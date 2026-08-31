#include "llama-pq.h"

#include "ggml.h"
#include "ggml-pq.h"
#include "llama-model-loader.h"
#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

namespace {

constexpr int PQ_K = GGML_PQ_K;

void parallel_for(int n_items, const std::function<void(int, int)> & fn) {
    int nth = (int) std::thread::hardware_concurrency();
    if (nth < 1) nth = 1;
    if (nth > 128) nth = 128;
    if (n_items < 4 * nth) nth = std::max(1, n_items / 4);
    std::vector<std::thread> th;
    th.reserve(nth);
    for (int t = 0; t < nth; t++) {
        th.emplace_back([&, t]() {
            const int i0 = (int)((int64_t) n_items * t / nth);
            const int i1 = (int)((int64_t) n_items * (t + 1) / nth);
            fn(i0, i1);
        });
    }
    for (auto & th_ : th) th_.join();
}

// per-subspace k-means: grid init (L levels per dim, L^ds <= K, mixed-radix
// index), then `iters` Lloyd refinements. pts: n x ds (row-major).
void kmeans_subspace(const float * pts, int n, int ds, int iters, float * cb) {
    int L = 1;
    while (L < PQ_K) {
        double p = 1;
        for (int d = 0; d < ds; d++) p *= (double)(L + 1);
        if (p <= (double) PQ_K) L++; else break;
    }

    float lo[GGML_PQ_MAX_DS], hi[GGML_PQ_MAX_DS];
    for (int d = 0; d < ds; d++) {
        lo[d] = 1e30f; hi[d] = -1e30f;
        for (int p = 0; p < n; p++) {
            const float v = pts[(size_t) p * ds + d];
            lo[d] = std::min(lo[d], v);
            hi[d] = std::max(hi[d], v);
        }
    }
    // grid init: centroid k = sum_d l_d * L^d, l_d on a uniform grid
    for (int k = 0; k < PQ_K; k++) {
        for (int d = 0; d < ds; d++) {
            const int stride = (int) std::pow((double) L, d);
            const int l = (k / stride) % L;
            cb[(size_t) k * ds + d] = (L > 1) ? lo[d] + (hi[d] - lo[d]) * (float) l / (float)(L - 1) : lo[d];
        }
    }

    // Lloyd iterations
    std::vector<float> sums((size_t) PQ_K * ds);
    std::vector<int>   cnt(PQ_K);
    for (int it = 0; it < iters; it++) {
        std::fill(sums.begin(), sums.end(), 0.f);
        std::fill(cnt.begin(), cnt.end(), 0);
        for (int p = 0; p < n; p++) {
            const float * pt = pts + (size_t) p * ds;
            int best = 0;
            float bd = 1e30f;
            for (int k = 0; k < PQ_K; k++) {
                float dd = 0.f;
                for (int d = 0; d < ds; d++) {
                    const float e = pt[d] - cb[(size_t) k * ds + d];
                    dd += e * e;
                    if (dd >= bd) break;
                }
                if (dd < bd) { bd = dd; best = k; }
            }
            for (int d = 0; d < ds; d++) sums[(size_t) best * ds + d] += pt[d];
            cnt[best]++;
        }
        for (int k = 0; k < PQ_K; k++) {
            if (cnt[k] > 0) {
                for (int d = 0; d < ds; d++)
                    cb[(size_t) k * ds + d] = sums[(size_t) k * ds + d] / cnt[k];
            }
        }
    }
}

int nearest_centroid(const float * pt, const float * cb, int ds) {
    int best = 0;
    float bd = 1e30f;
    for (int k = 0; k < PQ_K; k++) {
        float dd = 0.f;
        for (int d = 0; d < ds; d++) {
            const float e = pt[d] - cb[(size_t) k * ds + d];
            dd += e * e;
            if (dd >= bd) break;
        }
        if (dd < bd) { bd = dd; best = k; }
    }
    return best;
}

constexpr int kIters = 2;

// S2: subspace i covers output rows [i*ds, i*ds+ds); points = columns.
bool pq_s2_quantize(const float * W, int n_out, int n_in, int ds,
                    std::vector<int8_t> & cb8, std::vector<float> & inv,
                    std::vector<uint8_t> & idx) {
    const int M = n_out / ds;
    std::vector<float> cb((size_t) M * PQ_K * ds);
    idx.assign((size_t) M * n_in, 0);

    parallel_for(M, [&](int i0, int i1) {
        std::vector<float> pts((size_t) n_in * ds);
        for (int i = i0; i < i1; i++) {
            for (int p = 0; p < n_in; p++)
                for (int s = 0; s < ds; s++)
                    pts[(size_t) p * ds + s] = W[(size_t)(i * ds + s) * n_in + p];
            float * cb_i = &cb[(size_t) i * PQ_K * ds];
            kmeans_subspace(pts.data(), n_in, ds, kIters, cb_i);
            for (int p = 0; p < n_in; p++)
                idx[(size_t) i * n_in + p] = (uint8_t) nearest_centroid(pts.data() + (size_t) p * ds, cb_i, ds);
        }
    });

    // pack per-(i,side) int8 + dequant scale
    cb8.assign((size_t) M * ds * PQ_K, 0);
    inv.resize((size_t) M * ds);
    for (int i = 0; i < M; i++) {
        for (int s = 0; s < ds; s++) {
            float amax = 0.f;
            for (int k = 0; k < PQ_K; k++)
                amax = std::max(amax, std::abs(cb[(size_t)(i * PQ_K + k) * ds + s]));
            inv[(size_t) i * ds + s] = (amax > 0) ? amax / 127.0f : 0.0f;
            const float sc = (amax > 0) ? 127.0f / amax : 0.0f;
            for (int k = 0; k < PQ_K; k++) {
                int v = (int) std::lround(cb[(size_t)(i * PQ_K + k) * ds + s] * sc);
                cb8[((size_t) i * ds + s) * PQ_K + k] = (int8_t) std::max(-127, std::min(127, v));
            }
        }
    }
    return true;
}

// S1: subspace i covers input cols [i*ds, i*ds+ds); points = weight rows.
bool pq_s1_quantize(const float * W, int n_out, int n_in, int ds,
                    std::vector<float> & cbf, std::vector<uint8_t> & idx) {
    const int M = n_in / ds;
    cbf.assign((size_t) M * PQ_K * ds, 0.f);
    idx.assign((size_t) M * n_out, 0);

    parallel_for(M, [&](int i0, int i1) {
        std::vector<float> pts((size_t) n_out * ds);
        for (int i = i0; i < i1; i++) {
            for (int j = 0; j < n_out; j++)
                for (int d = 0; d < ds; d++)
                    pts[(size_t) j * ds + d] = W[(size_t) j * n_in + i * ds + d];
            float * cb_i = &cbf[(size_t) i * PQ_K * ds];
            kmeans_subspace(pts.data(), n_out, ds, kIters, cb_i);
            for (int j = 0; j < n_out; j++)
                idx[(size_t) i * n_out + j] = (uint8_t) nearest_centroid(pts.data() + (size_t) j * ds, cb_i, ds);
        }
    });
    return true;
}

} // namespace

static int g_pq_ds   = 2;
static int g_pq_mode = 0; // 0=auto, 1=S1, 2=S2

void llama_pq_begin(const llama_model_params & params) {
    ggml_pq_reset();
    g_pq_ds   = params.pq_ds   > 0 && params.pq_ds   <= GGML_PQ_MAX_DS ? params.pq_ds   : 2;
    g_pq_mode = params.pq_mode >= 0 && params.pq_mode <= 2             ? params.pq_mode : 0;
}

void llama_pq_finish(void) {
    ggml_pq_set_enabled(true);
}

bool llama_pq_build_pack(struct ggml_tensor * t, int mode, int ds,
                         llama_pq_pack & out) {
    out = llama_pq_pack{};
    if (!t || !t->data) return false;
    if (t->ne[2] != 1 || t->ne[3] != 1) return false;
    if (!ggml_is_contiguous(t)) return false;

    const int64_t n_out = t->ne[1]; // rows
    const int64_t n_in  = t->ne[0]; // cols
    if (n_in < 256 || n_out < 256) return false;
    if (n_in % ds != 0) return false;
    if (mode == 1 && n_out % ds != 0) return false;

    const auto * tt = ggml_get_type_traits(t->type);
    if (!tt || !tt->to_float) return false;

    std::vector<float> W((size_t) n_out * n_in);
    tt->to_float(t->data, W.data(), (int64_t) n_out * n_in);

    out.mode = mode;
    out.ds   = ds;
    out.n_in = n_in;
    out.n_out = n_out;
    if (mode == 0) {
        pq_s1_quantize(W.data(), (int) n_out, (int) n_in, ds, out.cbf, out.idx);
    } else {
        pq_s2_quantize(W.data(), (int) n_out, (int) n_in, ds, out.cb8, out.inv, out.idx);
    }
    return true;
}

bool llama_pq_register_tensor(struct ggml_tensor * t, int mode, int ds) {
    llama_pq_pack pack;
    if (!llama_pq_build_pack(t, mode, ds, pack)) return false;
    if (pack.mode == 0) {
        return ggml_pq_register(t->name, 0, pack.ds, pack.cbf.data(), nullptr,
                                nullptr, pack.idx.data(), pack.n_in, pack.n_out);
    }
    return ggml_pq_register(t->name, 1, pack.ds, nullptr, pack.cb8.data(),
                            pack.inv.data(), pack.idx.data(), pack.n_in, pack.n_out);
}

// ---------------------------------------------------------------------------
// GGUF side-tensor fast path
// ---------------------------------------------------------------------------
void llama_pq_side_name(char * buf, size_t cap, const char * base, const char * suffix) {
    snprintf(buf, cap, "%s.%s", base, suffix);
}

namespace {

// read a side tensor's raw bytes from the model file(s)
bool pq_read_side(struct llama_model_loader & ml, const char * name,
                  void * dst, size_t expect_bytes, ggml_type expect_type) {
    struct ggml_tensor * t = ml.get_tensor_meta(name);
    if (!t || t->type != expect_type) return false;
    const size_t bytes = ggml_nbytes(t);
    if (bytes != expect_bytes) return false;
    const auto & w = ml.require_weight(name);
    if (ml.use_mmap) {
        const auto & mapping = ml.mappings.at(w.idx);
        memcpy(dst, (const uint8_t *)mapping->addr() + w.offs, bytes);
    } else {
        GGML_ASSERT(w.idx < ml.files.size());
        auto & f = ml.files.at(w.idx);
        f->seek(w.offs, SEEK_SET);
        f->read_raw(dst, bytes);
    }
    return true;
}

} // namespace

bool llama_pq_register_from_loader(struct llama_model_loader & ml,
                                   const char * base, int * ds_out) {
    char nm[512];
    llama_pq_side_name(nm, sizeof(nm), base, "pq_meta");
    struct ggml_tensor * mt = ml.get_tensor_meta(nm);
    if (!mt || mt->type != GGML_TYPE_I32 || ggml_nbytes(mt) != 4 * sizeof(int32_t)) {
        return false;
    }
    int32_t meta[4] = {0, 0, 0, 0};
    if (!pq_read_side(ml, nm, meta, sizeof(meta), GGML_TYPE_I32)) return false;

    const int mode = meta[0];
    const int ds   = meta[1];
    const int K    = meta[2];
    if (ds_out) *ds_out = ds;
    if (mode == 4) {
        if (ds != 4 || K != GGML_PQ_K) return false;
        const int64_t n_in = ml.get_tensor_meta(base)->ne[0];
        const int64_t n_out = ml.get_tensor_meta(base)->ne[1];
        const int M = (int)(n_in / ds);
        llama_pq_side_name(nm, sizeof(nm), base, "pq_idx");
        struct ggml_tensor * it = ml.get_tensor_meta(nm);
        if (!it || it->type != GGML_TYPE_I8 || it->ne[0] != n_out || it->ne[1] != M) return false;
        std::vector<uint8_t> idx((size_t)M * n_out);
        if (!pq_read_side(ml, nm, idx.data(), idx.size(), GGML_TYPE_I8)) return false;
        llama_pq_side_name(nm, sizeof(nm), base, "pq_cb");
        std::vector<_Float16> cb((size_t)n_in * GGML_PQ_K);
        if (!pq_read_side(ml, nm, cb.data(), cb.size() * sizeof(_Float16), GGML_TYPE_F16)) return false;
        llama_pq_side_name(nm, sizeof(nm), base, "pq_row_scale");
        std::vector<float> row_scale((size_t)n_out);
        if (!pq_read_side(ml, nm, row_scale.data(), row_scale.size() * sizeof(float), GGML_TYPE_F32)) return false;
        return ggml_pq_register_raw_scaled(base, ds, cb.data(), idx.data(), row_scale.data(), n_in, n_out);
    }
    if (mode < 0 || mode > 1 || ds < 1 || ds > GGML_PQ_MAX_DS || K != GGML_PQ_K) {
        LLAMA_LOG_WARN("%s: unsupported PQ meta for '%s' (mode=%d ds=%d K=%d)\n",
                       __func__, base, mode, ds, K);
        return false;
    }

    const int64_t M = (mode == 0) ? meta[3] : meta[3]; // n_subspaces stored in meta[3]
    llama_pq_pack pack;
    pack.mode = mode;
    pack.ds   = ds;
    pack.n_in  = (mode == 0) ? M * ds : -1;
    pack.n_out = (mode == 0) ? -1 : M * ds;

    // read indices first to learn the missing extent (n_out for S1 / n_in for S2)
    llama_pq_side_name(nm, sizeof(nm), base, "pq_idx");
    struct ggml_tensor * it = ml.get_tensor_meta(nm);
    if (!it || it->type != GGML_TYPE_I8 || it->ne[2] != 1 || it->ne[3] != 1) return false;
    if (mode == 0) {
        pack.n_out = it->ne[0];
        if (pack.n_in != M * ds || it->ne[1] != M) return false;
        pack.idx.resize((size_t) M * it->ne[0]);
        if (!pq_read_side(ml, nm, pack.idx.data(), pack.idx.size(), GGML_TYPE_I8)) return false;
        llama_pq_side_name(nm, sizeof(nm), base, "pq_cb");
        pack.cbf.resize((size_t) M * GGML_PQ_K * ds);
        if (!pq_read_side(ml, nm, pack.cbf.data(),
                          pack.cbf.size() * sizeof(float), GGML_TYPE_F32)) return false;
        return ggml_pq_register(base, 0, ds, pack.cbf.data(), nullptr, nullptr,
                                pack.idx.data(), pack.n_in, pack.n_out);
    }
    pack.n_in = it ? it->ne[0] : -1;
    if (pack.n_in < 0 || it->ne[1] != M) return false;
    pack.idx.resize((size_t) M * pack.n_in);
    if (!pq_read_side(ml, nm, pack.idx.data(), pack.idx.size(), GGML_TYPE_I8)) return false;

    llama_pq_side_name(nm, sizeof(nm), base, "pq_cb");
    pack.cb8.resize((size_t) M * ds * GGML_PQ_K);
    if (!pq_read_side(ml, nm, pack.cb8.data(), pack.cb8.size(), GGML_TYPE_I8)) return false;
    llama_pq_side_name(nm, sizeof(nm), base, "pq_inv");
    pack.inv.resize((size_t) M * ds);
    if (!pq_read_side(ml, nm, pack.inv.data(), pack.inv.size() * sizeof(float), GGML_TYPE_F32)) return false;

    return ggml_pq_register(base, 1, ds, nullptr, pack.cb8.data(), pack.inv.data(),
                            pack.idx.data(), pack.n_in, pack.n_out);
}

// ---------------------------------------------------------------------------
// Graph-level QKV fusion
// ---------------------------------------------------------------------------
struct ggml_tensor * llama_pq_build_fused_weight(struct ggml_tensor ** ts, int n_ts,
        const char * name_from, const char * name_to, int64_t * row_offs) {
    if (n_ts < 1 || n_ts > 4) return nullptr;
    for (int m = 0; m < n_ts; m++) {
        if (!ts[m]) return nullptr;
        // rows must be memcpy-compatible: one common row-major type, equal ne[0]
        if (ts[m]->type != ts[0]->type) return nullptr;
        if (ts[m]->type != GGML_TYPE_F16 && ts[m]->type != GGML_TYPE_F32 &&
            ts[m]->type != GGML_TYPE_Q8_0) return nullptr;
        if (ts[m]->ne[0] != ts[0]->ne[0]) return nullptr;
        if (!ggml_is_contiguous(ts[m])) return nullptr;
    }

    int64_t n_out = 0;
    for (int m = 0; m < n_ts; m++) {
        row_offs[m] = n_out;
        n_out += ts[m]->ne[1];
    }

    static struct ggml_context * ctx = nullptr;   // tensor metadata, process lifetime
    if (!ctx) {
        struct ggml_init_params ip = {
            /*mem_size =*/ ggml_tensor_overhead() * 128,
            /*mem_buffer =*/ nullptr,
            /*no_alloc =*/ true,
        };
        ctx = ggml_init(ip);
    }
    // fused tensor name: rename a suffix of ts[0], e.g.
    //   blk.N.attn_q.weight -> blk.N.attn_qkv
    //   blk.N.ffn_gate.weight -> blk.N.ffn_gate_up
    std::string nm = ts[0]->name;
    const size_t p = nm.find(name_from);
    if (p == std::string::npos) return nullptr;
    nm.replace(p, strlen(name_from), name_to);

    const int64_t fne[2] = { ts[0]->ne[0], n_out };
    struct ggml_tensor * f = ggml_new_tensor(ctx, ts[0]->type, 2, fne);
    ggml_set_name(f, nm.c_str());
    f->data = malloc(ggml_nbytes(f));
    if (!f->data) return nullptr;
    size_t off = 0;
    for (int m = 0; m < n_ts; m++) {
        memcpy((char *) f->data + off, ts[m]->data, ggml_nbytes(ts[m]));
        off += ggml_nbytes(ts[m]);
    }
    return f;
}

struct ggml_tensor * llama_pq_build_qkv_weight(struct ggml_tensor * wq,
        struct ggml_tensor * wk, struct ggml_tensor * wv, int64_t * row_offs) {
    struct ggml_tensor * ts[3] = { wq, wk, wv };
    return llama_pq_build_fused_weight(ts, 3, ".attn_q.weight", ".attn_qkv", row_offs);
}

bool llama_pq_register_qkv_group(struct llama_model_loader & ml,
        const char * fused_name, struct ggml_tensor * wq,
        struct ggml_tensor * wk, struct ggml_tensor * wv, int * ds_out) {
    struct ggml_tensor * members[3] = { wq, wk, wv };
    int ds = 0;
    for (auto * t : members) {
        int mds = 0;
        if (!llama_pq_register_from_loader(ml, t->name, &mds)) {
            // no side tensors: fall back to on-the-fly k-means
            if (!llama_pq_register_tensor(t, 0, 2)) return false;
            mds = 2;
        }
        if (ds == 0) ds = mds;
        else if (ds != mds) return false;
    }
    const int64_t offs[3] = { 0, wq->ne[1], wq->ne[1] + wk->ne[1] };
    const char * names[3] = { wq->name, wk->name, wv->name };
    if (!ggml_pq_register_group(fused_name, wq->ne[0],
                                wq->ne[1] + wk->ne[1] + wv->ne[1], names, 3, offs)) {
        return false;
    }
    if (ds_out) *ds_out = ds;
    return true;
}

bool llama_pq_register_fused_group(struct llama_model_loader & ml,
        const char * fused_name, struct ggml_tensor ** members, int n_members,
        const int64_t * row_offs, int * ds_out) {
    int ds = 0;
    for (int m = 0; m < n_members; m++) {
        int mds = 0;
        if (!llama_pq_register_from_loader(ml, members[m]->name, &mds)) {
            // no side tensors: fall back to on-the-fly k-means
            if (!llama_pq_register_tensor(members[m], 0, 2)) return false;
            mds = 2;
        }
        if (ds == 0) ds = mds;
        else if (ds != mds) return false;
    }
    int64_t n_in = members[0]->ne[0], n_out = 0;
    const char * names[4];
    for (int m = 0; m < n_members; m++) {
        names[m] = members[m]->name;
        n_out += members[m]->ne[1];
    }
    if (!ggml_pq_register_group(fused_name, n_in, n_out, names, n_members, row_offs)) {
        return false;
    }
    if (ds_out) *ds_out = ds;
    return true;
}
