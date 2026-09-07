# PQ Decode GEMV 指南（编译 / 转换 / 运行 / 调优）

本 fork 在 llama.cpp 中集成了 **PQ（乘积量化）decode-time GEMV**：token 生成阶段（`n_tokens==1`）
使用 int8 byte-LUT PQ 内核（AVX-512 FP16 + VBMI）计算 linear 层，**prefill 与批量推理保持原始权重**。
内核源自 pq_cpu 项目（`bench_pq.cpp`），量化方法参照 TFLOP 的 faiss ProductQuantizer。

---

## 1. 前置条件

### 硬件

- CPU 需支持 **AVX512-FP16** 和 **AVX512-VBMI**（如 Xeon 6759P-C / Granite Rapids）。检查：
  ```bash
  grep -o 'avx512[a-z_0-9]*' /proc/cpuinfo | sort -u | grep -E 'fp16|vbmi'
  # 需要同时看到 avx512_fp16 和 avx512vbmi
  ```
  不支持时编译不会报错（`pq-gemv.cpp` 有编译期守卫），但 `--pq-decode` 会静默回退原始路径，**没有 PQ 加速**。

### 软件

- GCC 13+（需 `-march=native` 生成 FP16/VBMI 指令）
- CMake ≥ 3.14、Ninja
- **必须在未激活 conda 环境的 shell 中构建**（见 §2.2 坑位 #1——这是最常见的失败原因）

---

## 2. 编译

### 2.1 推荐配置（一条命令）

**首先退出 conda 环境**（`conda deactivate` 直到提示符不再显示环境名，然后 `hash -r`）。
激活的 conda 环境会：把 PATH 里的 `gcc`/`ld` 换成 conda shim（其内嵌的 binutils 与系统
glibc 冲突）、注入 `-L/-rpath/-isystem` 等 CFLAGS/LIBRARY_PATH、且 conda gcc 通常缺
OpenMP——仅 `env -u CC -u CXX` 清不掉这些。

```bash
cd <llama.cpp 检出目录>

which gcc          # 必须输出 /usr/bin/gcc

cmake -S . -B build-amx -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=ON \
      -DGGML_CCACHE=OFF

ninja -C build-amx llama-cli llama-pq-convert -j$(nproc)
```

产物在 `build-amx/bin/`：`llama-cli`、`llama-pq-convert`。

无法退出 conda 时的替代方案（完全干净的环境执行）：

```bash
env -i HOME=$HOME USER=$USER TERM=$TERM PATH=/usr/bin:/bin:/usr/local/bin \
  cmake -S . -B build-amx -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++
env -i HOME=$HOME USER=$USER TERM=$TERM PATH=/usr/bin:/bin:/usr/local/bin \
  ninja -C build-amx llama-cli llama-pq-convert -j$(nproc)
```

### 2.2 关键要点

| 要点 | 原因 |
|---|---|
| `-G Ninja` | CMake 默认生成 Makefile；不加会在 `ninja -C build-amx` 时报 `build.ninja: No such file or directory` |
| `PATH` 里混入 conda 的 gcc/ld（报 `GLIBC_PRIVATE` 或 `dladdr@GLIBC_2.2.5`） | **`conda deactivate` + `hash -r` 后重新配置**（§2.1）。仅 `env -u CC -u CXX` 不够：conda 还注入 LIBRARY_PATH/CPATH，且 conda gcc 内嵌自己的 binutils |
| `OpenMP not found`（CMake 警告） | 用的 conda gcc（缺 OpenMP）；退出 conda 用系统 gcc |
| `env -u CC -u CXX ...` | 清掉 shell 里残留的编译器环境变量 |

### 2.3 增量编译

```bash
PATH=/usr/bin:/bin ninja -C build-amx llama-cli        # 只改源文件时
```

修改 CMakeLists（新增源文件/目标）后需重新运行 2.1 的 cmake 配置。

### 2.4 多机共享文件系统

若多个系统挂载同一份 home（如 `/home/user` 与 `/opt/raid5/users/user` 是同一 NFS），
**每个系统用独立的 build 目录**（`build-amx`、`build-node2`），不要共享：
CMakeCache 记录了各系统的编译器与 cmake 绝对路径，交叉使用会报
`CMake Error: The current CMakeCache.txt directory ... is different than ...` 或
`/usr/bin/cmake: No such file or directory`（ninja 自动重新生成时调用了别的系统的 cmake）。
出现该问题时的恢复方法：删除 build 目录，用本机 cmake 重新执行 2.1。

---

## 3. 模型转换（一次性）

`llama-pq-convert` 读取普通 GGUF，对 linear 层权重做 PQ 量化，把码本/索引作为
**侧挂张量**写进新 GGUF（原始权重原样保留供 prefill 使用）：

```bash
./build-amx/bin/llama-pq-convert <input.gguf> <output.gguf> [--pq-ds N] [--pq-mode auto|s1|s2]

# Llama-2-7B 实例
./build-amx/bin/llama-pq-convert Llama-2-7b-chat-hf.gguf Llama-2-7b-chat-hf-pq.gguf --pq-ds 2 --pq-mode auto
```

| 参数 | 说明 |
|---|---|
| `--pq-ds N` | 子空间维度（默认 2；支持 1..8）。ds↑ 压缩比更高但精度下降（见 §7） |
| `--pq-mode auto` | q/k/v/o、ffn_gate/ffn_up 沿输入维划分（S1）；ffn_down 沿输出维划分（S2） |
| `--pq-mode s1` / `s2` | 全部层用输入维划分 / 输出维划分 |

开销参考（7B F16）：

- 转换耗时 ~100 s（k-means，与机器负载相关），输出文件 12.55 → 16.35 GiB（+3.8 GiB）；
- 转换收益：加载时 PQ 注册从 ~100 s（现场量化）降到 **~2.5 s**（直接读文件），**~40×**。

转换后模型内每个被量化权重 `W` 附带侧挂张量：

| 张量 | 类型与形状 | 内容 |
|---|---|---|
| `W.pq_meta` | I32[4] | {mode(0=S1/1=S2), ds, K, 子空间数 M} |
| `W.pq_cb` | S1: F32 [ds,K,M]；S2: I8 [K,M·ds] | 码本 |
| `W.pq_idx` | I8，S1: [n_out,M]；S2: [n_in,M] | 索引（1 byte/d_sub 权重） |
| `W.pq_inv` | F32 [M·ds] | 仅 S2：dequant scale |

同时写入 GGUF KV：`pq.ds`、`pq.mode`。

> 不转换也能用：对普通 GGUF 直接 `--pq-decode`，加载时会现场做 k-means 量化（7B 约 100 s）。

---

## 4. 运行

### 4.1 基本用法

```bash
./build-amx/bin/llama-cli -m base-pq.gguf --pq-decode -t 60 \
    --cpu-range 0-59 --cpu-strict 1 \
    --no-conversation -st \
    -p "The meaning of life" -n 128
```

| 参数 | 说明 |
|---|---|
| `--pq-decode` | 启用 PQ decode GEMV（无侧挂张量的模型会现场量化） |
| `--pq-ds N` | 覆盖 ds；**含侧挂张量的模型会使用文件里记录的 ds，此参数无效** |
| `--pq-mode ...` | 同转换参数；对已转换模型同样以文件记录为准 |
| `-t 60 --cpu-range 0-59 --cpu-strict 1` | 绑定单 socket 的 60 个物理核（推荐，见 §6） |

`llama-server` 同样可用（`--pq-decode` 参数一致），适合服务化部署。

#### llama-bench 支持

`llama-bench` 已兼容 PQ（`-pd/--pq-decode <0|1>`、`--pq-ds N`），输出表含 `pq` 列，
且 PQ 设置不同的实例会自动分别加载模型：

```bash
./build-amx/bin/llama-bench -m base-pq.gguf -p 32 -n 32 -t 60 -fa off --pq-decode 0
./build-amx/bin/llama-bench -m base-pq.gguf -p 32 -n 32 -t 60 -fa off --pq-decode 1
# pp32 = prefill（恒为原始权重路径）; tg32 = decode（pq=on 时走 PQ 内核）
# 实测: tg32  15.8 t/s (pq=off) -> 30.2 t/s (pq=on/ds=2) ≈ 1.95×
```

双路 CPU 测试可使用 `--numa distribute`，并设置 `GGML_PQ_STRIPE=1` 让 PQ
codebook/index 的 huge-page buffer 采用双节点 first-touch；单节点测试可省略该变量：

```bash
GGML_PQ_STRIPE=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores \
  ./build-amx/bin/llama-bench -m base-pq.gguf -p 0 -n 32 -pg 0,32 \
  -t 60 --numa distribute -pd 1 -r 3
```

注意：pp 测试（batch>1）永远走原始权重路径，`--pq-decode` 不改变 prefill 性能。

### 4.2 必知行为（踩坑记录）

| 症状 | 原因与解法 |
|---|---|
| 后台/管道运行时无限刷 `> ` 提示符 | llama-cli 进入了交互对话模式。加 `--no-conversation -st`（单轮后退出），或用 `-cnv` + 终端交互 |
| 日志里看不到 `PQ decode enabled` | 库日志默认精简，加 `-v` 或 `--log-file xx.log`。成功注册会打印：`load_tensors: PQ decode enabled: N tensors quantized (ds=2, mode=0) in X ms` |
| `done_getting_tensors: wrong number of tensors` | 用了 PQ GGUF 但没加 `--pq-decode`（文件含额外侧挂张量，需要启用 PQ 让 loader 走 partial 模式） |
| 已转换模型上 `--pq-ds 4` 不生效 | 侧挂张量记录的 ds 优先；要换 ds 需重新转换 |

### 4.3 诊断与性能剖析

```bash
# 逐节点计时（按 op 类型聚合 busy/straggler-barrier + MUL_MAT 按张量 top 榜）
GGML_NODE_TIMING=1 ./build-amx/bin/llama-cli -m base-pq.gguf --pq-decode ... 2>&1 | grep NODE-TIMING -A 14
```

输出示例（Llama-2-7B，60 线程）：`MUL_MAT` 占每 token 时间的 ~89%（PQ 权重流为主），
barrier/调度合计 ~12-15%。若你的机器上 MUL_MAT 占比低而 barrier 占比高，说明瓶颈在
调度而非带宽，再考虑节点融合/线程池类优化。

### 4.4 验证 PQ 内核生效

```bash
... --pq-decode -v 2>&1 | grep "PQ decode enabled"
# 应看到: PQ decode enabled: 224 tensors quantized (ds=2, mode=0) in 2533 ms
```

转换过的模型 `in 2533 ms`（~2.5s）；未转换模型是 ~100 s（现场量化）。

---

## 5. 性能参考（Llama-2-7B F16，Xeon 6759P-C 60 线程单 socket）

| 配置 | decode | prefill |
|---|---|---|
| 原始 F16 权重 | 16.2 t/s | 不变 |
| **PQ decode（S1i8+S2i8）** | **~30 t/s（≈2.0×）** | 不变 |

- 加速来源：decode 每 token 流式读取全部权重，字节数从 2 B/权重（F16）压到 ~0.55 B/权重，DRAM 带宽-bound 下吞吐近似按压缩比提升（扣除 attention/调度等固定开销后约 2×）。
- **共享机器注意**：冷流 DRAM 带宽会被其他负载显著压低（实测同一台机从 430 GB/s 掉到 190 GB/s，吞吐同步下降）。跨时段对比数字时先确认机器状态。
- 长上下文建议追加 `-ctk q8_0 -ctv q8_0`（KV 量化收益需 2K+ 上下文才可测）。
- `GGML_NODE_TIMING=1` 的逐节点分解见 §4.3。

---

## 6. 精度

- PQ 量化（ds=2, K=256）后 GEMV 输出相对 dense 的 MSE ≈ 1.6e-2（与 TFLOP/faiss 同配置的 PQ 理论水平一致）；
- int8 码本/距离表引入的额外误差 ≤ 4e-4，被 PQ 本身误差掩盖；
- Llama-2-7B 实测生成文本连贯，与 F16 基线逐 token 对比仅有轻微分叉；
- **未做 perplexity 评估**——部署前建议用 `llama-perplexity` 对比量化前后（当前 `--pq-decode` 不影响 prefill，perplexity 结果反映的是原始权重路径，需另行支持）。

ds 压缩比/精度权衡（合成乘积码本实测，K=256）：

| d_sub | 每维级数 | 索引字节/权重 | 相对误差量级 |
|---|---|---|---|
| 1 | 256 | 1.0 | ~1e-4 |
| 2 | 16 | 0.5 | ~1e-2 |
| 4 | 4 | 0.25 | ~0.5（需残差量化） |
| 8 | 2 | 0.125 | 不可用 |

---

## 7. 源码结构与内核要点

```
ggml/include/ggml-pq.h          # PQ registry C API（模式/布局/计时接口）
ggml/src/ggml-cpu/pq-gemv.cpp   # 注册表 + ith/nth 并行 S1i8/S2i8 内核（huge_vec 巨页缓冲）
src/llama-pq.h/.cpp             # k-means 量化、GGUF 侧挂读写、加载路径
tools/pq-convert/pq-convert.cpp # GGUF 转换工具
ggml/src/ggml-cpu/ggml-cpu.c    # 拦截点：ggml_compute_forward 中 MUL_MAT，
                                # 位于 extra-buffer (repack/AMX) 拦截之前
```

内核要点（AVX-512）：

- 256 项 int8 LUT 借助 `vpermi2b` 索引越界清零：2×vpermi2b + 1×blend 覆盖 0..255；
- S2：码本离线 int8 量化（每侧 512B L1），even/odd 双查表 + fp16 FMA；x 预除 128 + scale×128 防 fp16 累加溢出；
- S1：距离表每 token 运行时构建 + int8 量化 + fp16 dequant 累加，两阶段（建表→`ggml_barrier`→归约）；
- PQ 缓冲使用 2MB 对齐巨页（`huge_vec`，TLB miss 敏感的 GB 级索引流）。

---

## 8. 故障排查 FAQ

| 现象 | 排查 |
|---|---|
| 链接报 `GLIBC_PRIVATE` / `dladdr@GLIBC_2.2.5` | conda 环境未退出：`conda deactivate` + `hash -r`，删除 build 目录重新配置（§2.1） |
| `build.ninja: No such file or directory` | 配置时漏了 `-G Ninja`（默认生成 Makefile） |
| `CMakeCache.txt directory is different` | build 目录被另一台机器的 cmake 配置过；删除重建，或该机器用独立 build 目录 |
| 编译通过但 `--pq-decode` 无加速 | ① CPU 缺 AVX512-FP16/VBMI（§1 检查）；② 模型未转换且未意识到现场量化耗时；③ 确认 `-v` 日志有 `PQ decode enabled` |
| decode 速度与预期不符 | ① 机器共享负载压低带宽（`GGML_NODE_TIMING=1` 看 MUL_MAT 有效带宽）；② 线程数/绑核（单 socket 物理核最优，SMT 通常无益）；③ 长上下文记得 KV 量化 |
| 加载很慢 | 未转换的模型会现场 k-means（7B ~100 s）；用 `llama-pq-convert` 预转换 |

## PQ-4c8b checkpoints

TFLOP `pq-4c8b` checkpoints can be exported into precomputed GGUF side tensors:

```bash
env -u LD_LIBRARY_PATH python3 tools/pq-convert/export_pq4c8b.py \
  ../TFLOP/best-formal-hard /tmp/pq4c8b-side \
  --model-config ../Llama-2-7b-hf/config.json
llama-pq-convert base-pq.gguf base-pq-4c8b.gguf \
  --pq4c8b-dir /tmp/pq4c8b-side
```

The exporter folds block and dimension scales into the FP16 codebook and keeps
per-output vector scales as `pq_row_scale`. Side tensors are written in the
canonical row order of the final GGUF weight. In particular, Q/K codes and
per-output scales receive the same Llama RoPE row permutation as the dense
HF-to-GGUF conversion. Query and key head counts are read separately from the
model configuration so grouped-query attention is handled correctly. Decode
uses the S1 AVX-512 LUT kernel with `ds=4`; prefill continues to use the
original F16 tensors.

## Interactive text generation

Build and launch the lightweight text frontend through the artifact Makefile:

```bash
make chat GGML_CUDA=OFF
```

The resulting `build/bin/llama-pq-chat` reads `tokenizer.chat_template` from
the GGUF. In the default `auto` mode, a model with an embedded template uses
llama.cpp's chat-template API and retains system/user/assistant message
history. A model without a template uses independent completion prompts and
does not receive synthetic role markers.

The mode can be overridden for diagnostics or for another compatible GGUF:

```bash
make chat GGML_CUDA=OFF CHAT_MODE=completion
make chat GGML_CUDA=OFF CHAT_MODEL=/data/chat-edgepq.gguf CHAT_MODE=auto
```

`CHAT_MODE=chat` fails if the GGUF has no embedded template. The terminal
prints the selected mode and Prompt/Generation Token/s. Use `/clear` to reset
message history and `/exit` to quit.
