# Llama-2-7b-chat CPU 推理性能测试

基于 Intel Xeon 6759P-C (支持 AVX512 全集 + AMX) 的 llama.cpp CPU 推理性能测试项目。
覆盖 14 种 GGUF 量化格式的 prompt 处理 (pp) 与 token 生成 (tg) 吞吐,并支持交互对话。

- 硬件: Intel(R) Xeon(R) 6759P-C, 240 逻辑核 (2 socket x 60 核 x 2 SMT), 双 NUMA
- 指令集: AVX512F/DQ/BW/VL/CD/VBMI/VBMI2/VNNI/BF16/FP16/IFMA, AMX (BF16/INT8/TILE)
- 软件: GCC 13.3, CMake 3.31, Ninja 1.11, Ubuntu 24.04
- 上游: llama.cpp commit `efb3036c1` (build 9985)
- 模型: Llama-2-7b-chat-hf GGUF,位于 `/home/shared/models/gguf-models/Llama-2-7b-chat-hf-GGUF/`

## 目录结构

```
scripts/
  run-cpu-bench.sh    # 批量基准测试脚本 (14 种量化)
  chat.sh             # 交互对话脚本 (按量化名切换)
bench-results/
  README.md           # 本文档
  SUMMARY.md          # 性能结果汇总 + 量化算法原理
  llama2-7b-*.csv     # llama-bench 原始 CSV 数据
  llama2-7b-*.md      # llama-bench 原始 markdown 表
build-amx/            # 构建目录 (AMX + AVX512)
  bin/llama-bench     # 基准测试工具
  bin/llama-cli       # 交互对话工具
```

## 构建

### 配置 CMake (启用 AMX + AVX512)

```bash
cmake -S . -B build-amx -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_LTO=ON \
  -DGGML_AVX512=ON \
  -DGGML_AVX512_VBMI=ON \
  -DGGML_AVX512_VNNI=ON \
  -DGGML_AVX512_BF16=ON \
  -DGGML_AMX_TILE=ON \
  -DGGML_AMX_INT8=ON \
  -DGGML_AMX_BF16=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON
```

关键选项说明:
- `GGML_NATIVE=ON`: 使用 `-march=native`,启用当前 CPU 支持的全部指令集
- `GGML_LTO=ON`: 链接时优化,跨文件内联提升 GEMM 性能
- `GGML_AVX512*=ON`: AVX512 各子集,提升 batch GEMM 吞吐
- `GGML_AMX_TILE/INT8/BF16=ON`: 启用 AMX 后端,`ggml_backend_amx_mul_mat` 路径
- `LLAMA_BUILD_SERVER=ON`: `llama-cli` 依赖 `llama-server-impl`,必须开启

### 编译目标

```bash
# 基准测试 + 对话工具
cmake --build build-amx --target llama-bench llama-cli -j
```

### 验证 AMX 已编入二进制

```bash
nm -DC build-amx/bin/libggml-cpu.so | grep -i amx
# 应包含: ggml_backend_amx_mul_mat, ggml_backend_amx_buffer_type,
#         ggml_cpu_has_amx_int8
```

## 基准测试

### 一键测试全部量化

```bash
./scripts/run-cpu-bench.sh
```

默认配置 (可通过环境变量覆盖):
- `THREADS=120` (单 NUMA node 逻辑核数)
- `PP=512` (prompt 处理 512 tokens)
- `TG=128` (生成 128 tokens)
- `REPS=3` (每个测试重复 3 次)
- `NUMA=isolate` (NUMA 隔离,避免跨 socket 内存访问)

输出写入 `bench-results/`:
- `llama2-7b-<时间戳>.md` — markdown 表
- `llama2-7b-<时间戳>.csv` — CSV 原始数据 (可用 `scripts/compare-llama-bench.py` 对比)

### 自定义参数

```bash
THREADS=64 PP=1024 TG=256 REPS=5 NUMA=distribute ./scripts/run-cpu-bench.sh
```

### 单独运行 llama-bench

```bash
./build-amx/bin/llama-bench \
  -m /home/shared/models/gguf-models/Llama-2-7b-chat-hf-GGUF/Llama-2-7b-chat-hf.Q4_K_M.gguf \
  -p 512 -n 128 \
  -t 120 --numa isolate \
  -r 5 \
  -o md
```

输出格式: `-o md|csv|json|jsonl|sql`

### 线程数与 NUMA 选择

LLM 推理是**内存带宽受限**任务,不是核越多越好:
- 单 NUMA node (120 线程) 通常最优,因为模型权重集中在本地内存
- 双 socket (240 线程) 会引入跨 NUMA 内存访问,解码反而变慢
- `--numa isolate`: 每个线程绑定到固定核,避免迁移
- `--numa distribute`: 在 NUMA 节点间分布,适合大 batch

## 交互对话

### 用脚本 (推荐)

```bash
./scripts/chat.sh              # 默认 Q4_K_M
./scripts/chat.sh Q4_0         # 换 Q4_0
./scripts/chat.sh Q8_0         # 换 Q8_0
./scripts/chat.sh Q6_K -n 256  # 附加 llama-cli 参数
```

可用量化: `Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_0 Q4_1 Q4_K_S Q4_K_M Q5_0 Q5_1 Q5_K_S Q5_K_M Q6_K Q8_0`

进入后多轮对话,命令:
- `/exit` 或 `Ctrl+C` — 退出
- `/clear` — 清空对话历史
- `/regen` — 重新生成上一条回复
- `/read <file>` — 加入文本文件内容

### 直接用 llama-cli

```bash
./build-amx/bin/llama-cli \
  -m /home/shared/models/gguf-models/Llama-2-7b-chat-hf-GGUF/Llama-2-7b-chat-hf.Q4_K_M.gguf \
  -t 120 --numa isolate \
  -c 4096 \
  -cnv
```

切换量化只需改 `-m` 路径中的量化名。关键参数:

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `-m` | GGUF 模型路径 | 按量化切换 |
| `-t` | 线程数 | 120 (单 NUMA node) |
| `--numa` | NUMA 策略 | isolate |
| `-c` | 上下文长度 | 2048-4096 |
| `-n` | 生成 token 数,-1 直到填满上下文 | -1 |
| `-cnv` | 对话模式,自动套用 chat template | 默认开 |
| `-sys` | 系统提示词 | 可选 |
| `-p` | 单次 prompt (配合 `-no-cnv`) | 可选 |

### 单次生成 (非交互,快速验证)

```bash
./build-amx/bin/llama-cli \
  -m /home/shared/models/gguf-models/Llama-2-7b-chat-hf-GGUF/Llama-2-7b-chat-hf.Q4_0.gguf \
  -t 120 --numa isolate -c 2048 -n 48 \
  -p "Say one sentence about cats." \
  -no-cnv --no-display-prompt
```

## 量化算法原理

llama.cpp 的 GGUF 量化分两大家族,均为 **block-wise 线性量化**: 每 N 个权重共享一个 scale `d` (可选 min `m`),
重建公式 `x = d * q + m`。block 大小与 scale 存储方式是各方案的差异核心。

有效 bits-per-weight (bpw) = (权重位数 + scale 位数) / block 大小。
LLM 推理带宽受限,bpw 越低解码越快,但精度损失越大。

### Legacy block quants (block size = 32)

每 32 个权重一组,单一 scale,无 super-block 结构。解码简单快,但单 scale 对 outlier 适配差。

| 类型 | Block | 布局 (每 32 权重) | bpw | 重建 |
|------|------:|------------------|----:|------|
| Q4_0 | 32 | 1 x fp16 scale + 32 x 4-bit (对称) | 4.50 | `x = d * q` |
| Q4_1 | 32 | 1 x fp16 scale + 1 x fp16 min + 32 x 4-bit | 5.00 | `x = d * q + m` (非对称) |
| Q5_0 | 32 | 1 x fp16 scale + 1 x u32 hi-bit + 32 x 4-bit | 5.50 | `x = d * q` (5-bit) |
| Q5_1 | 32 | 1 x fp16 scale + 1 x fp16 min + 1 x u32 hi-bit + 32 x 4-bit | 6.00 | `x = d * q + m` (5-bit 非对称) |
| Q8_0 | 32 | 1 x fp16 scale + 32 x int8 | 8.50 | `x = d * q` |

- **对称 (Q4_0/Q5_0/Q8_0)**: 范围 `[-d, +d]` 围绕零,最省,适合近似零均值权重
- **非对称 (Q4_1/Q5_1)**: 加 min `m`,范围不必居中,对非零均值 block 精度更好,多 ~0.5 bpw
- **Q8_0**: int8 + per-block fp16 scale,本组最高保真,常作低比特方案的精度参考

### K-quants (super-block = 256, `QK_K = 256`)

两级 scale。一个 256 权重的 **super-block** 分成 16 或 32 个 sub-block,每个 sub-block 有自己的 6-bit scale,
这些 sub-scale 再被单个 fp16 super-block scale `d` (和 `dmin`) 量化。两级结构大幅降低 per-weight scale 开销,
腾出位数给权重本身。

| 类型 | Sub-blocks | 权重位 | Scale 存储 | bpw | 重建 |
|------|-----------:|-------:|-----------|----:|------|
| Q2_K | 16 x 16 | 2 | 4-bit sub-scale + fp16 d/dmin | 2.625 | `x = d*s*q + dmin*smin` |
| Q3_K | 16 x 16 | 3 | 6-bit sub-scale + fp16 d | 3.4375 | `x = d*s*q` |
| Q4_K | 8 x 32 | 4 | 6-bit sub-scale + fp16 d/dmin | 4.5 | `x = d*s*q + dmin*smin` |
| Q5_K | 8 x 32 | 5 | 6-bit sub-scale + fp16 d/dmin | 5.5 | `x = d*s*q + dmin*smin` |
| Q6_K | 16 x 16 | 6 | 8-bit int8 sub-scale + fp16 d | 6.5625 | `x = d*s*q` |

- **Q2_K**: 2-bit 权重 + 4-bit sub-scale + min,最低 bpw,带宽收益最大但精度损失最大
- **Q3_K**: 3-bit + 6-bit scale,三档变体:
  - **Q3_K_S** (Small): 4-bit scale,精度略低
  - **Q3_K_M** (Medium): 6-bit scale (默认)
  - **Q3_K_L** (Large): 6-bit scale + 1-bit 张量保留高精度
- **Q4_K / Q5_K**: 4/5-bit,8 个 32 权重 sub-block,6-bit scale + min (非对称)
  - **_S**: 部分 attention/mlp 层降级到更低精度
  - **_M**: 全部保持 Q4_K/Q5_K
- **Q6_K**: 6-bit + int8 sub-scale (未再量化),接近 fp16, K-quant 精度参考

### K-quant 精度优势

Legacy 32-block 每 32 权重付一个完整 fp16 scale = 0.5 bpw 开销 (有 min 再 +0.5)。
K-quant 每 16/32 sub-block 只付几位 (sub-scale 被量化),同等 bpw 下留给权重的位数更多。
所以同尺寸下 Q4_K_M 困惑度低于 Q4_1,K-quant 在结果中占优。

## 性能结果

测试条件: 120 线程, NUMA isolate, pp512 + tg128, 每项 3 次重复, KV cache f16, mmap 开启。

| Quant | Model size | pp512 t/s | tg128 t/s |
|-------|-----------|----------:|----------:|
| Q2_K | 2.36 GiB | 75.21 ± 1.58 | 24.10 ± 21.34 |
| Q3_K_S | 2.75 GiB | 141.65 ± 24.58 | 31.66 ± 1.00 |
| Q3_K_M | 3.07 GiB | 72.37 ± 5.61 | 15.36 ± 16.91 |
| Q3_K_L | 3.35 GiB | 176.30 ± 66.92 | 35.72 ± 0.07 |
| Q4_0 | 3.56 GiB | 567.64 ± 332.81 | 45.36 ± 0.79 |
| Q4_1 | 3.95 GiB | 459.57 ± 183.58 | 42.98 ± 0.35 |
| Q4_K_S | 3.59 GiB | 658.61 ± 84.84 | 39.54 ± 0.46 |
| Q4_K_M | 3.80 GiB | 810.60 ± 7.17 | 36.31 ± 2.27 |
| Q5_0 | 4.33 GiB | 233.09 ± 1.83 | 22.97 ± 0.04 |
| Q5_1 | 4.72 GiB | 113.54 ± 7.22 | 21.98 ± 0.06 |
| Q5_K_S | 4.33 GiB | 554.75 ± 161.93 | 30.44 ± 0.68 |
| Q5_K_M | 4.45 GiB | 604.90 ± 32.50 | 28.99 ± 0.56 |
| Q6_K | 5.15 GiB | 439.42 ± 125.80 | 27.13 ± 0.70 |
| Q8_0 | 6.67 GiB | 450.52 ± 104.59 | 25.86 ± 0.06 |

### 关键观察

- **生成 (tg128)** 带宽受限,bpw 越低越快: Q4_0 (45.4 t/s) > Q4_1 (43.0) > Q4_K_S (39.5) > Q4_K_M (36.3)
- **Prompt 处理 (pp512)** 计算受限,AMX 加速明显: Q4_K_M 达 810 t/s,K-quant GEMM 内核在 AMX bf16/int8 tile 路径上高度优化
- **Q4_0/Q4_1 生成快于 K-quant 同位宽**: 解码内核更简单 (无 super-block scale 解包),AMX 有直接 Q4_0/Q8_0 mul_mat 路径
- **标准差偏大** 的项 (Q2_K tg、Q3_K_M tg、Q4_0 pp) 主要来自首次运行 mmap 冷页错误,可加 `--no-warmup` 关闭预热后取稳态,或预先 `cat` 模型文件预热 page cache

### 选型建议

| 场景 | 推荐 | 理由 |
|------|------|------|
| 最大吞吐,可接受精度损失 | Q4_0 | 生成最快 45 t/s,AMX 直接路径 |
| 精度与速度平衡 | Q4_K_M | 生成 36 t/s,prompt 810 t/s,困惑度优于 Q4_1 |
| 高精度,延迟不敏感 | Q6_K | 接近 fp16 精度,生成 27 t/s |
| 极致带宽节省 | Q2_K / Q3_K_S | 生成 24-32 t/s,精度损失明显 |

## 故障排查

**`ninja: error: unknown target 'llama-cli'`**
旧版目标名是 `llama-cli`,新版可能重命名为 `llama-app`。本构建已确认 `llama-cli` 可用,
若失败检查 `LLAMA_BUILD_SERVER=ON` (llama-cli 依赖 llama-server-impl)。

**AMX 符号缺失**
确认 CMake 配置中 `GGML_AMX_BF16/INT8/TILE=ON`,用 `nm -DC build-amx/bin/libggml-cpu.so | grep amx` 验证。

**解码速度比 bench 慢**
- 检查是否用了 `--numa isolate` 和 `-t 120`
- 首次运行有 mmap 冷启动,跑第二轮取稳态
- 对话模式 `-cnv` 比 `-no-cnv` 单次生成略慢 (要处理 chat template)

**模型加载失败 / OOM**
- Llama-2-7b Q8_0 约 6.7 GiB,确保内存充足
- 用 `-c` 调小上下文 (默认从模型 metadata 读取,可能很大)

## 参考

- 性能结果详情: [SUMMARY.md](SUMMARY.md)
- 原始 CSV 数据: `llama2-7b-*.csv`
- llama-bench 工具: `tools/llama-bench/llama-bench.cpp`
- 量化实现: `ggml/src/ggml-quants.c`, `ggml/src/ggml-cpu/amx/amx.cpp`
- 量化结构定义: `ggml/src/ggml-common.h`
- llama.cpp 上游: https://github.com/ggml-org/llama.cpp
