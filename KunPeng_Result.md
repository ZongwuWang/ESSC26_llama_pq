# 鲲鹏评测结果

华为鲲鹏（aarch64）decode 吞吐结果

---

## 鲲鹏结果


| Model      | TG128 (tok/s)    | TG512 (tok/s)    | WikiText-2 PPL |
| ---------- | ---------------- | ---------------- | -------------- |
| F16        | 8.63 ± 0.16      | 8.24 ± 0.13      | —              |
| Q2_K       | 33.11 ± 0.32     | 31.22 ± 0.03     | —              |
| **EdgePQ** | **18.30 ± 0.01** | **16.93 ± 3.42** | **—**          |


EdgePQ 相对 F16 为 **2.12× / 2.05×**（TG128 / TG512），相对 Q2_K 为 **0.55× / 0.54×**（约慢 **45% / 46%**）。

---

## Main Results（对照）

 **x86-64 + AVX-512**，**60 CPU 线程**：


| Model      | TG128 (tok/s)    | TG512 (tok/s)    | WikiText-2 PPL |
| ---------- | ---------------- | ---------------- | -------------- |
| F16        | 25.24 ± 0.38     | 24.10 ± 0.77     | 6.09           |
| Q2_K       | 38.11 ± 0.89     | 34.24 ± 0.37     | 7.75           |
| **EdgePQ** | **56.50 ± 0.64** | **53.04 ± 0.58** | **6.33**       |


结论：EdgePQ 相对 F16 快 **2.24× / 2.20×**，相对 Q2_K 快 **48.3% / 54.9%**（TG128 / TG512）。

---

## 运行指令

**60 线程**、**TG128 / TG512**。

```bash
# 1) 环境与编译
make kunpeng-check
make kunpeng-build
make kunpeng-selftest

# 2) 准备模型
make env                 
make prepare-inputs
make check-benchmark-inputs

# 3) 吞吐评测（结果写入 output/throughput.csv）
make kunpeng-benchmark \
  AE_THREADS=60 \
  AE_NUMA=distribute \
  AE_GENERATIONS=128,512 \
  AE_REPETITIONS=3 \
  AE_WARMUP=1 \
  AE_CONTEXT=2048
```



```bash
./build/bin/llama-perplexity \
  -m models/Llama-2-7b-chat-hf-f16.gguf \
  -f datasets/wikitext-2-raw-v1.test.txt \
  -c 3072 --ppl-stride 2048 \
  -t 60 --numa distribute
```

