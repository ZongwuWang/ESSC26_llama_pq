SHELL := /bin/bash

CXX := /usr/bin/g++
CC := /usr/bin/gcc
CMAKE ?= cmake
NINJA ?= ninja
UV ?= uv
CUDACXX ?=

VENV ?= .venv
PYTHON := $(VENV)/bin/python # 
HF := $(VENV)/bin/hf

LLAMA_SRC ?= llama.cpp
LLAMA_BUILD ?= build
OUTPUT_DIR ?= output
MODEL_DIR ?= models
DATASET_DIR ?= datasets

EDGE_PQ_REPO ?= ZongwuWang/EdgePQ-4c8b
EDGE_PQ_REVISION ?= e0d5a4e1bc91370f866d6bc729ed24a602666029
EDGE_PQ_SHA256 ?= 189c984206891d84076581cf6b2bd3a11864790c20e6a9aa27a4919e31b69bdf
FP16_REPO ?= second-state/Llama-2-7B-Chat-GGUF
FP16_REVISION ?= 064fe43ea8c1e1f93477ef4a170bdc2b244ef02c
Q2_REPO ?= $(EDGE_PQ_REPO)
Q2_REVISION ?= $(EDGE_PQ_REVISION)
FP16_FILE ?= Llama-2-7b-chat-hf-f16.gguf
Q2_FILE ?= Llama-2-7b-chat-hf.Q2_K.gguf
PQ_MODEL ?= $(MODEL_DIR)/base-pq-4c8b.gguf
PQ_MODEL_VERIFY_STAMP ?= $(PQ_MODEL).verified
FP16_MODEL ?= $(MODEL_DIR)/$(FP16_FILE)
Q2_MODEL ?= $(MODEL_DIR)/$(Q2_FILE)
PQ_CHECKPOINT ?= $(MODEL_DIR)/best-formal-hard
PPL_DATASET ?= $(DATASET_DIR)/wikitext-2-raw-v1
TOKENIZER ?= $(MODEL_DIR)/tokenizer

CUDA_DEVICE ?= 0

# Architecture defaults: Kunpeng/aarch64 is CPU-native (no CUDA); x86 keeps
# the paper artifact defaults (CUDA ON, 60 threads, cores 0-59).
UNAME_M := $(shell uname -m)
NPROC := $(shell nproc 2>/dev/null || echo 1)
ifeq ($(UNAME_M),aarch64)
GGML_CUDA ?= OFF
# llama-bench/PQ shared-LUT barriers: nproc (e.g. 640) usually hurts vs ~48–64.
AE_THREADS ?= 60
# Single LUT replica avoids N_nodes×memcpy on multi-NUMA Kunpeng hosts.
GGML_PQ_NUMA_LUT ?= 0
AE_NUMA ?= distribute
CHAT_CPU_RANGE ?= 0-$(shell expr $(NPROC) - 1 || echo 0)
PPL_THREADS ?= $(NPROC)
PPL_NUMA ?= distribute
else
GGML_CUDA ?= ON
AE_THREADS ?= 60
AE_NUMA ?= distribute
CHAT_CPU_RANGE ?= 0-59
PPL_THREADS ?= 60
PPL_NUMA ?= distribute
endif

AE_REPETITIONS ?= 3
AE_WARMUP ?= 1
AE_GENERATIONS ?= 128,512
AE_CONTEXT ?= 2048

CHAT_MODEL ?= $(PQ_MODEL)
CHAT_MODE ?= auto
CHAT_THREADS ?= $(AE_THREADS)
CHAT_CONTEXT ?= 4096
CHAT_MAX_TOKENS ?= 256
CHAT_SYSTEM ?=

GGUF_PPL_CONTEXT ?= 3072
GGUF_PPL_STRIDE ?= 2048
PQ_PPL_CONTEXT ?= 4096
PQ_PPL_STRIDE ?= 4096

THROUGHPUT_OUTPUT := $(OUTPUT_DIR)/throughput.csv
PPL_OUTPUT := $(OUTPUT_DIR)/perplexity.csv

ifeq ($(strip $(CUDACXX)),)
CUDACXX := $(firstword $(wildcard /usr/local/cuda/bin/nvcc /usr/local/cuda-12.4/bin/nvcc /usr/local/cuda-12.9/bin/nvcc))
endif
CUDA_COMPILER_ARG := $(if $(CUDACXX),-DCMAKE_CUDA_COMPILER="$(CUDACXX)",)

SYSTEM_BUILD_ENV := env -u CFLAGS -u CXXFLAGS -u LDFLAGS \
	-u LD_LIBRARY_PATH -u LIBRARY_PATH -u CPATH \
	-u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u GCC_EXEC_PREFIX

CPPFLAGS := -O2 -std=c++17 -Wall -Wextra -pedantic
CPPFLAGS += -I$(LLAMA_SRC)/include -I$(LLAMA_SRC)/ggml/include
LLAMA_LDFLAGS := -L$(LLAMA_BUILD)/bin -Wl,-rpath,$(abspath $(LLAMA_BUILD)/bin)
LLAMA_LDLIBS := -lllama -lggml -lggml-cpu -lggml-base -lpthread -ldl -lm

.PHONY: help all demo env prepare-edgepq prepare-baselines prepare-dataset prepare-inputs \
	check-benchmark-inputs check-ppl-inputs check-pq-ppl-inputs check-inputs \
	llama-build llama_pq prepare-chat-model chat selftest smoke benchmark ppl pq-ppl \
	plot clean distclean kunpeng-check kunpeng-verify-isa kunpeng-build kunpeng-selftest \
	kunpeng-smoke kunpeng-benchmark kunpeng-chat kunpeng-pq-meta kunpeng-export-s2 \
	kunpeng-export-k64 kunpeng-export-mix kunpeng-benchmark-k64 kunpeng-pq-k64-accept \
	kunpeng-thread-sweep

help:
	@echo "EdgePQ artifact targets:"
	@echo "  make help           List the public artifact targets"
	@echo "  make env            Create the locked Python environment"
	@echo "  make prepare-inputs Download all public models and WikiText-2"
	@echo "  make check-inputs   Validate all required external inputs"
	@echo "  make llama-build    Build llama.cpp, PPL, quantizer, and PQ tools"
	@echo "  make selftest       Run the PQ correctness self-test"
	@echo "  make smoke          Run an 8-token three-model smoke test"
	@echo "  make chat           Auto-detect Chat vs completion mode from the GGUF"
	@echo "  make demo           Run cache-aware end-to-end reproduction"
	@echo "  make benchmark      Reproduce decode throughput CSV"
	@echo "  make ppl            Evaluate F16 and Q2_K perplexity"
	@echo "  make pq-ppl         Evaluate PQ reconstruction perplexity"
	@echo "  make all            Rerun the complete paper evaluation"
	@echo "  make plot           Render throughput and PPL charts"
	@echo ""
	@echo "Kunpeng / aarch64 targets (see docs/DEPLOY_KUNPENG.md):"
	@echo "  make kunpeng-check      Verify host is aarch64 and print CPU info"
	@echo "  make kunpeng-verify-isa Build + confirm pq-gemv-arm.cpp has fp16+dotprod"
	@echo "  make kunpeng-build      CPU-only build (GGML_CUDA=OFF, armv8.2+fp16+dotprod)"
	@echo "  make kunpeng-selftest   Build + PQ numeric self-test"
	@echo "  make kunpeng-smoke      8-token F16/Q2_K/EdgePQ smoke test"
	@echo "  make kunpeng-benchmark  Decode throughput -> output/throughput.csv"
	@echo "  make kunpeng-chat       Interactive EdgePQ chat on all CPU cores"
	@echo "  make kunpeng-pq-meta    Print pq_meta modes in PQ_MODEL (expect mode=4)"
	@echo "  make kunpeng-export-s2  Online S2 K=256 A/B from F16 (not trained 4c8b)"
	@echo "  make kunpeng-export-k64 Online scaled S1 K=64 ds=4 (mode-4 GGUF) from F16"
	@echo "  make kunpeng-export-mix Online auto S1/S2 K=64 mix from F16"
	@echo "  make kunpeng-benchmark-k64 Throughput with base-pq-k64-kunpeng.gguf"
	@echo "  make kunpeng-pq-k64-accept Export K64 + PPL |Δ|<=0.2 vs $(PQ_MODEL)"
	@echo "  make kunpeng-thread-sweep Phase0 stable thread sweep (unset S1_PARTIALS)"


all:
	@echo
	@echo "==> [1/7] Creating the locked Python environment"
	$(MAKE) env
	@echo
	@echo "==> [2/7] Preparing public models, checkpoint, tokenizer, and WikiText-2"
	$(MAKE) prepare-inputs
	@echo
	@echo "==> [3/7] Validating every external input"
	$(MAKE) check-inputs
	@echo
	@echo "==> [4/7] Building the bundled llama.cpp and PQ tools"
	$(MAKE) llama-build
	@echo
	@echo "==> [5/7] Running the PQ correctness self-test"
	$(MAKE) selftest
	@echo
	@echo "==> [6/7] Measuring F16, Q2_K, and EdgePQ decode throughput"
	$(MAKE) benchmark
	@echo
	@echo "==> [7/7] Measuring F16, Q2_K, and EdgePQ perplexity"
	$(MAKE) ppl
	$(MAKE) pq-ppl
	@echo
	@echo "==> Complete: CSV results are in $(OUTPUT_DIR)/; run 'make plot' to render charts."

demo:
	@echo
	@echo "==> [1/7] Creating or reusing the locked Python environment"
	$(MAKE) env
	@echo
	@echo "==> [2/7] Detecting cached inputs and downloading only missing files"
	$(MAKE) prepare-inputs
	@echo
	@echo "==> [3/7] Validating models, checkpoint, tokenizer, and dataset"
	$(MAKE) check-inputs
	@echo
	@echo "==> [4/7] Reusing or building the bundled runtime"
	$(MAKE) llama-build
	@echo
	@echo "==> [5/7] Running the standalone PQ correctness test"
	$(MAKE) selftest
	@echo
	@echo "==> [6/7] Loading all three models and generating eight tokens each"
	$(MAKE) smoke
	@echo
	@echo "==> [7/7] Producing the paper results and terminal summary"
	@if test -s "$(THROUGHPUT_OUTPUT)" && test -s "$(PPL_OUTPUT)"; then \
		echo "[REUSE] $(THROUGHPUT_OUTPUT)"; \
		echo "[REUSE] $(PPL_OUTPUT)"; \
	else \
		echo "[RUN] Cached CSVs are incomplete; continuing with the full paper evaluation"; \
		$(MAKE) benchmark; \
		$(MAKE) ppl; \
		$(MAKE) pq-ppl; \
	fi
	$(MAKE) plot
	@echo
	@echo "==> Demo complete. Generated artifacts:"
	@echo "    - $(THROUGHPUT_OUTPUT)"
	@echo "    - $(PPL_OUTPUT)"
	@echo "    - $(OUTPUT_DIR)/throughput.png"
	@echo "    - $(OUTPUT_DIR)/ppl.png"

env: uv.lock pyproject.toml
	$(UV) sync --frozen
	@echo "[OK] Python environment: $(VENV)"

prepare-edgepq: env
	mkdir -p "$(MODEL_DIR)"
	@set -eu; \
	model="$(PQ_MODEL)"; \
	stamp="$(PQ_MODEL_VERIFY_STAMP)"; \
	expected="$(EDGE_PQ_SHA256)"; \
	test -n "$$expected" || { echo "[ERROR] EDGE_PQ_SHA256 is empty" >&2; exit 1; }; \
	verify_model() { \
		if test -f "$$stamp" && test "$$stamp" -nt "$$model"; then \
			cached=$$(cat "$$stamp"); \
			if test "$$cached" = "$$expected"; then \
				return 0; \
			fi; \
			echo "[STALE] EdgePQ GGUF verification stamp targets a different SHA-256" >&2; \
			echo "        required: $$expected" >&2; \
			echo "        cached:   $$cached" >&2; \
			return 1; \
		fi; \
		echo "[CHECK] Verifying EdgePQ GGUF SHA-256: $$model"; \
		actual=$$(sha256sum "$$model" | awk '{print $$1}'); \
		test "$$actual" = "$$expected" || { \
			echo "[STALE] EdgePQ GGUF SHA-256 mismatch" >&2; \
			echo "        expected: $$expected" >&2; \
			echo "        actual:   $$actual" >&2; \
			return 1; \
		}; \
		printf '%s\n' "$$expected" > "$$stamp"; \
	}; \
	if test -f "$$model" && verify_model; then \
		echo "[REUSE] Verified EdgePQ GGUF: $$model"; \
	else \
		rm -f "$$stamp"; \
		echo "[DOWNLOAD] EdgePQ GGUF from $(EDGE_PQ_REPO)@$(EDGE_PQ_REVISION)"; \
		$(HF) download "$(EDGE_PQ_REPO)" base-pq-4c8b.gguf \
			--revision "$(EDGE_PQ_REVISION)" --local-dir "$(MODEL_DIR)" --force-download; \
		test -f "$$model" || { echo "[ERROR] EdgePQ download did not create $$model" >&2; exit 1; }; \
		verify_model || { echo "[ERROR] Downloaded EdgePQ GGUF failed verification" >&2; exit 1; }; \
		echo "[OK] Downloaded and verified EdgePQ GGUF: $$model"; \
	fi
	@count=$$(find "$(PQ_CHECKPOINT)/pq_states" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l | tr -d ' '); \
	if test -f "$(PQ_CHECKPOINT)/non_pq_state.pt" && test "$$count" -eq 224; then \
		echo "[REUSE] EdgePQ checkpoint: $(PQ_CHECKPOINT) (224/224 PQ states)"; \
	else \
		echo "[DOWNLOAD] EdgePQ checkpoint from $(EDGE_PQ_REPO)@$(EDGE_PQ_REVISION)"; \
		$(HF) download "$(EDGE_PQ_REPO)" \
			--revision "$(EDGE_PQ_REVISION)" \
			--include "best-formal-hard/*" --local-dir "$(MODEL_DIR)"; \
	fi
	@if test -f "$(TOKENIZER)/tokenizer.json" && \
		test -f "$(TOKENIZER)/tokenizer.model" && \
		test -f "$(TOKENIZER)/tokenizer_config.json"; then \
		echo "[REUSE] Llama-2 tokenizer: $(TOKENIZER)"; \
	else \
		echo "[DOWNLOAD] Llama-2 tokenizer from $(EDGE_PQ_REPO)@$(EDGE_PQ_REVISION)"; \
		$(HF) download "$(EDGE_PQ_REPO)" tokenizer.json tokenizer.model tokenizer_config.json \
			--revision "$(EDGE_PQ_REVISION)" --local-dir "$(TOKENIZER)"; \
	fi

prepare-baselines: env
	mkdir -p "$(MODEL_DIR)"
	@if test -f "$(FP16_MODEL)"; then \
		echo "[REUSE] FP16 baseline: $(FP16_MODEL)"; \
	else \
		echo "[DOWNLOAD] FP16 baseline from $(FP16_REPO)@$(FP16_REVISION)"; \
		$(HF) download "$(FP16_REPO)" "$(FP16_FILE)" \
			--revision "$(FP16_REVISION)" --local-dir "$(MODEL_DIR)"; \
	fi
	@if test -f "$(Q2_MODEL)"; then \
		echo "[REUSE] Q2_K baseline: $(Q2_MODEL)"; \
	else \
		echo "[DOWNLOAD] Q2_K baseline from $(Q2_REPO)@$(Q2_REVISION)"; \
		$(HF) download "$(Q2_REPO)" "$(Q2_FILE)" \
			--revision "$(Q2_REVISION)" --local-dir "$(MODEL_DIR)"; \
	fi

prepare-dataset: env
	$(PYTHON) ppl_gguf_compare.py prepare-dataset --output "$(PPL_DATASET)"

prepare-inputs: prepare-edgepq prepare-baselines prepare-dataset
	@echo "[OK] Input preparation complete"

check-benchmark-inputs:
	@test -f "$(FP16_MODEL)" || { echo "missing FP16_MODEL: $(FP16_MODEL)"; exit 1; }
	@test -f "$(Q2_MODEL)" || { echo "missing Q2_MODEL: $(Q2_MODEL)"; exit 1; }
	@test -f "$(PQ_MODEL)" || { echo "missing PQ_MODEL: $(PQ_MODEL)"; exit 1; }

check-ppl-inputs:
	@test -f "$(FP16_MODEL)" || { echo "missing FP16_MODEL: $(FP16_MODEL)"; exit 1; }
	@test -f "$(Q2_MODEL)" || { echo "missing Q2_MODEL: $(Q2_MODEL)"; exit 1; }
	@test -d "$(PPL_DATASET)" || { echo "missing PPL_DATASET: $(PPL_DATASET)"; exit 1; }

check-pq-ppl-inputs:
	@test -f "$(PQ_CHECKPOINT)/non_pq_state.pt" || { echo "missing non-PQ state: $(PQ_CHECKPOINT)/non_pq_state.pt"; exit 1; }
	@count=$$(find "$(PQ_CHECKPOINT)/pq_states" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l | tr -d ' '); \
		test "$$count" -eq 224 || { echo "expected 224 PQ states, found $$count: $(PQ_CHECKPOINT)/pq_states"; exit 1; }
	@test -f "$(TOKENIZER)/tokenizer.json" || { echo "missing tokenizer.json: $(TOKENIZER)"; exit 1; }
	@test -f "$(TOKENIZER)/tokenizer.model" || { echo "missing tokenizer.model: $(TOKENIZER)"; exit 1; }
	@test -f "$(TOKENIZER)/tokenizer_config.json" || { echo "missing tokenizer_config.json: $(TOKENIZER)"; exit 1; }
	@test -d "$(PPL_DATASET)" || { echo "missing PPL_DATASET: $(PPL_DATASET)"; exit 1; }

check-inputs: check-benchmark-inputs check-ppl-inputs check-pq-ppl-inputs
	@echo "[OK] All required models, checkpoint files, tokenizer files, and dataset are present"

# On Kunpeng/aarch64, force armv8.2-a+fp16+dotprod so pq-gemv-arm.cpp
# compiles the FP16 vector + SDOT paths (not scalar / vmull fallbacks).
ifeq ($(UNAME_M),aarch64)
GGML_CPU_ARM_ARCH ?= armv8.2-a+fp16+dotprod
CMAKE_ARM_ARCH_ARG := -DGGML_CPU_ARM_ARCH="$(GGML_CPU_ARM_ARCH)"
else
CMAKE_ARM_ARCH_ARG :=
endif

$(LLAMA_BUILD)/build.ninja: $(LLAMA_SRC)/CMakeLists.txt
	$(SYSTEM_BUILD_ENV) $(CMAKE) -S "$(LLAMA_SRC)" -B "$(LLAMA_BUILD)" -G Ninja \
		-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=$(GGML_CUDA) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_UI=OFF \
		-DLLAMA_OPENSSL=OFF -DCMAKE_C_COMPILER="$(CC)" -DCMAKE_CXX_COMPILER="$(CXX)" \
		$(CMAKE_ARM_ARCH_ARG) $(CUDA_COMPILER_ARG)

llama-build: $(LLAMA_BUILD)/build.ninja
	$(SYSTEM_BUILD_ENV) $(NINJA) -C "$(LLAMA_BUILD)" \
		llama llama-perplexity llama-quantize pq-selftest llama-pq-chat llama-pq-convert -j$$(nproc)
	@echo "[OK] Runtime and PQ tools: $(LLAMA_BUILD)/bin"

llama_pq: llama_pq.cpp llama-build
	$(SYSTEM_BUILD_ENV) $(CXX) $(CPPFLAGS) $(LLAMA_LDFLAGS) -o $@ $< $(LLAMA_LDLIBS)

prepare-chat-model:
	@if test "$(CHAT_MODEL)" = "$(PQ_MODEL)"; then \
		$(MAKE) prepare-edgepq; \
	elif test -f "$(CHAT_MODEL)"; then \
		echo "[REUSE] User-provided chat model: $(CHAT_MODEL)"; \
	else \
		echo "[ERROR] CHAT_MODEL does not exist: $(CHAT_MODEL)" >&2; \
		exit 1; \
	fi

chat: prepare-chat-model llama-build
	env -u GGML_NODE_TIMING -u GGML_PQ_DISABLE_ATTN -u GGML_PQ_DISABLE_FFN \
		-u GGML_PQ_DISABLE_S1 -u GGML_PQ_DISABLE_S2 -u GGML_PQ_NO_FUSE -u GGML_PQ_NO_QKV \
		GGML_PQ_STRIPE=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores \
		taskset -c "$(CHAT_CPU_RANGE)" "$(LLAMA_BUILD)/bin/llama-pq-chat" \
		--model "$(CHAT_MODEL)" --mode "$(CHAT_MODE)" \
		--threads "$(CHAT_THREADS)" --ctx-size "$(CHAT_CONTEXT)" \
		--max-tokens "$(CHAT_MAX_TOKENS)" --system "$(CHAT_SYSTEM)"

selftest: llama-build
	env -u GGML_NODE_TIMING -u GGML_PQ_DISABLE_ATTN -u GGML_PQ_DISABLE_FFN \
		-u GGML_PQ_DISABLE_S1 -u GGML_PQ_DISABLE_S2 -u GGML_PQ_NO_FUSE -u GGML_PQ_NO_QKV \
		"$(LLAMA_BUILD)/bin/pq-selftest"
	@echo "[OK] PQ correctness self-test passed"

smoke: check-benchmark-inputs llama_pq
	env -u GGML_NODE_TIMING GGML_PQ_STRIPE=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores \
		./llama_pq --fp16 "$(FP16_MODEL)" --q2 "$(Q2_MODEL)" --pq "$(PQ_MODEL)" \
		--threads "$(AE_THREADS)" --repetitions 1 --warmup 1 \
		--generations 8 --context "$(AE_CONTEXT)" --numa "$(AE_NUMA)" \
		--output /dev/null
	@echo "[OK] F16, Q2_K, and EdgePQ smoke test passed"

benchmark: check-benchmark-inputs llama_pq
	mkdir -p "$(OUTPUT_DIR)"
	env -u GGML_NODE_TIMING -u GGML_PQ_S1_PARTIALS \
		GGML_PQ_STRIPE=1 GGML_PQ_NUMA_LUT="$(GGML_PQ_NUMA_LUT)" \
		OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores \
		./llama_pq --fp16 "$(FP16_MODEL)" --q2 "$(Q2_MODEL)" --pq "$(PQ_MODEL)" \
		--threads "$(AE_THREADS)" --repetitions "$(AE_REPETITIONS)" \
		--warmup "$(AE_WARMUP)" --generations "$(AE_GENERATIONS)" \
		--context "$(AE_CONTEXT)" --numa "$(AE_NUMA)" \
		--output "$(THROUGHPUT_OUTPUT)"

ppl: env check-ppl-inputs llama-build
	mkdir -p "$(OUTPUT_DIR)"
	CUDA_VISIBLE_DEVICES="$(CUDA_DEVICE)" $(PYTHON) ppl_gguf_compare.py gguf \
		--binary "$(LLAMA_BUILD)/bin/llama-perplexity" \
		--dataset "$(PPL_DATASET)" --context "$(GGUF_PPL_CONTEXT)" \
		--stride "$(GGUF_PPL_STRIDE)" --threads "$(PPL_THREADS)" \
		--numa "$(PPL_NUMA)" --device CUDA0 --output "$(PPL_OUTPUT)" \
		--model "F16=$(FP16_MODEL)" --model "Q2_K=$(Q2_MODEL)"

pq-ppl: env check-pq-ppl-inputs
	mkdir -p "$(OUTPUT_DIR)"
	CUDA_VISIBLE_DEVICES="$(CUDA_DEVICE)" $(PYTHON) ppl_gguf_compare.py pq \
		--checkpoint "$(PQ_CHECKPOINT)" --dataset "$(PPL_DATASET)" \
		--tokenizer "$(TOKENIZER)" --context "$(PQ_PPL_CONTEXT)" \
		--stride "$(PQ_PPL_STRIDE)" --device cuda:0 \
		--output "$(PPL_OUTPUT)" --append

plot: env
	$(PYTHON) plot_results.py --throughput "$(THROUGHPUT_OUTPUT)" \
		--perplexity "$(PPL_OUTPUT)" --output-dir "$(OUTPUT_DIR)"
	@echo "[OK] Charts rendered in $(OUTPUT_DIR)/"

kunpeng-check:
	@test "$(UNAME_M)" = "aarch64" || { \
		echo "[ERROR] kunpeng-* targets require aarch64 (got $(UNAME_M))" >&2; \
		exit 1; \
	}
	@echo "[OK] arch=$(UNAME_M) nproc=$(NPROC) GGML_CUDA=$(GGML_CUDA) AE_THREADS=$(AE_THREADS)"
	@echo "[OK] CHAT_CPU_RANGE=$(CHAT_CPU_RANGE) AE_NUMA=$(AE_NUMA)"
	@echo "[OK] GGML_CPU_ARM_ARCH=$(GGML_CPU_ARM_ARCH)"
	@command -v $(CMAKE) >/dev/null || { echo "[ERROR] cmake not found" >&2; exit 1; }
	@command -v $(NINJA) >/dev/null || { echo "[ERROR] ninja not found" >&2; exit 1; }
	@command -v $(CXX) >/dev/null || { echo "[ERROR] C++ compiler not found: $(CXX)" >&2; exit 1; }
	@(grep -E 'Features|Flags' /proc/cpuinfo 2>/dev/null | head -3) || true
	@echo "[OK] Host toolchain ready for Kunpeng NEON PQ path (pq-gemv-arm.cpp)"
	@grep -q 'int32_t iacc\[kS1OutTile\]' llama.cpp/ggml/src/ggml-cpu/pq-gemv-arm.cpp && { \
		echo "[ERROR] pq-gemv-arm.cpp still has int32 phase2 (expect ~7 tok/s). Use float pq_fma_lut16_f32 path (see 6b942c9)." >&2; \
		exit 1; \
	} || true
	@grep -q 'pq_fma_lut16_f32' llama.cpp/ggml/src/ggml-cpu/pq-gemv-arm.cpp || { \
		echo "[ERROR] pq-gemv-arm.cpp missing pq_fma_lut16_f32 in shared phase2" >&2; \
		exit 1; \
	}
	@echo "[OK] PQ shared phase2 source check (float FMA, not int32 iacc)"

kunpeng-verify-isa: kunpeng-build
	@test -f "$(LLAMA_BUILD)/compile_commands.json" || { \
		echo "[ERROR] missing $(LLAMA_BUILD)/compile_commands.json (reconfigure with EXPORT_COMPILE_COMMANDS)" >&2; \
		exit 1; \
	}
	@echo "[CHECK] pq-gemv-arm.cpp march flags:"
	@python3 -c 'import json,sys; p="$(LLAMA_BUILD)/compile_commands.json"; \
		cmds=json.load(open(p)); \
		hits=[c for c in cmds if "pq-gemv-arm.cpp" in c.get("file","")]; \
		sys.exit("[ERROR] pq-gemv-arm.cpp not in compile_commands.json") if not hits else None; \
		cmd=hits[0].get("command") or " ".join(hits[0].get("arguments",[])); \
		print(cmd); \
		need=("fp16","dotprod"); \
		missing=[x for x in need if x not in cmd]; \
		sys.exit("[ERROR] missing ISA flags in march: "+",".join(missing)) if missing else print("[OK] fp16+dotprod present in compile flags")'

kunpeng-build: kunpeng-check
	$(MAKE) llama-build GGML_CUDA=OFF

kunpeng-selftest: kunpeng-build
	$(MAKE) selftest GGML_CUDA=OFF

kunpeng-smoke: kunpeng-check
	$(MAKE) smoke GGML_CUDA=OFF AE_THREADS="$(AE_THREADS)" AE_NUMA="$(AE_NUMA)"

kunpeng-benchmark: kunpeng-check
	$(MAKE) benchmark GGML_CUDA=OFF AE_THREADS="$(AE_THREADS)" AE_NUMA="$(AE_NUMA)" \
		AE_REPETITIONS="$(AE_REPETITIONS)" AE_GENERATIONS="$(AE_GENERATIONS)"

kunpeng-chat: kunpeng-check
	$(MAKE) chat GGML_CUDA=OFF CHAT_THREADS="$(CHAT_THREADS)" \
		CHAT_CPU_RANGE="$(CHAT_CPU_RANGE)"

# Inspect pq_meta modes in a GGUF (EdgePQ-4c8b should be all mode=4 / scaled S1).
kunpeng-pq-meta:
	@if test -x "$(PYTHON)"; then \
		$(PYTHON) scripts/kunpeng_pq_meta_check.py "$(PQ_MODEL)"; \
	else \
		python3 scripts/kunpeng_pq_meta_check.py "$(PQ_MODEL)"; \
	fi

# A/B: online S2 re-quant from F16 (NOT trained EdgePQ-4c8b codebooks).
PQ_S2_MODEL ?= $(MODEL_DIR)/base-pq-s2-kunpeng.gguf
PQ_K64_MODEL ?= $(MODEL_DIR)/base-pq-k64-kunpeng.gguf
PQ_MIX_MODEL ?= $(MODEL_DIR)/base-pq-mix-k64-kunpeng.gguf

kunpeng-export-s2: llama-build
	@test -f "$(FP16_MODEL)" || { echo "[ERROR] missing F16 model: $(FP16_MODEL)" >&2; exit 1; }
	$(LLAMA_BUILD)/bin/llama-pq-convert "$(FP16_MODEL)" "$(PQ_S2_MODEL)" \
		--pq-target kunpeng --pq-mode s2 --pq-ds 2 --pq-k 256 --train-from "$(FP16_MODEL)"
	@echo "[OK] wrote $(PQ_S2_MODEL) (online S2 K=256; PPL != EdgePQ-4c8b)"

# Phase2A: ARM-friendly scaled S1 K=64 (single vqtbl4q), ds=4, mode-4 side tensors.
kunpeng-export-k64: llama-build
	@test -f "$(FP16_MODEL)" || { echo "[ERROR] missing F16 model: $(FP16_MODEL)" >&2; exit 1; }
	$(LLAMA_BUILD)/bin/llama-pq-convert "$(FP16_MODEL)" "$(PQ_K64_MODEL)" \
		--pq-mode s1 --pq-ds 4 --pq-k 64 --pq-scaled --train-from "$(FP16_MODEL)"
	@echo "[OK] wrote $(PQ_K64_MODEL) (scaled S1 K=64 ds=4; run kunpeng-pq-k64-accept for PPL)"

kunpeng-benchmark-k64: kunpeng-check kunpeng-export-k64 llama_pq
	mkdir -p "$(OUTPUT_DIR)"
	env -u GGML_NODE_TIMING -u GGML_PQ_S1_SHARED \
		GGML_PQ_STRIPE=1 GGML_PQ_NUMA_LUT="$(GGML_PQ_NUMA_LUT)" \
		OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores \
		./llama_pq --fp16 "$(FP16_MODEL)" --q2 "$(Q2_MODEL)" --pq "$(PQ_K64_MODEL)" \
		--threads "$(AE_THREADS)" --repetitions "$(AE_REPETITIONS)" \
		--warmup "$(AE_WARMUP)" --generations "$(AE_GENERATIONS)" \
		--context "$(AE_CONTEXT)" --numa "$(AE_NUMA)" \
		--output "$(OUTPUT_DIR)/throughput_k64.csv"

kunpeng-pq-k64-accept: kunpeng-check kunpeng-export-k64 env check-ppl-inputs llama-build
	mkdir -p "$(OUTPUT_DIR)"
	@test -f "$(PQ_MODEL)" || { echo "[ERROR] missing baseline PQ: $(PQ_MODEL)" >&2; exit 1; }
	$(PYTHON) ppl_gguf_compare.py compare \
		--binary "$(LLAMA_BUILD)/bin/llama-perplexity" \
		--dataset "$(PPL_DATASET)" --context "$(PQ_PPL_CONTEXT)" \
		--stride "$(PQ_PPL_STRIDE)" --threads "$(PPL_THREADS)" \
		--numa "$(PPL_NUMA)" --device CPU \
		--baseline "PQ-4c8b=$(PQ_MODEL)" \
		--candidate "PQ-K64=$(PQ_K64_MODEL)" \
		--max-delta 0.2 \
		--output "$(OUTPUT_DIR)/perplexity_k64_accept.csv"

# Phase2B: auto mix (ffn_down->S2, rest S1) with K=64 for Kunpeng A/B.
kunpeng-export-mix: llama-build
	@test -f "$(FP16_MODEL)" || { echo "[ERROR] missing F16 model: $(FP16_MODEL)" >&2; exit 1; }
	$(LLAMA_BUILD)/bin/llama-pq-convert "$(FP16_MODEL)" "$(PQ_MIX_MODEL)" \
		--pq-mode auto --pq-ds 2 --pq-k 64 --train-from "$(FP16_MODEL)"
	@echo "[OK] wrote $(PQ_MIX_MODEL) (auto S1/S2 K=64; measure PPL vs 4c8b)"

# Phase0: thread sweep with GGML_PQ_S1_PARTIALS unset (stable-bench script).
kunpeng-thread-sweep:
	@chmod +x scripts/kunpeng_stable_bench.sh
	env -u GGML_PQ_S1_PARTIALS THREADS_LIST="$(or $(THREADS_LIST),16,24,32,48,60)" \
		AE_NUMA="$(AE_NUMA)" AE_REPETITIONS="$(AE_REPETITIONS)" \
		./scripts/kunpeng_stable_bench.sh

clean:
	rm -f llama_pq

distclean: clean
	rm -rf "$(LLAMA_BUILD)" "$(VENV)" "$(OUTPUT_DIR)"
