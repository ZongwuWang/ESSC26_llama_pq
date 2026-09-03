SHELL := /bin/bash

CXX := /usr/bin/g++
CC := /usr/bin/gcc
CMAKE ?= cmake
UV ?= uv
CUDACXX ?=

VENV ?= .venv
PYTHON := $(VENV)/bin/python
HF := $(VENV)/bin/hf

LLAMA_SRC ?= llama.cpp
LLAMA_BUILD ?= build
OUTPUT_DIR ?= output
MODEL_DIR ?= models
DATASET_DIR ?= datasets

EDGE_PQ_REPO ?= ZongwuWang/EdgePQ-4c8b
EDGE_PQ_REVISION ?= 15ab7f2d07f788b5b1124356fe18d77c81b0a50b
EDGE_PQ_SHA256 ?= c91e15df2ac571212fbdad3868af635026e72ba50980370b7ff3784e264253be
FP16_REPO ?= second-state/Llama-2-7B-Chat-GGUF
FP16_REVISION ?= 064fe43ea8c1e1f93477ef4a170bdc2b244ef02c
Q2_REPO ?= $(EDGE_PQ_REPO)
Q2_REVISION ?= $(EDGE_PQ_REVISION)
FP16_FILE ?= Llama-2-7b-chat-hf-f16.gguf
Q2_FILE ?= Llama-2-7b-chat-hf.Q2_K.gguf
PQ_MODEL ?= $(MODEL_DIR)/base-pq-4c8b.gguf
PQ_MODEL_VERIFY_STAMP ?= $(PQ_MODEL).verified
EDGE_PQ_AUX_VERIFY_STAMP ?= $(MODEL_DIR)/.edgepq-aux-revision
FP16_MODEL ?= $(MODEL_DIR)/$(FP16_FILE)
Q2_MODEL ?= $(MODEL_DIR)/$(Q2_FILE)
PQ_CHECKPOINT ?= $(MODEL_DIR)/best-formal-hard
PPL_DATASET ?= $(DATASET_DIR)/wikitext-2-raw-v1
TOKENIZER ?= $(MODEL_DIR)/tokenizer

CUDA_DEVICE ?= 0
GGML_CUDA ?= ON

AE_THREADS ?= 60
AE_REPETITIONS ?= 3
AE_WARMUP ?= 1
AE_GENERATIONS ?= 128,512
AE_CONTEXT ?= 2048
AE_NUMA ?= distribute

MODEL ?= pq
ifneq ($(filter $(MODEL),pq),)
CHAT_DEFAULT_MODEL := $(PQ_MODEL)
CHAT_DEFAULT_ARGS :=
CHAT_DEFAULT_TEMPERATURE := 0
else ifneq ($(filter $(MODEL),16 fp16),)
CHAT_DEFAULT_MODEL := $(FP16_MODEL)
CHAT_DEFAULT_ARGS := --no-pq
CHAT_DEFAULT_TEMPERATURE := 0.7
else ifneq ($(filter $(MODEL),k2 q2 q2_k),)
CHAT_DEFAULT_MODEL := $(Q2_MODEL)
CHAT_DEFAULT_ARGS := --no-pq
CHAT_DEFAULT_TEMPERATURE := 0.7
else
$(error Unsupported MODEL='$(MODEL)'; use pq, 16, or k2)
endif

CHAT_MODEL ?= $(CHAT_DEFAULT_MODEL)
CHAT_EXTRA_ARGS ?= $(CHAT_DEFAULT_ARGS)
CHAT_TEMPERATURE ?= $(CHAT_DEFAULT_TEMPERATURE)
CHAT_MODE ?= auto
CHAT_THREADS ?= $(AE_THREADS)
CHAT_CPU_RANGE ?= 0-59
CHAT_CONTEXT ?= 4096
CHAT_MAX_TOKENS ?= 256
CHAT_SYSTEM ?=

GGUF_PPL_CONTEXT ?= 3072
GGUF_PPL_STRIDE ?= 2048
PQ_PPL_CONTEXT ?= 4096
PQ_PPL_STRIDE ?= 4096
PPL_THREADS ?= 60
PPL_NUMA ?= distribute

THROUGHPUT_OUTPUT := $(OUTPUT_DIR)/throughput.csv
PPL_OUTPUT := $(OUTPUT_DIR)/perplexity.csv
RESULTS_CACHE_STAMP := $(OUTPUT_DIR)/.results-cache-key
RESULTS_PROTOCOL_VERSION ?= 2
RESULTS_CACHE_KEY = edgepq=$(EDGE_PQ_REVISION);fp16=$(FP16_REVISION);q2=$(Q2_REVISION);dataset=b08601e04326c79dfdd32d625aee71d232d685c3;protocol=$(RESULTS_PROTOCOL_VERSION);threads=$(AE_THREADS);repetitions=$(AE_REPETITIONS);warmup=$(AE_WARMUP);generations=$(AE_GENERATIONS);ae_context=$(AE_CONTEXT);ae_numa=$(AE_NUMA);gguf_ppl=$(GGUF_PPL_CONTEXT)/$(GGUF_PPL_STRIDE)/$(PPL_THREADS)/$(PPL_NUMA);pq_ppl=$(PQ_PPL_CONTEXT)/$(PQ_PPL_STRIDE)

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

.PHONY: help all demo env prepare-edgepq-model prepare-edgepq-assets prepare-edgepq \
	prepare-fp16 prepare-q2 prepare-baselines prepare-dataset prepare-inputs \
	check-benchmark-inputs check-ppl-inputs check-pq-ppl-inputs check-inputs \
	llama-configure llama-build llama_pq prepare-chat-model chat selftest smoke benchmark ppl pq-ppl \
	record-results-cache plot clean distclean

help:
	@echo "EdgePQ artifact targets:"
	@echo "  make help           List the public artifact targets"
	@echo "  make env            Create the locked Python environment"
	@echo "  make prepare-inputs Download all public models and WikiText-2"
	@echo "  make check-inputs   Validate all required external inputs"
	@echo "  make llama-build    Build llama.cpp, PPL, quantizer, and PQ tools"
	@echo "  make selftest       Run the PQ correctness self-test"
	@echo "  make smoke          Run an 8-token three-model smoke test"
	@echo "  make chat MODEL=pq  Chat with pq, 16 (FP16), or k2 (Q2_K)"
	@echo "  make demo           Run cache-aware end-to-end reproduction"
	@echo "  make benchmark      Reproduce decode throughput CSV"
	@echo "  make ppl            Evaluate F16 and Q2_K perplexity"
	@echo "  make pq-ppl         Evaluate PQ reconstruction perplexity"
	@echo "  make all            Rerun the complete paper evaluation"
	@echo "  make plot           Render throughput and PPL charts"


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
	$(MAKE) record-results-cache
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
	@if test -s "$(THROUGHPUT_OUTPUT)" && test -s "$(PPL_OUTPUT)" && \
		test -f "$(RESULTS_CACHE_STAMP)" && \
		grep -Fqx '$(RESULTS_CACHE_KEY)' "$(RESULTS_CACHE_STAMP)"; then \
		echo "[REUSE] $(THROUGHPUT_OUTPUT)"; \
		echo "[REUSE] $(PPL_OUTPUT)"; \
	else \
		echo "[RUN] Cached CSVs are missing or stale; continuing with the full evaluation"; \
		$(MAKE) benchmark; \
		$(MAKE) ppl; \
		$(MAKE) pq-ppl; \
		$(MAKE) record-results-cache; \
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

prepare-edgepq-model: env
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

prepare-edgepq-assets: prepare-edgepq-model
	mkdir -p "$(MODEL_DIR)"
	@count=$$(find "$(PQ_CHECKPOINT)/pq_states" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l | tr -d ' '); \
	if test -f "$(EDGE_PQ_AUX_VERIFY_STAMP)" && \
		grep -Fqx '$(EDGE_PQ_REVISION)' "$(EDGE_PQ_AUX_VERIFY_STAMP)" && \
		test -f "$(PQ_CHECKPOINT)/non_pq_state.pt" && test "$$count" -eq 224; then \
		echo "[REUSE] EdgePQ checkpoint: $(PQ_CHECKPOINT) (224/224 PQ states)"; \
	else \
		echo "[DOWNLOAD] Missing or stale EdgePQ checkpoint from $(EDGE_PQ_REPO)@$(EDGE_PQ_REVISION)"; \
		$(HF) download "$(EDGE_PQ_REPO)" \
			--revision "$(EDGE_PQ_REVISION)" \
			--include "best-formal-hard/*" --local-dir "$(MODEL_DIR)"; \
	fi
	@if test -f "$(EDGE_PQ_AUX_VERIFY_STAMP)" && \
		grep -Fqx '$(EDGE_PQ_REVISION)' "$(EDGE_PQ_AUX_VERIFY_STAMP)" && \
		test -f "$(TOKENIZER)/tokenizer.json" && \
		test -f "$(TOKENIZER)/tokenizer.model" && \
		test -f "$(TOKENIZER)/tokenizer_config.json"; then \
		echo "[REUSE] Llama-2 tokenizer: $(TOKENIZER)"; \
	else \
		echo "[DOWNLOAD] Missing or stale Llama-2 tokenizer from $(EDGE_PQ_REPO)@$(EDGE_PQ_REVISION)"; \
		$(HF) download "$(EDGE_PQ_REPO)" tokenizer.json tokenizer.model tokenizer_config.json \
			--revision "$(EDGE_PQ_REVISION)" --local-dir "$(TOKENIZER)"; \
	fi
	@count=$$(find "$(PQ_CHECKPOINT)/pq_states" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l | tr -d ' '); \
		test -f "$(PQ_CHECKPOINT)/non_pq_state.pt" && test "$$count" -eq 224 && \
		test -f "$(TOKENIZER)/tokenizer.json" && test -f "$(TOKENIZER)/tokenizer.model" && \
		test -f "$(TOKENIZER)/tokenizer_config.json" || { \
			echo "[ERROR] EdgePQ checkpoint or tokenizer preparation is incomplete" >&2; exit 1; \
		}; \
		printf '%s\n' '$(EDGE_PQ_REVISION)' > "$(EDGE_PQ_AUX_VERIFY_STAMP)"; \
		echo "[OK] Recorded EdgePQ checkpoint/tokenizer revision"

prepare-edgepq: prepare-edgepq-assets

prepare-fp16: env
	mkdir -p "$(MODEL_DIR)"
	@if test -f "$(FP16_MODEL)"; then \
		echo "[REUSE] FP16 baseline: $(FP16_MODEL)"; \
	else \
		echo "[DOWNLOAD] FP16 baseline from $(FP16_REPO)@$(FP16_REVISION)"; \
		$(HF) download "$(FP16_REPO)" "$(FP16_FILE)" \
			--revision "$(FP16_REVISION)" --local-dir "$(MODEL_DIR)"; \
	fi

prepare-q2: env
	mkdir -p "$(MODEL_DIR)"
	@if test -f "$(Q2_MODEL)"; then \
		echo "[REUSE] Q2_K baseline: $(Q2_MODEL)"; \
	else \
		echo "[DOWNLOAD] Q2_K baseline from $(Q2_REPO)@$(Q2_REVISION)"; \
		$(HF) download "$(Q2_REPO)" "$(Q2_FILE)" \
			--revision "$(Q2_REVISION)" --local-dir "$(MODEL_DIR)"; \
	fi

prepare-baselines: prepare-fp16 prepare-q2

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

llama-configure:
	@set -eu; \
	cache="$(LLAMA_BUILD)/CMakeCache.txt"; \
	if test -f "$$cache"; then \
		cached_source=$$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$$cache"); \
		cached_source=$$(realpath -m "$$cached_source"); \
		expected_source=$$(realpath -m "$(LLAMA_SRC)"); \
		if test -n "$$cached_source" && test "$$cached_source" != "$$expected_source"; then \
			echo "[STALE] CMake cache belongs to $$cached_source"; \
			echo "        reconfiguring for $$expected_source"; \
			$(CMAKE) -E rm -f "$$cache"; \
			$(CMAKE) -E rm -rf "$(LLAMA_BUILD)/CMakeFiles"; \
		fi; \
	fi
	$(SYSTEM_BUILD_ENV) $(CMAKE) -S "$(LLAMA_SRC)" -B "$(LLAMA_BUILD)" -G Ninja \
		-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=$(GGML_CUDA) \
		-DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_UI=OFF \
		-DLLAMA_OPENSSL=OFF -DCMAKE_C_COMPILER="$(CC)" -DCMAKE_CXX_COMPILER="$(CXX)" $(CUDA_COMPILER_ARG)

llama-build: llama-configure
	$(SYSTEM_BUILD_ENV) $(CMAKE) --build "$(LLAMA_BUILD)" --target \
		llama llama-perplexity llama-quantize pq-selftest llama-pq-chat llama-pq-convert \
		--parallel $$(nproc)
	@echo "[OK] Runtime and PQ tools: $(LLAMA_BUILD)/bin"

llama_pq: llama_pq.cpp llama-build
	$(SYSTEM_BUILD_ENV) $(CXX) $(CPPFLAGS) $(LLAMA_LDFLAGS) -o $@ $< $(LLAMA_LDLIBS)

prepare-chat-model:
	@if test "$(CHAT_MODEL)" = "$(PQ_MODEL)"; then \
		$(MAKE) prepare-edgepq-model; \
	elif test "$(CHAT_MODEL)" = "$(FP16_MODEL)"; then \
		$(MAKE) prepare-fp16; \
	elif test "$(CHAT_MODEL)" = "$(Q2_MODEL)"; then \
		$(MAKE) prepare-q2; \
	elif test -f "$(CHAT_MODEL)"; then \
		echo "[REUSE] User-provided chat model: $(CHAT_MODEL)"; \
	else \
		echo "[ERROR] CHAT_MODEL does not exist: $(CHAT_MODEL)" >&2; \
		exit 1; \
	fi

chat: GGML_CUDA := OFF
chat: prepare-chat-model llama-build
	env -u GGML_NODE_TIMING -u GGML_PQ_DISABLE_ATTN -u GGML_PQ_DISABLE_FFN \
		-u GGML_PQ_DISABLE_S1 -u GGML_PQ_DISABLE_S2 -u GGML_PQ_NO_FUSE -u GGML_PQ_NO_QKV \
		GGML_PQ_STRIPE=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores \
		taskset -c "$(CHAT_CPU_RANGE)" "$(LLAMA_BUILD)/bin/llama-pq-chat" \
		--model "$(CHAT_MODEL)" --mode "$(CHAT_MODE)" \
		--threads "$(CHAT_THREADS)" --ctx-size "$(CHAT_CONTEXT)" \
		--max-tokens "$(CHAT_MAX_TOKENS)" --temp "$(CHAT_TEMPERATURE)" \
		--system "$(CHAT_SYSTEM)" $(CHAT_EXTRA_ARGS)

selftest: llama-build
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
	env -u GGML_NODE_TIMING GGML_PQ_STRIPE=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores \
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

record-results-cache:
	mkdir -p "$(OUTPUT_DIR)"
	@printf '%s\n' '$(RESULTS_CACHE_KEY)' > "$(RESULTS_CACHE_STAMP)"
	@echo "[OK] Recorded result cache key: $(RESULTS_CACHE_STAMP)"

plot: env
	$(PYTHON) plot_results.py --throughput "$(THROUGHPUT_OUTPUT)" \
		--perplexity "$(PPL_OUTPUT)" --output-dir "$(OUTPUT_DIR)"
	@echo "[OK] Charts rendered in $(OUTPUT_DIR)/"

clean:
	rm -f llama_pq

distclean: clean
	rm -rf "$(LLAMA_BUILD)" "$(VENV)" "$(OUTPUT_DIR)"
