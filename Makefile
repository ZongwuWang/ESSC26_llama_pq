SHELL := /bin/bash

CXX := /usr/bin/g++
CC := /usr/bin/gcc
CMAKE ?= cmake
NINJA ?= ninja
UV ?= uv

VENV ?= .venv
PYTHON := $(VENV)/bin/python
HF := $(VENV)/bin/hf

LLAMA_SRC ?= llama.cpp
LLAMA_BUILD ?= build
OUTPUT_DIR ?= output
MODEL_DIR ?= models
DATASET_DIR ?= datasets

EDGE_PQ_REPO ?= ZongwuWang/EdgePQ-4c8b
EDGE_PQ_REVISION ?= 574ece53e87a10c12a2ea299996cf7622989634d
Q2_REPO ?= second-state/Llama-2-7B-Chat-GGUF
Q2_REVISION ?= 064fe43ea8c1e1f93477ef4a170bdc2b244ef02c
FP16_FILE ?= Llama-2-7b-chat-hf-f16.gguf
Q2_FILE ?= Llama-2-7b-chat-hf-Q2_K.gguf
PQ_MODEL ?= $(MODEL_DIR)/base-pq-4c8b.gguf
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

GGUF_PPL_CONTEXT ?= 3072
GGUF_PPL_STRIDE ?= 2048
PQ_PPL_CONTEXT ?= 4096
PQ_PPL_STRIDE ?= 4096
PPL_THREADS ?= 60
PPL_NUMA ?= distribute

THROUGHPUT_OUTPUT := $(OUTPUT_DIR)/throughput.csv
PPL_OUTPUT := $(OUTPUT_DIR)/perplexity.csv

SYSTEM_BUILD_ENV := env -u CFLAGS -u CXXFLAGS -u LDFLAGS \
	-u LD_LIBRARY_PATH -u LIBRARY_PATH -u CPATH \
	-u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u GCC_EXEC_PREFIX

CPPFLAGS := -O2 -std=c++17 -Wall -Wextra -pedantic
CPPFLAGS += -I$(LLAMA_SRC)/include -I$(LLAMA_SRC)/ggml/include
LLAMA_LDFLAGS := -L$(LLAMA_BUILD)/bin -Wl,-rpath,$(abspath $(LLAMA_BUILD)/bin)
LLAMA_LDLIBS := -lllama -lggml -lggml-cpu -lggml-base -lpthread -ldl -lm

.PHONY: help all env prepare-edgepq prepare-baselines prepare-dataset prepare-inputs \
	check-benchmark-inputs check-ppl-inputs check-pq-ppl-inputs check-inputs \
	llama-build llama_pq selftest smoke benchmark ppl pq-ppl plot clean distclean

help:
	@echo "EdgePQ artifact targets:"
	@echo "  make env            Create the locked Python environment"
	@echo "  make prepare-inputs Download all public models and WikiText-2"
	@echo "  make check-inputs   Validate all required external inputs"
	@echo "  make llama-build    Build llama.cpp, PPL, quantizer, and PQ tools"
	@echo "  make selftest       Run the PQ correctness self-test"
	@echo "  make smoke          Run an 8-token three-model smoke test"
	@echo "  make benchmark      Reproduce decode throughput CSV"
	@echo "  make ppl            Evaluate F16 and Q2_K perplexity"
	@echo "  make pq-ppl         Evaluate PQ reconstruction perplexity"
	@echo "  make all            Run the complete paper evaluation"
	@echo "  make plot           Render throughput and PPL charts"

all: env
	$(MAKE) prepare-inputs
	$(MAKE) check-inputs
	$(MAKE) llama-build
	$(MAKE) selftest
	$(MAKE) benchmark
	$(MAKE) ppl
	$(MAKE) pq-ppl

env: uv.lock pyproject.toml
	$(UV) sync --frozen

prepare-edgepq: env
	mkdir -p "$(MODEL_DIR)"
	$(HF) download "$(EDGE_PQ_REPO)" base-pq-4c8b.gguf \
		--revision "$(EDGE_PQ_REVISION)" --local-dir "$(MODEL_DIR)"
	$(HF) download "$(EDGE_PQ_REPO)" \
		--revision "$(EDGE_PQ_REVISION)" \
		--include "best-formal-hard/*" --local-dir "$(MODEL_DIR)"
	$(HF) download "$(EDGE_PQ_REPO)" tokenizer.json tokenizer.model tokenizer_config.json \
		--revision "$(EDGE_PQ_REVISION)" --local-dir "$(MODEL_DIR)/tokenizer"

prepare-baselines: env
	mkdir -p "$(MODEL_DIR)"
	$(HF) download "$(Q2_REPO)" "$(FP16_FILE)" "$(Q2_FILE)" \
		--revision "$(Q2_REVISION)" --local-dir "$(MODEL_DIR)"

prepare-dataset: env
	$(PYTHON) ppl_gguf_compare.py prepare-dataset --output "$(PPL_DATASET)"

prepare-inputs: prepare-edgepq prepare-baselines prepare-dataset

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
	@test -d "$(PQ_CHECKPOINT)/pq_states" || { echo "missing PQ states: $(PQ_CHECKPOINT)/pq_states"; exit 1; }
	@test -d "$(TOKENIZER)" || { echo "missing TOKENIZER: $(TOKENIZER)"; exit 1; }
	@test -d "$(PPL_DATASET)" || { echo "missing PPL_DATASET: $(PPL_DATASET)"; exit 1; }

check-inputs: check-benchmark-inputs check-ppl-inputs check-pq-ppl-inputs

$(LLAMA_BUILD)/build.ninja: $(LLAMA_SRC)/CMakeLists.txt
	$(SYSTEM_BUILD_ENV) $(CMAKE) -S "$(LLAMA_SRC)" -B "$(LLAMA_BUILD)" -G Ninja \
		-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=$(GGML_CUDA) \
		-DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_UI=OFF \
		-DLLAMA_OPENSSL=OFF -DCMAKE_C_COMPILER="$(CC)" -DCMAKE_CXX_COMPILER="$(CXX)"

llama-build: $(LLAMA_BUILD)/build.ninja
	$(SYSTEM_BUILD_ENV) $(NINJA) -C "$(LLAMA_BUILD)" \
		llama llama-perplexity llama-quantize pq-selftest llama-pq-convert -j$$(nproc)

llama_pq: llama_pq.cpp llama-build
	$(SYSTEM_BUILD_ENV) $(CXX) $(CPPFLAGS) $(LLAMA_LDFLAGS) -o $@ $< $(LLAMA_LDLIBS)

selftest: llama-build
	"$(LLAMA_BUILD)/bin/pq-selftest"

smoke: check-benchmark-inputs llama_pq
	GGML_PQ_STRIPE=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores \
		./llama_pq --fp16 "$(FP16_MODEL)" --q2 "$(Q2_MODEL)" --pq "$(PQ_MODEL)" \
		--threads "$(AE_THREADS)" --repetitions 1 --warmup 1 \
		--generations 8 --context "$(AE_CONTEXT)" --numa "$(AE_NUMA)" \
		--output /dev/null

benchmark: check-benchmark-inputs llama_pq
	mkdir -p "$(OUTPUT_DIR)"
	GGML_PQ_STRIPE=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores \
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

clean:
	rm -f llama_pq

distclean: clean
	rm -rf "$(LLAMA_BUILD)" "$(VENV)" "$(OUTPUT_DIR)"
