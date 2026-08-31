CXX := /usr/bin/g++
LLAMA_SRC ?= llama.cpp
LLAMA_BUILD ?= build
FP16_MODEL ?= ../../models/Llama-2-7b-chat-hf.gguf
Q2_MODEL ?= ../../models/Llama-2-7b-chat-hf-GGUF/Llama-2-7b-chat-hf.Q2_K.gguf
PQ_MODEL ?= ../../llama.cpp/base-pq-4c8b.gguf
AE_THREADS ?= 60
AE_REPETITIONS ?= 3
AE_WARMUP ?= 1
AE_GENERATIONS ?= 128,512
AE_CONTEXT ?= 2048
AE_NUMA ?= distribute
AE_OUTPUT ?= essc-pq-comparison.csv

SYSTEM_BUILD_ENV := env -u CC -u CXX -u CFLAGS -u CXXFLAGS -u LDFLAGS \
	-u LD_LIBRARY_PATH -u LIBRARY_PATH -u CPATH \
	-u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u GCC_EXEC_PREFIX \
	PATH=/usr/local/bin:/usr/bin:/bin:/home/wangzongwu/miniconda3/envs/million/bin
CPPFLAGS := -O2 -std=c++17 -Wall -Wextra -pedantic
CPPFLAGS += -I$(LLAMA_SRC)/include -I$(LLAMA_SRC)/ggml/include
LLAMA_LDFLAGS := -L$(LLAMA_BUILD)/bin -Wl,-rpath,$(abspath $(LLAMA_BUILD)/bin)
LLAMA_LDLIBS := -lllama -lggml -lggml-cpu -lggml-base -lpthread -ldl -lm

.PHONY: all benchmark llama-build clean

all: llama_pq

llama-build: $(LLAMA_BUILD)/bin/libllama.so

$(LLAMA_BUILD)/bin/libllama.so: $(LLAMA_SRC)/CMakeLists.txt
	$(SYSTEM_BUILD_ENV) cmake -S $(LLAMA_SRC) -B $(LLAMA_BUILD) -G Ninja \
	  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_BUILD_TOOLS=OFF \
	  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
	  -DCMAKE_ASM_COMPILER=/usr/bin/gcc
	$(SYSTEM_BUILD_ENV) ninja -C $(LLAMA_BUILD) llama -j$$(nproc)

llama_pq: llama_pq.cpp llama-build
	$(SYSTEM_BUILD_ENV) $(CXX) $(CPPFLAGS) $(LLAMA_LDFLAGS) -o $@ $< $(LLAMA_LDLIBS)

benchmark: llama_pq
	GGML_PQ_STRIPE=1 OMP_DYNAMIC=FALSE \
	OMP_PROC_BIND=spread OMP_PLACES=cores \
	./llama_pq \
	  --fp16 "$(FP16_MODEL)" \
	  --q2 "$(Q2_MODEL)" \
	  --pq "$(PQ_MODEL)" \
	  --threads "$(AE_THREADS)" \
	  --repetitions "$(AE_REPETITIONS)" \
	  --warmup "$(AE_WARMUP)" \
	  --generations "$(AE_GENERATIONS)" \
	  --context "$(AE_CONTEXT)" \
	  --numa "$(AE_NUMA)" \
	  --output "$(AE_OUTPUT)"

clean:
	rm -f llama_pq
