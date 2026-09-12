CXX ?= c++
NVCC ?= nvcc
# -march=native lets the host compiler use ADX/MULX for the field-arithmetic
# carry chains on x86; harmless on arm64. Only affects the host binaries.
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall
# fatbin for Turing..Blackwell; native-only builds are faster to compile: make GPU_ARCH="-arch=native"
# sm_100 (datacenter Blackwell) / sm_120 (consumer Blackwell, e.g. RTX 5090)
# need CUDA >= 12.8; on an older toolkit override GPU_ARCH (the fleet uses -arch=native).
GPU_ARCH ?= -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_89,code=sm_89 \
            -gencode arch=compute_90,code=sm_90 \
            -gencode arch=compute_100,code=sm_100 \
            -gencode arch=compute_120,code=sm_120 \
            -gencode arch=compute_120,code=compute_120
# -Xptxas -v prints per-kernel register/spill counts, needed to tune launch
# bounds and confirm local-memory changes.
NVCCFLAGS ?= -O3 -std=c++17 -Xptxas -v $(GPU_ARCH)

all: host

host: bin/test_host bin/cpu_vanity

gpu: bin/gpu_vanity

bin:
	mkdir -p bin

bin/test_host: src/test_host.cpp src/*.h | bin
	$(CXX) $(CXXFLAGS) -o $@ $<

bin/cpu_vanity: src/cpu_main.cpp src/*.h | bin
	$(CXX) $(CXXFLAGS) -o $@ $<

bin/gpu_vanity: src/main.cu src/*.h | bin
	$(NVCC) $(NVCCFLAGS) -o $@ $<

table.bin: tools/gen_table.py tools/ed25519_ref.py
	python3 tools/gen_table.py $@

test: host table.bin
	python3 -m pytest tests/ -v

clean:
	rm -rf bin

.PHONY: all host gpu test clean
