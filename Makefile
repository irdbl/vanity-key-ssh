CXX ?= c++
NVCC ?= nvcc
CXXFLAGS ?= -O2 -std=c++17 -Wall
# fatbin for Turing..Blackwell; native-only builds are faster to compile: make GPU_ARCH="-arch=native"
GPU_ARCH ?= -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_89,code=sm_89 \
            -gencode arch=compute_90,code=sm_90 \
            -gencode arch=compute_90,code=compute_90
NVCCFLAGS ?= -O3 -std=c++17 $(GPU_ARCH)

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
