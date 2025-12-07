# Barracuda Go Server - targets defined below with CUDA support

.PHONY: build run clean
BENCHMARK = bin/benchmark

# Default target
all: directories $(LIBRARY) $(BENCHMARK)

# Create directories
directories:
	@mkdir -p $(OBJDIR) $(LIBDIR) bin

# Compile C++ source
$(CPP_OBJECTS): $(CPP_SOURCES) $(SRCDIR)/barracuda_engine.hpp
	$(CXX) -O3 -fPIC $(INCLUDES) -c $< -o $@

# Compile CUDA kernels
$(CU_OBJECTS): $(CU_SOURCES) $(SRCDIR)/barracuda_engine.hpp
	$(CC) $(CFLAGS) -Xcompiler -fPIC $(INCLUDES) -c $< -o $@

# Create shared library
$(LIBRARY): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CC) -shared -o $@ $^ $(LIBS)

# Build benchmark executable
$(BENCHMARK): src/benchmark.cpp $(LIBRARY)
	$(CC) $(CFLAGS) $(INCLUDES) -L$(LIBDIR) -o $@ src/benchmark.cpp -lbarracuda $(LIBS)

# Go module build
# Build Go binary with CUDA support
build: $(LIBRARY)
	CGO_ENABLED=1 CGO_LDFLAGS="-L./lib -lbarracuda" go build -o barracuda ./cmd/server

# Build library target alias
build-lib: $(LIBRARY)

# Build Go binary without CUDA (mock mode)
build-mock:
	go build -o barracuda ./cmd/server

# Run the barracuda server
run:
	make build && ./barracuda

# Test targets
test-cpp: tests/bin/barracuda_test
	@echo "🧪 Running C++ Tests..."
	./tests/bin/barracuda_test

test-go: $(LIBRARY)
	@echo "🧪 Running Go Tests..."  
	cd tests && go test -v

test-cuda: $(BENCHMARK)
	@echo "🚀 Running CUDA Performance Test..."
	./$(BENCHMARK)

test-all: test-cpp test-go test-cuda
	@echo "✅ All tests completed!"

# Build C++ tests (simple version without gtest)
tests/bin/barracuda_test: tests/cpp/simple_test.cpp $(LIBRARY)
	@mkdir -p tests/bin
	$(CXX) -O3 -std=c++14 $(INCLUDES) -L$(LIBDIR) -o $@ $< -lbarracuda $(LIBS)

# Quick benchmark
benchmark-quick: $(BENCHMARK)
	@echo "🚀 Running quick CUDA benchmark..."
	./$(BENCHMARK)

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(LIBDIR) bin/ tests/bin/ barracuda

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt update
	sudo apt install -y build-essential nvidia-cuda-toolkit

# Check CUDA installation
cuda-info:
	nvcc --version
	nvidia-smi

.PHONY: all directories go-build test benchmark clean install-deps cuda-info