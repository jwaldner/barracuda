# 🎯 Barracuda CUDA Options Analyzer

High-performance options analysis engine with CUDA acceleration for S&P 500 put options strategies.

## ✨ Features

### 🚀 **Performance**
- **CUDA-accelerated** options calculations (falls back to CPU)
- **Batch processing** with configurable workload factors
- **Real-time benchmarking** and performance comparison
- **Rate-limited API calls** (350ms intervals for Alpaca Basic)

### 📊 **Data Integration**
- **Live S&P 500 data** via Alpaca Markets API
- **Dynamic symbol ranking** (top 25 by market cap)
- **Real options chains** with current pricing
- **Black-Scholes calculations** with Greeks

### 🎨 **Modern Web UI**
- **Live status indicators** with color-coded states
- **Bright yellow CUDA highlighting** for easy identification
- **Real-time workload monitoring** (IDLE → ACTIVE → COMPLETED)
- **Performance statistics** with processing times
- **Responsive design** with large, readable text

### ⚙️ **Configuration**
- **Flexible execution modes**: auto/cuda/cpu
- **Workload factor tuning** for benchmarking (1.0x = normal, 2.0x = double work)
- **Batch size optimization** for different hardware
- **Paper trading support** with Alpaca sandbox

## 🛠️ Installation

### Prerequisites
- **Go 1.21+**
- **NVIDIA CUDA Toolkit** (optional, for GPU acceleration)
- **GCC/G++** compiler
- **Alpaca Markets API account**

### Build
```bash
# Clone the repository
git clone <your-repo-url>
cd go-cuda

# Copy and configure settings
cp config.yaml.template config.yaml
# Edit config.yaml with your Alpaca API keys

# Build the application
make build

# Run the server
make run
```

## 🎮 Usage

### Web Interface
1. Open http://localhost:8080
2. Review status indicators:
   - **⚡ Compute Mode**: CUDA/CPU with yellow highlighting
   - **🔥 Workload**: Current multiplier and processing state
   - **📊 S&P 500 Symbols**: Available symbols count
   - **💰 Analysis Cash**: Available capital

3. Click **"🔍 Run Put Options Analysis"**
4. Watch real-time status changes and results

### Performance Benchmarking

#### Compare CUDA vs CPU:
```yaml
# config.yaml - Test with CUDA
engine:
  execution_mode: "auto"  # or "cuda"
  workload_factor: 2.0    # Double the computational load

# Run analysis, note timing

# config.yaml - Test with CPU
engine:
  execution_mode: "cpu"
  workload_factor: 2.0    # Same workload

# Restart server, run analysis, compare times
```

#### Stress Testing:
```yaml
engine:
  workload_factor: 5.0    # 5x computational intensity
  batch_size: 2000        # Larger batches
```

## 📋 Configuration Options

### `config.yaml` Structure:
```yaml
alpaca:
  api_key: "YOUR_KEY"
  secret_key: "YOUR_SECRET"

trading:
  default_cash: 70000      # Available capital
  target_delta: 0.25       # Target option delta
  default_stocks: []       # Empty = use dynamic S&P 500

engine:
  execution_mode: "auto"   # auto/cuda/cpu
  workload_factor: 1.0     # Computational multiplier
  batch_size: 1000         # Contracts per batch
  enable_benchmarks: true  # Performance tracking
```

### Workload Factor Guide:
- **1.0**: Normal computational load
- **2.0**: Double the work (good for differentiating CUDA vs CPU)
- **0.5**: Light load (faster processing)
- **5.0**: Heavy stress test (max differentiation)

## 🎯 UI Status Indicators

### Compute Mode Card:
- **Bright yellow background**: CUDA/CPU status
- **Real-time device count**: Shows available CUDA devices

### Workload Indicator:
- **🔥 WORKLOAD: 1.0x [NORMAL] | IDLE** (green background)
- **🔥 WORKLOAD: ACTIVE - Processing X symbols** (red background)
- **🔥 WORKLOAD: COMPLETED (X.XXs)** (green background)
- **🔥 WORKLOAD: ERROR** (dark red background)

## 📊 Performance Analysis

The system provides detailed timing statistics:
- **Processing time** in seconds
- **Symbols analyzed** count
- **Results generated** count
- **CUDA status** (true/false)

Example output: `"✅ Completed in 11.19 seconds | 25 symbols | 25 results | CUDA: true"`

## 🔧 Build Commands

```bash
make build    # Compile the application
make run      # Build and start server
make clean    # Remove binaries
```

## 🚀 Deployment

### Production Checklist:
1. ✅ Configure real Alpaca API keys
2. ✅ Set appropriate workload factor (1.0 for production)
3. ✅ Test CUDA detection on target hardware
4. ✅ Verify network connectivity to Alpaca APIs
5. ✅ Configure proper rate limiting (350ms intervals)

### Hardware Recommendations:
- **CUDA-capable GPU**: NVIDIA GTX 1060 or better
- **RAM**: 8GB+ for large symbol processing
- **Network**: Stable connection for real-time data

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

---

**Built with ❤️ using Go, CUDA, and modern web technologies**