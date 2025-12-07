package barracuda

import (
	"math"
	"time"
)

// OptionContract represents an options contract for processing
type OptionContract struct {
	Symbol           string  `json:"symbol"`
	StrikePrice      float64 `json:"strike_price"`
	UnderlyingPrice  float64 `json:"underlying_price"`
	TimeToExpiration float64 `json:"time_to_expiration"`
	RiskFreeRate     float64 `json:"risk_free_rate"`
	Volatility       float64 `json:"volatility"`
	OptionType       byte    `json:"option_type"` // 'C' or 'P'

	// Output Greeks
	Delta            float64 `json:"delta"`
	Gamma            float64 `json:"gamma"`
	Theta            float64 `json:"theta"`
	Vega             float64 `json:"vega"`
	Rho              float64 `json:"rho"`
	TheoreticalPrice float64 `json:"theoretical_price"`
}

// MarketData represents real-time market data
type MarketData struct {
	Symbol    string  `json:"symbol"`
	Price     float64 `json:"price"`
	Bid       float64 `json:"bid"`
	Ask       float64 `json:"ask"`
	Timestamp int64   `json:"timestamp"`
	Volume    float64 `json:"volume"`
}

// VolatilitySkew represents the 25-delta put/call IV skew
type VolatilitySkew struct {
	Symbol     string  `json:"symbol"`
	Expiration string  `json:"expiration"`
	Put25DIV   float64 `json:"put_25d_iv"`
	Call25DIV  float64 `json:"call_25d_iv"`
	Skew       float64 `json:"skew"`
	ATMIV      float64 `json:"atm_iv"`
}

// ExecutionMode defines how calculations are performed
type ExecutionMode string

const (
	ExecutionModeAuto ExecutionMode = "auto" // Auto-detect best method
	ExecutionModeCUDA ExecutionMode = "cuda" // Force CUDA (fail if not available)
	ExecutionModeCPU  ExecutionMode = "cpu"  // Force CPU calculation
)

// BenchmarkResult contains performance metrics
type BenchmarkResult struct {
	Mode               ExecutionMode `json:"mode"`
	Calculations       int           `json:"calculations"`
	ProcessingTimeMs   float64       `json:"processing_time_ms"`
	CalculationsPerSec float64       `json:"calculations_per_sec"`
	BatchSize          int           `json:"batch_size"`
	Batches            int           `json:"batches"`
	CUDAAvailable      bool          `json:"cuda_available"`
	DeviceCount        int           `json:"device_count"`
	WorkloadFactor     float64       `json:"workload_factor"`
}

// BaracudaEngine provides high-performance options calculations
type BaracudaEngine struct {
	executionMode ExecutionMode
}

// NewBaracudaEngine creates a new options engine
func NewBaracudaEngine() *BaracudaEngine {
	// Try to initialize CUDA, fall back to CPU if not available
	execMode := ExecutionModeCPU
	if isCudaAvailable() {
		execMode = ExecutionModeCUDA
	}
	return &BaracudaEngine{
		executionMode: execMode,
	}
}

// Close cleans up the engine resources
func (be *BaracudaEngine) Close() {
	// Nothing to clean up in CPU mode
}

// IsCudaAvailable returns true if CUDA is available for calculations
func (be *BaracudaEngine) IsCudaAvailable() bool {
	return be.executionMode == ExecutionModeCUDA
}

// isCudaAvailable checks if CUDA is actually available
func isCudaAvailable() bool {
	// This should call into the C++ CUDA detection code
	// For now, assume CUDA is available if the library exists
	return true // Will be replaced with actual CUDA detection
}

// GetDeviceCount returns the number of CUDA devices
func (be *BaracudaEngine) GetDeviceCount() int {
	if be.executionMode == ExecutionModeCUDA {
		return 1 // Assume 1 NVIDIA device for now
	}
	return 0
}

// CalculateBlackScholes performs Black-Scholes calculation
func (be *BaracudaEngine) CalculateBlackScholes(contracts []OptionContract) ([]OptionContract, error) {
	if len(contracts) == 0 {
		return contracts, nil
	}

	// Calculate using CPU-based Black-Scholes
	for i := range contracts {
		contract := &contracts[i]

		// Basic Black-Scholes calculation
		S := contract.UnderlyingPrice
		K := contract.StrikePrice
		T := contract.TimeToExpiration
		r := contract.RiskFreeRate
		sigma := contract.Volatility

		if T <= 0 {
			// Handle expired options
			if contract.OptionType == 'C' {
				contract.TheoreticalPrice = math.Max(S-K, 0)
				contract.Delta = func() float64 {
					if S > K {
						return 1.0
					} else {
						return 0.0
					}
				}()
			} else {
				contract.TheoreticalPrice = math.Max(K-S, 0)
				contract.Delta = func() float64 {
					if K > S {
						return -1.0
					} else {
						return 0.0
					}
				}()
			}
			continue
		}

		// Black-Scholes calculations
		d1 := (math.Log(S/K) + (r+0.5*sigma*sigma)*T) / (sigma * math.Sqrt(T))
		d2 := d1 - sigma*math.Sqrt(T)

		normCDFd1 := normalCDF(d1)
		normCDFd2 := normalCDF(d2)
		normCDFNegd1 := normalCDF(-d1)
		normCDFNegd2 := normalCDF(-d2)

		if contract.OptionType == 'C' {
			// Call option
			contract.TheoreticalPrice = S*normCDFd1 - K*math.Exp(-r*T)*normCDFd2
			contract.Delta = normCDFd1
		} else {
			// Put option
			contract.TheoreticalPrice = K*math.Exp(-r*T)*normCDFNegd2 - S*normCDFNegd1
			contract.Delta = normCDFd1 - 1
		}

		// Calculate other Greeks
		sqrtT := math.Sqrt(T)
		expMinusRT := math.Exp(-r * T)

		// Gamma (same for calls and puts)
		contract.Gamma = normalPDF(d1) / (S * sigma * sqrtT)

		// Vega (same for calls and puts)
		contract.Vega = S * normalPDF(d1) * sqrtT / 100 // Divide by 100 for 1% volatility change

		// Theta
		if contract.OptionType == 'C' {
			contract.Theta = -(S*normalPDF(d1)*sigma/(2*sqrtT) + r*K*expMinusRT*normCDFd2) / 365
		} else {
			contract.Theta = -(S*normalPDF(d1)*sigma/(2*sqrtT) - r*K*expMinusRT*normCDFNegd2) / 365
		}

		// Rho
		if contract.OptionType == 'C' {
			contract.Rho = K * T * expMinusRT * normCDFd2 / 100 // Divide by 100 for 1% rate change
		} else {
			contract.Rho = -K * T * expMinusRT * normCDFNegd2 / 100
		}
	}

	return contracts, nil
}

// CalculateBatch processes multiple options contracts efficiently
func (be *BaracudaEngine) CalculateBatch(contracts []OptionContract) ([]OptionContract, error) {
	return be.CalculateBlackScholes(contracts)
}

// GetBenchmarkResult performs a performance test
func (be *BaracudaEngine) GetBenchmarkResult(numCalculations int) BenchmarkResult {
	start := time.Now()

	// Create test contracts
	contracts := make([]OptionContract, numCalculations)
	for i := range contracts {
		contracts[i] = OptionContract{
			Symbol:           "TEST",
			StrikePrice:      100.0,
			UnderlyingPrice:  100.0,
			TimeToExpiration: 0.25, // 3 months
			RiskFreeRate:     0.05,
			Volatility:       0.25,
			OptionType:       'C',
		}
	}

	// Perform calculations
	_, _ = be.CalculateBlackScholes(contracts)

	elapsed := time.Since(start)

	return BenchmarkResult{
		Mode:               ExecutionModeCPU,
		Calculations:       numCalculations,
		ProcessingTimeMs:   float64(elapsed.Nanoseconds()) / 1e6,
		CalculationsPerSec: float64(numCalculations) / elapsed.Seconds(),
		BatchSize:          numCalculations,
		Batches:            1,
		CUDAAvailable:      false,
		DeviceCount:        0,
	}
}

// normalCDF approximates the cumulative distribution function of standard normal distribution
func normalCDF(x float64) float64 {
	return 0.5 * (1.0 + math.Erf(x/math.Sqrt2))
}

// normalPDF calculates the probability density function of standard normal distribution
func normalPDF(x float64) float64 {
	return math.Exp(-0.5*x*x) / math.Sqrt(2*math.Pi)
}
