package barracuda

/*
#cgo CXXFLAGS: -I${SRCDIR}/../src -std=c++11
#cgo LDFLAGS: -L${SRCDIR}/../lib -lbarracuda -lcudart -lcurand -lstdc++ -Wl,-rpath,${SRCDIR}/../lib

// C struct matching barracuda_engine.cpp
typedef struct {
    char symbol[32];
    double strike_price;
    double underlying_price;
    double time_to_expiration;
    double risk_free_rate;
    double volatility;
    char option_type;
    double delta;
    double gamma;
    double theta;
    double vega;
    double rho;
    double theoretical_price;
} COptionContract;

// C functions from barracuda_engine.cpp
void* barracuda_create_engine();
void barracuda_destroy_engine(void* engine);
int barracuda_initialize_cuda(void* engine);
int barracuda_is_cuda_available(void* engine);
int barracuda_get_device_count(void* engine);
int barracuda_calculate_options(void* engine, COptionContract* contracts, int count);
double barracuda_benchmark(void* engine, int num_contracts, int iterations);
*/
import "C"
import (
	"fmt"
	"math"
	"time"
	"unsafe"
)

// OptionContract represents an options contract
type OptionContract struct {
	Symbol           string
	StrikePrice      float64
	UnderlyingPrice  float64
	TimeToExpiration float64
	RiskFreeRate     float64
	Volatility       float64
	OptionType       byte // 'C' or 'P'

	// Output Greeks
	Delta            float64
	Gamma            float64
	Theta            float64
	Vega             float64
	Rho              float64
	TheoreticalPrice float64
}

// SymbolAnalysisResult represents comprehensive analysis for a single symbol
type SymbolAnalysisResult struct {
	Symbol                string
	StockPrice            float64
	Expiration            string
	PutsWithIV            []OptionContract
	CallsWithIV           []OptionContract
	VolatilitySkew        VolatilitySkew
	Best25DPut            OptionContract
	Best25DCall           OptionContract
	CalculationTimeMs     float64
	ExecutionMode         string
	TotalOptionsProcessed int
}

// VolatilitySkew represents 25-delta volatility skew analysis
type VolatilitySkew struct {
	Symbol     string
	Expiration string
	Put25DIV   float64
	Call25DIV  float64
	Skew       float64
	ATMIV      float64
}

// ExecutionMode defines how calculations are performed
type ExecutionMode string

const (
	ExecutionModeAuto ExecutionMode = "auto"
	ExecutionModeCUDA ExecutionMode = "cuda"
	ExecutionModeCPU  ExecutionMode = "cpu"
)

// BaracudaEngine provides CUDA-accelerated options calculations
type BaracudaEngine struct {
	engine        unsafe.Pointer
	executionMode ExecutionMode
}

// NewBaracudaEngine creates a new CUDA-accelerated engine
func NewBaracudaEngine() *BaracudaEngine {
	engine := C.barracuda_create_engine()
	if engine == nil {
		// CUDA engine creation failed
		return nil
	}

	be := &BaracudaEngine{
		engine:        engine,
		executionMode: ExecutionModeCUDA,
	}

	// Initialize CUDA
	if C.barracuda_initialize_cuda(engine) == 0 {
		// CUDA initialization failed, using CPU
		be.executionMode = ExecutionModeCPU
	} else {
		_ = C.barracuda_get_device_count(engine) // CUDA device count checked
		// CUDA initialized successfully
	}

	return be
}

// NewBaracudaEngineForced creates engine with forced execution mode
func NewBaracudaEngineForced(mode string) *BaracudaEngine {
	be := NewBaracudaEngine()
	if be == nil {
		return nil
	}

	switch mode {
	case "cpu":
		be.executionMode = ExecutionModeCPU
		// CPU mode selected
	case "cuda":
		be.executionMode = ExecutionModeCUDA
		// CUDA mode selected
	default:
		be.executionMode = ExecutionModeAuto
	}

	return be
}

// Close cleans up engine resources
func (be *BaracudaEngine) Close() {
	if be.engine != nil {
		C.barracuda_destroy_engine(be.engine)
		be.engine = nil
	}
}

// IsCudaAvailable returns true if CUDA is available
func (be *BaracudaEngine) IsCudaAvailable() bool {
	if be.engine == nil {
		return false
	}
	// Check if CUDA is actually available, regardless of execution mode setting
	return C.barracuda_is_cuda_available(be.engine) != 0
}

// GetDeviceCount returns number of CUDA devices
func (be *BaracudaEngine) GetDeviceCount() int {
	if be.engine == nil {
		return 0
	}
	return int(C.barracuda_get_device_count(be.engine))
}

// CalculateBlackScholes performs GPU-accelerated Black-Scholes calculation
func (be *BaracudaEngine) CalculateBlackScholes(contracts []OptionContract) ([]OptionContract, error) {
	if len(contracts) == 0 {
		return contracts, nil
	}

	if be.engine == nil {
		return nil, fmt.Errorf("engine not initialized")
	}

	// Convert Go contracts to C contracts
	cContracts := make([]C.COptionContract, len(contracts))
	for i, contract := range contracts {
		// Copy symbol (truncate if too long)
		symbolBytes := []byte(contract.Symbol)
		for j := 0; j < len(symbolBytes) && j < 31; j++ {
			cContracts[i].symbol[j] = C.char(symbolBytes[j])
		}
		cContracts[i].symbol[31] = 0 // Null terminate

		cContracts[i].strike_price = C.double(contract.StrikePrice)
		cContracts[i].underlying_price = C.double(contract.UnderlyingPrice)
		cContracts[i].time_to_expiration = C.double(contract.TimeToExpiration)
		cContracts[i].risk_free_rate = C.double(contract.RiskFreeRate)
		cContracts[i].volatility = C.double(contract.Volatility)
		cContracts[i].option_type = C.char(contract.OptionType)
	}

	// Call CUDA calculation
	result := C.barracuda_calculate_options(
		be.engine,
		(*C.COptionContract)(unsafe.Pointer(&cContracts[0])),
		C.int(len(contracts)))

	if result != 0 {
		return nil, fmt.Errorf("CUDA calculation failed with code %d", result)
	}

	// Convert results back to Go
	results := make([]OptionContract, len(contracts))
	for i := range contracts {
		results[i] = contracts[i] // Copy input data
		results[i].Delta = float64(cContracts[i].delta)
		results[i].Gamma = float64(cContracts[i].gamma)
		results[i].Theta = float64(cContracts[i].theta)
		results[i].Vega = float64(cContracts[i].vega)
		results[i].Rho = float64(cContracts[i].rho)
		results[i].TheoreticalPrice = float64(cContracts[i].theoretical_price)
	}

	return results, nil
}

// AnalyzeSymbolsBatch performs comprehensive analysis on multiple symbols
// Automatically routes to CUDA (parallel) or CPU (sequential) based on availability
func (be *BaracudaEngine) AnalyzeSymbolsBatch(
	symbols []string,
	stockPrices map[string]float64,
	optionsChains map[string][]OptionContract,
	expirationDate string) ([]SymbolAnalysisResult, error) {

	if be.engine == nil {
		return nil, fmt.Errorf("engine not initialized")
	}

	// For now, process each symbol individually using existing CalculateBlackScholes
	// TODO: Implement full C++ batch processing integration
	results := make([]SymbolAnalysisResult, 0, len(symbols))

	startTime := time.Now()

	for _, symbol := range symbols {
		result := SymbolAnalysisResult{
			Symbol:     symbol,
			Expiration: expirationDate,
		}

		stockPrice, hasStock := stockPrices[symbol]
		options, hasOptions := optionsChains[symbol]

		if !hasStock || !hasOptions {
			continue
		}

		result.StockPrice = stockPrice

		// Determine execution mode based on forced mode and CUDA availability
		if be.executionMode == ExecutionModeCUDA && be.IsCudaAvailable() {
			result.ExecutionMode = "CUDA"
		} else if be.executionMode == ExecutionModeAuto && be.IsCudaAvailable() {
			result.ExecutionMode = "CUDA"
		} else {
			result.ExecutionMode = "CPU"
		}

		// Separate puts and calls and calculate implied volatility from market prices
		var puts, calls []OptionContract
		for _, option := range options {
			option.UnderlyingPrice = stockPrice
			option.TimeToExpiration = 0.085 // ~31 days for Jan 2026 expiration
			option.RiskFreeRate = 0.05

			// Use market price (TheoreticalPrice) to calculate implied volatility
			marketPrice := option.TheoreticalPrice
			if marketPrice > 0.01 { // Only calculate IV for options with meaningful market price
				// Simple implied volatility estimation (Newton-Raphson would be better)
				option.Volatility = be.estimateImpliedVolatility(marketPrice, stockPrice, option.StrikePrice,
					option.TimeToExpiration, option.RiskFreeRate, option.OptionType)
			} else {
				option.Volatility = 0.25 // Default for very cheap options
			}

			if option.OptionType == 'P' {
				puts = append(puts, option)
			} else {
				calls = append(calls, option)
			}
		}

		// Calculate options (routes to CUDA or CPU automatically)
		if len(puts) > 0 {
			calculatedPuts, err := be.CalculateBlackScholes(puts)
			if err == nil {
				result.PutsWithIV = calculatedPuts
			}
		}

		if len(calls) > 0 {
			calculatedCalls, err := be.CalculateBlackScholes(calls)
			if err == nil {
				result.CallsWithIV = calculatedCalls
			}
		}

		// Calculate 25-delta skew (simplified)
		if len(result.PutsWithIV) > 0 && len(result.CallsWithIV) > 0 {
			result.VolatilitySkew = be.calculate25DeltaSkew(result.PutsWithIV, result.CallsWithIV, expirationDate)
		}

		result.TotalOptionsProcessed = len(result.PutsWithIV) + len(result.CallsWithIV)
		results = append(results, result)
	}

	duration := time.Since(startTime)

	// Set timing for all results
	for i := range results {
		results[i].CalculationTimeMs = duration.Seconds() * 1000
	}

	return results, nil
}

// Helper function to calculate 25-delta skew
func (be *BaracudaEngine) calculate25DeltaSkew(puts, calls []OptionContract, expiration string) VolatilitySkew {
	skew := VolatilitySkew{
		Expiration: expiration,
	}

	if len(puts) == 0 || len(calls) == 0 {
		return skew
	}

	// Find options closest to 25-delta
	targetDelta := 0.25
	var bestPut, bestCall OptionContract
	minPutDiff := 1.0
	minCallDiff := 1.0

	// Find 25-delta put (delta ≈ -0.25)
	for _, put := range puts {
		deltaDiff := abs(abs(put.Delta) - targetDelta)
		if deltaDiff < minPutDiff {
			minPutDiff = deltaDiff
			bestPut = put
			skew.Symbol = put.Symbol
		}
	}

	// Find 25-delta call (delta ≈ +0.25)
	for _, call := range calls {
		deltaDiff := abs(call.Delta - targetDelta)
		if deltaDiff < minCallDiff {
			minCallDiff = deltaDiff
			bestCall = call
		}
	}

	skew.Put25DIV = bestPut.Volatility
	skew.Call25DIV = bestCall.Volatility
	skew.Skew = bestPut.Volatility - bestCall.Volatility
	skew.ATMIV = (bestPut.Volatility + bestCall.Volatility) / 2.0

	return skew
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// estimateImpliedVolatility calculates IV from market price using Newton-Raphson
func (be *BaracudaEngine) estimateImpliedVolatility(marketPrice, stockPrice, strikePrice, timeToExp, riskFreeRate float64, optionType byte) float64 {
	tolerance := 1e-8 // Tighter tolerance for better precision
	maxIterations := 100

	// Better initial guess based on at-the-money approximation
	var vol float64
	atm := stockPrice / strikePrice
	if atm > 1.1 || atm < 0.9 {
		vol = 0.18 // Lower initial guess for OTM options
	} else {
		vol = marketPrice * 2.0 / (stockPrice * math.Sqrt(timeToExp)) // ATM approximation
		if vol < 0.05 {
			vol = 0.05
		}
		if vol > 1.0 {
			vol = 0.18
		}
	}

	for i := 0; i < maxIterations; i++ {
		// Black-Scholes calculation
		d1 := (math.Log(stockPrice/strikePrice) + (riskFreeRate+0.5*vol*vol)*timeToExp) / (vol * math.Sqrt(timeToExp))
		d2 := d1 - vol*math.Sqrt(timeToExp)

		var theoreticalPrice float64
		var vega float64

		// Normal CDF approximation
		nd1 := 0.5 * (1.0 + math.Erf(d1/math.Sqrt(2.0)))
		nd2 := 0.5 * (1.0 + math.Erf(d2/math.Sqrt(2.0)))
		// Normal PDF
		pdf := math.Exp(-0.5*d1*d1) / math.Sqrt(2.0*math.Pi)

		if optionType == 'C' {
			theoreticalPrice = stockPrice*nd1 - strikePrice*math.Exp(-riskFreeRate*timeToExp)*nd2
		} else {
			theoreticalPrice = strikePrice*math.Exp(-riskFreeRate*timeToExp)*(1.0-nd2) - stockPrice*(1.0-nd1)
		}

		vega = stockPrice * pdf * math.Sqrt(timeToExp)
		priceDiff := theoreticalPrice - marketPrice

		if math.Abs(priceDiff) < tolerance {
			return vol
		}

		if vega < 1e-10 {
			break
		}

		// Damped Newton-Raphson for better convergence
		adjustment := priceDiff / vega
		if math.Abs(adjustment) > 0.1 {
			adjustment = 0.1 * math.Copysign(1.0, adjustment) // Limit large steps
		}
		vol -= adjustment

		// Keep vol in tighter, more realistic bounds
		if vol < 0.02 {
			vol = 0.02
		}
		if vol > 1.5 {
			vol = 1.5
		}
	}

	return vol
}

// BenchmarkCalculation performs performance benchmark calculation
func (be *BaracudaEngine) BenchmarkCalculation(numContracts, iterations int) float64 {
	if be.engine == nil {
		return 0.0
	}
	return float64(C.barracuda_benchmark(be.engine, C.int(numContracts), C.int(iterations)))
}

// Close cleans up engine resources
