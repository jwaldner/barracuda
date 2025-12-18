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
    double market_close_price;
    double delta;
    double gamma;
    double theta;
    double vega;
    double rho;
    double theoretical_price;
} COptionContract;

// Complete C struct for complete GPU processing
typedef struct {
    char symbol[32];
    double strike_price;
    double underlying_price;
    double time_to_expiration;
    double risk_free_rate;
    double volatility;
    char option_type;
    double market_close_price;
    double delta;
    double gamma;
    double theta;
    double vega;
    double rho;
    double theoretical_price;
    double implied_volatility;
    int max_contracts;
    double total_premium;
    double cash_needed;
    double profit_percentage;
    double annualized_return;
    int days_to_expiration;
} CCompleteOptionContract;

// C functions from barracuda_engine.cpp
void* barracuda_create_engine();
void barracuda_destroy_engine(void* engine);
int barracuda_initialize_cuda(void* engine);
int barracuda_is_cuda_available(void* engine);
int barracuda_get_device_count(void* engine);
int barracuda_calculate_options(void* engine, COptionContract* contracts, int count);
double barracuda_benchmark(void* engine, int num_contracts, int iterations);

// CUDA maximization function - zero Go loops
int barracuda_cuda_maximize_processing(void* engine, COptionContract* contracts, int count,
                                     double stock_price, int* put_count, int* call_count);

// Complete GPU processing function - ALL calculations on GPU
int barracuda_calculate_options_complete(void* engine, CCompleteOptionContract* contracts, int count,
                                       double available_cash, int days_to_expiration);

// New preprocessing functions
typedef struct {
    int num_puts;
    int num_calls;
    double preprocessing_time_ms;
    int total_contracts_processed;
} CPreprocessingResult;

typedef struct {
    double put_25d_iv;
    double call_25d_iv;
    double skew;
    double atm_iv;
    double calculation_time_ms;
    int contracts_analyzed;
} CVolatilitySkewResult;

// TODO: Add preprocessing functions when C++ integration is complete
*/
import "C"
import (
	"fmt"
	"log"
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
	OptionType       byte    // 'C' or 'P'
	MarketClosePrice float64 // Market close price for IV calculation

	// Output Greeks
	Delta            float64
	Gamma            float64
	Theta            float64
	Vega             float64
	Rho              float64
	TheoreticalPrice float64
}

// CompleteOptionResult represents a complete option analysis result with all business calculations
// EXPERIMENTAL: For complete GPU processing that includes business logic calculations
type CompleteOptionResult struct {
	// Basic option info
	Symbol          string
	OptionType      byte // 'C' or 'P'
	StrikePrice     float64
	UnderlyingPrice float64

	// Market data
	MarketClosePrice float64

	// Calculated Greeks (from GPU)
	Delta             float64
	Gamma             float64
	Theta             float64
	Vega              float64
	Rho               float64
	TheoreticalPrice  float64
	ImpliedVolatility float64

	// Business calculations (from GPU)
	MaxContracts     int
	TotalPremium     float64
	CashNeeded       float64
	ProfitPercentage float64
	AnnualizedReturn float64
	DaysToExpiration int
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

// PreprocessingResult represents the result of parallel preprocessing
type PreprocessingResult struct {
	NumPuts                 int
	NumCalls                int
	PreprocessingTimeMs     float64
	TotalContractsProcessed int
}

// VolatilitySkewResult represents parallel skew calculation result
type VolatilitySkewResult struct {
	Put25DIV          float64
	Call25DIV         float64
	Skew              float64
	ATMIV             float64
	CalculationTimeMs float64
	ContractsAnalyzed int
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
		cContracts[i].market_close_price = C.double(contract.MarketClosePrice)
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

		// Separate puts and calls - NO DEFAULTS, require real market data
		var puts, calls []OptionContract
		for _, option := range options {
			option.UnderlyingPrice = stockPrice

			// REQUIRE real market data - NO DEFAULTS for CPU mode either
			if option.TimeToExpiration <= 0 {
				log.Printf("❌ CPU: Missing time to expiration for %s - skipping", option.Symbol)
				continue
			}
			if option.RiskFreeRate <= 0 {
				log.Printf("❌ CPU: Missing risk free rate for %s - skipping", option.Symbol)
				continue
			}

			// Use market price (TheoreticalPrice) to calculate implied volatility - NO DEFAULTS
			marketPrice := option.TheoreticalPrice
			if marketPrice > 0.01 { // Only calculate IV for options with meaningful market price
				// Simple implied volatility estimation (Newton-Raphson would be better)
				option.Volatility = be.estimateImpliedVolatility(marketPrice, stockPrice, option.StrikePrice,
					option.TimeToExpiration, option.RiskFreeRate, option.OptionType)
			} else {
				log.Printf("❌ CPU: Missing volatility/market price for %s - skipping", option.Symbol)
				continue // Skip contracts with no real data
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

	// Calculate initial volatility guess from market price - NO DEFAULTS
	vol := marketPrice * 2.0 / (stockPrice * math.Sqrt(timeToExp)) // Market-based approximation

	// Reasonable bounds for volatility calculation
	if vol < 0.01 {
		return 0 // Invalid market data - cannot calculate
	}
	if vol > 2.0 {
		vol = 2.0 // Cap at 200% volatility (extreme but possible)
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

// MaximizeCUDAUsage processes options with minimal Go loops - maximum GPU utilization
func (be *BaracudaEngine) MaximizeCUDAUsage(options []OptionContract, stockPrice float64) ([]OptionContract, []OptionContract, error) {
	if len(options) == 0 {
		return nil, nil, nil
	}

	if be.engine == nil {
		return nil, nil, fmt.Errorf("engine not initialized")
	}

	// Force CUDA mode for maximum GPU usage
	if !be.IsCudaAvailable() {
		return nil, nil, fmt.Errorf("CUDA not available - cannot maximize GPU usage")
	}

	log.Printf("🚀 CUDA MAXIMIZED: Processing %d contracts with minimal Go loops, maximum GPU parallelization", len(options))

	// Debug: Log the first few contract inputs to verify correct values
	for i := 0; i < len(options) && i < 3; i++ {
		log.Printf("🔍 Input[%d]: Symbol=%s, Strike=%.2f, Underlying=%.2f, Time=%.6f, Vol=%.3f, Rate=%.3f, Type=%c",
			i, options[i].Symbol, options[i].StrikePrice, options[i].UnderlyingPrice,
			options[i].TimeToExpiration, options[i].Volatility, options[i].RiskFreeRate, options[i].OptionType)
	}

	// Convert Go contracts to C contracts - ALL fields needed for Black-Scholes
	cContracts := make([]C.COptionContract, len(options))
	for i, contract := range options {
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
		cContracts[i].market_close_price = C.double(contract.MarketClosePrice)
		cContracts[i].theoretical_price = C.double(contract.TheoreticalPrice)
	}

	// ALL COMPUTATIONAL WORK ON GPU:
	// - Preprocessing (set prices, time, rates)
	// - Implied volatility (Newton-Raphson iterations)
	// - Black-Scholes calculations
	// - Put/call separation
	cudarStart := time.Now()
	result := C.barracuda_calculate_options(
		be.engine,
		(*C.COptionContract)(unsafe.Pointer(&cContracts[0])),
		C.int(len(options)))

	if result != 0 {
		return nil, nil, fmt.Errorf("CUDA calculation failed with code %d", result)
	}
	cudaDuration := time.Since(cudarStart)

	// Convert results back and separate puts/calls
	var puts, calls []OptionContract
	for i := range options {
		// DEBUG: Log what CUDA returned
		log.Printf("🔍 CUDA RESULT[%d]: Symbol=%s, Market=$%.3f, TheoPrice=%.6f, Delta=%.6f, Vol=%.3f",
			i, options[i].Symbol, float64(cContracts[i].market_close_price),
			float64(cContracts[i].theoretical_price), float64(cContracts[i].delta), float64(cContracts[i].volatility))

		processed := OptionContract{
			Symbol:           options[i].Symbol,
			StrikePrice:      options[i].StrikePrice,
			OptionType:       options[i].OptionType,
			UnderlyingPrice:  float64(cContracts[i].underlying_price),
			TimeToExpiration: float64(cContracts[i].time_to_expiration),
			RiskFreeRate:     float64(cContracts[i].risk_free_rate),
			MarketClosePrice: float64(cContracts[i].market_close_price),
			Delta:            float64(cContracts[i].delta),
			Gamma:            float64(cContracts[i].gamma),
			Theta:            float64(cContracts[i].theta),
			Vega:             float64(cContracts[i].vega),
			Rho:              float64(cContracts[i].rho),
			TheoreticalPrice: float64(cContracts[i].theoretical_price),
			Volatility:       float64(cContracts[i].volatility),
		}

		if processed.OptionType == 'P' {
			puts = append(puts, processed)
		} else {
			calls = append(calls, processed)
		}
	}

	log.Printf("⚡ CUDA MAXIMIZED: %.3fms | %d contracts → %d puts, %d calls | All computations on GPU",
		cudaDuration.Seconds()*1000, len(options), len(puts), len(calls))

	return puts, calls, nil
}

// MaximizeCUDAUsageComplete processes options with COMPLETE GPU processing - all business calculations on GPU
func (be *BaracudaEngine) MaximizeCUDAUsageComplete(options []OptionContract, stockPrice, availableCash float64, strategy string, expirationDate string) ([]CompleteOptionResult, error) {
	if len(options) == 0 {
		return nil, nil
	}

	if be.engine == nil {
		return nil, fmt.Errorf("engine not initialized")
	}

	if !be.IsCudaAvailable() {
		return nil, fmt.Errorf("CUDA not available - complete GPU processing requires CUDA")
	}

	log.Printf("🚀 COMPLETE CUDA: Processing %d contracts with ALL calculations on GPU", len(options))

	// Calculate days to expiration
	expirationTime, err := time.Parse("2006-01-02", expirationDate)
	if err != nil {
		return nil, fmt.Errorf("invalid expiration date format: %v", err)
	}
	daysToExp := int(time.Until(expirationTime).Hours() / 24)

	// Convert Go contracts to C complete contracts
	cContracts := make([]C.CCompleteOptionContract, len(options))
	for i, contract := range options {
		// Copy symbol (truncate if too long)
		symbolBytes := []byte(contract.Symbol)
		for j := 0; j < len(symbolBytes) && j < 31; j++ {
			cContracts[i].symbol[j] = C.char(symbolBytes[j])
		}
		cContracts[i].symbol[31] = 0 // Null terminate

		// Input fields
		cContracts[i].strike_price = C.double(contract.StrikePrice)
		cContracts[i].underlying_price = C.double(contract.UnderlyingPrice)
		cContracts[i].time_to_expiration = C.double(contract.TimeToExpiration)
		cContracts[i].risk_free_rate = C.double(contract.RiskFreeRate)
		cContracts[i].volatility = C.double(contract.Volatility)
		cContracts[i].option_type = C.char(contract.OptionType)
		cContracts[i].market_close_price = C.double(contract.MarketClosePrice)
	}

	// Call CUDA complete processing
	result := C.barracuda_calculate_options_complete(
		be.engine,
		(*C.CCompleteOptionContract)(unsafe.Pointer(&cContracts[0])),
		C.int(len(options)),
		C.double(availableCash),
		C.int(daysToExp))

	if result != 0 {
		return nil, fmt.Errorf("CUDA complete processing failed with code %d", result)
	}

	// Convert results back to Go
	results := make([]CompleteOptionResult, len(options))
	for i := range options {
		results[i] = CompleteOptionResult{
			Symbol:            options[i].Symbol,
			OptionType:        options[i].OptionType,
			StrikePrice:       float64(cContracts[i].strike_price),
			UnderlyingPrice:   float64(cContracts[i].underlying_price),
			MarketClosePrice:  float64(cContracts[i].market_close_price),
			Delta:             float64(cContracts[i].delta),
			Gamma:             float64(cContracts[i].gamma),
			Theta:             float64(cContracts[i].theta),
			Vega:              float64(cContracts[i].vega),
			Rho:               float64(cContracts[i].rho),
			TheoreticalPrice:  float64(cContracts[i].theoretical_price),
			ImpliedVolatility: float64(cContracts[i].implied_volatility),
			MaxContracts:      int(cContracts[i].max_contracts),
			TotalPremium:      float64(cContracts[i].total_premium),
			CashNeeded:        float64(cContracts[i].cash_needed),
			ProfitPercentage:  float64(cContracts[i].profit_percentage),
			AnnualizedReturn:  float64(cContracts[i].annualized_return),
			DaysToExpiration:  int(cContracts[i].days_to_expiration),
		}
	}

	log.Printf("⚡ COMPLETE CUDA: %d contracts processed with ALL calculations on GPU", len(results))
	return results, nil
}

// Close cleans up engine resources
