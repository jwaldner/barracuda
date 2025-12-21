// Package barracuda provides CUDA-accelerated Black-Scholes option calculations
//
// 🚀 PRODUCTION FUNCTIONS (ONLY these are accessible from external Go code):
//   - NewBaracudaEngine()          - Create engine instance
//   - MaximizeCUDAUsageComplete()  - Complete GPU processing with business logic
//   - MaximizeCPUUsageComplete()   - Complete CPU processing with business logic  
//   - MaximizeCUDAUsage()          - Basic GPU processing for simple analysis
//   - Close()                      - Clean up engine resources
//   - IsCudaAvailable()            - Check CUDA availability
//   - GetDeviceCount()             - Get CUDA device count
//   - SetExecutionMode()           - Set execution mode (auto/cuda/cpu)
//   - BenchmarkCalculation()       - Performance benchmarking
//
// 🔒 PRIVATE FUNCTIONS (Not accessible from external Go code):
//   - calculateBlackScholes()      - Internal calculation function
//   - calculate25DeltaSkew()       - Internal volatility analysis  
//   - logAuditEntry()              - Internal audit logging
//   - logCompleteAuditEntry()      - Internal complete audit logging
//   - estimateImpliedVolatility()  - Internal IV calculation
//
// AI/Coder Note: External code can ONLY call the public batch functions!
// All internal calculation functions are private and inaccessible.
package barracuda

/*
#cgo CXXFLAGS: -I${SRCDIR}/../src -std=c++11
#cgo LDFLAGS: -L${SRCDIR}/../lib -lbarracuda -lcudart -lcurand -lstdc++ -Wl,-rpath,${SRCDIR}/../lib

#include <stdlib.h>

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
int barracuda_calculate_options_with_audit(void* engine, COptionContract* contracts, int count, const char* audit_symbol);
void barracuda_set_execution_mode(void* engine, const char* mode);
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

// BarracudaEngine preprocessing functions will be added when C++ integration is complete
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"
	"unsafe"

	"github.com/jwaldner/barracuda/internal/logger"
)

// safeLog safely logs a message, handling cases where logger might not be initialized (like in tests)
func safeLog(format string, args ...interface{}) {
	if logger.Info != nil {
		logger.Info.Printf(format, args...)
	}
}

// safeWarn safely logs a warning message, handling cases where logger might not be initialized
func safeWarn(format string, args ...interface{}) {
	if logger.Warn != nil {
		logger.Warn.Printf(format, args...)
	}
}

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

// calculateBlackScholes performs GPU-accelerated Black-Scholes calculation (PRIVATE - use batch functions)
// ⚠️  WARNING: FOR INTERNAL/TESTING USE ONLY - NOT FOR PRODUCTION
// ⚠️  Use MaximizeCUDAUsageComplete() or MaximizeCPUUsageComplete() batch functions instead
// ⚠️  This function lacks proper business logic and optimization
// ⚠️  PRIVATE FUNCTION - NOT ACCESSIBLE FROM EXTERNAL Go CODE
func (be *BaracudaEngine) calculateBlackScholes(contracts []OptionContract, auditSymbol *string) ([]OptionContract, error) {
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

	// Call calculation with audit symbol (C++ engine will choose CUDA or CPU path automatically)
	// ALWAYS pass audit symbol - either nil or actual symbol
	var auditPtr *C.char
	if auditSymbol != nil {
		// Convert Go string to C string
		auditBytes := []byte(*auditSymbol + "\000") // null terminate
		auditPtr = (*C.char)(unsafe.Pointer(&auditBytes[0]))
	}
	// auditPtr will be nil if auditSymbol is nil

	result := C.barracuda_calculate_options_with_audit(
		be.engine,
		(*C.COptionContract)(unsafe.Pointer(&cContracts[0])),
		C.int(len(contracts)),
		auditPtr)

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

	// AUDIT LOGGING: If audit symbol provided, log detailed Black-Scholes calculation
	if auditSymbol != nil {
			err := be.logAuditEntry(*auditSymbol, results, 0.0)
		if err != nil {
			safeWarn("⚠️ Failed to log audit entry: %v", err)
		} else {
			safeLog("📋 AUDIT: Logged BlackScholesCalculation for %s (%d contracts)", *auditSymbol, len(results))
		}
	}

	return results, nil
}

// logAuditEntry creates detailed audit log for Black-Scholes calculations
// logAuditEntry logs detailed Black-Scholes audit information (PRIVATE)
func (be *BaracudaEngine) logAuditEntry(symbol string, results []OptionContract, computeTimeMs float64) error {
	if len(results) == 0 {
		return nil
	}

	// Create audits directory if it doesn't exist
	auditDir := "audits"
	if err := os.MkdirAll(auditDir, 0755); err != nil {
		return fmt.Errorf("failed to create audit directory: %v", err)
	}

	// Create audit entry with detailed Black-Scholes calculation data
	auditEntry := map[string]interface{}{
		"ticker":    symbol,
		"timestamp": time.Now().Format(time.RFC3339),
		"type":      "BlackScholesCalculation",
		"success":   true,
		"calculation_details": map[string]interface{}{
			"formula_documentation": map[string]interface{}{
				"put_formula":     "P = K * exp(-r*T) * N(-d2) - S * N(-d1)",
				"call_formula":    "C = S * N(d1) - K * exp(-r*T) * N(d2)",
				"d1_formula":      "d1 = [ln(S/K) + (r + σ²/2)*T] / (σ * √T)",
				"d2_formula":      "d2 = d1 - σ * √T",
				"greeks": map[string]interface{}{
					"delta_put":  "-N(-d1)",
					"delta_call": "N(d1)",
					"gamma":      "N'(d1) / (S * σ * √T)",
					"theta_put":  "[-S * N'(d1) * σ / (2√T) + r * K * exp(-r*T) * N(-d2)] / 365",
					"theta_call": "[-S * N'(d1) * σ / (2√T) - r * K * exp(-r*T) * N(d2)] / 365",
					"vega":       "S * √T * N'(d1) / 100",
					"rho_put":    "-K * T * exp(-r*T) * N(-d2) / 100",
					"rho_call":   "K * T * exp(-r*T) * N(d2) / 100",
				},
			},
			"execution_type":      be.executionMode,
			"contracts_processed": len(results),
			"symbol":              symbol,
			"compute_time_ms":     computeTimeMs,
			"inputs": []map[string]interface{}{
				{
					"contract_symbol": results[0].Symbol,
					"option_type":     string(results[0].OptionType),
					"S":               results[0].UnderlyingPrice,  // Current stock price
					"K":               results[0].StrikePrice,      // Strike price
					"T":               results[0].TimeToExpiration, // Time to expiration in years
					"r":               results[0].RiskFreeRate,     // Risk-free interest rate (annual)
					"sigma":           results[0].Volatility,       // Implied volatility (annual)
					"dividend_yield":  0.0,                        // Dividend yield (assume 0 for now)
				},
			},
			"results": []map[string]interface{}{
				{
					"contract_symbol":   results[0].Symbol,
					"d1":                "needs_calculation", // TODO: Extract from engine
					"d2":                "needs_calculation", // TODO: Extract from engine
					"theoretical_price": results[0].TheoreticalPrice,
					"delta":             results[0].Delta,
					"gamma":             results[0].Gamma,
					"theta":             results[0].Theta,
					"vega":              results[0].Vega,
					"rho":               results[0].Rho,
				},
			},
			"validation": map[string]interface{}{
				"ticker_match":            results[0].Symbol == symbol,
				"sample_contract_symbol": results[0].Symbol,
				"main_ticker":             symbol,
			},
		},
	}

	// Write to timestamped audit file
	timestamp := time.Now().Format("2006-01-02_15-04-05")
	filename := filepath.Join(auditDir, fmt.Sprintf("audit_%s_%s.json", symbol, timestamp))

	data, err := json.MarshalIndent(auditEntry, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal audit data: %v", err)
	}

	err = os.WriteFile(filename, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write audit file: %v", err)
	}

	safeLog("📋 AUDIT: Created audit file: %s", filename)
	return nil
}

// logCompleteAuditEntry creates detailed audit log for complete CUDA/CPU processing
// logCompleteAuditEntry logs complete option analysis audit information (PRIVATE)
func (be *BaracudaEngine) logCompleteAuditEntry(symbol string, results []CompleteOptionResult, executionMode string, computeTimeMs float64) error {
	if len(results) == 0 {
		return nil
	}

	// Create audits directory if it doesn't exist
	auditDir := "audits"
	if err := os.MkdirAll(auditDir, 0755); err != nil {
		return fmt.Errorf("failed to create audit directory: %v", err)
	}

	// Create audit entry with detailed Black-Scholes calculation data
	auditEntry := map[string]interface{}{
		"ticker":    symbol,
		"timestamp": time.Now().Format(time.RFC3339),
		"type":      "BlackScholesCalculation",
		"success":   true,
		"calculation_details": map[string]interface{}{
			"formula_documentation": map[string]interface{}{
				"put_formula":     "P = K * exp(-r*T) * N(-d2) - S * N(-d1)",
				"call_formula":    "C = S * N(d1) - K * exp(-r*T) * N(d2)",
				"d1_formula":      "d1 = [ln(S/K) + (r + σ²/2)*T] / (σ * √T)",
				"d2_formula":      "d2 = d1 - σ * √T",
				"greeks": map[string]interface{}{
					"delta_put":  "-N(-d1)",
					"delta_call": "N(d1)",
					"gamma":      "N'(d1) / (S * σ * √T)",
					"theta_put":  "[-S * N'(d1) * σ / (2√T) + r * K * exp(-r*T) * N(-d2)] / 365",
					"theta_call": "[-S * N'(d1) * σ / (2√T) - r * K * exp(-r*T) * N(d2)] / 365",
					"vega":       "S * √T * N'(d1) / 100",
					"rho_put":    "-K * T * exp(-r*T) * N(-d2) / 100",
					"rho_call":   "K * T * exp(-r*T) * N(d2) / 100",
				},
			},
			"execution_type":      executionMode,
			"contracts_processed": len(results),
			"symbol":              symbol,
			"compute_time_ms":     computeTimeMs,
			"inputs": []map[string]interface{}{
				{
					"contract_symbol": results[0].Symbol,
					"option_type":     string(results[0].OptionType),
					"S":               results[0].UnderlyingPrice,                   // Current stock price
					"K":               results[0].StrikePrice,                       // Strike price
					"T":               float64(results[0].DaysToExpiration) / 365.0, // Time to expiration in years
					"r":               0.05,                                         // Risk-free interest rate (annual)
					"sigma":           results[0].ImpliedVolatility,                 // Implied volatility (annual)
					"dividend_yield":  0.0,                                         // Dividend yield (assume 0 for now)
				},
			},
			"results": []map[string]interface{}{
				{
					"contract_symbol":     results[0].Symbol,
					"d1":                  "needs_calculation", // TODO: Extract from engine
					"d2":                  "needs_calculation", // TODO: Extract from engine
					"theoretical_price":   results[0].TheoreticalPrice,
					"delta":               results[0].Delta,
					"gamma":               results[0].Gamma,
					"theta":               results[0].Theta,
					"vega":                results[0].Vega,
					"rho":                 results[0].Rho,
					"implied_volatility":  results[0].ImpliedVolatility,
				},
			},
			"business_calculations": []map[string]interface{}{
				{
					"contract_symbol":   results[0].Symbol,
					"max_contracts":     results[0].MaxContracts,
					"total_premium":     results[0].TotalPremium,
					"cash_needed":       results[0].CashNeeded,
					"profit_percentage": results[0].ProfitPercentage,
					"annualized_return": results[0].AnnualizedReturn,
				},
			},
			"validation": map[string]interface{}{
				"ticker_match":            results[0].Symbol == symbol,
				"sample_contract_symbol": results[0].Symbol,
				"main_ticker":             symbol,
			},
		},
	}

	// Write to timestamped audit file
	timestamp := time.Now().Format("2006-01-02_15-04-05")
	filename := filepath.Join(auditDir, fmt.Sprintf("audit_%s_%s.json", symbol, timestamp))

	data, err := json.MarshalIndent(auditEntry, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal audit data: %v", err)
	}

	err = os.WriteFile(filename, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write audit file: %v", err)
	}

	safeLog("📋 AUDIT: Created complete audit file: %s", filename)
	return nil
}

// Legacy AnalyzeSymbolsBatch function removed - replaced by complete GPU processing

// Helper function to calculate 25-delta skew
// ⚠️  WARNING: FOR INTERNAL/TESTING USE ONLY - NOT FOR PRODUCTION
// ⚠️  Use proper batch volatility analysis functions instead of this simplified calculation
// ⚠️  PRIVATE FUNCTION - NOT ACCESSIBLE FROM EXTERNAL Go CODE
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
// estimateImpliedVolatility calculates IV using Newton-Raphson (PRIVATE)
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

	safeLog("🚀 CUDA MAXIMIZED: Processing %d contracts with minimal Go loops, maximum GPU parallelization", len(options))

	// Debug: Log the first few contract inputs to verify correct values
	for i := 0; i < len(options) && i < 3; i++ {
		safeLog("🔍 Input[%d]: Symbol=%s, Strike=%.2f, Underlying=%.2f, Time=%.6f, Vol=%.3f, Rate=%.3f, Type=%c",
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
		safeLog("🔍 CUDA RESULT[%d]: Symbol=%s, Market=$%.3f, TheoPrice=%.6f, Delta=%.6f, Vol=%.3f",
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

	safeLog("⚡ CUDA MAXIMIZED: %.3fms | %d contracts → %d puts, %d calls | All computations on GPU",
		cudaDuration.Seconds()*1000, len(options), len(puts), len(calls))

	return puts, calls, nil
}

// MaximizeCUDAUsageComplete processes options with COMPLETE GPU processing - all business calculations on GPU
func (be *BaracudaEngine) MaximizeCUDAUsageComplete(options []OptionContract, stockPrice, availableCash float64, strategy string, expirationDate string, auditSymbol *string) ([]CompleteOptionResult, error) {
	if len(options) == 0 {
		return nil, nil
	}

	if be.engine == nil {
		return nil, fmt.Errorf("engine not initialized")
	}

	if !be.IsCudaAvailable() {
		return nil, fmt.Errorf("CUDA not available - complete GPU processing requires CUDA")
	}

	safeLog("🚀 COMPLETE CUDA: Processing %d contracts with ALL calculations on GPU", len(options))

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

	// Add audit message for complete processing if audit symbol provided
	if auditSymbol != nil && len(results) > 0 {
		// For CUDA complete processing, timing would need to be tracked from the CUDA call
		// Using 0.0 for now as timing needs to be properly integrated
		err := be.logCompleteAuditEntry(*auditSymbol, results, "CUDA", 0.0)
		if err != nil {
			logger.Error.Printf("⚠️ Failed to log complete audit entry: %v", err)
		} else {
			safeLog("📋 AUDIT: Logged complete CUDA BlackScholesCalculation for %s (%d contracts)", *auditSymbol, len(results))
		}
	}

	safeLog("⚡ COMPLETE CUDA: %d contracts processed with ALL calculations on GPU", len(results))
	return results, nil
}

// SetExecutionMode sets the execution mode (cpu, cuda, auto)
func (be *BaracudaEngine) SetExecutionMode(mode string) {
	if be.engine == nil {
		return
	}

	modeBytes := []byte(mode + "\000") // null terminate
	C.barracuda_set_execution_mode(
		be.engine,
		(*C.char)(unsafe.Pointer(&modeBytes[0])))
}

// MaximizeCPUUsageComplete processes options with COMPLETE CPU processing - all business calculations on CPU
func (be *BaracudaEngine) MaximizeCPUUsageComplete(options []OptionContract, stockPrice, availableCash float64, strategy string, expirationDate string, auditSymbol *string) ([]CompleteOptionResult, error) {
	if len(options) == 0 {
		return nil, nil
	}

	if be.engine == nil {
		return nil, fmt.Errorf("engine not initialized")
	}

	safeLog("🖥️  COMPLETE CPU: Processing %d contracts with ALL calculations on CPU", len(options))

	// Set engine to CPU mode
	be.SetExecutionMode("cpu")

	// Calculate days to expiration
	expirationTime, err := time.Parse("2006-01-02", expirationDate)
	if err != nil {
		return nil, fmt.Errorf("invalid expiration date format: %v", err)
	}
	daysToExp := int(time.Until(expirationTime).Hours() / 24)

	// Convert Go contracts to C complete contracts (same as CUDA version)
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

	// Measure CPU processing time
	cpuStart := time.Now()

	// Call CPU complete processing (C++ engine will use CPU path since mode is set to "cpu")
	result := C.barracuda_calculate_options_complete(
		be.engine,
		(*C.CCompleteOptionContract)(unsafe.Pointer(&cContracts[0])),
		C.int(len(options)),
		C.double(availableCash),
		C.int(daysToExp))

	if result != 0 {
		return nil, fmt.Errorf("CPU complete processing failed with code %d", result)
	}

	cpuDuration := time.Since(cpuStart)
	safeLog("🖥️  CPU Processing completed in %.2fms", float64(cpuDuration.Nanoseconds())/1e6)

	// Convert results back to Go (same as CUDA version)
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

	// Add CPU audit logging if audit symbol provided
	if auditSymbol != nil && len(results) > 0 {
		err := be.logCompleteAuditEntry(*auditSymbol, results, "CPU", cpuDuration.Seconds()*1000)
		if err != nil {
			logger.Error.Printf("⚠️ Failed to log CPU audit entry: %v", err)
		} else {
			safeLog("📋 AUDIT: Logged complete CPU BlackScholesCalculation for %s (%d contracts)", *auditSymbol, len(results))
		}
	}

	return results, nil
}

// Close cleans up engine resources
