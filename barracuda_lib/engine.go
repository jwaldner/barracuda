// Package barracuda provides CUDA-accelerated Black-Scholes option calculations
//
// 🚀 PRODUCTION FUNCTIONS (ONLY these are accessible from external Go code):
//   - NewBarracudaEngine()          - Create engine instance
//   - MaximizeCUDAUsageComplete()  - Complete GPU processing with business logic
//   - MaximizeCPUUsageComplete()   - Complete CPU processing with business logic
//   - MaximizeCUDAUsage()          - Basic GPU processing for simple analysis
//   - Close()                      - Clean up engine resources
//   - IsCudaAvailable()            - Check CUDA availability
//   - GetDeviceCount()             - Get CUDA device count
//   - SetExecutionMode()           - Set execution mode (auto/cuda/cpu)
//   - BenchmarkCalculation()       - Performance benchmarking
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
                                       double available_cash, int days_to_expiration, const char* audit_symbol);

*/
import "C"
import (
	"fmt"
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

// ExecutionMode defines how calculations are performed
type ExecutionMode string

const (
	ExecutionModeAuto ExecutionMode = "auto"
	ExecutionModeCUDA ExecutionMode = "cuda"
	ExecutionModeCPU  ExecutionMode = "cpu"
)

// BarracudaEngine provides CUDA-accelerated options calculations
type BarracudaEngine struct {
	engine        unsafe.Pointer
	executionMode ExecutionMode
}

// NewBarracudaEngine creates a new CUDA-accelerated engine
func NewBarracudaEngine() *BarracudaEngine {
	engine := C.barracuda_create_engine()
	if engine == nil {
		// CUDA engine creation failed
		return nil
	}

	be := &BarracudaEngine{
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

// NewBarracudaEngineForced creates engine with forced execution mode
func NewBarracudaEngineForced(mode string) *BarracudaEngine {
	be := NewBarracudaEngine()
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
func (be *BarracudaEngine) Close() {
	if be.engine != nil {
		C.barracuda_destroy_engine(be.engine)
		be.engine = nil
	}
}

// IsCudaAvailable returns true if CUDA is available
func (be *BarracudaEngine) IsCudaAvailable() bool {
	if be.engine == nil {
		return false
	}
	// Check if CUDA is actually available, regardless of execution mode setting
	return C.barracuda_is_cuda_available(be.engine) != 0
}

// GetDeviceCount returns number of CUDA devices
func (be *BarracudaEngine) GetDeviceCount() int {
	if be.engine == nil {
		return 0
	}
	return int(C.barracuda_get_device_count(be.engine))
}

// BenchmarkCalculation performs performance benchmark calculation
func (be *BarracudaEngine) BenchmarkCalculation(numContracts, iterations int) float64 {
	if be.engine == nil {
		return 0.0
	}
	return float64(C.barracuda_benchmark(be.engine, C.int(numContracts), C.int(iterations)))
}

// MaximizeCUDAUsage processes options with minimal Go loops - maximum GPU utilization
func (be *BarracudaEngine) MaximizeCUDAUsage(options []OptionContract, stockPrice float64) ([]OptionContract, []OptionContract, error) {
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
		safeLog("🔍 INPUT[%d]: %s Strike=%.2f Underlying=%.2f Vol=%.3f Type=%c Market=%.3f",
			i, options[i].Symbol, options[i].StrikePrice, options[i].UnderlyingPrice,
			options[i].Volatility, options[i].OptionType, options[i].MarketClosePrice)
	}

	// Convert Go contracts to C contracts
	cContracts := make([]C.COptionContract, len(options))
	for i, contract := range options {
		// Copy symbol (truncate if too long)
		symbolBytes := []byte(contract.Symbol)
		for j := 0; j < len(symbolBytes) && j < 31; j++ {
			cContracts[i].symbol[j] = C.char(symbolBytes[j])
		}
		cContracts[i].symbol[31] = 0 // Null terminate

		// Copy input fields
		cContracts[i].strike_price = C.double(contract.StrikePrice)
		cContracts[i].underlying_price = C.double(contract.UnderlyingPrice)
		cContracts[i].time_to_expiration = C.double(contract.TimeToExpiration)
		cContracts[i].risk_free_rate = C.double(contract.RiskFreeRate)
		cContracts[i].volatility = C.double(contract.Volatility)
		cContracts[i].option_type = C.char(contract.OptionType)
		cContracts[i].market_close_price = C.double(contract.MarketClosePrice)

		// Initialize output fields
		cContracts[i].delta = C.double(contract.Delta)
		cContracts[i].gamma = C.double(contract.Gamma)
		cContracts[i].theta = C.double(contract.Theta)
		cContracts[i].vega = C.double(contract.Vega)
		cContracts[i].rho = C.double(contract.Rho)
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
			Volatility:       float64(cContracts[i].volatility),
			MarketClosePrice: float64(cContracts[i].market_close_price),
			Delta:            float64(cContracts[i].delta),
			Gamma:            float64(cContracts[i].gamma),
			Theta:            float64(cContracts[i].theta),
			Vega:             float64(cContracts[i].vega),
			Rho:              float64(cContracts[i].rho),
			TheoreticalPrice: float64(cContracts[i].theoretical_price),
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
func (be *BarracudaEngine) MaximizeCUDAUsageComplete(options []OptionContract, stockPrice, availableCash float64, strategy string, expirationDate string, auditSymbol *string) ([]CompleteOptionResult, error) {
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
	var cAuditSymbol *C.char
	if auditSymbol != nil {
		auditBytes := []byte(*auditSymbol + "\000")
		cAuditSymbol = (*C.char)(unsafe.Pointer(&auditBytes[0]))
	}
	result := C.barracuda_calculate_options_complete(
		be.engine,
		(*C.CCompleteOptionContract)(unsafe.Pointer(&cContracts[0])),
		C.int(len(options)),
		C.double(availableCash),
		C.int(daysToExp),
		cAuditSymbol)

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

	safeLog("⚡ COMPLETE CUDA: %d contracts processed with ALL calculations on GPU", len(results))

	return results, nil
}

// SetExecutionMode sets the execution mode (cpu, cuda, auto)
func (be *BarracudaEngine) SetExecutionMode(mode string) {
	if be.engine == nil {
		return
	}

	modeBytes := []byte(mode + "\000") // null terminate
	C.barracuda_set_execution_mode(
		be.engine,
		(*C.char)(unsafe.Pointer(&modeBytes[0])))
}

// MaximizeCPUUsageComplete processes options with COMPLETE CPU processing - all business calculations on CPU
func (be *BarracudaEngine) MaximizeCPUUsageComplete(options []OptionContract, stockPrice, availableCash float64, strategy string, expirationDate string, auditSymbol *string) ([]CompleteOptionResult, error) {
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
	var cAuditSymbol *C.char
	if auditSymbol != nil {
		auditBytes := []byte(*auditSymbol + "\000")
		cAuditSymbol = (*C.char)(unsafe.Pointer(&auditBytes[0]))
	}
	result := C.barracuda_calculate_options_complete(
		be.engine,
		(*C.CCompleteOptionContract)(unsafe.Pointer(&cContracts[0])),
		C.int(len(options)),
		C.double(availableCash),
		C.int(daysToExp),
		cAuditSymbol)

	if result != 0 {
		return nil, fmt.Errorf("CPU complete processing failed with code %d", result)
	}

	cpuDuration := time.Since(cpuStart)

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

	safeLog("⚡ COMPLETE CPU: %.3fms | %d contracts processed with ALL calculations on CPU",
		cpuDuration.Seconds()*1000, len(results))

	return results, nil
}
