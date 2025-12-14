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
*/
import "C"
import (
	"fmt"
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
	return be.executionMode == ExecutionModeCUDA && C.barracuda_is_cuda_available(be.engine) != 0
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
