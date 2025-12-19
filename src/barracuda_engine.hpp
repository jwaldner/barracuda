#ifndef BARRACUDA_ENGINE_HPP
#define BARRACUDA_ENGINE_HPP

#include <vector>
#include <string>
#include <memory>
#include <map>

// C interface for Go integration
extern "C" {
    // C struct for complete option processing with all business calculations
    struct CCompleteOptionContract {
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
    };
}

namespace barracuda {

enum class ExecutionMode {
    Auto,
    CUDA,
    CPU
};

struct OptionContract {
    std::string symbol;
    double strike_price;
    double underlying_price;
    double time_to_expiration;
    double risk_free_rate;
    double volatility;
    char option_type; // 'C' for call, 'P' for put
    double market_close_price; // Market close price for IV calculation
    
    // Greeks (output)
    double delta;
    double gamma;
    double theta;
    double vega;
    double rho;
    double theoretical_price;
};

// Complete option contract with all business calculations
struct CompleteOptionContract {
    char symbol[32];
    double strike_price;
    double underlying_price;
    double time_to_expiration;
    double risk_free_rate;
    double volatility;
    char option_type; // 'C' for call, 'P' for put
    double market_close_price;
    
    // Greeks (output)
    double delta;
    double gamma;
    double theta;
    double vega;
    double rho;
    double theoretical_price;
    double implied_volatility;
    
    // Business calculations (output)
    int max_contracts;
    double total_premium;
    double cash_needed;
    double profit_percentage;
    double annualized_return;
    int days_to_expiration;
};

struct MarketData {
    std::string symbol;
    double price;
    double bid;
    double ask;
    long long timestamp;
    double volume;
};

struct VolatilitySkew {
    std::string symbol;
    std::string expiration;
    double put_25d_iv;
    double call_25d_iv;
    double skew;
    double atm_iv;
};

struct SymbolAnalysisResult {
    std::string symbol;
    double stock_price;
    std::string expiration;
    
    // Options data with calculated implied volatilities
    std::vector<OptionContract> puts_with_iv;
    std::vector<OptionContract> calls_with_iv;
    
    // 25-delta analysis
    VolatilitySkew volatility_skew;
    OptionContract best_25d_put;
    OptionContract best_25d_call;
    
    // Performance metadata
    double calculation_time_ms;
    std::string execution_mode;  // "CUDA" or "CPU"
    int total_options_processed;
};

class BarracudaEngine {
private:
    bool cuda_available_;
    int device_count_;
    ExecutionMode execution_mode_;
    
public:
    BarracudaEngine();
    ~BarracudaEngine();
    
    // Initialize CUDA context
    bool InitializeCUDA();
    
    // Options pricing and Greeks calculation
    std::vector<OptionContract> CalculateBlackScholes(
        const std::vector<OptionContract>& contracts,
        const char* audit_symbol = nullptr);
    
    // Batch processing for large option chains
    std::vector<OptionContract> BatchProcessOptions(
        const std::vector<OptionContract>& contracts,
        int batch_size = 1024);
    
    // Volatility surface calculations
    VolatilitySkew Calculate25DeltaSkew(
        const std::vector<OptionContract>& puts,
        const std::vector<OptionContract>& calls,
        const std::string& expiration);
    
    // Legacy batch analysis functions removed - replaced by complete GPU processing
    
    // Utility functions (public for C interface)
    bool IsCudaAvailable() const { return cuda_available_; }
    int GetDeviceCount() const { return device_count_; }
    std::string GetDeviceInfo(int device_id) const;
    
    // Execution mode setter (public for C interface)
    void SetExecutionMode(ExecutionMode mode) { execution_mode_ = mode; }
    
    // Performance benchmarking (public for C interface)
    double BenchmarkCalculation(int num_contracts, int iterations);
    
private:
    // Helper function to append audit messages to JSON file
    void appendAuditMessage(const std::string& message);
    
    // Helper function to append detailed calculation data to JSON file
    void appendAuditCalculation(const std::string& calculation_data);
    
    // Implied volatility calculation
    double CalculateImpliedVolatility(
        double market_price, double stock_price, double strike_price,
        double time_to_expiration, double risk_free_rate, char option_type);
    
    // Market data analysis
    std::vector<double> CalculateRollingVolatility(
        const std::vector<MarketData>& price_data,
        int window_size);
    
    // Portfolio risk calculations
    double CalculatePortfolioVar(
        const std::vector<OptionContract>& portfolio,
        const std::vector<std::vector<double>>& correlation_matrix,
        double confidence_level = 0.95);
    
    // Monte Carlo simulation
    std::vector<double> MonteCarloSimulation(
        const std::vector<OptionContract>& portfolio,
        int num_simulations,
        int time_steps);
};

// C interface for Go FFI
extern "C" {
    // Engine management
    void* barracuda_create_engine();
    void barracuda_destroy_engine(void* engine);
    bool barracuda_initialize_cuda(void* engine);
    
    // Options calculations
    int baracuda_calculate_options(
        void* engine,
        OptionContract* contracts,
        int count);
    
    // Complete option processing (all calculations on GPU)
    int barracuda_calculate_options_complete(
        void* engine,
        CompleteOptionContract* contracts,
        int count,
        double available_cash,
        int days_to_expiration);
    
    // Volatility skew
    int baracuda_calculate_skew(
        void* engine,
        const char* symbol,
        const char* expiration,
        OptionContract* puts,
        int put_count,
        OptionContract* calls,
        int call_count,
        VolatilitySkew* result);
    
    // Market data analysis
    int baracuda_rolling_volatility(
        void* engine,
        MarketData* data,
        int data_count,
        int window_size,
        double* results);
    
    // Performance info
    bool baracuda_is_cuda_available(void* engine);
    int baracuda_get_device_count(void* engine);
    double baracuda_benchmark(void* engine, int num_contracts, int iterations);
    
    // Monte Carlo PI estimation (for workload testing)
    int baracuda_monte_carlo_pi(void* engine, int samples);
}

} // namespace barracuda

#endif // BARRACUDA_ENGINE_HPP