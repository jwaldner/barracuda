#pragma once
#include "barracuda_engine.hpp"

namespace barracuda {

struct PreprocessingResult {
    std::vector<OptionContract> puts;
    std::vector<OptionContract> calls;
    double preprocessing_time_ms;
    int total_contracts_processed;
};

struct VolatilitySkewResult {
    double put_25d_iv;
    double call_25d_iv; 
    double skew;
    double atm_iv;
    double calculation_time_ms;
    int contracts_analyzed;
};

class PreprocessingEngine {
public:
    PreprocessingEngine(bool cuda_available);
    
    // Parallel preprocessing: set underlying price, time to exp, risk-free rate, calculate IV
    PreprocessingResult ProcessOptionsData(
        const std::vector<OptionContract>& raw_options,
        double underlying_price,
        double time_to_expiration,
        double risk_free_rate
    );
    
    // Parallel 25-delta skew calculation
    VolatilitySkewResult Calculate25DeltaSkew(
        const std::vector<OptionContract>& puts,
        const std::vector<OptionContract>& calls
    );
    
    // Parallel implied volatility calculation for batch
    std::vector<double> CalculateImpliedVolatilities(
        const std::vector<OptionContract>& contracts
    );

private:
    bool cuda_available_;
    
    // CPU fallback methods
    double EstimateImpliedVolatilityCPU(double market_price, double stock_price, 
                                       double strike_price, double time_to_exp, 
                                       double risk_free_rate, char option_type);
};

} // namespace barracuda