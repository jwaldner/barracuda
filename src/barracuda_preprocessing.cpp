#include "barracuda_preprocessing.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>

namespace barracuda {

// CUDA kernel declarations
extern "C" {
    void launch_preprocessing_kernel(OptionContract* d_contracts, int num_contracts,
                                   double underlying_price, double time_to_exp, double risk_free_rate);
    void launch_implied_volatility_kernel(OptionContract* d_contracts, int num_contracts);
    void launch_skew_calculation_kernel(OptionContract* d_puts, int num_puts,
                                      OptionContract* d_calls, int num_calls,
                                      double* d_result); // [put_25d_iv, call_25d_iv, skew, atm_iv]
    
    // New CUDA kernel declarations
    void launch_implied_volatility_black_scholes_kernel(OptionContract* d_contracts, int num_contracts);
    void launch_preprocess_contracts_kernel(OptionContract* d_contracts, int num_contracts,
                                          double underlying_price, double time_to_exp, double risk_free_rate);
    void launch_find_25delta_skew_kernel(OptionContract* d_puts, int num_puts,
                                       OptionContract* d_calls, int num_calls, double* d_result);
}

PreprocessingEngine::PreprocessingEngine(bool cuda_available) 
    : cuda_available_(cuda_available) {}

PreprocessingResult PreprocessingEngine::ProcessOptionsData(
    const std::vector<OptionContract>& raw_options,
    double underlying_price,
    double time_to_expiration,
    double risk_free_rate) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    PreprocessingResult result;
    std::vector<OptionContract> processed = raw_options;
    
    if (cuda_available_ && processed.size() > 10) { // Lower threshold for more CUDA usage
        // MAXIMUM CUDA PARALLEL PATH
        OptionContract* d_contracts;
        size_t size = processed.size() * sizeof(OptionContract);
        
        cudaMalloc(&d_contracts, size);
        cudaMemcpy(d_contracts, processed.data(), size, cudaMemcpyHostToDevice);
        
        // CUDA Step 1: Parallel preprocessing (set prices, time, rates)
        launch_preprocess_contracts_kernel(d_contracts, processed.size(), 
                                         underlying_price, time_to_expiration, risk_free_rate);
        
        // CUDA Step 2: Parallel implied volatility calculation (Newton-Raphson)
        launch_implied_volatility_black_scholes_kernel(d_contracts, processed.size());
        
        // Copy results back
        cudaMemcpy(processed.data(), d_contracts, size, cudaMemcpyDeviceToHost);
        cudaFree(d_contracts);
        
    } else {
        // CPU SEQUENTIAL PATH
        for (auto& option : processed) {
            option.underlying_price = underlying_price;
            option.time_to_expiration = time_to_expiration;
            option.risk_free_rate = risk_free_rate;
            
            // Calculate implied volatility if market price available
            if (option.theoretical_price > 0.01) {
                option.volatility = EstimateImpliedVolatilityCPU(
                    option.theoretical_price, underlying_price, option.strike_price,
                    time_to_expiration, risk_free_rate, option.option_type);
            } else {
                option.volatility = 0.25; // Default
            }
        }
    }
    
    // Separate puts and calls
    for (const auto& option : processed) {
        if (option.option_type == 'P') {
            result.puts.push_back(option);
        } else {
            result.calls.push_back(option);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.preprocessing_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    result.total_contracts_processed = processed.size();
    
    return result;
}

VolatilitySkewResult PreprocessingEngine::Calculate25DeltaSkew(
    const std::vector<OptionContract>& puts,
    const std::vector<OptionContract>& calls) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    VolatilitySkewResult result = {};
    
    if (puts.empty() || calls.empty()) {
        return result;
    }
    
    if (cuda_available_ && (puts.size() + calls.size()) > 5) { // Lower threshold for more CUDA usage
        // MAXIMUM CUDA PARALLEL PATH
        OptionContract* d_puts;
        OptionContract* d_calls; 
        double* d_result;
        
        size_t puts_size = puts.size() * sizeof(OptionContract);
        size_t calls_size = calls.size() * sizeof(OptionContract);
        
        cudaMalloc(&d_puts, puts_size);
        cudaMalloc(&d_calls, calls_size);
        cudaMalloc(&d_result, 4 * sizeof(double)); // [put_25d_iv, call_25d_iv, skew, atm_iv]
        
        cudaMemcpy(d_puts, puts.data(), puts_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_calls, calls.data(), calls_size, cudaMemcpyHostToDevice);
        
        // CUDA: Parallel 25-delta skew search and calculation
        launch_find_25delta_skew_kernel(d_puts, puts.size(), d_calls, calls.size(), d_result);
        
        double host_result[4];
        cudaMemcpy(host_result, d_result, 4 * sizeof(double), cudaMemcpyDeviceToHost);
        
        result.put_25d_iv = host_result[0];
        result.call_25d_iv = host_result[1]; 
        result.skew = host_result[2];
        result.atm_iv = host_result[3];
        
        cudaFree(d_puts);
        cudaFree(d_calls);
        cudaFree(d_result);
        
    } else {
        // CPU SEQUENTIAL PATH
        double target_delta = 0.25;
        OptionContract best_put, best_call;
        double min_put_diff = 1.0;
        double min_call_diff = 1.0;
        
        // Find 25-delta put
        for (const auto& put : puts) {
            double delta_diff = std::abs(std::abs(put.delta) - target_delta);
            if (delta_diff < min_put_diff) {
                min_put_diff = delta_diff;
                best_put = put;
            }
        }
        
        // Find 25-delta call
        for (const auto& call : calls) {
            double delta_diff = std::abs(call.delta - target_delta);
            if (delta_diff < min_call_diff) {
                min_call_diff = delta_diff;
                best_call = call;
            }
        }
        
        result.put_25d_iv = best_put.volatility;
        result.call_25d_iv = best_call.volatility;
        result.skew = best_put.volatility - best_call.volatility;
        result.atm_iv = (best_put.volatility + best_call.volatility) / 2.0;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.calculation_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    result.contracts_analyzed = puts.size() + calls.size();
    
    return result;
}

double PreprocessingEngine::EstimateImpliedVolatilityCPU(double market_price, double stock_price,
                                                        double strike_price, double time_to_exp,
                                                        double risk_free_rate, char option_type) {
    const double tolerance = 1e-8;
    const int max_iterations = 100;
    
    // Initial volatility guess
    double vol = market_price * 2.0 / (stock_price * std::sqrt(time_to_exp));
    vol = std::max(0.05, std::min(vol, 1.0));
    
    for (int i = 0; i < max_iterations; i++) {
        // Black-Scholes calculation
        double d1 = (std::log(stock_price / strike_price) + (risk_free_rate + 0.5 * vol * vol) * time_to_exp) / 
                    (vol * std::sqrt(time_to_exp));
        double d2 = d1 - vol * std::sqrt(time_to_exp);
        
        // Normal CDF approximation
        double nd1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
        double nd2 = 0.5 * (1.0 + std::erf(d2 / std::sqrt(2.0)));
        double pdf = std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * M_PI);
        
        double theoretical_price;
        if (option_type == 'C') {
            theoretical_price = stock_price * nd1 - strike_price * std::exp(-risk_free_rate * time_to_exp) * nd2;
        } else {
            theoretical_price = strike_price * std::exp(-risk_free_rate * time_to_exp) * (1.0 - nd2) - 
                               stock_price * (1.0 - nd1);
        }
        
        double vega = stock_price * pdf * std::sqrt(time_to_exp);
        double price_diff = theoretical_price - market_price;
        
        if (std::abs(price_diff) < tolerance) {
            return vol;
        }
        
        if (vega > 1e-10) {
            vol = vol - price_diff / vega;
            vol = std::max(0.01, std::min(vol, 3.0)); // Clamp volatility
        } else {
            break;
        }
    }
    
    return vol;
}

} // namespace barracuda