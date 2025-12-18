#include "barracuda_engine.hpp"
#include "barracuda_preprocessing.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <iostream>
#include <chrono>
#include <algorithm>

namespace barracuda {

// External CUDA kernel declarations
extern "C" {
    void launch_black_scholes_kernel(OptionContract* d_contracts, int num_contracts);
    void launch_monte_carlo_kernel(double* d_paths, double S0, double r, double sigma, 
                                  double T, int num_steps, int num_paths, curandState* d_states);
    void launch_setup_kernel(curandState* d_states, unsigned long seed, int num_paths);
    
    // New CUDA kernels for maximum GPU utilization
    void launch_implied_volatility_newtonraphson_kernel(OptionContract* d_contracts, int num_contracts);
    void launch_preprocess_contracts_kernel(OptionContract* d_contracts, int num_contracts,
                                          double underlying_price, double time_to_exp, double risk_free_rate);
    void launch_find_25delta_skew_kernel(OptionContract* d_puts, int num_puts,
                                       OptionContract* d_calls, int num_calls, double* d_results);
    void launch_separate_puts_calls_kernel(OptionContract* d_contracts, int num_contracts,
                                         int* d_put_indices, int* d_call_indices,
                                         int* d_num_puts, int* d_num_calls);
}



BarracudaEngine::BarracudaEngine() : cuda_available_(false), device_count_(0), execution_mode_(ExecutionMode::Auto) {
    InitializeCUDA();
}

BarracudaEngine::~BarracudaEngine() {
    // Cleanup CUDA resources if needed
}

bool BarracudaEngine::InitializeCUDA() {
    cudaError_t error = cudaGetDeviceCount(&device_count_);
    
    if (error != cudaSuccess || device_count_ == 0) {
        std::cerr << "CUDA not available: " << cudaGetErrorString(error) << std::endl;
        cuda_available_ = false;
        return false;
    }
    
    // Set device 0 as default
    cudaSetDevice(0);
    cuda_available_ = true;
    
    // CUDA initialized silently
    return true;
}

std::vector<OptionContract> BarracudaEngine::CalculateBlackScholes(
    const std::vector<OptionContract>& contracts) {
    
    std::vector<OptionContract> results = contracts;
    
    // Choose execution path based on CUDA availability and execution mode
    if (cuda_available_ && 
        (execution_mode_ == ExecutionMode::Auto || execution_mode_ == ExecutionMode::CUDA)) {
        
        // CUDA PARALLEL PATH
        OptionContract* d_contracts;
        size_t size = contracts.size() * sizeof(OptionContract);
        
        cudaMalloc(&d_contracts, size);
        cudaMemcpy(d_contracts, results.data(), size, cudaMemcpyHostToDevice);
        
        // Launch kernel via wrapper function
        launch_black_scholes_kernel(d_contracts, contracts.size());
        
        // Copy results back
        cudaMemcpy(results.data(), d_contracts, size, cudaMemcpyDeviceToHost);
        cudaFree(d_contracts);
        
    } else {
        
        // CPU SEQUENTIAL PATH  
        for (auto& contract : results) {
            // Black-Scholes parameters
            double S = contract.underlying_price;
            double K = contract.strike_price;
            double T = contract.time_to_expiration;
            double r = contract.risk_free_rate;
            double sigma = contract.volatility;
            
            // Calculate d1 and d2
            double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
            double d2 = d1 - sigma * sqrt(T);
            
            // Standard normal CDF approximation
            auto norm_cdf = [](double x) -> double {
                return 0.5 * (1.0 + erf(x / sqrt(2.0)));
            };
            
            // Standard normal PDF
            auto norm_pdf = [](double x) -> double {
                return exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
            };
            
            double Nd1 = norm_cdf(d1);
            double Nd2 = norm_cdf(d2);
            double nd1 = norm_pdf(d1);
            
            if (contract.option_type == 'C') {
                // Call option
                contract.theoretical_price = S * Nd1 - K * exp(-r * T) * Nd2;
                contract.delta = Nd1;
            } else {
                // Put option
                contract.theoretical_price = K * exp(-r * T) * (1.0 - Nd2) - S * (1.0 - Nd1);
                contract.delta = Nd1 - 1.0;
            }
            
            // Greeks calculations
            contract.gamma = nd1 / (S * sigma * sqrt(T));
            contract.theta = -(S * nd1 * sigma) / (2.0 * sqrt(T)) - r * K * exp(-r * T) * Nd2;
            contract.vega = S * nd1 * sqrt(T);
            contract.rho = K * T * exp(-r * T) * Nd2;
            
            if (contract.option_type == 'P') {
                contract.theta += r * K * exp(-r * T);
                contract.rho = -contract.rho;
            }
        }
    }
    
    return results;
}

std::vector<OptionContract> BarracudaEngine::BatchProcessOptions(
    const std::vector<OptionContract>& contracts, int batch_size) {
    
    std::vector<OptionContract> results;
    results.reserve(contracts.size());
    
    for (size_t i = 0; i < contracts.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, contracts.size());
        std::vector<OptionContract> batch(contracts.begin() + i, contracts.begin() + end);
        
        auto batch_results = CalculateBlackScholes(batch);
        results.insert(results.end(), batch_results.begin(), batch_results.end());
    }
    
    return results;
}

VolatilitySkew BarracudaEngine::Calculate25DeltaSkew(
    const std::vector<OptionContract>& puts,
    const std::vector<OptionContract>& calls,
    const std::string& expiration) {
    
    VolatilitySkew skew;
    skew.expiration = expiration;
    
    // Find 25-delta options (simplified - would need iterative search in practice)
    double target_delta = 0.25;
    double best_put_iv = 0.0, best_call_iv = 0.0;
    double min_put_diff = 1.0, min_call_diff = 1.0;
    
    // Calculate options first to get deltas
    auto calculated_puts = CalculateBlackScholes(puts);
    auto calculated_calls = CalculateBlackScholes(calls);
    
    for (const auto& put : calculated_puts) {
        double delta_diff = std::abs(std::abs(put.delta) - target_delta);
        if (delta_diff < min_put_diff) {
            min_put_diff = delta_diff;
            best_put_iv = put.volatility;
            skew.symbol = put.symbol;
        }
    }
    
    for (const auto& call : calculated_calls) {
        double delta_diff = std::abs(call.delta - target_delta);
        if (delta_diff < min_call_diff) {
            min_call_diff = delta_diff;
            best_call_iv = call.volatility;
        }
    }
    
    skew.put_25d_iv = best_put_iv;
    skew.call_25d_iv = best_call_iv;
    skew.skew = best_put_iv - best_call_iv;
    skew.atm_iv = (best_put_iv + best_call_iv) / 2.0;
    
    return skew;
}

// Newton-Raphson method for implied volatility calculation
double BarracudaEngine::CalculateImpliedVolatility(
    double market_price, double stock_price, double strike_price,
    double time_to_expiration, double risk_free_rate, char option_type) {
    
    const double tolerance = 1e-6;
    const int max_iterations = 100;
    double vol = 0.25; // Initial guess: 25%
    
    for (int i = 0; i < max_iterations; i++) {
        // Calculate theoretical price and vega using Black-Scholes
        double d1 = (log(stock_price / strike_price) + (risk_free_rate + 0.5 * vol * vol) * time_to_expiration) / (vol * sqrt(time_to_expiration));
        double d2 = d1 - vol * sqrt(time_to_expiration);
        
        auto norm_cdf = [](double x) -> double {
            return 0.5 * (1.0 + erf(x / sqrt(2.0)));
        };
        
        auto norm_pdf = [](double x) -> double {
            return exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
        };
        
        double Nd1 = norm_cdf(d1);
        double Nd2 = norm_cdf(d2);
        double nd1 = norm_pdf(d1);
        
        double theoretical_price;
        if (option_type == 'C') {
            theoretical_price = stock_price * Nd1 - strike_price * exp(-risk_free_rate * time_to_expiration) * Nd2;
        } else {
            theoretical_price = strike_price * exp(-risk_free_rate * time_to_expiration) * (1.0 - Nd2) - stock_price * (1.0 - Nd1);
        }
        
        double vega = stock_price * nd1 * sqrt(time_to_expiration);
        
        double price_diff = theoretical_price - market_price;
        
        if (abs(price_diff) < tolerance) {
            return vol;
        }
        
        if (vega < 1e-10) {
            break; // Avoid division by zero
        }
        
        vol -= price_diff / vega;
        
        // Keep volatility in reasonable bounds
        if (vol < 0.001) vol = 0.001;
        if (vol > 5.0) vol = 5.0;
    }
    
    return vol; // Return best estimate even if not converged
}

// Main batch processing function - routes to CUDA or CPU
std::vector<SymbolAnalysisResult> BarracudaEngine::AnalyzeSymbolsBatch(
    const std::vector<std::string>& symbols,
    const std::map<std::string, double>& stock_prices,
    const std::map<std::string, std::vector<OptionContract>>& options_chains,
    const std::string& expiration_date) {
    
    // Simplified implementation - these batch functions need refactoring
    return std::vector<SymbolAnalysisResult>();
}

// CUDA parallel processing implementation - STUB FOR NOW
std::vector<SymbolAnalysisResult> BarracudaEngine::AnalyzeSymbolsBatchParallel(
    const std::vector<std::string>& symbols,
    const std::map<std::string, double>& stock_prices,
    const std::map<std::string, std::vector<OptionContract>>& options_chains,
    const std::string& expiration_date) {
    
    // TODO: Implement parallel batch processing
    return std::vector<SymbolAnalysisResult>();
}

// CPU sequential processing implementation - STUB FOR NOW
std::vector<SymbolAnalysisResult> BarracudaEngine::AnalyzeSymbolsBatchSequential(
    const std::vector<std::string>& symbols,
    const std::map<std::string, double>& stock_prices,
    const std::map<std::string, std::vector<OptionContract>>& options_chains,
    const std::string& expiration_date) {
    
    // TODO: Implement sequential batch processing 
    return std::vector<SymbolAnalysisResult>();
}

std::vector<double> BarracudaEngine::CalculateRollingVolatility(
    const std::vector<MarketData>& price_data, int window_size) {
    
    std::vector<double> volatilities;
    
    for (size_t i = window_size; i < price_data.size(); i++) {
        std::vector<double> returns;
        
        for (int j = 0; j < window_size - 1; j++) {
            double ret = log(price_data[i - j].price / price_data[i - j - 1].price);
            returns.push_back(ret);
        }
        
        // Calculate standard deviation
        double mean = 0.0;
        for (double ret : returns) mean += ret;
        mean /= returns.size();
        
        double variance = 0.0;
        for (double ret : returns) {
            variance += (ret - mean) * (ret - mean);
        }
        variance /= returns.size() - 1;
        
        // Annualized volatility (assuming daily data)
        double vol = sqrt(variance * 252);
        volatilities.push_back(vol);
    }
    
    return volatilities;
}

std::vector<double> BarracudaEngine::MonteCarloSimulation(
    const std::vector<OptionContract>& portfolio,
    int num_simulations, int time_steps) {
    
    if (!cuda_available_) {
        return std::vector<double>(num_simulations, 0.0);
    }
    
    // For simplicity, simulate just the first contract
    if (portfolio.empty()) return {};
    
    const auto& contract = portfolio[0];
    
    double* d_paths;
    curandState* d_states;
    
    cudaMalloc(&d_paths, num_simulations * sizeof(double));
    cudaMalloc(&d_states, num_simulations * sizeof(curandState));
    
    // Setup random states and run Monte Carlo via wrapper functions
    launch_setup_kernel(d_states, time(nullptr), num_simulations);
    launch_monte_carlo_kernel(d_paths, contract.underlying_price, contract.risk_free_rate,
                             contract.volatility, contract.time_to_expiration,
                             time_steps, num_simulations, d_states);
    
    std::vector<double> results(num_simulations);
    cudaMemcpy(results.data(), d_paths, 
               num_simulations * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_paths);
    cudaFree(d_states);
    
    return results;
}

double BarracudaEngine::BenchmarkCalculation(int num_contracts, int iterations) {
    // Create test contracts
    std::vector<OptionContract> test_contracts(num_contracts);
    for (int i = 0; i < num_contracts; i++) {
        test_contracts[i] = {
            "TEST", 100.0 + i, 100.0, 0.25, 0.05, 0.20, 'C',
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        };
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        CalculateBlackScholes(test_contracts);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    return duration.count() / 1000.0; // Return milliseconds
}

std::string BarracudaEngine::GetDeviceInfo(int device_id) const {
    if (device_id >= device_count_) return "Invalid device ID";
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    return std::string("Device ") + std::to_string(device_id) + ": " + 
           prop.name + " (Compute " + std::to_string(prop.major) + "." + 
           std::to_string(prop.minor) + ")";
}

// C interface implementations
extern "C" {
    void* barracuda_create_engine() {
        return new BarracudaEngine();
    }
    
    void barracuda_destroy_engine(void* engine) {
        delete static_cast<BarracudaEngine*>(engine);
    }
    
    bool barracuda_initialize_cuda(void* engine) {
        return static_cast<BarracudaEngine*>(engine)->InitializeCUDA();
    }
    
    int barracuda_calculate_options(void* engine, OptionContract* c_contracts, int count) {
        auto* eng = static_cast<BarracudaEngine*>(engine);
        
        // C struct layout: char[32] symbol, then doubles, char, then output doubles
        // Offsets: symbol=0, strike=32, underlying=40, time=48, rate=56, vol=64, type=72
        // Outputs: delta=80, gamma=88, theta=96, vega=104, rho=112, price=120
        
        // Convert C contracts to C++ contracts
        std::vector<OptionContract> cpp_contracts;
        cpp_contracts.reserve(count);
        
        for (int i = 0; i < count; i++) {
            OptionContract cpp_contract;
            // Copy char array to std::string
            char* c_ptr = (char*)&c_contracts[i];
            char symbol_buf[33];
            memcpy(symbol_buf, c_ptr, 32);
            symbol_buf[32] = '\0';
            cpp_contract.symbol = std::string(symbol_buf);
            
            // Copy input fields using correct offsets
            cpp_contract.strike_price = *(double*)(c_ptr + 32);
            cpp_contract.underlying_price = *(double*)(c_ptr + 40);
            cpp_contract.time_to_expiration = *(double*)(c_ptr + 48);
            cpp_contract.risk_free_rate = *(double*)(c_ptr + 56);
            cpp_contract.volatility = *(double*)(c_ptr + 64);
            cpp_contract.option_type = *(char*)(c_ptr + 72);
            
            cpp_contracts.push_back(cpp_contract);
        }
        
        // Calculate with C++ engine
        auto results = eng->CalculateBlackScholes(cpp_contracts);
        
        // Copy results back using correct offsets
        for (size_t i = 0; i < results.size() && i < (size_t)count; i++) {
            char* c_ptr = (char*)&c_contracts[i];
            *(double*)(c_ptr + 80) = results[i].delta;
            *(double*)(c_ptr + 88) = results[i].gamma;
            *(double*)(c_ptr + 96) = results[i].theta;
            *(double*)(c_ptr + 104) = results[i].vega;
            *(double*)(c_ptr + 112) = results[i].rho;
            *(double*)(c_ptr + 120) = results[i].theoretical_price;
        }
        
        return 0; // Success
    }
    
    bool barracuda_is_cuda_available(void* engine) {
        return static_cast<BarracudaEngine*>(engine)->IsCudaAvailable();
    }

    int barracuda_get_device_count(void* engine) {
        return static_cast<BarracudaEngine*>(engine)->GetDeviceCount();
    }

    double barracuda_benchmark(void* engine, int num_contracts, int iterations) {
        return static_cast<BarracudaEngine*>(engine)->BenchmarkCalculation(num_contracts, iterations);
    }
    
    // CUDA-MAXIMIZED: Zero Go loops, 100% GPU processing
    int barracuda_cuda_maximize_processing(void* engine, OptionContract* c_contracts, int count,
                                         double stock_price, int* put_count, int* call_count) {
        auto* eng = static_cast<BarracudaEngine*>(engine);
        
        if (!eng->IsCudaAvailable()) {
            return -1; // CUDA required
        }
        
        // GPU memory allocation
        OptionContract* d_contracts;
        int* d_put_indices;
        int* d_call_indices;
        int* d_num_puts;
        int* d_num_calls;
        
        size_t contract_size = count * sizeof(OptionContract);
        cudaMalloc(&d_contracts, contract_size);
        cudaMalloc(&d_put_indices, count * sizeof(int));
        cudaMalloc(&d_call_indices, count * sizeof(int));
        cudaMalloc(&d_num_puts, sizeof(int));
        cudaMalloc(&d_num_calls, sizeof(int));
        
        // Initialize counters on GPU
        cudaMemset(d_num_puts, 0, sizeof(int));
        cudaMemset(d_num_calls, 0, sizeof(int));
        
        // Copy contracts to GPU
        cudaMemcpy(d_contracts, c_contracts, contract_size, cudaMemcpyHostToDevice);
        
        // CUDA Phase 1: Parallel preprocessing (set prices, time, rates)
        launch_preprocess_contracts_kernel(d_contracts, count, stock_price, 0.085, 0.05);
        
        // CUDA Phase 2: Parallel implied volatility (Newton-Raphson)
        launch_implied_volatility_newtonraphson_kernel(d_contracts, count);
        
        // CUDA Phase 3: Parallel Black-Scholes calculation
        launch_black_scholes_kernel(d_contracts, count);
        
        // CUDA Phase 4: Parallel put/call separation
        launch_separate_puts_calls_kernel(d_contracts, count, d_put_indices, d_call_indices, 
                                        d_num_puts, d_num_calls);
        
        // Copy results back to CPU
        cudaMemcpy(c_contracts, d_contracts, contract_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(put_count, d_num_puts, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(call_count, d_num_calls, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Cleanup GPU memory
        cudaFree(d_contracts);
        cudaFree(d_put_indices);
        cudaFree(d_call_indices);
        cudaFree(d_num_puts);
        cudaFree(d_num_calls);
        
        return 0; // Success
    }
}

} // namespace barracuda