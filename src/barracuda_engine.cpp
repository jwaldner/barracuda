#include "barracuda_engine.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <iomanip>

namespace barracuda {

// External CUDA kernel declarations
extern "C" {
    // New CUDA kernels for maximum GPU utilization
    // NOTE: Old kernel removed - using combined IV+Black-Scholes kernel
    void launch_implied_volatility_black_scholes_kernel(OptionContract* d_contracts, int num_contracts);
    void launch_preprocess_contracts_kernel(OptionContract* d_contracts, int num_contracts,
                                          double underlying_price, double time_to_exp, double risk_free_rate);
    // launch_find_25delta_skew_kernel removed - was only used by legacy preprocessing engine
    void launch_separate_puts_calls_kernel(OptionContract* d_contracts, int num_contracts,
                                         int* d_put_indices, int* d_call_indices,
                                         int* d_num_puts, int* d_num_calls);
    void launch_complete_option_analysis_kernel(CompleteOptionContract* d_contracts, 
                                              int num_contracts, double available_cash, 
                                              int days_to_expiration);
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

// Options pricing and Greeks calculation
// ⚠️  WARNING: FOR INTERNAL/TESTING USE ONLY - NOT FOR PRODUCTION
// ⚠️  Use complete batch processing functions instead of this basic calculation
// ⚠️  This function lacks business logic optimization and proper error handling
std::vector<OptionContract> BarracudaEngine::CalculateBlackScholes(
    const std::vector<OptionContract>& contracts,
    const char* audit_symbol) {
    
    std::vector<OptionContract> results = contracts;
    
    // Choose execution path based on CUDA availability and execution mode
    if (cuda_available_ && 
        (execution_mode_ == ExecutionMode::Auto || execution_mode_ == ExecutionMode::CUDA)) {
        
        // CUDA PARALLEL PATH
        OptionContract* d_contracts;
        size_t size = contracts.size() * sizeof(OptionContract);
        
        cudaMalloc(&d_contracts, size);
        cudaMemcpy(d_contracts, results.data(), size, cudaMemcpyHostToDevice);
        
        // Launch combined IV + Black-Scholes kernel
        launch_implied_volatility_black_scholes_kernel(d_contracts, contracts.size());
        
        // Copy results back
        cudaMemcpy(results.data(), d_contracts, size, cudaMemcpyDeviceToHost);
        cudaFree(d_contracts);
        
        // Add audit message if audit_symbol is provided
        if (audit_symbol != nullptr) {
            // Log detailed CUDA processing results - find matching audit symbol
            for (const auto& contract : results) {
                if (contract.symbol == std::string(audit_symbol)) {
                    // Calculate d1 and d2 for plugin formula (CUDA results don't return these)
                    double S = contract.underlying_price;
                    double K = contract.strike_price;
                    double T = contract.time_to_expiration;
                    double r = contract.risk_free_rate;
                    double sigma = contract.volatility;
                    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
                    double d2 = d1 - sigma * sqrt(T);
                    
                    std::string plugin_formula = buildPluginFormula(contract, d1, d2);
                    
                    std::stringstream audit_data;
                    audit_data << "{"
                              << "\"execution_type\": \"CUDA\", "
                              << "\"symbol\": \"" << audit_symbol << "\", "
                              << "\"formula\": \"" << (contract.option_type == 'C' ? "C = S*N(d1) - K*e^(-r*T)*N(d2)" : "P = K*e^(-r*T)*N(-d2) - S*N(-d1)") << "\", "
                              << "\"plug_in_formula\": \"" << plugin_formula << "\", "
                              << "\"contract_data\": {"
                              << "\"symbol\": \"" << contract.symbol << "\", "
                              << "\"strike_price\": " << contract.strike_price << ", "
                              << "\"underlying_price\": " << contract.underlying_price << ", "
                              << "\"time_to_expiration\": " << contract.time_to_expiration << ", "
                              << "\"risk_free_rate\": " << contract.risk_free_rate << ", "
                              << "\"volatility\": " << contract.volatility << ", "
                              << "\"option_type\": \"" << contract.option_type << "\", "
                              << "\"market_close_price\": " << contract.market_close_price
                              << "}, "
                              << "\"calculated_results\": {"
                              << "\"theoretical_price\": " << contract.theoretical_price << ", "
                              << "\"d1\": " << d1 << ", "
                              << "\"d2\": " << d2 << ", "
                              << "\"delta\": " << contract.delta << ", "
                              << "\"gamma\": " << contract.gamma << ", "
                              << "\"theta\": " << contract.theta << ", "
                              << "\"vega\": " << contract.vega << ", "
                              << "\"rho\": " << contract.rho
                              << "}, "
                              << "\"contracts_processed\": " << results.size()
                              << "}";
                    appendAuditCalculation(audit_data.str());
                    break; // Only audit the matching symbol once
                }
            }
        }
        
    } else {
        
        // CPU SEQUENTIAL PATH  
        for (auto& contract : results) {
            // Black-Scholes parameters
            double S = contract.underlying_price;
            double K = contract.strike_price;
            double T = contract.time_to_expiration;
            double r = contract.risk_free_rate;
            double sigma = contract.volatility;
            
            // Protect against zero volatility (causes division by zero)
            if (sigma <= 1e-8) {
                // For zero volatility, option value is intrinsic value only
                if (contract.option_type == 'C') {
                    contract.theoretical_price = fmax(0.0, S - K * exp(-r * T));
                    contract.delta = (S > K * exp(-r * T)) ? 1.0 : 0.0;
                } else {
                    contract.theoretical_price = fmax(0.0, K * exp(-r * T) - S);
                    contract.delta = (S < K * exp(-r * T)) ? -1.0 : 0.0;
                }
                contract.gamma = 0.0;
                contract.theta = 0.0;
                contract.vega = 0.0;
                contract.rho = 0.0;
                continue;
            }
            
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
            contract.vega = S * nd1 * sqrt(T);
            
            if (contract.option_type == 'C') {
                contract.theta = -(S * nd1 * sigma) / (2.0 * sqrt(T)) - r * K * exp(-r * T) * Nd2;
                contract.rho = K * T * exp(-r * T) * Nd2;
            } else {
                // Put theta: -S*N'(d1)*σ/(2√T) + r*K*e^(-r*T)*N(-d2)
                contract.theta = -(S * nd1 * sigma) / (2.0 * sqrt(T)) + r * K * exp(-r * T) * (1.0 - Nd2);
                contract.rho = -K * T * exp(-r * T) * (1.0 - Nd2);
            }
            
            // Apply market standard scaling
            contract.theta /= 365.0;  // Convert to daily decay
            contract.vega /= 100.0;   // Convert to per 1% volatility change
            contract.rho /= 100.0;    // Convert to per 1% rate change
        }
        
        // Add audit message if audit_symbol is provided
        if (audit_symbol != nullptr) {
            // Log detailed CPU processing results with plugin formula
            for (const auto& contract : results) {
                if (contract.symbol == std::string(audit_symbol)) {
                    // Calculate d1 and d2 for this specific contract
                    double S = contract.underlying_price;
                    double K = contract.strike_price;
                    double T = contract.time_to_expiration;
                    double r = contract.risk_free_rate;
                    double sigma = contract.volatility;
                    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
                    double d2 = d1 - sigma * sqrt(T);
                    
                    std::string plugin_formula = buildPluginFormula(contract, d1, d2);
                    
                    std::stringstream audit_data;
                    audit_data << "{"
                              << "\"execution_type\": \"CPU\", "
                              << "\"symbol\": \"" << audit_symbol << "\", "
                              << "\"formula\": \"" << (contract.option_type == 'C' ? "C = S*N(d1) - K*e^(-r*T)*N(d2)" : "P = K*e^(-r*T)*N(-d2) - S*N(-d1)") << "\", "
                              << "\"plug_in_formula\": \"" << plugin_formula << "\", "
                              << "\"contract_data\": {"
                              << "\"symbol\": \"" << contract.symbol << "\", "
                              << "\"strike_price\": " << contract.strike_price << ", "
                              << "\"underlying_price\": " << contract.underlying_price << ", "
                              << "\"time_to_expiration\": " << contract.time_to_expiration << ", "
                              << "\"risk_free_rate\": " << contract.risk_free_rate << ", "
                              << "\"volatility\": " << contract.volatility << ", "
                              << "\"option_type\": \"" << contract.option_type << "\", "
                              << "\"market_close_price\": " << contract.market_close_price
                              << "}, "
                              << "\"calculated_results\": {"
                              << "\"theoretical_price\": " << contract.theoretical_price << ", "
                              << "\"d1\": " << d1 << ", "
                              << "\"d2\": " << d2 << ", "
                              << "\"delta\": " << contract.delta << ", "
                              << "\"gamma\": " << contract.gamma << ", "
                              << "\"theta\": " << contract.theta << ", "
                              << "\"vega\": " << contract.vega << ", "
                              << "\"rho\": " << contract.rho
                              << "}, "
                              << "\"contracts_processed\": " << results.size()
                              << "}";
                    appendAuditCalculation(audit_data.str());
                    break; // Only audit the matching symbol once
                }
            }
        }
    }
    
    return results;
}

double BarracudaEngine::BenchmarkCalculation(int num_contracts, int iterations) {
    // Create test contracts with realistic market-like data
    std::vector<OptionContract> test_contracts(num_contracts);
    for (int i = 0; i < num_contracts; i++) {
        double base_price = 150.0 + (i * 5.0);  // Vary stock prices: 150, 155, 160...
        double strike = base_price * 0.95;      // 5% OTM strikes
        double vol = 0.15 + (i * 0.02);        // Vary IV: 15%, 17%, 19%...
        test_contracts[i] = {
            ("BENCH_" + std::to_string(i)).c_str(), strike, base_price, vol, 0.045, vol, 'P',
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        };
    }
    
    clock_t start = clock();
    
    for (int i = 0; i < iterations; i++) {
        CalculateBlackScholes(test_contracts);
    }
    
    clock_t end = clock();
    double duration = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    return duration; // Return milliseconds
}

extern "C" {
    int barracuda_calculate_options_with_audit(void* engine, OptionContract* c_contracts, int count, const char* audit_symbol);
    void barracuda_set_execution_mode(void* engine, const char* mode);
    int barracuda_calculate_options_complete(void* engine, CompleteOptionContract* c_contracts, int count,
                                           double available_cash, int days_to_expiration, const char* audit_symbol);
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
        return barracuda_calculate_options_with_audit(engine, c_contracts, count, nullptr);
    }
    
    int barracuda_calculate_options_with_audit(void* engine, OptionContract* c_contracts, int count, const char* audit_symbol) {
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
            
            // Copy input fields using correct offsets (accounting for 8-byte alignment)
            cpp_contract.strike_price = *(double*)(c_ptr + 32);
            cpp_contract.underlying_price = *(double*)(c_ptr + 40);
            cpp_contract.time_to_expiration = *(double*)(c_ptr + 48);
            cpp_contract.risk_free_rate = *(double*)(c_ptr + 56);
            cpp_contract.volatility = *(double*)(c_ptr + 64);
            cpp_contract.option_type = *(char*)(c_ptr + 72);
            cpp_contract.market_close_price = *(double*)(c_ptr + 80); // 8-byte aligned after option_type // After char (padded to 8-byte boundary)
            
            cpp_contracts.push_back(cpp_contract);
        }
        
        // Calculate with C++ engine
        auto results = eng->CalculateBlackScholes(cpp_contracts, audit_symbol);
        
        // Copy results back using correct offsets (after market_close_price at offset 80)
        for (size_t i = 0; i < results.size() && i < (size_t)count; i++) {
            char* c_ptr = (char*)&c_contracts[i];
            *(double*)(c_ptr + 88) = results[i].delta;        // Greeks start at offset 88
            *(double*)(c_ptr + 96) = results[i].gamma;
            *(double*)(c_ptr + 104) = results[i].theta;
            *(double*)(c_ptr + 112) = results[i].vega;
            *(double*)(c_ptr + 120) = results[i].rho;
            *(double*)(c_ptr + 128) = results[i].theoretical_price;
        }
        
        return 0; // Success
    }
    
    bool barracuda_is_cuda_available(void* engine) {
        return static_cast<BarracudaEngine*>(engine)->IsCudaAvailable();
    }

    int barracuda_get_device_count(void* engine) {
        return static_cast<BarracudaEngine*>(engine)->GetDeviceCount();
    }

    void barracuda_set_execution_mode(void* engine, const char* mode) {
        auto* eng = static_cast<BarracudaEngine*>(engine);
        std::string mode_str(mode);
        
        if (mode_str == "cpu") {
            eng->SetExecutionMode(ExecutionMode::CPU);
        } else if (mode_str == "cuda") {
            eng->SetExecutionMode(ExecutionMode::CUDA);
        } else {
            eng->SetExecutionMode(ExecutionMode::Auto);
        }
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
        
        // CUDA Phase 2+3: Combined implied volatility calculation + Black-Scholes (single kernel)
        launch_implied_volatility_black_scholes_kernel(d_contracts, count);
        
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
    
    // Complete option processing - ALL calculations on GPU
    int barracuda_calculate_options_complete(void* engine, CompleteOptionContract* c_contracts, int count,
                                           double available_cash, int days_to_expiration, const char* audit_symbol) {
        auto* eng = static_cast<BarracudaEngine*>(engine);
        
        if (!eng->IsCudaAvailable()) {
            return -1; // CUDA required for complete processing
        }
        
        // GPU memory allocation
        CompleteOptionContract* d_contracts;
        size_t contract_size = count * sizeof(CompleteOptionContract);
        
        cudaMalloc(&d_contracts, contract_size);
        if (d_contracts == nullptr) {
            return -2; // GPU memory allocation failed
        }
        
        // Copy contracts to GPU
        cudaMemcpy(d_contracts, c_contracts, contract_size, cudaMemcpyHostToDevice);
        
        // Launch complete processing kernel (IV + Black-Scholes + Business logic)
        extern void launch_complete_option_analysis_kernel(CompleteOptionContract* d_contracts, 
                                                         int num_contracts, double available_cash, 
                                                         int days_to_expiration);
        launch_complete_option_analysis_kernel(d_contracts, count, available_cash, days_to_expiration);
        
        // Copy complete results back to CPU
        cudaMemcpy(c_contracts, d_contracts, contract_size, cudaMemcpyDeviceToHost);
        
        // Handle audit logging through the engine method
        eng->ProcessOptionsCompleteAudit(c_contracts, count, available_cash, audit_symbol);
        
        // Cleanup GPU memory
        cudaFree(d_contracts);
        
        return 0; // Success
    }
}

// Complete option processing audit logging only (processing already done)
void BarracudaEngine::ProcessOptionsCompleteAudit(CompleteOptionContract* contracts, int count, double available_cash, const char* audit_symbol) {
    // Only log if audit_symbol is provided
    if (audit_symbol == nullptr) {
        return;
    }
    
    std::string target_symbol(audit_symbol);
    
    // Find matching contract for the audit symbol
    for (int i = 0; i < count; i++) {
        std::string contract_symbol(contracts[i].symbol, 
            strnlen(contracts[i].symbol, sizeof(contracts[i].symbol)));
        
        if (target_symbol == contract_symbol) {
            // Calculate d1 and d2 for plugin formula using IMPLIED VOLATILITY
            double S = contracts[i].underlying_price;
            double K = contracts[i].strike_price;
            double T = contracts[i].time_to_expiration;
            double r = contracts[i].risk_free_rate;
            double sigma = contracts[i].implied_volatility;  // Use IMPLIED volatility, not input volatility
            double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
            double d2 = d1 - sigma * sqrt(T);
            
            // Convert CompleteOptionContract to OptionContract for buildPluginFormula
            OptionContract contract_for_formula;
            contract_for_formula.symbol = target_symbol;
            contract_for_formula.strike_price = contracts[i].strike_price;
            contract_for_formula.underlying_price = contracts[i].underlying_price;
            contract_for_formula.time_to_expiration = contracts[i].time_to_expiration;
            contract_for_formula.risk_free_rate = contracts[i].risk_free_rate;
            contract_for_formula.volatility = contracts[i].implied_volatility;  // Use IMPLIED volatility
            contract_for_formula.option_type = contracts[i].option_type;
            contract_for_formula.theoretical_price = contracts[i].theoretical_price;
            
            std::string plugin_formula = buildPluginFormula(contract_for_formula, d1, d2);
            
            // Create detailed audit entry for this contract
            std::stringstream audit_data;
            audit_data << "{"
                      << "\"execution_type\": \"CUDA\", "
                      << "\"symbol\": \"" << target_symbol << "\", "
                      << "\"formula\": \"" << (contracts[i].option_type == 'C' ? "C = S*N(d1) - K*e^(-r*T)*N(d2)" : "P = K*e^(-r*T)*N(-d2) - S*N(-d1)") << "\", "
                      << "\"plug_in_formula\": \"" << plugin_formula << "\", "
                      << "\"available_cash\": " << available_cash << ", "
                      << "\"contract_data\": {"
                      << "\"symbol\": \"" << target_symbol << "\", "
                      << "\"strike_price\": " << contracts[i].strike_price << ", "
                      << "\"underlying_price\": " << contracts[i].underlying_price << ", "
                      << "\"time_to_expiration\": " << contracts[i].time_to_expiration << ", "
                      << "\"risk_free_rate\": " << contracts[i].risk_free_rate << ", "
                      << "\"volatility\": " << contracts[i].implied_volatility << ", "  // Use IMPLIED volatility in contract_data
                      << "\"option_type\": \"" << contracts[i].option_type << "\", "
                      << "\"market_close_price\": " << contracts[i].market_close_price << ", "
                      << "\"days_to_expiration\": " << contracts[i].days_to_expiration
                      << "}, "
                      << "\"calculated_results\": {"
                      << "\"theoretical_price\": " << contracts[i].theoretical_price << ", "
                      << "\"implied_volatility\": " << contracts[i].implied_volatility << ", "
                      << "\"delta\": " << contracts[i].delta << ", "
                      << "\"gamma\": " << contracts[i].gamma << ", "
                      << "\"theta\": " << contracts[i].theta << ", "
                      << "\"vega\": " << contracts[i].vega << ", "
                      << "\"rho\": " << contracts[i].rho << ", "
                      << "\"max_contracts\": " << contracts[i].max_contracts << ", "
                      << "\"total_premium\": " << contracts[i].total_premium << ", "
                      << "\"cash_needed\": " << contracts[i].cash_needed << ", "
                      << "\"profit_percentage\": " << contracts[i].profit_percentage << ", "
                      << "\"annualized_return\": " << contracts[i].annualized_return
                      << "}, "
                      << "\"contracts_processed\": " << count
                      << "}";
            appendAuditCalculation(audit_data.str());
            break; // Only log once per symbol
        }
    }
}

void BarracudaEngine::appendAuditCalculation(const std::string& calculation_data) {
    // DEBUG: Log entry to this function
    std::ofstream debugFile("/tmp/barracuda_audit_debug.txt", std::ios::app);
    if (debugFile.is_open()) {
        debugFile << "appendAuditCalculation called with " << calculation_data.length() << " bytes of data" << std::endl;
        debugFile.close();
    }
    
    std::ifstream inFile("audit.json");
    if (!inFile.is_open()) {
        std::ofstream debugFile2("/tmp/barracuda_audit_debug.txt", std::ios::app);
        if (debugFile2.is_open()) {
            debugFile2 << "ERROR: Cannot open audit.json for reading" << std::endl;
            debugFile2.close();
        }
        return; // File doesn't exist, return silently
    }
    
    // Read entire file content
    std::string content((std::istreambuf_iterator<char>(inFile)),
                        std::istreambuf_iterator<char>());
    inFile.close();
    
    // Get current timestamp
    time_t now = time(0);
    struct tm* timeinfo = localtime(&now);
    std::stringstream ss;
    ss << std::put_time(timeinfo, "%Y-%m-%dT%H:%M:%S");
    
    // Find the entries array and insert before the closing ]
    size_t entries_pos = content.find("\"entries\":");
    if (entries_pos != std::string::npos) {
        // Find the array content
        size_t array_start = content.find("[", entries_pos);
        size_t array_end = content.find_last_of("]");
        
        if (array_start != std::string::npos && array_end != std::string::npos) {
            // Check if array is empty by looking for content between [ and ]
            std::string array_content = content.substr(array_start + 1, array_end - array_start - 1);
            // Remove whitespace to check if truly empty
            array_content.erase(std::remove_if(array_content.begin(), array_content.end(), ::isspace), array_content.end());
            bool is_empty = array_content.empty();
            
            // Create detailed audit entry
            std::string audit_entry = (is_empty ? "\n    {\n" : ",\n    {\n");
            audit_entry += "      \"type\": \"BlackScholesCalculation\",\n"
                          "      \"calculation_details\": " + calculation_data + ",\n"
                          "      \"timestamp\": \"" + ss.str() + "-06:00\"\n"
                          "    }";
            
            // Insert before the closing ]
            content.insert(array_end, audit_entry);
            
            // Write back to file
            std::ofstream outFile("audit.json");
            if (outFile.is_open()) {
                outFile << content;
                outFile.close();
                
                std::ofstream debugFile3("/tmp/barracuda_audit_debug.txt", std::ios::app);
                if (debugFile3.is_open()) {
                    debugFile3 << "SUCCESS: audit.json written with calculation_details" << std::endl;
                    debugFile3.close();
                }
            } else {
                std::ofstream debugFile4("/tmp/barracuda_audit_debug.txt", std::ios::app);
                if (debugFile4.is_open()) {
                    debugFile4 << "ERROR: Cannot open audit.json for writing" << std::endl;
                    debugFile4.close();
                }
            }
        } else {
            std::ofstream debugFile5("/tmp/barracuda_audit_debug.txt", std::ios::app);
            if (debugFile5.is_open()) {
                debugFile5 << "ERROR: Cannot find array start/end positions" << std::endl;
                debugFile5.close();
            }
        }
    } else {
        std::ofstream debugFile6("/tmp/barracuda_audit_debug.txt", std::ios::app);
        if (debugFile6.is_open()) {
            debugFile6 << "ERROR: Cannot find entries in audit.json" << std::endl;
            debugFile6.close();
        }
    }
}

// Private helper function to build plugin formula with actual values
std::string BarracudaEngine::buildPluginFormula(const OptionContract& contract, double d1, double d2) const {
    std::stringstream formula;
    formula << std::fixed << std::setprecision(4);
    
    double S = contract.underlying_price;
    double K = contract.strike_price;
    double T = contract.time_to_expiration;
    double r = contract.risk_free_rate;
    double sigma = contract.volatility;
    
    // Show the d1 and d2 calculations with actual values
    formula << "d1 = (ln(" << S << "/" << K << ") + (" << r << " + " << (sigma*sigma/2) << ")*" << T << ") / (" << sigma << "*√" << T << ")\\n";
    formula << "d1 = " << d1 << "\\n";
    formula << "d2 = " << d1 << " - " << sigma << "*√" << T << " = " << d2 << "\\n";
    
    if (contract.option_type == 'C') {
        // Call option: C = S*N(d1) - K*e^(-r*T)*N(d2)
        formula << "C = " << S << " * N(" << d1 << ") - " << K << " * e^(-" << r << "*" << T << ") * N(" << d2 << ")\\n";
        formula << "C = " << contract.theoretical_price;
    } else {
        // Put option: P = K*e^(-r*T)*N(-d2) - S*N(-d1)  
        // Show the actual d1/d2 values, then show what goes into N()
        formula << "P = " << K << " * e^(-" << r << "*" << T << ") * N(" << (-d2) << ") - " << S << " * N(" << (-d1) << ")\\n";
        formula << "P = " << contract.theoretical_price;
    }
    
    return formula.str();
}

} // namespace barracuda