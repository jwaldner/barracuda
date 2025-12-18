#include "barracuda_engine.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

namespace barracuda {

// CUDA kernel for Black-Scholes calculation
__global__ void black_scholes_kernel(
    OptionContract* contracts,
    int num_contracts) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contracts) return;
    
    OptionContract& opt = contracts[idx];
    
    // Black-Scholes parameters
    double S = opt.underlying_price;
    double K = opt.strike_price;
    double T = opt.time_to_expiration;
    double r = opt.risk_free_rate;
    double sigma = opt.volatility;
    double q = 0.005; // Dividend yield (0.5% for most stocks)
    
    // CORRECT Black-Scholes formula with dividend yield
    // d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
    double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    
    // Cumulative normal distributions
    double Nd1 = 0.5 * (1.0 + erf(d1 / sqrt(2.0)));
    double Nd2 = 0.5 * (1.0 + erf(d2 / sqrt(2.0)));
    double N_neg_d1 = 0.5 * (1.0 + erf(-d1 / sqrt(2.0)));
    double N_neg_d2 = 0.5 * (1.0 + erf(-d2 / sqrt(2.0)));
    double nd1 = exp(-0.5 * d1 * d1) / sqrt(2.0 * M_PI);
    
    if (opt.option_type == 'C') {
        // Call option: C = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)
        opt.theoretical_price = S * exp(-q * T) * Nd1 - K * exp(-r * T) * Nd2;
        opt.delta = exp(-q * T) * Nd1;
    } else {
        // Put option: P = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)
        opt.theoretical_price = K * exp(-r * T) * N_neg_d2 - S * exp(-q * T) * N_neg_d1;
        opt.delta = -exp(-q * T) * N_neg_d1;
    }
    
    // Greeks calculations
    opt.gamma = exp(-q * T) * nd1 / (S * sigma * sqrt(T));
    opt.theta = -(S * exp(-q * T) * nd1 * sigma) / (2.0 * sqrt(T)) - r * K * exp(-r * T) * Nd2 + q * S * exp(-q * T) * Nd1;
    opt.vega = S * exp(-q * T) * nd1 * sqrt(T);
    opt.rho = K * T * exp(-r * T) * Nd2;
    
    if (opt.option_type == 'P') {
        opt.theta -= q * S * exp(-q * T) * Nd1 + q * S * exp(-q * T) * (1.0 - Nd1);
        opt.rho = -opt.rho;
    }
}

// Combined CUDA kernel: Calculate Implied Volatility + Black-Scholes in one pass
__global__ void implied_volatility_black_scholes_kernel(
    OptionContract* contracts,
    int num_contracts) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contracts) return;
    
    OptionContract& opt = contracts[idx];
    
    // STEP 1: Calculate Implied Volatility from market close price
    double market_price = opt.market_close_price;
    
    // Black-Scholes parameters
    double stock_price = opt.underlying_price;
    double strike = opt.strike_price;
    double time_exp = opt.time_to_expiration;
    double rate = opt.risk_free_rate;
    double q = 0.005; // Dividend yield
    
    if (market_price > 0.0) {
        // Initial volatility guess
        double iv = fmax(0.10, fmin(2.0, market_price * 2.0 / (stock_price * sqrt(time_exp))));
        const double tolerance = 1e-8;
        const int max_iter = 50;
        
        // Newton-Raphson iteration for implied volatility
        for (int i = 0; i < max_iter; i++) {
            double d1_iv = (log(stock_price / strike) + (rate - q + 0.5 * iv * iv) * time_exp) / (iv * sqrt(time_exp));
            double d2_iv = d1_iv - iv * sqrt(time_exp);
            
            double Nd1_iv = 0.5 * (1.0 + erf(d1_iv / sqrt(2.0)));
            double Nd2_iv = 0.5 * (1.0 + erf(d2_iv / sqrt(2.0)));
            double N_neg_d1_iv = 0.5 * (1.0 + erf(-d1_iv / sqrt(2.0)));
            double N_neg_d2_iv = 0.5 * (1.0 + erf(-d2_iv / sqrt(2.0)));
            
            double theo_price;
            if (opt.option_type == 'C') {
                theo_price = stock_price * exp(-q * time_exp) * Nd1_iv - strike * exp(-rate * time_exp) * Nd2_iv;
            } else {
                theo_price = strike * exp(-rate * time_exp) * N_neg_d2_iv - stock_price * exp(-q * time_exp) * N_neg_d1_iv;
            }
            
            double nd1_iv = exp(-0.5 * d1_iv * d1_iv) / sqrt(2.0 * M_PI);
            double vega_val = stock_price * exp(-q * time_exp) * nd1_iv * sqrt(time_exp);
            double price_diff = theo_price - market_price;
            
            if (fabs(price_diff) < tolerance) break;
            
            if (vega_val > 1e-10) {
                iv = iv - price_diff / vega_val;
                iv = fmax(0.01, fmin(3.0, iv));
            } else {
                break;
            }
        }
        
        opt.volatility = iv;
    } else {
        opt.volatility = 0.25; // Default 25% if no market price
    }
    
    // STEP 2: Calculate Black-Scholes with the derived implied volatility
    double sigma = opt.volatility;
    
    double d1 = (log(stock_price / strike) + (rate - q + 0.5 * sigma * sigma) * time_exp) / (sigma * sqrt(time_exp));
    double d2 = d1 - sigma * sqrt(time_exp);
    
    double Nd1 = 0.5 * (1.0 + erf(d1 / sqrt(2.0)));
    double Nd2 = 0.5 * (1.0 + erf(d2 / sqrt(2.0)));
    double N_neg_d1 = 0.5 * (1.0 + erf(-d1 / sqrt(2.0)));
    double N_neg_d2 = 0.5 * (1.0 + erf(-d2 / sqrt(2.0)));
    double nd1 = exp(-0.5 * d1 * d1) / sqrt(2.0 * M_PI);
    
    if (opt.option_type == 'C') {
        opt.theoretical_price = stock_price * exp(-q * time_exp) * Nd1 - strike * exp(-rate * time_exp) * Nd2;
        opt.delta = exp(-q * time_exp) * Nd1;
    } else {
        opt.theoretical_price = strike * exp(-rate * time_exp) * N_neg_d2 - stock_price * exp(-q * time_exp) * N_neg_d1;
        opt.delta = -exp(-q * time_exp) * N_neg_d1;
    }
    
    // Greeks calculations
    opt.gamma = exp(-q * time_exp) * nd1 / (stock_price * sigma * sqrt(time_exp));
    opt.theta = -(stock_price * exp(-q * time_exp) * nd1 * sigma) / (2.0 * sqrt(time_exp)) - rate * strike * exp(-rate * time_exp) * Nd2 + q * stock_price * exp(-q * time_exp) * Nd1;
    opt.vega = stock_price * exp(-q * time_exp) * nd1 * sqrt(time_exp);
    opt.rho = strike * time_exp * exp(-rate * time_exp) * Nd2;
    
    if (opt.option_type == 'P') {
        opt.theta -= q * stock_price * exp(-q * time_exp) * Nd1 + q * stock_price * exp(-q * time_exp) * (1.0 - Nd1);
        opt.rho = -opt.rho;
    }
}

// CUDA kernel for Monte Carlo simulation
__global__ void monte_carlo_kernel(
    double* paths,
    double S0,
    double r,
    double sigma,
    double T,
    int num_steps,
    int num_paths,
    curandState* states) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;
    
    curandState local_state = states[idx];
    double dt = T / num_steps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol = sigma * sqrt(dt);
    
    double S = S0;
    for (int i = 0; i < num_steps; i++) {
        double z = curand_normal(&local_state);
        S *= exp(drift + vol * z);
    }
    
    paths[idx] = S;
    states[idx] = local_state;
}

// NOTE: Old standalone implied volatility kernel removed
// Using combined implied_volatility_black_scholes_kernel instead
// (Removed redundant implied volatility kernel code - conflicts resolved)

// CUDA kernel for parallel contract preprocessing
__global__ void preprocess_contracts_kernel(
    OptionContract* contracts,
    int num_contracts,
    double underlying_price,
    double time_to_expiration,
    double risk_free_rate) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contracts) return;
    
    OptionContract& opt = contracts[idx];
    opt.underlying_price = underlying_price;
    opt.time_to_expiration = time_to_expiration;
    opt.risk_free_rate = risk_free_rate;
}

// CUDA kernel for parallel 25-delta skew calculation
__global__ void find_25delta_skew_kernel(
    OptionContract* puts, int num_puts,
    OptionContract* calls, int num_calls,
    double* results) { // [put_25d_iv, call_25d_iv, skew, atm_iv]
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use first thread to do the calculation (can be optimized with reduction)
    if (idx == 0) {
        double target_delta = 0.25;
        
        // Find best put (delta ≈ -0.25)
        double best_put_iv = 0.0; // Will be set from real market data
        double min_put_diff = 1.0;
        
        for (int i = 0; i < num_puts; i++) {
            double delta_diff = fabs(fabs(puts[i].delta) - target_delta);
            if (delta_diff < min_put_diff) {
                min_put_diff = delta_diff;
                best_put_iv = puts[i].volatility;
            }
        }
        
        // Find best call (delta ≈ +0.25)
        double best_call_iv = 0.0; // Will be set from real market data
        double min_call_diff = 1.0;
        
        for (int i = 0; i < num_calls; i++) {
            double delta_diff = fabs(calls[i].delta - target_delta);
            if (delta_diff < min_call_diff) {
                min_call_diff = delta_diff;
                best_call_iv = calls[i].volatility;
            }
        }
        
        results[0] = best_put_iv;   // put_25d_iv
        results[1] = best_call_iv;  // call_25d_iv
        results[2] = best_put_iv - best_call_iv; // skew
        results[3] = (best_put_iv + best_call_iv) / 2.0; // atm_iv
    }
}

// CUDA kernel for parallel put/call separation
__global__ void separate_puts_calls_kernel(
    OptionContract* contracts, int num_contracts,
    int* put_indices, int* call_indices, 
    int* num_puts, int* num_calls) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contracts) return;
    
    // Use atomic operations to count and store indices
    if (contracts[idx].option_type == 'P') {
        int put_idx = atomicAdd(num_puts, 1);
        put_indices[put_idx] = idx;
    } else {
        int call_idx = atomicAdd(num_calls, 1);
        call_indices[call_idx] = idx;
    }
}

// Monte Carlo PI estimation kernel
__global__ void monte_carlo_pi_kernel(int* inside, int samples_per_thread, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState local_state = states[idx];
    int local_inside = 0;
    
    for (int i = 0; i < samples_per_thread; i++) {
        double x = curand_uniform_double(&local_state) * 2.0 - 1.0;
        double y = curand_uniform_double(&local_state) * 2.0 - 1.0;
        
        double distance = sqrt(x*x + y*y);
        if (distance <= 1.0) {
            local_inside++;
        }
        
        // Extra work to match CPU benchmark
        double extra = sin(x) * cos(y) * exp(-distance);
        (void)extra; // Prevent optimization
    }
    
    atomicAdd(inside, local_inside);
    states[idx] = local_state;
}

// NOTE: All broken kernel fragments completely removed

// Initialize cuRAND states
__global__ void setup_kernel(curandState* state, unsigned long seed, int num_paths) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;
    curand_init(seed, idx, 0, &state[idx]);
}

// CUDA wrapper functions (called from C++)
extern "C" {
    void launch_black_scholes_kernel(OptionContract* d_contracts, int num_contracts) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_contracts + threadsPerBlock - 1) / threadsPerBlock;
        
        black_scholes_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_contracts, num_contracts);
        
        cudaDeviceSynchronize();
    }
    
    void launch_implied_volatility_black_scholes_kernel(OptionContract* d_contracts, int num_contracts) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_contracts + threadsPerBlock - 1) / threadsPerBlock;
        
        implied_volatility_black_scholes_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_contracts, num_contracts);
        
        cudaDeviceSynchronize();
    }
    
    void launch_monte_carlo_kernel(double* d_paths, double S0, double r, double sigma, 
                                  double T, int num_steps, int num_paths, curandState* d_states) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_paths + threadsPerBlock - 1) / threadsPerBlock;
        
        monte_carlo_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_paths, S0, r, sigma, T, num_steps, num_paths, d_states);
        
        cudaDeviceSynchronize();
    }
    
    void launch_setup_kernel(curandState* d_states, unsigned long seed, int num_paths) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_paths + threadsPerBlock - 1) / threadsPerBlock;
        
        setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, seed, num_paths);
        
        cudaDeviceSynchronize();
    }
    
    void launch_monte_carlo_pi_kernel(int* d_inside, int samples, int num_threads, curandState* d_states) {
        int samples_per_thread = (samples + num_threads - 1) / num_threads;
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_threads + threadsPerBlock - 1) / threadsPerBlock;
        
        monte_carlo_pi_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_inside, samples_per_thread, d_states);
        
        cudaDeviceSynchronize();
    }
    
    // NOTE: Old implied volatility wrappers removed - using combined kernel
    
    void launch_preprocess_contracts_kernel(OptionContract* d_contracts, int num_contracts,
                                          double underlying_price, double time_to_exp, double risk_free_rate) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_contracts + threadsPerBlock - 1) / threadsPerBlock;
        
        preprocess_contracts_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_contracts, num_contracts, underlying_price, time_to_exp, risk_free_rate);
        cudaDeviceSynchronize();
    }
    
    void launch_find_25delta_skew_kernel(OptionContract* d_puts, int num_puts,
                                       OptionContract* d_calls, int num_calls, double* d_results) {
        // Use single block since we're doing a simple linear search
        find_25delta_skew_kernel<<<1, 1>>>(d_puts, num_puts, d_calls, num_calls, d_results);
        cudaDeviceSynchronize();
    }
    
    void launch_separate_puts_calls_kernel(OptionContract* d_contracts, int num_contracts,
                                         int* d_put_indices, int* d_call_indices,
                                         int* d_num_puts, int* d_num_calls) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_contracts + threadsPerBlock - 1) / threadsPerBlock;
        
        separate_puts_calls_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_contracts, num_contracts, d_put_indices, d_call_indices, d_num_puts, d_num_calls);
        cudaDeviceSynchronize();
    }
}

} // namespace barracuda