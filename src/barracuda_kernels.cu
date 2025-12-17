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
    
    if (opt.option_type == 'C') {
        // Call option
        opt.theoretical_price = S * Nd1 - K * exp(-r * T) * Nd2;
        opt.delta = Nd1;
    } else {
        // Put option
        opt.theoretical_price = K * exp(-r * T) * (1.0 - Nd2) - S * (1.0 - Nd1);
        opt.delta = Nd1 - 1.0;
    }
    
    // Greeks calculations
    opt.gamma = nd1 / (S * sigma * sqrt(T));
    opt.theta = -(S * nd1 * sigma) / (2.0 * sqrt(T)) - r * K * exp(-r * T) * Nd2;
    opt.vega = S * nd1 * sqrt(T);
    opt.rho = K * T * exp(-r * T) * Nd2;
    
    if (opt.option_type == 'P') {
        opt.theta += r * K * exp(-r * T);
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

// CUDA kernel for parallel implied volatility calculation (Newton-Raphson)
__global__ void implied_volatility_kernel(
    OptionContract* contracts,
    int num_contracts,
    double tolerance = 1e-8,
    int max_iterations = 100) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contracts) return;
    
    OptionContract& opt = contracts[idx];
    double market_price = opt.theoretical_price; // Using theoretical_price as market price
    
    if (market_price <= 0.01) {
        opt.volatility = 0.25; // Default for very cheap options
        return;
    }
    
    double S = opt.underlying_price;
    double K = opt.strike_price;
    double T = opt.time_to_expiration;
    double r = opt.risk_free_rate;
    
    // Initial volatility guess
    double vol = market_price * 2.0 / (S * sqrt(T));
    vol = fmax(0.05, fmin(vol, 1.0));
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // Black-Scholes calculation
        double d1 = (log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * sqrt(T));
        double d2 = d1 - vol * sqrt(T);
        
        // Normal CDF and PDF
        double nd1 = 0.5 * (1.0 + erf(d1 / sqrt(2.0)));
        double nd2 = 0.5 * (1.0 + erf(d2 / sqrt(2.0)));
        double pdf = exp(-0.5 * d1 * d1) / sqrt(2.0 * M_PI);
        
        double theoretical_price;
        if (opt.option_type == 'C') {
            theoretical_price = S * nd1 - K * exp(-r * T) * nd2;
        } else {
            theoretical_price = K * exp(-r * T) * (1.0 - nd2) - S * (1.0 - nd1);
        }
        
        double vega = S * pdf * sqrt(T);
        double price_diff = theoretical_price - market_price;
        
        if (fabs(price_diff) < tolerance) {
            opt.volatility = vol;
            return;
        }
        
        if (vega > 1e-10) {
            vol = vol - price_diff / vega;
            vol = fmax(0.01, fmin(vol, 3.0)); // Clamp volatility
        } else {
            break;
        }
    }
    
    opt.volatility = vol;
}

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
        double best_put_iv = 0.25;
        double min_put_diff = 1.0;
        
        for (int i = 0; i < num_puts; i++) {
            double delta_diff = fabs(fabs(puts[i].delta) - target_delta);
            if (delta_diff < min_put_diff) {
                min_put_diff = delta_diff;
                best_put_iv = puts[i].volatility;
            }
        }
        
        // Find best call (delta ≈ +0.25)
        double best_call_iv = 0.25;
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

// CUDA kernel for parallel implied volatility calculation
__global__ void implied_volatility_kernel(
    OptionContract* contracts,
    double* market_prices,
    int num_contracts) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contracts) return;
    
    OptionContract& opt = contracts[idx];
    double market_price = market_prices[idx];
    
    if (market_price <= 0) {
        opt.volatility = 0.25; // Default volatility
        return;
    }
    
    // Newton-Raphson method for implied volatility
    const double tolerance = 1e-6;
    const int max_iterations = 50; // Reduced for GPU
    double vol = 0.25; // Initial guess
    
    double S = opt.underlying_price;
    double K = opt.strike_price;
    double T = opt.time_to_expiration;
    double r = opt.risk_free_rate;
    
    for (int i = 0; i < max_iterations; i++) {
        double d1 = (log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * sqrt(T));
        double d2 = d1 - vol * sqrt(T);
        
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
        
        double theoretical_price;
        if (opt.option_type == 'C') {
            theoretical_price = S * Nd1 - K * exp(-r * T) * Nd2;
        } else {
            theoretical_price = K * exp(-r * T) * (1.0 - Nd2) - S * (1.0 - Nd1);
        }
        
        double vega = S * nd1 * sqrt(T);
        double price_diff = theoretical_price - market_price;
        
        if (abs(price_diff) < tolerance) {
            break;
        }
        
        if (vega < 1e-10) {
            break; // Avoid division by zero
        }
        
        vol -= price_diff / vega;
        
        // Keep volatility in reasonable bounds
        if (vol < 0.001) vol = 0.001;
        if (vol > 5.0) vol = 5.0;
    }
    
    opt.volatility = vol;
}

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
    
    void launch_implied_volatility_kernel(OptionContract* d_contracts, double* d_market_prices, int num_contracts) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_contracts + threadsPerBlock - 1) / threadsPerBlock;
        
        implied_volatility_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_contracts, d_market_prices, num_contracts);
        
        cudaDeviceSynchronize();
    }
    
    // New CUDA wrapper functions for maximum GPU utilization
    void launch_implied_volatility_newtonraphson_kernel(OptionContract* d_contracts, int num_contracts) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_contracts + threadsPerBlock - 1) / threadsPerBlock;
        
        implied_volatility_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_contracts, num_contracts);
        cudaDeviceSynchronize();
    }
    
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