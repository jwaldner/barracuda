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
}

} // namespace barracuda