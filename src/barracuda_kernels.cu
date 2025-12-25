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
    double q = 0.0; // Zero dividend yield assumption for calculations
    
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
    opt.vega = S * exp(-q * T) * nd1 * sqrt(T);
    
    if (opt.option_type == 'C') {
        opt.theta = -(S * exp(-q * T) * nd1 * sigma) / (2.0 * sqrt(T)) - r * K * exp(-r * T) * Nd2 + q * S * exp(-q * T) * Nd1;
        opt.rho = K * T * exp(-r * T) * Nd2;
    } else {
        opt.theta = -(S * exp(-q * T) * nd1 * sigma) / (2.0 * sqrt(T)) + r * K * exp(-r * T) * N_neg_d2 - q * S * exp(-q * T) * N_neg_d1;
        opt.rho = -K * T * exp(-r * T) * N_neg_d2;
    }
    
    // Apply market standard scaling
    opt.theta /= 365.0;  // Convert to daily decay
    opt.vega /= 100.0;   // Convert to per 1% volatility change
    opt.rho /= 100.0;    // Convert to per 1% rate change
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
    double q = 0.0; // No dividend assumption for IV calculation
    
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
        // Use input volatility as-is (already validated in Go layer)
        // opt.volatility is already set from input, no need to override
    }
    
    // STEP 2: Calculate Black-Scholes with the derived implied volatility
    double sigma = opt.volatility;
    
    // Protect against zero volatility (causes division by zero)
    if (sigma <= 1e-8) {
        // For zero volatility, option value is intrinsic value only
        if (opt.option_type == 'C') {
            opt.theoretical_price = fmax(0.0, stock_price * exp(-q * time_exp) - strike * exp(-rate * time_exp));
            opt.delta = (stock_price > strike * exp((rate - q) * time_exp)) ? exp(-q * time_exp) : 0.0;
        } else {
            opt.theoretical_price = fmax(0.0, strike * exp(-rate * time_exp) - stock_price * exp(-q * time_exp));
            opt.delta = (stock_price < strike * exp((rate - q) * time_exp)) ? -exp(-q * time_exp) : 0.0;
        }
        opt.gamma = 0.0;
        opt.theta = 0.0;
        opt.vega = 0.0;
        opt.rho = 0.0;
        return;
    }
    
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
        opt.theta = -(stock_price * exp(-q * time_exp) * nd1 * sigma) / (2.0 * sqrt(time_exp)) - rate * strike * exp(-rate * time_exp) * Nd2 + q * stock_price * exp(-q * time_exp) * Nd1;
        opt.rho = strike * time_exp * exp(-rate * time_exp) * Nd2;
    } else {
        opt.theoretical_price = strike * exp(-rate * time_exp) * N_neg_d2 - stock_price * exp(-q * time_exp) * N_neg_d1;
        opt.delta = -exp(-q * time_exp) * N_neg_d1;
        opt.theta = -(stock_price * exp(-q * time_exp) * nd1 * sigma) / (2.0 * sqrt(time_exp)) + rate * strike * exp(-rate * time_exp) * N_neg_d2 - q * stock_price * exp(-q * time_exp) * N_neg_d1;
        opt.rho = -strike * time_exp * exp(-rate * time_exp) * N_neg_d2;
    }
    
    // Greeks calculations (common for both calls and puts)
    opt.gamma = exp(-q * time_exp) * nd1 / (stock_price * sigma * sqrt(time_exp));
    opt.vega = stock_price * exp(-q * time_exp) * nd1 * sqrt(time_exp);
    
    // Rho is already calculated correctly above in the option-specific sections
    
    // Apply market standard scaling
    opt.theta /= 365.0;  // Convert to daily decay
    opt.vega /= 100.0;   // Convert to per 1% volatility change
    opt.rho /= 100.0;    // Convert to per 1% rate change
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

// Legacy 25-delta skew kernel removed - not used by complete GPU processing

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
    
    // Legacy 25-delta skew launcher removed - not used by complete GPU processing
    
    void launch_separate_puts_calls_kernel(OptionContract* d_contracts, int num_contracts,
                                         int* d_put_indices, int* d_call_indices,
                                         int* d_num_puts, int* d_num_calls) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_contracts + threadsPerBlock - 1) / threadsPerBlock;
        
        separate_puts_calls_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_contracts, num_contracts, d_put_indices, d_call_indices, d_num_puts, d_call_indices);
        cudaDeviceSynchronize();
    }

// COMPLETE OPTION PROCESSING KERNEL - Calculates ALL business logic on GPU
__global__ void complete_option_analysis_kernel(
    CompleteOptionContract* contracts,
    int num_contracts,
    double available_cash,
    int days_to_expiration) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contracts) return;
    
    CompleteOptionContract& opt = contracts[idx];
    
    // STEP 1: Calculate implied volatility (same as existing kernel)
    double stock_price = opt.underlying_price;
    double strike = opt.strike_price;
    double time_exp = opt.time_to_expiration;
    double rate = opt.risk_free_rate;
    double market_price = opt.market_close_price;
    double q = 0.0; // Market-specific dividend should be passed in
    
    double iv = fmax(0.10, fmin(1.0, market_price / (stock_price * 0.1))); // Market-derived estimate
    if (market_price > 0.01) {
        iv = fmax(0.10, fmin(2.0, market_price * 2.0 / (stock_price * sqrt(time_exp))));
        
        for (int iter = 0; iter < 50; iter++) {
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
            
            if (fabs(price_diff) < 1e-8) break;
            
            if (vega_val > 1e-10) {
                iv = iv - price_diff / vega_val;
                iv = fmax(0.01, fmin(3.0, iv));
            } else {
                break;
            }
        }
    }
    
    opt.implied_volatility = iv;
    
    // STEP 2: Calculate Black-Scholes with derived IV
    double sigma = iv;
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
    opt.vega = stock_price * exp(-q * time_exp) * nd1 * sqrt(time_exp);
    
    if (opt.option_type == 'C') {
        opt.theta = -(stock_price * exp(-q * time_exp) * nd1 * sigma) / (2.0 * sqrt(time_exp)) - rate * strike * exp(-rate * time_exp) * Nd2 + q * stock_price * exp(-q * time_exp) * Nd1;
        opt.rho = strike * time_exp * exp(-rate * time_exp) * Nd2;
    } else {
        opt.theta = -(stock_price * exp(-q * time_exp) * nd1 * sigma) / (2.0 * sqrt(time_exp)) + rate * strike * exp(-rate * time_exp) * N_neg_d2 - q * stock_price * exp(-q * time_exp) * N_neg_d1;
        opt.rho = -strike * time_exp * exp(-rate * time_exp) * N_neg_d2;
    }
    
    // Apply market standard scaling
    opt.theta /= 365.0;  // Convert to daily decay
    opt.vega /= 100.0;   // Convert to per 1% volatility change
    opt.rho /= 100.0;    // Convert to per 1% rate change
    
    // STEP 3: BUSINESS LOGIC CALCULATIONS ON GPU
    double premium_per_share = opt.theoretical_price;
    double cash_needed_per_contract;
    
    // Validate inputs to prevent unrealistic calculations
    if (premium_per_share < 0.01 || available_cash <= 0 || strike <= 0) {
        opt.max_contracts = 0;
        opt.total_premium = 0.0;
        opt.cash_needed = 0.0;
        opt.profit_percentage = 0.0;
        opt.annualized_return = 0.0;
        return;
    }
    
    if (opt.option_type == 'P') {
        // Puts: need cash = strike price × 100 (obligation to buy)
        cash_needed_per_contract = strike * 100.0;
    } else {
        // Calls: pay premium × 100 per contract
        cash_needed_per_contract = premium_per_share * 100.0;
    }
    
    // Calculate max contracts from available cash
    opt.max_contracts = (int)(available_cash / cash_needed_per_contract);
    if (opt.max_contracts < 0) opt.max_contracts = 0;
    
    // Calculate totals
    opt.total_premium = (double)opt.max_contracts * opt.theoretical_price * 100.0;
    opt.cash_needed = (double)opt.max_contracts * cash_needed_per_contract;
    
    // Calculate profit percentage based on theoretical price vs cash needed
    if (opt.cash_needed > 0) {
        opt.profit_percentage = (opt.total_premium / opt.cash_needed) * 100.0;
    } else {
        opt.profit_percentage = 0.0;
    }
    opt.annualized_return = opt.profit_percentage * (365.0 / (double)days_to_expiration);
    opt.days_to_expiration = days_to_expiration;
}

// Launch function for complete processing kernel
void launch_complete_option_analysis_kernel(
    CompleteOptionContract* d_contracts,
    int num_contracts,
    double available_cash,
    int days_to_expiration) {
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_contracts + threadsPerBlock - 1) / threadsPerBlock;
    
    complete_option_analysis_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_contracts, num_contracts, available_cash, days_to_expiration);
    cudaDeviceSynchronize();
}

}

} // namespace barracuda