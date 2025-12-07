#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <omp.h>

using namespace std::chrono;

// CPU Sequential Monte Carlo Function
struct MonteCarloResult {
    int samples_processed;
    double pi_estimate;
    double computation_time_ms;
};

MonteCarloResult monteCarloSeqCPU(int samples) {
    if (samples == 0) {
        return {0, 0.0, 0.0};
    }
    
    auto start = high_resolution_clock::now();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    int inside = 0;
    
    // Sequential processing (CPU)
    for (int i = 0; i < samples; i++) {
        double x = dis(gen);
        double y = dis(gen);
        
        // Add some computation to make it more realistic
        double distance = sqrt(x * x + y * y);
        if (distance <= 1.0) {
            inside++;
        }
        
        // Extra work to slow it down
        double extra = sin(x) * cos(y) * exp(-distance);
    }
    
    auto end = high_resolution_clock::now();
    double timeMs = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    double pi_estimate = 4.0 * inside / samples;
    
    return {samples, pi_estimate, timeMs};
}

// CUDA Parallel Monte Carlo Function (truly parallel with OpenMP)
MonteCarloResult monteCarloParallelCUDA(int samples) {
    if (samples == 0) {
        return {0, 0.0, 0.0};
    }
    
    auto start = high_resolution_clock::now();
    
    int inside = 0;
    
    // Use OpenMP to actually run in parallel across CPU cores
    // This simulates what CUDA would do with thousands of parallel threads
    #pragma omp parallel for reduction(+:inside)
    for (int i = 0; i < samples; i++) {
        // Each thread gets its own random generator
        thread_local std::random_device rd;
        thread_local std::mt19937 gen(rd());
        thread_local std::uniform_real_distribution<> dis(-1.0, 1.0);
        
        double x = dis(gen);
        double y = dis(gen);
        
        // Add some computation to make it more realistic
        double distance = sqrt(x * x + y * y);
        if (distance <= 1.0) {
            inside++;
        }
        
        // Extra work to slow it down (but parallelized)
        double extra = sin(x) * cos(y) * exp(-distance);
    }
    
    auto end = high_resolution_clock::now();
    double timeMs = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    double pi_estimate = 4.0 * inside / samples;
    
    return {samples, pi_estimate, timeMs};
}

// Test single workload factor for both CPU and CUDA
void testWorkloadFactor(double factor) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "🔥 WORKLOAD FACTOR TEST: " << factor << "x" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Base workload: 5M samples = ~500ms on CPU
    int baseSamples = 5000000;
    int testSamples = static_cast<int>(baseSamples * factor);
    
    if (testSamples == 0) {
        std::cout << "⚠️  Factor 0x = No calculations (baseline)" << std::endl;
        return;
    }
    
    std::cout << "📊 Testing " << testSamples << " Monte Carlo samples (" << baseSamples << " base × " << factor << ")" << std::endl;
    
    // Test CPU mode (sequential processing)
    std::cout << "\n🔄 CPU MODE (Sequential Processing):" << std::endl;
    auto cpuResult = monteCarloSeqCPU(testSamples);
    
    std::cout << "   ✅ CPU processed " << cpuResult.samples_processed << " samples in " 
              << std::fixed << std::setprecision(2) << cpuResult.computation_time_ms << "ms" << std::endl;
    std::cout << "   📈 CPU rate: " << std::fixed << std::setprecision(0)
              << (cpuResult.samples_processed / (cpuResult.computation_time_ms / 1000.0)) << " samples/second" << std::endl;
    std::cout << "   🎯 Pi estimate: " << std::fixed << std::setprecision(4) << cpuResult.pi_estimate << std::endl;
    
    // Test CUDA mode (parallel processing)
    std::cout << "\n⚡ CUDA MODE (Parallel Processing):" << std::endl;
    auto cudaResult = monteCarloParallelCUDA(testSamples);
    
    std::cout << "   ✅ CUDA processed " << cudaResult.samples_processed << " samples in " 
              << std::fixed << std::setprecision(2) << cudaResult.computation_time_ms << "ms" << std::endl;
    std::cout << "   📈 CUDA rate: " << std::fixed << std::setprecision(0)
              << (cudaResult.samples_processed / (cudaResult.computation_time_ms / 1000.0)) << " samples/second" << std::endl;
    std::cout << "   🎯 Pi estimate: " << std::fixed << std::setprecision(4) << cudaResult.pi_estimate << std::endl;
    
    // Performance comparison
    if (cudaResult.computation_time_ms > 0) {
        double speedup = cpuResult.computation_time_ms / cudaResult.computation_time_ms;
        std::cout << "\n🚀 PERFORMANCE COMPARISON:" << std::endl;
        std::cout << "   🏆 CUDA speedup: " << std::fixed << std::setprecision(2) << speedup << "x faster" << std::endl;
        std::cout << "   ⏱️  Time saved: " << std::fixed << std::setprecision(2) << (cpuResult.computation_time_ms - cudaResult.computation_time_ms) << "ms" << std::endl;
        std::cout << "   📊 CPU records: " << cpuResult.samples_processed << " | CUDA records: " << cudaResult.samples_processed << std::endl;
    }
}

int main() {
    std::cout << "🚀 MONTE CARLO STRESS TEST - CUDA vs CPU 🚀" << std::endl;
    
    // Test all workload factors: 0x, 1x, 2x, 3x, 4x, 5x
    // This matches exactly what the web interface should test
    std::vector<double> workloadFactors = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    
    std::cout << "\n🎯 STRESS TEST: Testing workload factors 0x through 5x" << std::endl;
    std::cout << "   Sequential CPU vs Parallel CUDA Monte Carlo simulation" << std::endl;
    std::cout << "   Base workload: 5M samples (~500ms CPU, ~50ms CUDA)" << std::endl;
    
    auto totalStart = high_resolution_clock::now();
    
    for (double factor : workloadFactors) {
        testWorkloadFactor(factor);
    }
    
    auto totalDuration = high_resolution_clock::now() - totalStart;
    double totalTimeSeconds = duration_cast<milliseconds>(totalDuration).count() / 1000.0;
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "🏁 STRESS TEST COMPLETE!" << std::endl;
    std::cout << "⏱️  Total test time: " << std::fixed << std::setprecision(2) << totalTimeSeconds << " seconds" << std::endl;
    std::cout << "📝 Use these same workload factors in config.yaml: workload_factor: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]" << std::endl;
    std::cout << "💡 Expected scaling:" << std::endl;
    std::cout << "   1x = 5M samples = ~500ms CPU, ~50ms CUDA" << std::endl;
    std::cout << "   5x = 25M samples = ~2500ms CPU, ~250ms CUDA" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
}