#include "baracuda_engine.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace baracuda;
using namespace std::chrono;

void runBenchmark(const std::string& mode, int numContracts, int iterations) {
    std::cout << "\n=== " << mode << " Benchmark ===" << std::endl;
    std::cout << "Contracts: " << numContracts << ", Iterations: " << iterations << std::endl;
    
    BaracudaEngine engine;
    
    // Force execution mode
    if (mode == "CUDA" && !engine.IsCudaAvailable()) {
        std::cout << "CUDA not available, skipping..." << std::endl;
        return;
    }
    
    // Generate test contracts
    std::vector<OptionContract> contracts(numContracts);
    for (int i = 0; i < numContracts; i++) {
        contracts[i] = {
            "TEST" + std::to_string(i % 100),
            95.0 + (i % 50),        // Strike: $95-$145
            100.0 + (i % 100),      // Underlying: $100-$200  
            0.25,                   // 3 months
            0.05,                   // 5% risk-free rate
            0.20 + (i % 30) * 0.01, // Volatility: 20%-50%
            (i % 2 == 0) ? 'C' : 'P', // Alternate calls/puts
            0, 0, 0, 0, 0, 0        // Greeks (output)
        };
    }
    
    std::cout << "Starting " << mode << " calculations..." << std::endl;
    
    auto start = high_resolution_clock::now();
    
    // Run iterations
    for (int iter = 0; iter < iterations; iter++) {
        auto results = engine.CalculateBlackScholes(contracts);
        
        // Progress for long tests
        if (iterations > 10 && iter % (iterations / 10) == 0) {
            int progress = (iter * 100) / iterations;
            std::cout << "Progress: " << progress << "%" << std::endl;
        }
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    
    // Calculate metrics
    double totalTime = duration.count() / 1000.0; // milliseconds
    int totalCalculations = numContracts * iterations;
    double calculationsPerSec = totalCalculations / (totalTime / 1000.0);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "✅ " << mode << " Results:" << std::endl;
    std::cout << "   Total Time: " << totalTime << " ms" << std::endl;
    std::cout << "   Total Calculations: " << totalCalculations << std::endl;
    std::cout << "   Performance: " << calculationsPerSec << " calc/sec" << std::endl;
    std::cout << "   Time per Contract: " << totalTime / totalCalculations << " ms" << std::endl;
}

void comparePerformance(int numContracts, int iterations) {
    std::cout << "\n🚀 BARACUDA PERFORMANCE COMPARISON 🚀" << std::endl;
    std::cout << "Testing " << numContracts << " contracts × " << iterations << " iterations" << std::endl;
    
    auto start = high_resolution_clock::now();
    
    // Run CPU benchmark
    runBenchmark("CPU", numContracts, iterations);
    
    // Run CUDA benchmark
    runBenchmark("CUDA", numContracts, iterations);
    
    auto totalTime = duration_cast<seconds>(high_resolution_clock::now() - start);
    std::cout << "\n⏱️  Total benchmark time: " << totalTime.count() << " seconds" << std::endl;
}

int main() {
    std::cout << "🔥 BARACUDA ENGINE BENCHMARK 🔥" << std::endl;
    
    BaracudaEngine engine;
    
    // System info
    std::cout << "\n📊 System Information:" << std::endl;
    std::cout << "   CUDA Available: " << (engine.IsCudaAvailable() ? "YES" : "NO") << std::endl;
    std::cout << "   CUDA Devices: " << engine.GetDeviceCount() << std::endl;
    
    if (engine.IsCudaAvailable()) {
        for (int i = 0; i < engine.GetDeviceCount(); i++) {
            std::cout << "   Device " << i << ": " << engine.GetDeviceInfo(i) << std::endl;
        }
    }
    
    // Progressive workload tests
    std::vector<std::pair<int, int>> workloads = {
        {100, 100},      // Small: 10K calculations
        {1000, 100},     // Medium: 100K calculations  
        {10000, 100},    // Large: 1M calculations
        {50000, 20},     // Very Large: 1M calculations
        {100000, 10}     // Massive: 1M calculations
    };
    
    std::cout << "\n📈 Running Progressive Workload Tests..." << std::endl;
    
    for (const auto& workload : workloads) {
        int contracts = workload.first;
        int iterations = workload.second;
        int totalCalcs = contracts * iterations;
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "WORKLOAD: " << contracts << " contracts × " << iterations << " iterations = " << totalCalcs << " total calculations" << std::endl;
        
        comparePerformance(contracts, iterations);
    }
    
    std::cout << "\n🏁 Benchmark Complete!" << std::endl;
    
    return 0;
}