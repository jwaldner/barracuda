#include "barracuda_engine.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>

using namespace barracuda;

void testBasicCalculation() {
    std::cout << "🧪 Testing Basic Black-Scholes Calculation..." << std::endl;
    
    BarracudaEngine engine;
    
    std::vector<OptionContract> contracts = {
        {"TEST", 100.0, 100.0, 0.25, 0.05, 0.20, 'C', 0, 0, 0, 0, 0, 0}
    };
    
    auto results = engine.CalculateBlackScholes(contracts);
    
    assert(results.size() == 1);
    assert(results[0].theoretical_price > 0);
    assert(results[0].delta > 0 && results[0].delta < 1);
    assert(results[0].gamma > 0);
    
    std::cout << "   ✅ Theoretical Price: " << results[0].theoretical_price << std::endl;
    std::cout << "   ✅ Delta: " << results[0].delta << std::endl;
    std::cout << "   ✅ Gamma: " << results[0].gamma << std::endl;
}

void testBatchProcessing() {
    std::cout << "🧪 Testing Batch Processing..." << std::endl;
    
    BarracudaEngine engine;
    const int numContracts = 1000;
    
    std::vector<OptionContract> contracts;
    for (int i = 0; i < numContracts; i++) {
        contracts.push_back({
            "BATCH" + std::to_string(i % 10),
            95.0 + (i % 10),
            100.0,
            0.25,
            0.05,
            0.20,
            (i % 2 == 0) ? 'C' : 'P',
            0, 0, 0, 0, 0, 0
        });
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto results = engine.CalculateBlackScholes(contracts);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    assert(results.size() == numContracts);
    
    // Verify all results are valid
    for (const auto& result : results) {
        assert(result.theoretical_price > 0);
        assert(std::isfinite(result.delta));
        assert(std::isfinite(result.gamma));
    }
    
    std::cout << "   ✅ Processed " << numContracts << " contracts in " 
              << duration.count() << " microseconds" << std::endl;
}

void testPerformance() {
    std::cout << "🧪 Testing Performance..." << std::endl;
    
    BarracudaEngine engine;
    const int numContracts = 10000;
    const int iterations = 5;
    
    std::vector<OptionContract> contracts;
    for (int i = 0; i < numContracts; i++) {
        contracts.push_back({
            "PERF" + std::to_string(i % 100),
            90.0 + (i % 20),
            100.0,
            0.01 + (i % 50) * 0.01,
            0.05,
            0.15 + (i % 30) * 0.01,
            (i % 2 == 0) ? 'C' : 'P',
            0, 0, 0, 0, 0, 0
        });
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; iter++) {
        auto results = engine.CalculateBlackScholes(contracts);
        assert(results.size() == numContracts);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    int totalCalculations = numContracts * iterations;
    double calculationsPerSec = totalCalculations / (duration.count() / 1000.0);
    
    std::cout << "   ✅ Performance: " << calculationsPerSec << " calculations/second" << std::endl;
    
    // Should be at least 500K calculations per second
    assert(calculationsPerSec > 500000);
}

void testCudaAvailability() {
    std::cout << "🧪 Testing CUDA Availability..." << std::endl;
    
    BarracudaEngine engine;
    
    std::cout << "   ✅ CUDA Available: " << (engine.IsCudaAvailable() ? "YES" : "NO") << std::endl;
    std::cout << "   ✅ CUDA Devices: " << engine.GetDeviceCount() << std::endl;
    
    if (engine.IsCudaAvailable()) {
        for (int i = 0; i < engine.GetDeviceCount(); i++) {
            std::cout << "   ✅ Device " << i << ": " << engine.GetDeviceInfo(i) << std::endl;
        }
    }
}

int main() {
    std::cout << "🔥 BARRACUDA ENGINE TESTS 🔥" << std::endl << std::endl;
    
    try {
        testCudaAvailability();
        std::cout << std::endl;
        
        testBasicCalculation();
        std::cout << std::endl;
        
        testBatchProcessing();
        std::cout << std::endl;
        
        testPerformance();
        std::cout << std::endl;
        
        std::cout << "🎉 All tests passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown error" << std::endl;
        return 1;
    }
    
    return 0;
}