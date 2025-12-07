#include "baracuda_engine.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace baracuda;

class BaracudaEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<BaracudaEngine>();
    }

    void TearDown() override {
        engine.reset();
    }

    std::unique_ptr<BaracudaEngine> engine;
    const double TOLERANCE = 1e-6;
};

// Test CUDA initialization
TEST_F(BaracudaEngineTest, CudaInitialization) {
    EXPECT_TRUE(engine->IsCudaAvailable());
    EXPECT_GT(engine->GetDeviceCount(), 0);
    std::cout << "CUDA Device: " << engine->GetDeviceInfo(0) << std::endl;
}

// Test basic Black-Scholes calculation
TEST_F(BaracudaEngineTest, BasicBlackScholesCall) {
    std::vector<OptionContract> contracts = {
        {"AAPL", 100.0, 100.0, 0.25, 0.05, 0.20, 'C', 0, 0, 0, 0, 0, 0}
    };
    
    auto results = engine->CalculateBlackScholes(contracts);
    
    ASSERT_EQ(results.size(), 1);
    
    // For ATM call with 25% time, 5% rate, 20% vol
    // Expected theoretical price around 3.99
    EXPECT_NEAR(results[0].theoretical_price, 3.99, 0.5);
    EXPECT_GT(results[0].delta, 0.4);
    EXPECT_LT(results[0].delta, 0.6);
    EXPECT_GT(results[0].gamma, 0);
    EXPECT_LT(results[0].theta, 0); // Time decay
    EXPECT_GT(results[0].vega, 0);  // Positive vega
}

// Test basic Black-Scholes put
TEST_F(BaracudaEngineTest, BasicBlackScholesPut) {
    std::vector<OptionContract> contracts = {
        {"AAPL", 100.0, 100.0, 0.25, 0.05, 0.20, 'P', 0, 0, 0, 0, 0, 0}
    };
    
    auto results = engine->CalculateBlackScholes(contracts);
    
    ASSERT_EQ(results.size(), 1);
    
    // For ATM put
    EXPECT_GT(results[0].theoretical_price, 2.0);
    EXPECT_LT(results[0].delta, 0);     // Negative delta for puts
    EXPECT_GT(results[0].delta, -0.6);
    EXPECT_GT(results[0].gamma, 0);
    EXPECT_LT(results[0].theta, 0);     // Time decay
    EXPECT_GT(results[0].vega, 0);      // Positive vega
}

// Test batch processing
TEST_F(BaracudaEngineTest, BatchProcessing) {
    const int numContracts = 1000;
    std::vector<OptionContract> contracts;
    
    for (int i = 0; i < numContracts; i++) {
        contracts.push_back({
            "BATCH" + std::to_string(i),
            95.0 + i % 10,          // Strike
            100.0,                  // Underlying
            0.25,                   // Time
            0.05,                   // Rate
            0.20,                   // Vol
            (i % 2 == 0) ? 'C' : 'P', // Alternate calls/puts
            0, 0, 0, 0, 0, 0
        });
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto results = engine->CalculateBlackScholes(contracts);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    ASSERT_EQ(results.size(), numContracts);
    
    // All contracts should have valid prices
    for (const auto& result : results) {
        EXPECT_GT(result.theoretical_price, 0);
        EXPECT_NE(result.delta, 0);
        EXPECT_GT(result.gamma, 0);
    }
    
    std::cout << "Processed " << numContracts << " contracts in " 
              << duration.count() << " microseconds" << std::endl;
}

// Test performance comparison
TEST_F(BaracudaEngineTest, PerformanceTest) {
    const int numContracts = 10000;
    const int iterations = 10;
    
    std::vector<OptionContract> contracts;
    for (int i = 0; i < numContracts; i++) {
        contracts.push_back({
            "PERF" + std::to_string(i % 100),
            90.0 + (i % 20),        // Strike: 90-110
            100.0,                  // Underlying
            0.01 + (i % 50) * 0.01, // Time: 0.01-0.5
            0.05,                   // Rate
            0.15 + (i % 30) * 0.01, // Vol: 15%-45%
            (i % 2 == 0) ? 'C' : 'P',
            0, 0, 0, 0, 0, 0
        });
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; iter++) {
        auto results = engine->CalculateBlackScholes(contracts);
        EXPECT_EQ(results.size(), numContracts);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    int totalCalculations = numContracts * iterations;
    double calculationsPerSec = totalCalculations / (duration.count() / 1000.0);
    
    std::cout << "Performance: " << calculationsPerSec << " calculations/second" << std::endl;
    
    // Expect at least 1M calculations per second
    EXPECT_GT(calculationsPerSec, 1000000);
}

// Test edge cases
TEST_F(BaracudaEngineTest, EdgeCases) {
    std::vector<OptionContract> contracts = {
        // Very short time to expiration
        {"SHORT", 100.0, 100.0, 0.001, 0.05, 0.20, 'C', 0, 0, 0, 0, 0, 0},
        // Very high volatility
        {"HIGHVOL", 100.0, 100.0, 0.25, 0.05, 2.0, 'C', 0, 0, 0, 0, 0, 0},
        // Deep ITM call
        {"DEEPITM", 50.0, 100.0, 0.25, 0.05, 0.20, 'C', 0, 0, 0, 0, 0, 0},
        // Deep OTM put
        {"DEEPOTM", 150.0, 100.0, 0.25, 0.05, 0.20, 'P', 0, 0, 0, 0, 0, 0}
    };
    
    auto results = engine->CalculateBlackScholes(contracts);
    
    ASSERT_EQ(results.size(), 4);
    
    // All should have valid prices
    for (const auto& result : results) {
        EXPECT_GT(result.theoretical_price, 0);
        EXPECT_TRUE(std::isfinite(result.delta));
        EXPECT_TRUE(std::isfinite(result.gamma));
        EXPECT_TRUE(std::isfinite(result.theta));
        EXPECT_TRUE(std::isfinite(result.vega));
    }
}

// Test volatility skew calculation
TEST_F(BaracudaEngineTest, VolatilitySkew) {
    // Create puts and calls around 25-delta
    std::vector<OptionContract> puts = {
        {"AAPL", 85.0, 100.0, 0.25, 0.05, 0.25, 'P', 0, 0, 0, 0, 0, 0},
        {"AAPL", 90.0, 100.0, 0.25, 0.05, 0.23, 'P', 0, 0, 0, 0, 0, 0},
        {"AAPL", 95.0, 100.0, 0.25, 0.05, 0.21, 'P', 0, 0, 0, 0, 0, 0}
    };
    
    std::vector<OptionContract> calls = {
        {"AAPL", 105.0, 100.0, 0.25, 0.05, 0.19, 'C', 0, 0, 0, 0, 0, 0},
        {"AAPL", 110.0, 100.0, 0.25, 0.05, 0.21, 'C', 0, 0, 0, 0, 0, 0},
        {"AAPL", 115.0, 100.0, 0.25, 0.05, 0.23, 'C', 0, 0, 0, 0, 0, 0}
    };
    
    auto skew = engine->Calculate25DeltaSkew(puts, calls, "2024-03-15");
    
    EXPECT_EQ(skew.symbol, "AAPL");
    EXPECT_EQ(skew.expiration, "2024-03-15");
    EXPECT_GT(skew.put_25d_iv, 0);
    EXPECT_GT(skew.call_25d_iv, 0);
    EXPECT_NE(skew.skew, 0); // Should have some skew
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "🧪 Running Baracuda Engine Tests..." << std::endl;
    
    return RUN_ALL_TESTS();
}