package main

import (
	"testing"

	barracuda "github.com/jwaldner/barracuda/barracuda_lib"
)

func TestEngineCreation(t *testing.T) {
	engine := barracuda.NewBaracudaEngine()
	if engine == nil {
		t.Fatal("Failed to create engine")
	}
	defer engine.Close()

	if !engine.IsCudaAvailable() {
		t.Skip("CUDA not available, skipping CUDA tests")
	}

	deviceCount := engine.GetDeviceCount()
	if deviceCount == 0 {
		t.Error("Expected at least 1 CUDA device")
	}
	t.Logf("✅ CUDA devices available: %d", deviceCount)
}

func TestBlackScholesCalculation(t *testing.T) {
	engine := barracuda.NewBaracudaEngine()
	if engine == nil {
		t.Fatal("Failed to create engine")
	}
	defer engine.Close()

	contracts := []barracuda.OptionContract{
		{
			Symbol:           "TEST",
			StrikePrice:      100.0,
			UnderlyingPrice:  100.0,
			TimeToExpiration: 91.0 / 365.0, // ~3 months
			RiskFreeRate:     0.05,
			Volatility:       0.20,
			OptionType:       'C',
			MarketClosePrice: 5.0,
		},
	}

	// Use CUDA batch function instead of single function
	results, err := engine.MaximizeCUDAUsageComplete(contracts, 100.0, 10000.0, "calls", "2026-01-16", nil)
	if err != nil {
		t.Fatalf("CUDA batch calculation failed: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	result := results[0]
	t.Logf("✅ Theoretical Price: %.6f", result.TheoreticalPrice)
	t.Logf("✅ Delta: %.6f", result.Delta)

	if result.TheoreticalPrice <= 0 {
		t.Error("Theoretical price should be positive")
	}
	if result.Delta <= 0 || result.Delta >= 1 {
		t.Error("Delta should be between 0 and 1 for call option")
	}
	if result.Gamma <= 0 {
		t.Error("Gamma should be positive")
	}
}

func TestCPUMode(t *testing.T) {
	engine := barracuda.NewBaracudaEngineForced("cpu")
	if engine == nil {
		t.Fatal("Failed to create CPU engine")
	}
	defer engine.Close()

	contracts := []barracuda.OptionContract{
		{
			Symbol:           "CPU_TEST",
			StrikePrice:      100.0,
			UnderlyingPrice:  100.0,
			TimeToExpiration: 91.0 / 365.0, // ~3 months
			RiskFreeRate:     0.05,
			Volatility:       0.20,
			OptionType:       'P', // Put option
			MarketClosePrice: 3.0,
		},
	}

	// Use CPU batch function instead of single function
	results, err := engine.MaximizeCPUUsageComplete(contracts, 100.0, 10000.0, "puts", "2026-01-16", nil)
	if err != nil {
		t.Fatalf("CPU batch calculation failed: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	result := results[0]
	t.Logf("✅ CPU Put Theoretical Price: %.6f", result.TheoreticalPrice)
	t.Logf("✅ CPU Put Delta: %.6f", result.Delta)

	if result.TheoreticalPrice <= 0 {
		t.Error("Put theoretical price should be positive")
	}
	if result.Delta >= 0 || result.Delta <= -1 {
		t.Error("Put delta should be between -1 and 0")
	}
}
