package main

import (
	"fmt"
	"time"

	barracuda "github.com/jwaldner/barracuda/barracuda_lib"
	testdata "github.com/jwaldner/barracuda/test_data"
)

func main() {
	fmt.Println("🔬 Testing CUDA vs CPU Engine with Apple Options Data")
	fmt.Println("📅 Data: AAPL options expiring 2026-01-16 (3rd Friday)")
	fmt.Println("💰 Stock Price: $274.115 (captured 2025-12-16)")
	fmt.Println()

	// Get mock Apple data
	symbol, stockPrice, mockOptionsChain, expiration := testdata.GetAppleTestData()

	fmt.Printf("📊 Symbol: %s, Price: $%.2f, Expiration: %s\n", symbol, stockPrice, expiration)
	fmt.Printf("📈 Options in chain: %d contracts\n", len(mockOptionsChain[symbol]))
	fmt.Println()

	// Convert mock data to engine format
	var engineOptions []barracuda.OptionContract
	for _, mockOpt := range mockOptionsChain[symbol] {
		engineOpt := barracuda.OptionContract{
			Symbol:           mockOpt.Symbol,
			StrikePrice:      mockOpt.Strike,
			UnderlyingPrice:  stockPrice,
			TimeToExpiration: testdata.MockAppleOptionsData.TimeToExpiration,
			RiskFreeRate:     testdata.MockAppleOptionsData.RiskFreeRate,
			Volatility:       0.25, // Initial guess - will calculate IV from market price
			OptionType:       mockOpt.Type,
			TheoreticalPrice: mockOpt.GetMidPrice(), // Use mid price as "market" price
		}
		engineOptions = append(engineOptions, engineOpt)
	}

	// Test both engines
	testEngine("CPU", engineOptions, symbol, stockPrice, expiration)
	testEngine("CUDA", engineOptions, symbol, stockPrice, expiration)
	testEngine("AUTO", engineOptions, symbol, stockPrice, expiration)
}

func testEngine(mode string, options []barracuda.OptionContract, symbol string, stockPrice float64, expiration string) {
	fmt.Printf("🧪 Testing %s Engine\n", mode)
	fmt.Println("=" + fmt.Sprintf("%*s", len(mode)+15, "="))

	// Create engine with specific mode
	engine := barracuda.NewBaracudaEngineForced(mode)
	if engine == nil {
		fmt.Printf("❌ Failed to create %s engine\n\n", mode)
		return
	}
	defer engine.Close()

	startTime := time.Now()

	// Use appropriate batch calculation function based on engine mode
	var calculatedOptions []barracuda.CompleteOptionResult
	var err error

	if mode == "CPU" {
		calculatedOptions, err = engine.MaximizeCPUUsageComplete(options, stockPrice, 10000.0, "mixed", expiration, nil)
	} else {
		calculatedOptions, err = engine.MaximizeCUDAUsageComplete(options, stockPrice, 10000.0, "mixed", expiration, nil)
	}

	if err != nil {
		fmt.Printf("❌ Error: %v\n\n", err)
		return
	}

	duration := time.Since(startTime)

	if len(calculatedOptions) == 0 {
		fmt.Printf("❌ No calculation results for %s\n\n", symbol)
		return
	}

	// Create a simple result structure for display
	result := struct {
		Symbol                string
		StockPrice            float64
		Expiration            string
		TotalOptionsProcessed int
		ExecutionMode         string
		CalculationTimeMs     float64
	}{
		Symbol:                symbol,
		StockPrice:            stockPrice,
		Expiration:            expiration,
		TotalOptionsProcessed: len(calculatedOptions),
		ExecutionMode:         mode,
		CalculationTimeMs:     duration.Seconds() * 1000,
	}

	fmt.Printf("✅ Success! Processed in %.2fms\n", duration.Seconds()*1000)
	fmt.Printf("🏃 Execution Mode: %s\n", result.ExecutionMode)
	fmt.Printf("📊 Options Processed: %d\n", result.TotalOptionsProcessed)

	// Separate puts and calls from calculated options
	var putsCount, callsCount int
	for _, option := range calculatedOptions {
		if option.OptionType == 'P' {
			putsCount++
		} else {
			callsCount++
		}
	}
	fmt.Printf("🎯 Puts calculated: %d, Calls calculated: %d\n", putsCount, callsCount)

	// Show sample calculations
	if len(calculatedOptions) > 0 {
		fmt.Printf("📊 Sample Results:\n")
		for i, option := range calculatedOptions {
			if i >= 3 {
				break // Show only first 3
			}
			fmt.Printf("   %s: Strike $%.0f, Premium $%.2f, Delta %.3f, Gamma %.3f\n",
				option.Symbol, option.StrikePrice, option.TheoreticalPrice, option.Delta, option.Gamma)
		}
	}

	fmt.Printf("✅ Test completed successfully!\n")

	fmt.Println()
}
