package main

import (
	"fmt"
	"log"
	"math"

	"github.com/jwaldner/barracuda/barracuda_lib"
)

// Test the improved CDF accuracy with Grok's expected values
func main() {
	fmt.Println("🎯 Testing CDF Accuracy Against Grok Analysis")
	fmt.Println("===========================================")

	// Grok's inputs from the analysis
	S := 188.36    // stock price
	K := 166.0     // strike price
	T := 0.057534246575342465 // time to expiration
	r := 0.03983   // risk-free rate
	sigma := 0.39963570444400937 // volatility

	// Grok's expected values
	expected_price := 0.706300
	expected_delta := -0.082000

	fmt.Printf("📊 Input Parameters:\n")
	fmt.Printf("   Stock Price (S): $%.2f\n", S)
	fmt.Printf("   Strike Price (K): $%.0f\n", K)
	fmt.Printf("   Time to Exp (T): %.6f years\n", T)
	fmt.Printf("   Risk-free Rate (r): %.5f\n", r)
	fmt.Printf("   Volatility (σ): %.6f\n", sigma)
	fmt.Println()

	// Create a test contract
	engine := barracuda.NewBaracudaEngine()
	defer engine.Close()
	
	contract := barracuda.OptionContract{
		Symbol:           "NVDA",
		StrikePrice:      K,
		UnderlyingPrice:  S,
		TimeToExpiration: T,
		RiskFreeRate:     r,
		Volatility:       sigma,
		OptionType:       'P', // Put option
		MarketClosePrice: 0.0, // Not needed for this test
	}

	// Calculate using our improved implementation
	contracts := []barracuda.OptionContract{contract}
	
	// Try CUDA first, then CPU fallback
	puts, calls, err := engine.MaximizeCUDAUsage(contracts, S)
	if err != nil {
		log.Fatalf("❌ CUDA calculation failed: %v", err)
	}

	var result barracuda.OptionContract
	var found bool
	
	// Check puts (since we're testing a put option)
	if len(puts) > 0 {
		result = puts[0]
		found = true
	} else if len(calls) > 0 {
		result = calls[0]
		found = true
	}
	
	if !found {
		// Try CPU fallback using complete function
		fmt.Println("⚠️  CUDA returned no results, trying CPU fallback...")
		complete_results, err := engine.MaximizeCPUUsageComplete(contracts, S, 10000.0, "puts", "2026-01-16", nil)
		if err != nil || len(complete_results) == 0 {
			log.Fatalf("❌ Both CUDA and CPU calculations failed")
		}
		
		// Convert CompleteOptionResult to OptionContract for display
		cr := complete_results[0]
		result = barracuda.OptionContract{
			Symbol:           cr.Symbol,
			StrikePrice:      cr.StrikePrice,
			UnderlyingPrice:  cr.UnderlyingPrice,
			OptionType:       cr.OptionType,
			TheoreticalPrice: cr.TheoreticalPrice,
			Delta:            cr.Delta,
			Gamma:            cr.Gamma,
			Theta:            cr.Theta,
			Vega:             cr.Vega,
			Rho:              cr.Rho,
		}
	}

	fmt.Printf("🔬 Calculation Results:\n")
	fmt.Printf("   Theoretical Price: $%.6f (Expected: $%.6f)\n", result.TheoreticalPrice, expected_price)
	fmt.Printf("   Delta: %.6f (Expected: %.6f)\n", result.Delta, expected_delta)
	fmt.Printf("   Gamma: %.6f\n", result.Gamma)
	fmt.Printf("   Theta: %.6f\n", result.Theta)
	fmt.Printf("   Vega: %.6f\n", result.Vega)
	fmt.Printf("   Rho: %.6f\n", result.Rho)
	fmt.Println()

	// Calculate differences
	price_diff := math.Abs(result.TheoreticalPrice - expected_price)
	delta_diff := math.Abs(result.Delta - expected_delta)

	fmt.Printf("📈 Accuracy Analysis:\n")
	fmt.Printf("   Price Difference: $%.6f (%.4f%%)\n", price_diff, (price_diff/expected_price)*100)
	fmt.Printf("   Delta Difference: %.6f (%.4f%%)\n", delta_diff, (delta_diff/math.Abs(expected_delta))*100)
	fmt.Println()

	// Assessment
	price_tolerance := 0.001  // ±0.001 as mentioned by user
	delta_tolerance := 0.001

	price_accurate := price_diff <= price_tolerance
	delta_accurate := delta_diff <= delta_tolerance

	if price_accurate && delta_accurate {
		fmt.Printf("✅ ACCURACY IMPROVED: Both values within ±%.3f tolerance\n", price_tolerance)
	} else {
		fmt.Printf("⚠️  NEEDS FURTHER IMPROVEMENT:\n")
		if !price_accurate {
			fmt.Printf("   - Price difference %.6f exceeds tolerance %.3f\n", price_diff, price_tolerance)
		}
		if !delta_accurate {
			fmt.Printf("   - Delta difference %.6f exceeds tolerance %.3f\n", delta_diff, delta_tolerance)
		}
	}

	// Show improvement from original error
	original_error := 0.014  // User mentioned 0.014 difference
	improvement := ((original_error - price_diff) / original_error) * 100

	if price_diff < original_error {
		fmt.Printf("🎯 IMPROVEMENT: Reduced error by %.1f%% (from $%.3f to $%.6f)\n", 
			improvement, original_error, price_diff)
	}
}