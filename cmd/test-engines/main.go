package main

import (
	"fmt"
	"math"
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

	// Use batch processing function
	symbols := []string{symbol}
	stockPrices := map[string]float64{symbol: stockPrice}
	optionsChains := map[string][]barracuda.OptionContract{symbol: options}

	results, err := engine.AnalyzeSymbolsBatch(symbols, stockPrices, optionsChains, expiration)
	if err != nil {
		fmt.Printf("❌ Error: %v\n\n", err)
		return
	}

	duration := time.Since(startTime)

	if len(results) == 0 {
		fmt.Printf("❌ No results returned\n\n")
		return
	}

	result := results[0]

	fmt.Printf("✅ Success! Processed in %.2fms\n", duration.Seconds()*1000)
	fmt.Printf("🏃 Execution Mode: %s\n", result.ExecutionMode)
	fmt.Printf("📊 Options Processed: %d\n", result.TotalOptionsProcessed)
	fmt.Printf("🎯 Puts with IV: %d, Calls with IV: %d\n", len(result.PutsWithIV), len(result.CallsWithIV))

	// Display 25-delta skew results
	skew := result.VolatilitySkew
	if skew.Symbol != "" {
		fmt.Printf("📈 25Δ Skew Analysis:\n")
		fmt.Printf("   • Put 25Δ IV: %.1f%%\n", skew.Put25DIV*100)
		fmt.Printf("   • Call 25Δ IV: %.1f%%\n", skew.Call25DIV*100)
		fmt.Printf("   • Skew: %.1f vol points\n", (skew.Put25DIV-skew.Call25DIV)*100)
		fmt.Printf("   • ATM IV: %.1f%%\n", skew.ATMIV*100)

		// Validate against expected results
		expected := testdata.Expected25DeltaResults
		skewPoints := (skew.Put25DIV - skew.Call25DIV) * 100

		if skewPoints >= expected.ExpectedSkewRange[0] && skewPoints <= expected.ExpectedSkewRange[1] {
			fmt.Printf("   ✅ Skew in expected range (%.1f-%.1f)\n", expected.ExpectedSkewRange[0], expected.ExpectedSkewRange[1])
		} else {
			fmt.Printf("   ⚠️  Skew outside expected range (%.1f-%.1f)\n", expected.ExpectedSkewRange[0], expected.ExpectedSkewRange[1])
		}

		if skew.Put25DIV > skew.Call25DIV {
			fmt.Printf("   ✅ Negative skew confirmed (puts > calls)\n")
		} else {
			fmt.Printf("   ⚠️  Unexpected skew direction\n")
		}
	} else {
		fmt.Printf("⚠️  No 25Δ skew data calculated\n")
	}

	// Show premiums for all 3 risk levels (only for first test)
	if mode == "CPU" {
		fmt.Printf("📊 Risk Level Analysis - Option Premiums:\n")

		riskLevels := []struct {
			name  string
			delta float64
			risk  string
		}{
			{"10-Delta", 0.10, "Conservative (10% ITM probability)"},
			{"25-Delta", 0.25, "Moderate (25% ITM probability)"},
			{"50-Delta", 0.50, "Aggressive (50% ITM probability)"},
		}

		for _, level := range riskLevels {
			var closestOption *barracuda.OptionContract
			minDeltaDiff := 1.0

			// Search ONLY puts for target negative deltas (-0.1, -0.25, -0.5)
			targetDelta := -level.delta // Convert to negative for puts
			for i := range result.PutsWithIV {
				put := &result.PutsWithIV[i]
				deltaDiff := math.Abs(put.Delta - targetDelta)
				if deltaDiff < minDeltaDiff {
					minDeltaDiff = deltaDiff
					closestOption = put
				}
			}

			if closestOption != nil {
				totalPremium := closestOption.TheoreticalPrice * 100
				// Display absolute delta value for frontend simplicity
				displayDelta := math.Abs(closestOption.Delta)
				fmt.Printf("🎯 %s Put: Strike $%.0f, Delta: %.3f (Real: %.3f)\n",
					level.name, closestOption.StrikePrice, displayDelta, closestOption.Delta)
				fmt.Printf("   💰 Premium: $%.2f/share | $%.0f/contract\n",
					closestOption.TheoreticalPrice, totalPremium)
				fmt.Printf("   📊 IV: %.2f%%\n", closestOption.Volatility*100)

				// Compare our calculation vs expected values for 50-delta ($275 strike)
				if level.name == "50-Delta" && closestOption.StrikePrice == 275 {
					fmt.Printf("   🔬 EXPECTED vs ACTUAL COMPARISON ($275 Put):\n")

					// Real ALPACA API data for Jan 16, 2026 expiration (31 days out)
					// Stock at $272.23, $275 Put slightly ITM with 31 days to expiration
					mockBid, mockAsk := 6.10, 6.30           // REAL API pricing with 31 days remaining
					mockBidSize, mockAskSize := 180.0, 140.0 // Real volume from API
					mockMid := (mockBid + mockAsk) / 2

					// Calculate volume-weighted price from mock data
					totalSize := mockBidSize + mockAskSize
					bidRatio := mockBidSize / totalSize
					mockVWAP := mockBid + (mockAsk-mockBid)*bidRatio

					// Platform comparison table
					fmt.Printf("\n     📊 REAL ALPACA API DATA (Jan 16, 2026 - 31 Days Out):\n")
					fmt.Printf("     ┌─────────────────────────────────────────┐\n")
					fmt.Printf("     │ REAL Market Data (31 Days to Exp):     │\n")
					fmt.Printf("     │   Stock: $272.23 | Strike: $275         │\n")
					fmt.Printf("     │   Time: 31 days to expiration           │\n")
					fmt.Printf("     │                                         │\n")
					fmt.Printf("     │ REAL Alpaca API Pricing:                │\n")
					fmt.Printf("     │   Bid: $%.2f (%d) | Ask: $%.2f (%d)    │\n", mockBid, int(mockBidSize), mockAsk, int(mockAskSize))
					fmt.Printf("     │   Mid: $%.2f | VWAP: $%.3f              │\n", mockMid, mockVWAP)
					fmt.Printf("     │   Volume: %.0f%% bid-weighted            │\n", bidRatio*100)
					fmt.Printf("     │                                         │\n")
					fmt.Printf("     │ Our Calculation vs EXPECTED:           │\n")
					fmt.Printf("     │   Our Premium: $%.2f | Delta: %.4f      │\n", closestOption.TheoreticalPrice, closestOption.Delta)
					fmt.Printf("     │   Expected: $6.20 | Delta: -0.5000      │\n")
					fmt.Printf("     │   Accuracy: %.2f%% (%.2f¢ diff)          │\n", (1.0-math.Abs(closestOption.TheoreticalPrice-6.20)/6.20)*100, math.Abs(closestOption.TheoreticalPrice-6.20)*100)
					fmt.Printf("     └─────────────────────────────────────────┘\n")

					// DETAILED CALCULATION BREAKDOWN
					fmt.Printf("\n     📋 OUR CALCULATION DETAILS:\n")
					fmt.Printf("     ┌─────────────────────────────────────────┐\n")
					fmt.Printf("     │ Black-Scholes Parameters:               │\n")
					fmt.Printf("     │   Stock Price (S):     $%.3f            │\n", closestOption.UnderlyingPrice)
					fmt.Printf("     │   Strike Price (K):    $%.3f            │\n", closestOption.StrikePrice)
					fmt.Printf("     │   Time to Exp (T):     %.4f years      │\n", closestOption.TimeToExpiration)
					fmt.Printf("     │   Risk-Free Rate (r):  %.2f%%            │\n", closestOption.RiskFreeRate*100)
					fmt.Printf("     │   Volatility (σ):      %.2f%%            │\n", closestOption.Volatility*100)
					fmt.Printf("     │                                         │\n")

					// Calculate d1 and d2 for detailed breakdown
					S := closestOption.UnderlyingPrice
					K := closestOption.StrikePrice
					T := closestOption.TimeToExpiration
					r := closestOption.RiskFreeRate
					sigma := closestOption.Volatility

					d1 := (math.Log(S/K) + (r+0.5*sigma*sigma)*T) / (sigma * math.Sqrt(T))
					d2 := d1 - sigma*math.Sqrt(T)

					fmt.Printf("     │ Intermediate Calculations:              │\n")
					fmt.Printf("     │   d1 = ln(S/K) + (r+σ²/2)T / σ√T       │\n")
					fmt.Printf("     │   d1 = %.6f                            │\n", d1)
					fmt.Printf("     │   d2 = d1 - σ√T = %.6f                 │\n", d2)
					fmt.Printf("     │                                         │\n")
					fmt.Printf("     │ Put Option Formula:                     │\n")
					fmt.Printf("     │   Put = K*e^(-rT)*N(-d2) - S*N(-d1)     │\n")
					fmt.Printf("     │   Put = $%.3f                           │\n", closestOption.TheoreticalPrice)
					fmt.Printf("     │                                         │\n")
					fmt.Printf("     │ Greeks Calculated:                      │\n")
					fmt.Printf("     │   Delta:  %.4f (price sensitivity)     │\n", closestOption.Delta)
					fmt.Printf("     │   Gamma:  %.4f (delta sensitivity)     │\n", closestOption.Gamma)
					fmt.Printf("     │   Theta:  %.4f (time decay/day)        │\n", closestOption.Theta)
					fmt.Printf("     │   Vega:   %.4f (vol sensitivity)       │\n", closestOption.Vega)
					fmt.Printf("     │   Rho:    %.4f (rate sensitivity)      │\n", closestOption.Rho)
					fmt.Printf("     └─────────────────────────────────────────┘\n")

					// Analysis table
					testMethods := []struct {
						name  string
						price float64
					}{
						{"Mock Mid Price (Near Exp)", mockMid},
						{"Mock VWAP (Near Exp)", mockVWAP},
						{"Our Theoretical", closestOption.TheoreticalPrice},
						{"Intrinsic Value", math.Max(275.0-274.50, 0)}, // $275 Put with stock at $274.50
					}

					for _, method := range testMethods {
						fmt.Printf("     %-25s | Price: $%.3f\n", method.name, method.price)
					}

					fmt.Printf("   🎯 Target: Expected IV = 19.35%%\n")
				}

				fmt.Printf("   📈 Risk: %s\n", level.risk)
			} else {
				fmt.Printf("❌ No %s option found\n", level.name)
			}
		}
	}

	fmt.Println()
}
