package main

import (
	"fmt"
	"time"

	"github.com/jwaldner/barracuda/internal/audit"
	"github.com/jwaldner/barracuda/internal/logger"
)

func main() {
	fmt.Println("Testing Grok handler integration...")

	// Initialize logger
	err := logger.InitWithLevel("debug")
	if err != nil {
		fmt.Printf("Failed to initialize logger: %v\n", err)
		return
	}

	// Create audit logger
	auditor := audit.NewOptionsAnalysisAuditLogger()

	// Simulate the full workflow:
	// 1. Create audit for IBM
	fmt.Println("=== Step 1: Create IBM audit ===")
	err = auditor.LogOptionsAnalysisOperation("IBM", "create", map[string]interface{}{
		"expiration": "2025-03-21",
		"strategy":   "covered_call",
	})
	if err != nil {
		fmt.Printf("Error creating IBM audit: %v\n", err)
		return
	}

	// 2. Add some option data
	fmt.Println("=== Step 2: Add option entries ===")
	err = auditor.LogOptionsAnalysisOperation("IBM", "option_analysis", map[string]interface{}{
		"strike":            180.0,
		"theoretical_price": 8.75,
		"delta":             -0.45, // Realistic put delta
		"gamma":             0.009, // Realistic gamma
		"theta":             -0.15,
		"vega":              0.22,
	})
	if err != nil {
		fmt.Printf("Error adding option data: %v\n", err)
		return
	}

	err = auditor.LogOptionsAnalysisOperation("IBM", "market_data", map[string]interface{}{
		"stock_price":    175.50,
		"implied_vol":    0.28,
		"risk_free_rate": 0.045,
		"time_to_expiry": 0.245,
	})
	if err != nil {
		fmt.Printf("Error adding market data: %v\n", err)
		return
	}

	// Wait for processing
	time.Sleep(100 * time.Millisecond)

	// 3. Simulate Grok handler call (what the fixed handler does)
	fmt.Println("=== Step 3: Grok analysis (simulating handler) ===")
	grokAnalysis := `# IBM Options Analysis - March 21, 2025

## Executive Summary
Analysis of IBM covered call strategy for March 21, 2025 expiration.

## Market Data Review
- **Current Stock Price**: $175.50
- **Strike Price**: $180.00
- **Implied Volatility**: 28%
- **Time to Expiry**: 89 days (0.245 years)

## Greeks Analysis
- **Delta**: -0.45 (moderate put directional exposure)
- **Gamma**: 0.009 (low convexity risk)
- **Theta**: -0.15 (moderate time decay)
- **Vega**: 0.22 (moderate volatility risk)

## Theoretical Pricing
- **Theoretical Value**: $8.75
- **Assessment**: Option appears fairly valued

## Risk Assessment
The covered call shows balanced risk/reward with moderate Greeks exposure.

## Recommendation
Position sizing appropriate given current market conditions.
`

	// This simulates what the Grok handler now does with the fix
	err = auditor.LogOptionsAnalysisOperation("", "analysis_result", map[string]interface{}{
		"grok_result":   grokAnalysis,
		"custom_prompt": "Analyze IBM options for covered call strategy",
		"tokens":        1250,
		"elapsed":       2.35,
	})
	if err != nil {
		fmt.Printf("Error sending Grok analysis: %v\n", err)
		return
	}

	// Wait for processing
	time.Sleep(100 * time.Millisecond)

	fmt.Println("\n=== Test Complete ===")
	fmt.Println("Check:")
	fmt.Println("1. audit.json should be gone (moved to audits/)")
	fmt.Println("2. audits/ should have IBM-2025-03-21.json with 2 entries")
	fmt.Println("3. audits/ should have IBM-2025-03-21.md with Grok analysis")
}
