package main

import (
	"fmt"
	"time"

	"github.com/jwaldner/barracuda/internal/audit"
	"github.com/jwaldner/barracuda/internal/logger"
)

func main() {
	fmt.Println("Testing analysis_result action...")

	// Initialize logger
	err := logger.InitWithLevel("debug")
	if err != nil {
		fmt.Printf("Failed to initialize logger: %v\n", err)
		return
	}

	// Create audit logger
	auditor := audit.NewOptionsAnalysisAuditLogger()

	// Create an audit for MSFT
	fmt.Println("=== Creating MSFT audit ===")
	err = auditor.LogOptionsAnalysisOperation("MSFT", "create", map[string]interface{}{
		"expiration": "2025-02-21",
		"test":       "analysis test",
	})
	if err != nil {
		fmt.Printf("Error creating MSFT audit: %v\n", err)
		return
	}

	// Add some entries
	err = auditor.LogOptionsAnalysisOperation("MSFT", "calculation", map[string]interface{}{
		"strike":            400.0,
		"theoretical_price": 15.25,
		"delta":             -0.35, // Realistic put delta
		"gamma":             0.008, // Realistic gamma
	})
	if err != nil {
		fmt.Printf("Error adding entry: %v\n", err)
		return
	}

	// Wait for processing
	time.Sleep(100 * time.Millisecond)

	// Test analysis_result action
	fmt.Println("=== Testing analysis_result ===")
	grokAnalysis := `# Options Analysis - MSFT

## Summary
This analysis covers the MSFT option chain for February 21, 2025.

## Calculations
- Strike: $400
- Theoretical Price: $15.25
- Delta: -0.35
- Gamma: 0.008

## Conclusion
The calculations appear consistent with market expectations.
`

	err = auditor.LogOptionsAnalysisOperation("", "analysis_result", map[string]interface{}{
		"grok_result": grokAnalysis,
	})
	if err != nil {
		fmt.Printf("Error creating analysis: %v\n", err)
		return
	}

	// Wait for processing
	time.Sleep(100 * time.Millisecond)

	fmt.Println("\n=== Test Complete ===")
	fmt.Println("Check:")
	fmt.Println("1. audit.json should be gone (moved to audits/)")
	fmt.Println("2. audits/ should have MSFT-2025-02-21.json")
	fmt.Println("3. audits/ should have MSFT-2025-02-21.md with grok analysis")
}
