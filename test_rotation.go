package main

import (
	"fmt"
	"os"
	"time"

	"github.com/jwaldner/barracuda/internal/audit"
)

func main() {
	fmt.Println("Testing new ticker+expiry rotation system...")

	// Create audit logger
	auditor := audit.NewOptionsAnalysisAuditLogger()

	// Send TSLA request with expiry - should archive old audit.json as GLD-expiry.json and create current.json
	fmt.Println("Sending TSLA-2025-01-17 audit request...")
	err := auditor.LogOptionsAnalysisOperation("TSLA", "GetStockPrice", map[string]interface{}{
		"symbol":     "TSLA",
		"price":      250.75,
		"expiration": "2025-01-17",
		"test":       "rotation-with-expiry",
	})

	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	// Give it a moment for the channel to process
	time.Sleep(100 * time.Millisecond)

	fmt.Println("First request sent!")

	// Send another entry for same ticker+expiry - should just append
	fmt.Println("Sending another TSLA-2025-01-17 entry...")
	err = auditor.LogOptionsAnalysisOperation("TSLA", "CalculateBlackScholes", map[string]interface{}{
		"strike":     200.0,
		"expiration": "2025-01-17",
		"iv":         0.25,
	})

	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	time.Sleep(100 * time.Millisecond)

	// Send different ticker+expiry - should rotate again
	fmt.Println("Sending AAPL-2025-02-21 entry...")
	err = auditor.LogOptionsAnalysisOperation("AAPL", "GetOptionsChain", map[string]interface{}{
		"symbol":     "AAPL",
		"expiration": "2025-02-21",
		"strikes":    []float64{150, 155, 160},
	})

	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	time.Sleep(100 * time.Millisecond)

	fmt.Println("All requests sent! Check:")
	fmt.Println("- current.json (should be AAPL-2025-02-21)")
	fmt.Println("- TSLA-2025-01-17.json (archived)")
	fmt.Println("- Any old audit files from previous system")
}
