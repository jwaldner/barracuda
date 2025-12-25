package main

import (
	"fmt"
	"time"

	"github.com/jwaldner/barracuda/internal/audit"
	"github.com/jwaldner/barracuda/internal/logger"
)

func main() {
	fmt.Println("Testing audit system...")

	// Initialize logger
	err := logger.InitWithLevel("debug")
	if err != nil {
		fmt.Printf("Failed to initialize logger: %v\n", err)
		return
	}

	// Create audit logger
	auditor := audit.NewOptionsAnalysisAuditLogger()

	// Test 1: Create first audit for AAPL
	fmt.Println("\n=== Test 1: Create AAPL audit ===")
	err = auditor.LogOptionsAnalysisOperation("AAPL", "create", map[string]interface{}{
		"expiration": "2025-01-17",
		"test":       "first audit",
	})
	if err != nil {
		fmt.Printf("Error creating AAPL audit: %v\n", err)
	}

	// Add some entries to AAPL
	fmt.Println("Adding entries to AAPL audit...")
	err = auditor.LogOptionsAnalysisOperation("AAPL", "option_data", map[string]interface{}{
		"strike": 200.0,
		"type":   "call",
		"price":  10.50,
	})
	if err != nil {
		fmt.Printf("Error adding AAPL entry: %v\n", err)
	}

	err = auditor.LogOptionsAnalysisOperation("AAPL", "calculation", map[string]interface{}{
		"delta": -0.42, // Realistic put delta
		"gamma": 0.012, // Realistic gamma
	})
	if err != nil {
		fmt.Printf("Error adding AAPL calculation: %v\n", err)
	}

	// Wait a moment for processing
	time.Sleep(100 * time.Millisecond)

	// Test 2: Create second audit for TSLA (should archive AAPL first)
	fmt.Println("\n=== Test 2: Create TSLA audit (should archive AAPL) ===")
	err = auditor.LogOptionsAnalysisOperation("TSLA", "create", map[string]interface{}{
		"expiration": "2025-01-24",
		"test":       "second audit",
	})
	if err != nil {
		fmt.Printf("Error creating TSLA audit: %v\n", err)
	}

	// Add entries to TSLA
	fmt.Println("Adding entries to TSLA audit...")
	err = auditor.LogOptionsAnalysisOperation("TSLA", "option_data", map[string]interface{}{
		"strike": 350.0,
		"type":   "put",
		"price":  25.75,
	})
	if err != nil {
		fmt.Printf("Error adding TSLA entry: %v\n", err)
	}

	// Wait for processing
	time.Sleep(100 * time.Millisecond)

	fmt.Println("\n=== Test Complete ===")
	fmt.Println("Check:")
	fmt.Println("1. audit.json should exist with TSLA data")
	fmt.Println("2. audits/ directory should have AAPL_2025-01-17_*.json file")
	fmt.Println("3. AAPL should have 2 entries, TSLA should have 1 entry")
}
