package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"path/filepath"
)

// AuditData represents the structure of audit JSON files
type AuditData struct {
	Header struct {
		Ticker     string `json:"ticker"`
		ExpiryDate string `json:"expiry_date"`
		StartTime  string `json:"start_time"`
	} `json:"header"`
	Entries []struct {
		Data map[string]interface{} `json:"data"`
	} `json:"entries"`
}

// ValidationIssue represents a data consistency issue
type ValidationIssue struct {
	Type        string
	Description string
	Severity    string
	Contract    string
	Expected    interface{}
	Actual      interface{}
}

func main() {
	fmt.Println("🔍 Audit Data Validation Tool")
	fmt.Println("=============================")

	// Find all audit JSON files
	auditFiles, err := filepath.Glob("/home/joe/projects/go-cuda/audits/*.json")
	if err != nil {
		log.Fatalf("Error finding audit files: %v", err)
	}

	fmt.Printf("Found %d audit files to validate\n\n", len(auditFiles))

	for _, file := range auditFiles {
		fmt.Printf("📋 Analyzing: %s\n", filepath.Base(file))
		issues := validateAuditFile(file)

		if len(issues) == 0 {
			fmt.Println("✅ No issues found")
		} else {
			fmt.Printf("⚠️  Found %d issues:\n", len(issues))
			for _, issue := range issues {
				fmt.Printf("  • %s: %s\n", issue.Type, issue.Description)
				if issue.Expected != nil && issue.Actual != nil {
					fmt.Printf("    Expected: %v, Got: %v\n", issue.Expected, issue.Actual)
				}
			}
		}
		fmt.Println()
	}
}

func validateAuditFile(filename string) []ValidationIssue {
	var issues []ValidationIssue

	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return []ValidationIssue{{
			Type:        "FILE_ERROR",
			Description: fmt.Sprintf("Cannot read file: %v", err),
			Severity:    "HIGH",
		}}
	}

	var audit AuditData
	if err := json.Unmarshal(data, &audit); err != nil {
		return []ValidationIssue{{
			Type:        "JSON_ERROR",
			Description: fmt.Sprintf("Invalid JSON: %v", err),
			Severity:    "HIGH",
		}}
	}

	// Extract stock prices and Black-Scholes data
	var stockPrices []float64
	var bsCalculations []map[string]interface{}

	for _, entry := range audit.Entries {
		// Stock price from batch data
		if price, ok := entry.Data["price"].(float64); ok {
			stockPrices = append(stockPrices, price)
		}

		// Black-Scholes calculation data
		if calcDetails, ok := entry.Data["calculation_details"].(map[string]interface{}); ok {
			bsCalculations = append(bsCalculations, calcDetails)
		}
	}

	// Validation 1: Stock price consistency
	if len(stockPrices) > 1 {
		basePrice := stockPrices[0]
		for _, price := range stockPrices[1:] {
			if math.Abs(price-basePrice) > basePrice*0.01 { // More than 1% difference
				issues = append(issues, ValidationIssue{
					Type:        "STOCK_PRICE_INCONSISTENCY",
					Description: fmt.Sprintf("Stock price varies significantly across entries"),
					Severity:    "MEDIUM",
					Expected:    basePrice,
					Actual:      price,
				})
			}
		}
	}

	// Validation 2: Black-Scholes data consistency
	for i, calc := range bsCalculations {
		if contractData, ok := calc[audit.Header.Ticker+"_contract"].(map[string]interface{}); ok {
			if variables, ok := contractData["variables"].(map[string]interface{}); ok {
				// Check for zero volatility
				if sigma, ok := variables["sigma"].(float64); ok && sigma == 0 {
					issues = append(issues, ValidationIssue{
						Type:        "ZERO_VOLATILITY",
						Description: "Volatility is zero, causing calculation errors",
						Severity:    "HIGH",
						Contract:    fmt.Sprintf("Contract_%d", i+1),
					})
				}

				// Check stock price vs header ticker
				if stockPrice, ok := variables["S"].(float64); ok {
					if len(stockPrices) > 0 && math.Abs(stockPrice-stockPrices[0]) > stockPrices[0]*0.05 {
						issues = append(issues, ValidationIssue{
							Type:        "STOCK_PRICE_MISMATCH",
							Description: "Stock price in BS calculation doesn't match batch data",
							Severity:    "HIGH",
							Contract:    fmt.Sprintf("Contract_%d", i+1),
							Expected:    stockPrices[0],
							Actual:      stockPrice,
						})
					}
				}

				// Check for unrealistic option prices
				if results, ok := contractData["results"].(map[string]interface{}); ok {
					if price, ok := results["theoretical_price"].(float64); ok {
						if price < 0.001 {
							issues = append(issues, ValidationIssue{
								Type:        "UNREALISTIC_PRICE",
								Description: "Theoretical price extremely small",
								Severity:    "MEDIUM",
								Contract:    fmt.Sprintf("Contract_%d", i+1),
								Actual:      price,
							})
						}
					}
				}
			}
		}
	}

	// Validation 3: Ranking data consistency
	for _, entry := range audit.Entries {
		if rankings, ok := entry.Data["rankings"].([]interface{}); ok {
			for _, rankingData := range rankings {
				if ranking, ok := rankingData.(map[string]interface{}); ok {
					if premium, ok := ranking["premium"].(float64); ok {
						if premium < 0.01 && premium > 0 {
							symbol, _ := ranking["symbol"].(string)
							issues = append(issues, ValidationIssue{
								Type:        "TINY_PREMIUM",
								Description: fmt.Sprintf("Extremely small premium for %s", symbol),
								Severity:    "MEDIUM",
								Actual:      premium,
							})
						}
					}
				}
			}
		}
	}

	return issues
}
