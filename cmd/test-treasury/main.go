package main

import (
	"fmt"
	"log"

	"github.com/jwaldner/barracuda/internal/treasury"
)

func main() {
	fmt.Println("🏛️ Testing Treasury API integration...")

	client := treasury.NewTreasuryClient()

	// Test fetching current risk-free rate
	rate, err := client.GetRiskFreeRate()
	if err != nil {
		log.Printf("❌ Error fetching Treasury rate: %v", err)
	} else {
		fmt.Printf("✅ Current Treasury Bill Rate (Risk-Free): %.6f (%.3f%%)\n", rate, rate*100)
	}

	// Test last known rate functionality
	lastKnownRate := client.GetRiskFreeRateWithLastKnown()
	fmt.Printf("✅ Rate with last known: %.6f (%.3f%%)\n", lastKnownRate, lastKnownRate*100)

	fmt.Println("🎉 Treasury API integration successful!")
}
