package main

import (
	"fmt"
	"log"

	"github.com/jwaldner/barracuda/internal/alpaca"
)

func main() {
	// Use API keys directly for this test
	apiKey := "AKAH5A3VGCR3S9FNWIVC"
	secretKey := "dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy"

	client := alpaca.NewClient(apiKey, secretKey)

	fmt.Println("🔍 Fetching REAL volume data from Alpaca for AAPL $275 Put...")

	// Get options chain for AAPL expiring 2026-01-16
	optionsChains, err := client.GetOptionsChain([]string{"AAPL"}, "2026-01-16", "")
	if err != nil {
		log.Fatalf("Error fetching options chain: %v", err)
	}

	// Find the $275 put option
	if contracts, exists := optionsChains["AAPL"]; exists {
		for _, contract := range contracts {
			if contract.Type == "put" && contract.StrikePrice == "275" {
				fmt.Printf("📊 Found $275 Put: %s\n", contract.Symbol)

				// Get real-time quote with bid/ask sizes
				quote, err := client.GetOptionQuote(contract.Symbol)
				if err != nil {
					log.Printf("Error getting quote: %v", err)
					continue
				}

				fmt.Printf("💰 Real Market Data:\n")
				fmt.Printf("   Bid: $%.2f (Size: %d)\n", quote.BidPrice, quote.BidSize)
				fmt.Printf("   Ask: $%.2f (Size: %d)\n", quote.AskPrice, quote.AskSize)
				fmt.Printf("   Open Interest: %v\n", contract.OpenInterest)

				// Calculate volume-weighted price
				totalSize := quote.BidSize + quote.AskSize
				if totalSize > 0 {
					bidWeight := float64(quote.BidSize) / float64(totalSize)
					vwapPrice := quote.BidPrice*bidWeight + quote.AskPrice*(1-bidWeight)
					fmt.Printf("   VWAP: $%.3f (%.1f%% bid-weighted)\n", vwapPrice, bidWeight*100)
				}

				break
			}
		}
	} else {
		fmt.Println("❌ No AAPL options found")
	}
}
