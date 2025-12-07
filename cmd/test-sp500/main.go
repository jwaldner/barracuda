package main

import (
	"log"
	"time"

	"github.com/jwaldner/barracuda/internal/symbols"
)

func main() {
	log.Printf("🧪 Testing S&P 500 Symbol Service")

	// Create service
	service := symbols.NewSP500Service("test_data")

	// Test update
	log.Printf("📡 Fetching S&P 500 symbols...")
	start := time.Now()

	err := service.UpdateSymbols()
	if err != nil {
		log.Printf("❌ Update failed: %v", err)
		return
	}

	duration := time.Since(start)
	log.Printf("✅ Update completed in %v", duration)

	// Test loading
	symbols, err := service.LoadSymbols()
	if err != nil {
		log.Printf("❌ Load failed: %v", err)
		return
	}

	log.Printf("📊 Loaded %d S&P 500 symbols", len(symbols))

	// Show first 10 symbols
	log.Printf("📈 First 10 symbols:")
	for i, symbol := range symbols {
		if i >= 10 {
			break
		}
		log.Printf("   %s - %s", symbol.Symbol, symbol.Company)
	}

	// Test string list
	tickers, err := service.GetSymbolsAsStrings()
	if err != nil {
		log.Printf("❌ String list failed: %v", err)
		return
	}

	log.Printf("🎯 Got %d ticker symbols: %v...", len(tickers), tickers[:5])

	// Test info
	info, err := service.GetSymbolsInfo()
	if err != nil {
		log.Printf("❌ Info failed: %v", err)
		return
	}

	log.Printf("ℹ️ Info: %+v", info)

	log.Printf("🎉 All tests passed!")
}
