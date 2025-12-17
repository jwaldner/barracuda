// Example: How to use the performance monitoring wrapper

package main

import (
	"fmt"

	"github.com/jwaldner/barracuda/internal/alpaca"
)

func main() {
	// Create regular Alpaca client
	client := alpaca.NewClient("your-api-key", "your-secret-key")

	// Wrap it with performance monitoring
	perfClient := alpaca.NewPerformanceWrapper(client)

	// Use it exactly like the regular client
	price, err := perfClient.GetStockPrice("AAPL")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("AAPL price: $%.2f\n", price.Price)
	}

	// Get performance stats anytime
	fmt.Println(perfClient.GetPerformanceStats())

	// At shutdown, performance stats are automatically printed
	perfClient.Close()
}
