package providers

import (
	"context"
	"time"
)

// PerformanceMetrics tracks timing and performance data for provider operations
type PerformanceMetrics struct {
	RequestDuration time.Duration `json:"request_duration"`
	QueueTime       time.Duration `json:"queue_time"`   // Time waiting for rate limiter
	NetworkTime     time.Duration `json:"network_time"` // Actual HTTP request time
	ParseTime       time.Duration `json:"parse_time"`   // JSON parsing time
	CacheHit        bool          `json:"cache_hit"`
	RequestCount    int           `json:"request_count"` // Number of API calls made
	BytesReceived   int64         `json:"bytes_received"`
	RateLimitHit    bool          `json:"rate_limit_hit"` // Did we hit rate limiting?
	RetryAttempts   int           `json:"retry_attempts"`
}

// StockPrice represents a stock price with metadata
type StockPrice struct {
	Symbol    string    `json:"symbol"`
	Price     float64   `json:"price"`
	Timestamp time.Time `json:"timestamp"`
	Volume    int64     `json:"volume,omitempty"`
}

// OptionsContract represents an options contract
type OptionsContract struct {
	ID               string    `json:"id"`
	Symbol           string    `json:"symbol"`
	UnderlyingSymbol string    `json:"underlying_symbol"`
	StrikePrice      float64   `json:"strike_price"`
	ExpirationDate   time.Time `json:"expiration_date"`
	OptionType       string    `json:"option_type"` // "call" or "put"
}

// OptionQuote represents an option quote with bid/ask
type OptionQuote struct {
	ContractID string    `json:"contract_id"`
	Bid        float64   `json:"bid"`
	Ask        float64   `json:"ask"`
	Last       float64   `json:"last"`
	Volume     int64     `json:"volume"`
	Timestamp  time.Time `json:"timestamp"`
}

// PriceResult contains stock prices with performance metrics
type PriceResult struct {
	Data    map[string]*StockPrice `json:"data"`
	Metrics PerformanceMetrics     `json:"metrics"`
}

// ContractsResult contains options contracts with performance metrics
type ContractsResult struct {
	Data    []*OptionsContract `json:"data"`
	Metrics PerformanceMetrics `json:"metrics"`
}

// QuotesResult contains option quotes with performance metrics
type QuotesResult struct {
	Data    map[string]*OptionQuote `json:"data"`
	Metrics PerformanceMetrics      `json:"metrics"`
}

// MarketProvider defines the interface for market data providers
type MarketProvider interface {
	// GetStockPrices fetches current stock prices for given symbols
	GetStockPrices(ctx context.Context, symbols []string) (*PriceResult, error)

	// GetOptionsContracts fetches available options contracts for a symbol
	GetOptionsContracts(ctx context.Context, symbol string, expiration time.Time) (*ContractsResult, error)

	// GetOptionQuotes fetches current quotes for specific option contracts
	GetOptionQuotes(ctx context.Context, contractIDs []string) (*QuotesResult, error)

	// GetProviderName returns the name of the provider (e.g., "alpaca", "polygon")
	GetProviderName() string

	// GetPerformanceStats returns cumulative performance statistics
	GetPerformanceStats() PerformanceMetrics

	// Close cleans up any resources (connections, rate limiters, etc.)
	Close() error
}
