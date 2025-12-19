package alpaca

import (
	"fmt"
	"time"

	"github.com/jwaldner/barracuda/internal/logger"
)

// PerformanceWrapper wraps the Alpaca client with performance monitoring
type PerformanceWrapper struct {
	client           *Client
	totalRequests    int64
	totalDuration    time.Duration
	totalQueueTime   time.Duration
	slowRequestCount int64
}

// NewPerformanceWrapper creates a wrapper around an Alpaca client
func NewPerformanceWrapper(client *Client) *PerformanceWrapper {
	return &PerformanceWrapper{
		client: client,
	}
}

// GetStockPrice wraps the original method with performance monitoring
func (pw *PerformanceWrapper) GetStockPrice(symbol string, auditTicker *string) (*StockPrice, error) {
	start := time.Now()
	result, err := pw.client.GetStockPrice(symbol, auditTicker)
	duration := time.Since(start)

	pw.recordRequest(duration)

	logger.Debug.Printf("📡 API CALL: GetStockPrice(%s) took %v", symbol, duration)
	if duration > 2*time.Second {
		logger.Debug.Printf("⚠️  SLOW API CALL: GetStockPrice(%s) took %v", symbol, duration)
	}

	return result, err
}

// GetStockPricesBatch wraps the original method with performance monitoring
func (pw *PerformanceWrapper) GetStockPricesBatch(symbols []string, auditTicker *string) (map[string]*StockPrice, error) {
	start := time.Now()
	result, err := pw.client.GetStockPricesBatch(symbols, auditTicker)
	duration := time.Since(start)

	pw.recordRequest(duration)

	logger.Debug.Printf("📡 API CALL: GetStockPricesBatch(%d symbols) took %v", len(symbols), duration)
	if duration > 5*time.Second {
		logger.Debug.Printf("⚠️  SLOW API CALL: GetStockPricesBatch(%d symbols) took %v", len(symbols), duration)
	}

	return result, err
}

// GetOptionsChain wraps the original method with performance monitoring
func (pw *PerformanceWrapper) GetOptionsChain(symbols []string, expirationDate, strategy string, auditTicker *string) (map[string][]*OptionContract, error) {
	start := time.Now()
	result, err := pw.client.GetOptionsChain(symbols, expirationDate, strategy, auditTicker)
	duration := time.Since(start)

	pw.recordRequest(duration)

	logger.Debug.Printf("📡 API CALL: GetOptionsChain(%d symbols, %s) took %v", len(symbols), expirationDate, duration)
	if duration > 10*time.Second {
		logger.Debug.Printf("⚠️  SLOW API CALL: GetOptionsChain(%d symbols, %s) took %v", len(symbols), expirationDate, duration)
	}

	return result, err
}

// GetOptionQuote wraps the original method with performance monitoring
func (pw *PerformanceWrapper) GetOptionQuote(symbol string, auditTicker *string) (*OptionQuote, error) {
	start := time.Now()
	result, err := pw.client.GetOptionQuote(symbol, auditTicker)
	duration := time.Since(start)

	pw.recordRequest(duration)

	logger.Debug.Printf("📡 API CALL: GetOptionQuote(%s) took %v", symbol, duration)
	if duration > 2*time.Second {
		logger.Debug.Printf("⚠️  SLOW API CALL: GetOptionQuote(%s) took %v", symbol, duration)
	}

	return result, err
}

// recordRequest updates performance statistics
func (pw *PerformanceWrapper) recordRequest(duration time.Duration) {
	pw.totalRequests++
	pw.totalDuration += duration

	if duration > 5*time.Second {
		pw.slowRequestCount++
	}
}

// GetPerformanceStats returns current performance statistics
func (pw *PerformanceWrapper) GetPerformanceStats() string {
	avgDuration := time.Duration(0)
	if pw.totalRequests > 0 {
		avgDuration = time.Duration(int64(pw.totalDuration) / pw.totalRequests)
	}

	return fmt.Sprintf(`
📊 Alpaca Client Performance Stats
==================================
Total Requests:    %d
Average Duration:  %v
Total Time:        %v
Slow Requests:     %d (>5s)
Slow Request %%:    %.1f%%
`,
		pw.totalRequests,
		avgDuration,
		pw.totalDuration,
		pw.slowRequestCount,
		float64(pw.slowRequestCount)/float64(max(pw.totalRequests, 1))*100,
	)
}

// Forward other methods to the wrapped client
func (pw *PerformanceWrapper) Close() {
	// Print final performance report
	if pw.totalRequests > 0 {
		logger.Debug.Printf("📊 Alpaca Performance Report:%s", pw.GetPerformanceStats())
	}
}

func max(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
