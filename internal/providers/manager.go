package providers

import (
	"context"
	"fmt"
	"time"

	"github.com/jwaldner/barracuda/internal/logger"
)

// ProviderManager manages market data providers and provides performance monitoring
type ProviderManager struct {
	provider MarketProvider
}

// NewProviderManager creates a new provider manager
func NewProviderManager(provider MarketProvider) *ProviderManager {
	return &ProviderManager{
		provider: provider,
	}
}

// GetStockPrices is a convenience wrapper that adds logging
func (pm *ProviderManager) GetStockPrices(ctx context.Context, symbols []string) (*PriceResult, error) {
	result, err := pm.provider.GetStockPrices(ctx, symbols)

	if err != nil {
		return nil, fmt.Errorf("provider %s failed to get stock prices: %v",
			pm.provider.GetProviderName(), err)
	}

	// Log performance if request was slow
	if result.Metrics.RequestDuration > 5*time.Second {
		logger.Warn.Printf("⚠️  SLOW REQUEST: %s stock prices took %v (queue: %v, network: %v)",
			pm.provider.GetProviderName(),
			result.Metrics.RequestDuration,
			result.Metrics.QueueTime,
			result.Metrics.NetworkTime)
	}

	return result, nil
}

// GetOptionsContracts is a convenience wrapper that adds logging
func (pm *ProviderManager) GetOptionsContracts(ctx context.Context, symbol string, expiration time.Time) (*ContractsResult, error) {
	result, err := pm.provider.GetOptionsContracts(ctx, symbol, expiration)

	if err != nil {
		return nil, fmt.Errorf("provider %s failed to get options contracts: %v",
			pm.provider.GetProviderName(), err)
	}

	// Log performance if request was slow
	if result.Metrics.RequestDuration > 5*time.Second {
		logger.Warn.Printf("⚠️  SLOW REQUEST: %s options contracts took %v (queue: %v, network: %v)",
			pm.provider.GetProviderName(),
			result.Metrics.RequestDuration,
			result.Metrics.QueueTime,
			result.Metrics.NetworkTime)
	}

	return result, nil
}

// GetOptionQuotes is a convenience wrapper that adds logging
func (pm *ProviderManager) GetOptionQuotes(ctx context.Context, contractIDs []string) (*QuotesResult, error) {
	result, err := pm.provider.GetOptionQuotes(ctx, contractIDs)

	if err != nil {
		return nil, fmt.Errorf("provider %s failed to get option quotes: %v",
			pm.provider.GetProviderName(), err)
	}

	return result, nil
}

// GetProvider returns the underlying provider
func (pm *ProviderManager) GetProvider() MarketProvider {
	return pm.provider
}

// GetPerformanceReport returns a detailed performance report
func (pm *ProviderManager) GetPerformanceReport() string {
	stats := pm.provider.GetPerformanceStats()

	report := fmt.Sprintf(`
📊 Provider Performance Report (%s)
=====================================
Requests Made:     %d
Average Queue Time: %v
Average Network:   %v  
Average Parse:     %v
Total Duration:    %v
Rate Limit Hits:   %v
Retry Attempts:    %d
Cache Hits:        %v
`,
		pm.provider.GetProviderName(),
		stats.RequestCount,
		stats.QueueTime,
		stats.NetworkTime,
		stats.ParseTime,
		stats.RequestDuration,
		stats.RateLimitHit,
		stats.RetryAttempts,
		stats.CacheHit,
	)

	return report
}

// Close cleans up the provider
func (pm *ProviderManager) Close() error {
	return pm.provider.Close()
}
