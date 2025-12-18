package alpaca

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/jwaldner/barracuda/internal/providers"
)

const (
	// Rate limiting for Alpaca Basic Plan (200 requests per minute)
	basicPlanDelay = 350 * time.Millisecond

	// HTTP timeout
	defaultTimeout = 30 * time.Second
)

// AlpacaProvider implements the MarketProvider interface for Alpaca Markets
type AlpacaProvider struct {
	apiKey     string
	secretKey  string
	baseURL    string
	dataURL    string
	httpClient *http.Client

	// Rate limiting
	lastRequest time.Time
	rateMutex   sync.Mutex

	// Performance tracking
	totalRequests    int64
	totalQueueTime   time.Duration
	totalNetworkTime time.Duration
	totalParseTime   time.Duration
	totalRetries     int64
	cacheHits        int64
	rateLimitHits    int64
	statsMutex       sync.RWMutex
}

// NewAlpacaProvider creates a new Alpaca market data provider
func NewAlpacaProvider(apiKey, secretKey string) *AlpacaProvider {
	return &AlpacaProvider{
		apiKey:    apiKey,
		secretKey: secretKey,
		baseURL:   "https://api.alpaca.markets",
		dataURL:   "https://data.alpaca.markets",
		httpClient: &http.Client{
			Timeout: defaultTimeout,
		},
	}
}

// GetProviderName returns the provider name
func (a *AlpacaProvider) GetProviderName() string {
	return "alpaca"
}

// rateLimit enforces Alpaca's rate limiting (350ms between requests for Basic plan)
func (a *AlpacaProvider) rateLimit() time.Duration {
	a.rateMutex.Lock()
	defer a.rateMutex.Unlock()

	elapsed := time.Since(a.lastRequest)
	if elapsed < basicPlanDelay {
		waitTime := basicPlanDelay - elapsed
		time.Sleep(waitTime)
		a.lastRequest = time.Now()
		return waitTime
	}

	a.lastRequest = time.Now()
	return 0
}

// makeRequest handles HTTP requests with performance tracking
func (a *AlpacaProvider) makeRequest(ctx context.Context, endpoint string) ([]byte, providers.PerformanceMetrics, error) {
	metrics := providers.PerformanceMetrics{
		RequestCount: 1,
	}

	startTime := time.Now()

	// Rate limiting
	queueTime := a.rateLimit()
	metrics.QueueTime = queueTime
	metrics.RateLimitHit = queueTime > 0

	// Create request
	req, err := http.NewRequestWithContext(ctx, "GET", a.dataURL+endpoint, nil)
	if err != nil {
		return nil, metrics, fmt.Errorf("creating request: %v", err)
	}

	// Add auth headers
	req.Header.Set("APCA-API-KEY-ID", a.apiKey)
	req.Header.Set("APCA-API-SECRET-KEY", a.secretKey)

	// Make network request
	networkStart := time.Now()
	resp, err := a.httpClient.Do(req)
	networkTime := time.Since(networkStart)
	metrics.NetworkTime = networkTime

	if err != nil {
		return nil, metrics, fmt.Errorf("network request: %v", err)
	}
	defer resp.Body.Close()

	// Read response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, metrics, fmt.Errorf("reading response: %v", err)
	}

	metrics.BytesReceived = int64(len(body))
	metrics.RequestDuration = time.Since(startTime)

	// Check for rate limiting response
	if resp.StatusCode == 429 {
		metrics.RateLimitHit = true
		return nil, metrics, fmt.Errorf("rate limited by API")
	}

	if resp.StatusCode != 200 {
		return nil, metrics, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	// Update global stats
	a.updateStats(metrics)

	return body, metrics, nil
}

// updateStats updates cumulative performance statistics
func (a *AlpacaProvider) updateStats(metrics providers.PerformanceMetrics) {
	a.statsMutex.Lock()
	defer a.statsMutex.Unlock()

	a.totalRequests++
	a.totalQueueTime += metrics.QueueTime
	a.totalNetworkTime += metrics.NetworkTime
	a.totalParseTime += metrics.ParseTime
	a.totalRetries += int64(metrics.RetryAttempts)

	if metrics.CacheHit {
		a.cacheHits++
	}
	if metrics.RateLimitHit {
		a.rateLimitHits++
	}
}

// GetPerformanceStats returns cumulative performance statistics
func (a *AlpacaProvider) GetPerformanceStats() providers.PerformanceMetrics {
	a.statsMutex.RLock()
	defer a.statsMutex.RUnlock()

	avgQueueTime := time.Duration(0)
	avgNetworkTime := time.Duration(0)

	if a.totalRequests > 0 {
		avgQueueTime = time.Duration(int64(a.totalQueueTime) / a.totalRequests)
		avgNetworkTime = time.Duration(int64(a.totalNetworkTime) / a.totalRequests)
	}

	return providers.PerformanceMetrics{
		RequestDuration: avgNetworkTime + avgQueueTime,
		QueueTime:       avgQueueTime,
		NetworkTime:     avgNetworkTime,
		ParseTime:       time.Duration(int64(a.totalParseTime) / max(a.totalRequests, 1)),
		RequestCount:    int(a.totalRequests),
		RetryAttempts:   int(a.totalRetries),
		RateLimitHit:    a.rateLimitHits > 0,
	}
}

// Close cleans up resources
func (a *AlpacaProvider) Close() error {
	// Nothing to clean up for HTTP client
	return nil
}

// Alpaca API response structures
type alpacaBarResponse struct {
	Bars map[string][]alpacaBar `json:"bars"`
}

type alpacaBar struct {
	Close     float64   `json:"c"`
	High      float64   `json:"h"`
	Low       float64   `json:"l"`
	NumTrades int       `json:"n"`
	Open      float64   `json:"o"`
	Timestamp time.Time `json:"t"`
	Volume    int64     `json:"v"`
}

type alpacaOptionsResponse struct {
	OptionsContracts []alpacaOption `json:"options_contracts"`
}

type alpacaOption struct {
	ID               string    `json:"id"`
	Symbol           string    `json:"symbol"`
	Name             string    `json:"name"`
	UnderlyingSymbol string    `json:"underlying_symbol"`
	StrikePrice      float64   `json:"strike_price"`
	ExpirationDate   time.Time `json:"expiration_date"`
	OptionType       string    `json:"type"`
}

// GetStockPrices fetches current stock prices for given symbols
func (a *AlpacaProvider) GetStockPrices(ctx context.Context, symbols []string) (*providers.PriceResult, error) {
	if len(symbols) == 0 {
		return &providers.PriceResult{
			Data:    make(map[string]*providers.StockPrice),
			Metrics: providers.PerformanceMetrics{},
		}, nil
	}

	parseStart := time.Now()

	// Build endpoint with symbols
	symbolsParam := strings.Join(symbols, ",")
	endpoint := fmt.Sprintf("/v2/stocks/bars/latest?symbols=%s", symbolsParam)

	// Make request with performance tracking
	body, metrics, err := a.makeRequest(ctx, endpoint)
	if err != nil {
		return nil, fmt.Errorf("stock prices request: %v", err)
	}

	// Parse response
	var resp alpacaBarResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parsing stock prices response: %v", err)
	}

	metrics.ParseTime = time.Since(parseStart)

	// Convert to provider format
	results := make(map[string]*providers.StockPrice)
	for symbol, bars := range resp.Bars {
		if len(bars) > 0 {
			bar := bars[0] // Latest bar
			results[symbol] = &providers.StockPrice{
				Symbol:    symbol,
				Price:     bar.Close,
				Timestamp: bar.Timestamp,
				Volume:    bar.Volume,
			}
		}
	}

	return &providers.PriceResult{
		Data:    results,
		Metrics: metrics,
	}, nil
}

// GetOptionsContracts fetches available options contracts for a symbol
func (a *AlpacaProvider) GetOptionsContracts(ctx context.Context, symbol string, expiration time.Time) (*providers.ContractsResult, error) {
	parseStart := time.Now()

	// Build endpoint
	expirationStr := expiration.Format("2006-01-02")
	endpoint := fmt.Sprintf("/v2/options/contracts?underlying_symbols=%s&expiration_date=%s&limit=1000",
		symbol, expirationStr)

	// Make request with performance tracking
	body, metrics, err := a.makeRequest(ctx, endpoint)
	if err != nil {
		return nil, fmt.Errorf("options contracts request: %v", err)
	}

	// Parse response
	var resp alpacaOptionsResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parsing options contracts response: %v", err)
	}

	metrics.ParseTime = time.Since(parseStart)

	// Convert to provider format
	results := make([]*providers.OptionsContract, len(resp.OptionsContracts))
	for i, contract := range resp.OptionsContracts {
		results[i] = &providers.OptionsContract{
			ID:               contract.ID,
			Symbol:           contract.Symbol,
			UnderlyingSymbol: contract.UnderlyingSymbol,
			StrikePrice:      contract.StrikePrice,
			ExpirationDate:   contract.ExpirationDate,
			OptionType:       contract.OptionType,
		}
	}

	return &providers.ContractsResult{
		Data:    results,
		Metrics: metrics,
	}, nil
}

// GetOptionQuotes fetches current quotes for specific option contracts
func (a *AlpacaProvider) GetOptionQuotes(ctx context.Context, contractIDs []string) (*providers.QuotesResult, error) {
	if len(contractIDs) == 0 {
		return &providers.QuotesResult{
			Data:    make(map[string]*providers.OptionQuote),
			Metrics: providers.PerformanceMetrics{},
		}, nil
	}

	parseStart := time.Now()

	// Build endpoint with contract IDs
	contractsParam := strings.Join(contractIDs, ",")
	endpoint := fmt.Sprintf("/v1/options/quotes/latest?symbols=%s", contractsParam)

	// Make request with performance tracking
	_, metrics, err := a.makeRequest(ctx, endpoint)
	if err != nil {
		return nil, fmt.Errorf("option quotes request: %v", err)
	}

	// For now, return empty results as the quote structure is more complex
	// This can be expanded when needed
	metrics.ParseTime = time.Since(parseStart)

	return &providers.QuotesResult{
		Data:    make(map[string]*providers.OptionQuote),
		Metrics: metrics,
	}, nil
}

func max(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
