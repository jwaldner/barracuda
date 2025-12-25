package alpaca

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/jwaldner/barracuda/internal/audit"
	"github.com/jwaldner/barracuda/internal/logger"
)

// AlpacaInterface defines the methods that both Client and PerformanceWrapper implement
type AlpacaInterface interface {
	GetStockPrice(symbol string, auditTicker *string) (*StockPrice, error)
	GetStockPricesBatch(symbols []string, auditTicker *string) (map[string]*StockPrice, error)
	GetOptionsChain(symbols []string, expirationDate, strategy string, targetDelta float64, auditTicker *string) (map[string][]*OptionContract, error)
	GetOptionQuote(symbol string, auditTicker *string) (*OptionQuote, error)
}

const (
	// Rate limiting for Alpaca Basic Plan (200 requests per minute)
	BasicPlanDelay = 320 * time.Millisecond

	// Rate limiting for Alpaca Algo Trader Plus (10,000 requests per minute)
	AlgoPlusDelay = 10 * time.Millisecond

	// HTTP timeout
	DefaultTimeout = 30 * time.Second

	// Performance thresholds
	SlowRequestThreshold = 5 * time.Second
)

// AuditCallback is a function that can be called to append audit data
type AuditCallback func(symbol, operation string, data map[string]interface{})

type Client struct {
	APIKey        string
	SecretKey     string
	BaseURL       string
	DataURL       string
	HTTPClient    *http.Client
	AuditCallback AuditCallback
	auditLogger   audit.OptionsAnalysisAuditor
}

func NewClient(apiKey, secretKey string) *Client {
	baseURL := "https://api.alpaca.markets"
	dataURL := "https://data.alpaca.markets"

	return &Client{
		APIKey:    apiKey,
		SecretKey: secretKey,
		BaseURL:   baseURL,
		DataURL:   dataURL,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		auditLogger: audit.NewOptionsAnalysisAuditLogger(),
	}
}

// Stock Price Structures
type StockPrice struct {
	Symbol string  `json:"symbol"`
	Price  float64 `json:"price"`
}

type AlpacaBarResponse struct {
	Bar struct {
		Close     float64 `json:"c"`
		High      float64 `json:"h"`
		Low       float64 `json:"l"`
		NumTrades int     `json:"n"`
		Open      float64 `json:"o"`
		Timestamp string  `json:"t"`
		Volume    int     `json:"v"`
		VWAP      float64 `json:"vw"`
	} `json:"bar"`
	Symbol string `json:"symbol"`
}

// Options Quote Structures
type OptionQuote struct {
	AskPrice  float64 `json:"ap"`
	AskSize   int     `json:"as"`
	AskEx     string  `json:"ax"`
	BidPrice  float64 `json:"bp"`
	BidSize   int     `json:"bs"`
	BidEx     string  `json:"bx"`
	Condition string  `json:"c"`
	Timestamp string  `json:"t"`
}

type AlpacaOptionQuotesResponse struct {
	Quotes map[string]OptionQuote `json:"quotes"`
}

// Options Chain Structures
type OptionContract struct {
	ID                string      `json:"id"`
	Symbol            string      `json:"symbol"`
	Name              string      `json:"name"`
	Status            string      `json:"status"`
	Tradable          bool        `json:"tradable"`
	ExpirationDate    string      `json:"expiration_date"`
	RootSymbol        string      `json:"root_symbol"`
	UnderlyingSymbol  string      `json:"underlying_symbol"`
	UnderlyingAssetId string      `json:"underlying_asset_id"`
	Type              string      `json:"type"`
	Style             string      `json:"style"`
	StrikePrice       string      `json:"strike_price"`
	Multiplier        string      `json:"multiplier"`
	Size              string      `json:"size"`
	OpenInterest      interface{} `json:"open_interest"`
	OpenInterestDate  interface{} `json:"open_interest_date"`
	ClosePrice        interface{} `json:"close_price"`
	ClosePriceDate    interface{} `json:"close_price_date"`
	BidPrice          interface{} `json:"bid_price,omitempty"`
	AskPrice          interface{} `json:"ask_price,omitempty"`
	LastPrice         interface{} `json:"last_price,omitempty"`
	Ppind             bool        `json:"ppind"`
	Delta             float64     `json:"delta,omitempty"`
	Gamma             float64     `json:"gamma,omitempty"`
	Theta             float64     `json:"theta,omitempty"`
	Vega              float64     `json:"vega,omitempty"`
	ImpliedVol        float64     `json:"implied_volatility,omitempty"`
}

type AlpacaOptionsResponse struct {
	Options       []OptionContract `json:"option_contracts"`
	NextPageToken interface{}      `json:"next_page_token"`
}

// Get batch stock prices from Alpaca (up to 50 symbols per batch for rate limiting)
func (c *Client) GetStockPricesBatch(symbols []string, auditTicker *string) (map[string]*StockPrice, error) {
	results := make(map[string]*StockPrice)
	totalStartTime := time.Now()
	totalRequests := 0

	// Process in batches of 50 (rate limit optimization)
	batchSize := 50
	totalBatches := (len(symbols) + batchSize - 1) / batchSize
	logger.Info.Printf("📦 STOCK_BATCH_LOOP: Starting %d batches of %d symbols each", totalBatches, batchSize)

	for i := 0; i < len(symbols); i += batchSize {
		batchStartTime := time.Now()
		batchNum := (i / batchSize) + 1
		end := i + batchSize
		if end > len(symbols) {
			end = len(symbols)
		}

		batch := symbols[i:end]
		logger.Info.Printf("📦 STOCK_BATCH_LOOP: Batch %d/%d processing %d symbols", batchNum, totalBatches, len(batch))
		batchResults, err := c.getStockPricesBatchInternal(batch)
		batchDuration := time.Since(batchStartTime)
		totalRequests++

		if err != nil {
			return nil, fmt.Errorf("batch %d-%d failed: %v", i, end-1, err)
		}

		// Merge results
		for symbol, price := range batchResults {
			results[symbol] = price

			// Audit logging handled at handler level to avoid duplicates

			// Legacy audit callback for backward compatibility
			if auditTicker != nil && *auditTicker == symbol && c.AuditCallback != nil {
				c.AuditCallback(symbol, "GetStockPricesBatch", map[string]interface{}{
					"symbol":      symbol,
					"price":       price.Price,
					"batch_size":  len(batch),
					"duration_ms": batchDuration.Milliseconds(),
					"batch_index": i,
				})
			}
		}

		logger.Info.Printf("📦 STOCK_BATCH_LOOP: Batch %d/%d completed in %v", batchNum, totalBatches, batchDuration)

		// Rate limiting - 200 requests per minute
		if i+batchSize < len(symbols) {
			logger.Info.Printf("😴 STOCK_BATCH_LOOP: Sleeping %v for rate limiting", BasicPlanDelay)
			time.Sleep(BasicPlanDelay)
		}
	}

	totalDuration := time.Since(totalStartTime)
	reqPerMin := float64(totalRequests) / totalDuration.Minutes()
	logger.Info.Printf("🏁 STOCK_BATCH_LOOP: Finished %d batches in %v (%.1f req/min)", totalBatches, totalDuration, reqPerMin)

	return results, nil
}

// Internal method for single batch request
func (c *Client) getStockPricesBatchInternal(symbols []string) (map[string]*StockPrice, error) {
	if len(symbols) == 0 {
		return make(map[string]*StockPrice), nil
	}

	// Build symbols parameter
	symbolsParam := strings.Join(symbols, ",")
	endpoint := fmt.Sprintf("/v2/stocks/bars/latest?symbols=%s", symbolsParam)

	req, err := http.NewRequest("GET", c.DataURL+endpoint, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Add("APCA-API-KEY-ID", c.APIKey)
	req.Header.Add("APCA-API-SECRET-KEY", c.SecretKey)

	logger.Verbose.Printf("📡 ALPACA STOCK API CALL: %s", req.URL.String())
	startTime := time.Now()
	resp, err := c.HTTPClient.Do(req)
	callDuration := time.Since(startTime)
	if err != nil {
		logger.Verbose.Printf("❌ Stock API call failed after %v: %v", callDuration, err)
		return nil, err
	}
	logger.Verbose.Printf("⏱️ Stock API call completed in %v (status: %d)", callDuration, resp.StatusCode)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		logger.Verbose.Printf("❌ Alpaca Stock Price API error: Status %d, Body: %s", resp.StatusCode, string(body))
		return nil, fmt.Errorf("alpaca batch API error: %d - %s", resp.StatusCode, string(body))
	}

	// Read body for parsing
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	logger.Verbose.Printf("📈 Alpaca Stock Price Response for [%s] (%d bytes): %s", strings.Join(symbols, ","), len(body), string(body))

	// Parse the nested structure
	var batchResp struct {
		Bars map[string]struct {
			Close     float64 `json:"c"`
			High      float64 `json:"h"`
			Low       float64 `json:"l"`
			NumTrades int     `json:"n"`
			Open      float64 `json:"o"`
			Timestamp string  `json:"t"`
			Volume    int     `json:"v"`
			VWAP      float64 `json:"vw"`
		} `json:"bars"`
	}
	if err := json.Unmarshal(body, &batchResp); err != nil {
		return nil, fmt.Errorf("failed to decode batch response: %v - body: %s", err, string(body))
	}

	results := make(map[string]*StockPrice)
	for symbol, barData := range batchResp.Bars {
		results[symbol] = &StockPrice{
			Symbol: symbol,
			Price:  barData.Close,
		}
	}

	return results, nil
}

// Get real stock price from Alpaca using bars (more reliable than quotes)
func (c *Client) GetStockPrice(symbol string, auditTicker *string) (*StockPrice, error) {
	startTime := time.Now()
	endpoint := fmt.Sprintf("/v2/stocks/%s/bars/latest", symbol)

	req, err := http.NewRequest("GET", c.DataURL+endpoint, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Add("APCA-API-KEY-ID", c.APIKey)
	req.Header.Add("APCA-API-SECRET-KEY", c.SecretKey)

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("alpaca stock API error: %d - %s", resp.StatusCode, string(body))
	}

	var alpacaResp AlpacaBarResponse
	if err := json.NewDecoder(resp.Body).Decode(&alpacaResp); err != nil {
		return nil, err
	}

	stockPrice := &StockPrice{
		Symbol: symbol,
		Price:  alpacaResp.Bar.Close,
	}

	duration := time.Since(startTime)

	// Use unified audit package for logging
	if auditTicker != nil && *auditTicker == symbol {
		// Audit logging handled at handler level to avoid duplicates
	}

	// Legacy audit callback for backward compatibility
	if auditTicker != nil && *auditTicker == symbol && c.AuditCallback != nil {
		c.AuditCallback(symbol, "GetStockPrice", map[string]interface{}{
			"symbol":      symbol,
			"price":       stockPrice.Price,
			"endpoint":    endpoint,
			"duration_ms": duration.Milliseconds(),
			"status_code": resp.StatusCode,
		})
	}

	return stockPrice, nil
}

// Get options chain from Alpaca with filtering per symbol
func (c *Client) GetOptionsChain(symbols []string, expiration string, strategy string, targetDelta float64, auditTicker *string) (map[string][]*OptionContract, error) {
	contractsBySymbol := make(map[string][]*OptionContract)

	// Get stock prices in batches first to determine strike limits
	cleanSymbols := make([]string, 0, len(symbols))
	for _, symbol := range symbols {
		cleanSymbols = append(cleanSymbols, strings.TrimSpace(symbol))
	}

	stockPriceBatch, err := c.GetStockPricesBatch(cleanSymbols, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get stock prices: %v", err)
	}

	stockPrices := make(map[string]float64)
	for symbol, stockPrice := range stockPriceBatch {
		stockPrices[symbol] = stockPrice.Price
	}

	// Process each symbol individually for options
	optionsStartTime := time.Now()
	optionsRequests := 0
	logger.Info.Printf("📋 OPTIONS_LOOP: Starting options lookup for %d symbols", len(cleanSymbols))

	for symbolIndex, symbol := range cleanSymbols {
		symbolStartTime := time.Now()
		symbol = strings.TrimSpace(symbol)
		logger.Info.Printf("📋 OPTIONS_LOOP: Symbol %d/%d processing %s", symbolIndex+1, len(cleanSymbols), symbol)

		stockPrice, exists := stockPrices[symbol]
		if !exists {
			continue
		}

		endpoint := "/v2/options/contracts"
		req, err := http.NewRequest("GET", c.BaseURL+endpoint, nil)
		if err != nil {
			continue
		}

		// Add query parameters with filtering
		q := req.URL.Query()
		q.Add("underlying_symbols", symbol)
		if expiration != "" {
			q.Add("expiration_date", expiration)
		}

		// Filter by option type - PUTS ONLY (system never uses calls)
		q.Add("type", "put")
		// For puts: get OTM strikes only (below stock price) based on target delta
		var maxMultiplier float64
		if targetDelta <= 0.15 { // LOW risk (low delta)
			maxMultiplier = 0.88 // 12% OTM (far from stock price, low delta)
		} else if targetDelta <= 0.25 { // MOD risk
			maxMultiplier = 0.95 // 5% OTM (medium distance)
		} else { // HIGH risk (high delta)
			maxMultiplier = 0.98 // 2% OTM (close to stock price, high delta)
		}

		// Always OTM puts: strikes below stock price only
		maxStrike := fmt.Sprintf("%.0f", stockPrice*maxMultiplier) // Below stock price
		minStrike := fmt.Sprintf("%.0f", stockPrice*0.70)          // Deep OTM (70% of stock price)
		q.Add("strike_price_gte", minStrike)
		q.Add("strike_price_lte", maxStrike)
		logger.Verbose.Printf("🔍 ALPACA FILTER: %s PUTS (delta %.3f) - strikes $%s to $%s (%.0f%% to %.0f%% of stock $%.2f)",
			symbol, targetDelta, minStrike, maxStrike, 70.0, maxMultiplier*100, stockPrice)

		q.Add("limit", "1000")
		req.URL.RawQuery = q.Encode()

		req.Header.Add("APCA-API-KEY-ID", c.APIKey)
		req.Header.Add("APCA-API-SECRET-KEY", c.SecretKey)

		logger.Verbose.Printf("📡 ALPACA OPTIONS API CALL: %s", req.URL.String())
		startTime := time.Now()
		resp, err := c.HTTPClient.Do(req)
		callDuration := time.Since(startTime)
		if err != nil {
			logger.Verbose.Printf("❌ Options API call failed after %v: %v", callDuration, err)
			continue
		}
		defer resp.Body.Close()
		logger.Verbose.Printf("⏱️ Options API call completed in %v (status: %d)", callDuration, resp.StatusCode)

		body, _ := io.ReadAll(resp.Body)

		if resp.StatusCode != http.StatusOK {
			logger.Verbose.Printf("❌ Alpaca API error for %s: Status %d, Body: %s", symbol, resp.StatusCode, string(body))
			continue
		}

		logger.Verbose.Printf("📡 Alpaca API Response for %s (%d bytes): %s", symbol, len(body), string(body))

		// Reset body for JSON decoding
		resp.Body = io.NopCloser(strings.NewReader(string(body)))

		var alpacaResp AlpacaOptionsResponse
		if err := json.NewDecoder(resp.Body).Decode(&alpacaResp); err != nil {
			logger.Verbose.Printf("❌ JSON decode error for %s: %v", symbol, err)
			continue
		}

		logger.Verbose.Printf("✅ Parsed %d option contracts for %s", len(alpacaResp.Options), symbol)

		// Convert to pointers and sort by strike price
		contracts := make([]*OptionContract, len(alpacaResp.Options))
		for i := range alpacaResp.Options {
			contracts[i] = &alpacaResp.Options[i]
		}

		// Sort by strike price (ascending)
		for i := 0; i < len(contracts); i++ {
			for j := i + 1; j < len(contracts); j++ {
				strike1, _ := strconv.ParseFloat(contracts[i].StrikePrice, 64)
				strike2, _ := strconv.ParseFloat(contracts[j].StrikePrice, 64)
				if strike1 > strike2 {
					contracts[i], contracts[j] = contracts[j], contracts[i]
				}
			}
		}

		contractsBySymbol[symbol] = contracts

		// Use unified audit package for logging
		if auditTicker != nil && *auditTicker == symbol {
			// Collect all contract data
			var contractsData []map[string]interface{}
			for _, contract := range contracts {
				contractData := map[string]interface{}{
					"symbol":            contract.Symbol,
					"expiration_date":   contract.ExpirationDate,
					"strike_price":      contract.StrikePrice,
					"type":              contract.Type,
					"underlying_symbol": contract.UnderlyingSymbol,
					"multiplier":        contract.Multiplier,
					"status":            contract.Status,
					"tradable":          contract.Tradable,
				}

				if contract.BidPrice != nil {
					contractData["bid_price"] = contract.BidPrice
				}
				if contract.AskPrice != nil {
					contractData["ask_price"] = contract.AskPrice
				}
				if contract.ClosePrice != nil {
					contractData["close_price"] = contract.ClosePrice
				}
				if contract.LastPrice != nil {
					contractData["last_price"] = contract.LastPrice
				}
				if contract.OpenInterest != nil {
					contractData["open_interest"] = contract.OpenInterest
				}
				if contract.Delta != 0 {
					contractData["delta"] = contract.Delta
				}
				if contract.Gamma != 0 {
					contractData["gamma"] = contract.Gamma
				}
				if contract.Theta != 0 {
					contractData["theta"] = contract.Theta
				}
				if contract.Vega != 0 {
					contractData["vega"] = contract.Vega
				}
				if contract.ImpliedVol != 0 {
					contractData["implied_volatility"] = contract.ImpliedVol
				}

				contractsData = append(contractsData, contractData)
			}
		}

		// Legacy audit callback disabled - using unified audit system

		currentSymbolDuration := time.Since(symbolStartTime)
		optionsRequests++
		logger.Info.Printf("📋 OPTIONS_LOOP: Symbol %d/%d (%s) completed in %v", symbolIndex+1, len(cleanSymbols), symbol, currentSymbolDuration)

		// Rate limiting between options requests (300ms minimum for 200 req/min)
		logger.Info.Printf("😴 OPTIONS_LOOP: Sleeping %v for rate limiting", BasicPlanDelay)
		time.Sleep(BasicPlanDelay)
	}

	optionsDuration := time.Since(optionsStartTime)
	optionsReqPerMin := float64(optionsRequests) / optionsDuration.Minutes()
	logger.Info.Printf("🏁 OPTIONS_LOOP: Finished %d symbols in %v (%.1f req/min)", len(cleanSymbols), optionsDuration, optionsReqPerMin)

	return contractsBySymbol, nil
}

// GetOptionQuote gets real-time option quote (bid/ask) from Alpaca
func (c *Client) GetOptionQuote(optionSymbol string, auditTicker *string) (*OptionQuote, error) {
	startTime := time.Now()
	// Use the data API endpoint for option quotes
	endpoint := fmt.Sprintf("/v1beta1/options/quotes/latest?symbols=%s", optionSymbol)

	req, err := http.NewRequest("GET", c.DataURL+endpoint, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Add("APCA-API-KEY-ID", c.APIKey)
	req.Header.Add("APCA-API-SECRET-KEY", c.SecretKey)

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("alpaca option quote API error: %d - %s", resp.StatusCode, string(body))
	}

	var quoteResp AlpacaOptionQuotesResponse
	if err := json.NewDecoder(resp.Body).Decode(&quoteResp); err != nil {
		return nil, err
	}

	quote, exists := quoteResp.Quotes[optionSymbol]
	if !exists {
		return nil, fmt.Errorf("no quote found for option symbol: %s", optionSymbol)
	}

	duration := time.Since(startTime)

	// Use unified audit package for logging
	if auditTicker != nil && *auditTicker == optionSymbol {
		// Audit logging handled at handler level to avoid duplicates
	}

	// Legacy audit callback for backward compatibility
	if auditTicker != nil && *auditTicker == optionSymbol && c.AuditCallback != nil {
		c.AuditCallback(optionSymbol, "GetOptionQuote", map[string]interface{}{
			"option_symbol": optionSymbol,
			"bid_price":     quote.BidPrice,
			"ask_price":     quote.AskPrice,
			"endpoint":      endpoint,
			"duration_ms":   duration.Milliseconds(),
			"status_code":   resp.StatusCode,
		})
	}

	return &quote, nil
}

// SetAuditCallback sets the audit callback function for the client
func (c *Client) SetAuditCallback(callback AuditCallback) {
	c.AuditCallback = callback
}

// TestConnection tests connection to Alpaca API
func (c *Client) TestConnection() error {
	req, err := http.NewRequest("GET", c.BaseURL+"/v2/account", nil)
	if err != nil {
		return err
	}

	req.Header.Add("APCA-API-KEY-ID", c.APIKey)
	req.Header.Add("APCA-API-SECRET-KEY", c.SecretKey)

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("alpaca API connection failed: %d - %s", resp.StatusCode, string(body))
	}

	return nil
}
