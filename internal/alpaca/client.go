package alpaca

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

type Client struct {
	APIKey     string
	SecretKey  string
	BaseURL    string
	DataURL    string
	HTTPClient *http.Client
}

func NewClient(apiKey, secretKey string, paperTrading bool) *Client {
	baseURL := "https://api.alpaca.markets"
	dataURL := "https://data.alpaca.markets"

	if paperTrading {
		baseURL = "https://paper-api.alpaca.markets"
	}

	return &Client{
		APIKey:    apiKey,
		SecretKey: secretKey,
		BaseURL:   baseURL,
		DataURL:   dataURL,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
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

// Get batch stock prices from Alpaca (up to 100 symbols per request)
func (c *Client) GetStockPricesBatch(symbols []string) (map[string]*StockPrice, error) {
	results := make(map[string]*StockPrice)

	// Process in batches of 100 (Alpaca limit)
	batchSize := 100
	for i := 0; i < len(symbols); i += batchSize {
		end := i + batchSize
		if end > len(symbols) {
			end = len(symbols)
		}

		batch := symbols[i:end]
		batchResults, err := c.getStockPricesBatchInternal(batch)
		if err != nil {
			return nil, fmt.Errorf("batch %d-%d failed: %v", i, end-1, err)
		}

		// Merge results
		for symbol, price := range batchResults {
			results[symbol] = price
		}

		// Rate limiting - 200 requests per minute
		if i+batchSize < len(symbols) {
			time.Sleep(350 * time.Millisecond)
		}
	}

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

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("alpaca batch API error: %d - %s", resp.StatusCode, string(body))
	}

	// Read body for parsing
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

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
func (c *Client) GetStockPrice(symbol string) (*StockPrice, error) {
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

	return &StockPrice{
		Symbol: symbol,
		Price:  alpacaResp.Bar.Close,
	}, nil
}

// Get options chain from Alpaca with filtering per symbol
func (c *Client) GetOptionsChain(symbols []string, expiration string, strategy string) (map[string][]*OptionContract, error) {
	contractsBySymbol := make(map[string][]*OptionContract)

	// Get stock prices in batches first to determine strike limits
	cleanSymbols := make([]string, 0, len(symbols))
	for _, symbol := range symbols {
		cleanSymbols = append(cleanSymbols, strings.TrimSpace(symbol))
	}

	stockPriceBatch, err := c.GetStockPricesBatch(cleanSymbols)
	if err != nil {
		return nil, fmt.Errorf("failed to get stock prices: %v", err)
	}

	stockPrices := make(map[string]float64)
	for symbol, stockPrice := range stockPriceBatch {
		stockPrices[symbol] = stockPrice.Price
	}

	// Process each symbol individually for options
	for _, symbol := range cleanSymbols {
		symbol = strings.TrimSpace(symbol)

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

		// Filter by option type (puts or calls)
		if strategy == "puts" {
			q.Add("type", "put")
			// For puts: only get strikes below stock price
			q.Add("strike_price_lte", fmt.Sprintf("%.0f", stockPrice-1))
		} else {
			q.Add("type", "call")
			// For calls: only get strikes above stock price
			q.Add("strike_price_gte", fmt.Sprintf("%.0f", stockPrice+1))
		}

		q.Add("limit", "1000")
		req.URL.RawQuery = q.Encode()

		req.Header.Add("APCA-API-KEY-ID", c.APIKey)
		req.Header.Add("APCA-API-SECRET-KEY", c.SecretKey)

		resp, err := c.HTTPClient.Do(req)
		if err != nil {
			continue
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)

		if resp.StatusCode != http.StatusOK {
			continue
		}

		// Reset body for JSON decoding
		resp.Body = io.NopCloser(strings.NewReader(string(body)))

		var alpacaResp AlpacaOptionsResponse
		if err := json.NewDecoder(resp.Body).Decode(&alpacaResp); err != nil {
			continue
		}

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

		// Rate limiting between options requests
		time.Sleep(350 * time.Millisecond)
	}

	return contractsBySymbol, nil
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
