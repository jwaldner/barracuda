package handlers

import (
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	barracuda "github.com/jwaldner/barracuda/barracuda_lib"
	"github.com/jwaldner/barracuda/internal/alpaca"
	"github.com/jwaldner/barracuda/internal/config"
	"github.com/jwaldner/barracuda/internal/models"
	"github.com/jwaldner/barracuda/internal/symbols"
)

// getSymbolCount extracts symbol count from info map
func getSymbolCount(info map[string]interface{}) int {
	if count, ok := info["count"].(float64); ok {
		return int(count)
	}
	if count, ok := info["count"].(int); ok {
		return count
	}
	return 0
}

// OptionsHandler handles options analysis requests
type OptionsHandler struct {
	alpacaClient  *alpaca.Client
	symbolService *symbols.SP500Service
	config        *config.Config
	engine        *barracuda.BaracudaEngine
}

// NewOptionsHandler creates a new options handler
func NewOptionsHandler(alpacaClient *alpaca.Client, cfg *config.Config, engine *barracuda.BaracudaEngine) *OptionsHandler {
	return &OptionsHandler{
		alpacaClient:  alpacaClient,
		symbolService: symbols.NewSP500Service("assets/symbols"),
		config:        cfg,
		engine:        engine,
	}
}

// HomeHandler serves the main web interface
func (h *OptionsHandler) HomeHandler(w http.ResponseWriter, r *http.Request) {
	// Get S&P 500 symbols for the interface
	info, err := h.symbolService.GetSymbolsInfo()
	if err != nil {
		log.Printf("Warning: Could not get S&P 500 symbols: %v", err)
	}

		data := struct {
		Title           string
		DefaultStocks   []string
		DefaultCash     int
		DefaultStrategy string
		SP500Count      int
		CUDAAvailable   bool
		DeviceCount     int
		PaperTrading    bool
		WorkloadFactor  float64
	}{
		Title:           "Barracuda Options Analyzer",
		DefaultStocks:   h.config.DefaultStocks,
		DefaultCash:     h.config.DefaultCash,
		DefaultStrategy: h.config.DefaultStrategy,
		SP500Count:      getSymbolCount(info),
		CUDAAvailable:   h.engine.IsCudaAvailable(),
		DeviceCount:     h.engine.GetDeviceCount(),
		PaperTrading:    h.config.AlpacaPaperTrading,
		WorkloadFactor:  h.config.Engine.WorkloadFactor,
	}

	// Load template from file
	tmpl, err := template.ParseFiles("web/templates/home.html")
	if err != nil {
		http.Error(w, "Template error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	if err := tmpl.Execute(w, data); err != nil {
		http.Error(w, "Template execution error: "+err.Error(), http.StatusInternalServerError)
	}
}

// AnalyzeHandler handles options analysis requests using simplified Black-Scholes
func (h *OptionsHandler) AnalyzeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req models.AnalysisRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Validate request
	if len(req.Symbols) == 0 {
		http.Error(w, "At least one symbol is required", http.StatusBadRequest)
		return
	}
	if req.AvailableCash <= 0 {
		http.Error(w, "Available cash must be positive", http.StatusBadRequest)
		return
	}
	if req.Strategy != "puts" && req.Strategy != "calls" {
		http.Error(w, "Strategy must be 'puts' or 'calls'", http.StatusBadRequest)
		return
	}

	startTime := time.Now()
	log.Printf("📊 Analyzing %s options for %d symbols with batch processing", req.Strategy, len(req.Symbols))

	// Clean and prepare symbols
	cleanSymbols := make([]string, 0, len(req.Symbols))
	for _, symbol := range req.Symbols {
		if cleaned := strings.TrimSpace(symbol); cleaned != "" {
			cleanSymbols = append(cleanSymbols, cleaned)
		}
	}

	if len(cleanSymbols) == 0 {
		log.Printf("⚠️ No valid symbols provided")
		duration := time.Since(startTime)
		response := models.AnalysisResponse{
			Results:         []models.OptionResult{},
			RequestedDelta:  req.TargetDelta,
			Strategy:        req.Strategy,
			ExpirationDate:  req.ExpirationDate,
			Timestamp:       time.Now().Format(time.RFC3339),
			ProcessingTime:  duration.Seconds(),
			ProcessingStats: "No valid symbols provided",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		return
	}

	// Get stock prices in batches (up to 100 symbols per API call)
	log.Printf("📈 Fetching stock prices for %d symbols in batches...", len(cleanSymbols))
	stockPrices, err := h.alpacaClient.GetStockPricesBatch(cleanSymbols)
	if err != nil {
		log.Printf("❌ Error getting batch stock prices: %v", err)
		http.Error(w, "Failed to get stock prices: "+err.Error(), http.StatusServiceUnavailable)
		return
	}

	log.Printf("📊 Retrieved %d stock prices", len(stockPrices))
	for symbol, price := range stockPrices {
		log.Printf("💰 %s: $%.2f", symbol, price.Price)
	}

	// Process real options data using batch approach
	results := h.processBatchedRealOptions(stockPrices, req)

	// Sort results by total premium (highest first)
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].TotalPremium > results[i].TotalPremium {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Calculate processing statistics
	duration := time.Since(startTime)
	processingStats := fmt.Sprintf("✅ Completed in %.2f seconds | %d symbols | %d results | CUDA: %v",
		duration.Seconds(), len(cleanSymbols), len(results), h.engine.IsCudaAvailable())
	log.Printf(processingStats)

	response := models.AnalysisResponse{
		Results:         results,
		RequestedDelta:  req.TargetDelta,
		Strategy:        req.Strategy,
		ExpirationDate:  req.ExpirationDate,
		Timestamp:       time.Now().Format(time.RFC3339),
		ProcessingTime:  duration.Seconds(),
		ProcessingStats: processingStats,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// TestConnectionHandler tests Alpaca API connection
func (h *OptionsHandler) TestConnectionHandler(w http.ResponseWriter, r *http.Request) {
	// Test connection by getting account info
	if _, err := h.alpacaClient.GetStockPrice("AAPL"); err != nil {
		response := map[string]interface{}{
			"status":  "error",
			"message": "Alpaca API connection failed: " + err.Error(),
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(response)
		return
	}

	response := map[string]interface{}{
		"status":    "success",
		"message":   "Alpaca API connection successful",
		"timestamp": time.Now().Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// processBatchedRealOptions fetches real options with batching and parallel processing
func (h *OptionsHandler) processBatchedRealOptions(stockPrices map[string]*alpaca.StockPrice, req models.AnalysisRequest) []models.OptionResult {
	var results []models.OptionResult

	// Get symbols list
	symbols := make([]string, 0, len(stockPrices))
	for symbol := range stockPrices {
		symbols = append(symbols, symbol)
	}

	log.Printf("📊 Fetching real options data for %d symbols in batches...", len(symbols))

	// Process symbols in batches of 10 to avoid API rate limits
	batchSize := 10
	for i := 0; i < len(symbols); i += batchSize {
		end := i + batchSize
		if end > len(symbols) {
			end = len(symbols)
		}

		batch := symbols[i:end]
		log.Printf("🔄 Processing batch %d-%d (%d symbols)", i+1, end, len(batch))

		// Process calls and puts separately for better API performance
		if req.Strategy == "calls" || req.Strategy == "both" {
			callResults := h.processBatchOptionsType(batch, stockPrices, req, "calls")
			results = append(results, callResults...)
		}

		if req.Strategy == "puts" || req.Strategy == "both" {
			putResults := h.processBatchOptionsType(batch, stockPrices, req, "puts")
			results = append(results, putResults...)
		}

		// Rate limiting between batches
		if end < len(symbols) {
			time.Sleep(500 * time.Millisecond)
		}
	}

	log.Printf("✅ Processed %d symbol batches, %d total results", (len(symbols)+batchSize-1)/batchSize, len(results))
	return results
}

// processBatchOptionsType processes a batch of symbols for a specific option type (calls/puts)
func (h *OptionsHandler) processBatchOptionsType(symbols []string, stockPrices map[string]*alpaca.StockPrice, req models.AnalysisRequest, optionType string) []models.OptionResult {
	var results []models.OptionResult

	log.Printf("🎯 Fetching %s options for batch: %v", optionType, symbols)

	// Get options chain from Alpaca for this specific type
	optionsChain, err := h.alpacaClient.GetOptionsChain(symbols, req.ExpirationDate, optionType)
	if err != nil {
		log.Printf("❌ Error getting %s options chain: %v", optionType, err)
		return results
	}

	totalContracts := 0
	for _, contracts := range optionsChain {
		totalContracts += len(contracts)
	}
	log.Printf("📊 Retrieved %d %s contracts for batch", totalContracts, optionType)

	// Process each symbol's options
	for symbol, contracts := range optionsChain {
		stockPrice := stockPrices[symbol]
		if stockPrice == nil {
			continue
		}

		log.Printf("📈 Processing %d %s options for %s (stock: $%.2f)", len(contracts), optionType, symbol, stockPrice.Price)

		// Find best option based on target delta and available cash
		bestContract := h.findBestOptionContract(contracts, stockPrice.Price, req.TargetDelta, optionType)
		if bestContract != nil {
			result := h.convertRealOptionToResult(symbol, stockPrice.Price, bestContract, req.AvailableCash, req.ExpirationDate)
			if result != nil {
				results = append(results, *result)
				log.Printf("✅ Found best %s option for %s: Strike $%.2f, Premium $%.2f", optionType, symbol, result.Strike, result.Premium)
			}
		} else {
			log.Printf("⚠️ No suitable %s options found for %s", optionType, symbol)
		}
	}

	return results
}

// findBestOptionContract finds the best option contract based on criteria
func (h *OptionsHandler) findBestOptionContract(contracts []*alpaca.OptionContract, stockPrice float64, targetDelta float64, strategy string) *alpaca.OptionContract {
	if len(contracts) == 0 {
		return nil
	}

	// For puts, find contracts with strikes near the target delta range
	var bestContract *alpaca.OptionContract
	bestScore := float64(-1)

	for _, contract := range contracts {
		// Parse strike price
		strikePrice, err := strconv.ParseFloat(contract.StrikePrice, 64)
		if err != nil {
			continue
		}

		// Skip if not tradable
		if !contract.Tradable {
			continue
		}

		// Calculate score based on strike distance from current price
		var score float64
		if strategy == "puts" {
			// For puts, prefer strikes below current price
			if strikePrice < stockPrice {
				// Closer to target delta range (around 0.5 = 95% of stock price)
				targetStrike := stockPrice * 0.95
				score = 1.0 - math.Abs(strikePrice-targetStrike)/targetStrike
			}
		} else {
			// For calls, prefer strikes above current price
			if strikePrice > stockPrice {
				targetStrike := stockPrice * 1.05
				score = 1.0 - math.Abs(strikePrice-targetStrike)/targetStrike
			}
		}

		if score > bestScore {
			bestScore = score
			bestContract = contract
		}
	}

	return bestContract
}

// convertRealOptionToResult converts real Alpaca option contract to result format
func (h *OptionsHandler) convertRealOptionToResult(symbol string, stockPrice float64, contract *alpaca.OptionContract, availableCash float64, expirationDate string) *models.OptionResult {
	// Parse strike price
	strikePrice, err := strconv.ParseFloat(contract.StrikePrice, 64)
	if err != nil {
		log.Printf("❌ %s: Invalid strike price: %s", symbol, contract.StrikePrice)
		return nil
	}

	// Calculate time to expiration in days
	expDate, err := time.Parse("2006-01-02", expirationDate)
	if err != nil {
		return nil
	}
	daysToExp := int(time.Until(expDate).Hours() / 24)

	// Use a reasonable premium estimate (we'd need market data for real premium)
	// For now, use Black-Scholes as fallback for premium calculation
	timeToExp := time.Until(expDate).Hours() / (24 * 365.25)
	if timeToExp <= 0 {
		return nil
	}

	// Simple Black-Scholes for premium estimation
	riskFreeRate := 0.05
	volatility := 0.25
	optionPrice := h.estimateOptionPrice(stockPrice, strikePrice, timeToExp, riskFreeRate, volatility, contract.Type == "call")
	delta := h.estimateDelta(stockPrice, strikePrice, timeToExp, riskFreeRate, volatility, contract.Type == "call")

	// Calculate premium per contract (options are in contracts of 100 shares)
	premium := optionPrice * 100
	maxContracts := int(availableCash / premium)

	// Debug logging
	log.Printf("🔍 %s: Strike=%.2f, EstPrice=%.4f, Premium=%.2f, MaxContracts=%d",
		symbol, strikePrice, optionPrice, premium, maxContracts)

	if maxContracts <= 0 {
		log.Printf("⚠️ %s: Premium %.2f too high for available cash %.0f", symbol, premium, availableCash)
		return nil
	}

	return &models.OptionResult{
		Ticker:       symbol,
		OptionSymbol: contract.Symbol,
		OptionType:   contract.Type,
		Strike:       strikePrice,
		StockPrice:   stockPrice,
		Premium:      premium,
		MaxContracts: maxContracts,
		TotalPremium: float64(maxContracts) * premium,
		Delta:        delta,
		Expiration:   expirationDate,
		DaysToExp:    daysToExp,
	}
}

// estimateOptionPrice calculates Black-Scholes option price (helper method)
func (h *OptionsHandler) estimateOptionPrice(S, K, T, r, sigma float64, isCall bool) float64 {
	d1 := (math.Log(S/K) + (r+0.5*sigma*sigma)*T) / (sigma * math.Sqrt(T))
	d2 := d1 - sigma*math.Sqrt(T)

	if isCall {
		return S*h.normalCDF(d1) - K*math.Exp(-r*T)*h.normalCDF(d2)
	} else {
		return K*math.Exp(-r*T)*h.normalCDF(-d2) - S*h.normalCDF(-d1)
	}
}

// estimateDelta calculates option delta (helper method)
func (h *OptionsHandler) estimateDelta(S, K, T, r, sigma float64, isCall bool) float64 {
	d1 := (math.Log(S/K) + (r+0.5*sigma*sigma)*T) / (sigma * math.Sqrt(T))

	if isCall {
		return h.normalCDF(d1)
	} else {
		return h.normalCDF(d1) - 1
	}
}

// normalCDF approximates the cumulative distribution function of standard normal distribution
func (h *OptionsHandler) normalCDF(x float64) float64 {
	return 0.5 * (1.0 + math.Erf(x/math.Sqrt2))
}
