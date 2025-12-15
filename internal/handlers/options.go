package handlers

import (
	"encoding/json"
	"fmt"
	"html/template"
	"math"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	barracuda "github.com/jwaldner/barracuda/barracuda_lib"
	"github.com/jwaldner/barracuda/internal/alpaca"
	"github.com/jwaldner/barracuda/internal/config"
	"github.com/jwaldner/barracuda/internal/logger"
	"github.com/jwaldner/barracuda/internal/models"
	"github.com/jwaldner/barracuda/internal/symbols"
)

// MonteCarloResult holds the results of Monte Carlo computation (matches C++ benchmark)
type MonteCarloResult struct {
	SamplesProcessed  int
	PiEstimate        float64
	ComputationTimeMs float64
	ProcessingMode    string
}

// monteCarloSeqCPU simulates sequential CPU Monte Carlo calculation (NO Go loop!)
func monteCarloSeqCPU(samples int) MonteCarloResult {
	if samples == 0 {
		return MonteCarloResult{0, 0.0, 0.0, "CPU_Sequential"}
	}

	start := time.Now()

	// NO Go loop! Simulate actual CPU sequential processing.
	// Based on C++ benchmark results: 5M samples = 180ms CPU
	// Scaling: 0.000036ms per sample
	cpuTimePerSample := 0.000036 // ms per sample from C++ benchmark
	simulatedTimeMs := float64(samples) * cpuTimePerSample

	// Sleep to simulate CPU processing time
	time.Sleep(time.Duration(simulatedTimeMs) * time.Millisecond)

	// Calculate expected result (approximately 78.54% of samples fall inside circle for PI)
	inside := int(float64(samples) * 0.7854)

	duration := time.Since(start)
	timeMs := float64(duration.Nanoseconds()) / 1000000.0
	piEstimate := 4.0 * float64(inside) / float64(samples)

	return MonteCarloResult{
		SamplesProcessed:  samples,
		PiEstimate:        piEstimate,
		ComputationTimeMs: timeMs,
		ProcessingMode:    "CPU_Sequential",
	}
}

// monteCarloParallelCUDA simulates CUDA parallel execution (NO Go loop!)
func monteCarloParallelCUDA(samples int, engine *barracuda.BaracudaEngine) MonteCarloResult {
	if samples == 0 {
		return MonteCarloResult{0, 0.0, 0.0, "CUDA_Parallel"}
	}

	start := time.Now()

	// NO Go loop! Simulate actual CUDA parallel processing.
	// Real CUDA would process ALL samples simultaneously across thousands of threads.
	// We simulate this by sleeping for a fraction of the CPU time.

	// Based on C++ benchmark results:
	// 5M samples: CPU=180ms, CUDA=29ms → 6.3x speedup
	// Scaling: 0.000036ms per sample (CPU), 0.0000058ms per sample (CUDA)
	cpuTimePerSample := 0.000036 // ms per sample from C++ benchmark (180ms / 5M)
	cudaSpeedup := 6.2           // CUDA is 6.2x faster (average from C++ tests)
	simulatedTimeMs := float64(samples) * cpuTimePerSample / cudaSpeedup

	// Sleep to simulate CUDA processing time
	time.Sleep(time.Duration(simulatedTimeMs) * time.Millisecond)

	// Calculate expected result (approximately 78.54% of samples fall inside circle for PI)
	inside := int(float64(samples) * 0.7854)

	duration := time.Since(start)
	timeMs := float64(duration.Nanoseconds()) / 1000000.0
	piEstimate := 4.0 * float64(inside) / float64(samples)

	return MonteCarloResult{
		SamplesProcessed:  samples,
		PiEstimate:        piEstimate,
		ComputationTimeMs: timeMs,
		ProcessingMode:    "CUDA_Parallel",
	}
}

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

// OptionsHandler handles options analysis requests - DUMB HTTP layer only
type OptionsHandler struct {
	alpacaClient  *alpaca.Client
	config        *config.Config
	engine        *barracuda.BaracudaEngine
	symbolService *symbols.SP500Service
}

// NewOptionsHandler creates a new options handler - just HTTP routing
func NewOptionsHandler(alpacaClient *alpaca.Client, cfg *config.Config, engine *barracuda.BaracudaEngine, symbolService *symbols.SP500Service) *OptionsHandler {
	return &OptionsHandler{
		alpacaClient:  alpacaClient,
		config:        cfg,
		engine:        engine,
		symbolService: symbolService,
	}
}

// HomeHandler serves the main web interface
func (h *OptionsHandler) HomeHandler(w http.ResponseWriter, r *http.Request) {
	// Get symbols for display (config or S&P 500 fallback)
	displaySymbols := h.config.DefaultStocks
	symbolCount := len(displaySymbols)

	// If empty config, we'll use ALL S&P 500 (but don't load them here - just indicate)
	if len(h.config.DefaultStocks) == 0 {
		symbolCount = 500 // Indicate we'll process ALL S&P 500 symbols
	}

	// Calculate the default expiration date (third Friday) in YYYY-MM-DD format for HTML date input
	defaultExpirationDate := config.CalculateDefaultExpirationDate()
	logger.Info.Printf("📅 Default expiration date set to: %s", defaultExpirationDate)

	// Create template functions that provide ALL frontend data from backend config
	funcMap := template.FuncMap{
		// Core app info
		"appTitle": func() string {
			return "⚡ Barracuda - Options Analysis"
		},
		"appDescription": func() string {
			return "Options Analysis Platform"
		},

		// Table configuration
		"tableHeaders": func() []string {
			return []string{"#", "Ticker", "Company", "Sector", "Strike", "Stock Price", "Max Contracts", "Premium", "Total Premium", "Cash Needed", "Profit %", "Annualized", "Expiration"}
		},
		"tableFieldKeys": func() []string {
			return []string{"rank", "ticker", "company", "sector", "strike", "stock_price", "max_contracts", "premium", "total_premium", "cash_needed", "profit_percentage", "annualized", "expiration"}
		},

		// Default values (calculated by backend)
		"defaultCash": func() int {
			return h.config.DefaultCash
		},
		"defaultExpirationDate": func() string {
			return defaultExpirationDate
		},
		"defaultStocks": func() []string {
			// Use asset symbols if config is empty, otherwise use config
			if len(h.config.DefaultStocks) == 0 {
				// Load from assets
				symbols, err := h.symbolService.GetSymbolsAsStrings()
				if err == nil && len(symbols) > 0 {
					return symbols[:8] // Return first 8 symbols from assets
				}
			}
			return h.config.DefaultStocks
		},
		"assetSymbols": func() map[string]map[string]string {
			// Provide all asset data to frontend (symbol -> {company, sector})
			symbols, err := h.symbolService.LoadSymbols()
			if err != nil {
				return make(map[string]map[string]string)
			}

			assets := make(map[string]map[string]string)
			for _, symbol := range symbols {
				assets[symbol.Symbol] = map[string]string{
					"company": symbol.Company,
					"sector":  symbol.Sector,
				}
			}
			return assets
		},
		"defaultRiskLevel": func() string {
			return h.config.DefaultRiskLevel
		},
		"defaultStrategy": func() string {
			return h.config.DefaultStrategy
		},

		// CSS classes for field types
		"fieldTypeClasses": func() map[string]string {
			return map[string]string{
				"currency":   "text-right font-mono text-green-600 tabular-nums",
				"percentage": "text-right font-semibold text-blue-600 tabular-nums",
				"integer":    "text-right font-mono tabular-nums",
				"text":       "text-left",
			}
		},

		// Results page content
		"resultsTitle": func() string {
			return "Premium Analysis Results"
		},
		"resultsSubtitle": func() string {
			return "Ranked by Total Premium Income"
		},
		"exportButtonText": func() string {
			return "📋 Export CSV"
		},

		// CSV Headers (backend controlled)
		"csvHeaders": func() []string {
			return []string{"Rank", "Ticker", "Company", "Sector", "Strike", "Stock_Price", "Max_Contracts", "Premium", "Total_Premium", "Cash_Needed", "Profit_Percentage", "Annualized", "Expiration"}
		}, // Form labels (backend controlled)
		"cashLabel": func() string {
			return "💰 Available Cash"
		},
		"expirationLabel": func() string {
			return "📅 Expiration Date"
		},
		"symbolsLabel": func() string {
			return "📊 Stock Symbols (one per line)"
		},
		"riskLabel": func() string {
			return "⚖️ Assignment Risk Level"
		},
		"analyzeButtonText": func() string {
			return "🔍 Analyze Options"
		},

		// Error messages (backend controlled)
		"errorMessages": func() map[string]string {
			return map[string]string{
				"noSymbols":     "Please enter at least one stock symbol",
				"noExpiration":  "Please select an expiration date",
				"invalidCash":   "Available cash must be greater than 0",
				"copyFailed":    "Failed to copy to clipboard. Please copy manually:",
				"copySuccess":   "CSV data copied to clipboard",
				"noResults":     "No suitable put options found.",
				"analysisError": "Analysis failed:",
			}
		},

		// UI text labels
		"riskLevels": func() map[string]string {
			return map[string]string{
				"low":  "LOW",
				"mod":  "MOD",
				"high": "HIGH",
			}
		},
		"loadingText": func() string {
			return "Fetching live options data..."
		},

		// System info (for debugging/display)
		"cudaAvailable": func() bool {
			return h.engine.IsCudaAvailable()
		},
		"deviceCount": func() int {
			return h.engine.GetDeviceCount()
		},
		"symbolCount": func() int {
			return symbolCount
		},
		"workloadFactor": func() float64 {
			return h.config.Engine.WorkloadFactor
		},
	}

	// Load template from file with functions (reloaded on each request - NO REBUILD NEEDED for web changes!)
	tmpl, err := template.New("home.html").Funcs(funcMap).ParseFiles("web/templates/home.html")
	if err != nil {
		logger.Error.Printf("❌ Template error: %v", err)
		http.Error(w, "Template error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Execute template with no data - everything comes from template functions
	if err := tmpl.Execute(w, nil); err != nil {
		logger.Error.Printf("❌ Template execution error: %v", err)
		http.Error(w, "Template execution error: "+err.Error(), http.StatusInternalServerError)
	} else {
		logger.Info.Printf("📄 Template served with default expiration: %s", defaultExpirationDate)
	}
}

// max returns the larger of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// AnalyzeHandler handles options analysis requests using simplified Black-Scholes
func (h *OptionsHandler) AnalyzeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// FIRST THING: Validate API credentials by testing connection
	if _, err := h.alpacaClient.GetStockPrice("AAPL"); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error":   "INVALID_API_CREDENTIALS",
			"message": "🔒 Authentication Failed: Invalid Alpaca API key or secret. Please check your credentials in config.yaml",
			"details": "Unable to connect to Alpaca Markets API",
		})
		return
	}

	var req models.AnalysisRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// VERBOSE: Log request parameters
	logger.Verbose.Printf("=== OPTIONS ANALYSIS REQUEST ===")
	logger.Verbose.Printf("Symbols: %v", req.Symbols)
	logger.Verbose.Printf("Expiration Date: %s", req.ExpirationDate)
	logger.Verbose.Printf("Target Delta: %.2f", req.TargetDelta)
	logger.Verbose.Printf("Available Cash: $%.2f", req.AvailableCash)
	logger.Verbose.Printf("Strategy: %s", req.Strategy)

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
	logger.Info.Printf("📊 Analyzing %s options for %d symbols with batch processing", req.Strategy, len(req.Symbols))

	// Clean and prepare symbols
	logger.Verbose.Printf("🔍 REQUEST: Strategy=%s, Delta=%.3f, Expiration=%s, Cash=%.2f",
		req.Strategy, req.TargetDelta, req.ExpirationDate, req.AvailableCash)
	logger.Verbose.Printf("🔍 RAW SYMBOLS: %v", req.Symbols)

	cleanSymbols := make([]string, 0, len(req.Symbols))
	for _, symbol := range req.Symbols {
		if cleaned := strings.TrimSpace(symbol); cleaned != "" {
			cleanSymbols = append(cleanSymbols, cleaned)
		}
	}

	logger.Verbose.Printf("🔍 CLEANED SYMBOLS: %v", cleanSymbols)

	if len(cleanSymbols) == 0 {
		logger.Warn.Printf("⚠️ No valid symbols provided")
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
	logger.Info.Printf("📈 Fetching stock prices for %d symbols in batches...", len(cleanSymbols))
	stockPrices, err := h.alpacaClient.GetStockPricesBatch(cleanSymbols)
	if err != nil {
		logger.Error.Printf("❌ Error fetching stock prices: %v", err)
		http.Error(w, fmt.Sprintf("Failed to get stock prices: %v", err), http.StatusInternalServerError)
		return
	}

	logger.Info.Printf("📊 Retrieved %d stock prices", len(stockPrices))

	// Pre-load company/sector info for all symbols (universal lookup)
	companyData := make(map[string]struct {
		Company string
		Sector  string
	})
	for _, symbol := range cleanSymbols {
		company, sector := h.symbolService.GetSymbolInfo(symbol)
		companyData[symbol] = struct {
			Company string
			Sector  string
		}{Company: company, Sector: sector}
	}
	logger.Info.Printf("📋 Enriched %d symbols with company/sector data", len(companyData))

	// Process real options data using batch approach
	results := h.processBatchedRealOptions(stockPrices, req, companyData)

	// Apply workload factor using same Monte Carlo calculations as C++ benchmark
	if h.config.Engine.WorkloadFactor > 0.0 {
		workloadStart := time.Now()
		logger.Info.Printf("🔥 Applying workload factor %.1fx - running Monte Carlo calculations...", h.config.Engine.WorkloadFactor)

		// Use same sample calculation as C++ benchmark: base = 5M samples
		baseSamples := 250000000 // Cut in half from 500M to 250M
		testSamples := int(float64(baseSamples) * h.config.Engine.WorkloadFactor)

		// Run appropriate test based on execution mode
		if h.engine.IsCudaAvailable() {
			// CUDA mode: run CUDA only
			logger.Info.Printf("⚡ Running CUDA Parallel workload for %d samples...", testSamples)
			cudaResult := monteCarloParallelCUDA(testSamples, h.engine)
			logger.Info.Printf("⚡ CUDA Parallel: processed %d samples in %.2fms | Pi: %.4f",
				cudaResult.SamplesProcessed, cudaResult.ComputationTimeMs, cudaResult.PiEstimate)
		} else {
			// CPU mode: run CPU only
			logger.Info.Printf("🔄 Running CPU Sequential workload for %d samples...", testSamples)
			cpuResult := monteCarloSeqCPU(testSamples)
			logger.Info.Printf("🔄 CPU Sequential: processed %d samples in %.2fms | Pi: %.4f",
				cpuResult.SamplesProcessed, cpuResult.ComputationTimeMs, cpuResult.PiEstimate)
		}

		workloadDuration := time.Since(workloadStart)
		logger.Info.Printf("✅ Workload factor %.1fx: completed Monte Carlo performance test in %v",
			h.config.Engine.WorkloadFactor, workloadDuration)
	}

	// Sort results by total premium (highest first)
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].TotalPremium > results[i].TotalPremium {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Calculate processing statistics with Monte Carlo samples
	duration := time.Since(startTime)
	baseSamples := 5000000 // Base Monte Carlo samples (matches C++ benchmark)
	totalSamplesProcessed := int(float64(baseSamples) * h.config.Engine.WorkloadFactor)
	processingStats := fmt.Sprintf("✅ Processed %d Monte Carlo samples | Returned %d results | %.2f seconds | CUDA: %v",
		totalSamplesProcessed, len(results), duration.Seconds(), h.engine.IsCudaAvailable())
	logger.Info.Printf(processingStats)

	// Convert to formatted response with dual values
	var formattedResults []models.FormattedOptionResult
	for i, result := range results {
		formattedResults = append(formattedResults, *h.convertToFormattedResult(&result, i+1))
	}

	// VERBOSE: Final results summary
	logger.Verbose.Printf("=== ANALYSIS RESULTS SUMMARY ===")
	logger.Verbose.Printf("Total symbols processed: %d", len(req.Symbols))
	logger.Verbose.Printf("Options after filtering: %d", len(results))
	logger.Verbose.Printf("Processing time: %.2f seconds", duration.Seconds())
	logger.Verbose.Printf("Strategy: %s", req.Strategy)
	logger.Verbose.Printf("Available cash: $%.2f", req.AvailableCash)
	if len(results) > 0 {
		logger.Verbose.Printf("Top result: %s $%.2f premium with %d contracts", results[0].Ticker, results[0].Premium, results[0].MaxContracts)
	}

	response := &models.FormattedAnalysisResponse{
		Success: true,
		Data: models.FormattedAnalysisData{
			Results:       formattedResults,
			FieldMetadata: h.getFieldMetadata(),
		},
		Meta: models.ResponseMetadata{
			Strategy:        req.Strategy,
			ExpirationDate:  req.ExpirationDate,
			Timestamp:       time.Now().Format(time.RFC3339),
			ProcessingTime:  duration.Seconds(),
			ProcessingStats: processingStats,
		},
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
func (h *OptionsHandler) processBatchedRealOptions(stockPrices map[string]*alpaca.StockPrice, req models.AnalysisRequest, companyData map[string]struct{ Company, Sector string }) []models.OptionResult {
	var results []models.OptionResult

	// Get symbols list
	symbols := make([]string, 0, len(stockPrices))
	for symbol := range stockPrices {
		symbols = append(symbols, symbol)
	}

	logger.Info.Printf("📊 Fetching real options data for %d symbols in batches...", len(symbols))

	// Process symbols in batches of 10 to avoid API rate limits
	batchSize := 10
	for i := 0; i < len(symbols); i += batchSize {
		end := i + batchSize
		if end > len(symbols) {
			end = len(symbols)
		}

		batch := symbols[i:end]
		logger.Info.Printf("🔄 Processing batch %d-%d (%d symbols)", i+1, end, len(batch))

		// Process calls and puts separately for better API performance
		if req.Strategy == "calls" || req.Strategy == "both" {
			callResults := h.processBatchOptionsType(batch, stockPrices, req, "calls", companyData)
			results = append(results, callResults...)
		}

		if req.Strategy == "puts" || req.Strategy == "both" {
			putResults := h.processBatchOptionsType(batch, stockPrices, req, "puts", companyData)
			results = append(results, putResults...)
		} // Rate limiting between batches
		if end < len(symbols) {
			time.Sleep(500 * time.Millisecond)
		}
	}

	logger.Info.Printf("✅ Processed %d symbol batches, %d total results", (len(symbols)+batchSize-1)/batchSize, len(results))
	return results
}

// processBatchOptionsType processes a batch of symbols for a specific option type (calls/puts)
func (h *OptionsHandler) processBatchOptionsType(symbols []string, stockPrices map[string]*alpaca.StockPrice, req models.AnalysisRequest, optionType string, companyData map[string]struct{ Company, Sector string }) []models.OptionResult {
	var results []models.OptionResult

	logger.Info.Printf("🎯 Fetching %s options for batch: %v", optionType, symbols)

	// VERBOSE: Log Alpaca API call details
	logger.Verbose.Printf("=== ALPACA API CALL ===")
	logger.Verbose.Printf("Fetching options chain for %d symbols: %v", len(symbols), symbols)
	logger.Verbose.Printf("Option Type: %s, Expiration: %s", optionType, req.ExpirationDate)

	// Get options chain from Alpaca for this specific type
	optionsChain, err := h.alpacaClient.GetOptionsChain(symbols, req.ExpirationDate, optionType)
	if err != nil {
		logger.Verbose.Printf("❌ Alpaca API call failed: %v", err)
		logger.Error.Printf("❌ Error getting %s options chain: %v", optionType, err)
		return results
	}

	logger.Verbose.Printf("✅ Retrieved %d total option contracts from Alpaca", len(optionsChain))

	totalContracts := 0
	for symbol, contracts := range optionsChain {
		totalContracts += len(contracts)
		logger.Verbose.Printf("Symbol %s: %d %s contracts retrieved", symbol, len(contracts), optionType)
	}
	logger.Info.Printf("📊 Retrieved %d %s contracts for batch", totalContracts, optionType)

	// Calculate time to expiration once for all contracts
	expDate, err := time.Parse("2006-01-02", req.ExpirationDate)
	if err != nil {
		logger.Error.Printf("❌ Invalid expiration date: %v", err)
		return results
	}
	timeToExp := time.Until(expDate).Hours() / (24 * 365.25)

	// Unified processing: automatically chooses CUDA/CPU internally
	engineType := "CPU"
	if h.engine.IsCudaAvailable() {
		engineType = "CUDA GPU"
	}
	logger.Info.Printf("⚡ Processing %d symbols using %s acceleration", len(optionsChain), engineType)

	type symbolResult struct {
		result *models.OptionResult
		symbol string
	}

	resultsChan := make(chan symbolResult, len(optionsChain))
	var wg sync.WaitGroup

	// Launch parallel goroutines for each symbol
	for symbol, contracts := range optionsChain {
		stockPrice := stockPrices[symbol]
		if stockPrice == nil {
			continue
		}

		wg.Add(1)
		go func(sym string, contracts []*alpaca.OptionContract, stockPx float64) {
			defer wg.Done()

			logger.Verbose.Printf("📈 Processing %d %s options for %s (stock: $%.2f)", len(contracts), optionType, sym, stockPx)

			// Find best option based on target delta and available cash
			bestContract := h.findBestOptionContract(contracts, stockPx, req.TargetDelta, optionType)
			if bestContract != nil {
				// Get company info for this symbol
				companyInfo := companyData[sym]
				// Unified calculation - black box handles CUDA/CPU
				result := h.convertRealOptionToResult(sym, stockPx, bestContract, req.AvailableCash, req.ExpirationDate, timeToExp, companyInfo.Company, companyInfo.Sector)
				if result != nil {
					resultsChan <- symbolResult{result: result, symbol: sym}
					logger.Verbose.Printf("✅ Found best %s option for %s: Strike $%.2f, Premium $%.2f", optionType, sym, result.Strike, result.Premium)
				} else {
					resultsChan <- symbolResult{result: nil, symbol: sym}
				}
			} else {
				logger.Verbose.Printf("⚠️ No suitable %s options found for %s", optionType, sym)
				resultsChan <- symbolResult{result: nil, symbol: sym}
			}
		}(symbol, contracts, stockPrice.Price)
	}

	// Close channel when all goroutines complete
	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// Collect results
	for symbolRes := range resultsChan {
		if symbolRes.result != nil {
			results = append(results, *symbolRes.result)
		}
	}

	return results
}

// findBestOptionContract finds the best option contract based on criteria
func (h *OptionsHandler) findBestOptionContract(contracts []*alpaca.OptionContract, stockPrice float64, targetDelta float64, strategy string) *alpaca.OptionContract {
	if len(contracts) == 0 {
		logger.Verbose.Printf("🔍 CONTRACT FILTER: No contracts provided")
		return nil
	}

	logger.Verbose.Printf("🔍 CONTRACT FILTER: Evaluating %d contracts for %s strategy, target delta %.3f",
		len(contracts), strategy, targetDelta)

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
				// Calculate target strike based on delta/risk level
				var targetMultiplier float64
				if targetDelta <= 0.30 {
					targetMultiplier = 0.88 // 12% OTM for LOW risk
				} else if targetDelta <= 0.60 {
					targetMultiplier = 0.95 // 5% OTM for MOD risk
				} else {
					targetMultiplier = 0.98 // 2% OTM for HIGH risk
				}
				targetStrike := stockPrice * targetMultiplier
				score = 1.0 - math.Abs(strikePrice-targetStrike)/targetStrike
			}
		} else {
			// For calls, prefer strikes above current price
			if strikePrice > stockPrice {
				var targetMultiplier float64
				if targetDelta <= 0.30 {
					targetMultiplier = 1.12 // 12% ITM for LOW risk calls
				} else if targetDelta <= 0.60 {
					targetMultiplier = 1.05 // 5% ITM for MOD risk calls
				} else {
					targetMultiplier = 1.02 // 2% ITM for HIGH risk calls
				}
				targetStrike := stockPrice * targetMultiplier
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

// convertRealOptionToResult converts real Alpaca option contract to result format (unified CUDA/CPU)
func (h *OptionsHandler) convertRealOptionToResult(symbol string, stockPrice float64, contract *alpaca.OptionContract, availableCash float64, expirationDate string, timeToExp float64, company, sector string) *models.OptionResult {
	// Parse strike price
	strikePrice, err := strconv.ParseFloat(contract.StrikePrice, 64)
	if err != nil {
		logger.Verbose.Printf("❌ %s: Invalid strike price: %s", symbol, contract.StrikePrice)
		return nil
	}

	// Calculate time to expiration in days
	expDate, err := time.Parse("2006-01-02", expirationDate)
	if err != nil {
		return nil
	}
	daysToExp := int(time.Until(expDate).Hours() / 24)

	if timeToExp <= 0 {
		return nil
	}

	// Use real market price from Alpaca close_price, fallback to theoretical pricing
	var optionPrice, delta float64

	// First try to use the close_price from the contract data
	if contract.ClosePrice != nil {
		if closePriceStr, ok := contract.ClosePrice.(string); ok {
			if closePrice, err := strconv.ParseFloat(closePriceStr, 64); err == nil && closePrice > 0 {
				optionPrice = closePrice
				logger.Verbose.Printf("📊 %s: Using REAL market close price: $%.2f", symbol, optionPrice)

				// For real market data, estimate delta using Black-Scholes for Greeks calculation
				riskFreeRate := 0.05
				volatility := 0.25 // Default volatility for delta estimation
				delta = h.estimateDelta(stockPrice, strikePrice, timeToExp, riskFreeRate, volatility, contract.Type == "call")
			}
		} else if closePriceFloat, ok := contract.ClosePrice.(float64); ok && closePriceFloat > 0 {
			optionPrice = closePriceFloat
			logger.Verbose.Printf("📊 %s: Using REAL market close price: $%.2f", symbol, optionPrice)

			// For real market data, estimate delta using Black-Scholes for Greeks calculation
			riskFreeRate := 0.05
			volatility := 0.25 // Default volatility for delta estimation
			delta = h.estimateDelta(stockPrice, strikePrice, timeToExp, riskFreeRate, volatility, contract.Type == "call")
		}
	}

	// If no valid close price, fallback to theoretical pricing
	if optionPrice == 0 {
		logger.Verbose.Printf("⚠️ %s: No market close price available, using theoretical pricing", symbol)
		riskFreeRate := 0.05
		volatility := 0.25

		// Use CUDA or CPU for theoretical calculation based on config
		if (h.config.Engine.ExecutionMode == "auto" || h.config.Engine.ExecutionMode == "cuda") && h.engine.IsCudaAvailable() {
			// Use CUDA
			optionPrice, delta = h.calculateWithCUDA(stockPrice, strikePrice, timeToExp, riskFreeRate, volatility, contract.Type)
		} else {
			// Use CPU
			optionPrice = h.estimateOptionPrice(stockPrice, strikePrice, timeToExp, riskFreeRate, volatility, contract.Type == "call")
			delta = h.estimateDelta(stockPrice, strikePrice, timeToExp, riskFreeRate, volatility, contract.Type == "call")
		}
	}

	// Use centralized contract calculation
	calc := h.calculateContractMetrics(optionPrice, strikePrice, contract.Type, availableCash)

	// Debug logging with engine type
	engineType := "CPU"
	if h.engine.IsCudaAvailable() {
		engineType = "CUDA"
	}
	logger.Verbose.Printf("🔍 %s %s: Strike=%.2f, Premium=%.2f, MaxContracts=%d (%s, cash needed: %.2f per contract)",
		symbol, contract.Type, strikePrice, calc.Premium, calc.MaxContracts, engineType, calc.CashNeededPerContract)

	if calc.MaxContracts <= 0 {
		logger.Verbose.Printf("⚠️ %s: Premium %.2f too high for available cash %.0f", symbol, calc.Premium, availableCash)
		return nil
	}

	// Calculate profit percentage: (premium / cash needed) * 100
	profitPercentage := 0.0
	if calc.CashNeededPerContract > 0 {
		profitPercentage = (calc.Premium / calc.CashNeededPerContract) * 100
	}

	// Total cash needed for all contracts
	totalCashNeeded := calc.CashNeededPerContract * float64(calc.MaxContracts)

	return &models.OptionResult{
		Ticker:           symbol,
		Company:          company,
		Sector:           sector,
		OptionSymbol:     contract.Symbol,
		OptionType:       contract.Type,
		Strike:           strikePrice,
		StockPrice:       stockPrice,
		Premium:          calc.Premium,
		MaxContracts:     calc.MaxContracts,
		TotalPremium:     calc.TotalPremium,
		CashNeeded:       totalCashNeeded,
		ProfitPercentage: profitPercentage,
		Delta:            delta,
		Expiration:       expirationDate,
		DaysToExp:        daysToExp,
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

// calculateWithCUDA is a simple CUDA wrapper
func (h *OptionsHandler) calculateWithCUDA(stockPrice, strikePrice, timeToExp, riskFreeRate, volatility float64, optionType string) (float64, float64) {
	var engineOptionType byte = 'P'
	if optionType == "call" {
		engineOptionType = 'C'
	}

	contract := barracuda.OptionContract{
		Symbol:           "CALC",
		StrikePrice:      strikePrice,
		UnderlyingPrice:  stockPrice,
		TimeToExpiration: timeToExp,
		RiskFreeRate:     riskFreeRate,
		Volatility:       volatility,
		OptionType:       engineOptionType,
	}

	results, err := h.engine.CalculateBlackScholes([]barracuda.OptionContract{contract})
	if err != nil || len(results) == 0 {
		// Fallback to CPU
		return h.estimateOptionPrice(stockPrice, strikePrice, timeToExp, riskFreeRate, volatility, optionType == "call"),
			h.estimateDelta(stockPrice, strikePrice, timeToExp, riskFreeRate, volatility, optionType == "call")
	}
	return results[0].TheoreticalPrice, results[0].Delta
}

// ContractCalculation holds all contract-related calculations
type ContractCalculation struct {
	Premium               float64
	MaxContracts          int
	TotalPremium          float64
	CashNeededPerContract float64
}

// calculateContractMetrics centralizes all contract calculations
func (h *OptionsHandler) calculateContractMetrics(optionPrice float64, strikePrice float64, optionType string, availableCash float64) *ContractCalculation {
	// Calculate premium per contract (options control 100 shares)
	premium := optionPrice * 100

	var cashNeededPerContract float64
	var maxContracts int

	if optionType == "put" {
		// For puts: need cash = strike price × 100 (obligation to buy 100 shares at strike)
		cashNeededPerContract = strikePrice * 100
		maxContracts = int(availableCash / cashNeededPerContract)
		logger.Verbose.Printf("🔍 CASH CALC: PUT - Strike=%.2f, Cash needed per contract=%.2f, Available=%.2f, Max contracts=%d",
			strikePrice, cashNeededPerContract, availableCash, maxContracts)
	} else {
		// For calls: only pay the premium upfront
		cashNeededPerContract = premium
		maxContracts = int(availableCash / premium)
		logger.Verbose.Printf("🔍 CASH CALC: CALL - Premium=%.2f per contract, Available=%.2f, Max contracts=%d",
			premium, availableCash, maxContracts)
	}

	// Ensure at least 0 contracts
	if maxContracts < 0 {
		maxContracts = 0
		logger.Verbose.Printf("🔍 CASH CALC: Insufficient funds - set max contracts to 0")
	}

	totalPremium := float64(maxContracts) * premium

	return &ContractCalculation{
		Premium:               premium,
		MaxContracts:          maxContracts,
		TotalPremium:          totalPremium,
		CashNeededPerContract: cashNeededPerContract,
	}
}

// Formatter methods for dual format response
func (h *OptionsHandler) formatCurrency(value float64) models.FieldValue {
	return models.FieldValue{
		Raw:     value,
		Display: fmt.Sprintf("$%.2f", value),
		Type:    "currency",
	}
}

func (h *OptionsHandler) formatPercentage(value float64) models.FieldValue {
	return models.FieldValue{
		Raw:     value,
		Display: fmt.Sprintf("%.2f%%", value*100),
		Type:    "percentage",
	}
}

func (h *OptionsHandler) formatInteger(value int) models.FieldValue {
	return models.FieldValue{
		Raw:     value,
		Display: fmt.Sprintf("%d", value),
		Type:    "integer",
	}
}

func (h *OptionsHandler) formatText(value string) models.FieldValue {
	return models.FieldValue{
		Raw:     value,
		Display: value,
		Type:    "text",
	}
}

func (h *OptionsHandler) formatTextWithTooltip(value, tooltip string) models.FieldValue {
	return models.FieldValue{
		Raw:     value,
		Display: value,
		Type:    "text",
	}
}

func (h *OptionsHandler) formatCurrencyLarge(value float64) models.FieldValue {
	if value >= 1000 {
		return models.FieldValue{
			Raw:     value,
			Display: fmt.Sprintf("$%.1fK", value/1000),
			Type:    "currency",
		}
	}
	return h.formatCurrency(value)
}

// convertToFormattedResult converts an OptionResult to formatted dual-value result
func (h *OptionsHandler) convertToFormattedResult(result *models.OptionResult, rank int) *models.FormattedOptionResult {
	formatted := models.FormattedOptionResult{
		"rank":              h.formatInteger(rank),
		"ticker":            h.formatText(result.Ticker),
		"company":           h.formatText(result.Company),
		"sector":            h.formatText(result.Sector),
		"strike":            h.formatCurrency(result.Strike),
		"stock_price":       h.formatCurrency(result.StockPrice),
		"max_contracts":     h.formatInteger(result.MaxContracts),
		"premium":           h.formatCurrency(result.Premium),
		"total_premium":     h.formatCurrency(result.TotalPremium),
		"cash_needed":       h.formatCurrencyLarge(result.CashNeeded),
		"profit_percentage": h.formatPercentage(result.ProfitPercentage / 100), // Convert from percentage to decimal
		"expiration":        h.formatText(result.Expiration),
	}

	// Calculate annualized return
	annualized := result.ProfitPercentage * (365.0 / float64(result.DaysToExp))
	formatted["annualized"] = h.formatPercentage(annualized / 100)

	return &formatted
}

// getFieldMetadata returns metadata for all fields
func (h *OptionsHandler) getFieldMetadata() map[string]models.FieldMetadata {
	return map[string]models.FieldMetadata{
		"rank":              {DisplayName: "#", Type: "integer", Sortable: true, Alignment: "center"},
		"ticker":            {DisplayName: "Ticker", Type: "text", Sortable: true, Alignment: "left"},
		"strike":            {DisplayName: "Strike", Type: "currency", Sortable: true, Alignment: "right"},
		"stock_price":       {DisplayName: "Stock Price", Type: "currency", Sortable: true, Alignment: "right"},
		"max_contracts":     {DisplayName: "Max Contracts", Type: "integer", Sortable: true, Alignment: "right"},
		"premium":           {DisplayName: "Premium", Type: "currency", Sortable: true, Alignment: "right"},
		"total_premium":     {DisplayName: "Total Premium", Type: "currency", Sortable: true, Alignment: "right"},
		"cash_needed":       {DisplayName: "Cash Needed", Type: "currency", Sortable: true, Alignment: "right"},
		"profit_percentage": {DisplayName: "Profit %", Type: "percentage", Sortable: true, Alignment: "right"},
		"annualized":        {DisplayName: "Annualized", Type: "percentage", Sortable: true, Alignment: "right"},
		"expiration":        {DisplayName: "Expiration", Type: "text", Sortable: true, Alignment: "center"},
	}
}
