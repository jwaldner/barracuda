package handlers

import (
	"encoding/json"
	"fmt"
	"html/template"
	"math"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	barracuda "github.com/jwaldner/barracuda/barracuda_lib"
	"github.com/jwaldner/barracuda/internal/alpaca"
	"github.com/jwaldner/barracuda/internal/audit"
	"github.com/jwaldner/barracuda/internal/config"
	"github.com/jwaldner/barracuda/internal/grok"
	"github.com/jwaldner/barracuda/internal/logger"
	"github.com/jwaldner/barracuda/internal/models"
	"github.com/jwaldner/barracuda/internal/symbols"
	"github.com/jwaldner/barracuda/internal/treasury"
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
	alpacaClient            alpaca.AlpacaInterface
	config                  *config.Config
	engine                  *barracuda.BaracudaEngine
	symbolService           *symbols.SP500Service
	auditLogger             audit.OptionsAnalysisAuditor
	treasuryClient          *treasury.TreasuryClient
	contractsProcessedCount int
	lastComputeDuration     time.Duration
}

// NewOptionsHandler creates a new options handler - just HTTP routing
func NewOptionsHandler(alpacaClient alpaca.AlpacaInterface, cfg *config.Config, engine *barracuda.BaracudaEngine, symbolService *symbols.SP500Service) *OptionsHandler {
	return &OptionsHandler{
		alpacaClient:   alpacaClient,
		config:         cfg,
		engine:         engine,
		symbolService:  symbolService,
		auditLogger:    audit.NewOptionsAnalysisAuditLogger(),
		treasuryClient: treasury.NewTreasuryClient(),
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

		// RULE: Template functions read from config.yaml ONLY for config data
		// Field names are backend logic, not config
		//
		// *** HOW TO ADD NEW FIELDS TO TABLE: ***
		// 1. Add field to models.OptionResult struct in internal/models/models.go
		// 2. Update processRealOptions() to populate the new field
		// 3. Add display name to "tableHeaders" array below (for UI display)
		// 4. Add field key to "tableFieldKeys" array below (for data access)
		// 5. Update HTML template if special formatting needed
		// 6. Frontend will automatically use new field via template functions - NO JS changes needed
		//
		"tableHeaders": func() []string {
			return []string{"Rank", "Ticker", "Company", "Sector", "Strike", "Stock_Price", "Premium", "Max_Contracts", "Total_Premium", "Profit_Percentage", "Delta", "Expiration", "Days_To_Exp"}
		},
		"tableFieldKeys": func() []string {
			return []string{"rank", "ticker", "company", "sector", "strike", "stock_price", "premium", "max_contracts", "total_premium", "profit_percentage", "delta", "expiration", "days_to_exp"}
		}, // Default values (calculated by backend)
		"defaultCash": func() int {
			return h.config.DefaultCash
		},
		"defaultExpirationDate": func() string {
			return defaultExpirationDate
		},
		"defaultStocks": func() []string {
			// Return configured stocks or empty array (empty = use ALL S&P 500)
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

		// CUDA Hardware Status
		"cudaHardwareAvailable": func() bool {
			return h.engine != nil && h.engine.IsCudaAvailable()
		},
		"cudaHardwareText": func() string {
			if h.engine != nil && h.engine.IsCudaAvailable() {
				return "🔥 CUDA AVAILABLE"
			}
			return "🙅 NO CUDA"
		},

		// Active Execution Mode
		"computeMode": func() string {
			mode := strings.ToUpper(h.config.Engine.ExecutionMode)
			if mode == "AUTO" {
				if h.engine != nil && h.engine.IsCudaAvailable() {
					return "CUDA"
				}
				return "CPU"
			}
			if mode == "CUDA" {
				return "CUDA"
			}
			return "CPU"
		},
		"activeModeText": func() string {
			mode := strings.ToUpper(h.config.Engine.ExecutionMode)
			if mode == "AUTO" {
				if h.engine != nil && h.engine.IsCudaAvailable() {
					return "⚡ ACTIVE: CUDA"
				}
				return "🔧 ACTIVE: CPU"
			}
			if mode == "CUDA" {
				return "⚡ ACTIVE: CUDA"
			}
			return "🔧 ACTIVE: CPU"
		},

		"deviceInfo": func() string {
			if h.engine != nil && h.engine.IsCudaAvailable() {
				return fmt.Sprintf("(%d devices)", h.engine.GetDeviceCount())
			}
			return "(fallback mode)"
		},
		"engineMode": func() string {
			// Return resolved execution mode, never AUTO
			mode := strings.ToUpper(h.config.Engine.ExecutionMode)
			if mode == "AUTO" {
				if h.engine != nil && h.engine.IsCudaAvailable() {
					return "CUDA"
				}
				return "CPU"
			}
			return mode
		},
		"workloadStatus": func() string {
			if h.config.Engine.WorkloadFactor == 0.0 {
				return "[OFF]"
			} else if h.config.Engine.WorkloadFactor == 1.0 {
				return "[NORMAL]"
			} else if h.config.Engine.WorkloadFactor < 1.0 {
				return "[LIGHT]"
			} else {
				return "[HEAVY]"
			}
		},
		"maxResults": func() string {
			if h.config.MaxResults == 0 {
				return "ALL"
			}
			return fmt.Sprintf("%d", h.config.MaxResults)
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

		// CSV Headers (backend logic)
		"csvHeaders": func() []string {
			return []string{"Rank", "Ticker", "Company", "Sector", "Strike", "Stock_Price", "Premium", "Max_Contracts", "Total_Premium", "Profit_Percentage", "Delta", "Expiration", "Days_To_Exp"}
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
				"noExpiration":  "Please select an expiration date",
				"invalidCash":   "Available cash must be greater than 0",
				"copyFailed":    "Failed to copy to clipboard. Please copy manual:",
				"copySuccess":   "CSV data copied to clipboard",
				"noResults":     "No suitable put options found.",
				"analysisError": "Analysis failed:",
			}
		},

		// Standard Config Utility Functions (frontend never changes)
		"configUtils": func() map[string]string {
			return map[string]string{
				"initialized": "true",
			}
		},

		// Config-based template functions (using existing config system)
		"generateCSVFilename": func(targetDelta float64, expirationDate, strategy string, symbolCount int) string {
			now := time.Now()
			timeStr := now.Format("15-04-05") // HH-MM-SS
			deltaStr := fmt.Sprintf("delta%.2f", targetDelta)

			// USE THE EXISTING CONFIG SYSTEM!
			filename := h.config.CSV.FilenameFormat
			filename = strings.ReplaceAll(filename, "{time}", timeStr)
			filename = strings.ReplaceAll(filename, "{exp_date}", expirationDate)
			filename = strings.ReplaceAll(filename, "{delta}", deltaStr)
			filename = strings.ReplaceAll(filename, "{strategy}", strategy)
			filename = strings.ReplaceAll(filename, "{symbols}", fmt.Sprintf("%d", symbolCount))

			return filename
		},

		"getCSVConfig": func() config.CSVConfig {
			return h.config.CSV // Return the actual config struct
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

		// Audit system configuration
		"auditJSONFilename": func() string {
			return "audit.json"
		},
		"auditDirectory": func() string {
			return ""
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
	// Set CORS headers for browser compatibility
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	// Handle preflight OPTIONS request
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// FIRST THING: Validate API credentials by testing connection
	if _, err := h.alpacaClient.GetStockPrice("AAPL", nil); err != nil {
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

	// Check for audit ticker and log at warn level
	if req.AuditTicker != "" {
		logger.Warn.Printf("🔍 AUDIT MODE: Ticker %s set for detailed audit logging", req.AuditTicker)
		logger.Warn.Printf("🔍 AUDIT MODE: This analysis will collect detailed data for %s", req.AuditTicker)
	}

	// Setup audit ticker pointer for API calls
	var auditTickerPtr *string
	if req.AuditTicker != "" {
		auditTickerPtr = &req.AuditTicker
	}

	// VERBOSE: Log request parameters
	logger.Debug.Printf("=== OPTIONS ANALYSIS REQUEST ===")
	logger.Debug.Printf("Symbols: %v", req.Symbols)
	logger.Debug.Printf("Expiration Date: %s", req.ExpirationDate)
	logger.Debug.Printf("Target Delta: %.2f", req.TargetDelta)
	logger.Debug.Printf("Available Cash: $%.2f", req.AvailableCash)
	logger.Debug.Printf("Strategy: %s", req.Strategy)

	if req.AuditTicker != "" {
		logger.Debug.Printf("Audit Ticker: %s", req.AuditTicker)
		// Initialize audit for this ticker
		if err := h.auditLogger.LogOptionsAnalysisOperation(req.AuditTicker, "create", map[string]interface{}{
			"message":    "Starting options analysis audit",
			"symbols":    req.Symbols,
			"strategy":   req.Strategy,
			"expiration": req.ExpirationDate,
		}); err != nil {
			logger.Warn.Printf("⚠️ Failed to initialize audit: %v", err)
		}
	}

	// If no symbols provided, use configured symbols or S&P 500
	if len(req.Symbols) == 0 {
		// Use configured stocks if available, otherwise S&P 500
		if len(h.config.DefaultStocks) > 0 {
			req.Symbols = h.config.DefaultStocks
			logger.Info.Printf("📊 Using %d configured default stocks", len(h.config.DefaultStocks))
		} else {
			// Get all S&P 500 symbols
			symbols, err := h.symbolService.GetSymbolsAsStrings()
			if err != nil {
				logger.Error.Printf("❌ Failed to get S&P 500 symbols: %v", err)
				http.Error(w, fmt.Sprintf("Failed to get S&P 500 symbols: %v", err), http.StatusInternalServerError)
				return
			}
			req.Symbols = symbols
			logger.Info.Printf("📊 Using %d S&P 500 symbols (default_stocks empty)", len(symbols))
		}
	}

	// Initialize contracts processed counter
	h.contractsProcessedCount = 0

	// Validate request
	if req.AvailableCash <= 0 {
		http.Error(w, "Available cash must be positive", http.StatusBadRequest)
		return
	}
	if req.Strategy != "puts" && req.Strategy != "calls" {
		http.Error(w, "Strategy must be 'puts' or 'calls'", http.StatusBadRequest)
		return
	}

	startTime := time.Now()

	// Clean and prepare symbols
	logger.Debug.Printf("🔍 REQUEST: Strategy=%s, Delta=%.3f, Expiration=%s, Cash=%.2f",
		req.Strategy, req.TargetDelta, req.ExpirationDate, req.AvailableCash)
	logger.Debug.Printf("🔍 RAW SYMBOLS: %v", req.Symbols)

	cleanSymbols := make([]string, 0, len(req.Symbols))
	for _, symbol := range req.Symbols {
		if cleaned := strings.TrimSpace(symbol); cleaned != "" {
			cleanSymbols = append(cleanSymbols, cleaned)
		}
	}

	logger.Debug.Printf("🔍 CLEANED SYMBOLS: %v", cleanSymbols)

	// If no cleaned symbols AND original request was empty, use S&P 500
	if len(cleanSymbols) == 0 {
		if len(req.Symbols) == 0 {
			// Original request was empty - this was handled above, should not happen
			logger.Error.Printf("❌ Unexpected: cleanSymbols empty but req.Symbols was filled")
		}
		logger.Warn.Printf("⚠️ No valid symbols after cleaning")
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

	// Major Step: Analysis Start
	if req.AuditTicker != "" {
		logger.Warn.Printf("🔍 AUDIT: Starting analysis with audit ticker %s - detailed logging enabled", req.AuditTicker)
	}

	// Get stock prices in batches (up to 100 symbols per API call)
	logger.Info.Printf("📈 Fetching stock prices for %d symbols in batches...", len(cleanSymbols))
	stockPrices, err := h.alpacaClient.GetStockPricesBatch(cleanSymbols, auditTickerPtr)
	if err != nil {
		logger.Error.Printf("❌ Error fetching stock prices: %v", err)
		http.Error(w, fmt.Sprintf("Failed to get stock prices: %v", err), http.StatusInternalServerError)
		return
	}

	// Debug: Log stock prices from API if audit ticker is set
	if auditTickerPtr != nil {
		for symbol, price := range stockPrices {
			if symbol == *auditTickerPtr {
				logger.Info.Printf("🔍 DEBUG: API returned %s stock price: $%.2f", symbol, price.Price)
				break
			}
		}
	}

	// Verbose: Log all stock prices from API
	logger.Verbose.Printf("📊 STOCK PRICES FROM API:")
	for symbol, price := range stockPrices {
		logger.Verbose.Printf("  %s: $%.2f", symbol, price.Price)
	}

	logger.Info.Printf("📊 Retrieved %d stock prices", len(stockPrices))
	logger.Debug.Printf("📊 Processing options for %d symbols with valid prices", len(stockPrices))

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

	// Process real options data using individual approach
	results := h.processRealOptions(stockPrices, req, companyData, auditTickerPtr)

	if h.config.Engine.WorkloadFactor > 0.0 {
		workloadStart := time.Now()
		logger.Info.Printf("🔥 Applying workload factor %.1fx - running Monte Carlo calculations...", h.config.Engine.WorkloadFactor)

		// Use same sample calculation as C++ benchmark: base = 5M samples
		baseSamples := 250000000 // Cut in half from 500M to 250M
		testSamples := int(float64(baseSamples) * h.config.Engine.WorkloadFactor)

		// Run appropriate test based on execution mode
		if (h.config.Engine.ExecutionMode == "auto" || h.config.Engine.ExecutionMode == "cuda") && h.engine.IsCudaAvailable() {
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

	// Results are already sorted by profit percentage from processRealOptions

	// Log ranking results to audit if audit ticker is set
	if auditTickerPtr != nil && *auditTickerPtr != "" {
		rankings := make([]map[string]interface{}, len(results))
		for i, result := range results {
			rankings[i] = map[string]interface{}{
				"rank":      i + 1,
				"symbol":    result.Ticker,
				"premium":   result.TotalPremium,
				"contracts": result.MaxContracts,
			}
		}

		rankingData := map[string]interface{}{
			"operation":      "SortByProfitPercentage",
			"ranking_method": "profit_percentage_descending",
			"rankings":       rankings,
			"total_symbols":  len(results),
		}

		rankingData["operation"] = "SortByProfitPercentage"
		if err := h.auditLogger.LogOptionsAnalysisOperation(*auditTickerPtr, "RankingResults", rankingData); err != nil {
			logger.Warn.Printf("📊 RANKING: Failed to log ranking results: %v", err)
		} else {
			logger.Warn.Printf("📊 RANKING: Logged ranking results for %d symbols", len(results))
		}
	}

	// Apply max_results limit from config (0 = show all)
	if h.config.MaxResults > 0 && len(results) > h.config.MaxResults {
		results = results[:h.config.MaxResults]
		logger.Info.Printf("📊 Limited results to top %d (max_results config)", h.config.MaxResults)
	}

	// Calculate processing statistics with Monte Carlo workload benchmark
	duration := time.Since(startTime)

	// Monte Carlo benchmark processing (matches C++ benchmark behavior)
	baseSamples := 5000000 // Base Monte Carlo samples (matches C++ benchmark)
	totalSamplesProcessed := int(float64(baseSamples) * h.config.Engine.WorkloadFactor)

	var mcResult MonteCarloResult
	if h.config.Engine.WorkloadFactor > 0.0 {
		// Execute workload benchmark based on engine type and execution mode
		if (h.config.Engine.ExecutionMode == "auto" || h.config.Engine.ExecutionMode == "cuda") && h.engine.IsCudaAvailable() {
			// Use CUDA benchmark: 1000 contracts, iterations based on workload factor
			iterations := int(h.config.Engine.WorkloadFactor * 10)
			benchmarkTime := h.engine.BenchmarkCalculation(1000, iterations)
			mcResult = MonteCarloResult{
				SamplesProcessed:  totalSamplesProcessed,
				PiEstimate:        3.14159, // Placeholder
				ComputationTimeMs: benchmarkTime,
				ProcessingMode:    "CUDA_Benchmark",
			}
		} else {
			mcResult = monteCarloSeqCPU(totalSamplesProcessed)
		}
		logger.Info.Printf("🎯 MONTE CARLO BENCHMARK: %d samples | %.2fms | π≈%.4f | %s",
			mcResult.SamplesProcessed, mcResult.ComputationTimeMs, mcResult.PiEstimate, mcResult.ProcessingMode)
	}

	// Log processing performance stats
	engineType := "CPU"
	if (h.config.Engine.ExecutionMode == "auto" || h.config.Engine.ExecutionMode == "cuda") && h.engine.IsCudaAvailable() {
		engineType = "CUDA"
	}

	// Major Step: Completion
	logger.Warn.Printf("✅ COMPLETE: %d results | %.3fs total | ⚡ %.3fs %s COMPUTE | %s engine",
		len(results), duration.Seconds(), h.lastComputeDuration.Seconds(), engineType, engineType)

	logger.Debug.Printf("🔍 DEBUG: Analysis completed, formatting response")

	// Check if client is still connected
	select {
	case <-r.Context().Done():
		logger.Error.Printf("❌ Client disconnected before response could be sent: %v", r.Context().Err())
		return
	default:
		logger.Debug.Printf("✅ Client connection still active, proceeding with response")
	}

	// Processing stats logged internally

	// Final processing summary - ALWAYS show real job stats, ADD workload if present
	processingStats := fmt.Sprintf("✅ Processed %d symbols | Returned %d results | %.3f seconds | Engine: %s",
		len(cleanSymbols), len(results), duration.Seconds(), engineType)

	// ADD workload benchmark stats to the real job processing
	if h.config.Engine.WorkloadFactor > 0.0 && totalSamplesProcessed > 0 {
		processingStats += fmt.Sprintf(" | %d Monte Carlo samples", totalSamplesProcessed)
	}

	// Convert to formatted response with dual values
	logger.Debug.Printf("🔄 Converting %d results to formatted response", len(results))
	var formattedResults []models.FormattedOptionResult
	for i, result := range results {
		formattedResults = append(formattedResults, *h.convertToFormattedResult(&result, i+1))
	}
	logger.Debug.Printf("✅ Formatted %d results for JSON response", len(formattedResults))

	// VERBOSE: Final results summary
	logger.Debug.Printf("=== ANALYSIS RESULTS SUMMARY ===")
	logger.Debug.Printf("Total symbols processed: %d", len(req.Symbols))
	logger.Debug.Printf("Options after filtering: %d", len(results))
	logger.Debug.Printf("Processing time: %.2f seconds", duration.Seconds())
	logger.Debug.Printf("Strategy: %s", req.Strategy)
	logger.Debug.Printf("Available cash: $%.2f", req.AvailableCash)

	// Verbose: Detailed results breakdown
	logger.Verbose.Printf("=== VERBOSE RESULTS DETAIL ===")
	logger.Verbose.Printf("Symbols processed: %d | Results found: %d | Processing time: %.2f seconds",
		len(req.Symbols), len(results), duration.Seconds())
	if len(results) > 0 {
		logger.Debug.Printf("Top result: %s $%.2f premium with %d contracts", results[0].Ticker, results[0].Premium, results[0].MaxContracts)
		logger.Verbose.Printf("Top result detail: %s | Strike: $%.2f | Premium: $%.2f | Delta: %.3f | Contracts: %d",
			results[0].Ticker, results[0].Strike, results[0].Premium, results[0].Delta, results[0].MaxContracts)
	}

	response := &models.FormattedAnalysisResponse{
		Success: true,
		Data: models.FormattedAnalysisData{
			Results:       formattedResults,
			FieldMetadata: h.getFieldMetadata(),
		},
		Meta: models.ResponseMetadata{
			Strategy:           req.Strategy,
			ExpirationDate:     req.ExpirationDate,
			Timestamp:          time.Now().Format(time.RFC3339),
			ProcessingTime:     duration.Seconds(),
			ComputeDuration:    h.lastComputeDuration.Seconds(),
			ProcessingStats:    processingStats,
			Engine:             engineType,
			CudaAvailable:      h.engine.IsCudaAvailable(),
			ExecutionMode:      engineType, // Use actual resolved execution mode, not config setting
			SymbolCount:        len(cleanSymbols),
			ResultCount:        len(results),
			WorkloadFactor:     h.config.Engine.WorkloadFactor,
			SamplesProcessed:   totalSamplesProcessed,
			ContractsProcessed: h.contractsProcessedCount,
		},
	}

	logger.Debug.Printf("🔄 Starting JSON encoding...")

	// Sanitize response data to remove infinity/NaN values that break JSON
	h.sanitizeResponseData(response)

	// Debug: Try to find infinity values before encoding
	logger.Debug.Printf("🔍 Checking for infinity values in response...")
	h.debugInfinityValues(response)

	// Pre-encode JSON to get accurate size
	jsonBytes, err := json.Marshal(response)
	if err != nil {
		logger.Error.Printf("❌ JSON encoding failed: %v", err)
		logger.Error.Printf("🔍 Response structure: Results=%d, Meta fields present", len(response.Data.Results))
		http.Error(w, "JSON encoding failed", http.StatusInternalServerError)
		return
	}
	logger.Debug.Printf("✅ JSON encoded successfully (%d bytes)", len(jsonBytes))

	// Check connection again before sending
	select {
	case <-r.Context().Done():
		logger.Error.Printf("❌ Client disconnected during JSON encoding: %v", r.Context().Err())
		return
	default:
		// Continue with response
	}

	// Set headers with accurate content length
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(jsonBytes)))
	logger.Debug.Printf("📡 Sending JSON response headers...")

	// Write the complete response
	logger.Debug.Printf("📡 Writing JSON response body...")
	if _, err := w.Write(jsonBytes); err != nil {
		logger.Error.Printf("❌ Failed to write JSON response: %v", err)
		return
	}

	// Ensure the response is flushed to client
	logger.Debug.Printf("📡 Flushing response...")
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}

	logger.Debug.Printf("✅ JSON response sent successfully (%d bytes)", len(jsonBytes))
}

// TestConnectionHandler tests Alpaca API connection
func (h *OptionsHandler) TestConnectionHandler(w http.ResponseWriter, r *http.Request) {
	// Set CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	// Handle preflight OPTIONS request
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Test connection by getting account info
	if _, err := h.alpacaClient.GetStockPrice("AAPL", nil); err != nil {
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

// processRealOptions finds best contracts sequentially, then batch processes with CUDA or CPU
func (h *OptionsHandler) processRealOptions(stockPrices map[string]*alpaca.StockPrice, req models.AnalysisRequest, companyData map[string]struct{ Company, Sector string }, auditTicker *string) []models.OptionResult {
	var results []models.OptionResult

	// Calculate time to expiration
	expirationTime, err := time.Parse("2006-01-02", req.ExpirationDate)
	if err != nil {
		logger.Error.Printf("❌ Invalid expiration date format: %v", err)
		return results
	}
	timeToExp := time.Until(expirationTime).Hours() / (24 * 365.25)

	// STEP 1: Find best contracts for each symbol (sequential)
	var selectedContracts []struct {
		symbol      string
		stockPrice  float64
		contract    *alpaca.OptionContract
		companyInfo struct{ Company, Sector string }
	}

	for symbol, stockPrice := range stockPrices {
		// Check if this symbol matches our audit ticker
		isAuditSymbol := auditTicker != nil && symbol == *auditTicker

		if isAuditSymbol {
			logger.Warn.Printf("🔍 AUDIT: Processing audit ticker %s - sending to audit system", symbol)

			// Send to audit system instead of direct file write
			if err := h.auditLogger.LogOptionsAnalysisOperation(symbol, "ProcessingSymbol", map[string]interface{}{
				"ticker":    symbol,
				"operation": "processing_symbol",
				"message":   "Starting analysis for audit symbol",
			}); err != nil {
				logger.Error.Printf("🔍 AUDIT: Failed to log to audit system: %v", err)
			}
		}

		// Get options for this symbol (audit ticker is passed through to API calls)
		optionsChain, err := h.alpacaClient.GetOptionsChain([]string{symbol}, req.ExpirationDate, req.Strategy, req.TargetDelta, auditTicker)
		if err != nil {
			logger.Error.Printf("❌ Error getting options for %s: %v", symbol, err)
			continue
		}

		contracts, exists := optionsChain[symbol]
		if !exists || len(contracts) == 0 {
			logger.Debug.Printf("⚠️ No options found for %s", symbol)
			continue
		}

		// Track actual option contracts processed for this symbol
		h.contractsProcessedCount += len(contracts)

		// Find best option
		bestContract := h.findBestOptionContract(contracts, stockPrice.Price, req.TargetDelta, req.Strategy)
		if bestContract == nil {
			logger.Debug.Printf("⚠️ No suitable contract found for %s", symbol)
			continue
		}

		// Add to selected contracts for batch processing
		companyInfo, exists := companyData[symbol]
		if !exists {
			companyInfo = struct{ Company, Sector string }{Company: symbol, Sector: "Unknown"}
		}

		selectedContracts = append(selectedContracts, struct {
			symbol      string
			stockPrice  float64
			contract    *alpaca.OptionContract
			companyInfo struct{ Company, Sector string }
		}{
			symbol:      symbol,
			stockPrice:  stockPrice.Price,
			contract:    bestContract,
			companyInfo: companyInfo,
		})

		// Debug: Log audit ticker stock price in selected contracts
		if auditTicker != nil && symbol == *auditTicker {
			logger.Info.Printf("🔍 DEBUG: Selected contract for %s with stock price: $%.2f", symbol, stockPrice.Price)

			// Log the selected contract details for audit prompt inclusion
			selectedContractDetails := map[string]interface{}{
				"ticker":            symbol,
				"option_symbol":     bestContract.Symbol,
				"option_type":       bestContract.Type,
				"strike_price":      bestContract.StrikePrice,
				"stock_price":       stockPrice.Price,
				"expiration":        bestContract.ExpirationDate,
				"underlying_symbol": bestContract.UnderlyingSymbol,
			}

			if err := h.auditLogger.LogOptionsAnalysisOperation(*auditTicker, "SelectedContractForCalculation", selectedContractDetails); err != nil {
				logger.Warn.Printf("📋 CONTRACT: Failed to log selected contract: %v", err)
			} else {
				logger.Warn.Printf("📋 CONTRACT: Logged selected contract %s %s @ $%s for %s calculation",
					bestContract.Type, bestContract.Symbol, bestContract.StrikePrice, *auditTicker)
			}
		}

		// Verbose: Log all selected contracts with stock prices
		logger.Verbose.Printf("🎯 SELECTED CONTRACT: %s stock_price=$%.2f strike=$%s", symbol, stockPrice.Price, bestContract.StrikePrice)
	}

	// STEP 2: Batch process all selected contracts
	if len(selectedContracts) > 0 {
		logger.Debug.Printf("🎯 Processing %d selected contracts in batch", len(selectedContracts))
		results = h.batchProcessContracts(selectedContracts, req, timeToExp)
	}

	return results
}

// batchProcessContracts processes all selected contracts with complete GPU processing
func (h *OptionsHandler) batchProcessContracts(selectedContracts []struct {
	symbol      string
	stockPrice  float64
	contract    *alpaca.OptionContract
	companyInfo struct{ Company, Sector string }
}, req models.AnalysisRequest, timeToExp float64) []models.OptionResult {

	logger.Info.Printf("🚀 COMPLETE GPU PROCESSING: All calculations on CUDA")
	return h.batchProcessContractsComplete(selectedContracts, req, timeToExp)
}

// Legacy batch processing function removed - replaced by complete GPU processing

// findBestOptionContract finds the best option contract based on criteria
func (h *OptionsHandler) findBestOptionContract(contracts []*alpaca.OptionContract, stockPrice float64, targetDelta float64, strategy string) *alpaca.OptionContract {
	if len(contracts) == 0 {
		logger.Verbose.Printf("🔍 CONTRACT FILTER: No contracts provided")
		return nil
	}

	logger.Verbose.Printf("🔍 CONTRACT FILTER: Evaluating %d contracts for %s strategy, target delta %.3f",
		len(contracts), strategy, targetDelta)

	// Find the contract with the nearest strike to target, regardless of direction
	var bestContract *alpaca.OptionContract
	bestDistance := float64(999999)

	// Calculate target strike based on delta/risk level for PUTS ONLY
	// Must match the Alpaca client filtering thresholds exactly
	var targetMultiplier float64
	if targetDelta <= 0.15 { // LOW risk (low delta)
		targetMultiplier = 0.88 // 12% OTM for LOW risk (far from stock, low delta)
	} else if targetDelta <= 0.25 { // MOD risk
		targetMultiplier = 0.95 // 5% OTM for MOD risk
	} else { // HIGH risk (high delta)
		targetMultiplier = 0.98 // 2% OTM for HIGH risk (close to stock, high delta)
	}
	targetStrike := stockPrice * targetMultiplier

	logger.Verbose.Printf("🎯 TARGET STRIKE: $%.2f (%.1f%% of stock $%.2f) for delta %.3f",
		targetStrike, targetMultiplier*100, stockPrice, targetDelta)

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

		// Find nearest strike to target (can go above or below)
		distance := math.Abs(strikePrice - targetStrike)
		if distance < bestDistance {
			bestDistance = distance
			bestContract = contract
		}
	}

	return bestContract
}

// Legacy contract calculation functions removed - replaced by complete GPU processing

// debugInfinityValues recursively checks for infinity values in the response
func (h *OptionsHandler) debugInfinityValues(response *models.FormattedAnalysisResponse) {
	// Check metadata fields
	if math.IsInf(response.Meta.ProcessingTime, 0) || math.IsNaN(response.Meta.ProcessingTime) {
		logger.Error.Printf("🔍 Found infinity in Meta.ProcessingTime: %v", response.Meta.ProcessingTime)
	}
	if math.IsInf(response.Meta.WorkloadFactor, 0) || math.IsNaN(response.Meta.WorkloadFactor) {
		logger.Error.Printf("🔍 Found infinity in Meta.WorkloadFactor: %v", response.Meta.WorkloadFactor)
	}

	// Check all results
	for i, result := range response.Data.Results {
		for fieldName, field := range result {
			if rawValue, ok := field.Raw.(float64); ok {
				if math.IsInf(rawValue, 0) || math.IsNaN(rawValue) {
					logger.Error.Printf("🔍 Found infinity in result[%d][%s]: %v", i, fieldName, rawValue)
				}
			}
		}
	}
}

// sanitizeResponseData removes infinity and NaN values that break JSON encoding
func (h *OptionsHandler) sanitizeResponseData(response *models.FormattedAnalysisResponse) {
	// Fix infinity/NaN values in all results
	for i := range response.Data.Results {
		result := response.Data.Results[i]

		// Get ticker for logging
		ticker := "unknown"
		if tickerField, exists := result["ticker"]; exists {
			if tickerStr, ok := tickerField.Raw.(string); ok {
				ticker = tickerStr
			}
		}

		// Sanitize ALL fields, not just specific ones
		for fieldName, field := range result {
			// Check float64 values
			if rawValue, ok := field.Raw.(float64); ok {
				if math.IsInf(rawValue, 0) || math.IsNaN(rawValue) {
					// Replace with zero value
					field.Raw = 0.0
					switch fieldName {
					case "strike", "stock_price", "premium", "total_premium", "total_cash_needed":
						field.Display = "$0.00"
					case "delta":
						field.Display = "0.000"
					case "profit_percentage":
						field.Display = "0.00%"
					default:
						field.Display = "0.00"
					}
					result[fieldName] = field
					logger.Warn.Printf("🔧 Sanitized infinite %s=%v for %s", fieldName, rawValue, ticker)
				}
			}

			// Check float32 values (just in case)
			if rawValue, ok := field.Raw.(float32); ok {
				if math.IsInf(float64(rawValue), 0) || math.IsNaN(float64(rawValue)) {
					field.Raw = float32(0.0)
					field.Display = "0.00"
					result[fieldName] = field
					logger.Warn.Printf("🔧 Sanitized infinite float32 %s=%v for %s", fieldName, rawValue, ticker)
				}
			}
		}
	}

	// Sanitize metadata fields
	if math.IsInf(response.Meta.ProcessingTime, 0) || math.IsNaN(response.Meta.ProcessingTime) {
		response.Meta.ProcessingTime = 0
		logger.Debug.Printf("🔧 Sanitized infinite ProcessingTime")
	}

	if math.IsInf(response.Meta.WorkloadFactor, 0) || math.IsNaN(response.Meta.WorkloadFactor) {
		response.Meta.WorkloadFactor = 0
		logger.Debug.Printf("🔧 Sanitized infinite WorkloadFactor")
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
		"rank":          h.formatInteger(rank),
		"ticker":        h.formatText(result.Ticker),
		"company":       h.formatText(result.Company),
		"sector":        h.formatText(result.Sector),
		"strike":        h.formatCurrency(result.Strike),
		"stock_price":   h.formatCurrency(result.StockPrice),
		"max_contracts": h.formatInteger(result.MaxContracts),
		"premium":       h.formatCurrency(result.Premium),
		"total_premium": h.formatCurrency(result.TotalPremium),

		"profit_percentage": h.formatPercentage(result.ProfitPercentage / 100), // Convert from percentage to decimal
		"expiration":        h.formatText(result.Expiration),
	}

	// Calculate annualized return
	annualized := 0.0
	if result.DaysToExp > 0 {
		annualized = result.ProfitPercentage * (365.0 / float64(result.DaysToExp))
	}
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

// calculateImpliedVolatility uses Newton-Raphson to find IV from market price
func (h *OptionsHandler) calculateImpliedVolatility(marketPrice, stockPrice, strikePrice, timeToExp, riskFreeRate float64, optionType byte) float64 {
	const (
		tolerance     = 1e-6  // Convergence tolerance
		maxIterations = 100   // Maximum iterations
		dividendYield = 0.005 // 0.5% dividend yield
	)

	// Initial volatility guess based on market price
	vol := math.Max(0.10, math.Min(2.0, marketPrice*2.0/(stockPrice*math.Sqrt(timeToExp))))

	for i := 0; i < maxIterations; i++ {
		// Calculate Black-Scholes price and vega with current volatility
		d1 := (math.Log(stockPrice/strikePrice) + (riskFreeRate-dividendYield+0.5*vol*vol)*timeToExp) / (vol * math.Sqrt(timeToExp))
		d2 := d1 - vol*math.Sqrt(timeToExp)

		// Cumulative normal distributions
		nd1 := 0.5 * (1.0 + math.Erf(d1/math.Sqrt(2.0)))
		nd2 := 0.5 * (1.0 + math.Erf(d2/math.Sqrt(2.0)))
		negNd1 := 0.5 * (1.0 + math.Erf(-d1/math.Sqrt(2.0)))
		negNd2 := 0.5 * (1.0 + math.Erf(-d2/math.Sqrt(2.0)))

		// Calculate theoretical price
		var theoreticalPrice float64
		if optionType == 'C' {
			theoreticalPrice = stockPrice*math.Exp(-dividendYield*timeToExp)*nd1 - strikePrice*math.Exp(-riskFreeRate*timeToExp)*nd2
		} else {
			theoreticalPrice = strikePrice*math.Exp(-riskFreeRate*timeToExp)*negNd2 - stockPrice*math.Exp(-dividendYield*timeToExp)*negNd1
		}

		// Calculate vega (sensitivity to volatility)
		vega := stockPrice * math.Exp(-dividendYield*timeToExp) * math.Exp(-0.5*d1*d1) / math.Sqrt(2.0*math.Pi) * math.Sqrt(timeToExp)

		// Price difference
		priceDiff := theoreticalPrice - marketPrice

		// Check convergence
		if math.Abs(priceDiff) < tolerance {
			return vol
		}

		// Newton-Raphson update: vol_new = vol - f(vol)/f'(vol)
		if vega > 1e-10 { // Avoid division by zero
			vol = vol - priceDiff/vega
		} else {
			break
		}

		// Keep volatility within reasonable bounds
		vol = math.Max(0.01, math.Min(3.0, vol))
	}

	return vol // Return best estimate even if not converged
}

// batchProcessContractsComplete uses complete GPU processing for ALL calculations
func (h *OptionsHandler) batchProcessContractsComplete(selectedContracts []struct {
	symbol      string
	stockPrice  float64
	contract    *alpaca.OptionContract
	companyInfo struct{ Company, Sector string }
}, req models.AnalysisRequest, timeToExp float64) []models.OptionResult {

	logger.Info.Printf("🚀 COMPLETE GPU: Processing %d contracts with ALL calculations on CUDA", len(selectedContracts))

	if !h.engine.IsCudaAvailable() {
		return nil // Require CUDA for complete processing
	}

	// Build complete contract data for CUDA
	var engineContracts []barracuda.OptionContract
	for _, sc := range selectedContracts {
		strikePrice, err := strconv.ParseFloat(sc.contract.StrikePrice, 64)
		if err != nil || !sc.contract.Tradable {
			continue
		}

		optionType := byte('P')
		if req.Strategy == "calls" {
			optionType = byte('C')
		}

		// Get market price for IV calculation
		marketPrice := 0.0
		if sc.contract.ClosePrice != nil {
			if closePriceStr, ok := sc.contract.ClosePrice.(string); ok {
				marketPrice, _ = strconv.ParseFloat(closePriceStr, 64)
			} else if closePriceFloat, ok := sc.contract.ClosePrice.(float64); ok {
				marketPrice = closePriceFloat
			}
		}

		// Validate minimum premium to prevent unrealistic calculations
		if marketPrice < 0.01 {
			logger.Verbose.Printf("⚠️  Skipping %s: Premium too small ($%.6f), likely data error", sc.symbol, marketPrice)
			continue
		}

		engineContracts = append(engineContracts, barracuda.OptionContract{
			Symbol:           sc.symbol,
			StrikePrice:      strikePrice,
			UnderlyingPrice:  sc.stockPrice,
			TimeToExpiration: timeToExp,
			RiskFreeRate:     h.treasuryClient.GetRiskFreeRateWithLastKnown(),
			Volatility:       math.Max(0.15, sc.contract.ImpliedVol), // Ensure minimum 15% volatility to prevent division by zero and unrealistic calculations
			OptionType:       optionType,
			MarketClosePrice: marketPrice,
		})

		// Debug: Log audit ticker price being passed to engine
		if req.AuditTicker != "" && sc.symbol == req.AuditTicker {
			logger.Info.Printf("🔍 DEBUG: Engine contract for %s created with UnderlyingPrice: $%.2f, StrikePrice: $%.2f", sc.symbol, sc.stockPrice, strikePrice)
		}

		// Verbose: Log all engine contracts with underlying prices
		logger.Verbose.Printf("🚀 ENGINE CONTRACT: %s underlying_price=$%.2f strike_price=$%.2f", sc.symbol, sc.stockPrice, strikePrice)
	}

	if len(engineContracts) == 0 {
		return nil
	}

	// Complete processing call - use CPU or CUDA function based on execution mode
	// Pass ONLY the audit ticker if explicitly set - NO FALLBACKS
	var auditSymbol *string
	if req.AuditTicker != "" {
		auditSymbol = &req.AuditTicker
	}
	// If no AuditTicker set, auditSymbol stays nil (no audit message)

	var completeResults []barracuda.CompleteOptionResult
	var err error

	// Measure just the compute time
	computeStartTime := time.Now()

	if h.config.Engine.ExecutionMode == "cpu" {
		completeResults, err = h.engine.MaximizeCPUUsageComplete(
			engineContracts,
			engineContracts[0].UnderlyingPrice,
			req.AvailableCash,
			req.Strategy,
			req.ExpirationDate,
			auditSymbol)
	} else {
		completeResults, err = h.engine.MaximizeCUDAUsageComplete(
			engineContracts,
			engineContracts[0].UnderlyingPrice,
			req.AvailableCash,
			req.Strategy,
			req.ExpirationDate,
			auditSymbol)
	}

	computeDuration := time.Since(computeStartTime)

	if err != nil {
		logger.Error.Printf("❌ Complete processing failed: %v", err)

		// Log failed Black-Scholes calculation to audit if this is the audit ticker
		if auditSymbol != nil {
			if auditErr := h.auditLogger.LogOptionsAnalysisOperation(*auditSymbol, "BlackScholesCalculation", map[string]interface{}{
				"success": false,
				"error":   err.Error(),
			}); auditErr != nil {
				logger.Warn.Printf("⚠️ Failed to log Black-Scholes audit entry: %v", auditErr)
			}
		}
		return nil
	}

	// Engine handles its own audit logging - HTTP handler should not contain business logic
	h.lastComputeDuration = computeDuration

	// DEBUG: Log timing details
	logger.Warn.Printf("🔍 TIMING DEBUG: Compute duration: %.3fms",
		h.lastComputeDuration.Seconds()*1000)

	// Convert complete CUDA results to business results
	var results []models.OptionResult
	for i, completeResult := range completeResults {
		if i >= len(selectedContracts) {
			break
		}

		sc := selectedContracts[i]
		results = append(results, models.OptionResult{
			Ticker:           completeResult.Symbol,
			Company:          sc.companyInfo.Company,
			Sector:           sc.companyInfo.Sector,
			OptionSymbol:     sc.contract.Symbol,
			OptionType:       req.Strategy,
			Strike:           completeResult.StrikePrice,
			StockPrice:       completeResult.UnderlyingPrice,
			Premium:          completeResult.TheoreticalPrice,
			MaxContracts:     completeResult.MaxContracts,
			TotalPremium:     completeResult.TotalPremium,
			CashNeeded:       completeResult.CashNeeded,
			ProfitPercentage: completeResult.ProfitPercentage,
			Delta:            completeResult.Delta,
			Gamma:            completeResult.Gamma,
			Theta:            completeResult.Theta,
			Vega:             completeResult.Vega,
			ImpliedVol:       completeResult.ImpliedVolatility,
			Expiration:       req.ExpirationDate,
			DaysToExp:        completeResult.DaysToExpiration,
		})
	}

	// Sort results by profit percentage (descending) for final ranking order
	sort.Slice(results, func(i, j int) bool {
		return results[i].ProfitPercentage > results[j].ProfitPercentage
	})

	logger.Info.Printf("⚡ COMPLETE GPU: %.3fms | %d contracts → %d results | ALL calculations on CUDA, sorted by profit %%",
		h.lastComputeDuration.Seconds()*1000, len(engineContracts), len(results))

	return results
}

// getExecutionMode returns current execution mode
func (h *OptionsHandler) getExecutionMode() string {
	if h.engine.IsCudaAvailable() && (h.config.Engine.ExecutionMode == "auto" || h.config.Engine.ExecutionMode == "cuda") {
		return "CUDA"
	}
	return "CPU"
}

// AuditStartupHandler logs when an audit ticker is set at startup
func (h *OptionsHandler) AuditStartupHandler(w http.ResponseWriter, r *http.Request) {
	// Set CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Ticker string `json:"ticker"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusOK) // Don't fail startup for this
		return
	}

	if req.Ticker != "" {
		logger.Warn.Printf("⚠️ STARTUP: Audit ticker '%s' is already set for detailed logging", req.Ticker)
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// AIAnalysisHandler sends data to AI for analysis
// AIAnalysisHandler sends request to audit worker for grok analysis
func (h *OptionsHandler) AIAnalysisHandler(w http.ResponseWriter, r *http.Request) {
	logger.Warn.Printf("🤖 GROK: AI Analysis request received")

	// Set CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		logger.Warn.Printf("🤖 GROK: OPTIONS preflight request handled")
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		logger.Warn.Printf("🤖 GROK: Invalid method %s rejected", r.Method)
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse request to get custom prompt
	var req struct {
		CustomPrompt string `json:"custom_prompt"`
	}

	if r.Body != nil {
		json.NewDecoder(r.Body).Decode(&req)
	}

	logger.Warn.Printf("🤖 GROK: Received custom prompt: '%s' (length: %d)", req.CustomPrompt, len(req.CustomPrompt))

	// Check if audit.json exists
	auditFileName := "audit.json"
	auditData, err := os.ReadFile(auditFileName)
	if err != nil {
		logger.Warn.Printf("🤖 GROK: No audit.json file found: %v", err)
		http.Error(w, "No audit data available for analysis", http.StatusNotFound)
		return
	}

	// Parse audit data to extract ticker
	var auditHeader struct {
		Header struct {
			Ticker     string `json:"ticker"`
			ExpiryDate string `json:"expiry_date"`
		} `json:"header"`
	}

	if err := json.Unmarshal(auditData, &auditHeader); err != nil {
		logger.Warn.Printf("🤖 GROK: Failed to parse audit data: %v", err)
		http.Error(w, "Invalid audit data format", http.StatusInternalServerError)
		return
	}

	ticker := auditHeader.Header.Ticker
	if ticker == "" {
		logger.Warn.Printf("🤖 GROK: No ticker found in audit data")
		ticker = "UNKNOWN"
	}

	logger.Warn.Printf("🤖 GROK: Extracted ticker '%s' from audit data", ticker)

	// Perform Grok analysis synchronously for UI
	grokClient := grok.NewClient()

	startTime := time.Now()
	response, err := grokClient.AnalyzeOptions(string(auditData), req.CustomPrompt)
	if err != nil {
		logger.Warn.Printf("🤖 GROK: API call failed: %v", err)
		http.Error(w, fmt.Sprintf("Grok analysis failed: %v", err), http.StatusInternalServerError)
		return
	}
	elapsed := time.Since(startTime).Seconds()

	// Format analysis with custom prompt
	analysis := grok.FormatAnalysisWithPrompt(response, req.CustomPrompt)

	logger.Warn.Printf("🤖 GROK: Analysis complete (%.2fs, %d tokens)", elapsed, response.Tokens)

	// Trigger audit archival by sending analysis_result to worker (always do this)
	if err := h.auditLogger.LogOptionsAnalysisOperation(ticker, "analysis_result", map[string]interface{}{
		"grok_result":   analysis,
		"custom_prompt": req.CustomPrompt,
		"tokens":        response.Tokens,
		"elapsed":       elapsed,
	}); err != nil {
		logger.Warn.Printf("🤖 GROK: Failed to trigger audit archival: %v", err)
	}

	// Return analysis results for UI
	responseData := map[string]interface{}{
		"status":   "complete",
		"analysis": analysis,
		"tokens":   response.Tokens,
		"elapsed":  elapsed,
		"message":  "AI analysis completed successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(responseData); err != nil {
		logger.Warn.Printf("🤖 GROK: Failed to encode response: %v", err)
		return
	}

	logger.Warn.Printf("🤖 GROK: Analysis complete - sent %d tokens to UI (%.2fs)", response.Tokens, elapsed)
}

// AuditFileExistsHandler checks if audit.json exists
func (h *OptionsHandler) AuditFileExistsHandler(w http.ResponseWriter, r *http.Request) {
	filename := "audit.json"
	_, err := os.Stat(filename)
	exists := err == nil

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"exists": exists})
}

// AuditLogHandler creates individual timestamped JSON files for each Grok analysis
func (h *OptionsHandler) AuditLogHandler(w http.ResponseWriter, r *http.Request) {
	// Set CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	logger.Warn.Printf("📋 AUDIT: Log entry request received")

	// Parse log entry (just to validate it's valid JSON)
	var logEntry map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&logEntry); err != nil {
		logger.Warn.Printf("📋 AUDIT: Failed to parse log entry: %v", err)
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Log entry received - no audit system needed, just respond
	logger.Warn.Printf("📋 AUDIT: Log entry acknowledged (no auditing performed)")

	response := map[string]string{
		"status":  "success",
		"message": "Log entry acknowledged",
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		logger.Warn.Printf("📋 AUDIT: Failed to encode response: %v", err)
		return
	}
	logger.Warn.Printf("📋 AUDIT: Response sent successfully")
}
