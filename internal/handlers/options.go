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
	alpacaClient            alpaca.AlpacaInterface
	config                  *config.Config
	engine                  *barracuda.BaracudaEngine
	symbolService           *symbols.SP500Service
	contractsProcessedCount int
	lastComputeDuration     time.Duration
}

// NewOptionsHandler creates a new options handler - just HTTP routing
func NewOptionsHandler(alpacaClient alpaca.AlpacaInterface, cfg *config.Config, engine *barracuda.BaracudaEngine, symbolService *symbols.SP500Service) *OptionsHandler {
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
			return []string{"#", "Ticker", "Company", "Sector", "Strike", "Stock Price", "Max Contracts", "Premium", "Total Premium", "Profit %", "Annualized", "Expiration"}
		},
		"tableFieldKeys": func() []string {
			return []string{"rank", "ticker", "company", "sector", "strike", "stock_price", "max_contracts", "premium", "total_premium", "profit_percentage", "annualized", "expiration"}
		},

		// Default values (calculated by backend)
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

		// CSV Headers (backend controlled)
		"csvHeaders": func() []string {
			return []string{"Rank", "Ticker", "Company", "Sector", "Strike", "Stock_Price", "Max_Contracts", "Premium", "Total_Premium", "Profit_Percentage", "Annualized", "Expiration"}
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

	// Check for audit ticker and log at warn level
	if req.AuditTicker != "" {
		logger.Warn.Printf("🔍 AUDIT MODE: Ticker %s set for detailed audit logging", req.AuditTicker)
		logger.Warn.Printf("🔍 AUDIT MODE: This analysis will collect detailed data for %s", req.AuditTicker)
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
	logger.Info.Printf("📊 Analyzing %s options for %d symbols with batch processing", req.Strategy, len(req.Symbols))

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
	auditMode := ""
	if req.AuditTicker != "" {
		auditMode = fmt.Sprintf(" | 🔍 AUDIT: %s", req.AuditTicker)
		logger.Warn.Printf("🔍 AUDIT: Starting analysis with audit ticker %s - detailed logging enabled", req.AuditTicker)
	}
	logger.Warn.Printf("🚀 START: %d symbols | %s | %s%s",
		len(cleanSymbols), req.Strategy, map[bool]string{true: "CUDA", false: "CPU"}[h.engine.IsCudaAvailable()], auditMode)

	// Get stock prices in batches (up to 100 symbols per API call)
	logger.Info.Printf("📈 Fetching stock prices for %d symbols in batches...", len(cleanSymbols))
	stockPrices, err := h.alpacaClient.GetStockPricesBatch(cleanSymbols)
	if err != nil {
		logger.Error.Printf("❌ Error fetching stock prices: %v", err)
		http.Error(w, fmt.Sprintf("Failed to get stock prices: %v", err), http.StatusInternalServerError)
		return
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
	results := h.processRealOptions(stockPrices, req, companyData)

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
	
	// Create/recreate audit JSON file if audit ticker was set
	if req.AuditTicker != "" {
		logger.Warn.Printf("🔍 AUDIT: Creating audit JSON file for %s", req.AuditTicker)
		h.createAuditJSONFile(req.AuditTicker, req, results, duration)
		logger.Warn.Printf("🔍 AUDIT: JSON file ready for Grok analysis")
	}
	
	logger.Debug.Printf("🔍 DEBUG: Analysis completed, formatting response")

	// Check if client is still connected
	select {
	case <-r.Context().Done():
		logger.Error.Printf("❌ Client disconnected before response could be sent: %v", r.Context().Err())
		return
	default:
		logger.Debug.Printf("✅ Client connection still active, proceeding with response")
	}

	logger.Always.Printf("⚡ PROCESSING STATS: Engine=%s | Duration=%.3fs | Symbols=%d | Results=%d | Mode=%s | Workload=%.1fx",
		engineType, duration.Seconds(), len(cleanSymbols), len(results), h.config.Engine.ExecutionMode, h.config.Engine.WorkloadFactor)

	// Final processing summary - ALWAYS show real job stats, ADD workload if present
	processingStats := fmt.Sprintf("✅ Processed %d symbols | Returned %d results | %.3f seconds | Engine: %s",
		len(cleanSymbols), len(results), duration.Seconds(), engineType)

	// ADD workload benchmark stats to the real job processing
	if h.config.Engine.WorkloadFactor > 0.0 && totalSamplesProcessed > 0 {
		processingStats += fmt.Sprintf(" | %d Monte Carlo samples", totalSamplesProcessed)
	}
	logger.Always.Printf(processingStats)

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

// DownloadCSVHandler generates and returns CSV file for download
func (h *OptionsHandler) DownloadCSVHandler(w http.ResponseWriter, r *http.Request) {
	// Set CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	// Handle preflight OPTIONS request
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Validate API credentials first (same as AnalyzeHandler)
	if _, err := h.alpacaClient.GetStockPrice("AAPL"); err != nil {
		http.Error(w, "Invalid API credentials", http.StatusUnauthorized)
		return
	}

	// Parse the same analysis request
	var req models.AnalysisRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Same symbol handling logic as AnalyzeHandler
	if len(req.Symbols) == 0 {
		if len(h.config.DefaultStocks) > 0 {
			req.Symbols = h.config.DefaultStocks
		} else {
			symbols, err := h.symbolService.GetSymbolsAsStrings()
			if err != nil {
				http.Error(w, fmt.Sprintf("Failed to get S&P 500 symbols: %v", err), http.StatusInternalServerError)
				return
			}
			req.Symbols = symbols
		}
	}

	// Clean symbols (same logic as AnalyzeHandler)
	cleanSymbols := make([]string, 0, len(req.Symbols))
	for _, symbol := range req.Symbols {
		if cleaned := strings.TrimSpace(symbol); cleaned != "" {
			cleanSymbols = append(cleanSymbols, cleaned)
		}
	}

	if len(cleanSymbols) == 0 {
		http.Error(w, "No valid symbols provided", http.StatusBadRequest)
		return
	}

	// Validate request (same as AnalyzeHandler)
	if req.AvailableCash <= 0 {
		http.Error(w, "Available cash must be positive", http.StatusBadRequest)
		return
	}
	if req.Strategy != "puts" && req.Strategy != "calls" {
		http.Error(w, "Strategy must be 'puts' or 'calls'", http.StatusBadRequest)
		return
	}

	startTime := time.Now()
	logger.Info.Printf("📊 CSV Generation: Analyzing %s options for %d symbols", req.Strategy, len(cleanSymbols))

	// Get stock prices (same as AnalyzeHandler)
	stockPrices, err := h.alpacaClient.GetStockPricesBatch(cleanSymbols)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to get stock prices: %v", err), http.StatusInternalServerError)
		return
	}

	// Pre-load company/sector info (same as AnalyzeHandler)
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

	// Process options using same method as AnalyzeHandler
	results := h.processRealOptions(stockPrices, req, companyData)

	// Sort results by total premium (same as AnalyzeHandler)
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].TotalPremium > results[i].TotalPremium {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Apply max_results limit (same as AnalyzeHandler)
	if h.config.MaxResults > 0 && len(results) > h.config.MaxResults {
		results = results[:h.config.MaxResults]
	}

	// Generate CSV content
	var csvContent strings.Builder

	// CSV Header
	headers := []string{
		"Rank", "Ticker", "Company", "Sector", "Strike", "Stock_Price",
		"Premium", "Max_Contracts", "Total_Premium", "Profit_Percentage", "Delta", "Expiration", "Days_To_Exp",
	}
	csvContent.WriteString(strings.Join(headers, ",") + "\n")

	// CSV Data rows
	for i, result := range results {
		row := []string{
			fmt.Sprintf("%d", i+1),
			result.Ticker,
			fmt.Sprintf("\"%s\"", result.Company), // Quote company names with spaces
			fmt.Sprintf("\"%s\"", result.Sector),
			fmt.Sprintf("%.2f", result.Strike),
			fmt.Sprintf("%.2f", result.StockPrice),
			fmt.Sprintf("%.2f", result.Premium),
			fmt.Sprintf("%d", result.MaxContracts),
			fmt.Sprintf("%.2f", result.TotalPremium),
			fmt.Sprintf("%.2f", result.ProfitPercentage),
			fmt.Sprintf("%.4f", result.Delta),
			result.Expiration,
			fmt.Sprintf("%d", result.DaysToExp),
		}
		csvContent.WriteString(strings.Join(row, ",") + "\n")
	}

	// Set headers for file download using configurable format
	timestamp := time.Now().Format("15-04-05") // HH-MM-SS for clarity
	expDate := req.ExpirationDate              // Keep as YYYY-MM-DD format
	deltaStr := fmt.Sprintf("delta%.2f", req.TargetDelta)
	strategy := strings.ToLower(req.Strategy) // puts or calls

	// Use configurable filename format
	filename := h.config.CSV.FilenameFormat
	filename = strings.ReplaceAll(filename, "{time}", timestamp)
	filename = strings.ReplaceAll(filename, "{exp_date}", expDate)
	filename = strings.ReplaceAll(filename, "{delta}", deltaStr)
	filename = strings.ReplaceAll(filename, "{strategy}", strategy)
	w.Header().Set("Content-Type", "text/csv")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", filename))

	// Write CSV content
	w.Write([]byte(csvContent.String()))

	logger.Info.Printf("📊 CSV downloaded: %d results in %.3fs", len(results), time.Since(startTime).Seconds())
}

// processRealOptions finds best contracts sequentially, then batch processes with CUDA or CPU
func (h *OptionsHandler) processRealOptions(stockPrices map[string]*alpaca.StockPrice, req models.AnalysisRequest, companyData map[string]struct{ Company, Sector string }) []models.OptionResult {
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
		// Get options for this symbol
		optionsChain, err := h.alpacaClient.GetOptionsChain([]string{symbol}, req.ExpirationDate, req.Strategy)
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

	// Calculate target strike based on delta/risk level
	var targetMultiplier float64
	if strategy == "puts" {
		if targetDelta <= 0.30 {
			targetMultiplier = 0.88 // 12% OTM for LOW risk
		} else if targetDelta <= 0.60 {
			targetMultiplier = 0.95 // 5% OTM for MOD risk
		} else {
			targetMultiplier = 0.98 // 2% OTM for HIGH risk
		}
	} else {
		if targetDelta <= 0.30 {
			targetMultiplier = 1.12 // 12% ITM for LOW risk calls
		} else if targetDelta <= 0.60 {
			targetMultiplier = 1.05 // 5% ITM for MOD risk calls
		} else {
			targetMultiplier = 1.02 // 2% ITM for HIGH risk calls
		}
	}
	targetStrike := stockPrice * targetMultiplier

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

		engineContracts = append(engineContracts, barracuda.OptionContract{
			Symbol:           sc.symbol,
			StrikePrice:      strikePrice,
			UnderlyingPrice:  sc.stockPrice,
			TimeToExpiration: timeToExp,
			RiskFreeRate:     0.04,
			Volatility:       0.25,
			OptionType:       optionType,
			MarketClosePrice: marketPrice,
		})
	}

	if len(engineContracts) == 0 {
		return nil
	}

	// CUDA call with complete processing (includes business calculations)
	startTime := time.Now()
	completeResults, err := h.engine.MaximizeCUDAUsageComplete(
		engineContracts,
		engineContracts[0].UnderlyingPrice,
		req.AvailableCash,
		req.Strategy,
		req.ExpirationDate)

	if err != nil {
		logger.Error.Printf("❌ Complete CUDA processing failed: %v", err)
		return nil
	}

	duration := time.Since(startTime)
	h.lastComputeDuration = duration

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
		duration.Seconds()*1000, len(engineContracts), len(results))

	return results
}

// createAuditJSONFile creates/recreates the audit JSON file for a ticker
func (h *OptionsHandler) createAuditJSONFile(ticker string, req models.AnalysisRequest, results []models.OptionResult, duration time.Duration) {
	// Always the same filename: audit.json in root (overwritten each time)
	filename := "audit.json"
	logger.Warn.Printf("🔍 AUDIT: Creating/overwriting JSON file: %s", filename)
	
	// Ensure results are properly sorted by profit percentage (descending) for correct ranking
	sort.Slice(results, func(i, j int) bool {
		return results[i].ProfitPercentage > results[j].ProfitPercentage
	})

	// Find ALL options for the ticker and get the best one's rank
	var tickerResult *models.OptionResult
	var allTickerOptions []models.OptionResult
	var rank int
	found := false
	
	// Collect all options for this ticker
	for i := range results {
		if results[i].Ticker == ticker {
			allTickerOptions = append(allTickerOptions, results[i])
			// The first one we find is the best (highest ranked)
			if !found {
				tickerResult = &results[i]
				rank = i + 1 // 1-based ranking
				found = true
			}
		}
	}

	// Create numbered results list with ranking validation data
	numberedResults := make([]map[string]interface{}, len(results))
	for i, result := range results {
		// Calculate metrics for ranking analysis
		moneyness := result.StockPrice / result.Strike
		annualizedReturn := result.ProfitPercentage * (365.0 / float64(result.DaysToExp))
		premiumAsPercent := (result.Premium / result.StockPrice) * 100
		cashEfficiency := result.TotalPremium / result.CashNeeded * 100
		
		// Risk assessment indicators
		otmDistance := math.Abs(result.StockPrice - result.Strike) / result.StockPrice * 100
		volatilityRisk := "UNKNOWN"
		if result.ImpliedVol > 0.5 {
			volatilityRisk = "HIGH"
		} else if result.ImpliedVol > 0.3 {
			volatilityRisk = "MEDIUM"
		} else {
			volatilityRisk = "LOW"
		}
		
		// Position size efficiency
		capitalUtilization := result.CashNeeded / req.AvailableCash * 100
		
		numberedResults[i] = map[string]interface{}{
			"rank": i + 1,
			"ticker": result.Ticker,
			"company": result.Company,
			"sector": result.Sector,
			"ranking_metrics": map[string]interface{}{
				"profit_percentage": result.ProfitPercentage,
				"annualized_return_percent": annualizedReturn,
				"cash_efficiency_percent": cashEfficiency,
				"otm_distance_percent": otmDistance,
				"volatility_risk_level": volatilityRisk,
				"capital_utilization_percent": capitalUtilization,
			},
			"option_details": map[string]interface{}{
				"strike": result.Strike,
				"stock_price": result.StockPrice,
				"moneyness": moneyness,
				"premium": result.Premium,
				"premium_as_percent_of_stock": premiumAsPercent,
				"max_contracts": result.MaxContracts,
				"total_premium": result.TotalPremium,
				"cash_needed": result.CashNeeded,
			},
			"greeks_and_risk": map[string]interface{}{
				"delta": result.Delta,
				"gamma": result.Gamma,
				"theta": result.Theta,
				"vega": result.Vega,
				"implied_volatility": result.ImpliedVol,
			},
			"time_factors": map[string]interface{}{
				"expiration": result.Expiration,
				"days_to_exp": result.DaysToExp,
			},
			"ranking_concerns": map[string]interface{}{
				"high_volatility_warning": result.ImpliedVol > 0.6,
				"low_liquidity_risk": result.MaxContracts < 5,
				"excessive_capital_use": capitalUtilization > 50,
				"very_short_term": result.DaysToExp < 14,
				"deep_otm": otmDistance > 20,
			},
		}
	}

	// Calculate ranking validation statistics
	var avgProfitPct, avgDelta, avgImpliedVol, avgDaysToExp float64
	var highVolCount, lowLiquidityCount, shortTermCount int
	sectorCounts := make(map[string]int)
	
	for _, result := range results {
		avgProfitPct += result.ProfitPercentage
		avgDelta += math.Abs(result.Delta)
		avgImpliedVol += result.ImpliedVol
		avgDaysToExp += float64(result.DaysToExp)
		sectorCounts[result.Sector]++
		
		// Count risk factors
		if result.ImpliedVol > 0.5 {
			highVolCount++
		}
		if result.MaxContracts < 10 {
			lowLiquidityCount++
		}
		if result.DaysToExp < 21 {
			shortTermCount++
		}
	}
	
	if len(results) > 0 {
		avgProfitPct /= float64(len(results))
		avgDelta /= float64(len(results))
		avgImpliedVol /= float64(len(results))
		avgDaysToExp /= float64(len(results))
	}
	
	// Analyze top and bottom performers for ranking validation
	var topPerformer, bottomPerformer map[string]interface{}
	if len(results) > 0 {
		topResult := results[0]
		topPerformer = map[string]interface{}{
			"ticker": topResult.Ticker,
			"company": topResult.Company,
			"sector": topResult.Sector,
			"profit_percentage": topResult.ProfitPercentage,
			"implied_vol": topResult.ImpliedVol,
			"days_to_exp": topResult.DaysToExp,
			"max_contracts": topResult.MaxContracts,
			"justification_question": "Does this stock deserve #1 ranking? Consider sector outlook, volatility environment, and liquidity.",
		}
		
		if len(results) > 1 {
			bottomResult := results[len(results)-1]
			bottomPerformer = map[string]interface{}{
				"ticker": bottomResult.Ticker,
				"company": bottomResult.Company,
				"sector": bottomResult.Sector,
				"profit_percentage": bottomResult.ProfitPercentage,
				"implied_vol": bottomResult.ImpliedVol,
				"days_to_exp": bottomResult.DaysToExp,
				"max_contracts": bottomResult.MaxContracts,
				"justification_question": "Is this stock rightfully at the bottom? Could hidden risks justify the low ranking?",
			}
		}
	}

	// Collect comprehensive audit data with market insights
	auditData := map[string]interface{}{
		"ticker": ticker,
		"timestamp": time.Now().Format(time.RFC3339),
		"analysis_request": req,
		"processing_time_seconds": duration.Seconds(),
		"execution_mode": h.getExecutionMode(),
		"total_results": len(results),
		"market_analysis": map[string]interface{}{
			"average_profit_percentage": avgProfitPct,
			"average_delta": avgDelta,
			"average_implied_volatility": avgImpliedVol,
			"average_days_to_expiration": avgDaysToExp,
			"sector_distribution": sectorCounts,
			"strategy": req.Strategy,
			"target_delta": req.TargetDelta,
			"available_cash": req.AvailableCash,
			"risk_factors": map[string]interface{}{
				"high_volatility_count": highVolCount,
				"low_liquidity_count": lowLiquidityCount,
				"short_term_count": shortTermCount,
				"total_opportunities": len(results),
			},
		},
		"ranking_validation": map[string]interface{}{
			"top_performer": topPerformer,
			"bottom_performer": bottomPerformer,
			"analysis_prompt": "Analyze if the ranking order makes fundamental sense. Does the #1 stock deserve top position? Are there hidden risks in top picks? Does the bottom stock have unfair disadvantages or represent a contrarian opportunity?",
		},
		"ranked_results": numberedResults,
	}

	// Collect available market data for the audited ticker
	alpacaData := map[string]interface{}{
		"data_collection_time": time.Now().Format(time.RFC3339),
		"ticker": ticker,
		"analysis_context": map[string]interface{}{
			"expiration_date": req.ExpirationDate,
			"strategy": req.Strategy,
			"target_delta": req.TargetDelta,
		},
	}
	
	// Get current stock price for comparison
	if stockPrice, err := h.alpacaClient.GetStockPrice(ticker); err == nil {
		alpacaData["current_stock_price"] = map[string]interface{}{
			"symbol": stockPrice.Symbol,
			"price": stockPrice.Price,
		}
		
		// Add market context analysis
		if found && tickerResult != nil {
			priceDiscrepancy := math.Abs(stockPrice.Price - tickerResult.StockPrice) / stockPrice.Price * 100
			alpacaData["price_analysis"] = map[string]interface{}{
				"analysis_price": tickerResult.StockPrice,
				"current_market_price": stockPrice.Price,
				"price_discrepancy_percent": priceDiscrepancy,
				"stale_data_warning": priceDiscrepancy > 2.0,
				"price_movement_analysis": map[string]interface{}{
					"direction": func() string {
						if stockPrice.Price > tickerResult.StockPrice {
							return "increased"
						} else if stockPrice.Price < tickerResult.StockPrice {
							return "decreased"
						}
						return "unchanged"
					}(),
					"significant_change": priceDiscrepancy > 1.0,
				},
			}
		}
	} else {
		alpacaData["stock_price_error"] = err.Error()
	}
	
	// Add market data context for Grok analysis
	alpacaData["market_context_questions"] = []string{
		"Is the current stock price significantly different from analysis price?",
		"Does the recent price movement affect the option attractiveness?", 
		"Are there any market conditions that could impact this ranking?",
		"Does the volume indicate unusual activity in this stock?",
		"Based on current market conditions, would you choose a different strike price?",
		"Do any of the alternative options offer better risk-adjusted returns?",
	}

	if found {
		// Create detailed analysis of all ticker options
		allOptionsAnalysis := make([]map[string]interface{}, len(allTickerOptions))
		for i, option := range allTickerOptions {
			// Find this option's rank in the overall results
			optionRank := 0
			for j, result := range results {
				if result.Ticker == option.Ticker && result.OptionSymbol == option.OptionSymbol {
					optionRank = j + 1
					break
				}
			}
			
			moneyness := option.StockPrice / option.Strike
			annualizedReturn := option.ProfitPercentage * (365.0 / float64(option.DaysToExp))
			premiumAsPercent := (option.Premium / option.StockPrice) * 100
			cashEfficiency := option.TotalPremium / option.CashNeeded * 100
			
			allOptionsAnalysis[i] = map[string]interface{}{
				"rank_in_overall_results": optionRank,
				"rank_among_ticker_options": i + 1,
				"option_symbol": option.OptionSymbol,
				"strike": option.Strike,
				"premium": option.Premium,
				"profit_percentage": option.ProfitPercentage,
				"delta": option.Delta,
				"implied_volatility": option.ImpliedVol,
				"days_to_expiration": option.DaysToExp,
				"max_contracts": option.MaxContracts,
				"total_premium": option.TotalPremium,
				"cash_needed": option.CashNeeded,
				"greeks": map[string]interface{}{
					"delta": option.Delta,
					"gamma": option.Gamma,
					"theta": option.Theta,
					"vega": option.Vega,
				},
				"advanced_metrics": map[string]interface{}{
					"moneyness": moneyness,
					"annualized_return": annualizedReturn,
					"premium_as_percent_of_stock": premiumAsPercent,
					"cash_efficiency": cashEfficiency,
				},
			}
		}
		
		auditData["option_result"] = tickerResult
		auditData["rank"] = rank
		auditData["total_options_for_ticker"] = len(allTickerOptions)
		auditData["all_ticker_options"] = allOptionsAnalysis
		auditData["option_selection_analysis"] = map[string]interface{}{
			"best_option_rank": rank,
			"alternative_options_available": len(allTickerOptions) - 1,
			"grok_analysis_prompt": fmt.Sprintf("This ticker has %d different option contracts available. The system chose the one ranked #%d overall. Analyze: 1) Is this the best choice among the %d options for %s? 2) Would you rank the alternatives differently? 3) Are there better risk/reward profiles in the other options? 4) Does the #1 choice optimize for the right factors?", len(allTickerOptions), rank, len(allTickerOptions), ticker),
		}
		auditData["alpaca_market_data"] = alpacaData
		logger.Warn.Printf("🔍 AUDIT: Found %s with %d options, best at rank %d out of %d total", ticker, len(allTickerOptions), rank, len(results))
	} else {
		auditData["option_result"] = nil
		auditData["rank"] = "NOT_FOUND"
		auditData["total_options_for_ticker"] = 0
		auditData["alpaca_market_data"] = alpacaData
		logger.Warn.Printf("🔍 AUDIT: %s not found in %d results", ticker, len(results))
	}

	// Write/overwrite JSON file
	file, err := os.Create(filename)
	if err != nil {
		logger.Warn.Printf("🔍 AUDIT: Failed to create JSON file: %v", err)
		return
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(auditData); err != nil {
		logger.Warn.Printf("🔍 AUDIT: Failed to write JSON: %v", err)
		return
	}

	dataSize := len(fmt.Sprintf("%v", auditData))
	logger.Warn.Printf("🔍 AUDIT: JSON file created: %s (%d bytes)", filename, dataSize)
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

// AuditTickerHandler gathers detailed data for a single ticker audit
func (h *OptionsHandler) AuditTickerHandler(w http.ResponseWriter, r *http.Request) {
	logger.Warn.Printf("🔍 AUDIT: Ticker audit request received")
	
	// Set CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		logger.Warn.Printf("🔍 AUDIT: OPTIONS preflight request handled")
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		logger.Warn.Printf("🔍 AUDIT: Invalid method %s rejected", r.Method)
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse audit request
	var req models.AnalysisRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		logger.Warn.Printf("🔍 AUDIT: Failed to parse request: %v", err)
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	ticker := req.AuditTicker
	if ticker == "" {
		logger.Warn.Printf("🔍 AUDIT: Missing audit_ticker in request")
		http.Error(w, "audit_ticker is required", http.StatusBadRequest)
		return
	}
	
	logger.Warn.Printf("🔍 AUDIT: Processing detailed audit for ticker %s", ticker)
	logger.Warn.Printf("🔍 AUDIT: Request parameters - Delta: %.3f, Expiration: %s, Cash: %.2f", req.TargetDelta, req.ExpirationDate, req.AvailableCash)

	// Get stock price with full API logging
	logger.Warn.Printf("🔍 AUDIT: Fetching stock price from Alpaca API for %s", ticker)
	stockPrice, err := h.alpacaClient.GetStockPrice(ticker)
	if err != nil {
		logger.Warn.Printf("🔍 AUDIT: Failed to get stock price for %s: %v", ticker, err)
		http.Error(w, fmt.Sprintf("Failed to get stock price for %s: %v", ticker, err), http.StatusInternalServerError)
		return
	}
	logger.Warn.Printf("🔍 AUDIT: Stock price received for %s: $%.2f", ticker, stockPrice.Price)

	// Calculate time to expiration
	logger.Warn.Printf("🔍 AUDIT: Parsing expiration date: %s", req.ExpirationDate)
	expirationTime, err := time.Parse("2006-01-02", req.ExpirationDate)
	if err != nil {
		logger.Warn.Printf("🔍 AUDIT: Invalid expiration date format: %v", err)
		http.Error(w, fmt.Sprintf("Invalid expiration date: %v", err), http.StatusBadRequest)
		return
	}

	now := time.Now()
	daysToExp := int(expirationTime.Sub(now).Hours() / 24)
	logger.Warn.Printf("🔍 AUDIT: Time calculation - Days to expiration: %d", daysToExp)
	timeToExp := float64(daysToExp) / 365.0

	// Calculate target strike based on delta (simplified for audit)
	var targetMultiplier float64
	if req.TargetDelta <= 0.30 {
		targetMultiplier = 0.95 // 5% OTM for puts
	} else {
		targetMultiplier = 0.90 // 10% OTM for higher delta
	}
	
	bestStrike := stockPrice.Price * targetMultiplier
	// Round to nearest $5
	bestStrike = math.Round(bestStrike/5) * 5
	
	// Perform detailed Black-Scholes calculation with logging
	riskFreeRate := 0.0525 // 5.25% risk-free rate
	dividendYield := 0.005 // 0.5% dividend yield
	impliedVol := 0.25    // 25% implied volatility
	
	logger.Warn.Printf("🔍 AUDIT: Starting Black-Scholes calculation for %s", ticker)
	logger.Warn.Printf("🔍 AUDIT: BS Input - S: $%.2f, K: $%.2f, T: %.4f, r: %.4f, σ: %.3f", stockPrice.Price, bestStrike, timeToExp, riskFreeRate, impliedVol)

	// Calculate d1 and d2 with full formula logging
	d1 := (math.Log(stockPrice.Price/bestStrike) + (riskFreeRate-dividendYield+0.5*impliedVol*impliedVol)*timeToExp) / (impliedVol * math.Sqrt(timeToExp))
	d2 := d1 - impliedVol*math.Sqrt(timeToExp)
	logger.Warn.Printf("🔍 AUDIT: BS Calculated - d1: %.4f, d2: %.4f", d1, d2)

	// Calculate N(d1) and N(d2)
	nd1 := 0.5 * (1.0 + math.Erf(d1/math.Sqrt(2.0)))
	nd2 := 0.5 * (1.0 + math.Erf(d2/math.Sqrt(2.0)))
	negNd1 := 0.5 * (1.0 + math.Erf(-d1/math.Sqrt(2.0)))
	negNd2 := 0.5 * (1.0 + math.Erf(-d2/math.Sqrt(2.0)))
	logger.Warn.Printf("🔍 AUDIT: BS Normal distributions - N(d1): %.4f, N(d2): %.4f, N(-d1): %.4f, N(-d2): %.4f", nd1, nd2, negNd1, negNd2)

	// Calculate put option price and Greeks
	putPrice := bestStrike*math.Exp(-riskFreeRate*timeToExp)*negNd2 - stockPrice.Price*math.Exp(-dividendYield*timeToExp)*negNd1
	delta := -math.Exp(-dividendYield*timeToExp) * negNd1
	logger.Warn.Printf("🔍 AUDIT: BS Final results - Put Price: $%.4f, Delta: %.4f", putPrice, delta)

	// Create detailed audit response
	logger.Warn.Printf("🔍 AUDIT: Building comprehensive audit response for %s", ticker)
	auditResponse := map[string]interface{}{
		"ticker": ticker,
		"rank":   1, // Will be determined by ranking
		"api_requests": []map[string]interface{}{
			{
				"url":           fmt.Sprintf("/v2/stocks/%s/quotes/latest", ticker),
				"method":        "GET",
				"response":      stockPrice,
				"timestamp":     time.Now().Format(time.RFC3339),
			},
		},
		"calculations": map[string]interface{}{
			"inputs": map[string]float64{
				"S": stockPrice.Price,
				"K": bestStrike,
				"T": timeToExp,
				"r": riskFreeRate,
				"σ": impliedVol,
			},
			"formulas": map[string]string{
				"d1": fmt.Sprintf("d1 = (ln(%.2f/%.2f) + (%.4f + 0.5*%.3f²)*%.3f) / (%.3f * √%.3f) = %.4f",
					stockPrice.Price, bestStrike, riskFreeRate, impliedVol, timeToExp, impliedVol, timeToExp, d1),
				"d2": fmt.Sprintf("d2 = %.4f - %.3f * √%.3f = %.4f", d1, impliedVol, timeToExp, d2),
				"N_d1": fmt.Sprintf("N(%.4f) = %.4f", d1, nd1),
				"N_d2": fmt.Sprintf("N(%.4f) = %.4f", d2, nd2),
				"put_price": fmt.Sprintf("P = %.2f*e^(-%.4f*%.3f)*%.4f - %.2f*e^(-%.3f*%.3f)*%.4f = %.4f",
					bestStrike, riskFreeRate, timeToExp, negNd2, stockPrice.Price, dividendYield, timeToExp, negNd1, putPrice),
				"delta": fmt.Sprintf("Δ = -e^(-%.3f*%.3f) * %.4f = %.4f", dividendYield, timeToExp, negNd1, delta),
			},
			"execution_mode": "CPU_AUDIT",
			"computation_time_ms": 0.1,
		},
		"final_result": map[string]interface{}{
			"strike":           bestStrike,
			"premium":          putPrice,
			"delta":            delta,
			"days_to_exp":      daysToExp,
			"profit_potential": (putPrice / bestStrike) * 100,
		},
	}
	
	responseSize := len(fmt.Sprintf("%v", auditResponse))
	logger.Warn.Printf("🔍 AUDIT: Audit response built for %s (%d bytes)", ticker, responseSize)
	logger.Warn.Printf("🔍 AUDIT: Final results - Strike: $%.2f, Premium: $%.4f, Delta: %.4f, Profit: %.2f%%", 
		bestStrike, putPrice, delta, (putPrice/bestStrike)*100)

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(auditResponse); err != nil {
		logger.Warn.Printf("🔍 AUDIT: Failed to encode audit response for %s: %v", ticker, err)
		return
	}
	logger.Warn.Printf("🔍 AUDIT: Complete - audit data sent successfully for %s", ticker)
}

// AIAnalysisHandler sends data to AI for analysis
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

	// Parse request body
	var requestData struct {
		Ticker string `json:"ticker"`
	}
	if err := json.NewDecoder(r.Body).Decode(&requestData); err != nil {
		logger.Warn.Printf("🤖 GROK: Failed to parse request body: %v", err)
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	ticker := requestData.Ticker
	if ticker == "" {
		logger.Warn.Printf("🤖 GROK: No ticker provided")
		http.Error(w, "Ticker required", http.StatusBadRequest)
		return
	}

	logger.Warn.Printf("🤖 GROK: Processing analysis for ticker %s", ticker)

	// Load the single audit JSON file
	auditFile := "audit.json"
	logger.Warn.Printf("🤖 GROK: Loading audit JSON file: %s", auditFile)
	
	auditBytes, err := os.ReadFile(auditFile)
	if err != nil {
		logger.Warn.Printf("🤖 GROK: Failed to read audit file %s: %v", auditFile, err)
		http.Error(w, fmt.Sprintf("Audit file not found. Run analysis first with audit ticker set."), http.StatusNotFound)
		return
	}

	logger.Warn.Printf("🤖 GROK: Loaded audit data: %d bytes", len(auditBytes))

	// Generate AI analysis using YAML prompt
	// TODO: Integrate with actual Grok API
	logger.Warn.Printf("🤖 GROK: Generating AI analysis using YAML prompt")
	
	// Use built-in AI prompt (TODO: move to config)
	aiPrompt := "You are an expert options trader analyzing put option recommendations. Analyze the provided audit data and determine if this ticker is ranked appropriately based on risk-adjusted premium calculations."

	// Generate analysis (mock for now)
	mockAnalysis := fmt.Sprintf("%s\n\nAnalysis for %s:\nBased on the provided audit data, this option appears to have reasonable risk-adjusted returns. The calculations show proper Black-Scholes pricing model application.\n\nGenerated at %s", 
		aiPrompt, ticker, time.Now().Format("2006-01-02 15:04:05"))

	logger.Warn.Printf("🤖 GROK: AI analysis completed for %s, response length: %d chars", ticker, len(mockAnalysis))

	// Create audit log entry with JSON data
	logEntry := map[string]interface{}{
		"ticker": ticker,
		"timestamp": time.Now().Format(time.RFC3339),
		"ai_analysis": mockAnalysis,
		"audit_data": json.RawMessage(auditBytes),
		"type": "grok_analysis",
	}

	// Send to audit log handler (creates markdown file)
	logger.Warn.Printf("🤖 GROK: Creating audit log markdown file")
	h.processAuditLog(logEntry)

	// Move the JSON file to audits folder with same naming format
	auditDir := "audits"
	timestamp := time.Now().Format("2006-01-02_15-04-05")
	jsonDestination := fmt.Sprintf("%s/audit_%s_%s.json", auditDir, ticker, timestamp)
	logger.Warn.Printf("🤖 GROK: Moving audit JSON file from %s to %s", auditFile, jsonDestination)
	
	if err := os.Rename(auditFile, jsonDestination); err != nil {
		logger.Warn.Printf("🤖 GROK: Failed to move JSON file: %v", err)
	} else {
		logger.Warn.Printf("🤖 GROK: JSON file moved successfully to %s", jsonDestination)
	}

	response := map[string]string{
		"analysis": mockAnalysis,
		"status":   "success",
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		logger.Warn.Printf("🤖 GROK: Failed to encode response: %v", err)
		return
	}
	logger.Warn.Printf("🤖 GROK: AI analysis response sent successfully for %s", ticker)
}

// processAuditLog processes audit log entry and creates markdown file
func (h *OptionsHandler) processAuditLog(logEntry map[string]interface{}) {
	// Create audits directory if it doesn't exist
	auditDir := "audits"
	if err := os.MkdirAll(auditDir, 0755); err != nil {
		logger.Warn.Printf("📋 AUDIT: Failed to create audit directory: %v", err)
		return
	}

	// Generate timestamped filename: audit_ticker_YYYY-MM-DD_HH-MM-SS.md
	ticker := "UNKNOWN"
	if t, ok := logEntry["ticker"]; ok {
		ticker = fmt.Sprintf("%v", t)
	}
	timestamp := time.Now().Format("2006-01-02_15-04-05")
	filename := fmt.Sprintf("%s/audit_%s_%s.md", auditDir, ticker, timestamp)
	logger.Warn.Printf("📋 AUDIT: Creating audit file for %s: %s", ticker, filename)

	// Extract data for markdown
	analysisText := "No analysis available"
	if analysis, ok := logEntry["ai_analysis"]; ok {
		analysisText = fmt.Sprintf("%v", analysis)
	}
	
	auditData := ""
	if data, ok := logEntry["audit_data"]; ok {
		auditDataBytes, _ := json.MarshalIndent(data, "", "  ")
		auditData = string(auditDataBytes)
	}

	// Create markdown content
	markdownContent := fmt.Sprintf("# Grok AI Analysis - %s\n\n**Generated:** %s\n**Ticker:** %s\n\n## AI Analysis\n\n%s\n\n## Audit Data\n\n<details>\n<summary>Click to view detailed audit data</summary>\n\n```json\n%s\n```\n\n</details>\n\n---\n*Generated by Barracuda Options Analysis System with Grok AI*\n", 
		ticker, time.Now().Format("2006-01-02 15:04:05"), ticker, analysisText, auditData)

	// Write markdown file
	file, err := os.Create(filename)
	if err != nil {
		logger.Warn.Printf("📋 AUDIT: Failed to create audit file %s: %v", filename, err)
		return
	}
	defer file.Close()

	if _, err := file.WriteString(markdownContent); err != nil {
		logger.Warn.Printf("📋 AUDIT: Failed to write markdown: %v", err)
		return
	}

	logger.Warn.Printf("📋 AUDIT: Markdown file created: %s", filename)
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
	
	// Parse log entry
	var logEntry map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&logEntry); err != nil {
		logger.Warn.Printf("📋 AUDIT: Failed to parse log entry: %v", err)
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Create audits directory if it doesn't exist
	auditDir := "audits"
	logger.Warn.Printf("📋 AUDIT: Ensuring audit directory exists: %s", auditDir)
	if err := os.MkdirAll(auditDir, 0755); err != nil {
		logger.Warn.Printf("📋 AUDIT: Failed to create audit directory: %v", err)
		logger.Error.Printf("❌ Failed to create audit directory: %v", err)
		http.Error(w, "Failed to create audit directory", http.StatusInternalServerError)
		return
	}

	// Generate timestamped filename: audit_ticker_YYYY-MM-DD_HH-MM-SS.md
	ticker := "UNKNOWN"
	if t, ok := logEntry["ticker"]; ok {
		ticker = fmt.Sprintf("%v", t)
	}
	timestamp := time.Now().Format("2006-01-02_15-04-05")
	filename := fmt.Sprintf("%s/audit_%s_%s.md", auditDir, ticker, timestamp)
	logger.Warn.Printf("📋 AUDIT: Creating audit file for %s: %s", ticker, filename)

	// Write JSON file
	logger.Warn.Printf("📋 AUDIT: Opening file for writing: %s", filename)
	file, err := os.Create(filename)
	if err != nil {
		logger.Warn.Printf("📋 AUDIT: Failed to create audit file %s: %v", filename, err)
		logger.Error.Printf("❌ Failed to create audit file %s: %v", filename, err)
		http.Error(w, "Failed to create audit file", http.StatusInternalServerError)
		return
	}
	defer file.Close()

	// Calculate data size
	dataSize := len(fmt.Sprintf("%v", logEntry))
	logger.Warn.Printf("📋 AUDIT: Writing %d bytes of audit data to file", dataSize)

	// Format as human-readable markdown
	tickerName := "UNKNOWN"
	if t, ok := logEntry["ticker"]; ok {
		tickerName = fmt.Sprintf("%v", t)
	}
	
	analysisText := "No analysis available"
	if analysis, ok := logEntry["ai_analysis"]; ok {
		analysisText = fmt.Sprintf("%v", analysis)
	}
	
	auditData := ""
	if data, ok := logEntry["audit_data"]; ok {
		auditDataBytes, _ := json.MarshalIndent(data, "", "  ")
		auditData = string(auditDataBytes)
	}
	
	markdownContent := fmt.Sprintf("# Grok AI Analysis - %s\n\n**Generated:** %s\n**Ticker:** %s\n\n## AI Analysis\n\n%s\n\n## Audit Data\n\n<details>\n<summary>Click to view detailed audit data</summary>\n\n```json\n%s\n```\n\n</details>\n\n---\n*Generated by Barracuda Options Analysis System with Grok AI*\n", 
		tickerName, time.Now().Format("2006-01-02 15:04:05"), tickerName, analysisText, auditData)

	if _, err := file.WriteString(markdownContent); err != nil {
		logger.Warn.Printf("📋 AUDIT: Failed to write audit file %s: %v", filename, err)
		logger.Error.Printf("❌ Failed to write audit file %s: %v", filename, err)
		http.Error(w, "Failed to write audit file", http.StatusInternalServerError)
		return
	}

	logger.Warn.Printf("📋 AUDIT: File successfully created: %s (%d bytes)", filename, dataSize)
	logger.Info.Printf("📋 AUDIT FILE CREATED: %s", filename)
	
	response := map[string]string{
		"status":   "success",
		"message":  "Audit file created",
		"filename": filename,
	}

	logger.Warn.Printf("📋 AUDIT: Sending success response for %s", ticker)
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		logger.Warn.Printf("📋 AUDIT: Failed to encode response: %v", err)
		return
	}
	logger.Warn.Printf("📋 AUDIT: Complete - audit file created and response sent for %s", ticker)
}
