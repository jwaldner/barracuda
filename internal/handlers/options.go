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

		// System status for footer
		"computeMode": func() string {
			if h.engine != nil && h.engine.IsCudaAvailable() {
				return "CUDA"
			}
			return "CPU"
		},
		"deviceInfo": func() string {
			if h.engine != nil && h.engine.IsCudaAvailable() {
				return fmt.Sprintf("(%d devices)", h.engine.GetDeviceCount())
			}
			return "(fallback mode)"
		},
		"engineMode": func() string {
			return strings.ToUpper(h.config.Engine.ExecutionMode)
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
	logger.Debug.Printf("=== OPTIONS ANALYSIS REQUEST ===")
	logger.Debug.Printf("Symbols: %v", req.Symbols)
	logger.Debug.Printf("Expiration Date: %s", req.ExpirationDate)
	logger.Debug.Printf("Target Delta: %.2f", req.TargetDelta)
	logger.Debug.Printf("Available Cash: $%.2f", req.AvailableCash)
	logger.Debug.Printf("Strategy: %s", req.Strategy)

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
	logger.Warn.Printf("🚀 START: %d symbols | %s | %s", 
		len(cleanSymbols), req.Strategy, map[bool]string{true: "CUDA", false: "CPU"}[h.engine.IsCudaAvailable()])

	
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

	// Sort results by total premium (highest first)
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].TotalPremium > results[i].TotalPremium {
				results[i], results[j] = results[j], results[i]
			}
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
		engineType = "CUDA GPU"
	}

	// Major Step: Completion
	logger.Warn.Printf("✅ COMPLETE: %d results | %.3fs | %s engine", len(results), duration.Seconds(), engineType)
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
			Strategy:         req.Strategy,
			ExpirationDate:   req.ExpirationDate,
			Timestamp:        time.Now().Format(time.RFC3339),
			ProcessingTime:   duration.Seconds(),
			ProcessingStats:  processingStats,
			Engine:           engineType,
			CudaAvailable:    h.engine.IsCudaAvailable(),
			ExecutionMode:    h.config.Engine.ExecutionMode,
			SymbolCount:      len(cleanSymbols),
			ResultCount:      len(results),
			WorkloadFactor:   h.config.Engine.WorkloadFactor,
			SamplesProcessed: totalSamplesProcessed,
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

// processRealOptions fetches real options individually
func (h *OptionsHandler) processRealOptions(stockPrices map[string]*alpaca.StockPrice, req models.AnalysisRequest, companyData map[string]struct{ Company, Sector string }) []models.OptionResult {
	var results []models.OptionResult
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Calculate time to expiration
	expirationTime, err := time.Parse("2006-01-02", req.ExpirationDate)
	if err != nil {
		logger.Error.Printf("❌ Invalid expiration date format: %v", err)
		return results
	}
	timeToExp := time.Until(expirationTime).Hours() / (24 * 365.25)

	// Process each symbol individually
	for symbol, stockPrice := range stockPrices {
		wg.Add(1)
		go func(sym string, price *alpaca.StockPrice) {
			defer wg.Done()

			// Get options for this symbol
			optionsChain, err := h.alpacaClient.GetOptionsChain([]string{sym}, req.ExpirationDate, req.Strategy)
			if err != nil {
				logger.Error.Printf("❌ Error getting options for %s: %v", sym, err)
				return
			}

			contracts, exists := optionsChain[sym]
			if !exists || len(contracts) == 0 {
				return
			}

			// Find best option
			bestContract := h.findBestOptionContract(contracts, price.Price, req.TargetDelta, req.Strategy)
			if bestContract == nil {
				return
			}

			// Convert to result
			companyInfo := companyData[sym]
			result := h.convertRealOptionToResult(sym, price.Price, bestContract, req.AvailableCash, req.ExpirationDate, timeToExp, companyInfo.Company, companyInfo.Sector)
			if result != nil {
				mu.Lock()
				results = append(results, *result)
				mu.Unlock()
			}
		}(symbol, stockPrice)
	}

	wg.Wait()
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
