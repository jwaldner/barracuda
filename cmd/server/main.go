package main

import (
	"fmt"
	"log"
	"net/http"
	"strings"

	barracuda "github.com/jwaldner/barracuda/barracuda_lib"
	"github.com/jwaldner/barracuda/internal/alpaca"
	"github.com/jwaldner/barracuda/internal/config"
	"github.com/jwaldner/barracuda/internal/handlers"
	"github.com/jwaldner/barracuda/internal/logger"
	"github.com/jwaldner/barracuda/internal/symbols"

	"github.com/gorilla/mux"
)

func main() {
	cfg := config.Load()

	// Initialize proper logging with config level and file path
	if err := logger.InitWithConfig(cfg.Logging.LogLevel, cfg.Logging.LogFile); err != nil {
		log.Fatalf("Failed to initialize logging: %v", err)
	}
	logger.Always.Printf("🚀 Barracuda Options Analyzer starting - Port: %s", cfg.Port)

	if cfg.Logging.LogLevel == "verbose" {
		fmt.Printf("⚠️  VERBOSE LOGGING ENABLED - Detailed Alpaca API calls and calculations will be logged to %s\n", cfg.Logging.LogFile)
	}

	// Validate required config after loading from YAML
	if cfg.AlpacaAPIKey == "" {
		log.Fatal("ALPACA_API_KEY is required (set in config.yaml or environment variable)")
	}
	if cfg.AlpacaSecretKey == "" {
		log.Fatal("ALPACA_SECRET_KEY is required (set in config.yaml or environment variable)")
	}

	// Only reject obvious placeholder values - let web handle actual auth errors
	if strings.Contains(cfg.AlpacaAPIKey, "<") || strings.Contains(cfg.AlpacaAPIKey, ">") ||
		cfg.AlpacaAPIKey == "YOUR_API_KEY" || cfg.AlpacaAPIKey == "REPLACE_ME" {
		log.Fatal("❌ API key appears to be a placeholder - please set real credentials")
	}
	if strings.Contains(cfg.AlpacaSecretKey, "<") || strings.Contains(cfg.AlpacaSecretKey, ">") ||
		cfg.AlpacaSecretKey == "YOUR_SECRET_KEY" || cfg.AlpacaSecretKey == "REPLACE_ME" {
		log.Fatal("❌ Secret key appears to be a placeholder - please set real credentials")
	}

	// Initialize engine based on configuration
	var engine *barracuda.BaracudaEngine

	// Check execution mode from config
	executionMode := cfg.Engine.ExecutionMode
	if executionMode == "" {
		executionMode = "auto" // fallback to auto if not set
	}

	if executionMode == "cpu" {
		// Force CPU mode by creating engine differently
		engine = barracuda.NewBaracudaEngineForced("cpu")
		if engine != nil && engine.IsCudaAvailable() {
			logger.Always.Printf("🔧 EXECUTION MODE: CPU (CUDA hardware available but disabled by config)")
		} else {
			logger.Always.Printf("🔧 EXECUTION MODE: CPU (no CUDA hardware)")
		}
	} else {
		engine = barracuda.NewBaracudaEngine()
		if engine != nil && engine.IsCudaAvailable() {
			logger.Always.Printf("🔧 EXECUTION MODE: CUDA")
		} else {
			logger.Always.Printf("🔧 EXECUTION MODE: CPU (CUDA hardware not available)")
		}
	}

	if engine == nil {
		log.Fatal("Failed to initialize Barracuda engine")
	}
	defer engine.Close()

	// Show final active execution mode
	if engine != nil && engine.IsCudaAvailable() && (executionMode == "auto" || executionMode == "cuda") {
		log.Printf("⚡ ACTIVE MODE: CUDA (%d devices)", engine.GetDeviceCount())
	} else {
		log.Printf("🔧 ACTIVE MODE: CPU")
	}

	// Create Alpaca client
	log.Println("📡 Creating Alpaca client...")
	logger.Always.Printf("📡 Creating Alpaca client")
	logger.Info.Printf("📡 Alpaca client created - Base URL: https://api.alpaca.markets")

	baseClient := alpaca.NewClient(cfg.AlpacaAPIKey, cfg.AlpacaSecretKey)
	alpacaClient := alpaca.NewPerformanceWrapper(baseClient)

	// Create symbol service for company/sector lookups
	symbolService := symbols.NewSP500Service("assets/symbols")

	// Create options handler with CUDA engine
	// Initialize handlers
	optionsHandler := handlers.NewOptionsHandler(alpacaClient, cfg, engine, symbolService)
	sp500Handler := handlers.NewSP500Handler()

	// Setup router
	r := mux.NewRouter()

	// Serve static files (CSS, JS, images) - NO REBUILD NEEDED
	r.PathPrefix("/static/").Handler(http.StripPrefix("/static/", http.FileServer(http.Dir("web/static/"))))

	// Main application endpoints
	r.HandleFunc("/", optionsHandler.HomeHandler).Methods("GET")
	r.HandleFunc("/api/analyze", optionsHandler.AnalyzeHandler).Methods("POST")
	r.HandleFunc("/api/test-connection", optionsHandler.TestConnectionHandler).Methods("GET")

	// S&P 500 symbol management endpoints
	r.HandleFunc("/api/sp500/update", sp500Handler.UpdateSymbolsHandler).Methods("POST")
	r.HandleFunc("/api/sp500/symbols", sp500Handler.GetSymbolsHandler).Methods("GET")
	r.HandleFunc("/api/sp500/info", sp500Handler.GetSymbolsInfoHandler).Methods("GET")
	r.HandleFunc("/api/sp500/top25", sp500Handler.GetTop25Handler).Methods("GET")

	// Start server
	fmt.Printf("🌐 Server starting on http://localhost:%s\n", cfg.Port)
	logger.Always.Printf("🌐 Server starting on http://localhost:%s", cfg.Port)
	logger.Info.Printf("🌐 HTTP server started on port %s", cfg.Port)

	// Browser opening can be handled externally via OPEN_BROWSER env var if needed

	if err := http.ListenAndServe("0.0.0.0:"+cfg.Port, r); err != nil {
		log.Fatal("Server failed to start:", err)
	}
}
