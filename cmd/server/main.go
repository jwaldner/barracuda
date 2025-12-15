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
	log.Println("🚀 Starting Barracuda Options Analyzer...")
	// Load configuration (this will set env vars from YAML)
	cfg := config.Load()
	log.Println("⚙️  Configuration loaded")

	// Initialize proper logging with config level
	if err := logger.InitWithLevel(cfg.Logging.LogLevel); err != nil {
		log.Fatalf("Failed to initialize logging: %v", err)
	}
	logger.Info.Printf("📝 Logging system initialized - level: %s -> %s", cfg.Logging.LogLevel, cfg.Logging.LogFile)
	logger.Info.Printf("🚀 Barracuda Options Analyzer starting - Port: %s", cfg.Port)

	// Terminal warning if verbose logging is enabled
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
		log.Println("CPU FORCED MODE: CUDA disabled")
	} else {
		engine = barracuda.NewBaracudaEngine()
	}

	if engine == nil {
		log.Fatal("Failed to initialize Barracuda engine")
	}
	defer engine.Close()

	switch executionMode {
	case "cuda":
		if engine.IsCudaAvailable() {
			// CUDA engine forced mode
		} else {
			log.Fatal("❌ CUDA mode requested but CUDA not available")
		}
	case "cpu":
		// Force CPU-only mode
		log.Println("CPU FORCED MODE: CUDA disabled")
	case "auto":
		fallthrough
	default:
		if engine != nil && engine.IsCudaAvailable() {
			fmt.Printf("🔥 Compute Mode: CUDA (%d devices detected)\n", engine.GetDeviceCount())
			logger.Info.Printf("🔥 CUDA engine initialized with %d devices", engine.GetDeviceCount())
		} else {
			log.Println("CPU FALLBACK: CUDA not available")
		}
	}

	// Create Alpaca client
	log.Println("📡 Creating Alpaca client...")
	logger.Info.Printf("📡 Alpaca client created - Base URL: https://api.alpaca.markets")

	alpacaClient := alpaca.NewClient(cfg.AlpacaAPIKey, cfg.AlpacaSecretKey)

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
	logger.Info.Printf("🌐 HTTP server started on port %s", cfg.Port)

	// Browser opening can be handled externally via OPEN_BROWSER env var if needed

	if err := http.ListenAndServe("0.0.0.0:"+cfg.Port, r); err != nil {
		log.Fatal("Server failed to start:", err)
	}
}
