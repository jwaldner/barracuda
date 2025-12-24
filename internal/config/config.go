package config

import (
	"io/ioutil"
	"os"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v2"
)

// LoggingConfig represents logging configuration
type LoggingConfig struct {
	LogLevel string `yaml:"log_level"`
	LogFile  string `yaml:"log_file"`
}

type ComputeConfig struct {
	ExecutionMode         string `yaml:"execution_mode"`
	BenchmarkMode         bool   `yaml:"benchmark_mode"`
	BenchmarkCalculations int    `yaml:"benchmark_calculations"`
	BenchmarkBatchSize    int    `yaml:"benchmark_batch_size"`
	SimulationWorkload    bool   `yaml:"simulation_workload"`
}

// CSVConfig represents CSV export configuration
type CSVConfig struct {
	FilenameFormat string `yaml:"filename_format"`
}

// AuditLogConfig represents audit log configuration
type AuditLogConfig struct {
	AIAnalysisPrompt string `yaml:"ai_analysis_prompt"`
	FilenameFormat   string `yaml:"filename_format"`
}

type Config struct {
	// Server settings
	Port string

	// Alpaca API settings
	AlpacaAPIKey    string
	AlpacaSecretKey string

	// Grok AI API settings
	GrokAPIKey         string
	GrokEndpoint       string
	GrokModel          string
	GrokTimeoutMinutes int

	// Default application settings
	DefaultStocks    []string
	DefaultCash      int
	DefaultStrategy  string
	DefaultRiskLevel string
	MaxResults       int

	// Logging settings
	Logging LoggingConfig `yaml:"logging"`
	// Engine settings
	Engine EngineConfig `yaml:"engine"`
	// Compute settings (legacy)
	Compute ComputeConfig `yaml:"compute"`
	// CSV export settings
	CSV CSVConfig `yaml:"csv"`
}

// AlpacaConfig represents Alpaca API configuration
type AlpacaConfig struct {
	APIKey    string `yaml:"api_key"`
	SecretKey string `yaml:"secret_key"`
}

// EngineConfig represents computation engine configuration
type EngineConfig struct {
	ExecutionMode         string  `yaml:"execution_mode"`          // auto, cuda, cpu
	BatchSize             int     `yaml:"batch_size"`              // Max contracts per batch
	EnableBenchmarks      bool    `yaml:"enable_benchmarks"`       // Enable performance benchmarking
	WorkloadFactor        float64 `yaml:"workload_factor"`         // Computational workload multiplier for benchmarking
	CompleteGPUProcessing bool    `yaml:"complete_gpu_processing"` // Enable complete GPU processing (experimental)
}

type YAMLConfig struct {
	Alpaca             AlpacaConfig  `yaml:"alpaca"`
	GrokAPIKey         string        `yaml:"grok_api_key"`
	GrokEndpoint       string        `yaml:"grok_endpoint"`
	GrokModel          string        `yaml:"grok_model"`
	GrokTimeoutMinutes int           `yaml:"grok_timeout_minutes"`
	Logging            LoggingConfig `yaml:"logging"`

	Trading struct {
		DefaultCash      int      `yaml:"default_cash"`
		TargetDelta      float64  `yaml:"target_delta"`
		MaxResults       int      `yaml:"max_results"`
		DefaultStocks    []string `yaml:"default_stocks"`
		DefaultRiskLevel string   `yaml:"default_risk_level"`
	} `yaml:"trading"`

	Engine   EngineConfig   `yaml:"engine"`
	Compute  ComputeConfig  `yaml:"compute"`
	CSV      CSVConfig      `yaml:"csv"`
	AuditLog AuditLogConfig `yaml:"audit_log"`
}

func Load() *Config {
	cfg := &Config{
		Port:               getEnv("PORT", "8080"),
		AlpacaAPIKey:       getEnv("ALPACA_API_KEY", ""),
		AlpacaSecretKey:    getEnv("ALPACA_SECRET_KEY", ""),
		GrokAPIKey:         getEnv("GROK_API_KEY", ""),
		GrokEndpoint:       getEnv("GROK_ENDPOINT", "https://api.x.ai/v1/chat/completions"),
		GrokModel:          getEnv("GROK_MODEL", "grok-3"),
		GrokTimeoutMinutes: getEnvInt("GROK_TIMEOUT_MINUTES", 15),
		DefaultStocks:      getEnvStringSlice("DEFAULT_STOCKS", []string{}), // No defaults - use S&P 500 ranking
		DefaultCash:        getEnvInt("DEFAULT_CASH", 10000),
		DefaultStrategy:    getEnv("DEFAULT_STRATEGY", "puts"),
		DefaultRiskLevel:   getEnv("DEFAULT_RISK_LEVEL", "LOW"),
		MaxResults:         getEnvInt("MAX_RESULTS", 25), // Default logging configuration
		Logging: LoggingConfig{
			LogLevel: getEnv("LOG_LEVEL", "info"),
			LogFile:  getEnv("LOG_FILE", "barracuda.log"),
		},

		// Default engine configuration
		Engine: EngineConfig{
			ExecutionMode:    getEnv("ENGINE_EXECUTION_MODE", "auto"),
			BatchSize:        getEnvInt("ENGINE_BATCH_SIZE", 1000),
			EnableBenchmarks: getEnvBool("ENGINE_ENABLE_BENCHMARKS", false),
		},

		// Default compute configuration (legacy)
		Compute: ComputeConfig{
			ExecutionMode:         getEnv("EXECUTION_MODE", "auto"),
			BenchmarkMode:         getEnvBool("BENCHMARK_MODE", true),
			BenchmarkCalculations: getEnvInt("BENCHMARK_CALCULATIONS", 1000000),
			BenchmarkBatchSize:    getEnvInt("BENCHMARK_BATCH_SIZE", 10000),
			SimulationWorkload:    getEnvBool("SIMULATION_WORKLOAD", true),
		},
	}

	// Try to load from YAML file and set environment variables
	if yamlCfg := loadYAMLConfig(); yamlCfg != nil {
		// YAML config loaded
		// Set environment variables from YAML if not already set
		if yamlCfg.Alpaca.APIKey != "" && yamlCfg.Alpaca.APIKey != "YOUR_ALPACA_API_KEY" {
			if os.Getenv("ALPACA_API_KEY") == "" {
				os.Setenv("ALPACA_API_KEY", yamlCfg.Alpaca.APIKey)
				// ALPACA_API_KEY loaded
			}
			cfg.AlpacaAPIKey = yamlCfg.Alpaca.APIKey
		}
		if yamlCfg.Alpaca.SecretKey != "" && yamlCfg.Alpaca.SecretKey != "YOUR_ALPACA_SECRET_KEY" {
			if os.Getenv("ALPACA_SECRET_KEY") == "" {
				os.Setenv("ALPACA_SECRET_KEY", yamlCfg.Alpaca.SecretKey)
				// ALPACA_SECRET_KEY loaded
			}
			cfg.AlpacaSecretKey = yamlCfg.Alpaca.SecretKey
		}

		// Load Grok API config from YAML
		if yamlCfg.GrokAPIKey != "" && yamlCfg.GrokAPIKey != "YOUR_GROK_API_KEY" {
			if os.Getenv("GROK_API_KEY") == "" {
				os.Setenv("GROK_API_KEY", yamlCfg.GrokAPIKey)
			}
			cfg.GrokAPIKey = yamlCfg.GrokAPIKey
		}
		if yamlCfg.GrokEndpoint != "" {
			cfg.GrokEndpoint = yamlCfg.GrokEndpoint
		}
		if yamlCfg.GrokModel != "" {
			cfg.GrokModel = yamlCfg.GrokModel
		}
		if yamlCfg.GrokTimeoutMinutes > 0 {
			cfg.GrokTimeoutMinutes = yamlCfg.GrokTimeoutMinutes
		}

		if yamlCfg.Trading.DefaultCash > 0 {
			cfg.DefaultCash = yamlCfg.Trading.DefaultCash
		}
		if len(yamlCfg.Trading.DefaultStocks) > 0 {
			cfg.DefaultStocks = yamlCfg.Trading.DefaultStocks
		}
		if yamlCfg.Trading.DefaultRiskLevel != "" {
			cfg.DefaultRiskLevel = yamlCfg.Trading.DefaultRiskLevel
		}
		// Always use YAML max_results value (0 = show all)
		cfg.MaxResults = yamlCfg.Trading.MaxResults

		// Logging configuration from YAML
		if yamlCfg.Logging.LogLevel != "" {
			cfg.Logging.LogLevel = yamlCfg.Logging.LogLevel
		}
		if yamlCfg.Logging.LogFile != "" {
			cfg.Logging.LogFile = yamlCfg.Logging.LogFile
		}

		// Engine configuration from YAML
		if yamlCfg.Engine.ExecutionMode != "" {
			cfg.Engine = yamlCfg.Engine
			// Engine mode configured
		}

		// Compute configuration from YAML (legacy)
		cfg.Compute = yamlCfg.Compute

		// CSV configuration from YAML
		cfg.CSV = yamlCfg.CSV
		// Set default filename format if not specified
		if cfg.CSV.FilenameFormat == "" {
			cfg.CSV.FilenameFormat = "{time}_{exp_date}_{delta}_{strategy}_{symbols}symbols.csv"
		}

		// Validate execution mode
		if cfg.Compute.ExecutionMode == "" {
			cfg.Compute.ExecutionMode = "auto"
		}
		if cfg.Compute.BenchmarkCalculations == 0 {
			cfg.Compute.BenchmarkCalculations = 1000000
		}
		if cfg.Compute.BenchmarkBatchSize == 0 {
			cfg.Compute.BenchmarkBatchSize = 10000
		}
	}

	return cfg
}

func loadYAMLConfig() *YAMLConfig {
	data, err := ioutil.ReadFile("config.yaml")
	if err != nil {
		// Could not read config.yaml - silently return nil
		return nil
	}

	var yamlCfg YAMLConfig
	if err := yaml.Unmarshal(data, &yamlCfg); err != nil {
		// Could not parse config.yaml - silently return nil
		return nil
	}

	return &yamlCfg
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.ParseBool(value); err == nil {
			return parsed
		}
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.Atoi(value); err == nil {
			return parsed
		}
	}
	return defaultValue
}

func getEnvStringSlice(key string, defaultValue []string) []string {
	if value := os.Getenv(key); value != "" {
		return strings.Split(value, ",")
	}
	return defaultValue
}

// CalculateDefaultExpirationDate calculates the next options expiration (3rd Friday) in YYYY-MM-DD format
func CalculateDefaultExpirationDate() string {
	today := time.Now()
	currentMonth := today.Month()
	currentYear := today.Year()

	// Find 3rd Friday of current month
	firstDay := time.Date(currentYear, currentMonth, 1, 0, 0, 0, 0, time.UTC)
	firstFriday := firstDay.AddDate(0, 0, (5-int(firstDay.Weekday())+7)%7)
	thirdFriday := firstFriday.AddDate(0, 0, 14)

	// If current day is PAST the 3rd Friday, use next month's 3rd Friday
	// Otherwise use current month's 3rd Friday
	if today.After(thirdFriday) {
		// Use next month's 3rd Friday
		nextMonth := currentMonth + 1
		nextYear := currentYear
		if nextMonth > 12 {
			nextMonth = 1
			nextYear++
		}

		nextFirstDay := time.Date(nextYear, nextMonth, 1, 0, 0, 0, 0, time.UTC)
		nextFirstFriday := nextFirstDay.AddDate(0, 0, (5-int(nextFirstDay.Weekday())+7)%7)
		nextThirdFriday := nextFirstFriday.AddDate(0, 0, 14)
		return nextThirdFriday.Format("2006-01-02")
	}

	return thirdFriday.Format("2006-01-02")
}

// FormatAuditFilename formats audit filenames using the configured template
func FormatAuditFilename(format, ticker, expDate, timestamp string) string {
	result := format
	result = strings.ReplaceAll(result, "{ticker}", ticker)
	result = strings.ReplaceAll(result, "{exp_date}", expDate)
	result = strings.ReplaceAll(result, "{timestamp}", timestamp)
	return result
}

// GetAuditConfig returns audit log configuration from loaded YAML
func GetAuditConfig() *AuditLogConfig {
	if yamlCfg := loadYAMLConfig(); yamlCfg != nil {
		return &yamlCfg.AuditLog
	}
	// Return defaults if no config loaded
	return &AuditLogConfig{
		FilenameFormat:   "{ticker}-{exp_date}",
		AIAnalysisPrompt: "Please analyze the provided options data and calculations for accuracy and completeness.",
	}
}
