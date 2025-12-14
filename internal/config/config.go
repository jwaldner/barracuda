package config

import (
	"io/ioutil"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v2"
)

type ComputeConfig struct {
	ExecutionMode         string `yaml:"execution_mode"`
	BenchmarkMode         bool   `yaml:"benchmark_mode"`
	BenchmarkCalculations int    `yaml:"benchmark_calculations"`
	BenchmarkBatchSize    int    `yaml:"benchmark_batch_size"`
	SimulationWorkload    bool   `yaml:"simulation_workload"`
}

type Config struct {
	// Server settings
	Port string

	// Alpaca API settings
	AlpacaAPIKey       string
	AlpacaSecretKey    string
	AlpacaPaperTrading bool

	// Default application settings
	DefaultStocks    []string
	DefaultCash      int
	DefaultStrategy  string
	DefaultRiskLevel string

	// Engine settings
	Engine EngineConfig `yaml:"engine"`
	// Compute settings (legacy)
	Compute ComputeConfig `yaml:"compute"`
}

// AlpacaConfig represents Alpaca API configuration
type AlpacaConfig struct {
	APIKey    string `yaml:"api_key"`
	SecretKey string `yaml:"secret_key"`
}

// EngineConfig represents computation engine configuration
type EngineConfig struct {
	ExecutionMode    string  `yaml:"execution_mode"`    // auto, cuda, cpu
	BatchSize        int     `yaml:"batch_size"`        // Max contracts per batch
	EnableBenchmarks bool    `yaml:"enable_benchmarks"` // Enable performance benchmarking
	WorkloadFactor   float64 `yaml:"workload_factor"`   // Computational workload multiplier for benchmarking
}

type YAMLConfig struct {
	Alpaca AlpacaConfig `yaml:"alpaca"`

	Trading struct {
		DefaultCash      int      `yaml:"default_cash"`
		TargetDelta      float64  `yaml:"target_delta"`
		MaxResults       int      `yaml:"max_results"`
		DefaultStocks    []string `yaml:"default_stocks"`
		DefaultRiskLevel string   `yaml:"default_risk_level"`
	} `yaml:"trading"`

	Engine  EngineConfig  `yaml:"engine"`
	Compute ComputeConfig `yaml:"compute"`
}

func Load() *Config {
	cfg := &Config{
		Port:               getEnv("PORT", "8080"),
		AlpacaAPIKey:       getEnv("ALPACA_API_KEY", ""),
		AlpacaSecretKey:    getEnv("ALPACA_SECRET_KEY", ""),
		AlpacaPaperTrading: getEnvBool("ALPACA_PAPER_TRADING", false),
		DefaultStocks:      getEnvStringSlice("DEFAULT_STOCKS", []string{}), // No defaults - use S&P 500 ranking
		DefaultCash:        getEnvInt("DEFAULT_CASH", 10000),
		DefaultStrategy:    getEnv("DEFAULT_STRATEGY", "puts"),
		DefaultRiskLevel:   getEnv("DEFAULT_RISK_LEVEL", "LOW"),

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

		if yamlCfg.Trading.DefaultCash > 0 {
			cfg.DefaultCash = yamlCfg.Trading.DefaultCash
		}
		if len(yamlCfg.Trading.DefaultStocks) > 0 {
			cfg.DefaultStocks = yamlCfg.Trading.DefaultStocks
		}
		if yamlCfg.Trading.DefaultRiskLevel != "" {
			cfg.DefaultRiskLevel = yamlCfg.Trading.DefaultRiskLevel
		}

		// Engine configuration from YAML
		if yamlCfg.Engine.ExecutionMode != "" {
			cfg.Engine = yamlCfg.Engine
			// Engine mode configured
		}

		// Compute configuration from YAML (legacy)
		cfg.Compute = yamlCfg.Compute

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
		log.Printf("Could not read config.yaml: %v", err)
		return nil
	}

	var yamlCfg YAMLConfig
	if err := yaml.Unmarshal(data, &yamlCfg); err != nil {
		log.Printf("Could not parse config.yaml: %v", err)
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
