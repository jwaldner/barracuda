package config

import (
	"os"
	"testing"
)

func TestDefaultEnableBenchmarks(t *testing.T) {
	// Clear environment variable to test default
	os.Unsetenv("ENGINE_ENABLE_BENCHMARKS")
	
	cfg := Load()
	
	if !cfg.Engine.EnableBenchmarks {
		t.Errorf("Expected EnableBenchmarks to be true by default, got false")
	}
}

func TestEnableBenchmarksEnvOverride(t *testing.T) {
	// Test that environment variable can override the default
	os.Setenv("ENGINE_ENABLE_BENCHMARKS", "false")
	defer os.Unsetenv("ENGINE_ENABLE_BENCHMARKS")
	
	cfg := Load()
	
	if cfg.Engine.EnableBenchmarks {
		t.Errorf("Expected EnableBenchmarks to be false when env var is false, got true")
	}
}

func TestEnableBenchmarksEnvOverrideTrue(t *testing.T) {
	// Test that environment variable can explicitly set to true
	os.Setenv("ENGINE_ENABLE_BENCHMARKS", "true")
	defer os.Unsetenv("ENGINE_ENABLE_BENCHMARKS")
	
	cfg := Load()
	
	if !cfg.Engine.EnableBenchmarks {
		t.Errorf("Expected EnableBenchmarks to be true when env var is true, got false")
	}
}
