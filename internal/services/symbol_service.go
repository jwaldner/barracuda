package services

import (
	"fmt"

	"github.com/jwaldner/barracuda/internal/config"
	"github.com/jwaldner/barracuda/internal/symbols"
)

// SymbolService handles symbol selection logic
type SymbolService struct {
	config       *config.Config
	sp500Service *symbols.SP500Service
}

// NewSymbolService creates a new symbol service
func NewSymbolService(cfg *config.Config, sp500Service *symbols.SP500Service) *SymbolService {
	return &SymbolService{
		config:       cfg,
		sp500Service: sp500Service,
	}
}

// GetAnalysisSymbols returns symbols for analysis based on configuration
func (s *SymbolService) GetAnalysisSymbols() ([]string, error) {
	// Use configured default stocks if available
	if len(s.config.DefaultStocks) > 0 {
		// Using configured default stocks
		return s.config.DefaultStocks, nil
	}

	// Fallback to S&P 500 symbols
	sp500Symbols, err := s.sp500Service.GetSymbolsAsStrings()
	if err != nil {
		return nil, fmt.Errorf("failed to get S&P 500 symbols: %w", err)
	}

	// Limit to top 25 for performance
	const maxSymbols = 25
	if len(sp500Symbols) > maxSymbols {
		sp500Symbols = sp500Symbols[:maxSymbols]
	}

	// Using S&P 500 symbols
	return sp500Symbols, nil
}

// GetSymbolSource returns a description of the symbol source
func (s *SymbolService) GetSymbolSource() string {
	if len(s.config.DefaultStocks) > 0 {
		return fmt.Sprintf("%d Configured: %v", len(s.config.DefaultStocks), s.config.DefaultStocks)
	}
	return "Top 25 S&P 500 (no default_stocks configured)"
}
