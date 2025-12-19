package services

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/jwaldner/barracuda/internal/dto"
)

// RequestService handles HTTP request parsing
type RequestService struct{}

// NewRequestService creates a new request service
func NewRequestService() *RequestService {
	return &RequestService{}
}

// ParseAnalysisRequest parses an HTTP request into an AnalysisRequest
func (s *RequestService) ParseAnalysisRequest(r *http.Request) (*dto.AnalysisRequest, error) {
	if r.Method != http.MethodPost {
		return nil, fmt.Errorf("method not allowed: %s", r.Method)
	}

	var req dto.AnalysisRequest

	// Parse JSON request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return nil, fmt.Errorf("failed to decode request: %w", err)
	}

	// Validate required fields
	if len(req.Symbols) == 0 {
		return nil, fmt.Errorf("symbols are required")
	}

	if req.ExpirationDate == "" {
		return nil, fmt.Errorf("expiration_date is required")
	}

	// Set defaults
	if req.TargetDelta == 0 {
		req.TargetDelta = 0.10 // LOW risk default
	}

	if req.AvailableCash == 0 {
		req.AvailableCash = 70000
	}

	if req.Strategy == "" {
		req.Strategy = "puts"
	}

	// Clean symbols
	var cleanSymbols []string
	for _, symbol := range req.Symbols {
		symbol = strings.TrimSpace(strings.ToUpper(symbol))
		if symbol != "" {
			cleanSymbols = append(cleanSymbols, symbol)
		}
	}
	req.Symbols = cleanSymbols

	// Analysis request parsed
	return &req, nil
}

// ValidateExpirationDate validates that the expiration date is valid
func (s *RequestService) ValidateExpirationDate(dateStr string) error {
	_, err := time.Parse("2006-01-02", dateStr)
	if err != nil {
		return fmt.Errorf("invalid expiration date format: %w", err)
	}
	return nil
}

// ParseFloat64 safely parses a float64 with default fallback
func (s *RequestService) ParseFloat64(str string, defaultValue float64) float64 {
	if val, err := strconv.ParseFloat(str, 64); err == nil {
		return val
	}
	return defaultValue
}
