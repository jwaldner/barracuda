package audit

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// AlpacaAuditData holds audit data for Alpaca API calls
type AlpacaAuditData struct {
	APIRequests []APIRequest `json:"api_requests"`
	Summary     Summary      `json:"summary"`
}

// APIRequest represents a single API request audit entry
type APIRequest struct {
	Type       string                 `json:"type"`
	URL        string                 `json:"url"`
	Method     string                 `json:"method"`
	DurationMs int64                  `json:"duration_ms"`
	Timestamp  string                 `json:"timestamp"`
	Success    bool                   `json:"success"`
	Error      string                 `json:"error,omitempty"`
	Response   map[string]interface{} `json:"response,omitempty"`
}

// Summary holds summary information for the audit
type Summary struct {
	TotalRequests int   `json:"total_requests"`
	SuccessCount  int   `json:"success_count"`
	ErrorCount    int   `json:"error_count"`
	TotalTimeMs   int64 `json:"total_time_ms"`
}

// AlpacaAudit implements the Auditable interface for Alpaca API calls
type AlpacaAudit struct {
	requests []APIRequest
}

// NewAlpacaAudit creates a new Alpaca audit component
func NewAlpacaAudit() *AlpacaAudit {
	return &AlpacaAudit{
		requests: make([]APIRequest, 0),
	}
}

// AddRequest adds an API request to the audit log
func (aa *AlpacaAudit) AddRequest(reqType, url, method string, duration time.Duration, success bool, errorMsg string, response map[string]interface{}) {
	request := APIRequest{
		Type:       reqType,
		URL:        url,
		Method:     method,
		DurationMs: duration.Milliseconds(),
		Timestamp:  time.Now().Format(time.RFC3339),
		Success:    success,
		Response:   response,
	}

	if errorMsg != "" {
		request.Error = errorMsg
	}

	aa.requests = append(aa.requests, request)
}

// InitAppendRename is the ONLY function to interact with audit system
// It handles initialization, appending, and file management in one call
func (aa *AlpacaAudit) InitAppendRename(ticker string, operation string, data map[string]interface{}) error {
	// Step 1: INIT - Check if this is first audit for this ticker
	filename := "audit.json"
	var existingData AlpacaAuditData
	isNewAudit := false

	if file, err := os.Open(filename); err == nil {
		defer file.Close()
		decoder := json.NewDecoder(file)
		if err := decoder.Decode(&existingData); err != nil {
			// File corrupted, start fresh
			isNewAudit = true
			existingData = AlpacaAuditData{APIRequests: []APIRequest{}, Summary: Summary{}}
		}
	} else {
		// No file exists, start fresh
		isNewAudit = true
		existingData = AlpacaAuditData{APIRequests: []APIRequest{}, Summary: Summary{}}
	}

	// If this is new audit or first time seeing this ticker, add init message FIRST
	needsInit := isNewAudit
	if !needsInit {
		// Check if we already have init message for this ticker
		hasInit := false
		for _, req := range existingData.APIRequests {
			if req.Type == "TickerAuditInitialization" {
				if tickerVal, ok := req.Response["ticker"]; ok {
					if tickerStr, isStr := tickerVal.(string); isStr && tickerStr == ticker {
						hasInit = true
						break
					}
				}
			}
		}
		needsInit = !hasInit
	}

	if needsInit {
		// Step 1: Add ticker initialization as FIRST entry
		initRequest := APIRequest{
			Type:       "TickerAuditInitialization",
			URL:        "/audit/init",
			Method:     "INIT",
			DurationMs: 0,
			Timestamp:  time.Now().Format(time.RFC3339),
			Success:    true,
			Response: map[string]interface{}{
				"ticker":  ticker,
				"message": "🎯 AUDIT INIT: Now auditing ticker: " + ticker,
			},
		}

		// Insert at beginning
		existingData.APIRequests = append([]APIRequest{initRequest}, existingData.APIRequests...)
	}

	// Step 2: APPEND - Add the current audit entry
	if operation != "" {
		// Add current request data from aa.requests to existing data
		existingData.APIRequests = append(existingData.APIRequests, aa.requests...)
	}

	// Step 3: RENAME/SAVE - Calculate summary and save
	totalRequests := len(existingData.APIRequests)
	successCount := 0
	errorCount := 0
	totalTime := int64(0)

	for _, req := range existingData.APIRequests {
		if req.Success {
			successCount++
		} else {
			errorCount++
		}
		totalTime += req.DurationMs
	}

	existingData.Summary = Summary{
		TotalRequests: totalRequests,
		SuccessCount:  successCount,
		ErrorCount:    errorCount,
		TotalTimeMs:   totalTime,
	}

	// Write final file
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create audit file: %v", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(existingData)
}

// SaveToFile saves the current audit data to a JSON file
func (aa *AlpacaAudit) SaveToFile(filename string) error {
	// Calculate summary
	totalRequests := len(aa.requests)
	successCount := 0
	errorCount := 0
	totalTime := int64(0)

	for _, req := range aa.requests {
		if req.Success {
			successCount++
		} else {
			errorCount++
		}
		totalTime += req.DurationMs
	}

	summary := Summary{
		TotalRequests: totalRequests,
		SuccessCount:  successCount,
		ErrorCount:    errorCount,
		TotalTimeMs:   totalTime,
	}

	auditData := AlpacaAuditData{
		APIRequests: aa.requests,
		Summary:     summary,
	}

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create audit file %s: %v", filename, err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(auditData)
}

// AppendToFile appends new audit requests to an existing audit file
func (aa *AlpacaAudit) AppendToFile(filename string, ticker string) error {
	if len(aa.requests) == 0 {
		return nil // Nothing to append
	}

	// Try to read existing file
	var existingData AlpacaAuditData
	if file, err := os.Open(filename); err == nil {
		defer file.Close()
		decoder := json.NewDecoder(file)
		if err := decoder.Decode(&existingData); err != nil {
			// File exists but is corrupted, start fresh
			existingData = AlpacaAuditData{APIRequests: []APIRequest{}}
		}
	} else {
		// File doesn't exist, start fresh
		existingData = AlpacaAuditData{APIRequests: []APIRequest{}}
	}

	// Add ticker info to each request
	for i := range aa.requests {
		if aa.requests[i].Response == nil {
			aa.requests[i].Response = make(map[string]interface{})
		}
		aa.requests[i].Response["audit_ticker"] = ticker
	}

	// Append new requests
	existingData.APIRequests = append(existingData.APIRequests, aa.requests...)

	// Recalculate summary
	totalRequests := len(existingData.APIRequests)
	successCount := 0
	errorCount := 0
	totalTime := int64(0)

	for _, req := range existingData.APIRequests {
		if req.Success {
			successCount++
		} else {
			errorCount++
		}
		totalTime += req.DurationMs
	}

	existingData.Summary = Summary{
		TotalRequests: totalRequests,
		SuccessCount:  successCount,
		ErrorCount:    errorCount,
		TotalTimeMs:   totalTime,
	}

	// Write back to file
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create audit file %s: %v", filename, err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(existingData)
}

// Clear clears all collected audit data
func (aa *AlpacaAudit) Clear() {
	aa.requests = aa.requests[:0]
}
