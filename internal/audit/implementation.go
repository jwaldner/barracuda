package audit

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"strings"
	"time"

	"github.com/jwaldner/barracuda/internal/logger"
)

// OptionsAnalysisAuditLogger implements the OptionsAnalysisAuditor interface
// Provides concrete logging functionality for options trading analysis operations
type OptionsAnalysisAuditLogger struct {
	auditFilename string
}

// NewOptionsAnalysisAuditLogger creates a new options analysis audit logger
func NewOptionsAnalysisAuditLogger() OptionsAnalysisAuditor {
	return &OptionsAnalysisAuditLogger{
		auditFilename: "audit.json",
	}
}

// AuditEntry represents a single audit log entry
type AuditEntry struct {
	Type      string                 `json:"type"`
	URL       string                 `json:"url,omitempty"`
	Method    string                 `json:"method,omitempty"`
	Duration  int64                  `json:"duration_ms,omitempty"`
	Timestamp string                 `json:"timestamp"`
	Success   bool                   `json:"success"`
	Error     string                 `json:"error,omitempty"`
	Response  map[string]interface{} `json:"response,omitempty"`
	Message   string                 `json:"message,omitempty"`
	Ticker    string                 `json:"ticker,omitempty"`
}

// AuditFile represents the structure of the audit.json file
type AuditFile struct {
	Ticker      string       `json:"ticker"`
	APIRequests []AuditEntry `json:"api_requests"`
	Summary     struct {
		TotalRequests int   `json:"total_requests"`
		SuccessCount  int   `json:"success_count"`
		ErrorCount    int   `json:"error_count"`
		TotalTimeMs   int64 `json:"total_time_ms"`
	} `json:"summary"`
}

// LogOptionsAnalysisOperation handles the complete audit lifecycle for options analysis
// Initializes audit file, appends operation data, and archives completed audit
func (oaal *OptionsAnalysisAuditLogger) LogOptionsAnalysisOperation(ticker string, operation string, data interface{}) error {
	// Step 1: For "init" operation (start of new audit), delete existing file and start fresh
	auditFile := &AuditFile{}
	needsInit := false

	if operation == "init" {
		// New audit session - delete existing audit file and start completely fresh
		if err := os.Remove(oaal.auditFilename); err != nil && !os.IsNotExist(err) {
			// File removal failed but continue anyway
		}
		needsInit = true
		auditFile = &AuditFile{
			Ticker:      ticker,
			APIRequests: []AuditEntry{},
		}
	} else {
		// Continuing existing audit - load and preserve existing data
		if file, err := os.Open(oaal.auditFilename); err == nil {
			defer file.Close()
			decoder := json.NewDecoder(file)
			if err := decoder.Decode(auditFile); err != nil {
				// File corrupted, start fresh
				needsInit = true
				auditFile = &AuditFile{
					Ticker:      ticker,
					APIRequests: []AuditEntry{},
				}
			}
		} else {
			// No file exists, start fresh
			needsInit = true
			auditFile = &AuditFile{
				Ticker:      ticker,
				APIRequests: []AuditEntry{},
			}
		}
	}

	// Set or update ticker in existing audit file - ALWAYS set if ticker provided
	if ticker != "" {
		auditFile.Ticker = ticker
	}

	// Check if we need ticker initialization - only if no init entry exists for this ticker
	if ticker != "" && (needsInit || !oaal.hasInitEntry(auditFile, ticker)) {
		// Add ticker initialization as FIRST entry
		initEntry := AuditEntry{
			Type:      "TickerAuditInitialization",
			URL:       "/audit/init",
			Method:    "INIT",
			Timestamp: time.Now().Format(time.RFC3339),
			Success:   true,
			Message:   "🎯 AUDIT INIT: Now auditing ticker: " + ticker,
			Ticker:    ticker,
		}
		auditFile.APIRequests = append([]AuditEntry{initEntry}, auditFile.APIRequests...)
		logger.Warn.Printf("🎯 AUDIT INIT: Initialized audit for ticker %s", ticker)
	}

	// Step 2: APPEND - Add new audit data if provided
	if operation != "" && data != nil {
		entry := oaal.convertDataToEntry(operation, data)
		if entry != nil {
			auditFile.APIRequests = append(auditFile.APIRequests, *entry)
			logger.Debug.Printf("📝 AUDIT APPEND: Added %s entry", operation)
		}
	}

	// Step 3: Calculate summary
	oaal.updateSummary(auditFile)

	// Step 4: Save file
	if err := oaal.writeAuditFile(auditFile); err != nil {
		return fmt.Errorf("failed to write audit file: %v", err)
	}

	// Step 5: FINISH - Create markdown summary and archive both files
	if operation == "finish" {
		if ticker != "" {
			if err := oaal.createMarkdownSummary(ticker, auditFile, data); err != nil {
				logger.Warn.Printf("⚠️ Failed to create markdown summary: %v", err)
			}
			if err := oaal.archiveAuditFile(ticker); err != nil {
				logger.Warn.Printf("⚠️ Failed to archive audit file: %v", err)
			}
		}
	} else if operation == "complete" || operation == "archive" {
		// Step 5: RENAME - Archive file only for complete/archive operations
		if ticker != "" {
			if err := oaal.archiveAuditFile(ticker); err != nil {
				logger.Warn.Printf("⚠️ Failed to archive audit file: %v", err)
			}
		}
	}

	return nil
}

// hasInitEntry checks if there's already an initialization entry for this ticker
func (oaal *OptionsAnalysisAuditLogger) hasInitEntry(auditFile *AuditFile, ticker string) bool {
	for _, entry := range auditFile.APIRequests {
		if entry.Type == "TickerAuditInitialization" && entry.Ticker == ticker {
			return true
		}
	}
	return false
}

// convertDataToEntry converts various data types to AuditEntry with Infinity error handling
func (oaal *OptionsAnalysisAuditLogger) convertDataToEntry(operation string, data interface{}) *AuditEntry {
	entry := &AuditEntry{
		Type:      operation,
		Timestamp: time.Now().Format(time.RFC3339),
		Success:   true,
	}

	// Check for Infinity values in the data before processing
	if infinityErrors := oaal.checkForInfinity(data); len(infinityErrors) > 0 {
		entry.Error = fmt.Sprintf("Infinity values detected: %v", infinityErrors)
		entry.Success = false
		logger.Warn.Printf("🚨 AUDIT INFINITY ERROR: %s", entry.Error)
	}

	// Additional validation for options calculations
	if optionsErrors := oaal.validateOptionsCalculation(data); len(optionsErrors) > 0 {
		if entry.Error != "" {
			entry.Error += "; "
		}
		entry.Error += fmt.Sprintf("Options validation errors: %v", optionsErrors)
		entry.Success = false
		logger.Warn.Printf("🚨 AUDIT OPTIONS ERROR: %v", optionsErrors)
	}

	switch d := data.(type) {
	case map[string]interface{}:
		entry.Response = d
		if url, ok := d["url"].(string); ok {
			entry.URL = url
		}
		if method, ok := d["method"].(string); ok {
			entry.Method = method
		}
		if duration, ok := d["duration_ms"].(int64); ok {
			entry.Duration = duration
		}
		if errorMsg, ok := d["error"].(string); ok && errorMsg != "" {
			entry.Error = errorMsg
			entry.Success = false
		}
	case string:
		entry.Message = d
	default:
		// Convert to JSON and store in response
		if jsonData, err := json.Marshal(data); err == nil {
			var responseData map[string]interface{}
			if json.Unmarshal(jsonData, &responseData) == nil {
				entry.Response = responseData
			}
		}
	}

	return entry
}

// checkForInfinity recursively checks data for Infinity values in options calculations
func (oaal *OptionsAnalysisAuditLogger) checkForInfinity(data interface{}) []string {
	var infinityErrors []string

	switch v := data.(type) {
	case float64:
		if math.IsInf(v, 0) {
			infinityErrors = append(infinityErrors, fmt.Sprintf("float64 value: %v", v))
		}
	case float32:
		if math.IsInf(float64(v), 0) {
			infinityErrors = append(infinityErrors, fmt.Sprintf("float32 value: %v", v))
		}
	case map[string]interface{}:
		for key, val := range v {
			if errs := oaal.checkForInfinity(val); len(errs) > 0 {
				for _, err := range errs {
					infinityErrors = append(infinityErrors, fmt.Sprintf("%s.%s", key, err))
				}
			}
		}
	case []interface{}:
		for i, val := range v {
			if errs := oaal.checkForInfinity(val); len(errs) > 0 {
				for _, err := range errs {
					infinityErrors = append(infinityErrors, fmt.Sprintf("[%d].%s", i, err))
				}
			}
		}
	default:
		// Use reflection to handle struct types that might contain float fields
		val := reflect.ValueOf(data)
		if val.Kind() == reflect.Ptr && !val.IsNil() {
			val = val.Elem()
		}
		if val.Kind() == reflect.Struct {
			typ := val.Type()
			for i := 0; i < val.NumField(); i++ {
				field := val.Field(i)
				fieldName := typ.Field(i).Name
				if field.CanInterface() {
					if errs := oaal.checkForInfinity(field.Interface()); len(errs) > 0 {
						for _, err := range errs {
							infinityErrors = append(infinityErrors, fmt.Sprintf("%s.%s", fieldName, err))
						}
					}
				}
			}
		}
	}

	return infinityErrors
}

// validateOptionsCalculation specifically checks for common Infinity errors in Black-Scholes calculations
func (oaal *OptionsAnalysisAuditLogger) validateOptionsCalculation(data interface{}) []string {
	var errors []string

	// Check if this looks like options calculation data
	if m, ok := data.(map[string]interface{}); ok {
		// Check common Black-Scholes fields that can produce Infinity
		fieldsToCheck := []string{
			"call_price", "put_price", "delta", "gamma", "theta", "vega", "rho",
			"implied_volatility", "time_value", "intrinsic_value", "premium",
			"bid", "ask", "mark", "strike", "spot_price", "risk_free_rate",
		}

		for _, field := range fieldsToCheck {
			if val, exists := m[field]; exists {
				if floatVal, ok := val.(float64); ok {
					if math.IsInf(floatVal, 0) {
						errors = append(errors, fmt.Sprintf("Options calculation field '%s' has Infinity value", field))
					} else if math.IsNaN(floatVal) {
						errors = append(errors, fmt.Sprintf("Options calculation field '%s' has NaN value", field))
					} else if floatVal < -1e10 || floatVal > 1e10 {
						errors = append(errors, fmt.Sprintf("Options calculation field '%s' has extreme value: %e", field, floatVal))
					}
				}
			}
		}

		// Specific Black-Scholes validation
		if strike, hasStrike := m["strike"].(float64); hasStrike {
			if spot, hasSpot := m["spot_price"].(float64); hasSpot {
				if strike <= 0 {
					errors = append(errors, "Strike price must be positive")
				}
				if spot <= 0 {
					errors = append(errors, "Spot price must be positive")
				}
			}
		}

		if vol, hasVol := m["implied_volatility"].(float64); hasVol {
			if vol < 0 {
				errors = append(errors, "Implied volatility cannot be negative")
			}
			if vol > 10 { // 1000% volatility is unrealistic
				errors = append(errors, fmt.Sprintf("Implied volatility suspiciously high: %.2f%%", vol*100))
			}
		}
	}

	return errors
}

// updateSummary calculates and updates audit summary statistics
func (oaal *OptionsAnalysisAuditLogger) updateSummary(auditFile *AuditFile) {
	totalRequests := len(auditFile.APIRequests)
	successCount := 0
	errorCount := 0
	totalTime := int64(0)

	for _, entry := range auditFile.APIRequests {
		if entry.Success {
			successCount++
		} else {
			errorCount++
		}
		totalTime += entry.Duration
	}

	auditFile.Summary.TotalRequests = totalRequests
	auditFile.Summary.SuccessCount = successCount
	auditFile.Summary.ErrorCount = errorCount
	auditFile.Summary.TotalTimeMs = totalTime
}

// writeAuditFile writes the audit data to the JSON file
func (oaal *OptionsAnalysisAuditLogger) writeAuditFile(auditFile *AuditFile) error {
	file, err := os.Create(oaal.auditFilename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(auditFile)
}

// archiveAuditFile moves the audit file to the audits directory with timestamp
func (oaal *OptionsAnalysisAuditLogger) archiveAuditFile(ticker string) error {
	// If no ticker provided, try to read it from the audit file
	if ticker == "" {
		if file, err := os.Open(oaal.auditFilename); err == nil {
			defer file.Close()
			var auditFile AuditFile
			if json.NewDecoder(file).Decode(&auditFile) == nil && auditFile.Ticker != "" {
				ticker = auditFile.Ticker
			}
		}
		// Fallback to "UNKNOWN" if still no ticker
		if ticker == "" {
			ticker = "UNKNOWN"
		}
	}

	auditDir := "audits"
	if err := os.MkdirAll(auditDir, 0755); err != nil {
		return err
	}

	timestamp := time.Now().Format("2006-01-02_15-04-05")
	destination := fmt.Sprintf("%s/audit_%s_%s.json", auditDir, ticker, timestamp)

	if err := os.Rename(oaal.auditFilename, destination); err != nil {
		return err
	}

	logger.Warn.Printf("📁 AUDIT ARCHIVE: Moved %s to %s", oaal.auditFilename, destination)
	return nil
}

// createMarkdownSummary creates a markdown summary file for AI analysis
func (oaal *OptionsAnalysisAuditLogger) createMarkdownSummary(ticker string, auditFile *AuditFile, analysisData interface{}) error {
	auditDir := "audits"
	if err := os.MkdirAll(auditDir, 0755); err != nil {
		return fmt.Errorf("failed to create audit directory: %v", err)
	}

	timestamp := time.Now().Format("2006-01-02_15-04-05")
	filename := fmt.Sprintf("%s/audit_%s_%s.md", auditDir, ticker, timestamp)
	logger.Warn.Printf("📋 AUDIT: Creating markdown summary: %s", filename)

	// Extract AI analysis from the data
	var markdownContent string
	if data, ok := analysisData.(map[string]interface{}); ok {
		if analysis, exists := data["ai_analysis"]; exists {
			analysisText := fmt.Sprintf("%v", analysis)
			// Check if this is already formatted markdown (contains # headers)
			if strings.HasPrefix(analysisText, "# Grok AI Analysis") {
				// Use the pre-formatted markdown with audit data appended
				auditDataBytes, err := json.MarshalIndent(auditFile, "", "  ")
				if err != nil {
					return fmt.Errorf("failed to marshal audit data: %v", err)
				}

				markdownContent = analysisText + fmt.Sprintf("\n\n## Audit Data\n\n<details>\n<summary>Click to view detailed audit data</summary>\n\n```json\n%s\n```\n\n</details>", string(auditDataBytes))
			} else {
				// Fallback to old template format
				auditDataBytes, err := json.MarshalIndent(auditFile, "", "  ")
				if err != nil {
					return fmt.Errorf("failed to marshal audit data: %v", err)
				}

				markdownContent = fmt.Sprintf("# Grok AI Analysis - %s\n\n**Generated:** %s\n**Ticker:** %s\n\n## AI Analysis\n\n%s\n\n## Audit Data\n\n<details>\n<summary>Click to view detailed audit data</summary>\n\n```json\n%s\n```\n\n</details>\n\n---\n*Generated by Barracuda Options Analysis System with Grok AI*\n",
					ticker, time.Now().Format("2006-01-02 15:04:05"), ticker, analysisText, string(auditDataBytes))
			}
		}
	}

	if markdownContent == "" {
		// No analysis data found, create basic template
		auditDataBytes, err := json.MarshalIndent(auditFile, "", "  ")
		if err != nil {
			return fmt.Errorf("failed to marshal audit data: %v", err)
		}

		markdownContent = fmt.Sprintf("# Analysis - %s\n\n**Generated:** %s\n**Ticker:** %s\n\n## Analysis\n\nNo analysis available\n\n## Audit Data\n\n<details>\n<summary>Click to view detailed audit data</summary>\n\n```json\n%s\n```\n\n</details>\n\n---\n*Generated by Barracuda Options Analysis System*\n",
			ticker, time.Now().Format("2006-01-02 15:04:05"), ticker, string(auditDataBytes))
	}

	// Write markdown file
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create markdown file: %v", err)
	}
	defer file.Close()

	if _, err := file.WriteString(markdownContent); err != nil {
		return fmt.Errorf("failed to write markdown content: %v", err)
	}

	logger.Warn.Printf("📋 AUDIT: Markdown summary created: %s", filename)
	return nil
}
