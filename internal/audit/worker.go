package audit

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/jwaldner/barracuda/internal/config"
	"github.com/jwaldner/barracuda/internal/logger"
)

// AuditAction represents operations sent to the audit channel
type AuditAction struct {
	Type   string      `json:"type"`   // "create_audit", "append_entry", "analysis_result"
	Ticker string      `json:"ticker"` // ticker symbol
	Expiry string      `json:"expiry"` // expiry date
	Data   interface{} `json:"data"`   // audit data
}

// AuditHeader represents the header stored in audit.json
type AuditHeader struct {
	Ticker     string    `json:"ticker"`
	ExpiryDate string    `json:"expiry_date,omitempty"`
	Strategy   string    `json:"strategy,omitempty"`
	StartTime  time.Time `json:"start_time"`
}

// AuditFile represents the complete audit file structure
type AuditFile struct {
	Header  AuditHeader              `json:"header"`
	Entries []map[string]interface{} `json:"entries"`
}

// OptionsAnalysisAuditLogger implements the OptionsAnalysisAuditor interface
type OptionsAnalysisAuditLogger struct {
	auditChan chan AuditAction
}

var globalAuditChan chan AuditAction

// NewOptionsAnalysisAuditLogger creates a new options analysis audit logger
func NewOptionsAnalysisAuditLogger() OptionsAnalysisAuditor {
	if globalAuditChan == nil {
		globalAuditChan = make(chan AuditAction, 100)
		go auditWorker(globalAuditChan)
	}

	return &OptionsAnalysisAuditLogger{
		auditChan: globalAuditChan,
	}
}

// LogOptionsAnalysisOperation handles audit operations via channel
func (oaal *OptionsAnalysisAuditLogger) LogOptionsAnalysisOperation(ticker string, operation string, data interface{}) error {
	logger.Debug.Printf("🔍 AUDIT: Operation='%s', Ticker='%s'", operation, ticker)

	// Extract expiry from data
	var expiry string
	if dataMap, ok := data.(map[string]interface{}); ok {
		if exp, exists := dataMap["expiration"]; exists && exp != nil {
			if expStr, ok := exp.(string); ok {
				expiry = expStr
				logger.Debug.Printf("🔍 AUDIT: Extracted expiry='%s' from data", expiry)
			}
		} else {
			logger.Debug.Printf("🔍 AUDIT: No 'expiration' key found in data: %+v", dataMap)
		}
	} else {
		logger.Debug.Printf("🔍 AUDIT: Data is not a map: %T %+v", data, data)
	}

	var action AuditAction

	switch operation {
	case "create":
		action = AuditAction{Type: "create_audit", Ticker: ticker, Expiry: expiry, Data: data}
	case "analysis_result":
		action = AuditAction{Type: "analysis_result", Data: data}
	default:
		action = AuditAction{Type: "append_entry", Ticker: ticker, Expiry: expiry, Data: data}
	}

	// VALIDATE: Only 3 action types allowed
	if action.Type != "create_audit" && action.Type != "append_entry" && action.Type != "analysis_result" {
		return fmt.Errorf("INVALID ACTION TYPE: %s", action.Type)
	}

	select {
	case oaal.auditChan <- action:
		return nil
	default:
		return fmt.Errorf("audit channel full")
	}
}

// auditWorker processes all audit operations in a single goroutine - OWNS ALL FILE OPERATIONS
func auditWorker(ch chan AuditAction) {
	const auditFileName = "audit.json"

	for action := range ch {
		switch action.Type {
		case "create_audit":
			ticker := action.Ticker
			expiry := action.Expiry

			if ticker == "" {
				logger.Warn.Printf("⚠️ AUDIT: Cannot create audit without ticker")
				continue
			}

			// Check if audit.json exists and move it to audits/ using ticker from header
			if _, err := os.Stat(auditFileName); err == nil {
				var existingFile AuditFile
				if data, readErr := os.ReadFile(auditFileName); readErr == nil {
					if json.Unmarshal(data, &existingFile) == nil && existingFile.Header.Ticker != "" {
						// Move existing audit using ticker from its header
						if err := os.MkdirAll("audits", 0755); err == nil {
							auditConfig := config.GetAuditConfig()
							timestamp := time.Now().Format("2006-01-02_15-04-05")
							baseName := config.FormatAuditFilename(auditConfig.FilenameFormat, existingFile.Header.Ticker, existingFile.Header.ExpiryDate, timestamp)
							archiveName := fmt.Sprintf("audits/%s.json", baseName)
							if os.Rename(auditFileName, archiveName) == nil {
								logger.Warn.Printf("📁 AUDIT: Moved existing audit to %s", archiveName)
							}
						}
					}
				}
			}

			// Create new audit file
			newAudit := AuditFile{
				Header: AuditHeader{
					Ticker:     ticker,
					ExpiryDate: expiry,
					StartTime:  time.Now(),
				},
				Entries: []map[string]interface{}{},
			}

			if data, err := json.MarshalIndent(newAudit, "", "  "); err == nil {
				if err := os.WriteFile(auditFileName, data, 0644); err == nil {
					logger.Warn.Printf("📝 AUDIT: Created new audit for %s-%s", ticker, expiry)
				} else {
					logger.Warn.Printf("⚠️ AUDIT: Failed to write audit file: %v", err)
				}
			}

		case "append_entry":
			// Read existing audit file - FAIL if it doesn't exist!
			var currentFile AuditFile
			auditData, err := os.ReadFile(auditFileName)
			if err != nil {
				logger.Warn.Printf("⚠️ AUDIT: FAILED - No audit.json exists for append_entry: %v", err)
				continue
			}

			if err := json.Unmarshal(auditData, &currentFile); err != nil {
				logger.Warn.Printf("⚠️ AUDIT: FAILED - Corrupted audit.json: %v", err)
				continue
			}

			// Verify ticker matches (if provided)
			if action.Ticker != "" && currentFile.Header.Ticker != action.Ticker {
				logger.Warn.Printf("⚠️ AUDIT: FAILED - Ticker mismatch: existing=%s, new=%s", currentFile.Header.Ticker, action.Ticker)
				continue
			}

			// Add entry with timestamp to EXISTING file only
			entry := map[string]interface{}{
				"timestamp": time.Now().Format(time.RFC3339),
				"data":      action.Data,
			}
			currentFile.Entries = append(currentFile.Entries, entry)

			// Write back to file
			if data, err := json.MarshalIndent(currentFile, "", "  "); err == nil {
				if err := os.WriteFile(auditFileName, data, 0644); err == nil {
					if logger.Debug != nil {
						logger.Debug.Printf("📝 AUDIT: Added entry (total: %d)", len(currentFile.Entries))
					}
				} else {
					logger.Warn.Printf("⚠️ AUDIT: Failed to write audit file: %v", err)
				}
			}

		case "analysis_result":
			// Read audit.json to get ticker info
			auditData, err := os.ReadFile(auditFileName)
			if err != nil {
				logger.Warn.Printf("⚠️ AUDIT: No audit.json found for analysis")
				continue
			}

			var auditFile AuditFile
			if err := json.Unmarshal(auditData, &auditFile); err != nil {
				logger.Warn.Printf("⚠️ AUDIT: Corrupted audit.json: %v", err)
				continue
			}

			ticker := auditFile.Header.Ticker
			expiry := auditFile.Header.ExpiryDate

			// Create audits directory
			if err := os.MkdirAll("audits", 0755); err != nil {
				logger.Warn.Printf("⚠️ AUDIT: Failed to create audits directory: %v", err)
				continue
			}

			// Generate filename using config format
			auditConfig := config.GetAuditConfig()
			timestamp := time.Now().Format("2006-01-02_15-04-05")
			baseName := config.FormatAuditFilename(auditConfig.FilenameFormat, ticker, expiry, timestamp)

			// Move JSON to audits
			jsonName := fmt.Sprintf("audits/%s.json", baseName)
			if err := os.Rename(auditFileName, jsonName); err != nil {
				logger.Warn.Printf("⚠️ AUDIT: Failed to move audit.json: %v", err)
				continue
			}
			logger.Verbose.Printf("📁 AUDIT: Moved audit to %s", jsonName)

			// Extract Grok analysis from data and create markdown
			var grokResult string
			if action.Data != nil {
				if dataMap, ok := action.Data.(map[string]interface{}); ok {
					if result, exists := dataMap["grok_result"]; exists && result != nil {
						grokResult = result.(string)
					}
				}
			}

			// Create markdown with grok result
			if grokResult != "" {
				mdName := fmt.Sprintf("audits/%s.md", baseName)
				if err := os.WriteFile(mdName, []byte(grokResult), 0644); err != nil {
					logger.Warn.Printf("⚠️ AUDIT: Failed to create markdown: %v", err)
				} else {
					logger.Verbose.Printf("📋 AUDIT: Created analysis %s", mdName)
				}
			}
		default:
			logger.Warn.Printf("⚠️ AUDIT: INVALID ACTION TYPE '%s' - ONLY create_audit, append_entry, analysis_result ALLOWED", action.Type)
		}
	}
}
