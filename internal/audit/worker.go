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
	Type   string      `json:"type"`   // "append", "analyze"
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
	if logger.Debug != nil {
		logger.Debug.Printf("🔍 AUDIT: LogOptionsAnalysisOperation called - ticker='%s', operation='%s'", ticker, operation)
	}

	// Extract expiry date from data if available
	var expiry string
	if dataMap, ok := data.(map[string]interface{}); ok {
		if exp, exists := dataMap["expiration"]; exists {
			if expStr, ok := exp.(string); ok {
				expiry = expStr
			}
		}
	}

	var action AuditAction

	switch operation {
	case "finish", "complete", "archive":
		action = AuditAction{Type: "analyze", Data: data}
	default:
		// Send append message with ticker and expiry info
		auditData := map[string]interface{}{
			"operation": operation,
			"ticker":    ticker,
			"expiry":    expiry,
			"data":      data,
		}
		action = AuditAction{
			Type:   "append",
			Ticker: ticker,
			Expiry: expiry,
			Data:   auditData,
		}
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
		case "append":
			ticker := action.Ticker
			expiry := action.Expiry
			data := action.Data

			// Try to read existing audit file
			file, err := os.Open(auditFileName)
			var currentFile AuditFile
			fileExists := err == nil

			if fileExists {
				if decodeErr := json.NewDecoder(file).Decode(&currentFile); decodeErr != nil {
					logger.Warn.Printf("⚠️ AUDIT: Corrupted audit.json, treating as non-existent: %v", decodeErr)
					fileExists = false
				}
				file.Close()
			}

			// NEW AUDIT DETECTION: If ticker is provided and differs from current audit
			if ticker != "" && fileExists && currentFile.Header.Ticker != "" && currentFile.Header.Ticker != ticker {
				// Archive existing audit.json using configurable format
				if err := os.MkdirAll("audits", 0755); err != nil {
					logger.Warn.Printf("⚠️ AUDIT: Failed to create audits directory: %v", err)
				}
				auditConfig := config.GetAuditConfig()
				baseName := config.FormatAuditFilename(auditConfig.FilenameFormat, currentFile.Header.Ticker, currentFile.Header.ExpiryDate, "")
				archiveName := fmt.Sprintf("audits/%s.json", baseName)
				if err := os.Rename(auditFileName, archiveName); err != nil {
					if logger.Warn != nil {
						logger.Warn.Printf("⚠️ AUDIT: Failed to archive %s to %s: %v", auditFileName, archiveName, err)
					}
				} else {
					if logger.Warn != nil {
						logger.Warn.Printf("📁 AUDIT: Archived %s to %s (new audit detected)", auditFileName, archiveName)
					}
				}
				fileExists = false // Force creation of new audit
			}

			// Create new audit if none exists or new ticker detected
			if !fileExists && ticker != "" {
				currentFile = AuditFile{
					Header: AuditHeader{
						Ticker:     ticker,
						ExpiryDate: expiry,
						StartTime:  time.Now(),
					},
					Entries: []map[string]interface{}{},
				}
				if logger.Warn != nil {
					logger.Warn.Printf("📝 AUDIT: Creating new audit.json for %s-%s", ticker, expiry)
				}
			} else if !fileExists {
				if logger.Warn != nil {
					logger.Warn.Printf("⚠️ AUDIT: No audit.json exists and no ticker provided - skipping entry")
				}
				continue
			}

			// Add new entry to audit
			if entryMap, ok := data.(map[string]interface{}); ok {
				entryMap["timestamp"] = time.Now().Format(time.RFC3339)
				currentFile.Entries = append(currentFile.Entries, entryMap)
			} else {
				entry := map[string]interface{}{
					"timestamp": time.Now().Format(time.RFC3339),
					"data":      data,
				}
				currentFile.Entries = append(currentFile.Entries, entry)
			}

			// Write updated audit file
			outFile, err := os.Create(auditFileName)
			if err != nil {
				if logger.Warn != nil {
					logger.Warn.Printf("⚠️ AUDIT: Failed to create %s: %v", auditFileName, err)
				}
				continue
			}

			encoder := json.NewEncoder(outFile)
			encoder.SetIndent("", "  ")
			if err := encoder.Encode(currentFile); err != nil {
				if logger.Warn != nil {
					logger.Warn.Printf("⚠️ AUDIT: Failed to write %s: %v", auditFileName, err)
				}
				outFile.Close()
				continue
			}
			outFile.Close()

			if logger.Debug != nil {
				logger.Debug.Printf("📝 AUDIT: Added entry to %s-%s (total: %d)",
					currentFile.Header.Ticker, currentFile.Header.ExpiryDate, len(currentFile.Entries))
			}

		case "analyze":
			// Analysis operation - extract Grok content and create .md file
			var auditFile AuditFile
			file, err := os.Open(auditFileName)
			if err != nil {
				if logger.Warn != nil {
					logger.Warn.Printf("⚠️ AUDIT: Failed to open %s for analysis: %v", auditFileName, err)
				}
				continue
			}

			if err := json.NewDecoder(file).Decode(&auditFile); err != nil {
				if logger.Warn != nil {
					logger.Warn.Printf("⚠️ AUDIT: Failed to decode %s for analysis: %v", auditFileName, err)
				}
				file.Close()
				continue
			}
			file.Close()

			ticker := auditFile.Header.Ticker
			expiry := auditFile.Header.ExpiryDate

			// Create audits directory
			if err := os.MkdirAll("audits", 0755); err != nil {
				if logger.Warn != nil {
					logger.Warn.Printf("⚠️ AUDIT: Failed to create audits directory: %v", err)
				}
			}

			// Extract Grok analysis content from action data
			var markdownContent string
			if action.Data != nil {
				if dataMap, ok := action.Data.(map[string]interface{}); ok {
					if grokAnalysis, exists := dataMap["grok_analysis"]; exists && grokAnalysis != nil {
						markdownContent = grokAnalysis.(string)
					}
				}
			}

			// Fallback to generic content if no Grok analysis found
			if markdownContent == "" {
				markdownContent = fmt.Sprintf("# Audit Analysis - %s (Expiry: %s)\n\n**Generated:** %s\n\n## Analysis\n\nNo Grok analysis content available.\n\n---\n*Generated by Barracuda Audit System*\n",
					ticker, expiry, time.Now().Format("2006-01-02 15:04:05"))
			}

			// Save markdown in audits directory using same base name as JSON
			auditConfig := config.GetAuditConfig()
			baseName := config.FormatAuditFilename(auditConfig.FilenameFormat, ticker, expiry, "")
			mdName := fmt.Sprintf("audits/%s.md", baseName)
			if err := os.WriteFile(mdName, []byte(markdownContent), 0644); err != nil {
				logger.Warn.Printf("⚠️ AUDIT: Failed to create analysis file %s: %v", mdName, err)
			} else {
				logger.Warn.Printf("📋 AUDIT: Created analysis %s", mdName)
			}

			// Move audit.json to audits directory after analysis using same base name
			archiveName := fmt.Sprintf("audits/%s.json", baseName)
			if err := os.Rename(auditFileName, archiveName); err != nil {
				logger.Warn.Printf("⚠️ AUDIT: Failed to move %s to %s: %v", auditFileName, archiveName, err)
			} else {
				logger.Warn.Printf("📁 AUDIT: Moved %s to %s (analysis complete)", auditFileName, archiveName)
			}
		}
	}
}
