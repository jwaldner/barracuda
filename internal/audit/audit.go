package audit

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/jwaldner/barracuda/internal/logger"
)

// AuditManager handles audit data collection across all layers
type AuditManager struct {
	ticker    string
	auditData map[string]interface{}
	mutex     sync.RWMutex
	isActive  bool
}

var globalAudit *AuditManager
var auditMutex sync.RWMutex

// Init initializes audit for a specific ticker - overwrites any existing audit.json
func Init(ticker string) {
	auditMutex.Lock()
	defer auditMutex.Unlock()

	if ticker == "" {
		globalAudit = nil
		return
	}

	globalAudit = &AuditManager{
		ticker:    ticker,
		auditData: make(map[string]interface{}),
		isActive:  true,
	}

	// Initialize with basic info
	globalAudit.auditData["ticker"] = ticker
	globalAudit.auditData["audit_started"] = time.Now().Format(time.RFC3339)

	// Overwrite audit.json file
	globalAudit.writeToFile()

	logger.Warn.Printf("🔍 AUDIT: Initialized for ticker %s", ticker)
}

// IsActive returns true if audit is currently active for any ticker
func IsActive() bool {
	auditMutex.RLock()
	defer auditMutex.RUnlock()
	return globalAudit != nil && globalAudit.isActive
}

// GetTicker returns the current audit ticker
func GetTicker() string {
	auditMutex.RLock()
	defer auditMutex.RUnlock()
	if globalAudit == nil {
		return ""
	}
	return globalAudit.ticker
}

// AddSection adds audit data from a specific layer
func AddSection(layerName string, data interface{}) {
	auditMutex.Lock()
	defer auditMutex.Unlock()

	if globalAudit == nil || !globalAudit.isActive {
		return
	}

	globalAudit.mutex.Lock()
	defer globalAudit.mutex.Unlock()

	// Add layer data with timestamp
	layerData := map[string]interface{}{
		"timestamp": time.Now().Format(time.RFC3339),
		"data":      data,
	}

	globalAudit.auditData[layerName] = layerData

	// Write updated data to file
	globalAudit.writeToFile()

	logger.Warn.Printf("🔍 AUDIT: Added %s layer data for %s", layerName, globalAudit.ticker)
}

// AddMetrics adds simple key-value metrics from any layer
func AddMetrics(layerName string, metrics map[string]interface{}) {
	auditMutex.Lock()
	defer auditMutex.Unlock()

	if globalAudit == nil || !globalAudit.isActive {
		return
	}

	globalAudit.mutex.Lock()
	defer globalAudit.mutex.Unlock()

	// If layer already exists, merge metrics
	if existing, exists := globalAudit.auditData[layerName]; exists {
		if layerMap, ok := existing.(map[string]interface{}); ok {
			if data, ok := layerMap["data"].(map[string]interface{}); ok {
				for k, v := range metrics {
					data[k] = v
				}
			} else {
				layerMap["data"] = metrics
			}
			layerMap["last_updated"] = time.Now().Format(time.RFC3339)
		}
	} else {
		// Create new layer entry
		globalAudit.auditData[layerName] = map[string]interface{}{
			"timestamp": time.Now().Format(time.RFC3339),
			"data":      metrics,
		}
	}

	// Write updated data to file
	globalAudit.writeToFile()

	logger.Warn.Printf("🔍 AUDIT: Added metrics to %s layer for %s", layerName, globalAudit.ticker)
}

// Finalize completes the audit and moves files to audits folder
func Finalize() {
	auditMutex.Lock()
	defer auditMutex.Unlock()

	if globalAudit == nil || !globalAudit.isActive {
		return
	}

	globalAudit.mutex.Lock()
	defer globalAudit.mutex.Unlock()

	// Add completion timestamp
	globalAudit.auditData["audit_completed"] = time.Now().Format(time.RFC3339)

	// Write final version
	globalAudit.writeToFile()

	// Move to audits folder with timestamp
	timestamp := time.Now().Format("2006-01-02_15-04-05")

	// Ensure audits directory exists
	if err := os.MkdirAll("audits", 0755); err != nil {
		logger.Error.Printf("❌ AUDIT: Failed to create audits directory: %v", err)
		return
	}

	// Move JSON file
	jsonDest := fmt.Sprintf("audits/audit_%s_%s.json", globalAudit.ticker, timestamp)
	if err := os.Rename("audit.json", jsonDest); err != nil {
		logger.Error.Printf("❌ AUDIT: Failed to move JSON file: %v", err)
	} else {
		logger.Warn.Printf("🔍 AUDIT: Moved JSON to %s", jsonDest)
	}

	// Create markdown summary
	markdownDest := fmt.Sprintf("audits/audit_%s_%s.md", globalAudit.ticker, timestamp)
	globalAudit.createMarkdownSummary(markdownDest)

	logger.Warn.Printf("🔍 AUDIT: Finalized for ticker %s", globalAudit.ticker)

	// Deactivate
	globalAudit.isActive = false
	globalAudit = nil
}

// writeToFile writes current audit data to audit.json
func (am *AuditManager) writeToFile() {
	file, err := os.Create("audit.json")
	if err != nil {
		logger.Error.Printf("❌ AUDIT: Failed to create audit.json: %v", err)
		return
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")

	if err := encoder.Encode(am.auditData); err != nil {
		logger.Error.Printf("❌ AUDIT: Failed to write JSON: %v", err)
		return
	}
}

// createMarkdownSummary creates a markdown summary of the audit
func (am *AuditManager) createMarkdownSummary(filename string) {
	content := fmt.Sprintf("# Audit Summary - %s\n\n", am.ticker)
	content += fmt.Sprintf("**Generated:** %s\n\n", time.Now().Format("2006-01-02 15:04:05"))

	// Add summary of each layer
	for layerName, layerData := range am.auditData {
		if layerName == "ticker" || layerName == "audit_started" || layerName == "audit_completed" {
			continue
		}

		content += fmt.Sprintf("## %s Layer\n\n", layerName)

		if layerMap, ok := layerData.(map[string]interface{}); ok {
			if timestamp, ok := layerMap["timestamp"]; ok {
				content += fmt.Sprintf("**Timestamp:** %s\n\n", timestamp)
			}

			content += "```json\n"
			if data, err := json.MarshalIndent(layerMap["data"], "", "  "); err == nil {
				content += string(data)
			}
			content += "\n```\n\n"
		}
	}

	content += "---\n*Generated by Barracuda Audit System*\n"

	if err := os.WriteFile(filename, []byte(content), 0644); err != nil {
		logger.Error.Printf("❌ AUDIT: Failed to create markdown: %v", err)
	} else {
		logger.Warn.Printf("🔍 AUDIT: Created markdown summary: %s", filename)
	}
}
