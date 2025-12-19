package audit

import (
	"encoding/json"
	"os"
	"time"

	"github.com/jwaldner/barracuda/internal/models"
)

// Auditable defines the interface for components that can be audited
type Auditable interface {
	// Audit performs audit operations for the specified ticker
	// Returns audit data that can be serialized to JSON
	Audit(ticker string, request models.AnalysisRequest) (interface{}, error)
}

// AuditManager handles audit operations across different components
type AuditCoordinator struct {
	Components map[string]Auditable
}

// NewAuditCoordinator creates a new audit coordinator
func NewAuditCoordinator() *AuditCoordinator {
	ac := &AuditCoordinator{
		Components: make(map[string]Auditable),
	}
	
	// Try to append initialization message to existing audit.json, don't create new file
	ac.appendInitialization()
	
	return ac
}

// appendInitialization tries to add initialization message to existing audit.json
func (ac *AuditCoordinator) appendInitialization() {
	// Only append if audit.json already exists - don't create it
	if _, err := os.Stat("audit.json"); os.IsNotExist(err) {
		return // File doesn't exist, don't create it
	}
	
	// Read existing file
	data, err := os.ReadFile("audit.json")
	if err != nil {
		return // Can't read, skip
	}
	
	var auditData map[string]interface{}
	if err := json.Unmarshal(data, &auditData); err != nil {
		return // Invalid JSON, skip
	}
	
	// Add initialization message to api_requests array
	if apiRequests, ok := auditData["api_requests"].([]interface{}); ok {
		initMessage := map[string]interface{}{
			"type":      "AuditManagerInitialization",
			"message":   "AuditManager initialized and ready for component registration",
			"timestamp": time.Now().Format(time.RFC3339),
		}
		
		auditData["api_requests"] = append(apiRequests, initMessage)
		
		// Update summary
		if summary, ok := auditData["summary"].(map[string]interface{}); ok {
			if totalRequests, ok := summary["total_requests"].(float64); ok {
				summary["total_requests"] = totalRequests + 1
			}
			if successCount, ok := summary["success_count"].(float64); ok {
				summary["success_count"] = successCount + 1
			}
		}
		
		// Write back to file
		if file, err := os.Create("audit.json"); err == nil {
			defer file.Close()
			encoder := json.NewEncoder(file)
			encoder.SetIndent("", "  ")
			encoder.Encode(auditData)
		}
	}
}

// Register adds an auditable component with a given name
func (ac *AuditCoordinator) Register(name string, auditable Auditable) {
	ac.Components[name] = auditable
}

// AuditTicker performs audit for a specific ticker across all registered components
func (ac *AuditCoordinator) AuditTicker(ticker string, request models.AnalysisRequest) (map[string]interface{}, error) {
	// First, add a clear ticker audit initialization message
	ac.logTickerAuditInit(ticker)
	
	auditData := map[string]interface{}{
		"ticker":     ticker,
		"timestamp":  time.Now().Format(time.RFC3339),
		"request":    request,
		"components": make(map[string]interface{}),
	}

	for name, auditable := range ac.Components {
		componentData, err := auditable.Audit(ticker, request)
		if err != nil {
			return nil, err
		}
		auditData["components"].(map[string]interface{})[name] = componentData
	}

	return auditData, nil
}

// logTickerAuditInit adds a clear message showing we're starting to audit a specific ticker
func (ac *AuditCoordinator) logTickerAuditInit(ticker string) {
	// Only append if audit.json already exists - don't create it
	if _, err := os.Stat("audit.json"); os.IsNotExist(err) {
		return // File doesn't exist, don't create it
	}
	
	// Read existing file
	data, err := os.ReadFile("audit.json")
	if err != nil {
		return // Can't read, skip
	}
	
	var auditData map[string]interface{}
	if err := json.Unmarshal(data, &auditData); err != nil {
		return // Invalid JSON, skip
	}
	
	// Add ticker audit initialization message to api_requests array
	if apiRequests, ok := auditData["api_requests"].([]interface{}); ok {
		tickerInitMessage := map[string]interface{}{
			"type":      "TickerAuditInitialization",
			"message":   "🎯 AUDIT INIT: Now auditing ticker: " + ticker,
			"ticker":    ticker,
			"timestamp": time.Now().Format(time.RFC3339),
		}
		
		auditData["api_requests"] = append(apiRequests, tickerInitMessage)
		
		// Update summary
		if summary, ok := auditData["summary"].(map[string]interface{}); ok {
			if totalRequests, ok := summary["total_requests"].(float64); ok {
				summary["total_requests"] = totalRequests + 1
			}
			if successCount, ok := summary["success_count"].(float64); ok {
				summary["success_count"] = successCount + 1
			}
		}
		
		// Write back to file
		if file, err := os.Create("audit.json"); err == nil {
			defer file.Close()
			encoder := json.NewEncoder(file)
			encoder.SetIndent("", "  ")
			encoder.Encode(auditData)
		}
	}
}

// GetRegisteredComponents returns the names of all registered auditable components
func (ac *AuditCoordinator) GetRegisteredComponents() []string {
	components := make([]string, 0, len(ac.Components))
	for name := range ac.Components {
		components = append(components, name)
	}
	return components
}