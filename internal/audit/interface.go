package audit

import (
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
	return &AuditCoordinator{
		Components: make(map[string]Auditable),
	}
}

// Register adds an auditable component with a given name
func (ac *AuditCoordinator) Register(name string, auditable Auditable) {
	ac.Components[name] = auditable
}

// AuditTicker performs audit for a specific ticker across all registered components
func (ac *AuditCoordinator) AuditTicker(ticker string, request models.AnalysisRequest) (map[string]interface{}, error) {
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

// GetRegisteredComponents returns the names of all registered auditable components
func (ac *AuditCoordinator) GetRegisteredComponents() []string {
	components := make([]string, 0, len(ac.Components))
	for name := range ac.Components {
		components = append(components, name)
	}
	return components
}