package audit

// Global audit registry to store audit components
var auditRegistry = make(map[string]*AlpacaAudit)

// RegisterAudit registers an audit component with a key
func RegisterAudit(key string, audit *AlpacaAudit) {
	auditRegistry[key] = audit
}

// GetAudit retrieves an audit component by key
func GetAudit(key string) *AlpacaAudit {
	return auditRegistry[key]
}

// SaveAuditToFile saves audit data for a specific ticker and clears it
func SaveAuditToFile(ticker string) error {
	audit := GetAudit("alpaca")
	if audit == nil {
		return nil // No audit data
	}
	
	filename := "audit_" + ticker + ".json"
	err := audit.SaveToFile(filename)
	if err == nil {
		audit.Clear() // Clear after successful save
	}
	return err
}