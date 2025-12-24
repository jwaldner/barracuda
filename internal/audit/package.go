package audit

// OptionsAnalysisAuditor defines the interface for auditing options analysis operations
type OptionsAnalysisAuditor interface {
	LogOptionsAnalysisOperation(ticker string, operation string, data interface{}) error
}
