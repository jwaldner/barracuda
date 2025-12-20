package audit

// OptionsAnalysisAuditor defines the comprehensive interface for auditing
// options trading analysis operations in the Barracuda system.
type OptionsAnalysisAuditor interface {
	// LogOptionsAnalysisOperation is the single entry point for ALL audit operations.
	// This method handles the complete audit lifecycle for options trading analysis:
	//
	// INITIALIZE: When called with a new ticker for the first time, it creates
	// the audit.json file and writes a ticker initialization message as the
	// very first entry to clearly identify what stock symbol is being analyzed.
	//
	// APPEND: Subsequent calls add new audit entries (API calls, Black-Scholes calculations,
	// options chain data, Greeks calculations) to the existing audit trail. The system
	// automatically validates data for Infinity/NaN values and options calculation errors.
	//
	// ARCHIVE: When operation is "complete", "archive", or "finish", the audit file is moved
	// to the audits/ directory with a timestamped filename for permanent storage.
	// The "finish" operation is typically used at the end of AI analysis to both
	// append the analysis results AND archive the complete audit trail.
	//
	// Parameters:
	//   stockTicker: Stock symbol being analyzed (e.g., "AAPL", "TSLA"). Empty string for system operations.
	//   analysisOperation: Type of analysis being logged. Standard operations include:
	//     - "GetStockPrice", "GetOptionsChain" - API data retrieval
	//     - "CalculateBlackScholes" - CUDA computation results
	//     - "AIAnalysis" - AI/Grok analysis results
	//     - "finish" - Append final analysis data AND archive audit file
	//     - "complete", "archive" - Archive audit file only
	//   analysisData: The data to be logged - API responses, calculation results, or system messages
	//
	// Returns:
	//   error: Any error encountered during audit logging, nil on success
	//
	// Usage Examples:
	//   // Initialize audit for AAPL options analysis
	//   auditor.LogOptionsAnalysisOperation("AAPL", "InitializeAnalysis", nil)
	//
	//   // Log stock price retrieval
	//   auditor.LogOptionsAnalysisOperation("AAPL", "GetStockPrice", stockPriceData)
	//
	//   // Log Black-Scholes calculation
	//   auditor.LogOptionsAnalysisOperation("AAPL", "CalculateBlackScholes", calculationResults)
	//
	//   // Archive the analysis (moves to audits/audit_AAPL_2025-12-19_15-30-45.json)
	//   auditor.LogOptionsAnalysisOperation("AAPL", "CompleteAnalysis", nil)
	//
	//   // Add AI analysis and archive in one call
	//   auditor.LogOptionsAnalysisOperation("AAPL", "finish", aiAnalysisData)
	LogOptionsAnalysisOperation(stockTicker string, analysisOperation string, analysisData interface{}) error
}
