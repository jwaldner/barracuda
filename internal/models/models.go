package models

// FieldValue represents a field with both raw data and formatted display
type FieldValue struct {
	Raw     interface{} `json:"raw"`     // For CSV/sorting: 1234.56
	Display string      `json:"display"` // For UI: "$1,234.56"
	Type    string      `json:"type"`    // For CSS: "currency"
}

// FormattedOptionResult represents an option result with formatted fields
type FormattedOptionResult map[string]FieldValue

// FormattedAnalysisResponse represents the complete API response
type FormattedAnalysisResponse struct {
	Success bool                  `json:"success"`
	Data    FormattedAnalysisData `json:"data"`
	Meta    ResponseMetadata      `json:"meta"`
}

type FormattedAnalysisData struct {
	Results       []FormattedOptionResult  `json:"results"`
	FieldMetadata map[string]FieldMetadata `json:"field_metadata"`
}

type FieldMetadata struct {
	DisplayName string `json:"display_name"`
	Type        string `json:"type"`
	Sortable    bool   `json:"sortable"`
	Alignment   string `json:"alignment"`
}

type ResponseMetadata struct {
	Strategy           string  `json:"strategy"`
	ExpirationDate     string  `json:"expiration_date"`
	Timestamp          string  `json:"timestamp"`
	ProcessingTime     float64 `json:"processing_time"`
	ComputeDuration    float64 `json:"compute_duration"`
	ProcessingStats    string  `json:"processing_stats"`
	Engine             string  `json:"engine"`
	CudaAvailable      bool    `json:"cuda_available"`
	ExecutionMode      string  `json:"execution_mode"`
	SymbolCount        int     `json:"symbol_count"`
	ResultCount        int     `json:"result_count"`
	WorkloadFactor     float64 `json:"workload_factor"`
	SamplesProcessed   int     `json:"samples_processed"`
	ContractsProcessed int     `json:"contracts_processed"`
}

// AnalysisRequest represents a request for options analysis
type AnalysisRequest struct {
	Symbols        []string `json:"symbols"`
	ExpirationDate string   `json:"expiration_date"`
	TargetDelta    float64  `json:"target_delta"`
	AvailableCash  float64  `json:"available_cash"`
	Strategy       string   `json:"strategy"`     // "puts" or "calls"
	AuditTicker    string   `json:"audit_ticker"` // For audit logging
}

// OptionResult represents the result of analyzing an option
type OptionResult struct {
	Ticker           string  `json:"ticker"`
	Company          string  `json:"company"` // Company name from SP500 data
	Sector           string  `json:"sector"`  // Sector from SP500 data
	OptionSymbol     string  `json:"option_symbol"`
	OptionType       string  `json:"option_type"`
	Strike           float64 `json:"strike"`
	StockPrice       float64 `json:"stock_price"`
	Premium          float64 `json:"premium"`
	MaxContracts     int     `json:"max_contracts"`
	TotalPremium     float64 `json:"total_premium"`
	CashNeeded       float64 `json:"cash_needed"`
	ProfitPercentage float64 `json:"profit_percentage"`
	Delta            float64 `json:"delta"`
	Gamma            float64 `json:"gamma"`
	Theta            float64 `json:"theta"`
	Vega             float64 `json:"vega"`
	ImpliedVol       float64 `json:"implied_volatility"`
	Expiration       string  `json:"expiration"`
	DaysToExp        int     `json:"days_to_expiration"`
}

// AnalysisResponse represents the response from options analysis
type AnalysisResponse struct {
	Results         []OptionResult `json:"results"`
	TotalPremium    float64        `json:"total_premium"`
	RequestedDelta  float64        `json:"requested_delta"`
	Strategy        string         `json:"strategy"`
	ExpirationDate  string         `json:"expiration_date"`
	Timestamp       string         `json:"timestamp"`
	ProcessingTime  float64        `json:"processing_time"`
	ProcessingStats string         `json:"processing_stats"`
}

// SP500Symbol represents a symbol in the S&P 500
type SP500Symbol struct {
	Symbol      string `json:"symbol"`
	Company     string `json:"company"`
	Sector      string `json:"sector"`
	SubIndustry string `json:"sub_industry"`
	Location    string `json:"location"`
	DateAdded   string `json:"date_added"`
	CIK         string `json:"cik"`
	Founded     string `json:"founded"`
}

// CUDACalculationRequest for Black-Scholes calculations
type CUDACalculationRequest struct {
	StockPrice     float64 `json:"stock_price"`
	StrikePrice    float64 `json:"strike_price"`
	TimeToMaturity float64 `json:"time_to_maturity"`
	RiskFreeRate   float64 `json:"risk_free_rate"`
	Volatility     float64 `json:"volatility"`
	OptionType     string  `json:"option_type"` // "call" or "put"
}

// CUDACalculationResponse for Black-Scholes results
type CUDACalculationResponse struct {
	OptionPrice float64 `json:"option_price"`
	Delta       float64 `json:"delta"`
	Gamma       float64 `json:"gamma"`
	Theta       float64 `json:"theta"`
	Vega        float64 `json:"vega"`
	Rho         float64 `json:"rho"`
}

// BatchCalculationRequest for multiple calculations
type BatchCalculationRequest struct {
	Calculations []CUDACalculationRequest `json:"calculations"`
}

// BatchCalculationResponse for multiple results
type BatchCalculationResponse struct {
	Results           []CUDACalculationResponse `json:"results"`
	ProcessedIn       float64                   `json:"processed_in_ms"`
	UsedCUDA          bool                      `json:"used_cuda"`
	DeviceCount       int                       `json:"device_count"`
	TotalCalculations int                       `json:"total_calculations"`
}
