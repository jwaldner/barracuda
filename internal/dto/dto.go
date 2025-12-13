package dto

// TemplateData represents data passed to HTML templates
type TemplateData struct {
	Title                 string
	DefaultStocks         []string
	DefaultCash           int
	DefaultStrategy       string
	DefaultRiskLevel      string
	DefaultExpirationDate string
	SP500Count            int
	CUDAAvailable         bool
	DeviceCount           int
	PaperTrading          bool
	WorkloadFactor        float64
	SymbolSource          string
}

// AnalysisRequest represents an options analysis request
type AnalysisRequest struct {
	Symbols        []string `json:"symbols"`
	ExpirationDate string   `json:"expiration_date"`
	TargetDelta    float64  `json:"target_delta"`
	AvailableCash  float64  `json:"available_cash"`
	Strategy       string   `json:"strategy"`
}
