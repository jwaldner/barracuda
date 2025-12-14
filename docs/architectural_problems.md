
# Barracuda Project: Required Architectural Changes

## Current Codebase Analysis

### Backend Structure Issues

#### Monolithic Handler (704 lines)
`internal/handlers/options.go` contains:
- HTTP request parsing
- Business logic calculations  
- Monte Carlo simulations
- Options pricing algorithms
- Data formatting
- Error handling
- Logging
- Response building

#### Duplicate Data Models
```go
// internal/models/models.go
type OptionResult struct {
    Ticker     string  `json:"ticker"`
    Strike     float64 `json:"strike"`
    Premium    float64 `json:"premium"`
    // ... 18 more fields
}

// internal/dto/dto.go  
type AnalysisRequest struct {
    Symbols        []string `json:"symbols"`
    ExpirationDate string   `json:"expiration_date"`
    // ... duplicate structure
}
```

#### Configuration Chaos
`internal/config/config.go` (231 lines) mixes:
- Server configuration
- API credentials
- Business logic defaults
- Engine configuration
- Legacy compute settings
- Environment variable parsing

#### Service Layer Confusion
- `internal/services/request_service.go` - Only handles request parsing
- No business logic services
- No separation between API concerns and domain logic

## Required Backend Restructuring

### 1. Layered Architecture Implementation

```
api/
├── handlers/           # HTTP layer only
│   ├── options.go     # Route handling, HTTP concerns
│   └── middleware.go  # Auth, logging, CORS
├── dto/               # API contracts
│   ├── requests.go    # Input DTOs
│   └── responses.go   # Output DTOs (formatted for frontend)
└── validators/        # Request validation

business/
├── options/
│   ├── service.go     # Orchestrates business operations
│   ├── calculator.go  # Options pricing logic
│   ├── risk.go        # Risk analysis
│   └── portfolio.go   # Portfolio optimization
├── market/
│   ├── data.go        # Market data operations
│   └── cache.go       # Data caching strategy
└── compute/
    ├── cuda.go        # CUDA operations
    └── monte_carlo.go # Monte Carlo calculations

infrastructure/
├── alpaca/            # External API client
├── config/            # Configuration management
└── storage/           # Data persistence
```

### 2. API Response Standardization

Backend must format ALL display values:

```go
// api/dto/responses.go
type OptionsAnalysisResponse struct {
    Success bool                   `json:"success"`
    Data    OptionsAnalysisData    `json:"data"`
    Meta    ResponseMetadata       `json:"meta"`
}

type OptionsAnalysisData struct {
    Results []FormattedOptionResult `json:"results"`
}

type FormattedOptionResult struct {
    Rank         int    `json:"rank"`
    Symbol       string `json:"symbol"`
    Strike       string `json:"strike"`        // "$100.00"
    StockPrice   string `json:"stock_price"`   // "$105.50" 
    Contracts    int    `json:"contracts"`     // 10
    Premium      string `json:"premium"`       // "$5.25"
    TotalPremium string `json:"total_premium"` // "$52,500"
    CashNeeded   string `json:"cash_needed"`   // "$100,000"
    Yield        string `json:"yield"`         // "5.25%"
    Annualized   string `json:"annualized"`    // "95.5%"
    Expiration   string `json:"expiration"`    // "2025-12-20"
}
```

### 3. Business Logic Extraction

```go
// business/options/service.go
type OptionsService struct {
    calculator Calculator
    market     MarketData
    validator  Validator
}

func (s *OptionsService) AnalyzeOptions(ctx context.Context, req AnalysisRequest) (*AnalysisResult, error) {
    // Pure business logic - no HTTP concerns
}

// business/options/calculator.go
type Calculator struct {
    pricer PricingEngine
    risk   RiskAnalyzer
}

func (c *Calculator) CalculatePremiums(options []Option) []PremiumResult
func (c *Calculator) AssessRisk(delta float64) RiskProfile
func (c *Calculator) OptimizePortfolio(cash float64, options []Option) Portfolio
```

### 4. Configuration Refactoring

```go
// infrastructure/config/config.go
type Config struct {
    Server ServerConfig `yaml:"server"`
    API    APIConfig    `yaml:"api"`  
    Engine EngineConfig `yaml:"engine"`
}

// infrastructure/config/server.go
type ServerConfig struct {
    Port         string        `yaml:"port"`
    ReadTimeout  time.Duration `yaml:"read_timeout"`
    WriteTimeout time.Duration `yaml:"write_timeout"`
}

// infrastructure/config/api.go  
type APIConfig struct {
    Alpaca AlpacaConfig `yaml:"alpaca"`
}
```

## Frontend Requirements

### 1. API Client Layer
```javascript
// api/client.js - Single responsibility
class BarracudaAPI {
    constructor(baseURL) {
        this.baseURL = baseURL;
    }
    
    async analyzeOptions(request) {
        // Pure API communication
        // No business logic
        // No formatting
    }
}
```

### 2. Display Components
```javascript
// ui/results-table.js
class ResultsTable {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
    }
    
    render(formattedResults) {
        // Pure rendering - backend provides formatted values
        // No calculations, no formatting
    }
}
```

### 3. Application Controller
```javascript
// app/controller.js
class OptionsController {
    constructor(api, table, form) {
        this.api = api;
        this.table = table;  
        this.form = form;
    }
    
    async analyze() {
        const request = this.form.getFormData();
        const response = await this.api.analyzeOptions(request);
        this.table.render(response.data.results);
    }
}
```

## Implementation Priority

### Phase 1: Backend Service Layer (Week 1)
1. Extract business logic from handlers
2. Create proper service interfaces
3. Implement calculator and risk analysis services
4. Maintain current API contracts (no frontend changes)

### Phase 2: API Standardization (Week 2)  
1. Create formatted response DTOs
2. Move all formatting logic to backend
3. Implement proper error handling
4. Add response validation

### Phase 3: Frontend Simplification (Week 3)
1. Remove all formatting logic from JavaScript
2. Implement clean API client
3. Create modular UI components
4. Remove duplicate functions and global state

### Phase 4: Configuration & Infrastructure (Week 4)
1. Refactor configuration management
2. Implement proper logging
3. Add health checks and monitoring
4. Create deployment pipeline

## Success Criteria

### Backend Responsibilities
- ✅ All business logic contained in service layer
- ✅ Handlers only handle HTTP concerns  
- ✅ All display values formatted server-side
- ✅ Consistent API contracts with validation
- ✅ Proper error handling and logging

### Frontend Responsibilities  
- ✅ Pure API communication (no business logic)
- ✅ Clean UI rendering (no data formatting)
- ✅ Modular component structure
- ✅ Single source of truth for state

### Architecture Benefits
- ✅ Backend shapes frontend completely
- ✅ Independent testing of all layers
- ✅ Clear separation of concerns
- ✅ Maintainable and extensible codebase
- ✅ Predictable change impact