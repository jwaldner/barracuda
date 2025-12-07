# Barracuda - Complete Project Context

## Project Overview
Baracuda is a high-performance options trading and analysis system that combines:
- **Go backend** for web API and data handling
- **CUDA acceleration** for Black-Scholes calculations
- **Alpaca integration** for real-time market data
- **S&P 500 symbol management** with quarterly updates
- **Assembly calculator integration** for precision

## Architecture Components

### Core Go Application
- **Module**: `github.com/jwaldner/barracuda`
- **Main Server**: Handles HTTP API, CUDA integration, Alpaca data
- **Port**: Configurable (default 8080)
- **Environment**: Paper trading vs Live trading support

### CUDA Engine (Baracuda)
- **Language**: C++ with CUDA kernels
- **Purpose**: High-performance Black-Scholes calculations
- **Features**: 
  - Batch options processing
  - Monte Carlo simulations
  - 25-delta volatility skew analysis
  - Greek calculations (Delta, Gamma, Theta, Vega, Rho)

### Market Data Integration
- **Provider**: Alpaca Markets API
- **Real-time**: Stock prices, options chains, quotes
- **Historical**: Stock price history for volatility analysis
- **Filtering**: Strategy-based (puts/calls), strike price ranges

### S&P 500 Symbol Service
- **Source**: GitHub CSV (quarterly S&P Dow Jones updates)
- **Backup**: Local asset files for failover
- **Data**: Symbol, Company, Sector, Sub-industry, Headquarters, etc.
- **Update**: Quarterly refresh with asset backup

## Key Features

### Options Analysis
1. **Delta-Based Selection**: Maps target delta to assignment probability
2. **Strike Filtering**: Puts below stock price, calls above
3. **Real Premium Pricing**: Uses bid/ask spreads for accuracy
4. **Cash Allocation**: Optimizes contract quantities based on available cash
5. **Ranking System**: Sorts by total premium potential

### CUDA Acceleration
1. **Black-Scholes**: GPU-accelerated options pricing
2. **Batch Processing**: 1000+ contracts simultaneously
3. **Monte Carlo**: Portfolio simulations
4. **Volatility Skew**: 25-delta put/call analysis
5. **Performance**: Microsecond-level calculations

### Assembly Calculator Integration
- **fcalc**: High-precision floating-point calculations
- **calc**: Integer-based operations
- **Use Cases**: Financial precision, volatility analysis, drawdown calculations

## API Endpoints

### CUDA Endpoints
- `POST /cuda/calculate` - Black-Scholes calculations
- `POST /cuda/volatility-skew` - 25-delta skew analysis
- `POST /analyze-enhanced` - Alpaca + CUDA combined analysis
- `GET /cuda/benchmark` - Performance testing
- `GET /cuda/status` - Engine status

### S&P 500 Endpoints
- `POST /sp500/update` - Force symbol update
- `GET /sp500/symbols` - Get all symbols with metadata
- `GET /sp500/list` - Get symbol list only
- `GET /sp500/info` - Service information

### Traditional Endpoints
- `POST /analyze` - Basic options analysis
- `GET /stock-history` - Historical price data
- `POST /api/analyze-volatility` - Volatility analysis API
- `POST /calculate` - Assembly calculator operations

## Data Structures

### Option Contract (CUDA)
```go
type OptionContract struct {
    Symbol           string  `json:"symbol"`
    StrikePrice      float64 `json:"strike_price"`
    UnderlyingPrice  float64 `json:"underlying_price"`
    TimeToExpiration float64 `json:"time_to_expiration"`
    RiskFreeRate     float64 `json:"risk_free_rate"`
    Volatility       float64 `json:"volatility"`
    OptionType       byte    `json:"option_type"`
    
    // Greeks (output)
    Delta            float64 `json:"delta"`
    Gamma            float64 `json:"gamma"`
    Theta            float64 `json:"theta"`
    Vega             float64 `json:"vega"`
    Rho              float64 `json:"rho"`
    TheoreticalPrice float64 `json:"theoretical_price"`
}
```

### S&P 500 Symbol
```go
type Symbol struct {
    Symbol              string `json:"symbol"`
    Company             string `json:"company"`
    Sector              string `json:"sector"`
    SubIndustry         string `json:"sub_industry"`
    HeadquartersLocation string `json:"headquarters_location"`
    DateAdded           string `json:"date_added"`
    CIK                 string `json:"cik"`
    Founded             string `json:"founded"`
    LastUpdated         string `json:"last_updated"`
}
```

### Volatility Skew
```go
type VolatilitySkew struct {
    Symbol     string  `json:"symbol"`
    Expiration string  `json:"expiration"`
    Put25DIV   float64 `json:"put_25d_iv"`
    Call25DIV  float64 `json:"call_25d_iv"`
    Skew       float64 `json:"skew"`
    ATMIV      float64 `json:"atm_iv"`
}
```

## Configuration

### Environment Variables
- `ALPACA_API_KEY` - Alpaca API credentials
- `ALPACA_SECRET_KEY` - Alpaca secret key  
- `ALPACA_PAPER_TRADING` - Paper vs live trading
- `PORT` - Server port (default 8080)
- `DEFAULT_STOCKS` - Default symbol list
- `DEFAULT_CASH` - Default available cash
- `DEFAULT_STRATEGY` - Default strategy (puts/calls)

### File Structure
```
go-cuda/
├── main.go                    # Main server with CUDA integration
├── baracuda/
│   └── engine.go             # Go CUDA wrapper
├── internal/
│   ├── alpaca/               # Market data client
│   ├── config/               # Configuration management
│   ├── handlers/             # HTTP handlers
│   └── symbols/              # S&P 500 service
├── src/                      # C++ CUDA source
├── assets/symbols/           # S&P 500 backup data
├── cmd/test-sp500/          # S&P 500 test utility
└── docs/                    # Documentation
```

## Performance Characteristics

### CUDA Engine
- **Batch Size**: 1000+ contracts per call
- **Processing Time**: ~175ms for 503 symbols
- **Memory**: GPU memory management for large datasets
- **Scalability**: Linear scaling with GPU cores

### S&P 500 Service  
- **Update Time**: ~175ms for full refresh
- **Asset Size**: 64KB backup file
- **Symbols**: 503 current S&P 500 companies
- **Reliability**: Fallback to local assets on network failure

### Market Data
- **Latency**: Real-time via Alpaca WebSocket/REST
- **Rate Limits**: Alpaca API limits apply
- **Caching**: Local caching for repeated requests
- **Precision**: Assembly calculator for financial accuracy

## Trading Strategies

### Delta Mapping
- **0.75 Delta**: High assignment risk (2% from stock price)
- **0.50 Delta**: Medium assignment risk (5% from stock price)  
- **0.25 Delta**: Low assignment risk (12% from stock price)

### Strike Selection
- **Puts**: Only strikes below current stock price
- **Calls**: Only strikes above current stock price
- **Filtering**: Tradable contracts only
- **Sorting**: By distance from target delta

### Risk Management
- **Cash Allocation**: Per-contract cash requirements
- **Premium Calculation**: Real bid/ask spreads
- **Assignment Probability**: Distance-based risk assessment
- **Portfolio Diversification**: Multi-symbol analysis

## Integration Points

### External APIs
1. **Alpaca Markets**: Real-time market data and trading
2. **GitHub CSV**: S&P 500 quarterly updates
3. **Assembly Calculators**: Precision mathematical operations

### Internal Services
1. **CUDA Engine**: High-performance calculations  
2. **S&P 500 Service**: Symbol universe management
3. **Configuration Service**: Environment management
4. **HTTP Router**: API endpoint management

## Development & Testing

### Build System
- **Go Modules**: Dependency management
- **CGO**: C++ CUDA integration
- **CUDA Toolkit**: Required for GPU acceleration
- **Assembly Tools**: fcalc, calc binaries

### Testing
- **Unit Tests**: Go test suite
- **C++ Tests**: Google Test framework
- **Integration Tests**: Full API testing
- **Performance Tests**: Benchmark utilities

This comprehensive context covers the entire Baracuda system architecture, from CUDA acceleration to market data integration to precision calculations.