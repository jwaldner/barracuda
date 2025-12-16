# Project Alpaca Options Data Integration

## Overview

Project is an options premium analyzer that integrates with Alpaca Markets API to retrieve real-time stock quotes and options data. The application analyzes options contracts to find the best premium opportunities based on user-defined criteria including delta targets, expiration dates, and trading strategies.

## Data Flow Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   Web UI Form   │───▶│  Handler Layer   │───▶│  Alpaca Client  │───▶│  Alpaca Markets  │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
        │                       │                       │                       │
        ▼                       ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ User Parameters │    │ Request Parsing  │    │ API Requests    │    │ JSON Responses   │
│ - Symbols       │    │ - Validation     │    │ - Authentication│    │ - Stock Prices   │
│ - Delta Target  │    │ - Processing     │    │ - Filtering     │    │ - Options Chain  │
│ - Cash Amount   │    │ - Analysis       │    │ - Rate Limiting │    │ - Option Quotes  │
│ - Strategy      │    │ - Response       │    │                 │    │                  │
│ - Expiration    │    │                  │    │                 │    │                  │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
        │                       │                       │                       │
        ▼                       ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Results Display │◀───│  Analysis Engine │◀───│ Data Processing │◀───│ Contract Matching│
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
```

## Complete Risk Selector Flow Trace

### Step-by-Step Risk Level Processing

```
USER CLICKS RISK BUTTON
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. UI Event Handler (JavaScript)                                             │
│    • User clicks: <button data-delta="0.50">MOD Δ 0.50</button>             │
│    • Event listener captures click                                           │
│    • Extracts data-delta="0.50" attribute                                   │
│    • Sets selectedDelta = 0.50 (global variable)                           │
│    • Updates button visual styling (highlights selected button)              │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. Form Submission (JavaScript)                                             │
│    • User clicks "🔍 Analyze Options" button                                │
│    • handleSubmit() function triggered                                      │
│    • Validates selectedDelta (fallback to 0.50 if invalid)                  │
│    • Creates URLSearchParams with target_delta: "0.50"                      │
│    • Sends POST request to /analyze endpoint                                │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. Server Handler Processing (Go)                                           │
│    • AnalyzeHandler receives POST /analyze                                  │
│    • Extracts: targetDeltaStr := r.FormValue("target_delta")               │
│    • Converts: targetDelta, _ := strconv.ParseFloat(targetDeltaStr, 64)    │
│    • Passes targetDelta=0.50 to alpaca client                              │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. Delta-to-Distance Conversion (Go)                                        │
│    • findBestContract() receives targetDelta=0.50                          │
│    • Conversion logic:                                                       │
│      if targetDelta == 0.75 → targetDistance = 0.02 (2%)                   │
│      if targetDelta == 0.50 → targetDistance = 0.05 (5%) ✓ SELECTED        │
│      if targetDelta == 0.25 → targetDistance = 0.12 (12%)                  │
│    • Result: targetDistance = 0.05 (5% from stock price)                   │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. Strike Selection Algorithm (Go)                                          │
│    • For each option contract in filtered chain:                            │
│      - Calculate actual distance from stock price                           │
│      - For PUTS: distance = (stockPrice - strikePrice) / stockPrice        │
│      - For CALLS: distance = (strikePrice - stockPrice) / stockPrice       │
│      - Find strike with distance closest to 0.05 (5%)                      │
│    • Example with AAPL $178.25:                                            │
│      - Target: 5% below = $169.34                                          │
│      - Available strikes: $165, $170, $175                                 │
│      - $170 strike = 4.6% below (closest to 5% target) ✓ SELECTED          │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 6. Financial Calculations (Go)                                              │
│    • Selected contract: AAPL $170 PUT                                      │
│    • Get real-time quote: bid=$2.80, ask=$2.90 → premium=$2.85             │
│    • Cash per contract: $170 × 100 = $17,000                               │
│    • Max contracts: $70,000 ÷ $17,000 = 4 contracts                       │
│    • Total premium: $2.85 × 4 × 100 = $1,140                              │
│    • Premium yield: ($1,140 ÷ $68,000) × 100 = 1.68%                      │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 7. Results Display (JavaScript)                                             │
│    • Server returns JSON with calculated results                            │
│    • displayResults() renders table row:                                   │
│      - Delta column shows: 0.046 (actual 4.6% distance)                   │
│      - Strike column shows: $170.00                                        │
│      - Premium column shows: $2.85                                         │
│      - Total Premium column shows: $1,140 (highlighted green)              │
│    • User sees risk level translated to specific strike selection           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Risk Level Mapping Summary

| UI Button | Delta Value | Target Distance | Strike Selection | Assignment Risk |
|-----------|-------------|-----------------|------------------|-----------------|
| LOW Δ 0.25 | 0.25 | 12% from stock | Far OTM | ~25% chance |
| MOD Δ 0.50 | 0.50 | 5% from stock | Moderate OTM | ~50% chance |
| HIGH Δ 0.75 | 0.75 | 2% from stock | Close to ATM | ~75% chance |

## Detailed Code Trace: Risk Selector Implementation

### 1. HTML Risk Selector Structure
```html
<!-- Risk Level Buttons in index.html -->
<div class="risk-selector">
    <div class="grid grid-cols-3 gap-3">
        <button type="button" class="risk-btn" data-delta="0.25">
            <div class="text-base">LOW</div>
            <div class="text-xs opacity-80">Δ 0.25</div>
        </button>
        <button type="button" class="risk-btn" data-delta="0.50">
            <div class="text-base">MOD</div>
            <div class="text-xs opacity-80">Δ 0.50</div>
        </button>
        <button type="button" class="risk-btn" data-delta="0.75">
            <div class="text-base">HIGH</div>
            <div class="text-xs opacity-80">Δ 0.75</div>
        </button>
    </div>
</div>
```

### 2. JavaScript Event Listener Setup
```javascript
// In setupEventListeners() function
document.querySelectorAll('.risk-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // Extract delta value from button's data attribute
        const delta = parseFloat(btn.dataset.delta);
        
        // Update global selectedDelta variable
        selectRisk(delta);
        
        // Auto-trigger analysis if symbols are entered
        if (document.getElementById('symbols').value.trim()) {
            setTimeout(() => {
                document.getElementById('analyze-btn').click();
            }, 300);
        }
    });
});
```

### 3. Risk Selection Function
```javascript
function selectRisk(delta) {
    selectedDelta = delta;  // Update global variable
    
    // Reset ALL buttons to default state first
    document.querySelectorAll('.risk-btn').forEach(btn => {
        const btnDelta = parseFloat(btn.dataset.delta);
        
        // Reset to base classes
        btn.className = 'risk-btn px-4 py-3 rounded-xl text-sm font-bold text-white transition-all transform duration-300 hover:scale-105';
        
        // Add appropriate color classes based on delta
        if (btnDelta === 0.25) {
            btn.classList.add('bg-green-600/40', 'border-2', 'border-green-400');
        } else if (btnDelta === 0.50) {
            btn.classList.add('bg-orange-600/40', 'border-2', 'border-orange-400');
        } else if (btnDelta === 0.75) {
            btn.classList.add('bg-red-600/40', 'border-2', 'border-red-400');
        }
        
        // Highlight selected button
        if (Math.abs(btnDelta - delta) < 0.001) {
            btn.className = 'risk-btn px-4 py-3 rounded-xl text-sm font-bold transition-all transform duration-300 bg-white text-black border-4 border-white shadow-2xl scale-110';
        }
    });
}
```

### 4. Form Submission with Delta Value
```javascript
async function handleSubmit(e) {
    e.preventDefault();
    
    // Validate selectedDelta before proceeding
    if (isNaN(selectedDelta) || selectedDelta <= 0) {
        selectedDelta = 0.50;  // Default to moderate risk
    }
    
    // Create form data with delta value
    const params = new URLSearchParams();
    params.append('symbols', symbolsValue);
    params.append('expiration_date', expirationValue);
    params.append('target_delta', selectedDelta.toString());  // ← DELTA PASSED HERE
    params.append('available_cash', cashValue);
    params.append('strategy', currentStrategy);
    
    // Send to server
    const response = await fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: params.toString()
    });
}
```

### 5. Server Handler Receives Delta
```go
// In handlers/handlers.go - AnalyzeHandler function
func (h *Handler) AnalyzeHandler(w http.ResponseWriter, r *http.Request) {
    // Extract form values
    symbolsStr := r.FormValue("symbols")
    expirationDate := r.FormValue("expiration_date")
    targetDeltaStr := r.FormValue("target_delta")  // ← DELTA RECEIVED HERE
    availableCashStr := r.FormValue("available_cash")
    strategy := r.FormValue("strategy")
    
    // Convert delta string to float
    targetDelta, err := strconv.ParseFloat(targetDeltaStr, 64)
    if err != nil {
        targetDelta = 0.50  // Default fallback
    }
    
    // Parse other values...
    availableCash, _ := strconv.ParseFloat(availableCashStr, 64)
    
    // Call Alpaca client with delta value
    response, err := h.alpacaClient.AnalyzeOptions(
        cleanSymbols, 
        expirationDate, 
        targetDelta,     // ← DELTA PASSED TO ALPACA CLIENT
        availableCash, 
        strategy
    )
}
```

### 6. Alpaca Client Processes Delta
```go
// In internal/alpaca/alpaca.go - AnalyzeOptions function
func (c *Client) AnalyzeOptions(symbols []string, expiration string, targetDelta float64, availableCash float64, strategy string) (*models.AnalysisResponse, error) {
    
    // Get stock prices first
    stockPrices := make(map[string]float64)
    for _, symbol := range symbols {
        stockPrice, err := c.GetStockPrice(symbol)
        if err == nil {
            stockPrices[symbol] = stockPrice.Price
        }
    }
    
    // Get options chains for each symbol
    contractsBySymbol, err := c.GetOptionsChain(symbols, expiration, strategy)
    
    // Process each symbol's contracts
    var allResults []*models.OptionResult
    for symbol, contracts := range contractsBySymbol {
        if len(contracts) == 0 {
            continue
        }
        
        stockPrice := &StockPrice{Symbol: symbol, Price: stockPrices[symbol]}
        
        // Find best contract for this symbol using target delta
        result := c.findBestContract(
            contracts, 
            stockPrice, 
            targetDelta,    // ← DELTA USED IN CONTRACT SELECTION
            availableCash, 
            strategy
        )
        
        if result != nil {
            allResults = append(allResults, result)
        }
    }
}
```

### 7. Delta-to-Distance Conversion in Contract Selection
```go
// In findBestContract function
func (c *Client) findBestContract(contracts []*OptionContract, stockPrice *StockPrice, targetDelta float64, availableCash float64, strategy string) *models.OptionResult {
    
    // Convert delta to target distance percentage
    var targetDistance float64
    if targetDelta == 0.75 {
        targetDistance = 0.02    // 2% from stock price (HIGH RISK)
    } else if targetDelta == 0.50 {
        targetDistance = 0.05    // 5% from stock price (MODERATE RISK)
    } else {  // targetDelta == 0.25
        targetDistance = 0.12    // 12% from stock price (LOW RISK)
    }
    
    var bestContract *OptionContract
    bestDistanceDiff := 1.0
    
    // Find strike closest to target distance
    for _, contract := range contracts {
        strikePrice, _ := strconv.ParseFloat(contract.StrikePrice, 64)
        
        var strikeDistance float64
        if strategy == "puts" {
            if strikePrice >= stockPrice.Price {
                continue  // Skip ITM puts
            }
            strikeDistance = (stockPrice.Price - strikePrice) / stockPrice.Price
        } else { // calls
            if strikePrice <= stockPrice.Price {
                continue  // Skip ITM calls
            }
            strikeDistance = (strikePrice - stockPrice.Price) / stockPrice.Price
        }
        
        // Calculate how close this strike is to target
        distanceDiff := math.Abs(strikeDistance - targetDistance)
        
        if distanceDiff < bestDistanceDiff {
            bestDistanceDiff = distanceDiff
            bestContract = contract
        }
    }
    
    // Get real-time quote for selected contract
    if bestContract != nil {
        quote, err := c.GetOptionQuote(bestContract.Symbol)
        if err == nil {
            // Calculate financial metrics and return result
            return c.calculateContractMetrics(bestContract, quote, stockPrice, availableCash, strategy)
        }
    }
    
    return nil
}
```

### 8. Results Display Shows Selected Strike
```javascript
// In displayResults function - showing how delta affects final display
function displayResults(data) {
    data.results.forEach((result, index) => {
        const row = document.createElement('tr');
        
        row.innerHTML = 
            `<td class="px-4 py-3">${result.delta.toFixed(3)}</td>` +     // Shows actual distance (e.g., 0.046 for 4.6%)
            `<td class="px-4 py-3">$${result.strike.toFixed(2)}</td>` +   // Shows selected strike (e.g., $170.00)
            `<td class="px-4 py-3">$${result.premium.toFixed(2)}</td>` +  // Shows premium for that strike
            // ... other columns
    });
}
```

This complete trace shows how a single button click (data-delta="0.50") flows through the entire system to produce a specific strike selection and premium calculation.

## API Authentication

### Configuration
Project uses Alpaca API credentials configured via environment variables:

```bash
ALPACA_API_KEY=AKAH5A3VGCR3S9FNWIVC
ALPACA_SECRET_KEY=dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy

```

### Headers
All API requests include authentication headers:
```http
APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC
APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy
```

### Base URLs
- **Live Trading**: `https://api.alpaca.markets` (trading endpoints)
- **Market Data**: `https://data.alpaca.markets` (quotes and bars)

## User Interface Elements

### Strategy Selection
```html
<div class="flex bg-white/20 rounded-xl p-1">
    <button id="puts-tab" class="strategy-tab">Cash-Secured Puts</button>
    <button id="calls-tab" class="strategy-tab">Covered Calls</button>
</div>
```

**Parameters Sent:**
- `strategy`: "puts" or "calls"

### Risk Level (Delta Target)
```html
<div class="risk-selector">
    <button class="risk-btn" data-delta="0.25">LOW Δ 0.25</button>
    <button class="risk-btn" data-delta="0.50">MOD Δ 0.50</button>
    <button class="risk-btn" data-delta="0.75">HIGH Δ 0.75</button>
</div>
```

**Parameters Sent:**
- `target_delta`: 0.25, 0.50, or 0.75

### Input Fields
```html
<!-- Available Cash -->
<input type="number" name="available_cash" value="50000" step="1000">

<!-- Expiration Date -->
<input type="date" name="expiration_date">

<!-- Stock Symbols -->
<textarea name="symbols" rows="4">AAPL
MSFT
TSLA</textarea>
```

**Parameters Sent:**
- `available_cash`: Dollar amount (e.g., 50000)
- `expiration_date`: ISO date format (e.g., "2025-07-18")
- `symbols`: Newline or comma-separated symbols

## Stock Price Retrieval

### API Endpoint
```
GET https://data.alpaca.markets/v2/stocks/{symbol}/bars/latest
```

### Request Example
```bash
curl -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://data.alpaca.markets/v2/stocks/AAPL/bars/latest"
```

### Response Structure
```json
{
  "bar": {
    "c": 178.25,    // Close price (used as current price)
    "h": 179.50,    // High
    "l": 177.80,    // Low
    "n": 12450,     // Number of trades
    "o": 178.90,    // Open
    "t": "2025-12-11T21:00:00Z",  // Timestamp
    "v": 1234567,   // Volume
    "vw": 178.15    // Volume weighted average price
  },
  "symbol": "AAPL"
}
```

### Processing Logic
```go
func (c *Client) GetStockPrice(symbol string) (*StockPrice, error) {
    endpoint := fmt.Sprintf("/v2/stocks/%s/bars/latest", symbol)
    // ... authentication and HTTP request ...
    
    return &StockPrice{
        Symbol: symbol,
        Price:  alpacaResp.Bar.Close,  // Use close price as current
    }, nil
}
```

## Options Chain Retrieval

### API Endpoint
```
GET https://api.alpaca.markets/v2/options/contracts
```

### Request Parameters

**Base Parameters:**
- `underlying_symbols`: Stock symbol (e.g., "AAPL")
- `expiration_date`: Target expiration (e.g., "2025-07-18")
- `limit`: Maximum contracts (e.g., 1000)

**Default Filtering Applied:**
- `status`: "active" (only active contracts)
- `tradable`: "true" (only tradable contracts)
- `style`: "american" (American-style options only)
- `size`: "1" (standard contract size)
- `multiplier`: "100" (100 shares per contract)

**Strategy-Based Filtering:**

#### For Puts Strategy
```
type=put
strike_price_lte={stock_price - 1}
```
- Only retrieves PUT options
- Only strikes BELOW current stock price
- Example: If AAPL = $178, only gets strikes ≤ $177

#### For Calls Strategy  
```
type=call
strike_price_gte={stock_price + 1}
```
- Only retrieves CALL options
- Only strikes ABOVE current stock price
- Example: If AAPL = $178, only gets strikes ≥ $179

### Request Example
```bash
# For AAPL puts with stock price $178.25
curl -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://api.alpaca.markets/v2/options/contracts?underlying_symbols=AAPL&expiration_date=2025-07-18&type=put&strike_price_lte=177&limit=1000"
```

### Response Structure
```json
{
  "option_contracts": [
    {
      "id": "01234567-89ab-cdef-0123-456789abcdef",
      "symbol": "AAPL240719P00175000",
      "name": "AAPL Jul 19 2024 $175 Put",
      "status": "active",
      "tradable": true,
      "expiration_date": "2024-07-19",
      "root_symbol": "AAPL",
      "underlying_symbol": "AAPL",
      "underlying_asset_id": "b0b6dd9d-8b9b-48a9-ba46-b9d54906e415",
      "type": "put",
      "style": "american",
      "strike_price": "175.00",
      "multiplier": "100",
      "size": "1"
    }
  ],
  "next_page_token": null
}
```

### Default Query Parameters Applied

When making requests to the Alpaca options contracts endpoint, Project applies these default filters to reduce the dataset:

```http
GET /v2/options/contracts?
  underlying_symbols=AAPL&
  expiration_date=2025-07-18&
  status=active&
  tradable=true&
  style=american&
  size=1&
  multiplier=100&
  limit=1000&
  type=put&
  strike_price_lte=177
```

**Automatic Filters:**
- `status=active` - Excludes expired or delisted contracts
- `tradable=true` - Only contracts available for trading
- `style=american` - American-style options (can exercise anytime)
- `size=1` - Standard contract size
- `multiplier=100` - 100 shares per contract
- `limit=1000` - Maximum contracts per request

**Strategy Filters:**
- For puts: `type=put` + `strike_price_lte={stock_price-1}`
- For calls: `type=call` + `strike_price_gte={stock_price+1}`

### Options Exclusion Examples

Given AAPL stock price of $178.25 on 2025-12-12:

**PUT Strategy Exclusions:**
- ❌ All CALL options (different type)
- ❌ PUT strikes ≥ $178 (at or above stock price)
- ❌ Expired contracts (status != "active")
- ❌ Non-tradable contracts (tradable != "true")
- ❌ European-style options (style != "american")
- ❌ Weekly options with different multipliers
- ✅ Only PUT strikes ≤ $177 are retrieved

**CALL Strategy Exclusions:**
- ❌ All PUT options (different type)
- ❌ CALL strikes ≤ $178 (at or below stock price)
- ❌ Expired contracts (status != "active")
- ❌ Non-tradable contracts (tradable != "true")
- ❌ European-style options (style != "american")
- ❌ Weekly options with different multipliers
- ✅ Only CALL strikes ≥ $179 are retrieved

**Typical Result Set Size:**
- Before filtering: ~500-2000 contracts per expiration
- After filtering: ~20-50 contracts per symbol
- Final selection: 1 best contract per symbol based on delta target

### Contract Processing
```go
func (c *Client) GetOptionsChain(symbols []string, expiration string, strategy string) (map[string][]*OptionContract, error) {
    contractsBySymbol := make(map[string][]*OptionContract)
    
    // Get stock prices first for filtering
    stockPrices := make(map[string]float64)
    for _, symbol := range symbols {
        stockPrice, err := c.GetStockPrice(symbol)
        stockPrices[symbol] = stockPrice.Price
    }
    
    // Make separate API calls for each symbol
    for _, symbol := range symbols {
        stockPrice := stockPrices[symbol]
        
        // Build filtered request
        q := req.URL.Query()
        q.Add("underlying_symbols", symbol)
        if expiration != "" {
            q.Add("expiration_date", expiration)
        }
        
        // Apply strategy-based filtering
        if strategy == "puts" {
            q.Add("type", "put")
            q.Add("strike_price_lte", fmt.Sprintf("%.0f", stockPrice-1))
        } else {
            q.Add("type", "call")
            q.Add("strike_price_gte", fmt.Sprintf("%.0f", stockPrice+1))
        }
        
        // ... execute request and process response ...
    }
}
```

## Option Quote Retrieval

### API Endpoint
```
GET https://data.alpaca.markets/v1beta1/options/quotes/latest
```

### Request Parameters
- `symbols`: Option contract symbol (e.g., "AAPL240719P00175000")

### Request Example
```bash
curl -H "APCA-API-KEY-ID: AKAH5A3VGCR3S9FNWIVC" \
     -H "APCA-API-SECRET-KEY: dI2ack52IQtNdtiJuGBfvdKHvWxZewtwBcjdb5oy" \
     "https://data.alpaca.markets/v1beta1/options/quotes/latest?symbols=AAPL240719P00175000"
```

### Response Structure
```json
{
  "quotes": {
    "AAPL240719P00175000": {
      "ap": 2.85,         // Ask Price
      "as": 25,           // Ask Size
      "ax": "CBOE",       // Ask Exchange
      "bp": 2.80,         // Bid Price
      "bs": 15,           // Bid Size
      "bx": "NASDAQ",     // Bid Exchange
      "c": "R",           // Condition
      "t": "2025-12-11T20:59:45.123456789Z"  // Timestamp
    }
  }
}
```

### Premium Calculation
```go
func (c *Client) GetOptionQuote(optionSymbol string) (*OptionQuote, error) {
    // ... make API request ...
    
    // Use mid price (average of bid and ask)
    premium := (quote.BidPrice + quote.AskPrice) / 2
    if premium <= 0 {
        // Fallback to ask price if mid is zero
        premium = quote.AskPrice
    }
    
    return &quote, nil
}
```

## Assignment Risk Level System

### UI Risk Level Selection

Project presents three assignment risk levels in the user interface, each mapped to a specific delta value:

```html
<div class="risk-selector">
    <button class="risk-btn" data-delta="0.25">
        <div class="text-base">LOW</div>
        <div class="text-xs opacity-80">Δ 0.25</div>
    </button>
    <button class="risk-btn" data-delta="0.50">
        <div class="text-base">MOD</div>
        <div class="text-xs opacity-80">Δ 0.50</div>
    </button>
    <button class="risk-btn" data-delta="0.75">
        <div class="text-base">HIGH</div>
        <div class="text-xs opacity-80">Δ 0.75</div>
    </button>
</div>
```

### Risk Level Definitions

**LOW RISK (Δ 0.25):**
- **Assignment Probability:** Very Low (~25% chance)
- **Premium Income:** Lower but safer
- **Ideal For:** Conservative investors, steady income
- **Strike Selection:** Far out-of-the-money (12% from stock price)

**MODERATE RISK (Δ 0.50):** *(Default)*
- **Assignment Probability:** Moderate (~50% chance)  
- **Premium Income:** Balanced risk/reward
- **Ideal For:** Most investors, balanced approach
- **Strike Selection:** Moderately out-of-the-money (5% from stock price)

**HIGH RISK (Δ 0.75):**
- **Assignment Probability:** High (~75% chance)
- **Premium Income:** Higher but riskier
- **Ideal For:** Aggressive traders, willing to own/sell stocks
- **Strike Selection:** Close to current price (2% from stock price)

### API Parameters Passed

When user selects a risk level, the following parameter is sent to `/analyze`:

```javascript
// Form data construction
params.append('target_delta', selectedDelta.toString());

// Examples of what gets sent:
// LOW risk    → target_delta: "0.25"
// MODERATE    → target_delta: "0.50" 
// HIGH risk   → target_delta: "0.75"
```

### Server-Side Delta Processing

The server receives the delta value and converts it to strike distance targets:

```go
func convertDeltaToStrikeDistance(targetDelta float64, strategy string) float64 {
    var targetDistance float64
    
    if targetDelta == 0.75 {     // HIGH RISK
        targetDistance = 0.02    // 2% from stock price
    } else if targetDelta == 0.50 {  // MODERATE RISK
        targetDistance = 0.05    // 5% from stock price
    } else {  // targetDelta == 0.25   // LOW RISK
        targetDistance = 0.12    // 12% from stock price
    }
    
    return targetDistance
}
```

### Practical Strike Selection Examples

Given AAPL stock price of **$178.25** on December 12, 2025:

**PUT Options (Cash-Secured Puts):**
- **LOW Risk (Δ 0.25):** Targets strikes ~12% below = **$157** puts
  - Premium: Lower (~$1.50), Assignment chance: ~25%
- **MOD Risk (Δ 0.50):** Targets strikes ~5% below = **$169** puts  
  - Premium: Moderate (~$3.20), Assignment chance: ~50%
- **HIGH Risk (Δ 0.75):** Targets strikes ~2% below = **$175** puts
  - Premium: Higher (~$5.80), Assignment chance: ~75%

**CALL Options (Covered Calls):**
- **LOW Risk (Δ 0.25):** Targets strikes ~12% above = **$200** calls
  - Premium: Lower (~$1.20), Called away chance: ~25%
- **MOD Risk (Δ 0.50):** Targets strikes ~5% above = **$187** calls
  - Premium: Moderate (~$2.90), Called away chance: ~50%
- **HIGH Risk (Δ 0.75):** Targets strikes ~2% above = **$182** calls
  - Premium: Higher (~$4.70), Called away chance: ~75%

### How Project Selects Strikes

1. **Calculate Target Distance:** Convert delta (0.25/0.50/0.75) to percentage (12%/5%/2%)
2. **Filter Available Strikes:** Get all strikes in the right direction from stock price
3. **Find Closest Match:** Select strike closest to target distance
4. **Retrieve Quote:** Get real-time bid/ask for premium calculation

## Delta Target to Strike Distance Mapping

Project maps user-selected delta targets to strike distance targets based on assignment likelihood:

### For PUT Options
```go
var targetDistance float64
if targetDelta == 0.75 {     // HIGH RISK
    targetDistance = 0.02    // 2% below stock price (very likely assignment)
} else if targetDelta == 0.50 {  // MODERATE RISK
    targetDistance = 0.05    // 5% below stock price
} else {  // targetDelta == 0.25   // LOW RISK  
    targetDistance = 0.12    // 12% below stock price (unlikely assignment)
}
```

### For CALL Options
```go
var targetDistance float64
if targetDelta == 0.75 {     // HIGH RISK
    targetDistance = 0.02    // 2% above stock price (very likely called away)
} else if targetDelta == 0.50 {  // MODERATE RISK
    targetDistance = 0.05    // 5% above stock price
} else {  // targetDelta == 0.25   // LOW RISK
    targetDistance = 0.12    // 12% above stock price (unlikely called away)
}
```

### Strike Selection Algorithm
```go
func (c *Client) findBestContract(contracts []*OptionContract, stockPrice *StockPrice, targetDelta float64, availableCash float64, strategy string) *models.OptionResult {
    var bestContract *OptionContract
    bestDistanceDiff := 1.0

    for _, contract := range contracts {
        strikePrice, _ := strconv.ParseFloat(contract.StrikePrice, 64)
        
        // Calculate actual distance from stock price
        var strikeDistance float64
        if strategy == "puts" {
            if strikePrice >= stockPrice.Price {
                continue  // Skip strikes at/above stock price for puts
            }
            strikeDistance = (stockPrice.Price - strikePrice) / stockPrice.Price
        } else {
            if strikePrice <= stockPrice.Price {
                continue  // Skip strikes at/below stock price for calls
            }
            strikeDistance = (strikePrice - stockPrice.Price) / stockPrice.Price
        }
        
        // Find strike closest to target distance
        distanceDiff := abs(strikeDistance - targetDistance)
        
        if distanceDiff < bestDistanceDiff {
            bestDistanceDiff = distanceDiff
            bestContract = contract
        }
    }
    
    // ... calculate contract details and return result ...
}
```

## Analysis Calculations

### Contract Calculations
```go
// Cash required per contract
cashPerContract := strikePrice * 100  // For puts
if strategy == "calls" {
    cashPerContract = stockPrice.Price * 100  // For calls (covered calls need 100 shares)
}

// Maximum contracts within available cash
maxContracts := int(availableCash / cashPerContract)
if maxContracts <= 0 {
    maxContracts = 1
}

// Financial metrics
cashUsed := float64(maxContracts) * cashPerContract
totalPremium := premium * float64(maxContracts) * 100  // Premium × contracts × 100 shares
premiumYield := (totalPremium / cashUsed) * 100
daysToExpiration := 30.0  // Estimated
annualizedReturn := (premiumYield / daysToExpiration) * 365
```

### Result Ranking
Results are sorted by total premium income (highest first):
```go
// Sort by total premium (highest first)
for i := 0; i < len(allResults); i++ {
    for j := i + 1; j < len(allResults); j++ {
        if allResults[j].TotalPremium > allResults[i].TotalPremium {
            allResults[i], allResults[j] = allResults[j], allResults[i]
        }
    }
}
```

## Form Submission Process

if (!expirationValue) {
    showError('Please select an expiration date');
    return;
}
```

### 2. Form Data Construction
```javascript
const params = new URLSearchParams();
params.append('symbols', symbolsValue);
params.append('expiration_date', expirationValue);
params.append('target_delta', selectedDelta.toString());
params.append('available_cash', cashValue);
params.append('strategy', currentStrategy);
```

### 3. API Request
```javascript
const response = await fetch('/analyze', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: params.toString()
});
```

### 4. Handler Processing
```go
func (h *Handler) AnalyzeHandler(w http.ResponseWriter, r *http.Request) {
    // Parse form data
    symbolsStr := r.FormValue("symbols")
    expirationDate := r.FormValue("expiration_date")
    targetDeltaStr := r.FormValue("target_delta")
    availableCashStr := r.FormValue("available_cash")
    strategy := r.FormValue("strategy")
    
    // Parse and clean symbols
    symbols := strings.Split(symbolsStr, ",")
    var cleanSymbols []string
    for _, symbol := range symbols {
        symbol = strings.TrimSpace(strings.ToUpper(symbol))
        if symbol != "" {
            cleanSymbols = append(cleanSymbols, symbol)
        }
    }
    
    // Call alpaca client
    response, err := h.alpacaClient.AnalyzeOptions(cleanSymbols, expirationDate, targetDelta, availableCash, strategy)
    
    // Return JSON response
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}
```

## Response Format

### Analysis Response Structure
```json
{
  "results": [
    {
      "ticker": "AAPL",
      "current_price": 178.25,
      "strike": 175.00,
      "delta": 0.018,               // Actual distance from stock price
      "premium": 2.83,              // Mid price (bid + ask) / 2
      "max_contracts": 2,           // Based on available cash
      "cash_used": 35000,           // Strike × 100 × contracts
      "total_premium": 566,         // Premium × contracts × 100
      "premium_yield": 1.62,        // (Total premium / cash used) × 100
      "annualized_return": 19.7,    // (Yield / days) × 365
      "option_type": "put"
    }
  ],
  "total_premium": 0,
  "requested_delta": 0.50,
  "strategy": "puts",
  "expiration_date": "2025-07-18"
}
```

### Results Display
```javascript
function displayResults(data) {
    // Results are pre-sorted by total premium (highest first)
    data.results.forEach((result, index) => {
        const row = document.createElement('tr');
        row.className = index === 0 ? 'bg-green-100 font-bold' : 'hover:bg-gray-50';
        
        row.innerHTML = 
            `<td class="px-4 py-3 font-bold">${index + 1}</td>` +         // Rank
            `<td class="px-4 py-3 font-bold">${result.ticker}</td>` +     // Symbol
            `<td class="px-4 py-3">${optionType}</td>` +                  // PUT/CALL
            `<td class="px-4 py-3">$${result.current_price.toFixed(2)}</td>` + // Stock Price
            `<td class="px-4 py-3">$${result.strike.toFixed(2)}</td>` +   // Strike
            `<td class="px-4 py-3">${result.delta.toFixed(3)}</td>` +     // Distance
            `<td class="px-4 py-3">$${result.premium.toFixed(2)}</td>` +  // Premium
            `<td class="px-4 py-3">${result.max_contracts}</td>` +        // Contracts
            `<td class="px-4 py-3">$${result.cash_used.toLocaleString()}</td>` + // Cash Used
            `<td class="px-4 py-3 font-bold text-green-600">$${result.total_premium.toLocaleString()}</td>` + // Total Premium
            `<td class="px-4 py-3 font-bold text-green-600">${result.premium_yield.toFixed(2)}%</td>` + // Yield
            `<td class="px-4 py-3 font-bold">${result.annualized_return.toFixed(1)}%</td>`; // Annualized
        
        tbody.appendChild(row);
    });
}
```

## Error Handling

### API Error Responses
```go
if resp.StatusCode != http.StatusOK {
    body, _ := io.ReadAll(resp.Body)
    return nil, fmt.Errorf("alpaca API error: %d - %s", resp.StatusCode, string(body))
}
```

### Common Error Scenarios
1. **Invalid API Keys**: 401 Unauthorized
2. **Rate Limiting**: 429 Too Many Requests  
3. **Invalid Symbols**: Empty results in options chain
4. **Market Closed**: Delayed or no data available
5. **Invalid Date**: Past expiration dates rejected

### Client-Side Error Display
```javascript
function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
    setTimeout(() => errorDiv.classList.add('hidden'), 8000);
}
```

## Performance Optimizations

### 1. Parallel Stock Price Requests
```go
// Get stock prices first to determine strike limits
stockPrices := make(map[string]float64)
for _, symbol := range symbols {
    stockPrice, err := c.GetStockPrice(symbol)
    if err != nil {
        continue
    }
    stockPrices[symbol] = stockPrice.Price
}
```

### 2. Filtered Options Requests
- Only request relevant strikes (above/below stock price)
- Apply type filter (puts/calls) at API level
- Limit results to prevent oversized responses

### 3. Mid-Price Calculation
- Use (bid + ask) / 2 for realistic premium estimates
- Fallback to ask price if mid is zero
- Single quote request per selected contract

## Integration Requirements

To recreate this exact logic in another application, implement:

1. **Authentication**: Alpaca API key/secret headers
2. **Stock Prices**: Latest bars endpoint for current prices
3. **Options Chain**: Filtered contracts by strategy and strikes
4. **Option Quotes**: Real-time bid/ask for premium calculation
5. **Delta Mapping**: Convert delta targets to strike distances
6. **Contract Selection**: Find closest strike to target distance
7. **Financial Calculations**: Cash requirements and yield metrics
8. **Result Ranking**: Sort by total premium income
9. **Error Handling**: API failures and validation
10. **UI Components**: Strategy selection, risk level, input validation

The key insight is that Project doesn't use actual option Greeks (delta) from market data, but instead maps user risk preferences to strike price distances from the current stock price, then finds the closest available strike in the filtered options chain.

## Differences from Current Barracuda Project

### Current Project vs. Documented System

The documented system above describes a **basic Alpaca-only options analyzer**, while the current **Barracuda project** is a significantly more advanced system with the following key differences:

#### 1. **Computation Engine**
- **Documented**: Simple Go-based calculations using delta-to-distance approximations
- **Current**: Advanced system with BOTH CPU and CUDA Black-Scholes engines + multiple execution modes
- **Impact**: Current system offers precision (CPU/CUDA Black-Scholes) AND performance options vs. basic approximations only

#### 2. **Options Pricing Methodology**
- **Documented**: Uses delta-to-distance mapping (0.25→12%, 0.50→5%, 0.75→2%) as approximation
- **Current**: Calculates actual Greeks (Delta, Gamma, Theta, Vega, Rho) using Black-Scholes model
- **Impact**: Current system provides precise option values vs. rough distance estimates

#### 3. **Performance & Scale**
- **Documented**: Sequential API calls, basic caching, limited throughput
- **Current**: GPU batch processing, Monte Carlo simulations, microsecond calculations
- **Impact**: Current system handles high-frequency analysis vs. basic retail analysis

#### 4. **Data Sources & Management**
- **Documented**: Direct Alpaca API integration only
- **Current**: Alpaca + S&P 500 symbol service + quarterly data updates + local backup
- **Impact**: Current system has enterprise-grade data management vs. basic API calls

#### 5. **Analysis Capabilities**
- **Documented**: Basic premium analysis, strike selection, yield calculations
- **Current**: Volatility skew analysis, portfolio simulations, assembly calculator integration
- **Impact**: Current system supports advanced trading strategies vs. simple screening

#### 6. **Architecture Complexity**
- **Documented**: Simple web app (Go + HTML/JS + Alpaca)
- **Current**: Multi-component system (Go + CUDA + Assembly + Web + APIs)
- **Impact**: Current system is production-grade vs. prototype/educational tool

#### 7. **API Endpoints**
- **Documented**: Single `/analyze` endpoint
- **Current**: Multiple specialized endpoints:
  - `/cuda/calculate` - GPU Black-Scholes
  - `/cuda/volatility-skew` - 25-delta analysis  
  - `/analyze-enhanced` - Combined Alpaca+CUDA
  - `/sp500/*` - Symbol management
  - `/cuda/benchmark` - Performance testing

#### 8. **Precision & Accuracy**
- **Documented**: Approximations and estimates for educational purposes
- **Current**: Financial-grade precision using assembly calculators and CUDA
- **Impact**: Current system suitable for actual trading vs. learning/demo tool

### Migration Path

If implementing the documented system's simplicity in the current Barracuda project:

1. **Add Basic Mode**: Create simplified endpoint that mimics delta-distance mapping
2. **UI Toggle**: Allow switching between "Simple" and "Advanced" analysis modes  
3. **Backward Compatibility**: Maintain `/analyze` endpoint for basic functionality
4. **Performance Options**: Let users choose CPU vs. GPU processing based on needs

The documented system represents an excellent **starting point** or **educational version**, while the current Barracuda project is the **production-ready evolution** with institutional-grade capabilities.

## What Needs to Be Fixed/Implemented

**PROBLEM**: Our current S&P 500 automated analysis does NOT implement the working options contract selection logic from the documented system.

### CRITICAL MISSING IMPLEMENTATION:

### 1. **Delta-Distance Strike Selection** (Priority: CRITICAL)
- **Current**: S&P 500 analysis exists but lacks proper contract selection
- **Missing**: Delta-to-distance conversion (0.25→12%, 0.50→5%, 0.75→2%)
- **Implementation**: Add `findBestContract()` function to S&P 500 handler

### 2. **Options Chain Retrieval & Filtering** (Priority: CRITICAL)
- **Current**: May not be filtering options chains correctly
- **Missing**: Alpaca API integration for options contracts
- **Implementation**: Add `GetOptionsChain()` with proper PUT/CALL filtering

### 3. **Real-time Option Quotes** (Priority: CRITICAL)
- **Current**: Missing bid/ask premium calculation
- **Missing**: `GetOptionQuote()` for selected contracts
- **Implementation**: Mid-price calculation (bid + ask) / 2

### 4. **Strike Distance Algorithm** (Priority: HIGH)
- **Current**: Fixed delta 0.25 but no distance matching
- **Missing**: Strike selection based on percentage from stock price
- **Implementation**: Distance calculation and closest match selection

### 5. **Financial Metrics Calculation** (Priority: HIGH)
- **Current**: Unknown if contract metrics are calculated
- **Missing**: Cash per contract, max contracts, premium yield, annualized return
- **Implementation**: Complete financial analysis per the documented system

### 6. **Alpaca API Integration** (Priority: CRITICAL)
- **Current**: Has Alpaca credentials but may not be using options endpoints
- **Missing**: 
  - `/v2/stocks/{symbol}/bars/latest` for stock prices
  - `/v2/options/contracts` for options chain
  - `/v1beta1/options/quotes/latest` for real-time quotes
- **Implementation**: Complete API integration with error handling

### 7. **Results Sorting & Display** (Priority: MEDIUM)
- **Current**: Unknown result processing
- **Missing**: Sort by total premium (highest first)
- **Implementation**: Result ranking and JSON response formatting

### 8. **Contract Validation** (Priority: HIGH)
- **Current**: May not validate contract availability
- **Missing**: Active, tradable, American-style filtering
- **Implementation**: Proper contract filtering and validation

### WHAT WE NEED TO ADD TO EXISTING S&P 500 SYSTEM:

```text
Current Barracuda S&P 500 Analysis:
┌─────────────────────────────┐
│ S&P 500 Automated Analysis  │ 
│ ✅ 25 S&P symbols          │
│ ✅ Fixed delta 0.25        │
│ ✅ Puts analysis           │
│ ❌ No contract selection   │
│ ❌ No Alpaca integration   │
│ ❌ No premium calculation  │
└─────────────────────────────┘

Target (S&P 500 + Working Contract Selection):
┌─────────────────────────────┐
│ S&P 500 + Contract Logic    │
│ ✅ Keep S&P 500 symbols    │
│ ✅ Keep automated flow     │
│ ✅ Keep existing UI        │
│ ➕ Add contract selection  │
│ ➕ Add Alpaca API calls    │
│ ➕ Add premium calculation │
│ ➕ Add results display     │
└─────────────────────────────┘
```

### IMPLEMENTATION PRIORITY:

**Phase 1 - CORE CONTRACT LOGIC:**
1. Add Alpaca API client to S&P 500 handler
2. Implement `GetOptionsChain()` for contract retrieval
3. Add `findBestContract()` with delta-distance matching
4. Integrate `GetOptionQuote()` for real-time pricing

**Phase 2 - STRIKE SELECTION:**
1. Implement delta-to-distance conversion (0.25 → 12%)
2. Add strike distance algorithm
3. Add contract filtering (active, tradable, American)
4. Build financial metrics calculation

**Phase 3 - RESULTS & DISPLAY:**
1. Format results as JSON (like documented system)
2. Sort by total premium income
3. Update results display in existing UI
4. Add error handling and validation

### FILES THAT NEED ENHANCEMENT (NOT REWRITE):

- `internal/handlers/options.go` - Add contract selection logic to existing S&P 500 handler
- `internal/alpaca/client.go` - Create if missing, add options API methods
- `web/static/app.js` - Enhance results display to show contract details
- `internal/models/models.go` - Add option result structures

**The real issue: S&P 500 analysis exists but doesn't actually select and price real options contracts using the working algorithm from the documented system.**