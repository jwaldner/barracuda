# DeltaQuest Options Filtering and Table Calculations

This document shows the complete code for how options are filtered per symbol and how the results table fields are calculated.

## Options Filtering Process Per Symbol

### 1. Initial API Call Filtering (`GetOptionsChain` function)

The filtering starts at the API level to reduce the massive universe of options contracts to only relevant ones:

```go
func (c *Client) GetOptionsChain(symbols []string, expiration string, strategy string) (map[string][]*OptionContract, error) {
    contractsBySymbol := make(map[string][]*OptionContract)

    // Get stock prices first to determine strike limits
    stockPrices := make(map[string]float64)
    for _, symbol := range symbols {
        symbol = strings.TrimSpace(symbol)
        stockPrice, err := c.GetStockPrice(symbol)
        if err != nil {
            fmt.Printf("Error getting stock price for %s: %v\n", symbol, err)
            continue
        }
        stockPrices[symbol] = stockPrice.Price
    }

    // Make separate API calls for each symbol with proper filtering
    for _, symbol := range symbols {
        symbol = strings.TrimSpace(symbol)

        stockPrice, exists := stockPrices[symbol]
        if !exists {
            continue
        }

        endpoint := "/v2/options/contracts"
        req, err := http.NewRequest("GET", c.BaseURL+endpoint, nil)
        if err != nil {
            fmt.Printf("Error creating request for %s: %v\n", symbol, err)
            continue
        }

        // Add query parameters with filtering
        q := req.URL.Query()
        q.Add("underlying_symbols", symbol)
        if expiration != "" {
            q.Add("expiration_date", expiration)
        }

        // Filter by option type (puts or calls)
        if strategy == "puts" {
            q.Add("type", "put")
            // For puts: only get strikes below stock price
            q.Add("strike_price_lte", fmt.Sprintf("%.0f", stockPrice-1))
        } else {
            q.Add("type", "call")
            // For calls: only get strikes above stock price
            q.Add("strike_price_gte", fmt.Sprintf("%.0f", stockPrice+1))
        }

        q.Add("limit", "1000")
        req.URL.RawQuery = q.Encode()

        req.Header.Add("APCA-API-KEY-ID", c.APIKey)
        req.Header.Add("APCA-API-SECRET-KEY", c.SecretKey)

        fmt.Printf("Making options request for %s: %s\n", symbol, req.URL.String())

        resp, err := c.HTTPClient.Do(req)
        if err != nil {
            fmt.Printf("Error making request for %s: %v\n", symbol, err)
            continue
        }
        defer resp.Body.Close()

        if resp.StatusCode != http.StatusOK {
            body, _ := io.ReadAll(resp.Body)
            fmt.Printf("Alpaca API error for %s: %d - %s\n", symbol, resp.StatusCode, string(body))
            continue
        }

        var alpacaResp AlpacaOptionsResponse
        if err := json.NewDecoder(resp.Body).Decode(&alpacaResp); err != nil {
            fmt.Printf("Error decoding response for %s: %v\n", symbol, err)
            continue
        }

        // Convert to pointers and sort by strike price
        contracts := make([]*OptionContract, len(alpacaResp.Options))
        for i := range alpacaResp.Options {
            contracts[i] = &alpacaResp.Options[i]
        }

        // Sort by strike price (ascending)
        for i := 0; i < len(contracts); i++ {
            for j := i + 1; j < len(contracts); j++ {
                strike1, _ := strconv.ParseFloat(contracts[i].StrikePrice, 64)
                strike2, _ := strconv.ParseFloat(contracts[j].StrikePrice, 64)
                if strike1 > strike2 {
                    contracts[i], contracts[j] = contracts[j], contracts[i]
                }
            }
        }

        contractsBySymbol[symbol] = contracts
        fmt.Printf("Found %d %s contracts for %s (stock price: $%.2f)\n",
            len(contracts), strategy, symbol, stockPrice)
    }

    return contractsBySymbol, nil
}
```

### 2. Contract Selection Filtering (`findBestContract` function)

After getting the filtered contracts from the API, the system selects the best one based on user's risk preference:

```go
func (c *Client) findBestContract(contracts []*OptionContract, stockPrice *StockPrice, targetDelta float64, availableCash float64, strategy string) *models.OptionResult {
    var bestContract *OptionContract
    bestDistanceDiff := 1.0

    // Filter by strategy (puts or calls)
    targetType := "put"
    if strategy == "calls" {
        targetType = "call"
    }

    // Map delta to distance targets based on assignment likelihood
    var targetDistance float64
    if targetType == "put" {
        // For puts: higher delta = closer to stock price (below it)
        // Assignment likelihood increases as strike approaches stock price from below
        if targetDelta == 0.75 { // High risk = very likely assignment = very close to stock price
            targetDistance = 0.02 // 2% below stock price
        } else if targetDelta == 0.50 { // Medium risk = moderate assignment = moderately close
            targetDistance = 0.05 // 5% below stock price
        } else { // Low risk = unlikely assignment = further away
            targetDistance = 0.12 // 12% below stock price
        }
    } else {
        // For calls: higher delta = closer to stock price (above it)
        // Call away likelihood increases as strike approaches stock price from above
        if targetDelta == 0.75 { // High risk = very likely called = very close to stock price
            targetDistance = 0.02 // 2% above stock price
        } else if targetDelta == 0.50 { // Medium risk = moderate call = moderately close
            targetDistance = 0.05 // 5% above stock price
        } else { // Low risk = unlikely called = further away
            targetDistance = 0.12 // 12% above stock price
        }
    }

    for _, contract := range contracts {
        if contract.Type != targetType {
            continue
        }

        // Skip if not tradable
        if !contract.Tradable {
            continue
        }

        // Parse strike price
        strikePrice, err := strconv.ParseFloat(contract.StrikePrice, 64)
        if err != nil {
            continue
        }

        // Calculate distance based on assignment likelihood
        var strikeDistance float64
        if targetType == "put" {
            // For puts: only consider strikes BELOW stock price
            if strikePrice >= stockPrice.Price {
                continue // Skip strikes at/above stock price
            }
            // Distance = how far below stock price (positive value)
            strikeDistance = (stockPrice.Price - strikePrice) / stockPrice.Price
        } else {
            // For calls: only consider strikes ABOVE stock price
            if strikePrice <= stockPrice.Price {
                continue // Skip strikes at/below stock price
            }
            // Distance = how far above stock price (positive value)
            strikeDistance = (strikePrice - stockPrice.Price) / stockPrice.Price
        }

        // Find strike closest to target distance
        distanceDiff := abs(strikeDistance - targetDistance)

        // Debug output to see strike distance matching
        fmt.Printf("Contract %s: strike $%.2f, distance %.1f%% %s stock, target %.1f%%, diff %.1f%%\n",
            contract.Symbol, strikePrice, strikeDistance*100,
            map[bool]string{true: "below", false: "above"}[targetType == "put"],
            targetDistance*100, distanceDiff*100)

        if distanceDiff < bestDistanceDiff {
            bestDistanceDiff = distanceDiff
            bestContract = contract
        }
    }

    if bestContract == nil {
        fmt.Printf("❌ No contract found matching criteria\n")
        return nil
    }

    fmt.Printf("🏆 WINNER: %s selected as best match!\n", bestContract.Symbol)

    // ... continue to calculations section below ...
}
```

### Filtering Summary

**Per Symbol, the filtering process:**

1. **API Level** (reduces ~2000 to ~50 contracts):
   - `underlying_symbols=SYMBOL` (single symbol per request)
   - `type=put/call` (strategy-based)
   - `strike_price_lte/gte=PRICE` (directional from stock price)
   - `expiration_date=DATE` (specific expiration)
   - `limit=1000` (maximum results)

2. **Contract Level** (reduces ~50 to 1 contract):
   - Skip wrong option type
   - Skip non-tradable contracts  
   - Skip in-the-money options
   - Calculate distance from stock price
   - Select closest match to target distance based on risk level

3. **Quote Level** (gets real-time pricing):
   - Get bid/ask for selected contract
   - Use mid-price for premium calculation
   - Calculate financial metrics (contracts, yield, etc.)

## Results Table Field Calculations

### Core Calculation Logic

After selecting the best contract, the system calculates all the financial metrics displayed in the results table:

```go
// Parse strike price
strikePrice, err := strconv.ParseFloat(bestContract.StrikePrice, 64)
if err != nil {
    return nil
}

// Get real option quote for accurate premium
quote, err := c.GetOptionQuote(bestContract.Symbol)
if err != nil {
    fmt.Printf("Error getting quote for %s: %v\n", bestContract.Symbol, err)
    return nil
}

// Use mid price (average of bid and ask)
premium := (quote.BidPrice + quote.AskPrice) / 2
if premium <= 0 {
    // Fallback to ask price if mid is zero
    premium = quote.AskPrice
}

fmt.Printf("Real premium for %s: $%.2f (bid: $%.2f, ask: $%.2f)\n",
    bestContract.Symbol, premium, quote.BidPrice, quote.AskPrice)

// Calculate contract details
cashPerContract := strikePrice * 100
if strategy == "calls" {
    cashPerContract = stockPrice.Price * 100
}

maxContracts := int(availableCash / cashPerContract)
if maxContracts <= 0 {
    maxContracts = 1
}

cashUsed := float64(maxContracts) * cashPerContract
totalPremium := premium * float64(maxContracts) * 100

premiumYield := (totalPremium / cashUsed) * 100
daysToExpiration := 30.0
annualizedReturn := (premiumYield / daysToExpiration) * 365

// Calculate actual distance for display (not estimated delta)
actualDistance := abs(strikePrice-stockPrice.Price) / stockPrice.Price

// Final debug summary
fmt.Printf("📊 FINAL RESULT: %s at $%.2f (%.1f%% below stock) with $%.2f premium = $%.2f total\n",
    bestContract.Symbol, strikePrice, actualDistance*100, premium, totalPremium)

return &models.OptionResult{
    Ticker:           bestContract.UnderlyingSymbol,
    CurrentPrice:     stockPrice.Price,
    Strike:           strikePrice,
    Delta:            actualDistance, // Show actual distance instead of fake delta
    Premium:          premium,
    MaxContracts:     maxContracts,
    CashUsed:         cashUsed,
    TotalPremium:     totalPremium,
    PremiumYield:     premiumYield,
    AnnualizedReturn: annualizedReturn,
    OptionType:       bestContract.Type,
}
```

### Field-by-Field Breakdown

| Table Column | Calculation | Code |
|-------------|-------------|------|
| **Rank** | Position after sorting by total premium | `index + 1` |
| **Ticker** | Stock symbol | `bestContract.UnderlyingSymbol` |
| **Type** | PUT or CALL | `bestContract.Type` |
| **Stock Price** | Current stock price | `stockPrice.Price` |
| **Strike** | Option strike price | `strikePrice` |
| **Delta** | Distance from stock price | `abs(strikePrice-stockPrice.Price) / stockPrice.Price` |
| **Premium** | Mid price (bid+ask)/2 | `(quote.BidPrice + quote.AskPrice) / 2` |
| **Contracts** | Max contracts affordable | `int(availableCash / cashPerContract)` |
| **Cash Used** | Total cash required | `float64(maxContracts) * cashPerContract` |
| **Total Premium** | Total income | `premium * float64(maxContracts) * 100` |
| **Yield** | Return percentage | `(totalPremium / cashUsed) * 100` |
| **Annualized** | Yearly return estimate | `(premiumYield / 30.0) * 365` |

### Strategy-Specific Cash Calculation

```go
// For PUT options (Cash-Secured Puts):
cashPerContract := strikePrice * 100  // Need cash to buy 100 shares at strike

// For CALL options (Covered Calls):  
if strategy == "calls" {
    cashPerContract = stockPrice.Price * 100  // Need to own 100 shares at current price
}
```

### Data Model Definition

The calculated fields are returned in this structured format:

```go
// OptionResult represents the analysis result for one option
type OptionResult struct {
    Ticker           string  `json:"ticker"`
    CurrentPrice     float64 `json:"current_price"`
    Strike           float64 `json:"strike"`
    Delta            float64 `json:"delta"`
    Premium          float64 `json:"premium"`
    MaxContracts     int     `json:"max_contracts"`
    CashUsed         float64 `json:"cash_used"`
    TotalPremium     float64 `json:"total_premium"`
    PremiumYield     float64 `json:"premium_yield"`
    AnnualizedReturn float64 `json:"annualized_return"`
    OptionType       string  `json:"option_type"`
}
```

### Key Calculation Notes

1. **Premium**: Uses mid-price between bid/ask for realistic estimate
2. **Total Premium**: Multiplied by 100 because each contract = 100 shares  
3. **Max Contracts**: Integer division ensures whole contracts only
4. **Annualized Return**: Assumes 30 days to expiration (could be improved with actual days)
5. **Delta Field**: Actually shows price distance percentage, not real option delta
6. **Cash Requirements**: Different for puts (strike × 100) vs calls (stock price × 100)

This calculation logic determines the financial attractiveness of each option and ranks them by total premium income potential.