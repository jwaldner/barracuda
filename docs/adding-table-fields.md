# Adding New Fields to Options Analysis Table

## Overview
This document explains how to add new fields to the options analysis table and CSV exports without breaking the frontend or requiring JavaScript changes.

## Architecture Rules
- **Config.yaml**: Only for configuration data (filename formats, settings)  
- **Backend**: Business logic including field names, headers, data processing
- **Frontend**: Uses template functions only - NO hardcoded values

## Step-by-Step Process

### 1. Add Field to Data Model ✅
**File:** `/internal/models/models.go`

Add your new field to the `OptionResult` struct:
```go
type OptionResult struct {
    // Existing fields...
    YourNewField float64 `json:"your_new_field"`
}
```

### 2. Populate the Field ✅
**File:** `/internal/handlers/options.go`

In the `processRealOptions()` function, calculate and set your new field:
```go
result := models.OptionResult{
    // Existing field assignments...
    YourNewField: calculateYourValue(contract, stockPrice),
}
```

### 3. Add to Display Configuration ✅
**File:** `/internal/handlers/options.go` (around lines 164-171)

Update both arrays in the template functions:
```go
"tableHeaders": func() []string {
    return []string{"Rank", "Ticker", "Company", "Sector", "Strike", "Stock_Price", "Premium", "Max_Contracts", "Total_Premium", "Profit_Percentage", "Delta", "Expiration", "Days_To_Exp", "Your New Field"}
},
"tableFieldKeys": func() []string {
    return []string{"rank", "ticker", "company", "sector", "strike", "stock_price", "premium", "max_contracts", "total_premium", "profit_percentage", "delta", "expiration", "days_to_exp", "your_new_field"}
},
```

### 4. Optional: Custom Formatting
**File:** `/web/templates/home.html`

Only needed if you want special formatting. The field will display automatically with default formatting.

## What NOT to Touch ❌

- **JavaScript files** (`/web/static/app.js`) - Frontend automatically picks up new fields
- **Config.yaml** - Field names are not configuration data
- **Any hardcoded arrays in frontend** - Use template functions only

## Testing Your Changes

1. Restart the server: `./barracuda`
2. Run an analysis
3. Check that your new field appears in:
   - Web table display
   - CSV copy (📋 Copy CSV button)  
   - CSV download (💾 Download CSV button)

## Example: Adding "Implied Volatility" Field

### Step 1: Model
```go
type OptionResult struct {
    // ... existing fields
    ImpliedVolatility float64 `json:"implied_volatility"`
}
```

### Step 2: Population
```go
result := models.OptionResult{
    // ... existing assignments
    ImpliedVolatility: contract.ImpliedVol,
}
```

### Step 3: Display
```go
"tableHeaders": func() []string {
    return []string{..existing.., "IV"}
},
"tableFieldKeys": func() []string {
    return []string{..existing.., "implied_volatility"}
},
```

## Architecture Benefits

- ✅ **Single source of truth**: Field definitions in backend only
- ✅ **Zero frontend changes**: Template functions handle everything  
- ✅ **Automatic CSV inclusion**: New fields appear in exports automatically
- ✅ **Type safety**: Go compiler validates field usage
- ✅ **Future-proof**: Adding fields never breaks existing code

## Common Mistakes to Avoid

1. **Don't hardcode field names in JavaScript** - Use `window.tableFieldKeys()` 
2. **Don't put field names in config.yaml** - They're business logic, not config
3. **Don't forget both arrays** - Update both `tableHeaders` and `tableFieldKeys`
4. **Match the casing** - Use snake_case for keys, Title Case for headers

## Questions?

Check the inline comments in `/internal/handlers/options.go` around the template functions for the latest guidance.