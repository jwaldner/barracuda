# AI Analysis Prompt - Black-Scholes Validation

Extract and validate Black-Scholes calculations from the provided JSON. Handle variations such as multiple contracts, mismatched tickers, or missing fields by noting them explicitly. Be specific, constructive, and focus on mathematical accuracy. Use standard Black-Scholes assumptions unless specified (e.g., continuous compounding, no dividends for simplicity). If recomputing, use precise math if needed (e.g., scipy.stats.norm for CDF). Remain neutral and evidence-based.

## CHECK THESE CALCULATIONS:

Extract inputs from the JSON (e.g., S from stock_price in GetStockPricesBatch or response, K/strike from contracts, T from time-to-expiration calculation using timestamps/expiration_date in years as decimal, r and σ from variables in BlackScholesCalculation). If inputs are missing, inconsistent (e.g., ticker mismatch like KO vs TSLA), or spread across sections, flag them clearly.

1. Is the theoretical price correct? Recompute using the Black-Scholes formula:
   - For puts: P = K * exp(-r*T) * N(-d2) - S * N(-d1)
   - Where d1 = [ln(S/K) + (r + σ²/2)*T] / (σ * √T)
   - d2 = d1 - σ * √T
   - Show d1, d2, and full calculation steps with expected value (use 6-8 decimal places for precision).

2. Are the Greeks mathematically accurate? Recompute and compare:
   - Delta (put): -N(-d1)
   - Gamma: N'(d1) / (S * σ * √T)
   - Theta (put, daily): [-S * N'(d1) * σ / (2√T) + r * K * exp(-r*T) * N(-d2)] / 365
   - Vega (per 1% vol): S * √T * N'(d1) / 100
   - Rho (put, per 1% rate): -K * T * exp(-r*T) * N(-d2) / 100
   Show formulas and steps.

3. Any calculation errors or data inconsistencies? Check for issues like incorrect option type (call vs. put), time units (T in years?), volatility assumptions, JSON mismatches, or other anomalies. Suggest fixes if found. Allow ±0.001 tolerance in comparisons due to rounding.

## RESPONSE FORMAT:

For each contract analyzed (if multiple, prioritize the sample contract or first 3; note if summarizing):
- THEORETICAL PRICE: [correct/incorrect] - Expected X (from [brief calc, e.g., d1=..., d2=...]), got Y
- DELTA: [correct/incorrect] - Expected X (from formula), got Y
- GAMMA: [correct/incorrect] - Expected X (from formula), got Y
- THETA: [correct/incorrect] - Expected X (from formula), got Y
- VEGA: [correct/incorrect] - Expected X (from formula), got Y
- RHO: [correct/incorrect] - Expected X (from formula), got Y
- INCONSISTENCIES: [List any, e.g., "Ticker mismatch: Expected TSLA, got KO"]
- OVERALL: [PASS/FAIL] - [brief summary, including suggested fixes like "Adjust theta for daily decay"]

If no calculations present: "NO DATA: No valid Black-Scholes inputs found in JSON."
Be direct. Show expected vs actual values. Focus on math errors, not investment advice.

## Key Improvements Made:

1. **Better Input Extraction**: Explicit guidance on where to find inputs in JSON structure
2. **Comprehensive Formula Documentation**: Full Black-Scholes formulas with step-by-step calculations
3. **Proper Put Option Handling**: Correct formulas specifically for put options
4. **Intermediate Value Verification**: Shows d1, d2 calculations to catch errors early
5. **Error Handling**: Addresses JSON variations, missing fields, ticker mismatches
6. **Precision Specifications**: 6-8 decimal places, ±0.001 tolerance for rounding
7. **Units Clarification**: T in years, volatility as decimal, theta daily, etc.