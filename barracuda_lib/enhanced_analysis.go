package barracuda

import (
	"log"
	"time"
)

// Enhanced analysis that separates preprocessing timing from calculation timing
func (be *BaracudaEngine) AnalyzeSymbolsBatchWithTiming(symbols []string, stockPrices map[string]float64,
	optionsChains map[string][]OptionContract, expirationDate string) []SymbolAnalysisResult {

	var results []SymbolAnalysisResult

	var totalPreprocessingTime float64
	var totalBlackScholesTime float64
	var totalSkewTime float64
	var totalRecordsProcessed int

	for _, symbol := range symbols {
		result := SymbolAnalysisResult{
			Symbol:     symbol,
			Expiration: expirationDate,
		}

		stockPrice, hasStock := stockPrices[symbol]
		options, hasOptions := optionsChains[symbol]

		if !hasStock || !hasOptions {
			continue
		}

		result.StockPrice = stockPrice

		// Determine execution mode based on forced mode and CUDA availability
		if be.executionMode == ExecutionModeCUDA && be.IsCudaAvailable() {
			result.ExecutionMode = "CUDA"
		} else if be.executionMode == ExecutionModeAuto && be.IsCudaAvailable() {
			result.ExecutionMode = "CUDA"
		} else {
			result.ExecutionMode = "CPU"
		}

		// PHASE 1: PREPROCESSING WITH SEPARATE TIMING
		// Time the preprocessing operations separately from Black-Scholes calculation
		preprocessStart := time.Now()

		var puts, calls []OptionContract
		processedCount := 0

		for _, option := range options {
			option.UnderlyingPrice = stockPrice
			option.TimeToExpiration = 0.085 // ~31 days for Jan 2026 expiration
			option.RiskFreeRate = 0.05

			// Use market price (TheoreticalPrice) to calculate implied volatility
			marketPrice := option.TheoreticalPrice
			if marketPrice > 0.01 { // Only calculate IV for options with meaningful market price
				// Simple implied volatility estimation (Newton-Raphson would be better)
				option.Volatility = be.estimateImpliedVolatility(marketPrice, stockPrice, option.StrikePrice,
					option.TimeToExpiration, option.RiskFreeRate, option.OptionType)
			} else {
				option.Volatility = 0.25 // Default for very cheap options
			}

			if option.OptionType == 'P' {
				puts = append(puts, option)
			} else {
				calls = append(calls, option)
			}
			processedCount++
		}

		preprocessDuration := time.Since(preprocessStart)
		preprocessMs := preprocessDuration.Seconds() * 1000

		// Accumulate preprocessing metrics
		totalPreprocessingTime += preprocessMs
		totalRecordsProcessed += processedCount

		log.Printf("🔧 PREPROCESS %s: %.3fms | %d contracts → %d puts, %d calls | Mode: %s",
			symbol, preprocessMs, processedCount, len(puts), len(calls), result.ExecutionMode)

		// PHASE 2: BLACK-SCHOLES CALCULATIONS (CUDA/CPU)
		blackScholesStart := time.Now()

		if len(puts) > 0 {
			calculatedPuts, err := be.CalculateBlackScholes(puts)
			if err == nil {
				result.PutsWithIV = calculatedPuts
			}
		}

		if len(calls) > 0 {
			calculatedCalls, err := be.CalculateBlackScholes(calls)
			if err == nil {
				result.CallsWithIV = calculatedCalls
			}
		}

		blackScholesDuration := time.Since(blackScholesStart)
		totalBlackScholesTime += blackScholesDuration.Seconds() * 1000

		log.Printf("⚡ BLACK-SCHOLES %s: %.3fms | %d puts + %d calls",
			symbol, blackScholesDuration.Seconds()*1000, len(result.PutsWithIV), len(result.CallsWithIV))

		// PHASE 3: 25-DELTA SKEW CALCULATION WITH SEPARATE TIMING
		if len(result.PutsWithIV) > 0 && len(result.CallsWithIV) > 0 {
			skewStart := time.Now()
			result.VolatilitySkew = be.calculate25DeltaSkew(result.PutsWithIV, result.CallsWithIV, expirationDate)
			skewDuration := time.Since(skewStart)
			skewMs := skewDuration.Seconds() * 1000

			totalSkewTime += skewMs
			contractsAnalyzed := len(result.PutsWithIV) + len(result.CallsWithIV)

			log.Printf("📊 SKEW CALC %s: %.3fms | %.4f skew (%.4f put - %.4f call) | %d contracts analyzed",
				symbol, skewMs, result.VolatilitySkew.Skew,
				result.VolatilitySkew.Put25DIV, result.VolatilitySkew.Call25DIV, contractsAnalyzed)
		}

		result.TotalOptionsProcessed = len(result.PutsWithIV) + len(result.CallsWithIV)
		results = append(results, result)
	}

	// SUMMARY TIMING REPORT
	totalTime := totalPreprocessingTime + totalBlackScholesTime + totalSkewTime

	log.Printf("📈 PERFORMANCE SUMMARY:")
	log.Printf("  📦 Preprocessing: %.3fms (%.1f%%) | %d records",
		totalPreprocessingTime, (totalPreprocessingTime/totalTime)*100, totalRecordsProcessed)
	log.Printf("  ⚡ Black-Scholes: %.3fms (%.1f%%)",
		totalBlackScholesTime, (totalBlackScholesTime/totalTime)*100)
	log.Printf("  📊 Skew Calc:     %.3fms (%.1f%%)",
		totalSkewTime, (totalSkewTime/totalTime)*100)
	log.Printf("  🎯 Total Compute: %.3fms | Mode: %s", totalTime,
		func() string {
			if len(results) > 0 {
				return results[0].ExecutionMode
			} else {
				return "N/A"
			}
		}())

	// Set timing for all results
	for i := range results {
		results[i].CalculationTimeMs = totalTime
	}

	return results
}
