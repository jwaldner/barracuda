package treasury

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"
)

type TreasuryClient struct {
	httpClient    *http.Client
	baseURL       string
	lastKnownRate float64
	lastFetchTime time.Time
}

type TreasuryResponse struct {
	Data []TreasuryRate `json:"data"`
	Meta struct {
		Count int `json:"count"`
	} `json:"meta"`
}

type TreasuryRate struct {
	RecordDate            string `json:"record_date"`
	SecurityDesc          string `json:"security_desc"`
	AvgInterestRateAmount string `json:"avg_interest_rate_amt"`
}

func NewTreasuryClient() *TreasuryClient {
	client := &TreasuryClient{
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
		baseURL: "https://api.fiscaldata.treasury.gov/services/api/fiscal_service",
	}

	// Initialize with current rate on startup
	if rate, err := client.fetchRiskFreeRate(); err == nil {
		client.lastKnownRate = rate
		client.lastFetchTime = time.Now()
		fmt.Printf("🏛️ Initialized Treasury client with rate: %.6f (%.3f%%)\n", rate, rate*100)
	} else {
		// If initial fetch fails, use a reasonable default as last known
		client.lastKnownRate = 0.04 // 4% as emergency default
		fmt.Printf("⚠️ Failed to fetch initial Treasury rate: %v, using emergency default: 4%%\n", err)
	}

	return client
}

// fetchRiskFreeRate does the actual API call (internal method)
func (tc *TreasuryClient) fetchRiskFreeRate() (float64, error) {
	url := fmt.Sprintf("%s/v2/accounting/od/avg_interest_rates?fields=avg_interest_rate_amt,record_date&filter=security_desc:eq:Treasury%%20Bills&sort=-record_date&page[size]=1", tc.baseURL)

	resp, err := tc.httpClient.Get(url)
	if err != nil {
		return 0, fmt.Errorf("failed to fetch Treasury rate: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("Treasury API returned status %d", resp.StatusCode)
	}

	var treasuryResp TreasuryResponse
	if err := json.NewDecoder(resp.Body).Decode(&treasuryResp); err != nil {
		return 0, fmt.Errorf("failed to decode Treasury response: %w", err)
	}

	if len(treasuryResp.Data) == 0 {
		return 0, fmt.Errorf("no Treasury rate data returned")
	}

	// Convert percentage string to float64 (e.g., "3.983" -> 0.03983)
	rateStr := treasuryResp.Data[0].AvgInterestRateAmount
	rate, err := strconv.ParseFloat(rateStr, 64)
	if err != nil {
		return 0, fmt.Errorf("failed to parse rate %s: %w", rateStr, err)
	}

	// Convert percentage to decimal (3.983% -> 0.03983)
	return rate / 100.0, nil
}

// GetRiskFreeRate fetches the most recent Treasury Bill rate as the risk-free rate
func (tc *TreasuryClient) GetRiskFreeRate() (float64, error) {
	rate, err := tc.fetchRiskFreeRate()
	if err != nil {
		return 0, err
	}

	// Update cache on successful fetch
	tc.lastKnownRate = rate
	tc.lastFetchTime = time.Now()

	fmt.Printf("📈 Fetched Treasury Bill rate: %.3f%% (%.6f decimal) - updated cache\n",
		rate*100, rate)

	return rate, nil
}

// GetRiskFreeRateWithLastKnown tries to fetch current rate, uses last known if fetch fails
func (tc *TreasuryClient) GetRiskFreeRateWithLastKnown() float64 {
	// Try to fetch fresh rate
	if rate, err := tc.GetRiskFreeRate(); err == nil {
		return rate
	}

	// Use last known rate if fetch failed
	age := time.Since(tc.lastFetchTime)
	fmt.Printf("⚠️ Treasury API failed, using last known rate: %.6f (%.3f%%) from %v ago\n",
		tc.lastKnownRate, tc.lastKnownRate*100, age.Round(time.Minute))

	return tc.lastKnownRate
}

// GetCacheInfo returns information about the cached rate
func (tc *TreasuryClient) GetCacheInfo() (rate float64, age time.Duration, isInitialized bool) {
	if tc.lastFetchTime.IsZero() {
		return tc.lastKnownRate, 0, false
	}
	return tc.lastKnownRate, time.Since(tc.lastFetchTime), true
}
