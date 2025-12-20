package grok

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	"github.com/jwaldner/barracuda/internal/config"
	"github.com/jwaldner/barracuda/internal/logger"
)

// Client represents a Grok AI API client
type Client struct {
	apiKey   string
	endpoint string
	model    string
	client   *http.Client
}

// AnalysisRequest represents a request for options analysis
type AnalysisRequest struct {
	Ticker    string `json:"ticker"`
	AuditData string `json:"audit_data"`
}

// AnalysisResponse represents the response from Grok API
type AnalysisResponse struct {
	Content string `json:"content"`
	Tokens  int    `json:"tokens"`
}

// NewClient creates a new Grok API client
func NewClient() (*Client, error) {
	cfg := config.Load()

	if cfg.GrokAPIKey == "" {
		logger.Warn.Printf("⚠️🤖 GROK WARNING: API key not configured - check config.yaml")
		return nil, fmt.Errorf("Grok API key not configured")
	}

	logger.Warn.Printf("🤖 GROK: Client initialized - Endpoint: %s, Model: %s", cfg.GrokEndpoint, cfg.GrokModel)

	// Custom transport with proper timeouts based on Grok's recommendations
	transport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 300 * time.Second, // Increased to 5 minutes as recommended
		ExpectContinueTimeout: 1 * time.Second,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   10,
		IdleConnTimeout:       90 * time.Second,
	}

	// HTTP client with extended timeout for long responses
	httpClient := &http.Client{
		Transport: transport,
		Timeout:   3600 * time.Second, // 1 hour as recommended by Grok
	}

	return &Client{
		apiKey:   cfg.GrokAPIKey,
		endpoint: cfg.GrokEndpoint,
		model:    cfg.GrokModel,
		client:   httpClient,
	}, nil
}

// TestConnectivity performs a basic connectivity test to the X.AI endpoint
func (c *Client) TestConnectivity() error {
	logger.Warn.Printf("🤖 GROK: Testing connectivity to %s", c.endpoint)

	// Create a minimal test request
	testRequest := map[string]interface{}{
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": "test",
			},
		},
		"model":       c.model,
		"max_tokens":  1,
		"temperature": 0,
	}

	reqBytes, err := json.Marshal(testRequest)
	if err != nil {
		return fmt.Errorf("failed to marshal test request: %v", err)
	}

	req, err := http.NewRequest("POST", c.endpoint, bytes.NewBuffer(reqBytes))
	if err != nil {
		return fmt.Errorf("failed to create test request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("User-Agent", "barracuda-options-analyzer/1.0")

	// Set a shorter timeout for the connectivity test
	testClient := &http.Client{Timeout: 10 * time.Second}

	resp, err := testClient.Do(req)
	if err != nil {
		logger.Warn.Printf("⚠️🤖 GROK WARNING: Connectivity test failed: %v", err)
		return fmt.Errorf("connectivity test failed: %v", err)
	}
	defer resp.Body.Close()

	logger.Warn.Printf("🤖 GROK: Connectivity test successful - Status: %d", resp.StatusCode)
	return nil
}

// AnalyzeOptions calls the Grok API to analyze options data
func (c *Client) AnalyzeOptions(ticker, auditData string) (*AnalysisResponse, error) {
	logger.Warn.Printf("🤖 GROK: Starting analysis for ticker: %s", ticker)

	// Construct the expert prompt with audit data
	prompt := fmt.Sprintf(`You are an expert quantitative analyst specializing in Black-Scholes option pricing models. Validate the mathematical accuracy of the provided calculations for %s.

AUDIT DATA:
%s

Please provide:
1. **Calculation Verification**: Verify the Black-Scholes formula implementation and Greek calculations
2. **Input Validation**: Check if S, K, T, r, sigma inputs are reasonable and properly formatted
3. **Mathematical Accuracy**: Confirm delta, gamma, theta, vega, rho calculations match expected values
4. **Formula Consistency**: Validate the theoretical price against the Black-Scholes formula
5. **Data Integrity**: Identify any inconsistencies, errors, or anomalies in the calculations

Focus on mathematical correctness, not investment advice. Verify calculations using: C = S*N(d1) - K*e^(-r*T)*N(d2) for calls, P = K*e^(-r*T)*N(-d2) - S*N(-d1) for puts.`, ticker, auditData)

	// Prepare Grok API request
	requestBody := map[string]interface{}{
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": prompt,
			},
		},
		"model":       c.model,
		"stream":      false,
		"temperature": 0.3, // More focused, less creative
	}

	reqBytes, err := json.Marshal(requestBody)
	if err != nil {
		logger.Warn.Printf("⚠️🤖 GROK WARNING: Failed to marshal request body: %v", err)
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	logger.Warn.Printf("🤖 GROK: Sending API request to X.AI (payload: %d bytes)", len(reqBytes))

	// Make HTTP request to Grok API with proper context
	req, err := http.NewRequestWithContext(context.Background(), "POST", c.endpoint, bytes.NewBuffer(reqBytes))
	if err != nil {
		logger.Warn.Printf("⚠️🤖 GROK WARNING: Failed to create HTTP request: %v", err)
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("User-Agent", "barracuda-options-analyzer/1.0")

	resp, err := c.client.Do(req)
	if err != nil {
		// Enhanced error logging for network diagnosis
		logger.Warn.Printf("⚠️🤖 GROK WARNING: HTTP request failed - Endpoint: %s", c.endpoint)
		logger.Warn.Printf("⚠️🤖 GROK WARNING: Error details: %v", err)
		logger.Warn.Printf("⚠️🤖 GROK WARNING: This may indicate network connectivity issues or API service problems")
		return nil, fmt.Errorf("API request failed: %v", err)
	}
	defer resp.Body.Close()

	logger.Warn.Printf("🤖 GROK: Received response - Status: %d %s", resp.StatusCode, resp.Status)

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		logger.Warn.Printf("⚠️🤖 GROK WARNING: API error response (Status %d): %s", resp.StatusCode, string(body))
		return nil, fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var grokResponse struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&grokResponse); err != nil {
		logger.Warn.Printf("⚠️🤖 GROK WARNING: Failed to decode API response: %v", err)
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}

	if len(grokResponse.Choices) == 0 {
		logger.Warn.Printf("⚠️🤖 GROK WARNING: API returned empty choices array")
		return nil, fmt.Errorf("no response from Grok API")
	}

	analysis := grokResponse.Choices[0].Message.Content
	tokens := grokResponse.Usage.TotalTokens

	logger.Warn.Printf("🤖 GROK: Analysis completed successfully - %d tokens used, response: %d chars", tokens, len(analysis))

	return &AnalysisResponse{
		Content: analysis,
		Tokens:  tokens,
	}, nil
}

// FormatAnalysis formats the Grok analysis as markdown
func FormatAnalysis(ticker string, response *AnalysisResponse) string {
	timestamp := time.Now().Format("2006-01-02 15:04:05")

	return fmt.Sprintf(`# Grok AI Analysis - %s

**Generated:** %s
**Ticker:** %s
**Tokens Used:** %d

## AI Analysis

%s

---
*Generated by Barracuda Options Analysis System with Grok AI*`,
		ticker, timestamp, ticker, response.Tokens, response.Content)
}
