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

// AnalyzeOptions calls the Grok API to analyze options data with enhanced progress tracking
func (c *Client) AnalyzeOptions(auditData string) (*AnalysisResponse, error) {
	logger.Warn.Printf("🤖 GROK: Starting analysis (audit data: %d bytes)", len(auditData))

	// Optimize audit data size if too large (keep only essential parts)
	optimizedData := auditData
	if len(auditData) > 50000 { // If larger than 50KB, truncate
		logger.Warn.Printf("🤖 GROK: Large audit data detected (%d bytes), truncating for API efficiency", len(auditData))
		optimizedData = auditData[:50000] + "\n... [truncated for API efficiency]"
	}

	// Track start time for performance monitoring
	startTime := time.Now()

	// Construct a concise, cost-effective prompt with audit data
	prompt := fmt.Sprintf(`You are a financial analysis AI. Analyze this options trading data and provide insights.

Focus on:
1. Black-Scholes calculations validation
2. Greeks analysis (Delta, Gamma, Theta, Vega, Rho)
3. Risk assessment and recommendations
4. Market conditions and volatility analysis

Options Data:
%s

Be concise and focus on numerical validation.`, optimizedData)

	// Prepare Grok API request with cost-optimized parameters
	requestBody := map[string]interface{}{
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": prompt,
			},
		},
		"model":       c.model,
		"stream":      false,
		"temperature": 0.1,  // More focused, faster responses
		"max_tokens":  2000, // Reduced from 4000 to limit costs
		"top_p":       0.9,  // Add top_p for more efficient generation
	}

	reqBytes, err := json.Marshal(requestBody)
	if err != nil {
		logger.Warn.Printf("⚠️🤖 GROK WARNING: Failed to marshal request body: %v", err)
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	logger.Warn.Printf("🤖 GROK: Sending API request to X.AI (payload: %d bytes, max_tokens: 4000)", len(reqBytes))

	// Create context with extended timeout for Grok's potentially long processing time
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute) // Increased to 15 minutes
	defer cancel()

	// Make HTTP request to Grok API with proper context
	req, err := http.NewRequestWithContext(ctx, "POST", c.endpoint, bytes.NewBuffer(reqBytes))
	if err != nil {
		logger.Warn.Printf("⚠️🤖 GROK WARNING: Failed to create HTTP request: %v", err)
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("User-Agent", "barracuda-options-analyzer/1.0")

	// Progress tracking with periodic updates
	logger.Warn.Printf("🤖 GROK: Request sent, waiting for response...")

	// Start progress monitoring in a goroutine
	progressDone := make(chan bool)
	go func() {
		// Use time.Ticker for progress tracking
		ticker := time.NewTicker(1 * time.Minute)
		defer ticker.Stop()

		for {
			select {
			case <-progressDone:
				return
			case <-ticker.C:
				elapsed := time.Since(startTime)
				logger.Warn.Printf("🤖 GROK: Still waiting for response... elapsed: %v (timeout in %v)", elapsed, 15*time.Minute-elapsed)
			}
		}
	}()

	resp, err := c.client.Do(req)

	// Stop progress monitoring
	close(progressDone)
	if err != nil {
		duration := time.Since(startTime)
		// Enhanced error logging for network diagnosis
		if ctx.Err() == context.DeadlineExceeded {
			logger.Warn.Printf("⚠️🤖 GROK WARNING: Request timeout after %v - API may be experiencing high load", duration)
			return nil, fmt.Errorf("request timed out after %v - Grok may be experiencing high load", duration)
		}

		logger.Warn.Printf("⚠️🤖 GROK WARNING: HTTP request failed after %v - Endpoint: %s", duration, c.endpoint)
		logger.Warn.Printf("⚠️🤖 GROK WARNING: Error details: %v", err)
		logger.Warn.Printf("⚠️🤖 GROK WARNING: This may indicate network connectivity issues or API service problems")
		return nil, fmt.Errorf("API request failed: %v", err)
	}
	defer resp.Body.Close()

	duration := time.Since(startTime)
	logger.Warn.Printf("🤖 GROK: Received response after %v - Status: %d %s", duration, resp.StatusCode, resp.Status)

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		logger.Warn.Printf("⚠️🤖 GROK WARNING: API error response (Status %d) after %v: %s", resp.StatusCode, duration, string(body))

		// Enhanced error messages based on status codes
		switch resp.StatusCode {
		case 401:
			return nil, fmt.Errorf("authentication failed - check API key (Status %d)", resp.StatusCode)
		case 429:
			return nil, fmt.Errorf("rate limit exceeded - please try again later (Status %d)", resp.StatusCode)
		case 500, 502, 503, 504:
			return nil, fmt.Errorf("Grok service temporarily unavailable - try again in a few minutes (Status %d)", resp.StatusCode)
		default:
			return nil, fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))
		}
	}

	// Parse response with enhanced error handling
	var grokResponse struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			TotalTokens      int `json:"total_tokens"`
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&grokResponse); err != nil {
		logger.Warn.Printf("⚠️🤖 GROK WARNING: Failed to decode API response after %v: %v", duration, err)
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}

	// Check for API-level errors in response
	if grokResponse.Error.Message != "" {
		logger.Warn.Printf("⚠️🤖 GROK WARNING: API returned error after %v: %s (Type: %s)", duration, grokResponse.Error.Message, grokResponse.Error.Type)
		return nil, fmt.Errorf("Grok API error: %s", grokResponse.Error.Message)
	}

	if len(grokResponse.Choices) == 0 {
		logger.Warn.Printf("⚠️🤖 GROK WARNING: API returned empty choices array after %v", duration)
		return nil, fmt.Errorf("no response from Grok API - empty choices")
	}

	analysis := grokResponse.Choices[0].Message.Content
	tokens := grokResponse.Usage.TotalTokens

	// Validate response quality
	if len(analysis) < 100 {
		logger.Warn.Printf("⚠️🤖 GROK WARNING: Suspiciously short analysis response (%d chars) after %v", len(analysis), duration)
	}

	logger.Warn.Printf("🤖 GROK: Analysis completed successfully in %v - %d tokens used (prompt: %d, completion: %d), response: %d chars",
		duration, tokens, grokResponse.Usage.PromptTokens, grokResponse.Usage.CompletionTokens, len(analysis))

	return &AnalysisResponse{
		Content: analysis,
		Tokens:  tokens,
	}, nil
}

// FormatAnalysis formats the Grok analysis as markdown
func FormatAnalysis(response *AnalysisResponse) string {
	timestamp := time.Now().Format("2006-01-02 15:04:05")

	return fmt.Sprintf(`# Grok AI Analysis

**Generated:** %s

## Analysis

%s

---
*Tokens used: %d | Generated at %s*
`,
		timestamp, response.Content, response.Tokens, timestamp)
}
