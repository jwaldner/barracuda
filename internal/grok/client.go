package grok

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/jwaldner/barracuda/internal/config"
)

type Client struct {
	client   *http.Client
	endpoint string
	apiKey   string
	model    string
}

type AnalysisResponse struct {
	Content string `json:"content"`
	Tokens  int    `json:"tokens"`
}

func NewClient() *Client {
	cfg := config.Load()

	return &Client{
		client: &http.Client{
			Timeout: time.Duration(cfg.GrokTimeoutMinutes) * time.Minute,
		},
		endpoint: cfg.GrokEndpoint,
		apiKey:   cfg.GrokAPIKey,
		model:    cfg.GrokModel,
	}
}

// AnalyzeOptions calls the Grok API to analyze options data
func (c *Client) AnalyzeOptions(auditData, customPrompt string) (*AnalysisResponse, error) {
	// Use custom prompt if provided, otherwise use config prompt
	prompt := customPrompt
	if prompt == "" {
		auditConfig := config.GetAuditConfig()
		prompt = auditConfig.AIAnalysisPrompt
	}

	// Use full audit data - no truncation
	optimizedData := auditData

	// Get timeout from config
	cfg := config.Load()
	timeoutDuration := time.Duration(cfg.GrokTimeoutMinutes) * time.Minute

	// Construct prompt with audit data using selected prompt (custom or config)
	fullPrompt := fmt.Sprintf(`%s

Options Data:
%s`, prompt, optimizedData)

	// Prepare Grok API request
	requestBody := map[string]interface{}{
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": fullPrompt,
			},
		},
		"model":       c.model,
		"stream":      false,
		"temperature": 0.1,
		"max_tokens":  2000,
		"top_p":       0.9,
	}

	reqBytes, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	// Create context with configurable timeout
	ctx, cancel := context.WithTimeout(context.Background(), timeoutDuration)
	defer cancel()

	// Make HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", c.endpoint, bytes.NewBuffer(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("API request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
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
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}

	// Check for API-level errors in response
	if grokResponse.Error.Message != "" {
		return nil, fmt.Errorf("Grok API error: %s", grokResponse.Error.Message)
	}

	if len(grokResponse.Choices) == 0 {
		return nil, fmt.Errorf("no response from Grok API - empty choices")
	}

	analysis := grokResponse.Choices[0].Message.Content
	tokens := grokResponse.Usage.TotalTokens

	return &AnalysisResponse{
		Content: analysis,
		Tokens:  tokens,
	}, nil
}

// FormatAnalysisWithPrompt formats the analysis response as markdown with the custom prompt included
func FormatAnalysisWithPrompt(response *AnalysisResponse, customPrompt string) string {
	var markdown strings.Builder

	// Add custom prompt section if provided
	if customPrompt != "" {
		markdown.WriteString("## Custom Analysis Prompt\n\n")
		markdown.WriteString(fmt.Sprintf("```\n%s\n```\n\n", customPrompt))
	}

	// Add the AI analysis
	markdown.WriteString("## AI Analysis\n\n")
	markdown.WriteString(response.Content)
	markdown.WriteString("\n\n")

	// Add metadata
	markdown.WriteString("---\n")
	markdown.WriteString(fmt.Sprintf("*Analysis generated using %d tokens*\n", response.Tokens))
	markdown.WriteString(fmt.Sprintf("*Generated at: %s*\n", time.Now().Format("2006-01-02 15:04:05")))

	return markdown.String()
}
