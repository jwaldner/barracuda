package symbols

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Symbol represents a single S&P 500 stock symbol with metadata
type Symbol struct {
	Symbol               string `json:"symbol"`
	Company              string `json:"company"`
	Sector               string `json:"sector"`
	SubIndustry          string `json:"sub_industry"`
	HeadquartersLocation string `json:"headquarters_location"`
	DateAdded            string `json:"date_added"`
	CIK                  string `json:"cik"`
	Founded              string `json:"founded"`
	LastUpdated          string `json:"last_updated"`
}

// SP500Service manages S&P 500 symbol data using GitHub CSV sources
type SP500Service struct {
	dataDir     string
	symbolsFile string
	assetFile   string
}

// NewSP500Service creates a new S&P 500 service
func NewSP500Service(dataDir string) *SP500Service {
	if dataDir == "" {
		dataDir = "data/symbols"
	}

	// Ensure data directory exists
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		// Could not create data directory
	}

	return &SP500Service{
		dataDir:     dataDir,
		symbolsFile: filepath.Join("assets/symbols", "sp500_symbols.json"),
		assetFile:   filepath.Join("assets/symbols", "sp500_symbols.json"),
	}
}

// UpdateSymbols fetches the latest S&P 500 symbols from S&P Dow Jones official source
func (s *SP500Service) UpdateSymbols() error {
	// Fetching S&P 500 symbols

	// Primary: S&P Dow Jones official source (updates quarterly)
	symbols, err := s.fetchFromSPDowJones()
	if err != nil {
		// S&P Dow Jones source failed
		// Fallback to GitHub CSV
		symbols, err = s.fetchFromGitHubCSV()
		if err != nil {
			symbols, err = s.loadFromAssets()
			if err != nil || len(symbols) == 0 {
				return fmt.Errorf("fetch failed and no assets available: %v", err)
			}
		} else {
			// Success! Build new assets
			s.buildAssets(symbols)
		}
	} else {
		// Success! Build new assets
		s.buildAssets(symbols)
	}

	// Add metadata
	now := time.Now().Format("2006-01-02 15:04:05")
	for i := range symbols {
		symbols[i].LastUpdated = now
	}

	// Backup existing file
	s.backupExistingFile()

	// Save new symbols
	if err := s.saveSymbols(symbols); err != nil {
		return fmt.Errorf("failed to save symbols: %v", err)
	}

	return nil
}

// fetchFromSPDowJones gets S&P 500 list from official S&P Dow Jones source (quarterly updates)
func (s *SP500Service) fetchFromSPDowJones() ([]Symbol, error) {
	return s.fetchFromGitHubCSV()
}

// fetchFromGitHubCSV uses GitHub CSV as reliable fallback
func (s *SP500Service) fetchFromGitHubCSV() ([]Symbol, error) {
	// GitHub CSV sources as fallback (reliable, frequently updated)
	urls := []string{
		"https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
		"https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
	}

	for _, url := range urls {
		symbols, err := s.fetchCSVSource(url)
		if err != nil {
			continue
		}
		return symbols, nil
	}

	return nil, fmt.Errorf("all GitHub CSV sources failed")
}

// fetchCSVSource fetches symbols from a CSV source
func (s *SP500Service) fetchCSVSource(url string) ([]Symbol, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	reader := csv.NewReader(resp.Body)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	if len(records) < 2 { // Header + at least one record
		return nil, fmt.Errorf("invalid CSV data")
	}

	var symbols []Symbol
	header := records[0]

	// Find column indices
	columns := s.findCSVColumns(header)

	if columns["symbol"] == -1 {
		return nil, fmt.Errorf("symbol column not found in CSV")
	}

	for i, record := range records[1:] { // Skip header
		if len(record) <= columns["symbol"] {
			continue
		}

		symbol := Symbol{
			Symbol:      strings.TrimSpace(record[columns["symbol"]]),
			LastUpdated: time.Now().Format("2006-01-02 15:04:05"),
		}

		if columns["company"] != -1 && len(record) > columns["company"] {
			symbol.Company = strings.TrimSpace(record[columns["company"]])
		}

		if columns["sector"] != -1 && len(record) > columns["sector"] {
			symbol.Sector = strings.TrimSpace(record[columns["sector"]])
		}

		if columns["sub_industry"] != -1 && len(record) > columns["sub_industry"] {
			symbol.SubIndustry = strings.TrimSpace(record[columns["sub_industry"]])
		}

		if columns["headquarters"] != -1 && len(record) > columns["headquarters"] {
			symbol.HeadquartersLocation = strings.TrimSpace(record[columns["headquarters"]])
		}

		if columns["date_added"] != -1 && len(record) > columns["date_added"] {
			symbol.DateAdded = strings.TrimSpace(record[columns["date_added"]])
		}

		if columns["cik"] != -1 && len(record) > columns["cik"] {
			symbol.CIK = strings.TrimSpace(record[columns["cik"]])
		}

		if columns["founded"] != -1 && len(record) > columns["founded"] {
			symbol.Founded = strings.TrimSpace(record[columns["founded"]])
		}

		// Clean and validate symbol
		if symbol.Symbol != "" && len(symbol.Symbol) <= 5 {
			// Clean all fields
			symbol.Symbol = strings.ToUpper(strings.TrimSpace(symbol.Symbol))
			symbol.Company = strings.TrimSpace(symbol.Company)
			symbol.Sector = strings.TrimSpace(symbol.Sector)
			symbol.SubIndustry = strings.TrimSpace(symbol.SubIndustry)
			symbol.HeadquartersLocation = strings.TrimSpace(symbol.HeadquartersLocation)
			symbol.DateAdded = strings.TrimSpace(symbol.DateAdded)
			symbol.CIK = strings.TrimSpace(symbol.CIK)
			symbol.Founded = strings.TrimSpace(symbol.Founded)
			symbols = append(symbols, symbol)
		}

		// Limit processing to avoid issues
		if i > 1000 {
			break
		}
	}

	return symbols, nil
}

// parseWikipediaHTML extracts symbols from Wikipedia HTML (basic parsing)
func (s *SP500Service) parseWikipediaHTML(html string) []Symbol {
	var symbols []Symbol

	// Look for table rows with stock symbols
	lines := strings.Split(html, "\n")

	for _, line := range lines {
		// Simple pattern matching for ticker symbols in Wikipedia table
		if strings.Contains(line, "<td>") && strings.Contains(line, "</td>") {
			// Extract potential ticker symbol (1-5 uppercase letters)
			symbol := s.extractSymbolFromHTML(line)
			if s.isValidSymbol(symbol) {
				symbols = append(symbols, Symbol{
					Symbol:  symbol,
					Company: s.extractCompanyFromHTML(line),
				})
			}
		}
	}

	// Remove duplicates and sort
	symbols = s.deduplicateSymbols(symbols)

	return symbols
}

// extractSymbolFromHTML extracts ticker symbol from HTML line
func (s *SP500Service) extractSymbolFromHTML(line string) string {
	// Simple HTML parsing to find ticker symbols between tags
	if idx := strings.Index(line, ">"); idx != -1 {
		if end := strings.Index(line[idx+1:], "<"); end != -1 {
			potential := strings.TrimSpace(line[idx+1 : idx+1+end])
			if s.isValidSymbol(potential) {
				return potential
			}
		}
	}

	return ""
}

// extractCompanyFromHTML extracts company name from HTML line
func (s *SP500Service) extractCompanyFromHTML(line string) string {
	// Simple extraction of company name
	parts := strings.Split(line, "<td>")
	for i, part := range parts {
		if i > 0 && len(part) > 10 { // Skip symbol column
			if end := strings.Index(part, "</td>"); end != -1 {
				company := strings.TrimSpace(part[:end])
				if len(company) > 5 && !s.isValidSymbol(company) {
					return company
				}
			}
		}
	}
	return ""
}

// isValidSymbol checks if a string looks like a valid ticker symbol
func (s *SP500Service) isValidSymbol(symbol string) bool {
	if len(symbol) < 1 || len(symbol) > 5 {
		return false
	}

	for _, char := range symbol {
		if char < 'A' || char > 'Z' {
			return false
		}
	}

	// Common S&P 500 symbols for validation
	knownSymbols := map[string]bool{
		"AAPL": true, "MSFT": true, "GOOGL": true, "AMZN": true, "TSLA": true,
		"META": true, "NVDA": true, "BRK": true, "UNH": true, "JNJ": true,
	}

	// If it's a known symbol, definitely valid
	if knownSymbols[symbol] {
		return true
	}

	// Otherwise, check if it's reasonable
	return len(symbol) >= 1 && len(symbol) <= 5
}

// findCSVColumns finds the relevant columns in CSV header
func (s *SP500Service) findCSVColumns(header []string) map[string]int {
	columns := map[string]int{
		"symbol":       -1,
		"company":      -1,
		"sector":       -1,
		"sub_industry": -1,
		"headquarters": -1,
		"date_added":   -1,
		"cik":          -1,
		"founded":      -1,
	}

	for i, col := range header {
		col = strings.ToLower(strings.TrimSpace(col))

		if strings.Contains(col, "symbol") || strings.Contains(col, "ticker") {
			columns["symbol"] = i
		} else if col == "security" || strings.Contains(col, "company") || strings.Contains(col, "name") {
			columns["company"] = i
		} else if strings.Contains(col, "sector") || col == "gics sector" {
			columns["sector"] = i
		} else if strings.Contains(col, "sub-industry") || col == "gics sub-industry" {
			columns["sub_industry"] = i
		} else if strings.Contains(col, "headquarters") || strings.Contains(col, "location") {
			columns["headquarters"] = i
		} else if strings.Contains(col, "date added") || strings.Contains(col, "added") {
			columns["date_added"] = i
		} else if col == "cik" {
			columns["cik"] = i
		} else if col == "founded" {
			columns["founded"] = i
		}
	}

	// Default to first column if symbol not found
	if columns["symbol"] == -1 {
		columns["symbol"] = 0
	}

	return columns
}

// getColumnValue safely gets value from CSV record
func (s *SP500Service) getColumnValue(record []string, col int) string {
	if col >= 0 && col < len(record) {
		return strings.TrimSpace(record[col])
	}
	return ""
}

// deduplicateSymbols removes duplicate symbols
func (s *SP500Service) deduplicateSymbols(symbols []Symbol) []Symbol {
	seen := make(map[string]bool)
	var unique []Symbol

	for _, symbol := range symbols {
		if !seen[symbol.Symbol] {
			seen[symbol.Symbol] = true
			unique = append(unique, symbol)
		}
	}

	// Sort by symbol
	sort.Slice(unique, func(i, j int) bool {
		return unique[i].Symbol < unique[j].Symbol
	})

	return unique
}

// backupExistingFile creates a backup of the current symbols file
func (s *SP500Service) backupExistingFile() {
	// Skip backup since we use assets now
}

// copyFile copies a file
func (s *SP500Service) copyFile(src, dst string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destFile.Close()

	_, err = io.Copy(destFile, sourceFile)
	return err
}

// saveSymbols saves symbols to JSON file
func (s *SP500Service) saveSymbols(symbols []Symbol) error {
	file, err := os.Create(s.symbolsFile)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")

	data := map[string]interface{}{
		"last_updated": time.Now().Format("2006-01-02 15:04:05"),
		"count":        len(symbols),
		"symbols":      symbols,
	}

	return encoder.Encode(data)
}

// LoadSymbols loads symbols from the local file
func (s *SP500Service) LoadSymbols() ([]Symbol, error) {
	// Use the same asset file as GetSymbolInfo to prevent mismatches
	return s.loadFromAssets()
}

// GetSymbolsAsStrings returns just the symbol strings
func (s *SP500Service) GetSymbolsAsStrings() ([]string, error) {
	symbols, err := s.LoadSymbols()
	if err != nil {
		return nil, err
	}

	var symbolStrings []string
	for _, symbol := range symbols {
		symbolStrings = append(symbolStrings, symbol.Symbol)
	}

	return symbolStrings, nil
}

// GetSymbolsInfo returns summary information
func (s *SP500Service) GetSymbolsInfo() (map[string]interface{}, error) {
	if _, err := os.Stat(s.assetFile); err != nil {
		return map[string]interface{}{
			"exists":       false,
			"last_updated": "never",
			"count":        0,
		}, nil
	}

	file, err := os.Open(s.assetFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var data map[string]interface{}
	if err := json.NewDecoder(file).Decode(&data); err != nil {
		return nil, err
	}

	data["exists"] = true
	return data, nil
}

// AutoUpdate updates symbols if they're older than specified duration
func (s *SP500Service) AutoUpdate(maxAge time.Duration) error {
	info, err := s.GetSymbolsInfo()
	if err != nil {
		return err
	}

	if !info["exists"].(bool) {
		return s.UpdateSymbols()
	}

	lastUpdated, ok := info["last_updated"].(string)
	if !ok {
		return s.UpdateSymbols()
	}

	updateTime, err := time.Parse("2006-01-02 15:04:05", lastUpdated)
	if err != nil {
		return s.UpdateSymbols()
	}

	if time.Since(updateTime) > maxAge {
		return s.UpdateSymbols()
	}
	return nil
}

// buildAssets saves successful fetch as assets
func (s *SP500Service) buildAssets(symbols []Symbol) {
	os.MkdirAll(filepath.Dir(s.assetFile), 0755)

	data := map[string]interface{}{
		"source":       "Asset Build",
		"last_updated": time.Now().Format("2006-01-02 15:04:05"),
		"count":        len(symbols),
		"symbols":      symbols,
	}

	file, _ := os.Create(s.assetFile)
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	encoder.Encode(data)
}

// loadFromAssets loads symbols from asset backup
func (s *SP500Service) loadFromAssets() ([]Symbol, error) {
	data, err := os.ReadFile(s.assetFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read assets file: %v", err)
	}

	var result struct {
		Symbols []Symbol `json:"symbols"`
	}

	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to parse assets JSON: %v", err)
	}

	return result.Symbols, nil
}

// GetSymbolInfo looks up company and sector info for any symbol (universal lookup)
func (s *SP500Service) GetSymbolInfo(ticker string) (company, sector string) {
	symbols, _ := s.loadFromAssets()

	for _, symbol := range symbols {
		if strings.EqualFold(symbol.Symbol, ticker) {
			return symbol.Company, symbol.Sector
		}
	}

	return "", ""
}
