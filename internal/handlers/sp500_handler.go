package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/jwaldner/barracuda/internal/logger"
	"github.com/jwaldner/barracuda/internal/symbols"
)

// SP500Handler handles S&P 500 symbol management endpoints
type SP500Handler struct {
	symbolService *symbols.SP500Service
}

// NewSP500Handler creates a new S&P 500 handler
func NewSP500Handler() *SP500Handler {
	return &SP500Handler{
		symbolService: symbols.NewSP500Service("assets/symbols"),
	}
}

// UpdateSymbolsHandler manually triggers symbol update
func (h *SP500Handler) UpdateSymbolsHandler(w http.ResponseWriter, r *http.Request) {
	logger.Info.Printf("📡 Manual S&P 500 symbol update requested")

	startTime := time.Now()
	err := h.symbolService.UpdateSymbols()
	duration := time.Since(startTime)

	if err != nil {
		logger.Error.Printf("❌ Symbol update failed: %v", err)
		http.Error(w, fmt.Sprintf("Update failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Get updated info
	info, _ := h.symbolService.GetSymbolsInfo()

	response := map[string]interface{}{
		"status":          "success",
		"message":         "S&P 500 symbols updated successfully",
		"update_duration": duration.Milliseconds(),
		"timestamp":       time.Now().Unix(),
		"info":            info,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)

	logger.Info.Printf("✅ Symbol update completed in %v", duration)
}

// GetSymbolsHandler returns current S&P 500 symbols
func (h *SP500Handler) GetSymbolsHandler(w http.ResponseWriter, r *http.Request) {
	// Auto-update if symbols are older than 7 days
	if err := h.symbolService.AutoUpdate(7 * 24 * time.Hour); err != nil {
		logger.Warn.Printf("⚠️ Auto-update failed: %v", err)
	}

	symbols, err := h.symbolService.LoadSymbols()
	if err != nil {
		http.Error(w, fmt.Sprintf("Could not load symbols: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"symbols":   symbols,
		"count":     len(symbols),
		"timestamp": time.Now().Unix(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GetSymbolListHandler returns just the ticker symbols as a simple list
func (h *SP500Handler) GetSymbolListHandler(w http.ResponseWriter, r *http.Request) {
	// Auto-update if needed
	if err := h.symbolService.AutoUpdate(7 * 24 * time.Hour); err != nil {
		logger.Warn.Printf("⚠️ Auto-update failed: %v", err)
	}

	tickers, err := h.symbolService.GetSymbolsAsStrings()
	if err != nil {
		http.Error(w, fmt.Sprintf("Could not load symbols: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"symbols":   tickers,
		"count":     len(tickers),
		"timestamp": time.Now().Unix(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GetSymbolsInfoHandler returns metadata about symbols
func (h *SP500Handler) GetSymbolsInfoHandler(w http.ResponseWriter, r *http.Request) {
	info, err := h.symbolService.GetSymbolsInfo()
	if err != nil {
		http.Error(w, fmt.Sprintf("Could not get info: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(info)
}

// AutoUpdateSymbols updates symbols if they're older than maxAge (for startup initialization)
func (h *SP500Handler) AutoUpdateSymbols(maxAge time.Duration) error {
	return h.symbolService.AutoUpdate(maxAge)
}

// GetTop25Handler returns top 25 S&P 500 stocks for analysis
func (h *SP500Handler) GetTop25Handler(w http.ResponseWriter, r *http.Request) {
	// Get first 25 S&P 500 symbols as a basic implementation
	symbols, err := h.symbolService.GetSymbolsAsStrings()
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to get symbols: %v", err), http.StatusInternalServerError)
		return
	}

	// Return first 25 symbols
	top25 := symbols[:25]
	if len(symbols) < 25 {
		top25 = symbols
	}

	response := map[string]interface{}{
		"status":  "success",
		"symbols": top25,
		"count":   len(top25),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
