package logger

import (
	"io"
	"log"
	"os"
)

var (
	Info    *log.Logger
	Warn    *log.Logger
	Debug   *log.Logger
	Verbose *log.Logger
	Error   *log.Logger
	Always  *log.Logger // Always logs to file regardless of log level

	// Current log level for filtering
	currentLogLevel string
)

func Init() error {
	return InitWithLevel("info")
}

func InitWithLevel(logLevel string) error {
	return InitWithConfig(logLevel, "barracuda.log")
}

func InitWithConfig(logLevel, logFilePath string) error {
	currentLogLevel = logLevel

	// Open log file
	logFile, err := os.OpenFile(logFilePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		return err
	}

	// Create null writer for disabled log levels
	nullWriter := io.Discard

	// Standard Go logging best practices with emojis
	Info = log.New(getWriter("info", logFile, nullWriter), "ℹ️  INFO: ", log.Ldate|log.Ltime)
	Warn = log.New(getWriter("warn", logFile, nullWriter), "⚠️  WARN: ", log.Ldate|log.Ltime|log.Lshortfile)
	Debug = log.New(getWriter("debug", logFile, nullWriter), "🐛 DEBUG: ", log.Ldate|log.Ltime|log.Lshortfile)
	Verbose = log.New(getWriter("verbose", logFile, nullWriter), "🔍 VERBOSE: ", log.Ldate|log.Ltime|log.Lshortfile)
	Error = log.New(io.MultiWriter(os.Stderr, logFile), "❌ ERROR: ", log.Ldate|log.Ltime|log.Lshortfile)
	Always = log.New(logFile, "📝 ALWAYS: ", log.Ldate|log.Ltime) // Always logs to file, bypasses level filtering

	return nil
}

// getWriter returns the appropriate writer based on log level
func getWriter(level string, activeWriter, disabledWriter io.Writer) io.Writer {
	if shouldLog(level) {
		return activeWriter
	}
	return disabledWriter
}

// shouldLog determines if a log level should be active
func shouldLog(level string) bool {
	levels := map[string]int{
		"error":   0,
		"warn":    1,
		"info":    2,
		"debug":   3,
		"verbose": 4,
	}

	currentLevel, exists := levels[currentLogLevel]
	if !exists {
		currentLevel = 2 // default to info
	}

	requiredLevel, exists := levels[level]
	if !exists {
		return false
	}

	return currentLevel >= requiredLevel
}
