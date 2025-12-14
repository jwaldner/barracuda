package logger

import (
	"io"
	"log"
	"os"
)

var (
	Info  *log.Logger
	Warn  *log.Logger
	Debug *log.Logger
	Error *log.Logger
)

func Init() error {
	// Open log file
	logFile, err := os.OpenFile("barracuda.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		return err
	}

	// Standard Go logging best practices with emojis
	Info = log.New(os.Stdout, "ℹ️  INFO: ", log.Ldate|log.Ltime)
	Warn = log.New(logFile, "⚠️  WARN: ", log.Ldate|log.Ltime|log.Lshortfile)
	Debug = log.New(logFile, "🐛 DEBUG: ", log.Ldate|log.Ltime|log.Lshortfile)
	Error = log.New(io.MultiWriter(os.Stderr, logFile), "❌ ERROR: ", log.Ldate|log.Ltime|log.Lshortfile)
	
	return nil
}