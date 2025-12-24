package test_audit

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestAuditFilenames(t *testing.T) {
	// Create persistent test directory
	testDir := "./test_files"
	os.RemoveAll(testDir)
	os.MkdirAll(testDir, 0755)

	ch := make(chan AuditAction, 10)
	go testAuditWorker(ch, testDir)

	// Test TSLA audit
	ch <- AuditAction{Type: "rotate", Ticker: "TSLA"}
	ch <- AuditAction{Type: "append", Data: map[string]interface{}{"price": 420.69}}
	ch <- AuditAction{Type: "analyze"}
	time.Sleep(50 * time.Millisecond)

	// Test AAPL audit
	ch <- AuditAction{Type: "rotate", Ticker: "AAPL"}
	ch <- AuditAction{Type: "append", Data: map[string]interface{}{"price": 150.00}}
	ch <- AuditAction{Type: "analyze"}
	time.Sleep(50 * time.Millisecond)

	close(ch)

	// Check files
	auditsDir := filepath.Join(testDir, "audits")
	files, err := os.ReadDir(auditsDir)
	if err != nil {
		t.Fatal("No audits directory created")
	}

	t.Logf("📁 Files in %s:", auditsDir)
	for _, file := range files {
		t.Logf("  %s", file.Name())
	}

	// Keep files for inspection - don't clean up
	t.Logf("🔍 Files kept in %s for inspection", testDir)
}
