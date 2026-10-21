package store

import (
	"strings"
	"testing"
)

func TestSafeLogTextCredentials(t *testing.T) {
	for _, message := range []string{
		`{"authorization":"Bearer secret-value"}`,
		`{"password":"secret-value with spaces"}`,
		`password='secret-value with spaces'`,
		"Authorization: Basic secret-value",
		"https://user:secret-value@example.com/path",
		`{"nested":{"api_key":"secret-value"}}`,
	} {
		t.Run(message, func(t *testing.T) {
			if result := SafeLogText(message); strings.Contains(result, "secret-value") {
				t.Errorf("credential survived redaction: %s", result)
			}
		})
	}
	if result := SafeLogText("perplexity: PERPLEXITY_API_KEY is required"); result != "perplexity: PERPLEXITY_API_KEY is required" {
		t.Fatal("redaction removed the actionable error")
	}
	if result := SafeLogText(strings.Repeat("x", 10000)); len(result) > 8192 || !strings.Contains(result, "[truncated]") {
		t.Fatal("long messages must be bounded and marked")
	}
}

func TestSafeLogDetailsKeepsTheSFUKeys(t *testing.T) {
	details := safeLogDetails(map[string]any{
		"sfu_edge": "sfu-a1",
		"sfu_host": "sfu-a1.example.com",
		"token":    "secret-value",
	})
	if len(details) != 2 || details["sfu_edge"] != "sfu-a1" || details["sfu_host"] != "sfu-a1.example.com" {
		t.Fatalf("details = %v, want only the two SFU keys", details)
	}
}
