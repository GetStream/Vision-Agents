package store

import (
	"errors"
	"log/slog"
	"regexp"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestMaskPhoneKeepsTheCountryCodeAndTheLastTwoDigits(t *testing.T) {
	for number, want := range map[string]string{
		"+15550001101":   "+1•••01",
		"+442079460018":  "+44•••18",
		"+380441234567":  "+380•••67",
		"%2B15550001101": "+1•••01",
		"5550001101":     "•••01",
		"+99912345601":   "+•••01",
	} {
		require.Equal(t, want, maskPhone(number), number)
	}
}

func TestRedactLogAttrMasksANumberInEveryKindOfValue(t *testing.T) {
	for name, line := range map[string]string{
		"string attr": logged(t, true, "called", "to", "+15550001101"),
		"message":     logged(t, true, "calling +15550001101"),
		"error":       logged(t, true, "failed", "error", errors.New("telnyx: +15550001101 is not reachable")),
		"map":         logged(t, true, "failed", "details", map[string]any{"to": "+15550001101"}),
		"group":       logged(t, true, "failed", slog.Group("call", "to", "+15550001101")),
		"spaces":      logged(t, true, "calling +1 555 000 1101"),
		"brackets":    logged(t, true, "calling +1 (555) 000-1101"),
		"url":         logged(t, true, "served", "request", "GET /v1/x?to=%2B15550001101"),
		"lower url":   logged(t, true, "served", "request", "GET /v1/x?to=%2b15550001101"),
	} {
		require.Contains(t, line, "+1•••01", name)
		for _, leaked := range []string{"5550001101", "555 000 1101", "000-1101"} {
			require.NotContains(t, line, leaked, name)
		}
	}
}

func TestRedactLogAttrLogsACallerAsAPhoneCaller(t *testing.T) {
	for _, line := range []string{
		logged(t, true, "joined", "participant", "sip-+15550001101"),
		logged(t, true, "joined", "participant", "sip-00442079460018"),
	} {
		require.Contains(t, line, "participant=phone_caller")
		require.NotContains(t, line, "sip-")
	}
	line := logged(t, true, "sip-+15550001101 left")
	require.Contains(t, line, `msg="phone_caller left"`)
	require.NotContains(t, line, "sip-")
}

func TestRedactLogAttrMasksANationalNumberUnderAPhoneAttribute(t *testing.T) {
	require.Contains(t, logged(t, true, "called", "caller", "5550001101"), "caller=•••01")
	require.Contains(t, logged(t, true, "called", "number", "5550001101"), "number=•••01")
	require.Contains(t, logged(t, true, "called", "caller", "555-0101"), "caller=•••01")
	require.Contains(t, logged(t, true, "called", "to", "(555) 000-1101"), "to=•••01")
	require.Contains(t, logged(t, true, "called", "to", "sip:5550001101@trunk.example"), "to=sip:•••01@trunk.example")
	require.Contains(t, logged(t, true, "called", "from", "tel:555-000-1101"), "from=tel:•••01")
	require.Contains(t, logged(t, true, "called", "to", "1.555.000.1101"), "to=•••01")
	require.Contains(t, logged(t, true, "called", "to", "5550001 10.0.0.2"), "to=•••02")
	require.Contains(t, logged(t, true, "called", "to", "5550001101 10.0.0.2"), "to=•••02")
	require.Contains(t, logged(t, true, "called", "to", "10.0.0.2 5550001"), "to=•••01")
}

func TestRedactLogAttrKeepsWhatIsNotANumber(t *testing.T) {
	for _, args := range [][]any{
		{"from", "queued", "to", "running"},
		{"count", 1234567},
		{"session_id", "6f1c2a9e-4b7d-4a8e-9c3f-0d2e1b5a7c44"},
		{"at", "2026-11-10T10:00:00+01:00"},
		{"from", "2026-11-10T10:00:00Z"},
		{"to", "10.20.30.40"},
		{"host", "sip-trunk.example"},
		{"user", "phone-router"},
		{"flag", "--sip-lookup-from"},
		{"from", "refusing/handing-over-1731234567"},
		{"from", "anthropic/claude-sonnet-4-20250514"},
		{"from", "/tmp/router-17312345/config.yaml"},
		{"from", "20261007200000_add_index.sql"},
		{"to", "192.168.1.1 10.0.0.2"},
		{"to", "10.0.0.1 10.0.0.2 10.0.0.3"},
		{"host", "sip-1.example.com"},
		{"host", "sip-01.trunk.example"},
		{"status", "sip-2 trunk up"},
		{"delta", "+0.00672757"},
		{"delta", "+123456"},
		{"to", "10.0.0.1:50601"},
		{"to", "sip:12345@10.0.0.1"},
		{"customer_id", "1732479"},
		{"app", "1257545"},
		{"to", "2026-11-10"},
		{"from", "2026-11-10 12:34:56"},
		{"to", "1234567890123456"},
	} {
		require.Equal(t, logged(t, false, "x", args...), logged(t, true, "x", args...), args)
	}
}

func TestRedactLogAttrMasksTheNumberInAnID(t *testing.T) {
	require.Contains(t, logged(t, true, "x", "call", "default:phone-+15550001101"), "call=default:phone-+1•••01")
}

func TestRedactJSONLogAttrMasksANumber(t *testing.T) {
	line := loggedJSON(t, true, "x", "to", "+15550001101", "error", errors.New("no route to +15550001101"))
	require.Contains(t, line, `"to":"+1•••01"`)
	require.Contains(t, line, `"error":"no route to +1•••01"`)
	require.NotContains(t, line, "5550001101")
	require.Contains(t, loggedJSON(t, true, "x", "participant", "sip-+15550001101"), `"participant":"phone_caller"`)
}

func TestANewLogMaskMasksEveryLineWithNoChangeToTheCall(t *testing.T) {
	original := logMasks
	t.Cleanup(func() { logMasks = original })
	logMasks = append(logMasks[:len(logMasks):len(logMasks)], logMask{
		may:     func(s string) bool { return strings.Contains(s, "@") },
		pattern: regexp.MustCompile(`[a-z0-9.]+@[a-z0-9.]+\.[a-z]+`),
		mask:    func(string) string { return "[email]" },
	})
	line := logged(t, true, "wrote to ali@example.com", "contact", "ali@example.com")
	require.Equal(t, 2, strings.Count(line, "[email]"))
	require.NotContains(t, line, "ali@example.com")

	logMasks = original
	require.Contains(t, logged(t, true, "wrote to ali@example.com", "contact", "ali@example.com"), "ali@example.com")
}
