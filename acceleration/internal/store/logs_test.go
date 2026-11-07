package store

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/getsentry/sentry-go"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
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

// A provider's key travels in a URL query or a URL-encoded header, and a Go http or websocket
// error prints that URL whole. The rows are the names we send a key in (key, token), the names
// credentialText took before this change (api_key, apikey, access_token, client_secret) and
// headers written into a URL.
func TestSafeLogTextQueryAndEncodedCredentials(t *testing.T) {
	secret := rand.Text()
	backslash := `\`
	for name, message := range map[string]string{
		"key after a json-escaped ampersand":         `{"url":"wss://live.example/ws?alt=sse` + backslash + `u0026key=` + secret + `"}`,
		"token after a json-escaped ampersand":       `{"url":"https://ipinfo.io/json?lang=en` + backslash + `u0026token=` + secret + `"}`,
		"key in a twice-encoded url":                 `redirect=https%253A%252F%252Flive.example%252Fws%253Fkey%253D` + secret,
		"key after a twice-encoded ampersand":        `redirect=https%253A%252F%252Flive.example%252Fws%253Fa%253D1%2526key%253D` + secret,
		"key after a json-escaped equals sign":       `{"url":"wss://live.example/ws?key` + backslash + `u003d` + secret + `"}`,
		"authorization behind a twice-encoded quote": `q=%257B%2522Authorization%2522%253A%2522Basic%2520` + secret + `%2522%257D`,
		"encoded authorization with an equals sign":  `h=authorization%3DBasic%20` + secret,
		"key after an html-escaped ampersand":        `<a href="https://live.example/ws?alt=sse&amp;key=` + secret + `">`,
		"twice-encoded bearer":                       `h=Bearer%2520` + secret,
		"twice-encoded authorization":                `h=Authorization%253A%2520Basic%2520` + secret,
		"key first (gemini stt and sts)":             `Get "wss://live.example/ws?key=` + secret + `": dial tcp: i/o timeout`,
		"key at the start of a json query":           `{"method":"GET","query":"key=` + secret + backslash + `u0026echo_status=404"}`,
		"token at the start of a quoted query":       `query="token=` + secret + `"`,
		"key after another parameter":                `wss://live.example/ws?alt=sse&key=` + secret + `&model=m`,
		"key in upper case":                          `wss://live.example/ws?KEY=` + secret,
		"key in an encoded url":                      `redirect=https%3A%2F%2Flive.example%2Fws%3Fkey%3D` + secret,
		"key after an encoded ampersand":             `redirect=https%3A%2F%2Flive.example%2Fws%3Fa%3D1%26key%3D` + secret,
		"token (ipinfo)":                             `Get "https://ipinfo.io/203.0.113.10/json?token=` + secret + `": EOF`,
		"token after another parameter":              `https://ipinfo.io/json?lang=en&token=` + secret,
		"api_key":                                    `https://api.example/v1?api_key=` + secret,
		"apikey":                                     `https://api.example/v1?apikey=` + secret,
		"access_token":                               `https://api.example/v1?access_token=` + secret,
		"client_secret":                              `https://auth.example/token?client_secret=` + secret,
		"encoded authorization bearer":               `h=Authorization%3A%20Bearer%20` + secret,
		"encoded authorization plus":                 `h=Authorization%3A+Bearer+` + secret,
		"encoded authorization basic":                `h=authorization%3A%20Basic%20` + secret,
		"encoded authorization json":                 `q=%7B%22Authorization%22%3A%22Basic%20` + secret + `%22%7D`,
		"encoded authorization quoted":               `q=%7B%27authorization%27%3A%27Basic%20` + secret + `%27%7D`,
		"encoded authorization param":                `redirect=https%3A%2F%2Fapi.example%2Fv1%3Fauthorization%3DBearer%2520` + secret,
		"encoded set-cookie":                         `h=Set-Cookie%3A%20session%3D` + secret,
		"encoded proxy-authorization":                `h=Proxy-Authorization%3A%20Basic%20` + secret,
		"encoded cookie":                             `h=Cookie%3A%20session%3D` + secret,
		"encoded bearer alone":                       `h=Bearer%20` + secret,
		"encoded bearer after a space":               `q=sent%20Bearer%20` + secret,
		"form-encoded bearer alone":                  `h=Bearer+` + secret,
	} {
		t.Run(name, func(t *testing.T) {
			if result := SafeLogText(message); strings.Contains(result, secret) {
				t.Errorf("credential survived redaction: %s", result)
			}
		})
	}
}

// The query rule names whole parameters: the parameters around a key stay, and parameters
// that only start with key or token, or prose that names them, are kept byte for byte.
func TestSafeLogTextKeepsTheRestOfTheURL(t *testing.T) {
	secret := rand.Text()
	backslash := `\`
	// The characters that end a redacted value: a space, #, ' and " each stop it, in the query
	// rule, the encoded authorization rule and the encoded bearer rule.
	for message, want := range map[string]string{
		`dial wss://live.example/ws?key=` + secret + ` failed: 401`:                  `dial wss://live.example/ws?key=[redacted] failed: 401`,
		`wss://live.example/ws?key=` + secret + `#top`:                               `wss://live.example/ws?key=[redacted]#top`,
		`url='wss://live.example/ws?key=` + secret + `' failed`:                      `url='wss://live.example/ws?key=[redacted]' failed`,
		`h=Authorization%3A%20Basic%20` + secret + ` failed: 401`:                    `h=[redacted] failed: 401`,
		`"h=Authorization%3A%20Basic%20` + secret + `" failed`:                       `"h=[redacted]" failed`,
		`h=Authorization%3A%20Basic%20` + secret + `#top`:                            `h=[redacted]#top`,
		`h=Bearer%20` + secret + ` failed: 401`:                                      `h=[redacted] failed: 401`,
		`h=Bearer%20` + secret + `#top`:                                              `h=[redacted]#top`,
		`{"url":"wss://live.example/ws?key=` + secret + backslash + `u0026model=m"}`: `{"url":"wss://live.example/ws?key=[redacted]` + backslash + `u0026model=m"}`,
		`h=Authorization%3A%20Basic%20` + secret + backslash + `u0026model=m`:        `h=[redacted]` + backslash + `u0026model=m`,
		`h=Bearer%20` + secret + backslash + `u0026model=m`:                          `h=[redacted]` + backslash + `u0026model=m`,
		`Get "wss://live.example/ws?alt=sse&key=` + secret + `&model=m": EOF`:        `Get "wss://live.example/ws?alt=sse&key=[redacted]&model=m": EOF`,
		`h=Authorization%3A%20Basic%20` + secret + `&model=m`:                        `h=[redacted]&model=m`,
		`h=Bearer%20` + secret + `&model=m`:                                          `h=[redacted]&model=m`,
		`{"query":"key=` + secret + backslash + `u0026echo_status=404"}`:             `{"query":"key=[redacted]` + backslash + `u0026echo_status=404"}`,
	} {
		if result := SafeLogText(message); result != want {
			t.Errorf("SafeLogText(%q) = %q, want %q", message, result, want)
		}
	}
	// The answers base 6b9627f99 gives for them, the same as the input; the bearer rule takes no
	// word after bearer, so the last one's "bearer of" stays too.
	for message, want := range map[string]string{
		`wss://stt.example/v1/listen?model=nova-3&keyterm=Stream&keyterms=Vision&keyterms_prompt=%5B%22Agents%22%5D&language_code=en`: `wss://stt.example/v1/listen?model=nova-3&keyterm=Stream&keyterms=Vision&keyterms_prompt=%5B%22Agents%22%5D&language_code=en`,
		`wss://stt.example/v1/listen?monkey=banana&keys=3&tokens=12&token_type_hint=access_token&max_tokens=256`:                      `wss://stt.example/v1/listen?monkey=banana&keys=3&tokens=12&token_type_hint=access_token&max_tokens=256`,
		`the key=value pair and token=12 in prose`:                                                        `the key=value pair and token=12 in prose`,
		`Get "https://api.example/v1/models?page=2&limit=50": dial tcp: lookup api.example: no such host`: `Get "https://api.example/v1/models?page=2&limit=50": dial tcp: lookup api.example: no such host`,
		`authorization%20failed for model gpt and the bearer of bad news`:                                 `authorization%20failed for model gpt and the bearer of bad news`,
	} {
		if result := SafeLogText(message); result != want {
			t.Errorf("SafeLogText(%q) = %q, want %q", message, result, want)
		}
	}
}

// Our own errors name a bearer token, a secret or a password in prose, and a WWW-Authenticate
// challenge starts with Bearer and an auth-param (RFC 6750 section 3): with no credential in
// them, they are kept byte for byte.
func TestRedactLogTextKeepsProseAndChallenges(t *testing.T) {
	for _, text := range []string{
		"bearer: the credential is not a bearer credential",
		"sandbox: the connection's scheme sends no bearer token",
		"auth: open secret: cipher: message authentication failed",
		"phone: open sip trunk password: cipher: message authentication failed",
		"store: rotate guest secret: context canceled",
		"store: set guest secret: EOF",
		"auth: open secret: ERROR #23505 duplicate key value",
		"apikey: the key is required",
		"needs_reauthorization: the provider no longer takes the credential",
		"usage: USAGE_INGEST_TOKEN is required: the bearer token producers and routers call with",
		"a Bearer Token. And a bearer or api_key connection",
		`Bearer error="insufficient_scope", scope="files:read files:write"`,
		`Bearer realm="", authorization_uri="https://x"`,
		`Bearer resource_metadata="https://resource.example.com/.well-known/oauth-protected-resource"`,
	} {
		if result := RedactLogText(text); result != text {
			t.Errorf("RedactLogText(%q) = %q, want it unchanged", text, result)
		}
	}
}

// A value that is not a word is a credential, after a word the rule passed over too.
func TestRedactLogTextTakesAValueThatIsNotAWord(t *testing.T) {
	secret := rand.Text()
	for text, want := range map[string]string{
		"Authorization: Bearer sk-test-" + secret:             "[redacted]",
		"sent Bearer " + secret + " and failed":               "sent [redacted] and failed",
		"sent bearer " + strings.ToLower(secret) + "0 failed": "sent [redacted] failed",
		"https://auth.example/token?client_secret=" + secret:  "https://auth.example/token?client_[redacted]",
		`{"password":"` + secret + `"}`:                       `{"[redacted]}`,
		`{"password":"word"}`:                                 `{"[redacted]}`,
		"password: hunter2 rejected":                          "[redacted] rejected",
		"secret: password: " + secret:                         "secret: [redacted]",
		"bearer of Bearer " + secret:                          "bearer of [redacted]",
		"Bearer private-token":                                "[redacted]",
		"Bearer ABCDEFGHI":                                    "[redacted]",
		`Bearer error="invalid_token", Bearer ` + secret:      `Bearer error="invalid_token", [redacted]`,
	} {
		if result := RedactLogText(text); result != want {
			t.Errorf("RedactLogText(%q) = %q, want %q", text, result, want)
		}
	}
}

// mayHoldCredential lets through every text a rule redacts: one row per rule and per word it
// needs, in any case, with the Kelvin sign and the long s the rules fold to k and s.
func TestEveryTextARuleRedactsPassesTheCheck(t *testing.T) {
	secret := rand.Text()
	for _, text := range []string{
		"sent BEARER " + secret,
		"h=bearer%20" + secret,
		"Proxy-Authorization: Basic " + secret,
		"Set-Cookie: session=" + secret,
		"h=Authorization%3A%20Basic%20" + secret,
		"h=COOKIE%3D" + secret,
		"x_api_key=" + secret,
		"ApiKey: " + secret,
		"refresh-token=" + secret,
		"ACCESS_TOKEN=" + secret,
		"client_SECRET=" + secret,
		`{"Password":"` + secret + `"}`,
		"wss://live.example/ws?KEY=" + secret,
		"https://ipinfo.io/json?Token=" + secret,
		"https://user:" + secret + "@example.com",
		"used sk-" + secret,
		"https://ipinfo.io/json?to\u212aen=" + secret,
		"pa\u017f\u017fword=" + secret,
	} {
		if redactCredentials(text) == text {
			t.Errorf("no rule redacts %q: the row tests nothing", text)
		}
		if !mayHoldCredential(text) {
			t.Errorf("the check skips %q, which a rule redacts", text)
		}
	}
	if mayHoldCredential(`llmrouter: post "https://api.example.com/v1/chat/completions": context deadline exceeded`) {
		t.Error("the check lets a line with none of the words through")
	}
}

// The size cap is SafeLogText's, not the redaction's: 8192 bytes are kept whole, 8193 are cut to
// 2048 runes and a marker, and RedactLogText never cuts.
func TestSafeLogTextCapsWhatItKeeps(t *testing.T) {
	secret := rand.Text()
	atCap, overCap := strings.Repeat("x", 8192), strings.Repeat("x", 8193)
	long := overCap + "?key=" + secret
	for name, got := range map[string][2]string{
		"8192 bytes are kept":      {SafeLogText(atCap), atCap},
		"8193 bytes are cut":       {SafeLogText(overCap), strings.Repeat("x", 2048) + "… [truncated]"},
		"runes are cut, not bytes": {SafeLogText(strings.Repeat("é", 5000)), strings.Repeat("é", 2048) + "… [truncated]"},
		"redact never cuts":        {RedactLogText(long), overCap + "?key=[redacted]"},
	} {
		if got[0] != got[1] {
			t.Errorf("%s: got %d bytes, want %d", name, len(got[0]), len(got[1]))
		}
	}
	if strings.Contains(SafeLogText(long), secret) {
		t.Error("the cap path kept the credential")
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

func TestSafeLogDetailsKeepsTheCallPlacedKeys(t *testing.T) {
	details := safeLogDetails(map[string]any{
		"stream_sip_host": "trunk-1.sip.example.com",
		"vendor":          "twilio",
		"vendor_call_id":  "twilio-call",
		"token":           "secret-value",
	})
	if len(details) != 3 || details["stream_sip_host"] != "trunk-1.sip.example.com" ||
		details["vendor"] != "twilio" || details["vendor_call_id"] != "twilio-call" {
		t.Fatalf("details = %v, want only the three call placed keys", details)
	}
}

func TestSafeLogDetailsKeepsTheBridgeMediaKeys(t *testing.T) {
	details := safeLogDetails(map[string]any{
		"stream_media_addr":   "203.0.113.10",
		"customer_media_addr": "198.51.100.7",
		"token":               "secret-value",
	})
	if len(details) != 2 || details["stream_media_addr"] != "203.0.113.10" || details["customer_media_addr"] != "198.51.100.7" {
		t.Fatalf("details = %v, want only the two media addresses", details)
	}
}

// logged is what a text handler with RedactLogAttr, or with no ReplaceAttr when redact is
// false, prints for one record at a pinned time, so two runs of it can be compared byte for
// byte.
func logged(t *testing.T, redact bool, message string, args ...any) string {
	t.Helper()
	var out bytes.Buffer
	options := &slog.HandlerOptions{}
	if redact {
		options.ReplaceAttr = RedactLogAttr
	}
	record := slog.NewRecord(time.Date(2026, 11, 6, 12, 0, 0, 0, time.UTC), slog.LevelError, message, 0)
	record.Add(args...)
	if err := slog.NewTextHandler(&out, options).Handle(context.Background(), record); err != nil {
		t.Fatal(err)
	}
	return out.String()
}

// echoedKey is a provider's 502 that echoes the Authorization header it was sent, and the key
// in it.
func echoedKey() (error, string) {
	key := rand.Text()
	return errors.New("502 Bad Gateway: upstream error: failed to connect to local service\n" +
		"request headers: Authorization: Bearer sk-test-" + key), key
}

func TestRedactLogAttrKeepsAnErrorsReasonAndNotTheKeyItEchoed(t *testing.T) {
	err, key := echoedKey()
	line := logged(t, true, "agent failure", "context", "llm", "error", fmt.Errorf("gemma: %w", err))
	if strings.Contains(line, key) {
		t.Fatalf("the key survived: %s", line)
	}
	want := `error="gemma: 502 Bad Gateway: upstream error: failed to connect to local service\nrequest headers: [redacted]"`
	if !strings.Contains(line, want) || !strings.Contains(line, "context=llm") {
		t.Fatalf("the line lost the rest of the error: %s", line)
	}
}

// A string attribute and the message are what a line holds when its caller logged err.Error()
// or formatted the error into the message.
func TestRedactLogAttrTakesTheKeyOutOfAStringAndTheMessage(t *testing.T) {
	err, key := echoedKey()
	line := logged(t, true, "refused: "+err.Error(), "reason", err.Error())
	if strings.Contains(line, key) {
		t.Fatalf("the key survived: %s", line)
	}
	if strings.Count(line, `failed to connect to local service\nrequest headers: [redacted]`) != 2 {
		t.Fatalf("the line lost the rest of the error: %s", line)
	}
}

func TestRedactLogAttrTakesTheKeyOutOfAGroup(t *testing.T) {
	err, key := echoedKey()
	line := logged(t, true, "agent failure", slog.Group("turn", "error", err))
	if strings.Contains(line, key) || !strings.Contains(line, `turn.error="502 Bad Gateway`) {
		t.Fatalf("the group kept the key or lost the error: %s", line)
	}
}

// An alert logs its details as a map, which can hold an error's text.
func TestRedactLogAttrTakesTheKeyOutOfAMapOfDetails(t *testing.T) {
	err, key := echoedKey()
	line := logged(t, true, "usage alert", "extra", map[string]any{"listing": err.Error()})
	if strings.Contains(line, key) || !strings.Contains(line, "listing:502 Bad Gateway") {
		t.Fatalf("the details kept the key or lost the error: %s", line)
	}
}

// The text handler prints a []byte as a string, so the key in a provider's body logged as
// bytes is taken out of that string.
func TestRedactLogAttrTakesTheKeyOutOfBytes(t *testing.T) {
	err, key := echoedKey()
	line := logged(t, true, "provider answered", "body", []byte(err.Error()))
	if strings.Contains(line, key) || !strings.Contains(line, `body="502 Bad Gateway: upstream error: failed to connect to local service\nrequest headers: [redacted]"`) {
		t.Fatalf("the bytes kept the key or lost the body: %s", line)
	}
}

// The text handler prints an error as %+v, which for a stack.Wrap error is the error and its
// stack; the redacted error keeps the stack.
func TestRedactLogAttrKeepsTheStackOfAnErrorItRedacts(t *testing.T) {
	err, key := echoedKey()
	line := logged(t, true, "agent failure", "error", stack.Wrap(err))
	if strings.Contains(line, key) || !strings.Contains(line, "TestRedactLogAttrKeepsTheStackOfAnErrorItRedacts") {
		t.Fatalf("the redacted error kept the key or lost its stack: %s", line)
	}
}

// A log line has no cap: a long error is logged whole, as it was before, with only its key
// taken out.
func TestRedactLogAttrKeepsALongErrorWhole(t *testing.T) {
	err, key := echoedKey()
	long := fmt.Errorf("%s: %w", strings.Repeat("upstream said no. ", 600), err)
	line := logged(t, true, "agent failure", "error", long)
	if strings.Contains(line, key) || strings.Count(line, "upstream said no.") != 600 ||
		!strings.Contains(line, "request headers: [redacted]") {
		t.Fatalf("the long error kept the key or lost its text: %d bytes", len(line))
	}
}

// The control is the line a text handler with no ReplaceAttr prints, which is the line the
// router printed before RedactLogAttr: with nothing to redact, it is the same byte for byte.
func TestRedactLogAttrLeavesALineWithNoKeyAsItWas(t *testing.T) {
	args := []any{
		"error", stack.Wrap(errors.New("502 Bad Gateway: upstream error: failed to connect to local service")),
		"reason", "perplexity: PERPLEXITY_API_KEY is required",
		"tokens", 42,
		"took", 1500 * time.Millisecond,
		"extra", map[string]any{"day": "2026-11-06", "rows": 3},
		"payload", []byte(`{"type":"ping"}`),
	}
	want := logged(t, false, "agent failure", args...)
	if got := logged(t, true, "agent failure", args...); got != want {
		t.Fatalf("the line changed:\n got %s\nwant %s", got, want)
	}
}

// loggedJSON is what a JSON handler with RedactJSONLogAttr, or with no ReplaceAttr when redact
// is false, prints for one record at a pinned time.
func loggedJSON(t *testing.T, redact bool, message string, args ...any) string {
	t.Helper()
	var out bytes.Buffer
	options := &slog.HandlerOptions{}
	if redact {
		options.ReplaceAttr = RedactJSONLogAttr
	}
	record := slog.NewRecord(time.Date(2026, 11, 6, 12, 0, 0, 0, time.UTC), slog.LevelError, message, 0)
	record.Add(args...)
	if err := slog.NewJSONHandler(&out, options).Handle(context.Background(), record); err != nil {
		t.Fatal(err)
	}
	return out.String()
}

// hiddenText is an error whose %v and %+v show less than its Error(), which is what the JSON
// handler prints.
type hiddenText struct{ text string }

func (e hiddenText) Error() string                 { return e.text }
func (e hiddenText) Format(f fmt.State, verb rune) { fmt.Fprint(f, "provider failed") }

// The JSON handler prints an error's Error(): the key is taken out of that, though %+v hides
// it, and a stack.Wrap error stays its Error(), with no stack added.
func TestRedactJSONLogAttrReadsAnErrorAsTheHandlerPrintsIt(t *testing.T) {
	err, key := echoedKey()
	line := loggedJSON(t, true, "agent failure", "hidden", hiddenText{err.Error()}, "wrapped", stack.Wrap(err),
		"body", []byte(err.Error()), "extra", map[string]any{"listing": err.Error()})
	if strings.Contains(line, key) {
		t.Fatalf("the key survived: %s", line)
	}
	reason := `502 Bad Gateway: upstream error: failed to connect to local service\nrequest headers: [redacted]`
	for _, want := range []string{`"hidden":"` + reason + `"`, `"wrapped":"` + reason + `"`, `"body":"` + reason + `"`, `listing`} {
		if !strings.Contains(line, want) {
			t.Fatalf("the line lost %s: %s", want, line)
		}
	}
}

// A message and a string attribute are redacted too, whatever their key.
func TestRedactJSONLogAttrRedactsAMessageAndAStringAttribute(t *testing.T) {
	err, key := echoedKey()
	line := loggedJSON(t, true, err.Error(), "reason", err.Error())
	if strings.Contains(line, key) {
		t.Fatalf("the key survived: %s", line)
	}
	if !strings.Contains(line, `"reason":"502 Bad Gateway`) {
		t.Fatalf("the line lost the reason: %s", line)
	}
}

// marshalsOtherText is an error whose Error() is clean and whose JSON is not: the handler
// prints the JSON.
type marshalsOtherText struct{ text string }

func (e marshalsOtherText) Error() string { return "provider failed" }
func (e marshalsOtherText) MarshalJSON() ([]byte, error) {
	return json.Marshal(e.text)
}

func TestRedactJSONLogAttrRedactsWhatAMarshalingErrorPrints(t *testing.T) {
	err, key := echoedKey()
	line := loggedJSON(t, true, "agent failure", "error", marshalsOtherText{err.Error()})
	if strings.Contains(line, key) {
		t.Fatalf("the key survived: %s", line)
	}
}

func TestRedactJSONLogAttrLeavesALineWithNoKeyAsItWas(t *testing.T) {
	args := []any{
		"error", stack.Wrap(errors.New("502 Bad Gateway: upstream error: failed to connect to local service")),
		"reason", "perplexity: PERPLEXITY_API_KEY is required",
		"tokens", 42,
		"extra", map[string]any{"day": "2026-11-06", "rows": 3},
		"payload", []byte(`{"type":"ping"}`),
	}
	want := loggedJSON(t, false, "agent failure", args...)
	if got := loggedJSON(t, true, "agent failure", args...); got != want {
		t.Fatalf("the line changed:\n got %s\nwant %s", got, want)
	}
}

// A Sentry event carries an error's text in its message, its exceptions and its breadcrumbs.
func TestRedactSentryEventKeepsTheReasonAndNotTheKey(t *testing.T) {
	err, key := echoedKey()
	event := RedactSentryEvent(&sentry.Event{
		Message:     "router stopped: " + err.Error(),
		Exception:   []sentry.Exception{{Type: "*errors.errorString", Value: err.Error()}},
		Breadcrumbs: []*sentry.Breadcrumb{{Message: err.Error()}},
	}, nil)
	reason := "502 Bad Gateway: upstream error: failed to connect to local service\nrequest headers: [redacted]"
	if event.Message != "router stopped: "+reason || event.Exception[0].Value != reason || event.Breadcrumbs[0].Message != reason {
		t.Fatalf("the event kept the key or lost the reason: %+v", event)
	}
	if event.Exception[0].Type != "*errors.errorString" || strings.Contains(fmt.Sprint(event.Breadcrumbs[0]), key) {
		t.Fatalf("the event changed what it should keep: %+v", event)
	}
}

// A usage alert carries a provider's error in a context (internal/alerts), and a tag can name
// what failed: every string in them is redacted, at any depth and in a value of any type, and
// the reason before the key is kept.
func TestRedactSentryEventTakesTheKeyOutOfContextsAndTags(t *testing.T) {
	err, key := echoedKey()
	reason := "502 Bad Gateway: upstream error: failed to connect to local service\nrequest headers: [redacted]"
	type reading struct {
		Error string
		Rows  int64
	}
	event := RedactSentryEvent(&sentry.Event{
		Tags: map[string]string{"item": err.Error()},
		Contexts: map[string]sentry.Context{"usage_alert": {
			"reading": err.Error(),
			"nested":  map[string]any{"error": err.Error()},
			"listing": []any{err.Error()},
			"typed":   map[string][]reading{"apps": {{Error: err.Error(), Rows: 1<<60 + 1}}},
		}},
	}, nil)
	if sent, _ := json.Marshal(event); strings.Contains(string(sent), key) {
		t.Fatalf("the event Sentry is sent keeps the key: %s", sent)
	}
	context := event.Contexts["usage_alert"]
	if event.Tags["item"] != reason || context["reading"] != reason {
		t.Fatalf("a tag or a context kept the key or lost the reason: %+v", event)
	}
	if context["nested"].(map[string]any)["error"] != reason || context["listing"].([]any)[0] != reason {
		t.Fatalf("a nested map or slice kept the key or lost the reason: %+v", context)
	}
	typed, _ := json.Marshal(context["typed"])
	if want := `{"apps":[{"Error":` + strconv.Quote(reason) + `,"Rows":1152921504606846977}]}`; string(typed) != want {
		t.Fatalf("a typed value kept the key or lost the reason:\n got %s\nwant %s", typed, want)
	}
}

// The scope an event came from shares its contexts' nested values with the next event, so
// redacting one event leaves them as they were.
func TestRedactSentryEventLeavesTheScopesContextsAlone(t *testing.T) {
	err, key := echoedKey()
	shared := map[string]any{"error": err.Error()}
	listed := []any{err.Error()}
	context := sentry.Context{"nested": shared, "listing": listed}
	RedactSentryEvent(&sentry.Event{Contexts: map[string]sentry.Context{"scope": context}}, nil)
	if !strings.Contains(shared["error"].(string), key) || !strings.Contains(listed[0].(string), key) {
		t.Fatalf("the scope's own context was changed: %+v %+v", shared, listed)
	}
	if context["nested"].(map[string]any)["error"] != err.Error() {
		t.Fatalf("the context the event shares with its scope was changed: %+v", context)
	}
}

// A context that holds itself would overflow the stack, which ends the process where Sentry's
// own encoding of it only fails: the walk stops at a depth, and what it reached above that is
// redacted as before.
func TestRedactSentryEventReturnsFromAContextThatHoldsItself(t *testing.T) {
	err, key := echoedKey()
	reason := "502 Bad Gateway: upstream error: failed to connect to local service\nrequest headers: [redacted]"
	loop := map[string]any{"error": err.Error()}
	loop["self"] = loop
	listing := []any{err.Error()}
	listing = append(listing, listing)
	listing[1] = listing

	event := RedactSentryEvent(&sentry.Event{Contexts: map[string]sentry.Context{
		"usage_alert": {"loop": loop, "listing": listing},
	}}, nil)

	context := event.Contexts["usage_alert"]
	if got := context["loop"].(map[string]any)["error"]; got != reason {
		t.Fatalf("a context that holds itself kept the key or lost the reason: %v", got)
	}
	if got := context["listing"].([]any)[0]; got != reason {
		t.Fatalf("a slice that holds itself kept the key or lost the reason: %v", got)
	}
	if strings.Contains(fmt.Sprint(context["loop"].(map[string]any)["error"]), key) {
		t.Fatal("the key is kept")
	}
}

// What an event says with no key in it reaches Sentry as it did before: the same text and
// the same values, of the same types. A typed value is the very value, not a copy of it.
func TestRedactSentryEventLeavesAnEventWithNoKeyAsItWas(t *testing.T) {
	type reading struct{ Rows int }
	typed := &reading{Rows: 3}
	event := RedactSentryEvent(&sentry.Event{
		Tags: map[string]string{"usage_alert": "sandbox_mismatch", "source": "router"},
		Contexts: map[string]sentry.Context{"usage_alert": {
			"day": "2026-11-06", "rows": 3, "increase": 1.5, "over": nil, "rerun": "router usage sandboxes --day 2026-11-06",
			"nested": map[string]any{"value": int64(1) << 60}, "listing": []any{"apps", true}, "typed": typed,
		}},
	}, nil)
	want := map[string]any{
		"day": "2026-11-06", "rows": 3, "increase": 1.5, "over": nil, "rerun": "router usage sandboxes --day 2026-11-06",
		"nested": map[string]any{"value": int64(1) << 60}, "listing": []any{"apps", true}, "typed": typed,
	}
	context := event.Contexts["usage_alert"]
	if fmt.Sprintf("%#v", map[string]any(context)) != fmt.Sprintf("%#v", want) || context["typed"] != any(typed) {
		t.Fatalf("the context changed:\n got %#v\nwant %#v", context, want)
	}
	if event.Tags["usage_alert"] != "sandbox_mismatch" || event.Tags["source"] != "router" || len(event.Tags) != 2 {
		t.Fatalf("the tags changed: %+v", event.Tags)
	}
}

// BenchmarkLogLine is the cost of RedactLogAttr on a router line with no key in it, typical and
// with a 4 KB error, against the same handler with no ReplaceAttr, which is what the router
// logged with before it.
func BenchmarkLogLine(b *testing.B) {
	typical := errors.New(`llmrouter: post "https://api.example.com/v1/chat/completions": context deadline exceeded`)
	long := errors.New(strings.Repeat("upstream said the model is overloaded, retry later; ", 80))
	for _, run := range []struct {
		name    string
		replace func([]string, slog.Attr) slog.Attr
		err     error
	}{
		{"typical/plain", nil, typical},
		{"typical/redacted", RedactLogAttr, typical},
		{"long/plain", nil, long},
		{"long/redacted", RedactLogAttr, long},
	} {
		b.Run(run.name, func(b *testing.B) {
			logger := slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{ReplaceAttr: run.replace}))
			b.ReportAllocs()
			for b.Loop() {
				logger.Info("served a request", "request_id", "01HZX3Q9V8M2", "model", "openai/gpt-4o-mini",
					"status", 200, "duration", 312*time.Millisecond, "error", run.err)
			}
		})
	}
}
