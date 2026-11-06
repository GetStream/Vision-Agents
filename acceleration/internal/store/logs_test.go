package store

import (
	"crypto/rand"
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

// A provider's key travels in a URL query or a URL-encoded header, and a Go http or websocket
// error prints that URL whole. The rows are the names we send a key in (key, token), the names
// credentialText took before this change (api_key, apikey, access_token, client_secret) and
// headers written into a URL.
func TestSafeLogTextQueryAndEncodedCredentials(t *testing.T) {
	secret := rand.Text()
	backslash := `\`
	for name, message := range map[string]string{
		"key after a json-escaped ampersand":   `{"url":"wss://live.example/ws?alt=sse` + backslash + `u0026key=` + secret + `"}`,
		"token after a json-escaped ampersand": `{"url":"https://ipinfo.io/json?lang=en` + backslash + `u0026token=` + secret + `"}`,
		"key in a twice-encoded url":           `redirect=https%253A%252F%252Flive.example%252Fws%253Fkey%253D` + secret,
		"key after a twice-encoded ampersand":  `redirect=https%253A%252F%252Flive.example%252Fws%253Fa%253D1%2526key%253D` + secret,
		"key after an html-escaped ampersand":  `<a href="https://live.example/ws?alt=sse&amp;key=` + secret + `">`,
		"twice-encoded bearer":                 `h=Bearer%2520` + secret,
		"twice-encoded authorization":          `h=Authorization%253A%2520Basic%2520` + secret,
		"key first (gemini stt and sts)":       `Get "wss://live.example/ws?key=` + secret + `": dial tcp: i/o timeout`,
		"key after another parameter":          `wss://live.example/ws?alt=sse&key=` + secret + `&model=m`,
		"key in upper case":                    `wss://live.example/ws?KEY=` + secret,
		"key in an encoded url":                `redirect=https%3A%2F%2Flive.example%2Fws%3Fkey%3D` + secret,
		"key after an encoded ampersand":       `redirect=https%3A%2F%2Flive.example%2Fws%3Fa%3D1%26key%3D` + secret,
		"token (ipinfo)":                       `Get "https://ipinfo.io/203.0.113.10/json?token=` + secret + `": EOF`,
		"token after another parameter":        `https://ipinfo.io/json?lang=en&token=` + secret,
		"api_key":                              `https://api.example/v1?api_key=` + secret,
		"apikey":                               `https://api.example/v1?apikey=` + secret,
		"access_token":                         `https://api.example/v1?access_token=` + secret,
		"client_secret":                        `https://auth.example/token?client_secret=` + secret,
		"encoded authorization bearer":         `h=Authorization%3A%20Bearer%20` + secret,
		"encoded authorization plus":           `h=Authorization%3A+Bearer+` + secret,
		"encoded authorization basic":          `h=authorization%3A%20Basic%20` + secret,
		"encoded authorization json":           `q=%7B%22Authorization%22%3A%22Basic%20` + secret + `%22%7D`,
		"encoded authorization quoted":         `q=%7B%27authorization%27%3A%27Basic%20` + secret + `%27%7D`,
		"encoded authorization param":          `redirect=https%3A%2F%2Fapi.example%2Fv1%3Fauthorization%3DBearer%2520` + secret,
		"encoded set-cookie":                   `h=Set-Cookie%3A%20session%3D` + secret,
		"encoded proxy-authorization":          `h=Proxy-Authorization%3A%20Basic%20` + secret,
		"encoded cookie":                       `h=Cookie%3A%20session%3D` + secret,
		"encoded bearer alone":                 `h=Bearer%20` + secret,
		"encoded bearer after a space":         `q=sent%20Bearer%20` + secret,
		"form-encoded bearer alone":            `h=Bearer+` + secret,
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
	for message, want := range map[string]string{
		`{"url":"wss://live.example/ws?key=` + secret + backslash + `u0026model=m"}`: `{"url":"wss://live.example/ws?key=[redacted]` + backslash + `u0026model=m"}`,
		`Get "wss://live.example/ws?alt=sse&key=` + secret + `&model=m": EOF`:        `Get "wss://live.example/ws?alt=sse&key=[redacted]&model=m": EOF`,
		`h=Authorization%3A%20Basic%20` + secret + `&model=m`:                        `h=[redacted]&model=m`,
		`h=Bearer%20` + secret + `&model=m`:                                          `h=[redacted]&model=m`,
	} {
		if result := SafeLogText(message); result != want {
			t.Errorf("SafeLogText(%q) = %q, want %q", message, result, want)
		}
	}
	// The answers base 6b9627f99 gives for them, the same as the input but for the last one,
	// whose "bearer of" the bearer rule took before this change too.
	for message, want := range map[string]string{
		`wss://stt.example/v1/listen?model=nova-3&keyterm=Stream&keyterms=Vision&keyterms_prompt=%5B%22Agents%22%5D&language_code=en`: `wss://stt.example/v1/listen?model=nova-3&keyterm=Stream&keyterms=Vision&keyterms_prompt=%5B%22Agents%22%5D&language_code=en`,
		`wss://stt.example/v1/listen?monkey=banana&keys=3&tokens=12&token_type_hint=access_token&max_tokens=256`:                      `wss://stt.example/v1/listen?monkey=banana&keys=3&tokens=12&token_type_hint=access_token&max_tokens=256`,
		`the key=value pair and token=12 in prose`:                                                        `the key=value pair and token=12 in prose`,
		`Get "https://api.example/v1/models?page=2&limit=50": dial tcp: lookup api.example: no such host`: `Get "https://api.example/v1/models?page=2&limit=50": dial tcp: lookup api.example: no such host`,
		`authorization%20failed for model gpt and the bearer of bad news`:                                 `authorization%20failed for model gpt and the [redacted] bad news`,
	} {
		if result := SafeLogText(message); result != want {
			t.Errorf("SafeLogText(%q) = %q, want %q", message, result, want)
		}
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
