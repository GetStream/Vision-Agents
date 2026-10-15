package session

import (
	"net/url"
	"regexp"
	"unicode/utf8"
)

// providerMessageBytes bounds provider_message on a "connector call failed" line
// (dispatcher.record): 1 KiB, the ticket's choice (AI-1242), enough for a provider's error
// sentence while a gateway's HTML page cannot fill the log.
const providerMessageBytes = 1 << 10

// providerMessage is what err, the error a failed call returned, says, for its log line: the
// tool's own words for an MCP isError (core.ToolError, whose Error is its Message), the SDK's
// or the JSON-RPC error's text, or the refusal's reason (dispatcher.recheck). Empty for nil.
//
// Every http or https URL in it keeps only its scheme and host, since its userinfo, path or
// query can carry a token. The text is searched rather than a *url.Error unwrapped: the MCP
// SDK also flattens one into text with %v, where errors.As no longer finds it (go-sdk v1.8.0
// mcp/streamable.go:2174 and :2560), and a connector's endpoints are all https
// (internal/connectors/core/AGENTS.md, «Untrusted values cannot steer a request»). It is then
// cut to providerMessageBytes at a UTF-8 boundary.
//
// Example: `Post "https://crm.example/mcp?access_token=x": EOF` is logged as
// `Post "https://crm.example": EOF`.
func providerMessage(err error) string {
	if err == nil {
		return ""
	}
	return providerText(err.Error())
}

// providerText is text a provider wrote, such as the start of its error body
// (core.Exchange.ErrorBody), with every http or https URL in it kept to its scheme and host,
// cut to providerMessageBytes at a UTF-8 boundary: what providerMessage does to an error's text.
func providerText(text string) string {
	return cutAtRune(anyURL.ReplaceAllStringFunc(text, origin), providerMessageBytes)
}

// providerBody is the start of a 4xx's or a 5xx's body as its log line has it: what the
// transport kept (core.Exchange.ErrorBody) without the UTF-8 sequence its 1 KiB cut may end
// inside, through providerText.
func providerBody(body string) string {
	return providerText(wholeRunes(body))
}

// wholeRunes is text without the UTF-8 sequence its last bytes start and do not finish; text
// that ends on a whole rune, or is not UTF-8, is unchanged.
func wholeRunes(text string) string {
	for back := 1; back < utf8.UTFMax && back <= len(text); back++ {
		if start := len(text) - back; utf8.RuneStart(text[start]) {
			if !utf8.FullRuneInString(text[start:]) {
				return text[:start]
			}
			return text
		}
	}
	return text
}

// anyURL is an http or https URL in an error's text, up to the space or quote that ends it.
var anyURL = regexp.MustCompile(`(?i)https?://[^\s"]+`)

// origin is raw's scheme and host, without the userinfo, path, query or fragment it had;
// "<url>" when raw does not parse as a URL with a host.
func origin(raw string) string {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Host == "" {
		return "<url>"
	}
	return parsed.Scheme + "://" + parsed.Host
}

// cutAtRune is text cut to at most limit bytes, never splitting a UTF-8 sequence: the cut
// moves back to the start of the rune it falls in, at most utf8.UTFMax-1 bytes, so text that
// is not UTF-8 is still cut near limit. core.CutResult does the same for a result, at
// core.MaxResultBytes and with a marker the model reads, so it does not fit here.
func cutAtRune(text string, limit int) string {
	if len(text) <= limit {
		return text
	}
	end := limit
	for end > limit-(utf8.UTFMax-1) && !utf8.RuneStart(text[end]) {
		end--
	}
	return text[:end]
}
