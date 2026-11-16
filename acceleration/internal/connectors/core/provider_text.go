package core

import (
	"net/url"
	"regexp"
	"unicode/utf8"
)

// DescriptionBytes bounds a provider's error_description on a log line: 256 B, the owner's
// choice (AI-1245, agreed 2026-10-14), enough for the sentence RFC 6749 section 5.2 means it
// to be while a provider's page of text cannot fill the line.
const DescriptionBytes = 256

// LoggedDescription is a provider's error_description (RFC 6749 sections 4.1.2.1 and 5.2) as a
// log line has it: each of secrets replaced by [redacted], every http or https URL kept to its
// scheme and host (KeepHosts), cut to DescriptionBytes at a UTF-8 boundary. secrets are what
// the request carried that the provider may echo, such as the refresh token and the client
// secret; one of a byte or none is left alone, as redactionOf leaves it.
//
// Example: "refresh token rt-1 revoked, see https://idp.example/e?t=rt-1" with secret rt-1 is
// "refresh token [redacted] revoked, see https://idp.example".
func LoggedDescription(text string, secrets ...string) string {
	return CutAtRune(KeepHosts(redactionFrom(secrets).apply(text, true)), DescriptionBytes)
}

// MessageBytes bounds provider_message on a log line (LoggedMessage): 1 KiB, the ticket's choice
// (AI-1242), enough for a provider's error sentence while a gateway's HTML page cannot fill the
// log.
const MessageBytes = 1 << 10

// LoggedMessage is what err, the error a failed call returned, says, for its log line: the
// tool's own words for an MCP isError (ToolError, whose Error is its Message), the SDK's
// or the JSON-RPC error's text, or the refusal's reason (session's dispatcher.recheck). Empty
// for nil.
//
// Every http or https URL in it keeps only its scheme and host, since its userinfo, path or
// query can carry a token. The text is searched rather than a *url.Error unwrapped: the MCP
// SDK also flattens one into text with %v, where errors.As no longer finds it (go-sdk v1.8.0
// mcp/streamable.go:2174 and :2560), and a connector's endpoints are all https
// (internal/connectors/core/AGENTS.md, «Untrusted values cannot steer a request»). It is
// then passed through redact, when it is not nil: the call's Exchange.Redact, which takes
// out the last request's path and query and the strings of its credential, since the MCP
// client puts a JSON-RPC error's message, and a tool its isError text, in the error, and a
// provider may echo either there (AI-1244). It is then cut to MessageBytes at a UTF-8
// boundary. The session's dispatcher logs it for a session's tool call, and a connection's
// validate for a credential the provider refused (api.checkConnection).
//
// Example: `Post "https://crm.example/mcp?access_token=x": EOF` is logged as
// `Post "https://crm.example": EOF`, and `invalid token <the token> for /primary/mcp` as
// `invalid token [redacted] for [redacted]`.
func LoggedMessage(err error, redact func(string) string) string {
	if err == nil {
		return ""
	}
	text := KeepHosts(err.Error())
	if redact != nil {
		text = redact(text)
	}
	return CutAtRune(text, MessageBytes)
}

// LoggedBody is the start of a 4xx's or a 5xx's body (Exchange.ErrorBody, already redacted) as
// a log line has it: without the UTF-8 sequence its 1 KiB cut may end inside, with every http or
// https URL kept to its scheme and host (KeepHosts), at most maxErrorExcerpt bytes.
func LoggedBody(body string) string {
	return CutAtRune(KeepHosts(wholeRunes(body)), maxErrorExcerpt)
}

// anyURL is an http or https URL in a text, up to the space or quote that ends it.
var anyURL = regexp.MustCompile(`(?i)https?://[^\s"]+`)

// KeepHosts is text with every http or https URL in it kept to its scheme and host, since a
// URL's userinfo, path or query can carry a token. A URL that does not parse with a host
// becomes "<url>".
//
// Example: `Post "https://crm.example/mcp?access_token=x": EOF` is
// `Post "https://crm.example": EOF`.
func KeepHosts(text string) string {
	return anyURL.ReplaceAllStringFunc(text, origin)
}

// origin is raw's scheme and host, without the userinfo, path, query or fragment it had;
// "<url>" when raw does not parse as a URL with a host.
func origin(raw string) string {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Host == "" {
		return "<url>"
	}
	return parsed.Scheme + "://" + parsed.Host
}

// CutAtRune is text cut to at most limit bytes, never splitting a UTF-8 sequence: the cut
// moves back to the start of the rune it falls in, at most utf8.UTFMax-1 bytes, so text that
// is not UTF-8 is still cut near limit. CutResult does the same for a result, at
// MaxResultBytes and with a marker the model reads, so it does not fit a log line.
func CutAtRune(text string, limit int) string {
	if len(text) <= limit {
		return text
	}
	end := limit
	for end > limit-(utf8.UTFMax-1) && !utf8.RuneStart(text[end]) {
		end--
	}
	return text[:end]
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
