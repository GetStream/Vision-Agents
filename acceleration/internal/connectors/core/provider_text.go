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

// LoggedBody is the start of a 4xx's or a 5xx's body (Exchange.ErrorBody, already redacted) as
// a log line has it: without the UTF-8 sequence its 1 KiB cut may end inside, each of secrets
// replaced by [redacted], with every http or https URL kept to its scheme and host (KeepHosts),
// at most maxErrorExcerpt bytes. secrets are what else the request carried that the provider
// may echo, such as the caller's cookies a direct call forwards. The body may be cut, so one
// it ends with the start of is cut off too; one of a byte or none is left alone, as
// redactionOf leaves it.
//
// Example: `{"header":{"Cookie":["sid=s-1"]}}` with secret s-1 is `{"header":{"Cookie":["sid=[redacted]"]}}`.
func LoggedBody(body string, secrets ...string) string {
	return CutAtRune(KeepHosts(redactionFrom(secrets).apply(wholeRunes(body), false)), maxErrorExcerpt)
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
