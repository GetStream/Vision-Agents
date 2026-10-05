package oauth2code

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// Classify maps a provider's answer to the one outcome the core acts on. It reads token
// endpoint answers (RFC 6749 section 5.2) and protected resource answers (RFC 6750 section
// 3) alike, since AccessCredential and a tool source both hand it theirs and the two cannot always be told
// apart. In order:
//
//   - err with no response: a request that was never written (errNotSent, or a failed
//     dial) is Transient, since nothing reached the provider. Any other failure is
//     Uncertain: the provider may have acted on it, so a rotated refresh token must not be
//     replayed. The prototype treated every transport failure of a refresh that way
//     (internal/mcp/oauth.go:560-572 on codex/connector-support at cf62af0d).
//   - err with a response, a body that could not be read: the status and headers that did
//     arrive decide, as below without the body. Only a 1xx-3xx stays Uncertain, since a
//     success whose body was lost may have rotated the refresh token.
//   - 429 is RateLimited (RFC 6585 section 4), with Retry-After.
//   - A Bearer challenge in WWW-Authenticate: insufficient_scope is ScopeRequired with the
//     challenge's scope (RFC 6750 section 3.1); insufficient_claims on a 401 is
//     ScopeRequired with the decoded claims request, which is a Microsoft convention, not an
//     RFC («Claims challenges, claims requests and client capabilities»,
//     learn.microsoft.com/en-us/entra/identity-platform/claims-challenge); invalid_token is
//     InvalidGrant, since the resolver hands out only a token it believes live and RFC 6750
//     section 3.1 says the provider found it «expired, revoked, malformed, or invalid».
//   - An error member in a JSON body that errorCodes names, at any status, because some
//     servers answer errors with 200 (RFC 6749 section 5.2 does not, the fake's Slack shape
//     does). An error member it does not name is a resource's own error, such as a 404's
//     not_found, and falls through: AccessCredential's redeem makes it a refusal of the refresh.
//   - 503 is Transient (RFC 9110 section 15.6.4: the server «is currently unable to handle
//     the request»). Any other 5xx is Uncertain: a 500 (section 15.6.1) does not say nothing
//     happened, and a 502 or 504 (sections 15.6.3, 15.6.5) says a gateway lost the answer
//     of a server that may have acted. The prototype made every non-2xx refresh answer
//     Uncertain (oauth.go:582-584).
//   - Anything else is OK: nothing about the credential for the core to act on. A 404 is
//     still the caller's to report, and so is a resource's own error code.
//
// RetryAfter is set whenever the answer carries a Retry-After the outcome can use.
func (s *Scheme) Classify(resp *http.Response, body []byte, err error) core.Outcome {
	if resp == nil {
		var dial *net.OpError
		if errors.Is(err, errNotSent) || (errors.As(err, &dial) && dial.Op == "dial") {
			return core.Outcome{Kind: core.OutcomeTransient}
		}
		return core.Outcome{Kind: core.OutcomeUncertain}
	}
	if err != nil {
		// What arrived of the body is not trusted.
		body = nil
	}
	retryAfter := s.retryAfter(resp.Header.Get("Retry-After"))
	if resp.StatusCode == http.StatusTooManyRequests {
		return core.Outcome{Kind: core.OutcomeRateLimited, RetryAfter: retryAfter}
	}
	if challenge, ok := bearerChallenge(resp.Header.Values("WWW-Authenticate")); ok {
		switch challenge["error"] {
		case "insufficient_scope":
			// RFC 6750 section 3: scope is «a space-delimited list».
			return core.Outcome{Kind: core.OutcomeScopeRequired, Scopes: strings.Fields(challenge["scope"])}
		case "insufficient_claims":
			// Microsoft: the status «Must be 401 Unauthorized» and claims is «Required when
			// error is "insufficient_claims"».
			if resp.StatusCode == http.StatusUnauthorized {
				return core.Outcome{Kind: core.OutcomeScopeRequired, Claims: decodeClaims(challenge["claims"])}
			}
		case "invalid_token":
			return core.Outcome{Kind: core.OutcomeInvalidGrant}
		}
	}
	if kind, known := errorCodes[errorCode(body)]; known {
		return core.Outcome{Kind: kind, RetryAfter: retryAfter}
	}
	switch {
	case resp.StatusCode == http.StatusServiceUnavailable:
		return core.Outcome{Kind: core.OutcomeTransient, RetryAfter: retryAfter}
	case resp.StatusCode >= http.StatusInternalServerError:
		return core.Outcome{Kind: core.OutcomeUncertain, RetryAfter: retryAfter}
	case err != nil && resp.StatusCode < http.StatusBadRequest:
		return core.Outcome{Kind: core.OutcomeUncertain}
	}
	return core.Outcome{Kind: core.OutcomeOK}
}

// errorCodes are the error members Classify knows, and what each means.
var errorCodes = map[string]core.OutcomeKind{
	// RFC 6749 section 5.2: the refresh token is «invalid, expired, revoked» or was issued
	// to another client. Only a new consent helps.
	"invalid_grant": core.OutcomeInvalidGrant,
	// Not an RFC 6749 code: Slack's oauth.v2.access answers it for a refresh, «The given
	// refresh token is invalid» (docs.slack.dev/reference/methods/oauth.v2.access), and the
	// prototype mapped it with invalid_grant (internal/mcp/oauth.go:573 at cf62af0d).
	"invalid_refresh_token": core.OutcomeInvalidGrant,
	// RFC 6750 section 3.1, from a resource that puts it in the body instead of a challenge.
	"insufficient_scope": core.OutcomeScopeRequired,
	// RFC 6749 section 4.1.2.1: «currently unable to handle the request», the code for a
	// 503 that a redirect cannot carry.
	"temporarily_unavailable": core.OutcomeTransient,
	// RFC 6749 section 4.1.2.1: the code for a 500, which says nothing about what happened.
	"server_error": core.OutcomeUncertain,
	// Slack's two, each «It's possible some aspect of the operation succeeded before the
	// error was raised» (docs.slack.dev/reference/methods/oauth.v2.access); the prototype
	// made both Uncertain (internal/mcp/oauth.go:576-578 at cf62af0d).
	"internal_error": core.OutcomeUncertain,
	"fatal_error":    core.OutcomeUncertain,
	// RFC 6749 section 5.2's other codes, which only a token endpoint answers: a refusal,
	// so nothing was spent and a later attempt may pass. The prototype kept such a
	// connection connected and «temporarily unavailable» (internal/connectors/runtime.go:
	// 114-121 at cf62af0d). invalid_request is left out: RFC 6750 section 3.1 has a
	// resource answer it for a malformed call, which says nothing about the credential.
	"invalid_client":         core.OutcomeTransient,
	"unauthorized_client":    core.OutcomeTransient,
	"unsupported_grant_type": core.OutcomeTransient,
	"invalid_scope":          core.OutcomeTransient,
}

// errorCode is the error member of a JSON object body, or "" when there is none.
func errorCode(body []byte) string {
	if len(body) == 0 {
		return ""
	}
	object, err := decodeObject(body)
	if err != nil {
		return ""
	}
	code, _ := object["error"].(string)
	return code
}

// retryAfter reads Retry-After (RFC 9110 section 10.2.3): delay-seconds, or an HTTP-date
// counted from Config.Now. Anything else, or a date already past, is zero: not said.
func (s *Scheme) retryAfter(value string) time.Duration {
	if value == "" {
		return 0
	}
	// delay-seconds is 1*DIGIT. 32 bits of seconds is 136 years, which fits a Duration.
	if seconds, err := strconv.ParseUint(value, 10, 32); err == nil {
		return time.Duration(seconds) * time.Second
	}
	// http.ParseTime takes the three date forms RFC 9110 section 5.6.7 has a recipient
	// accept.
	if at, err := http.ParseTime(value); err == nil {
		if wait := at.Sub(s.cfg.Now()); wait > 0 {
			return wait
		}
	}
	return 0
}

// decodeClaims turns the base64 claims of a claims challenge into the claims request to
// send back. Microsoft: «A quoted string containing a base 64 encoded claims request», and to
// use it «Decode the base64 string received earlier» and pass it as the claims parameter.
// Which base64 alphabet is not said; the page's example is the standard one with padding, so
// that is tried first. A value that does not decode to JSON gives no claims.
func decodeClaims(value string) string {
	for _, encoding := range []*base64.Encoding{base64.StdEncoding, base64.RawStdEncoding, base64.URLEncoding, base64.RawURLEncoding} {
		if raw, err := encoding.DecodeString(value); err == nil && json.Valid(raw) {
			return string(raw)
		}
	}
	return ""
}

// bearerChallenge returns the auth-params, names lowercased, of the first Bearer challenge
// in the WWW-Authenticate field lines. RFC 9110 section 11.6.1: the field is a list of
// challenges, each an auth-scheme and then a token68 or auth-params (section 11.2); the
// scheme and the parameter names are case-insensitive. A token followed by "=" is a
// parameter and any other token starts the next challenge, which is how one comma-separated
// list holds both.
func bearerChallenge(lines []string) (map[string]string, bool) {
	for _, line := range lines {
		lexer := challengeLexer{text: line}
		for {
			scheme, params, ok := lexer.challenge()
			if !ok {
				break
			}
			if strings.EqualFold(scheme, "Bearer") {
				return params, true
			}
		}
	}
	return nil, false
}

type challengeLexer struct {
	text string
	at   int
}

// challenge reads one challenge: its scheme and its parameters.
func (l *challengeLexer) challenge() (string, map[string]string, bool) {
	l.skip(" \t,")
	scheme := l.token()
	if scheme == "" {
		return "", nil, false
	}
	params := map[string]string{}
	for {
		l.skip(" \t,")
		start := l.at
		name := l.token()
		l.skip(" \t")
		if name == "" {
			// Not a parameter: the rest of a token68, which may end in "=" padding (RFC 9110
			// section 11.2), or text this reader cannot parse. The next comma is where the
			// next parameter or challenge can start.
			if next := strings.IndexByte(l.text[l.at:], ','); next >= 0 {
				l.at += next
			} else {
				l.at = len(l.text)
			}
			return scheme, params, true
		}
		if !l.consume('=') {
			// The next challenge, or a token68 without padding.
			l.at = start
			return scheme, params, true
		}
		l.skip(" \t")
		params[strings.ToLower(name)] = l.value()
	}
}

func (l *challengeLexer) skip(chars string) {
	for l.at < len(l.text) && strings.IndexByte(chars, l.text[l.at]) >= 0 {
		l.at++
	}
}

func (l *challengeLexer) consume(c byte) bool {
	if l.at < len(l.text) && l.text[l.at] == c {
		l.at++
		return true
	}
	return false
}

// token reads RFC 9110 section 5.6.2's token: one or more tchar.
func (l *challengeLexer) token() string {
	start := l.at
	for l.at < len(l.text) && isTchar(l.text[l.at]) {
		l.at++
	}
	return l.text[start:l.at]
}

// value reads a token or a quoted-string with its quoted-pairs undone (RFC 9110 section
// 5.6.4).
func (l *challengeLexer) value() string {
	if !l.consume('"') {
		return l.token()
	}
	var out strings.Builder
	for l.at < len(l.text) {
		c := l.text[l.at]
		l.at++
		switch {
		case c == '"':
			return out.String()
		case c == '\\' && l.at < len(l.text):
			out.WriteByte(l.text[l.at])
			l.at++
		default:
			out.WriteByte(c)
		}
	}
	return out.String()
}

// isTchar is RFC 9110 section 5.6.2: «!#$%&'*+-.^_`|~», DIGIT and ALPHA.
func isTchar(c byte) bool {
	return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || strings.IndexByte("!#$%&'*+-.^_`|~", c) >= 0
}
