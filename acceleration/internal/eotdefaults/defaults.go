// Package eotdefaults contains the fixed hosted EOT demo destination shared by the
// command-line demo and the ordinary router defaults.
package eotdefaults

import (
	"net/url"
	"strings"
)

// HostedDemoEndpoint is the packaged EU demo service. It is intentionally fixed so
// the anonymous client cannot be repointed at an arbitrary host.
const HostedDemoEndpoint = "https://audioturn-demo-eu-5gdhza7snq-ez.a.run.app/v1/eot"

const hostedDemoHostname = "audioturn-demo-eu-5gdhza7snq-ez.a.run.app"

// IsHostedDemoOrigin identifies the fixed demo service even when its endpoint was
// written using the root URL form accepted by the EOT client. Callers use this before
// deciding whether credentials may be attached.
func IsHostedDemoOrigin(raw string) bool {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed == nil || parsed.User != nil {
		return false
	}
	port := parsed.Port()
	return strings.EqualFold(parsed.Scheme, "https") &&
		strings.EqualFold(strings.TrimSuffix(parsed.Hostname(), "."), hostedDemoHostname) &&
		(port == "" || port == "443")
}

// IsHostedDemoEndpoint accepts the canonical EOT URL and the root URL form that the
// EOT client normalizes to it. Other paths or URL decorations must not be silently
// redirected to the fixed public scorer.
func IsHostedDemoEndpoint(raw string) bool {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed == nil || parsed.User != nil || parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" || parsed.RawPath != "" {
		return false
	}
	if !IsHostedDemoOrigin(raw) {
		return false
	}
	return parsed.Path == "" || parsed.Path == "/" || parsed.Path == "/v1/eot"
}
