package api

import (
	"io/fs"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
)

// ProxyTargetSuite is which provider URL a direct call's path makes, and which paths are
// refused before anything is sent (AI-958).
type ProxyTargetSuite struct {
	suite.Suite
}

func TestProxyTargetSuite(t *testing.T) {
	suite.Run(t, new(ProxyTargetSuite))
}

// TestAnEscapedSlashBackslashOrDotSegmentIsRefused: the ticket's two paths, every case of
// %2F, %5C and %2E, and a dot segment spelled half escaped.
func (s *ProxyTargetSuite) TestAnEscapedSlashBackslashOrDotSegmentIsRefused() {
	for _, path := range []string{
		"a/%2e%2e%2fx", "a%2F..%2F..%2Fx",
		"a%2fb", "a%2Fb", "a/%2F", "%2F%2Fevil.example%2Fx",
		"a%5cb", "a%5Cb", "a/%5C..%5Cx",
		"..", ".", "a/../b", "a/./b",
		"%2e", "%2E", "%2e%2e", "%2E%2E", "%2e%2E", ".%2e", "%2e.", ".%2E", "a/.%2e/x", "a/%2E./x",
	} {
		_, err := proxyTarget("https://slack.com/api", path, "")
		s.Error(err, path)
	}
}

// TestAnOrdinaryPathStillGoesAsItCame: the paths a provider's API is called with, an escaped
// character other than a slash or a backslash included, are unchanged by the rule.
func (s *ProxyTargetSuite) TestAnOrdinaryPathStillGoesAsItCame() {
	for path, want := range map[string]string{
		"chat.postMessage":         "https://slack.com/api/chat.postMessage",
		"v1/items/a%20b":           "https://slack.com/api/v1/items/a%20b",
		"users/me":                 "https://slack.com/api/users/me",
		"files/.well-known":        "https://slack.com/api/files/.well-known",
		"a/...":                    "https://slack.com/api/a/...",
		"a/..b":                    "https://slack.com/api/a/..b",
		"a/b.":                     "https://slack.com/api/a/b.",
		"a/%2e%2e%2e":              "https://slack.com/api/a/%2e%2e%2e",
		"repos/o/r/contents/x.txt": "https://slack.com/api/repos/o/r/contents/x.txt",
		"@evil.example/x":          "https://slack.com/api/@evil.example/x",
	} {
		target, err := proxyTarget("https://slack.com/api", path, "")
		s.Require().NoError(err, path)
		s.Equal(want, target.String(), path)
	}
}

// TestSlackBotTakesDirectCallsToTheWebAPI: revision 5 of the built-in slack_bot has an
// api_base, so a call to chat.postMessage goes to Slack's Web API.
func (s *ProxyTargetSuite) TestSlackBotTakesDirectCallsToTheWebAPI() {
	raw, err := fs.ReadFile(providers.FS, "slack_bot.yaml")
	s.Require().NoError(err)
	manifest, err := core.ParseManifest(raw)
	s.Require().NoError(err)

	base, found := manifest.Endpoints[proxyBase]

	s.Require().True(found, "slack_bot has no %s", proxyBase)
	target, err := proxyTarget(base, "chat.postMessage", "")
	s.Require().NoError(err)
	s.Equal("https://slack.com/api/chat.postMessage", target.String())
}
