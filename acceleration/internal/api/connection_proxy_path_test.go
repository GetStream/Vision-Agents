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

// TestGitHubTakesDirectCallsToTheRESTAPI: revision 4 of the built-in github has an api_base
// (AI-1308), so a call to repos/octo/hello/issues goes to GitHub's REST API.
func (s *ProxyTargetSuite) TestGitHubTakesDirectCallsToTheRESTAPI() {
	raw, err := fs.ReadFile(providers.FS, "github.yaml")
	s.Require().NoError(err)
	manifest, err := core.ParseManifest(raw)
	s.Require().NoError(err)

	base, found := manifest.Endpoints[proxyBase]

	s.Require().True(found, "github has no %s", proxyBase)
	target, err := proxyTarget(base, "repos/octo/hello/issues", "")
	s.Require().NoError(err)
	s.Equal("https://api.github.com/repos/octo/hello/issues", target.String())
}

// ProxiedLinksSuite is which of a provider's Link header targets a direct call's answer sends
// to the connection's proxy instead (AI-1308).
type ProxiedLinksSuite struct {
	suite.Suite
}

func TestProxiedLinksSuite(t *testing.T) {
	suite.Run(t, new(ProxiedLinksSuite))
}

// linkProxy is a connection's proxy URL under a router at https://router.example.
const linkProxy = "https://router.example/v1/agents/connections/c1/proxy"

// TestAGitHubPageLinkGoesToTheProxy: GitHub REST pages with absolute links to api.github.com,
// so each, in one value or in several, is the same path and query under the proxy, as written.
// The first value is what GET https://api.github.com/repositories?per_page=1 answered on
// 2026-10-15 (a URI template in rel="first" included); the second is from
// https://docs.github.com/en/rest/using-the-rest-api/using-pagination-in-the-rest-api (opened
// 2026-10-15).
func (s *ProxiedLinksSuite) TestAGitHubPageLinkGoesToTheProxy() {
	links := proxiedLinks([]string{
		`<https://api.github.com/repositories?per_page=1&since=371>; rel="next", <https://api.github.com/repositories{?since}>; rel="first"`,
		`<https://api.github.com/repositories/1300192/issues?per_page=2&page=2>; rel="next"`,
	}, "https://api.github.com", linkProxy)

	s.Equal([]string{
		`<` + linkProxy + `/repositories?per_page=1&since=371>; rel="next", <` + linkProxy + `/repositories{?since}>; rel="first"`,
		`<` + linkProxy + `/repositories/1300192/issues?per_page=2&page=2>; rel="next"`,
	}, links)
}

// TestALinkUnderABasePathKeepsWhatFollowsIt: under https://slack.com/api the link's path after
// /api follows the proxy URL.
func (s *ProxiedLinksSuite) TestALinkUnderABasePathKeepsWhatFollowsIt() {
	links := proxiedLinks([]string{`<https://slack.com/api/conversations.list?cursor=x>; rel="next"`},
		"https://slack.com/api", linkProxy)

	s.Equal([]string{`<` + linkProxy + `/conversations.list?cursor=x>; rel="next"`}, links)
}

// TestALinkOutsideTheBaseIsLeftAsItIs: another host, a host that only starts with the base's,
// a userinfo naming it, another scheme or port, a path beside the base's, and a «<» in a quoted
// parameter are not under the base, so none is rewritten.
func (s *ProxiedLinksSuite) TestALinkOutsideTheBaseIsLeftAsItIs() {
	for _, link := range []string{
		`<https://docs.github.com/rest>; rel="help"`,
		`<https://api.github.com.evil.example/user?page=2>; rel="next"`,
		`<https://api.github.com@evil.example/user>; rel="next"`,
		`<http://api.github.com/user>; rel="next"`,
		`<https://api.github.com:8443/user>; rel="next"`,
		`<https://api.github.com\@evil.example/user>; rel="next"`,
		`<https://evil.example/x>; title="<https://api.github.com/user>"`,
	} {
		s.Equal([]string{link}, proxiedLinks([]string{link}, "https://api.github.com", linkProxy), link)
	}
	beside := `<https://slack.com/apix/y>; rel="next"`
	s.Equal([]string{beside}, proxiedLinks([]string{beside}, "https://slack.com/api", linkProxy))
}

// TestWithNoPublicURLThereAreNoLinks: a router with no public URL has no proxy URL to send a
// link to, and one left as it is could take the router's credential to the provider.
func (s *ProxiedLinksSuite) TestWithNoPublicURLThereAreNoLinks() {
	s.Empty(proxiedLinks([]string{`<https://api.github.com/user?page=2>; rel="next"`}, "https://api.github.com", ""))
}
