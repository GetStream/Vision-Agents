package plugins

import (
	"bytes"
	"context"
	"log/slog"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/getsentry/sentry-go"
	"github.com/stretchr/testify/suite"
)

// loginLine is the line a login logged before uses went to Sentry too, taken from the base
// commit; nothing in it may change.
const loginLine = `level=WARN msg="deprecated plugin path used" path=login customer=cus_1 config=cfg_1 plugin=github via=plugins`

type ReportUseSuite struct {
	suite.Suite
	logged    bytes.Buffer
	logger    *slog.Logger
	transport *sentry.MockTransport
	client    *sentry.Client
}

func TestReportUseSuite(t *testing.T) {
	suite.Run(t, new(ReportUseSuite))
}

func (s *ReportUseSuite) SetupTest() {
	s.logged.Reset()
	s.logger = slog.New(slog.NewTextHandler(&s.logged, &slog.HandlerOptions{
		ReplaceAttr: func(_ []string, attr slog.Attr) slog.Attr {
			if attr.Key == slog.TimeKey {
				return slog.Attr{}
			}
			return attr
		},
	}))
	s.transport = &sentry.MockTransport{}
	client, err := sentry.NewClient(sentry.ClientOptions{Transport: s.transport})
	s.Require().NoError(err)
	s.client = client
}

func (s *ReportUseSuite) TestWithNoClientAUseIsTheLogLineAlone() {
	ctx := sentry.SetHubOnContext(context.Background(), sentry.NewHub(nil, sentry.NewScope()))

	s.login(ctx)

	s.Equal(loginLine, s.line())
}

func (s *ReportUseSuite) TestAUseIsAWarningIssueOfItsPathAndVia() {
	s.login(s.request())

	s.Equal(loginLine, s.line())
	s.Require().Len(s.transport.Events(), 1)
	event := s.transport.Events()[0]
	s.Equal(sentry.LevelWarning, event.Level)
	s.Equal(DeprecatedUse, event.Message)
	s.Equal([]string{DeprecatedUse, PathLogin, ViaPlugins}, event.Fingerprint)
	s.Equal(map[string]string{
		"path": PathLogin, "customer": "cus_1", "config": "cfg_1", "plugin": "github", "via": ViaPlugins,
	}, event.Tags)
}

func (s *ReportUseSuite) TestAUseWithNoViaIsAnIssueOfItsPathAlone() {
	ReportUse(s.request(), s.logger, PathSessionTools,
		"customer", "cus_1", "config", "cfg_1", "plugin", []string{"github"}, "mcp_server", []string{})

	s.Require().Len(s.transport.Events(), 1)
	event := s.transport.Events()[0]
	s.Equal([]string{DeprecatedUse, PathSessionTools}, event.Fingerprint)
	s.Equal(map[string]string{
		"path": PathSessionTools, "customer": "cus_1", "config": "cfg_1", "plugin": "[github]", "mcp_server": "[]",
	}, event.Tags)
}

func (s *ReportUseSuite) TestATagValueOverTheLimitIsCutToIt() {
	long := strings.Repeat("é", maxTagValue+50)

	ReportUse(s.request(), s.logger, PathSessionTools, "plugin", []string{long})

	s.Require().Len(s.transport.Events(), 1)
	tag := s.transport.Events()[0].Tags["plugin"]
	s.Equal(maxTagValue, len([]rune(tag)))
	s.Equal("[", tag[:1])
}

func (s *ReportUseSuite) TestAUseSendsNothingOfTheRequest() {
	s.login(s.request())

	s.Require().Len(s.transport.Events(), 1)
	s.Equal(&sentry.Request{}, s.transport.Events()[0].Request)
}

func (s *ReportUseSuite) TestAUseOutsideARequestGoesToTheCurrentHub() {
	previous := sentry.CurrentHub().Client()
	sentry.CurrentHub().BindClient(s.client)
	defer sentry.CurrentHub().BindClient(previous)

	s.login(context.Background())

	s.Len(s.transport.Events(), 1)
}

// request is the context sentryhttp gives a request to the login callback: a hub of its own,
// whose scope holds the request with the OAuth code and state in its query.
func (s *ReportUseSuite) request() context.Context {
	hub := sentry.NewHub(s.client, sentry.NewScope())
	hub.Scope().SetRequest(httptest.NewRequest("GET", "/v1/agents/plugins/callback?code=the-code&state=the-state", nil))
	return sentry.SetHubOnContext(context.Background(), hub)
}

func (s *ReportUseSuite) login(ctx context.Context) {
	ReportUse(ctx, s.logger, PathLogin, "customer", "cus_1", "config", "cfg_1", "plugin", "github", "via", ViaPlugins)
}

func (s *ReportUseSuite) line() string {
	return strings.TrimSuffix(s.logged.String(), "\n")
}
