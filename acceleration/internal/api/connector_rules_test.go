package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ConnectorRulesSuite covers the rules about connectors that are decided before anything
// is read or written: what a binding may be called, where a finished login sends the
// browser, which origins are trusted with one, and what a fork inherits. None of them
// needs a database, so none of them waits for one.
type ConnectorRulesSuite struct {
	suite.Suite
}

func TestConnectorRulesSuite(t *testing.T) {
	suite.Run(t, new(ConnectorRulesSuite))
}

func (s *ConnectorRulesSuite) TestABindingNameCannotBreakToolNamespacing() {
	// A tool is named binding__tool, so a binding with a double underscore in its name
	// would make the split ambiguous.
	bindings := []AgentConnectorBinding{fixedBinding("slack__workspace", "slack", strings.Repeat("a", 64))}

	complaint, ok := connectorBindingsComplaint(&bindings)

	s.False(ok)
	s.Contains(complaint, "connector binding names")
}

func (s *ConnectorRulesSuite) TestAToolGrantNeedsAValidSchemaDigest() {
	bindings := []AgentConnectorBinding{fixedBinding("crm", "salesforce", "")}

	complaint, ok := connectorBindingsComplaint(&bindings)

	s.False(ok)
	s.Contains(complaint, "schema_digest")
}

func (s *ConnectorRulesSuite) TestStoringABindingKeepsTheSchemaDigestOfEveryTool() {
	digest := strings.Repeat("b", 64)

	stored := connectorBindingsFromAPI([]AgentConnectorBinding{fixedBinding("crm", "salesforce", digest)})
	s.Require().Len(stored, 1)
	s.Equal(digest, stored[0].Tools[0].SchemaDigest)

	read := connectorBindingsToAPI(stored)
	s.Require().Len(read, 1)
	s.Equal(digest, read[0].Tools[0].SchemaDigest)
}

func (s *ConnectorRulesSuite) TestAFinishedLoginReturnsToTheConfiguredDashboardPage() {
	// The dashboard's own selection is kept, and only what the login decided is replaced.
	server := &Server{dashboardURL: "https://dashboard.example/organization/123/connections/?config_id=agent&status=old&connection_id=old"}

	for _, status := range []string{"connected", "failed"} {
		recorder := httptest.NewRecorder()
		server.redirectAfterConnectorLogin(recorder, "connection&value", status)

		s.Equal(http.StatusFound, recorder.Code)
		s.Equal("https://dashboard.example/organization/123/connections/?config_id=agent&connection_id=connection%26value&status="+status,
			recorder.Header().Get("Location"))
	}
}

func (s *ConnectorRulesSuite) TestADashboardURLWithoutAnOriginIsRefused() {
	_, err := NewServer(Options{DashboardURL: "/connections"})

	s.ErrorContains(err, "invalid dashboard URL")
}

func (s *ConnectorRulesSuite) TestADashboardURLThatIsNotAWebPageIsRefused() {
	_, err := NewServer(Options{DashboardURL: "javascript:alert(1)"})

	s.ErrorContains(err, "invalid dashboard URL")
}

func (s *ConnectorRulesSuite) TestADashboardURLCarryingCredentialsIsRefused() {
	_, err := NewServer(Options{DashboardURL: "https://user:password@dashboard.example"})

	s.ErrorContains(err, "invalid dashboard URL")
}

func (s *ConnectorRulesSuite) TestAnOriginIsSerializedTheWayABrowserSendsIt() {
	// A login handoff compares the Origin a browser sent with the one configured, so the
	// configured one has to be spelled the browser's way.
	s.Equal("https://router.example", s.origin("https://ROUTER.example:443/base"))
	s.Equal("http://router.example", s.origin("http://router.example:80"))
	s.Equal("https://router.example:8443", s.origin("https://router.example:8443"))
	s.Equal("https://[2001:db8::1]", s.origin("https://[2001:db8::1]:443"))
}

func (s *ConnectorRulesSuite) TestANamedOriginMayMakeCredentialedBrowserRequests() {
	recorder := s.fromOrigin([]string{"https://dash.example"}, "https://dash.example")

	s.Equal("https://dash.example", recorder.Header().Get("Access-Control-Allow-Origin"))
	s.Equal("true", recorder.Header().Get("Access-Control-Allow-Credentials"))
}

func (s *ConnectorRulesSuite) TestAWildcardOriginDoesNotAllowCredentialedBrowserRequests() {
	// Any page on the internet would otherwise read the API with the visitor's cookies,
	// which is what a connector login's cookie is.
	recorder := s.fromOrigin([]string{"*"}, "https://dash.example")

	s.Equal("https://dash.example", recorder.Header().Get("Access-Control-Allow-Origin"))
	s.Empty(recorder.Header().Get("Access-Control-Allow-Credentials"))
}

func (s *ConnectorRulesSuite) TestForkingToAnotherConfigUsesItsGrantsWithoutTransferringAccountSelections() {
	parent := store.AgentSession{
		ID: "session-1",
		ConnectorSelections: []store.SessionConnectorSelection{{
			Name: "crm", ConnectionID: "old-account",
		}},
	}
	config := &store.AgentConfig{
		ID: "new-config",
		Connectors: []store.ConnectorBinding{{
			Name: "calendar", ConnectorID: "calendly",
			Connection: store.ConnectionBinding{Type: "fixed", ConnectionID: "new-account"},
			Tools:      []store.ToolGrant{{Name: "list_events", SchemaDigest: "current-schema"}},
		}},
	}

	spec, err := forkSpec(session.Found{Stored: &parent}, ForkSessionRequest{}, config)

	s.Require().NoError(err)
	s.Equal(config.Connectors, spec.ConnectorBindings)
	s.Empty(spec.ConnectorSelections,
		"a different agent gets its own grants and cannot inherit the parent's choice of account")
}

func (s *ConnectorRulesSuite) origin(raw string) string {
	serialized, err := connectorOrigin(raw)
	s.Require().NoError(err)
	return serialized
}

// fromOrigin is how a router allowing allowed answers a browser at origin.
func (s *ConnectorRulesSuite) fromOrigin(allowed []string, origin string) *httptest.ResponseRecorder {
	answered := withCORS(allowed, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/health", nil)
	request.Header.Set("Origin", origin)
	answered.ServeHTTP(recorder, request)
	return recorder
}

// fixedBinding binds connectorID as name to one account, granting a tool whose schema
// hashes to digest.
func fixedBinding(name, connectorID, digest string) AgentConnectorBinding {
	connectionID := "connection-1"
	return AgentConnectorBinding{
		Name:        name,
		ConnectorId: connectorID,
		Connection: AgentConnectorSelection{
			Type:         AgentConnectorSelectionTypeFixed,
			ConnectionId: &connectionID,
		},
		Tools: []ConnectorToolGrant{{Name: "lookup_contact", SchemaDigest: digest}},
	}
}
