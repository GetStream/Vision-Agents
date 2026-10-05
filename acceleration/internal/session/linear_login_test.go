//go:build integration

package session

import (
	"context"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// LinearLoginSuite is the login an end user is asked for, end to end against the real
// Linear: its own discovery documents, its own dynamic client registration, its own
// consent page. UserPluginsSuite covers the same path against a provider we wrote, which
// cannot tell us Linear's metadata is where the catalog says it is.
//
// Linear is the plugin to do this with because it registers a client for whoever asks, so
// the whole flow runs with nothing configured. A vendor needing a client id could only be
// tested this far with a secret in the environment.
type LinearLoginSuite struct {
	suite.Suite
	store  *store.Store
	linear plugins.Plugin
	runner *userPluginRunner
	auth   *plugins.Auth
}

func TestLinearLoginSuite(t *testing.T) {
	suite.Run(t, new(LinearLoginSuite))
}

func (s *LinearLoginSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN is not set")
	}
	linear, ok := plugins.Lookup("linear")
	s.Require().True(ok, "the catalog has no linear to log into")
	s.linear = linear
	s.skipWithoutLinear()

	db, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(context.Background()))
	s.store = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })
}

func (s *LinearLoginSuite) SetupTest() {
	s.auth = &plugins.Auth{PublicURL: "https://router.example"}
	s.runner = &userPluginRunner{
		customerID: "customer-" + uuid.NewString(),
		configID:   uuid.NewString(),
		userID:     "alice",
		named:      map[string]plugins.Plugin{"linear": s.linear},
		db:         s.store,
		auth:       s.auth,
		logger:     slog.New(slog.DiscardHandler),
		open:       map[string]*plugins.Runtime{},
	}
	s.T().Cleanup(s.runner.Close)
}

func (s *LinearLoginSuite) TestAskingAboutLinearBeforeConnectingItAsksForTheLogin() {
	found := s.asked(s.run("linear__list_tools", ""))

	s.Equal(plugins.AuthorizationType, found.Type)
	s.Equal("linear", found.PluginID)
	s.Equal("Connect Linear", found.Title)
	s.Equal("mcp.linear.app", s.authorizeURL(found).Host)
}

// The three fields a Chat client with no renderer for this type draws a card from. They
// are checked against the catalog when the attachment is read back, so a card drawn from
// anything else would be dropped on the way home.
func (s *LinearLoginSuite) TestTheCardCarriesWhatIsNeededToDrawTheButton() {
	found := s.asked(s.run("linear__list_tools", ""))

	s.Equal(s.linear.Description, found.Text)
	s.Equal("https://router.example/v1/agents/plugins/linear/logo", found.ThumbURL)
	s.Equal(found.AuthorizeURL, found.TitleLink)
	s.True(plugins.ValidAuthorization(found), "a card the conversation would refuse to keep")
}

func (s *LinearLoginSuite) TestLinearRegistersAClientForThisDeployment() {
	query := s.authorizeURL(s.asked(s.run("linear__list_tools", ""))).Query()

	s.NotEmpty(query.Get("client_id"), "Linear advertises dynamic registration, so it minted one")
	s.NotEmpty(query.Get("code_challenge"))
	s.Equal("S256", query.Get("code_challenge_method"))
	s.Equal("code", query.Get("response_type"))
	s.Equal("https://router.example"+plugins.CallbackPath, query.Get("redirect_uri"))
	s.Equal([]string{"read", "write"}, s.linear.Scopes)
	s.Equal("read write", query.Get("scope"))
}

// What the callback needs to turn the code Linear sends back into a token. Without the
// verifier and the token endpoint the login cannot be finished, and the button is a
// round trip to nowhere.
func (s *LinearLoginSuite) TestThePendingLoginHoldsWhatTheCallbackWillNeed() {
	found := s.asked(s.run("linear__list_tools", ""))

	conn, err := s.store.UserPluginConnection(context.Background(),
		s.runner.customerID, s.runner.configID, "alice", "linear")
	s.Require().NoError(err)
	s.Equal(store.PluginPending, conn.Status)
	s.NotEmpty(conn.CodeVerifier)
	s.NotEmpty(conn.ClientID)
	s.Equal(conn.OAuthState, s.authorizeURL(found).Query().Get("state"),
		"the callback finds this login by the state Linear hands back")
	s.Contains(conn.TokenEndpoint, "linear.app")
}

func (s *LinearLoginSuite) TestTheButtonOpensLinearsOwnConsentPage() {
	found := s.asked(s.run("linear__list_tools", ""))

	status, body := s.get(found.AuthorizeURL)

	s.Equal(http.StatusOK, status)
	s.Contains(body, "Authorization Request",
		"Linear answered something other than the consent page: %.200s", body)
}

// The read-only endpoint is a resource of its own at Linear that accepts only read, so a
// login for it that asked for write, or named the other resource, would be refused.
func (s *LinearLoginSuite) TestAReadonlyLoginOpensLinearsConsentPageForTheReadOnlyServer() {
	readonly, err := ConfiguredPlugin("linear", []store.PluginOptions{{Plugin: "linear", Readonly: true}})
	s.Require().NoError(err)
	s.runner.named["linear"] = readonly

	found := s.asked(s.run("linear__list_tools", ""))
	query := s.authorizeURL(found).Query()
	s.Equal("https://mcp.linear.app/mcp/readonly", query.Get("resource"))
	s.Equal("read", query.Get("scope"))

	status, body := s.get(found.AuthorizeURL)
	s.Equal(http.StatusOK, status)
	s.Contains(body, "Authorization Request",
		"Linear answered something other than the consent page: %.200s", body)
}

func (s *LinearLoginSuite) TestTheLogoTheCardPointsAtIsOneWeServe() {
	found := s.asked(s.run("linear__list_tools", ""))

	// The host is this deployment, which is not running in a unit test, so what is
	// checked is that the path names a logo the router has to serve.
	raw, ok := plugins.Logo("linear")
	s.Require().True(ok)
	s.Contains(string(raw), "<svg")
	s.Equal(plugins.LogoPath("linear"), s.authorizeOrLogo(found.ThumbURL).Path)
}

// A token Linear does not know is a login that has to be asked for again, rather than a
// tool that fails and a conversation that cannot say why.
func (s *LinearLoginSuite) TestATokenLinearRefusesIsAskedForAgain() {
	s.Require().NoError(s.store.UpsertPluginConnection(context.Background(), &store.PluginConnection{
		CustomerID: s.runner.customerID, ConfigID: s.runner.configID, PluginID: "linear",
		UserID: "alice", Status: store.PluginConnected, AccessToken: "not-a-linear-token",
	}))

	found := s.asked(s.run("linear__list_tools", ""))

	s.Equal("Connect Linear", found.Title)
	conn, err := s.store.UserPluginConnection(context.Background(),
		s.runner.customerID, s.runner.configID, "alice", "linear")
	s.Require().NoError(err)
	s.Equal(store.PluginPending, conn.Status)
	s.Empty(conn.AccessToken, "the token Linear refused is not kept")
}

// run calls one of the plugin's tools the way the model would.
func (s *LinearLoginSuite) run(name, arguments string) string {
	parts, err := s.runner.Run(context.Background(), llm.ToolCall{
		ID: uuid.NewString(), Name: name, Arguments: arguments,
	})
	s.Require().NoError(err)
	s.Require().Len(parts, 1)
	return parts[0].Text
}

// asked reads the login out of a tool result, failing the test when the tool answered
// something else.
func (s *LinearLoginSuite) asked(result string) plugins.Authorization {
	found, ok := plugins.RequestedAuthorization("linear__list_tools", result)
	s.Require().True(ok, "the tool did not ask for a login: %s", result)
	return found
}

func (s *LinearLoginSuite) authorizeURL(found plugins.Authorization) *url.URL {
	parsed, err := url.Parse(found.AuthorizeURL)
	s.Require().NoError(err)
	s.Equal("https", parsed.Scheme)
	return parsed
}

func (s *LinearLoginSuite) authorizeOrLogo(raw string) *url.URL {
	parsed, err := url.Parse(raw)
	s.Require().NoError(err)
	return parsed
}

func (s *LinearLoginSuite) get(raw string) (int, string) {
	client := &http.Client{Timeout: 20 * time.Second}
	request, err := http.NewRequestWithContext(s.T().Context(), http.MethodGet, raw, nil)
	s.Require().NoError(err)
	response, err := client.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	body := make([]byte, 4096)
	n, _ := response.Body.Read(body)
	return response.StatusCode, string(body[:n])
}

// skipWithoutLinear leaves the suite alone when Linear is not reachable, so a machine with
// no network does not read as a broken catalog.
func (s *LinearLoginSuite) skipWithoutLinear() {
	client := &http.Client{Timeout: 10 * time.Second}
	request, err := http.NewRequestWithContext(context.Background(), http.MethodGet,
		"https://mcp.linear.app/.well-known/oauth-protected-resource/mcp", nil)
	s.Require().NoError(err)
	response, err := client.Do(request)
	if err != nil {
		s.T().Skipf("Linear is not reachable: %v", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		s.T().Skipf("Linear answered %d for its protected resource metadata", response.StatusCode)
	}
}
