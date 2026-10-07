//go:build integration

package api

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"testing"
	"testing/fstest"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/verifiers/hmacheader"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// eventsSecret is the operator's signing secret for both connectors in ConnectorEventsSuite.
// Made up.
const eventsSecret = "synthetic-operator-signing-secret"

// eventsBot is a built-in the suite seeds beside the real ones: a Slack-shaped bot whose
// channel block reads messages as well as signals, which the built-in Slack connector does
// not. Its id and content are fixed, so seeding it again in a later run changes nothing.
const eventsBot = `id: events_test_bot
revision: 1
name: Events test bot
schemes: [oauth2_code]
client:
  registration: [operator]
  env: EVENTS_TEST_BOT
capture:
  - name: team_id
    from: token_response
    path: $.team.id
identity: [team_id]
channel:
  verifier:
    kind: hmac_header
    secret: operator
    header: X-Slack-Signature
    algorithm: sha256
    encoding: hex
    prefix: v0=
    signed: "v0:{timestamp}:{body}"
    timestamp_header: X-Slack-Request-Timestamp
    max_age: 5m
  format: json
  challenge: $.challenge
  messages:
    match:
      $.event.type: message
    skip_if_present: [$.event.bot_id]
    provider_unit_id: $.team_id
    thread_key:
      - name: channel
        path: $.event.channel
      - name: thread_ts
        path: $.event.thread_ts
        fallback: $.event.ts
    author_id: $.event.user
    provider_message_id: $.event.ts
    text: $.event.text
  reply:
    url: https://slack.com/api/chat.postMessage
    body:
      channel: "{thread.channel}"
      thread_ts: "{thread.thread_ts}"
      text: "{text}"
  signals:
    - kind: uninstalled
      match:
        $.event.type: app_uninstalled
      identity:
        team_id: $.team_id
`

// ConnectorEventsSuite is POST /v1/agents/connectors/events/{connector_id} against the
// built-in Slack manifest: a provider delivers an event with no API credential, signed with
// the operator's secret, and the connections of the account it names move, through the
// router's own resolver, so the next resolve fails before reaching any scheme.
type ConnectorEventsSuite struct {
	RouterSuite
	delivered *deliveries
}

func TestConnectorEventsSuite(t *testing.T) {
	runSuite(t, new(ConnectorEventsSuite))
}

// SetupSuite registers hmac_header and a scheme named oauth2_code, which the built-ins name;
// it connects nothing, so a resolve that reaches it fails with its own error, not
// ErrNotConnected. The operator's signing secrets are in the suite's environment.
func (s *ConnectorEventsSuite) SetupSuite() {
	verifier := hmacheader.New()
	s.connectors = core.Registry{
		Schemes:   map[string]core.Scheme{oauth2code.Name: namedScheme(oauth2code.Name)},
		Verifiers: map[string]core.Verifier{verifier.Name(): verifier},
	}
	environment := map[string]string{
		"SLACK_MCP_SIGNING_SECRET":           eventsSecret,
		"EVENTS_TEST_BOT_MCP_SIGNING_SECRET": eventsSecret,
	}
	s.eventSecrets = ConnectorEventSecrets(func(name string) string { return environment[name] })
	s.delivered = &deliveries{}
	s.bridge = s.delivered
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(),
		fstest.MapFS{"events_test_bot.yaml": {Data: []byte(eventsBot)}}))
}

func (s *ConnectorEventsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ConnectorEventsSuite) TestAValidRevocationMovesTheUsersConnectionAndTheNextResolveFailsFast() {
	team := s.team()
	alice := s.connected("slack", team, "UALICE")
	bob := s.connected("slack", team, "UBOB")
	_, err := s.resolver.Resolve(context.Background(), alice, core.CredentialRequest{})
	s.Require().ErrorContains(err, "a named scheme does not connect", "a connected row goes to its scheme")

	status, _, _ := s.deliver("slack", s.signed(revocation(team, "UALICE"), time.Now()))

	s.Equal(http.StatusOK, status)
	s.Equal(store.ConnectionNeedsReauthorization, s.status(alice))
	s.Equal(store.ConnectionConnected, s.status(bob), "only the user Slack named")
	_, err = s.resolver.Resolve(context.Background(), alice, core.CredentialRequest{})
	s.ErrorIs(err, resolver.ErrNotConnected, "the row says so before any scheme is asked")
}

func (s *ConnectorEventsSuite) TestAnUnsignedRevocationIsRefusedAndChangesNothing() {
	team := s.team()
	alice := s.connected("slack", team, "UALICE")
	request := s.signed(revocation(team, "UALICE"), time.Now())
	request.Header.Del("X-Slack-Signature")

	status, body, _ := s.deliver("slack", request)

	s.Equal(http.StatusUnauthorized, status)
	s.Contains(body, `"code":"unauthenticated"`)
	s.Equal(store.ConnectionConnected, s.status(alice))
}

// Slack: refuse a request whose timestamp is «more than five minutes from local time»
// (https://docs.slack.dev/authentication/verifying-requests-from-slack).
func (s *ConnectorEventsSuite) TestAStaleRevocationIsRefusedAndChangesNothing() {
	team := s.team()
	alice := s.connected("slack", team, "UALICE")

	status, _, _ := s.deliver("slack", s.signed(revocation(team, "UALICE"), time.Now().Add(-6*time.Minute)))

	s.Equal(http.StatusUnauthorized, status)
	s.Equal(store.ConnectionConnected, s.status(alice))
}

func (s *ConnectorEventsSuite) TestARevocationSignedWithAnotherSecretIsRefused() {
	team := s.team()
	alice := s.connected("slack", team, "UALICE")

	status, _, _ := s.deliver("slack", s.signedWith(revocation(team, "UALICE"), time.Now(), "another-secret"))

	s.Equal(http.StatusUnauthorized, status)
	s.Equal(store.ConnectionConnected, s.status(alice))
}

// One operator app serves every customer, so its uninstall ends every app's connections in
// that workspace, and none in another.
func (s *ConnectorEventsSuite) TestAnUninstallMovesEveryConnectionOfTheWorkspaceInEveryApp() {
	team, elsewhere := s.team(), s.team()
	alice := s.connected("slack", team, "UALICE")
	untouched := s.connected("slack", elsewhere, "UALICE")
	s.useApp(s.data.createApp())
	bob := s.connected("slack", team, "UBOB")

	status, _, _ := s.deliver("slack", s.signed(uninstall(team), time.Now()))

	s.Equal(http.StatusOK, status)
	s.Equal(store.ConnectionNeedsReauthorization, s.status(alice))
	s.Equal(store.ConnectionNeedsReauthorization, s.status(bob))
	s.Equal(store.ConnectionConnected, s.status(untouched))
}

// A connection deleted between the lookup and its revocation has no grant left to end: the
// delivery still revokes the rest and answers 200, so Slack does not deliver it again. The
// later connection's lock is held until the earlier one has moved, which proves the lookup
// listed both, and it is deleted before the lock is let go.
func (s *ConnectorEventsSuite) TestAConnectionDeletedDuringAnUninstallCountsAsRevoked() {
	team := s.team()
	first, last := s.connected("slack", team, "UALICE"), s.connected("slack", team, "UBOB")
	if last.ConnectionID < first.ConnectionID {
		first, last = last, first
	}
	release := s.hold(last)
	answered := make(chan int, 1)
	go func() {
		status, _, _ := s.deliver("slack", s.signed(uninstall(team), time.Now()))
		answered <- status
	}()

	s.Require().Eventually(func() bool {
		return s.status(first) == store.ConnectionNeedsReauthorization
	}, 10*time.Second, 20*time.Millisecond, "the loop reached the held connection")
	s.Require().NoError(s.store.DeleteConnectorConnection(context.Background(), last.CustomerID, last.ConnectionID))
	release()

	select {
	case status := <-answered:
		s.Equal(http.StatusOK, status)
	case <-time.After(10 * time.Second):
		s.Fail("the delivery was not answered")
	}
}

func (s *ConnectorEventsSuite) TestAURLVerificationIsAnsweredWithItsChallenge() {
	body := []byte(`{"token":"synthetic","challenge":"synthetic-challenge-value","type":"url_verification"}`)

	status, answer, header := s.deliver("slack", s.signed(body, time.Now()))

	s.Equal(http.StatusOK, status)
	s.Equal("synthetic-challenge-value", answer)
	s.Equal("text/plain; charset=utf-8", header.Get("Content-Type"))
}

func (s *ConnectorEventsSuite) TestAnUnsignedURLVerificationIsNotAnswered() {
	request := s.signed([]byte(`{"challenge":"synthetic-challenge-value","type":"url_verification"}`), time.Now())
	request.Header.Set("X-Slack-Signature", "v0="+strings.Repeat("0", sha256.Size*2))

	status, answer, _ := s.deliver("slack", request)

	s.Equal(http.StatusUnauthorized, status)
	s.NotContains(answer, "synthetic-challenge-value")
}

// The bridge registers in the server's options; the handler hands it what the verifier read.
func (s *ConnectorEventsSuite) TestASignedMessageIsHandedToTheBridge() {
	team := s.team()
	body := []byte(`{"team_id":"` + team + `","type":"event_callback","event":{"type":"message","channel":"C0000CHAN","user":"U0000USER","text":"hello","ts":"1759740000.000200"}}`)

	status, _, _ := s.deliver("events_test_bot", s.signed(body, time.Now()))

	s.Equal(http.StatusOK, status)
	s.Equal([]core.InboundMessage{{
		ConnectorID:       "events_test_bot",
		ProviderUnitID:    team,
		ThreadKey:         "C0000CHAN:1759740000.000200",
		AuthorID:          "U0000USER",
		Text:              "hello",
		ProviderMessageID: "1759740000.000200",
		Raw:               body,
	}}, s.delivered.of(team))
}

func (s *ConnectorEventsSuite) TestAnUnsignedMessageNeverReachesTheBridge() {
	team := s.team()
	body := []byte(`{"team_id":"` + team + `","type":"event_callback","event":{"type":"message","channel":"C0000CHAN","user":"U0000USER","text":"hello","ts":"1759740000.000200"}}`)

	status, _, _ := s.deliver("events_test_bot", s.signedWith(body, time.Now(), "another-secret"))

	s.Equal(http.StatusUnauthorized, status)
	s.Empty(s.delivered.of(team))
}

func (s *ConnectorEventsSuite) TestAConnectorThatIsNotABuiltInTakesNoEvents() {
	for _, id := range []string{"no_such_connector", "custom_" + strings.ReplaceAll(s.utils.uuid(), "-", "")} {
		status, body, _ := s.deliver(id, s.signed(uninstall(s.team()), time.Now()))
		s.Equal(http.StatusNotFound, status, id)
		s.Contains(body, "this connector takes no events here", id)
	}
}

// Linear's manifest has no channel block, so nothing could verify an event for it.
func (s *ConnectorEventsSuite) TestABuiltInWithoutAChannelTakesNoEvents() {
	status, _, _ := s.deliver("linear", s.signed(uninstall(s.team()), time.Now()))

	s.Equal(http.StatusNotFound, status)
}

// 256 KiB is the cap the router's other provider webhooks use (channels.MaxDeliveryBytes).
func (s *ConnectorEventsSuite) TestABodyOverTheCapIsRefusedBeforeItIsVerified() {
	body := bytes.Repeat([]byte(" "), maxConnectorEventBytes+1)

	status, answer, _ := s.deliver("slack", s.signed(body, time.Now()))

	s.Equal(http.StatusRequestEntityTooLarge, status)
	s.Contains(answer, "an event is at most 256 KiB")
}

// team is a Slack team id of the test's own, so connections other tests and suites made in
// the shared database never match its events.
func (s *ConnectorEventsSuite) team() string {
	return "T" + strings.ToUpper(strings.ReplaceAll(s.utils.uuid(), "-", ""))
}

// connected is a connection of the test's app to connector for user in team, made through
// the API and then given credentials and the account as a consent leaves them.
func (s *ConnectorEventsSuite) connected(connector, team, user string) core.ConnectionRef {
	var created Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections", appOwned(connector), &created))
	ref := core.ConnectionRef{CustomerID: s.customerID(), ConnectionID: created.ID}
	credentials, err := pgsealed.New(s.store, s.sealer)
	s.Require().NoError(err)
	s.Require().NoError(credentials.Update(context.Background(), ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.Credentials = core.StoredCredentials{Scheme: oauth2code.Name, Version: 1, Payload: []byte(`{}`)}
		state.Status = store.ConnectionConnected
		state.AccountID = team + ":" + user
		state.Metadata = map[string]string{"team_id": team, "user_id": user}
		return true, nil
	}))
	return ref
}

// hold takes ref's credential lock and keeps it until the returned release is called, as a
// refresh in flight on another router would.
func (s *ConnectorEventsSuite) hold(ref core.ConnectionRef) (release func()) {
	credentials, err := pgsealed.New(s.store, s.sealer)
	s.Require().NoError(err)
	locked, released, done := make(chan struct{}), make(chan struct{}), make(chan error, 1)
	go func() {
		done <- credentials.Update(context.Background(), ref, func(*core.CredentialState, func() error) (bool, error) {
			close(locked)
			<-released
			return false, nil
		})
	}()
	<-locked
	var once sync.Once
	release = func() {
		once.Do(func() {
			close(released)
			<-done
		})
	}
	s.T().Cleanup(release)
	return release
}

func (s *ConnectorEventsSuite) status(ref core.ConnectionRef) string {
	connection, err := s.store.ConnectorConnection(context.Background(), ref.CustomerID, ref.ConnectionID)
	s.Require().NoError(err)
	return connection.Status
}

// deliver sends request to connector's events route, with no API credential, as Slack does.
func (s *ConnectorEventsSuite) deliver(connector string, request *http.Request) (int, string, http.Header) {
	target, err := http.NewRequest(http.MethodPost, s.server.URL+connectorEventsPath+connector, request.Body)
	s.Require().NoError(err)
	target.Header = request.Header
	response, err := http.DefaultClient.Do(target)
	s.Require().NoError(err)
	defer response.Body.Close()
	answer, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	return response.StatusCode, string(answer), response.Header
}

// signed is body as Slack sends it at at, signed with the operator's secret.
func (s *ConnectorEventsSuite) signed(body []byte, at time.Time) *http.Request {
	return s.signedWith(body, at, eventsSecret)
}

// signedWith signs body the way Slack's page says: v0= and the hex HMAC-SHA256 of
// v0:{timestamp}:{body}.
func (s *ConnectorEventsSuite) signedWith(body []byte, at time.Time, secret string) *http.Request {
	timestamp := strconv.FormatInt(at.Unix(), 10)
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte("v0:" + timestamp + ":"))
	mac.Write(body)
	request, err := http.NewRequest(http.MethodPost, "/", bytes.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Slack-Request-Timestamp", timestamp)
	request.Header.Set("X-Slack-Signature", "v0="+hex.EncodeToString(mac.Sum(nil)))
	return request
}

// revocation is Slack's tokens_revoked for users in team
// (https://docs.slack.dev/reference/events/tokens_revoked).
func revocation(team string, users ...string) []byte {
	return []byte(`{"token":"synthetic","team_id":"` + team + `","api_app_id":"A0000APP","event":{"type":"tokens_revoked","tokens":{"oauth":["` +
		strings.Join(users, `","`) + `"]}},"type":"event_callback","event_id":"Ev0000REVOKED","event_time":1759740000}`)
}

// uninstall is Slack's app_uninstalled for team
// (https://docs.slack.dev/reference/events/app_uninstalled).
func uninstall(team string) []byte {
	return []byte(`{"token":"synthetic","team_id":"` + team + `","api_app_id":"A0000APP","event":{"type":"app_uninstalled"},"type":"event_callback","event_id":"Ev0000UNINSTALLED","event_time":1759740000}`)
}

// deliveries is a channel bridge that keeps what it was handed, the state a test reads in
// place of the thread channels the real bridge (T57) writes.
type deliveries struct {
	mu       sync.Mutex
	messages []core.InboundMessage
}

func (d *deliveries) Deliver(_ context.Context, _ store.ConnectorOAuthClient, messages []core.InboundMessage) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.messages = append(d.messages, messages...)
	return nil
}

// of is the messages delivered for one provider unit.
func (d *deliveries) of(unit string) []core.InboundMessage {
	d.mu.Lock()
	defer d.mu.Unlock()
	var found []core.InboundMessage
	for _, message := range d.messages {
		if message.ProviderUnitID == unit {
			found = append(found, message)
		}
	}
	return found
}
