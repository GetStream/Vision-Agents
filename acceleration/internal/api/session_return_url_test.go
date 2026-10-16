//go:build integration

package api

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"time"

	"github.com/gorilla/websocket"
)

// A session's return_url (AI-1384) is where the logins and step-ups its chat asks for go back
// to, as createAuthorization's return_url is for a consent the backend begins. These run in
// ChatLoginsSuite, whose model reaches for crm's echo whenever a person asks.

// appOrigin is appReturnURL's origin, which the launch page takes the handoff from.
const appOrigin = "https://app.example"

// badReturnURL is plain http on a host that is not loopback, which checkReturnURL refuses
// (RFC 8252 sections 7.3 and 8.3).
const badReturnURL = "http://app.example/connectors/done"

func (s *ChatLoginsSuite) TestAChatLoginOfASessionWithAReturnURLIsLaunchedFromItsOriginAndGoesBackThere() {
	connector, grant := s.connector()
	opened := s.client.createSession(s.returningTo(s.session(s.config(connector, grant), nil), appReturnURL))
	events := s.client.opens("/v1/agents/sessions/" + opened.Id + "/events")

	s.ask(opened.Id)
	asked := s.loginOn(events, "")
	page, location, _ := s.loggedIn(asked)

	s.Equal(launchPageFrom(appOrigin), page, "the end user's device chose the page, as a backend would")
	s.Equal(s.backAtTheApp(asked["connection_id"].(string)), location)
	s.Equal(connectorEchoText, s.carryOn(events).ran["result"], "the chat carries on as after any login")
}

func (s *ChatLoginsSuite) TestAStepUpOfASessionWithAReturnURLIsLaunchedFromItsOriginAndGoesBackThere() {
	mine, events, opened := s.steppingUpReturningTo(pointerTo(appReturnURL))

	s.ask(opened)
	page, location, _ := s.loggedIn(s.stepUpOn(events))

	s.Equal(launchPageFrom(appOrigin), page)
	s.Equal(s.backAtTheApp(mine), location)
}

// TestAChatLoginOfASessionWithoutAReturnURLIsLaunchedAndEndsExactlyAsOnBase: the launch page,
// Location and body base b7cea98f served for a chat login (probe-base-author.txt in the
// AI-1384 ledger, 2026-10-16).
func (s *ChatLoginsSuite) TestAChatLoginOfASessionWithoutAReturnURLIsLaunchedAndEndsExactlyAsOnBase() {
	for _, returnURL := range []*string{nil, pointerTo("")} {
		connector, grant := s.connector()
		request := s.session(s.config(connector, grant), nil)
		request.ReturnUrl = returnURL
		opened := s.client.createSession(request)
		events := s.client.opens("/v1/agents/sessions/" + opened.Id + "/events")

		s.ask(opened.Id)
		asked := s.loginOn(events, "")
		page, location, body := s.loggedIn(asked)

		s.assertEndsAsOnBase(asked["connection_id"].(string), page, location, body)
	}
}

// TestAStepUpOfASessionWithoutAReturnURLIsLaunchedAndEndsExactlyAsOnBase: as base b7cea98f
// served a step-up (probe-base-author.txt).
func (s *ChatLoginsSuite) TestAStepUpOfASessionWithoutAReturnURLIsLaunchedAndEndsExactlyAsOnBase() {
	mine, events, opened := s.steppingUp()

	s.ask(opened)
	page, location, body := s.loggedIn(s.stepUpOn(events))

	s.assertEndsAsOnBase(mine, page, location, body)
}

// TestAChatCarriedOnAfterItEndedStillGoesBackToItsReturnURL: the session ended, so the next
// message carries it on from its row (reopenedFromRow), as another process would.
func (s *ChatLoginsSuite) TestAChatCarriedOnAfterItEndedStillGoesBackToItsReturnURL() {
	connector, grant := s.connector()
	opened := s.client.createSession(s.returningTo(s.session(s.config(connector, grant), nil), appReturnURL))
	s.Eventually(func() bool { return s.storedReturnURL(opened.Id) == appReturnURL }, settleFor, 20*time.Millisecond,
		"the row keeps the return_url")
	s.client.stopSession(opened.Id)

	s.ask(opened.Id)

	s.Equal(launchPageFrom(appOrigin), s.launchedFor(connector))
}

// TestAForkOfAnEndedSessionKeepsItsParentsReturnURL: the parent is read from its row, and the
// fork takes the return_url as it takes the parent's connector_bindings. The fork's row is what
// its own chat is carried on from. (A fork of a stored parent runs on the default model rather
// than the config's, so the logging-in model does not ask for a login here; the next test does.)
func (s *ChatLoginsSuite) TestAForkOfAnEndedSessionKeepsItsParentsReturnURL() {
	connector, grant := s.connector()
	opened := s.client.createSession(s.returningTo(s.session(s.config(connector, grant), nil), appReturnURL))
	s.Require().Eventually(func() bool { return s.storedReturnURL(opened.Id) == appReturnURL }, settleFor, 20*time.Millisecond,
		"the parent's row is written")
	s.client.stopSession(opened.Id)

	forked := s.fork(opened.Id, ForkSessionRequest{Messages: pointerTo(false)})

	s.Eventually(func() bool { return s.storedReturnURL(forked.Id) == appReturnURL }, settleFor, 20*time.Millisecond,
		"the fork's row keeps the parent's return_url")
}

// TestAForkNamingAConfigGoesBackToItsParentsReturnURL: a config named on the fork replaces the
// parent's models, not where the caller's consents go back to.
func (s *ChatLoginsSuite) TestAForkNamingAConfigGoesBackToItsParentsReturnURL() {
	connector, grant := s.connector()
	config := s.config(connector, grant)
	opened := s.client.createSession(s.returningTo(s.session(config, nil), appReturnURL))

	forked := s.fork(opened.Id, ForkSessionRequest{Messages: pointerTo(false), ConfigId: &config})
	s.ask(forked.Id)

	s.Equal(launchPageFrom(appOrigin), s.launchedFor(connector))
}

func (s *SessionCreateSuite) TestASessionWithAPlainHTTPReturnURLIsRefusedAndNotCreated() {
	id := s.utils.uuid()
	request := textSession(&id)
	request.ReturnUrl = pointerTo(badReturnURL)

	status, message := s.client.failure(http.MethodPost, "/v1/agents/sessions", request)

	s.Equal(http.StatusBadRequest, status)
	s.Equal("return_url: "+checkReturnURL(badReturnURL).Error(), message)
	s.Equal(http.StatusNotFound, s.client.do(http.MethodGet, "/v1/agents/sessions/"+id, nil, nil))
}

// A start frame is refused the return_url POST /v1/agents/sessions refuses, with the same
// message, code and type.
func (s *SocketSessionSuite) TestASocketSessionWithAPlainHTTPReturnURLIsRefused() {
	sent := map[string]any{"return_url": badReturnURL}
	_, body := s.client.call(http.MethodPost, "/v1/agents/sessions", sent)
	var created ErrorResponse
	s.Require().NoError(json.Unmarshal(body, &created))

	connection := s.client.opens("/v1/agents/socket")
	s.Require().NoError(connection.WriteJSON(map[string]any{"type": "start", "session": sent}))
	refused := s.readFrame(connection)

	s.Equal("error", refused["type"])
	s.Equal("return_url: "+checkReturnURL(badReturnURL).Error(), refused["error"])
	s.Equal(created.Error.Message, refused["error"])
	s.Equal(created.Error.Code, refused["code"])
	s.Equal(string(created.Error.Type), refused["error_type"])
	_, _, err := connection.ReadMessage()
	s.True(websocket.IsCloseError(err, websocket.ClosePolicyViolation), "the socket closes: %v", err)
}

// returningTo is request with returnURL as its return_url.
func (s *ChatLoginsSuite) returningTo(request CreateSessionRequest, returnURL string) CreateSessionRequest {
	request.ReturnUrl = &returnURL
	return request
}

// loggedIn is the consent asked opened and finished in one browser: its launch page with the
// nonce written NONCE, and the callback's Location and body.
func (s *ChatLoginsSuite) loggedIn(asked map[string]any) (string, string, string) {
	b := newBrowser(&s.RouterSuite, s.provider)
	page := launchPageIn(&s.RouterSuite, b.get(asked["launch_url"].(string)))
	finished := b.get(s.consent(b.handOff(s.started(asked))).String())
	defer finished.Body.Close()
	body, err := readAll(finished)
	s.Require().NoError(err)
	return page, finished.Header.Get("Location"), string(body)
}

// assertEndsAsOnBase checks a consent on connection was launched and ended as base b7cea98f's
// did (probe-base-author.txt).
func (s *ChatLoginsSuite) assertEndsAsOnBase(connection, page, location, body string) {
	s.T().Helper()
	s.Equal(baseLaunchPage, page)
	s.Equal("https://dashboard.example/connections?connection_id="+connection+"&status=connected", location)
	s.Equal(`<a href="https://dashboard.example/connections?connection_id=`+connection+`&amp;status=connected">Found</a>.`+"\n\n", body)
}

// backAtTheApp is where the callback sends the browser of a connected consent on connection
// begun with appReturnURL: its own query kept, connection_id and status added.
func (s *ChatLoginsSuite) backAtTheApp(connection string) string {
	return "https://app.example/connectors/done?" + url.Values{
		"connection_id": {connection}, "status": {consentConnected}, "tab": {"tools"}, "from": {"chat"},
	}.Encode()
}

// storedReturnURL is the session row's return_url, empty for NULL or no row yet.
func (s *ChatLoginsSuite) storedReturnURL(id string) string {
	var stored sql.NullString
	err := s.store.ShardDB().QueryRowContext(context.Background(),
		"SELECT return_url FROM agent_sessions WHERE id = ?", id).Scan(&stored)
	if errors.Is(err, sql.ErrNoRows) {
		return ""
	}
	s.Require().NoError(err)
	return stored.String
}

// launchedFor is the launch page, nonce written NONCE, of the one consent the chat began on
// s.client's connection to connector, waiting for it to be begun.
func (s *ChatLoginsSuite) launchedFor(connector string) string {
	var attempt string
	s.Require().Eventually(func() bool {
		return s.store.DB().QueryRowContext(context.Background(),
			`SELECT a.id FROM connector_authorization_attempts a JOIN connector_connections c ON c.id = a.connection_id
			WHERE c.connector_id = ? AND c.owner_id = ?`, connector, s.client.userID).Scan(&attempt) == nil
	}, settleFor, 20*time.Millisecond, "the chat begins a consent")
	return launchPageIn(&s.RouterSuite, newBrowser(&s.RouterSuite, s.provider).get(consentPublicURL+connectorLaunchPath+attempt))
}

// fork is s.client's backend forking id.
func (s *ChatLoginsSuite) fork(id string, request ForkSessionRequest) Session {
	var forked Session
	s.Require().Equal(http.StatusCreated, s.serverClient.actingFor(s.client).do(http.MethodPost,
		"/v1/agents/sessions/"+id+"/fork", request, &forked))
	return forked
}
