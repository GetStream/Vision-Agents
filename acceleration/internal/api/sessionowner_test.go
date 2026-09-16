package api

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// The two apps these tests are run as, with the secret each one's tokens are signed with.
const (
	acmeKey     = "vak_live_0123456789abcdef00000000"
	acmeSecret  = "vas_live_acme"
	otherKey    = "vak_live_fedcba987654321000000000"
	otherSecret = "vas_live_other"
)

// caller is how a request says who it is: the credentials it carries, as headers on a
// request and as a query string on a socket. It is one type for both because a browser
// WebSocket cannot set headers, and a rule that only held for requests would leave the
// socket — the one path a conversation is actually read over — deciding for itself.
type caller struct {
	key      string
	token    string
	authType string
	// claimed is a user id named in a header rather than in a token, which is all an
	// anonymous caller can do.
	claimed string
}

// SessionOwnershipSuite runs the router the way a deployment verifying tokens does, and
// asks whether one person can reach another's conversation.
//
// It is end to end over HTTP rather than against the session manager, because the answer
// has to survive the whole chain: a token is parsed, a kind is decided from it, an owner is
// built out of both, and only then is a session matched. Any link getting it wrong is one
// person reading another's messages, so the test holds the outside of it.
type SessionOwnershipSuite struct {
	suite.Suite

	server *httptest.Server
}

func TestSessionOwnershipSuite(t *testing.T) {
	suite.Run(t, new(SessionOwnershipSuite))
}

func (s *SessionOwnershipSuite) SetupTest() {
	logger := slog.New(slog.DiscardHandler)

	ears := &quietSTT{emitter: stt.NewEmitter(64)}
	transcription := sttrouter.NewRegistry()
	transcription.Register("stub", func(routing.Spec) (stt.STT, error) { return ears, nil })
	transcriber, err := sttrouter.New(sttrouter.Options{
		Config: routableConfig(), Registry: transcription, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(transcriber.Close)

	reasoning := llmrouter.NewRegistry()
	reasoning.Register("stub", func(routing.Spec) (llmrouter.Provider, error) {
		return &scriptedLLM{reply: "Hello."}, nil
	})
	reasoner, err := llmrouter.New(llmrouter.Options{
		Config: routableConfig(), Registry: reasoning, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(reasoner.Close)

	voice := &recordingTTS{emitter: tts.NewEmitter(64)}
	speech := ttsrouter.NewRegistry()
	speech.Register("stub", func(routing.Spec) (tts.TTS, error) { return voice, nil })
	speaker, err := ttsrouter.New(ttsrouter.Options{
		Config: routableConfig(), Registry: speech, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(speaker.Close)

	sessions, err := session.NewManager(session.ManagerOptions{
		LLM:    reasoner,
		STT:    transcriber,
		TTS:    speaker,
		Logger: logger,
		Edge: func(session.Spec, *slog.Logger) (agent.Edge, error) {
			return &silentEdge{inbound: make(chan agent.InboundAudio, 4)}, nil
		},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { sessions.Shutdown() })

	apps := map[string]auth.App{
		acmeKey:  {OrganizationID: "org-1", AppID: "acme", Secret: acmeSecret},
		otherKey: {OrganizationID: "org-2", AppID: "other", Secret: otherSecret},
	}
	authenticator, err := auth.New(auth.APIKey, func(_ context.Context, presented string) (auth.App, error) {
		app, known := apps[presented]
		if !known {
			return auth.App{}, auth.ErrUnauthenticated
		}
		return app, nil
	})
	s.Require().NoError(err)

	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{
			routing.STT: transcriber,
			routing.TTS: speaker,
			routing.LLM: reasoner,
		},
		Sessions: sessions,
		Auth:     authenticator,
		Logger:   logger,
	})
	s.Require().NoError(err)

	s.server = httptest.NewServer(server.Handler())
	s.T().Cleanup(s.server.Close)
}

// signed mints a token for an app, with whichever claims the kind of caller carries.
func (s *SessionOwnershipSuite) signed(secret string, claims jwt.MapClaims) string {
	claims["exp"] = time.Now().Add(time.Hour).Unix()
	token, err := jwt.NewWithClaims(jwt.SigningMethodHS256, claims).SignedString([]byte(secret))
	s.Require().NoError(err)
	return token
}

// authenticated is a signed-in end user: a token naming them, minted by the customer's
// backend and held by their device.
func (s *SessionOwnershipSuite) authenticated(user string) caller {
	return caller{
		key:      acmeKey,
		token:    s.signed(acmeSecret, jwt.MapClaims{"user_id": user}),
		authType: auth.AuthTypeJWT,
	}
}

// guest is an end user Stream issued a temporary account to. The token is as real as an
// authenticated one; the role claim is the whole of the difference.
func (s *SessionOwnershipSuite) guest(user string) caller {
	return caller{
		key:      acmeKey,
		token:    s.signed(acmeSecret, jwt.MapClaims{"user_id": user, "role": "guest"}),
		authType: auth.AuthTypeJWT,
	}
}

// anonymous is a caller with no user of its own: a valid app token that names nobody, plus
// whatever name it feels like claiming.
func (s *SessionOwnershipSuite) anonymous(claimed string) caller {
	return caller{
		key:      acmeKey,
		token:    s.signed(acmeSecret, jwt.MapClaims{}),
		authType: auth.AuthTypeJWT,
		claimed:  claimed,
	}
}

// backend is the customer's own process, which runs the application and so may have every
// session belonging to it.
func (s *SessionOwnershipSuite) backend() caller {
	return caller{
		key:      acmeKey,
		token:    s.signed(acmeSecret, jwt.MapClaims{"server": true}),
		authType: auth.AuthTypeServer,
	}
}

// elsewhere is a signed-in user of a different app, holding a perfectly good token of
// their own.
func (s *SessionOwnershipSuite) elsewhere(user string) caller {
	return caller{
		key:      otherKey,
		token:    s.signed(otherSecret, jwt.MapClaims{"user_id": user}),
		authType: auth.AuthTypeJWT,
	}
}

func (s *SessionOwnershipSuite) send(method, path string, who caller, body any) *http.Response {
	payload := bytes.NewReader(nil)
	if body != nil {
		encoded, err := json.Marshal(body)
		s.Require().NoError(err)
		payload = bytes.NewReader(encoded)
	}

	request, err := http.NewRequest(method, s.server.URL+path, payload)
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(auth.APIKeyHeader, who.key)
	request.Header.Set("Authorization", "Bearer "+who.token)
	request.Header.Set(auth.AuthTypeHeader, who.authType)
	if who.claimed != "" {
		request.Header.Set(auth.UserHeader, who.claimed)
	}

	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	s.T().Cleanup(func() { response.Body.Close() })
	return response
}

// opens starts a conversation in writing, which needs no call and no speech targets.
func (s *SessionOwnershipSuite) opens(who caller) Session {
	target, text := "en-low-latency", true
	response := s.send(http.MethodPost, "/v1/agents/sessions", who,
		CreateSessionRequest{Text: &text, Llm: &target})
	if response.StatusCode != http.StatusCreated {
		var failure Error
		s.Require().NoError(json.NewDecoder(response.Body).Decode(&failure))
		s.T().Fatalf("open a session: %s", failure.Error)
	}

	var created Session
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&created))
	return created
}

// listed is the sessions a caller is told it has.
func (s *SessionOwnershipSuite) listed(who caller) []Session {
	response := s.send(http.MethodGet, "/v1/agents/sessions", who, nil)
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var sessions []Session
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&sessions))
	return sessions
}

// watch opens the events socket the way a browser does, with the credentials in the query
// string, and reports the status if the router refused before the upgrade.
//
// The auth type has no query parameter, so a browser cannot claim to be a backend; a
// backend is a process and sends the header, which is why that one is set here.
func (s *SessionOwnershipSuite) watch(id string, who caller) (*websocket.Conn, int) {
	query := url.Values{auth.APIKeyParam: {who.key}, auth.TokenParam: {who.token}}
	if who.claimed != "" {
		query.Set(auth.UserParam, who.claimed)
	}
	address := "ws" + strings.TrimPrefix(s.server.URL, "http") +
		"/v1/agents/sessions/" + id + "/events?" + query.Encode()

	header := http.Header{}
	if who.authType == auth.AuthTypeServer {
		header.Set(auth.AuthTypeHeader, who.authType)
	}

	connection, response, err := websocket.DefaultDialer.Dial(address, header)
	if err != nil {
		s.Require().ErrorIs(err, websocket.ErrBadHandshake)
		return nil, response.StatusCode
	}
	s.T().Cleanup(func() { connection.Close() })
	return connection, response.StatusCode
}

// reaches asserts that a caller may have a session: it is listed to them, and they may
// read the conversation as it happens, which is what a device has a session for.
func (s *SessionOwnershipSuite) reaches(id string, who caller) {
	connection, status := s.watch(id, who)
	s.Equal(http.StatusSwitchingProtocols, status, "watching the conversation")
	if connection != nil {
		connection.Close()
	}

	var ids []string
	for _, listed := range s.listed(who) {
		ids = append(ids, listed.Id)
	}
	s.Contains(ids, id, "it is one of their sessions")
}

// refused asserts that a caller may not, and is told the session does not exist rather
// than that it does and is somebody else's.
func (s *SessionOwnershipSuite) refused(id string, who caller) {
	_, status := s.watch(id, who)
	s.Equal(http.StatusNotFound, status, "watching the conversation")

	s.Equal(http.StatusNotFound, s.send(http.MethodDelete, "/v1/agents/sessions/"+id, who, nil).StatusCode,
		"closing the session")

	for _, listed := range s.listed(who) {
		s.NotEqual(id, listed.Id, "it must not be listed either")
	}
}

func (s *SessionOwnershipSuite) TestTheUserWhoOpenedASessionReachesIt() {
	alice := s.authenticated("alice")

	opened := s.opens(alice)

	s.reaches(opened.Id, alice)
	s.Equal(http.StatusNoContent,
		s.send(http.MethodDelete, "/v1/agents/sessions/"+opened.Id, alice, nil).StatusCode)
}

func (s *SessionOwnershipSuite) TestWritingIntoASessionIsNotForDevicesEvenTheOwners() {
	// Ownership is not the only thing between a device and a session. Everything the spec
	// does not open is server-side whoever asks, so the owner is refused these too, and
	// with a 403 rather than the 404 a stranger gets: what is being withheld here is the
	// operation and not the session.
	//
	// Reading is not among them. A device may have the conversation it is having, which
	// is why getSession is marked client-accessible where respond and say are not.
	alice := s.authenticated("alice")
	opened := s.opens(alice)

	s.Equal(http.StatusOK,
		s.send(http.MethodGet, "/v1/agents/sessions/"+opened.Id, alice, nil).StatusCode,
		"reading one session by id")
	s.Equal(http.StatusForbidden,
		s.send(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/respond", alice,
			SayRequest{Text: "hello"}).StatusCode,
		"making the agent answer")
	s.Equal(http.StatusForbidden,
		s.send(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/say", alice,
			SayRequest{Text: "hello"}).StatusCode,
		"putting words in the agent's mouth")
}

func (s *SessionOwnershipSuite) TestOneUsersConversationIsOutOfAnothersReach() {
	// The whole point of the token: two signed-in users of the same app, and neither can
	// read what the other said.
	opened := s.opens(s.authenticated("alice"))

	s.refused(opened.Id, s.authenticated("bob"))
}

func (s *SessionOwnershipSuite) TestClaimingAUsersNameWithoutATokenReachesNothing() {
	// A user id is only worth what proved it. An anonymous caller may name itself
	// anything, so if the name were the whole of the identity, typing somebody else's
	// into a header would be enough to read their conversation.
	opened := s.opens(s.authenticated("alice"))

	s.refused(opened.Id, s.anonymous("alice"))
}

func (s *SessionOwnershipSuite) TestAGuestIsNotTheSignedInUserOfTheSameName() {
	// Guest ids are issued per device and a customer's user ids are their own, so the two
	// namespaces can collide. Nothing in a token tells them apart, which is why the
	// declaration does.
	opened := s.opens(s.authenticated("alice"))

	s.refused(opened.Id, s.guest("alice"))
}

func (s *SessionOwnershipSuite) TestAGuestReachesItsOwnConversationAndNobodyElsesReachesIt() {
	guest := s.guest("guest-7")

	opened := s.opens(guest)

	s.reaches(opened.Id, guest)
	s.refused(opened.Id, s.guest("guest-8"))
	s.refused(opened.Id, s.authenticated("guest-7"))
}

func (s *SessionOwnershipSuite) TestAnAnonymousCallerReachesTheSessionItOpenedAndNotAnothers() {
	// Anonymous is a supported way to hold a conversation, not a way to be refused one:
	// the name is unverified, so it is kept apart from every verified user of the same
	// name, and from other anonymous callers by being a different name.
	someone := s.anonymous("device-7")

	opened := s.opens(someone)

	s.reaches(opened.Id, someone)
	s.refused(opened.Id, s.anonymous("device-8"))
}

func (s *SessionOwnershipSuite) TestAnAnonymousCallerThatNamesNobodyIsListedNothing() {
	// It has the session id, so it can go on with the conversation it opened. What it
	// cannot have is a listing, because every caller naming nobody is the same owner and
	// a listing would hand one stranger another's conversation.
	nameless := s.anonymous("")

	opened := s.opens(nameless)

	connection, status := s.watch(opened.Id, nameless)
	s.Equal(http.StatusSwitchingProtocols, status, "the conversation it opened")
	if connection != nil {
		connection.Close()
	}
	s.Empty(s.listed(nameless))
}

func (s *SessionOwnershipSuite) TestAServerTokenInASocketsQueryStringIsNotABackend() {
	// A socket's credentials travel in the query string, where a URL is copied into
	// places a header never goes. If that were enough to be server-side, a leaked server
	// token would be every session the customer has; without the header it is a caller
	// naming nobody, which reaches nothing it did not open.
	opened := s.opens(s.authenticated("alice"))

	backend := s.backend()
	backend.authType = ""

	_, status := s.watch(opened.Id, backend)
	s.Equal(http.StatusNotFound, status)
}

func (s *SessionOwnershipSuite) TestTheCustomersBackendReachesEverySessionItHas() {
	// It runs the application, so cleaning up after a device that went away is its job.
	alice := s.opens(s.authenticated("alice"))
	bob := s.opens(s.authenticated("bob"))

	backend := s.backend()
	s.reaches(alice.Id, backend)
	s.reaches(bob.Id, backend)
	s.Len(s.listed(backend), 2)
	s.Equal(http.StatusNoContent,
		s.send(http.MethodDelete, "/v1/agents/sessions/"+alice.Id, backend, nil).StatusCode)
}

func (s *SessionOwnershipSuite) TestAnotherCustomersUserReachesNothingEvenByTheSameName() {
	opened := s.opens(s.authenticated("alice"))

	s.refused(opened.Id, s.elsewhere("alice"))
}

func (s *SessionOwnershipSuite) TestASessionWithNoTokenAtAllIsRefusedBeforeItIsOwned() {
	// The ownership rules are about telling verified callers apart. A caller with no
	// credential never gets far enough to have any.
	request, err := http.NewRequest(http.MethodGet, s.server.URL+"/v1/agents/sessions", nil)
	s.Require().NoError(err)

	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()

	s.Equal(http.StatusUnauthorized, response.StatusCode)
}
