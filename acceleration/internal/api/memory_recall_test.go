//go:build integration

package api

import (
	"context"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/memory/mem0"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// extractionGrace is how long mem0 is given to turn a conversation into memories, which it
// does on its own side after the conversation has moved on.
const extractionGrace = time.Minute

// recallEvery is how often a test waiting on mem0 opens another session to ask.
const recallEvery = 3 * time.Second

// MemorySuite runs the router against mem0 rather than a stand-in: whose memories a session
// reads is decided by the filters mem0 applies, so a stand-in would only prove itself. The
// model is the echo stub, so what a session recalled is what it answers with.
type MemorySuite struct {
	RouterSuite

	// known is a user of the standard app who told the agent they are allergic to peanuts,
	// under the label company=acme. Tests read what is known about them and add nothing.
	known *testClient
}

func TestMemorySuite(t *testing.T) {
	runSuite(t, new(MemorySuite))
}

func (s *MemorySuite) SetupSuite() {
	if os.Getenv("MEM0_API_KEY") == "" {
		s.T().Skip("MEM0_API_KEY must be set")
	}
	memories, err := mem0.New(mem0.Options{})
	s.Require().NoError(err)
	s.memoryStore = memories
	s.RouterSuite.SetupSuite()

	s.useFixture("standard")
	s.known = s.data.createUser()
	s.tell(s.known, remembering(s.known.userID, acme()), "I am allergic to peanuts.")
	s.learns(s.known, remembering(s.known.userID, acme()), "peanut")
}

func (s *MemorySuite) SetupTest() {
	s.useFixture("standard")
}

func (s *MemorySuite) TestWhatAUserToldThreeSessionsIsKnownInTheFourth() {
	owner := s.data.createUser()

	s.tell(owner, remembering(owner.userID, nil), "I am allergic to peanuts.")
	s.tell(owner, remembering(owner.userID, nil), "I live in Amsterdam.")
	s.tell(owner, remembering(owner.userID, nil), "My dog is called Rex.")

	s.learns(owner, remembering(owner.userID, nil), "peanut", "amsterdam", "rex")
}

func (s *MemorySuite) TestNothingIsRememberedAboutAnAnonymousCaller() {
	name := s.anonymousClient.userID

	// Refusing the session and keeping nothing from it are both an answer.
	opened, admitted := s.tries(s.anonymousClient, remembering(name, nil))
	if !admitted {
		return
	}
	s.Require().NotEmpty(s.ask(s.anonymousClient, opened, "I am allergic to peanuts."),
		"an exchange without an answer is not remembered")
	s.anonymousClient.stopSession(opened.Id)

	backend := s.serverClient.actingFor(s.anonymousClient)
	for deadline := time.Now().Add(extractionGrace); time.Now().Before(deadline); time.Sleep(recallEvery) {
		s.Require().NotContains(s.recalled(backend, remembering(name, nil)), "peanut")
	}
}

func (s *MemorySuite) TestAnotherAppRecallsNothingAboutTheSameUser() {
	stranger := s.data.backendOfAnotherApp().actingFor(s.known)

	s.NotContains(s.recalled(stranger, remembering(s.known.userID, nil)), "peanut")
}

func (s *MemorySuite) TestAUserCannotRecallAnotherUsersMemories() {
	stranger := s.data.createUser()

	// Refusing the session and opening it knowing nothing are both an answer.
	opened, admitted := s.tries(stranger, asking(remembering(s.known.userID, nil)))
	if !admitted {
		return
	}
	defer stranger.stopSession(opened.Id)
	s.NotContains(s.ask(stranger, opened, "What do you know about me?"), "peanut")
}

func (s *MemorySuite) TestAFilterNarrowsWhatIsRecalled() {
	elsewhere := remembering(s.known.userID, map[string]string{"company": "globex"})

	s.NotContains(s.recalled(s.known, elsewhere), "peanut")
}

func (s *MemorySuite) TestAFilterMatchingAnotherUsersLabelsRecallsNothingOfTheirs() {
	stranger := s.data.createUser()

	s.NotContains(s.recalled(stranger, remembering(stranger.userID, acme())), "peanut")
}

func (s *MemorySuite) TestAFilterNamingAnotherUserRecallsNothingOfTheirs() {
	stranger := s.data.createUser()
	impersonating := map[string]string{"user_id": s.known.userID, "app_id": s.customerID()}

	s.NotContains(s.recalled(stranger, remembering(stranger.userID, impersonating)), "peanut")
}

func (s *MemorySuite) TestAFilterNamingAnotherAppRecallsNothingOfItsMemories() {
	stranger := s.data.backendOfAnotherApp().actingFor(s.known)
	request := remembering(s.known.userID, map[string]string{"app_id": s.customerID(), "company": "acme"})
	request.Memory.AppId = pointerTo(s.customerID())

	s.NotContains(s.recalled(stranger, request), "peanut")
}

func (s *MemorySuite) TestDeletingASessionForgetsWhatItLearnedAndNothingElse() {
	owner := s.data.createUser()
	first := s.tell(owner, remembering(owner.userID, nil), "I am allergic to peanuts.")
	s.tell(owner, remembering(owner.userID, nil), "My dog is called Rex.")
	s.learns(owner, remembering(owner.userID, nil), "peanut", "rex")

	owner.deleteSession(first)

	recalled := s.recalled(owner, remembering(owner.userID, nil))
	s.NotContains(recalled, "peanut")
	s.Contains(recalled, "rex")
}

func (s *MemorySuite) TestAClaimedGuestsMemoriesMoveToTheUserWhoClaimedThem() {
	guest := s.data.createGuest()
	s.Require().NoError(s.store.RecordGuest(context.Background(),
		&store.GuestUser{ID: guest.userID, CustomerID: s.customerID()}))
	s.tell(guest, remembering(guest.userID, nil), "I am allergic to peanuts.")
	s.learns(guest, remembering(guest.userID, nil), "peanut")
	account := s.data.createUser()

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/guests/claim",
		ClaimGuestRequest{GuestId: guest.userID, UserId: account.userID}, nil))

	s.learns(account, remembering(account.userID, nil), "peanut")
}

// tell says something in a session of its own and ends it, returning the session's id.
// What was said is handed to mem0 once the agent has answered.
func (s *MemorySuite) tell(as *testClient, request CreateSessionRequest, said string) string {
	opened := as.createSession(request)
	s.Require().NotEmpty(s.ask(as, opened, said), "an exchange without an answer is not remembered")
	as.stopSession(opened.Id)
	return opened.Id
}

// learns waits until a session opened with request knows every fact, since mem0 extracts
// what it was told after the conversation has moved on.
func (s *MemorySuite) learns(as *testClient, request CreateSessionRequest, facts ...string) {
	deadline := time.Now().Add(extractionGrace)
	for {
		recalled := s.recalled(as, request)
		missing := false
		for _, fact := range facts {
			missing = missing || !strings.Contains(recalled, fact)
		}
		if !missing {
			return
		}
		s.Require().True(time.Now().Before(deadline), "the agent never learned %v, it knew %q", facts, recalled)
		time.Sleep(recallEvery)
	}
}

// recalled is what a session opened with request knew about the person before they said
// anything. The session writes nothing down, so asking does not teach mem0 the answer.
func (s *MemorySuite) recalled(as *testClient, request CreateSessionRequest) string {
	opened := as.createSession(asking(request))
	defer as.stopSession(opened.Id)
	return s.ask(as, opened, "What do you know about me?")
}

// tries opens a session the router may refuse, reporting whether it was opened.
func (s *MemorySuite) tries(as *testClient, request CreateSessionRequest) (Session, bool) {
	var opened Session
	status := as.do(http.MethodPost, "/v1/agents/sessions", request, &opened)
	if status == http.StatusBadRequest || status == http.StatusForbidden {
		return opened, false
	}
	s.Require().Equal(http.StatusCreated, status)
	return opened, true
}

// ask says something in a session over its socket, the way a device does, and returns what
// the agent answered, in lower case.
func (s *MemorySuite) ask(as *testClient, opened Session, text string) string {
	connection := as.opens("/v1/agents/sessions/" + opened.Id + "/events")
	command := frame{"type": "respond", "text": text}
	// A conversation kept in Stream Chat takes each message once, by the id it is sent with.
	if value(opened.ConversationId) != "" {
		command["command_id"] = s.utils.uuid()
	}
	s.Require().NoError(connection.WriteJSON(command))

	for {
		s.Require().NoError(connection.SetReadDeadline(time.Now().Add(settleFor)))
		var received frame
		s.Require().NoError(connection.ReadJSON(&received))
		s.Require().NotEqual("error", received["type"], "the session reported %v", received)
		if received["type"] == "responded" && received["pending_work"] == false {
			answer, _ := received["text"].(string)
			return strings.ToLower(answer)
		}
	}
}

// remembering is a conversation in writing whose memories are about userID, narrowed by
// filter. Its model answers every turn, since only an exchange with an answer is kept.
func remembering(userID string, filter map[string]string) CreateSessionRequest {
	request := textSession(nil)
	request.Llm = pointerTo("noted/noted-model")
	request.Memory = &SessionMemory{UserId: &userID}
	if filter != nil {
		request.Memory.Filter = &filter
	}
	return request
}

// asking is request as a session that recalls and keeps nothing, answered by the echo
// stub, which says what it was told before the conversation started.
func asking(request CreateSessionRequest) CreateSessionRequest {
	request.Llm = pointerTo("echo/echo-model")
	request.Incognito = pointerTo(true)
	return request
}

// acme is the label the known user's memories were written under.
func acme() map[string]string {
	return map[string]string{"company": "acme"}
}
