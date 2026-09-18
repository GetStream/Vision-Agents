//go:build integration

package api

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// MessageHookIntegrationSuite covers the half of the message hook the unit suite cannot:
// going from a channel somebody wrote in to the customer whose worker should answer is a
// database question.
type MessageHookIntegrationSuite struct {
	suite.Suite
	ctx        context.Context
	store      *store.Store
	pool       *dispatch.Pool
	handler    http.Handler
	customerID string
	channelID  string
}

func TestMessageHookIntegrationSuite(t *testing.T) {
	suite.Run(t, new(MessageHookIntegrationSuite))
}

func (s *MessageHookIntegrationSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN must be set")
	}

	s.ctx = context.Background()

	pgStore, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(pgStore.Migrate(s.ctx))
	s.store = pgStore
}

func (s *MessageHookIntegrationSuite) TearDownSuite() {
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
}

func (s *MessageHookIntegrationSuite) SetupTest() {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	speech, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)

	s.pool = dispatch.NewPool()
	server, err := NewServer(Options{
		Routers:      map[routing.Modality]routing.Inspector{routing.STT: speech},
		Store:        s.store,
		Dispatch:     s.pool,
		StreamSecret: hookSecret,
	})
	s.Require().NoError(err)
	s.handler = server.Handler()

	unique := time.Now().UnixNano()
	s.customerID = fmt.Sprintf("customer-%d", unique)
	s.channelID = fmt.Sprintf("support-%d", unique)
}

// holds stores an agent config for the current customer and returns its id.
func (s *MessageHookIntegrationSuite) holds(name string) string {
	config := store.AgentConfig{CustomerID: s.customerID, Name: name, Mode: "text"}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, &config))
	return config.ID
}

// ran records a conversation having been held on the channel, which is what a call leaves
// behind.
func (s *MessageHookIntegrationSuite) ran(configID string) {
	s.Require().NoError(s.store.StartCall(s.ctx, &store.Call{
		ID:         "session-" + s.channelID,
		CustomerID: s.customerID,
		CallID:     "call-" + s.channelID,
		AgentID:    s.channelID,
		ConfigID:   configID,
		StartedAt:  time.Now().UTC(),
	}))
}

// wrote delivers a signed message.new for the current channel, as Stream would. The custom
// data is whatever the channel was created with.
func (s *MessageHookIntegrationSuite) wrote(custom string) *httptest.ResponseRecorder {
	if custom == "" {
		custom = "{}"
	}
	body := fmt.Sprintf(`{
  "type": "message.new",
  "cid": "agent:%s",
  "channel_id": "%s",
  "channel_type": "agent",
  "channel_custom": %s,
  "message": {
    "id": "message-1",
    "text": "does the react sdk retry a failed upload?",
    "user": {"id": "sam", "name": "Sam"}
  }
}`, s.channelID, s.channelID, custom)

	mac := hmac.New(sha256.New, []byte(hookSecret))
	mac.Write([]byte(body))

	request := httptest.NewRequestWithContext(s.ctx,
		http.MethodPost, "/v1/chat/hooks/stream", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(signatureHeader, hex.EncodeToString(mac.Sum(nil)))
	recorder := httptest.NewRecorder()
	s.handler.ServeHTTP(recorder, request)
	return recorder
}

func (s *MessageHookIntegrationSuite) TestAMessageOnAChannelAConversationAlreadyRanOnReachesItsOwner() {
	configID := s.holds("support")
	s.ran(configID)
	worker, release := s.pool.Register(s.customerID, 1)
	defer release()

	recorder := s.wrote("")
	s.Require().Equal(http.StatusOK, recorder.Code, recorder.Body.String())

	select {
	case message := <-worker.Messages():
		s.Equal(s.channelID, message.ChannelID)
		s.Equal(s.channelID, message.AgentID, "a session opened on anything else answers elsewhere")
		s.Equal(configID, message.ConfigID)
		s.Equal("sam", message.UserID)
		s.Equal("does the react sdk retry a failed upload?", message.Text)
	case <-time.After(2 * time.Second):
		s.Fail("the message never reached the worker")
	}
}

func (s *MessageHookIntegrationSuite) TestAChannelThatNamesItsConfigIsAnsweredWithoutEverHavingHeldAConversation() {
	// This is a conversation that starts in writing: somebody opened a support chat
	// rather than ringing a number, so there is no call row to say whose channel it is.
	// Without the channel saying, nobody could ever be written to first.
	configID := s.holds("support")
	worker, release := s.pool.Register(s.customerID, 1)
	defer release()

	recorder := s.wrote(fmt.Sprintf(`{"%s": "%s"}`, ConfigField, configID))
	s.Require().Equal(http.StatusOK, recorder.Code, recorder.Body.String())

	select {
	case message := <-worker.Messages():
		s.Equal(s.channelID, message.ChannelID)
		s.Equal(configID, message.ConfigID, "the worker has to know which agent was written to")
	case <-time.After(2 * time.Second):
		s.Fail("the message never reached the worker")
	}
}

func (s *MessageHookIntegrationSuite) TestWhatTheChannelWasCreatedWithIsCarriedToTheWorker() {
	// This service has no opinion about an organization or a locale and should not need
	// one. It carries what the channel was created with, and the worker, which is where
	// the agent runs, decides what any of it means.
	configID := s.holds("support")
	worker, release := s.pool.Register(s.customerID, 1)
	defer release()

	recorder := s.wrote(fmt.Sprintf(
		`{"%s": "%s", "organization_id": "1234", "locale": "en-GB", "seats": 12}`,
		ConfigField, configID))
	s.Require().Equal(http.StatusOK, recorder.Code, recorder.Body.String())

	select {
	case message := <-worker.Messages():
		s.Equal("1234", message.Custom["organization_id"])
		s.Equal("en-GB", message.Custom["locale"])
		s.NotContains(message.Custom, "seats",
			"Stream takes arbitrary JSON here and a worker reads strings; a number rendered into one would be believed")
	case <-time.After(2 * time.Second):
		s.Fail("the message never reached the worker")
	}
}

func (s *MessageHookIntegrationSuite) TestAChannelCannotSendItsMessagesToAnotherCustomersWorkers() {
	// Whoever creates a channel decides what is on it, so the customer is worked out from
	// the config rather than read off the channel. A channel naming somebody else's config
	// is answered by that somebody else, and a channel claiming a customer is not answered
	// on that claim at all.
	configID := s.holds("support")
	somebodyElse, release := s.pool.Register("somebody-else", 1)
	defer release()

	recorder := s.wrote(fmt.Sprintf(`{"customer_id": "somebody-else", "%s": "%s"}`, ConfigField, configID))
	s.Require().Equal(http.StatusOK, recorder.Code)

	select {
	case message := <-somebodyElse.Messages():
		s.Failf("a message was misrouted", "%s went to another customer's worker", message.ChannelID)
	case <-time.After(200 * time.Millisecond):
	}
}

func (s *MessageHookIntegrationSuite) TestAChannelNamingAConfigNobodyHoldsIsAcceptedAndDropped() {
	worker, release := s.pool.Register(s.customerID, 1)
	defer release()

	recorder := s.wrote(fmt.Sprintf(`{"%s": "config-nobody-has"}`, ConfigField))
	s.Require().Equal(http.StatusOK, recorder.Code, "Stream retries a non-2xx, and no retry finds an owner")

	select {
	case message := <-worker.Messages():
		s.Failf("a message was answered", "%s names a config nobody holds", message.ChannelID)
	case <-time.After(200 * time.Millisecond):
	}
}

func (s *MessageHookIntegrationSuite) TestAChannelWithNoHistoryAndNoConfigIsAcceptedAndDropped() {
	// Every message in the app arrives at this hook. One in a channel nothing claims is
	// not answerable on a retry either.
	worker, release := s.pool.Register(s.customerID, 1)
	defer release()

	recorder := s.wrote("")
	s.Require().Equal(http.StatusOK, recorder.Code)

	select {
	case message := <-worker.Messages():
		s.Failf("a message was answered", "nothing says whose channel %s is", message.ChannelID)
	case <-time.After(200 * time.Millisecond):
	}
}

func (s *MessageHookIntegrationSuite) TestAConfigThatWasDeletedNoLongerClaimsAChannel() {
	configID := s.holds("support")
	s.Require().NoError(s.store.DeleteAgentConfig(s.ctx, s.customerID, configID))
	worker, release := s.pool.Register(s.customerID, 1)
	defer release()

	recorder := s.wrote(fmt.Sprintf(`{"%s": "%s"}`, ConfigField, configID))
	s.Require().Equal(http.StatusOK, recorder.Code)

	select {
	case message := <-worker.Messages():
		s.Failf("a message was answered", "%s names a config that was deleted", message.ChannelID)
	case <-time.After(200 * time.Millisecond):
	}
}
