//go:build integration

package api

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ChatEpisodesSuite is the app chat's episode card: a person's message in an agent channel,
// delivered by Stream Chat (/v1/chat/internal/messages) and answered by the queue's worker,
// opens an episode in the person's omni-channel, keyed by their Stream user id, under an agent
// config with episode_cards on. A database of its own, as the queue's worker claims every row
// in it and the sweeper closes every idle episode.
type ChatEpisodesSuite struct {
	RouterSuite
	chatInbounds

	// user is the Stream user who writes, unique to the test.
	user string
}

func TestChatEpisodesSuite(t *testing.T) {
	runSuite(t, new(ChatEpisodesSuite))
}

func (s *ChatEpisodesSuite) SetupSuite() {
	s.chatInbounds.keys(&s.RouterSuite)
	s.deploymentKnown = true
	s.database = "chatcards"
	s.RouterSuite.SetupSuite()
	s.wake(&s.RouterSuite)
}

func (s *ChatEpisodesSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.channelID = "chat-" + s.utils.uuid()
	s.user = "user-" + s.utils.uuid()
	s.answering(&s.RouterSuite)
}

// Catches a chat message that opens no episode, a card with the wrong source or thread, and a
// second card for the same run of messages.
func (s *ChatEpisodesSuite) TestAChatMessageWritesOneCardAndTheNextOneKeepsItsEpisodeOpen() {
	worker := s.worker()
	config := s.config(true, "llm-flow")

	s.Require().Equal(http.StatusOK, s.writes(config.ID, "where is my order?"))

	s.True(s.reachesWorker(worker), "the message is answered")
	omni := s.omniChannel()
	card := s.cards(omni, 1)[0]
	custom, _ := card["custom"].(map[string]any)
	s.Equal(store.EpisodeChat, custom["source"])
	s.Equal("agent:"+s.channelID, custom["thread_channel"], "the channel the chat is in")
	s.Equal("in_progress", custom["status"])
	s.NotContains(custom, "call_id")
	s.Regexp(`^omni-[0-9a-f-]{36}$`, omni)
	first := s.lastMessage()

	s.Require().Equal(http.StatusOK, s.writes(config.ID, "it was due on Monday"))

	s.True(s.reachesWorker(worker), "the second message is answered")
	s.Require().Eventually(func() bool { return s.lastMessage().After(first) }, settleFor, 10*time.Millisecond,
		"the second message keeps the episode open")
	s.Never(func() bool { return len(s.chat.Stored(omni)) > 1 }, dropped, 20*time.Millisecond, "a second card")
	s.Equal(1, s.episodeCount())
	s.Equal(1, s.contacts())
}

// Catches the message webhook's path left out: Stream waits on the hook, so the card is
// written off its request, and is written all the same.
func (s *ChatEpisodesSuite) TestAChatMessageTheWebhookBringsWritesACard() {
	worker := s.worker()
	config := s.config(true, "llm-flow")

	s.Require().Equal(http.StatusOK, s.hooked(&s.RouterSuite, "/v1/chat/hooks/stream", s.written(config.ID, "where is my order?"), suiteStreamSecret))

	s.True(s.reachesWorker(worker), "the message is answered")
	custom, _ := s.cards(s.omniChannel(), 1)[0]["custom"].(map[string]any)
	s.Equal(store.EpisodeChat, custom["source"])
}

// Catches a redelivery that opens a second episode or writes a second card: the queue's worker
// tries a row again when its lease runs out, after a try that opened the episode.
func (s *ChatEpisodesSuite) TestAQueuedMessageTriedAgainOpensNoSecondEpisode() {
	worker := s.worker()
	config := s.config(true, "llm-flow")
	s.Require().Equal(http.StatusOK, s.writes(config.ID, "where is my order?"))
	s.True(s.reachesWorker(worker))
	omni := s.omniChannel()
	s.cards(omni, 1)
	var seq int64
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT seq FROM chat_inbound WHERE position(? in body) > 0", []byte(s.channelID)).Scan(&seq))
	row, err := s.store.ChatInboundBySeq(context.Background(), seq)
	s.Require().NoError(err)

	s.Require().NoError(s.router.AnswerChatInbound(context.Background(), row))

	s.False(s.reachesWorker(worker), "one delivery is one answer")
	s.Never(func() bool { return len(s.chat.Stored(omni)) > 1 }, dropped, 20*time.Millisecond, "a second card")
	s.Equal(1, s.episodeCount())
	s.Equal(1, s.cardWrites(omni), "the card is written once, not again under its own id")
}

// The control: a config without episode_cards answers as at base c9a61fc90, where the probe
// of this delivery made no contact map row, no episode and asked Stream nothing.
func (s *ChatEpisodesSuite) TestAConfigWithoutCardsWritesAndReadsNothingNew() {
	worker := s.worker()
	config := s.config(false, "llm-flow")
	before := len(s.chat.Requests(suiteStreamKey))

	s.Require().Equal(http.StatusOK, s.writes(config.ID, "where is my order?"))

	s.True(s.reachesWorker(worker), "the message is answered as before")
	s.Never(func() bool { return s.contacts() > 0 || s.episodeCount() > 0 }, dropped, 20*time.Millisecond, "a contact or an episode")
	s.Equal(before, len(s.chat.Requests(suiteStreamKey)), "Stream was asked something")
}

// Catches the agent's own reply opening an episode: it carries a source, as everything the
// agent stores does, so it is not answered and opens nothing.
func (s *ChatEpisodesSuite) TestTheAgentsOwnReplyOpensNothing() {
	worker := s.worker()
	config := s.config(true, "llm-flow")

	s.Require().Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.message(suiteStreamApp, config.ID, `"agent"`), s.current))

	s.False(s.reachesWorker(worker))
	s.Never(func() bool { return s.contacts() > 0 || s.episodeCount() > 0 }, dropped, 20*time.Millisecond)
}

// Catches a chat episode on a thread channel, whose episode is the channel bridge's: a thread
// channel no thread is linked to yet is answered by its config, and still opens nothing.
func (s *ChatEpisodesSuite) TestAMessageInAThreadChannelOpensNoChatEpisode() {
	worker := s.worker()
	config := s.config(true, "llm-flow")
	s.channelID = "thread-" + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.writes(config.ID, "where is my order?"))

	s.True(s.reachesWorker(worker), "answered as before")
	s.Never(func() bool { return s.contacts() > 0 || s.episodeCount() > 0 }, dropped, 20*time.Millisecond)
}

// Catches a session running on the channel being passed over: its spec is the config that
// answers, so its message opens the episode as a worker's does.
func (s *ChatEpisodesSuite) TestAMessageToARunningSessionWritesACard() {
	s.worker()
	config := s.config(true, "en-low-latency")
	s.serverClient.createSession(CreateSessionRequest{AgentId: &s.channelID, ConfigId: &config.ID})

	s.Require().Equal(http.StatusOK, s.writes(config.ID, "where is my order?"))

	custom, _ := s.cards(s.omniChannel(), 1)[0]["custom"].(map[string]any)
	s.Equal(store.EpisodeChat, custom["source"])
}

// Catches an incognito session's message being kept: it records nothing of the person.
func (s *ChatEpisodesSuite) TestAMessageToAnIncognitoSessionOpensNothing() {
	worker := s.worker()
	config := s.config(true, "en-low-latency")
	s.serverClient.createSession(CreateSessionRequest{AgentId: &s.channelID, ConfigId: &config.ID, Incognito: pointerTo(true)})

	s.Require().Equal(http.StatusOK, s.writes(config.ID, "where is my order?"))

	s.NotEmpty(s.handedTo(worker).SessionID, "the running session answers")
	s.Never(func() bool { return s.contacts() > 0 || s.episodeCount() > 0 }, dropped, 20*time.Millisecond)
}

// Catches a chat episode opened in a call's channel while the call runs: the call's card names
// that channel, and a chat episode started in it would end the call card's lines there.
func (s *ChatEpisodesSuite) TestAMessageToACallsSessionOpensNoChatEpisode() {
	config := s.config(true, "llm-flow")
	call := s.utils.uuid()
	s.chat.PutCall("agent", call, "someone-in-a-browser")
	s.serverClient.createSession(CreateSessionRequest{Id: &call, StartVoice: pointerTo(true), ConfigId: &config.ID})
	s.channelID = call

	s.Require().Equal(http.StatusOK, s.writes(config.ID, "where is my order?"))

	s.Never(func() bool { return s.contacts() > 0 || s.episodeCount() > 0 }, dropped, 20*time.Millisecond)
}

// Catches a chat episode the idle sweeper leaves open or cannot summarize: a chat episode has
// no session, so it closes as idle, and its channel, which the customer's app made with the
// config it names and none of the router's stamps, is read for the summary. recites answers
// with what it was handed, so the summary holds the person's line.
func (s *ChatEpisodesSuite) TestAnIdleChatEpisodeIsClosedAndSummarized() {
	ctx := context.Background()
	s.worker()
	config := s.config(true, "recites/recites-model")
	bound, err := s.stream.For(ctx, s.customerID())
	s.Require().NoError(err)
	_, err = bound.Client.Chat().GetOrCreateChannel(ctx, "agent", s.channelID, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{CreatedByID: &s.user, Custom: map[string]any{ConfigField: config.ID}},
	})
	s.Require().NoError(err)
	text := "I need to move my appointment to Tuesday"
	_, err = bound.Client.Chat().SendMessage(ctx, "agent", s.channelID, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{Text: &text, UserID: &s.user},
	})
	s.Require().NoError(err)
	s.Require().Equal(http.StatusOK, s.writes(config.ID, text))
	omni := s.omniChannel()
	s.cards(omni, 1)
	// Idle for two hours, past the closer's hour.
	_, err = s.store.DB().ExecContext(ctx,
		"UPDATE episodes SET started_at = now() - interval '2 hours' WHERE customer_id = ?", s.customerID())
	s.Require().NoError(err)
	_, err = s.store.DB().ExecContext(ctx, `UPDATE episode_activity SET last_message_at = now() - interval '2 hours'
WHERE episode_id IN (SELECT id FROM episodes WHERE customer_id = ?)`, s.customerID())
	s.Require().NoError(err)

	s.Require().NoError(s.episodes.Sweep(ctx))

	var status string
	s.Require().NoError(s.store.DB().QueryRowContext(ctx, "SELECT status FROM episodes WHERE customer_id = ?", s.customerID()).Scan(&status))
	s.Equal(store.EpisodeSummarized, status)
	stored := s.chat.Stored(omni)
	s.Require().Len(stored, 1, "the card is updated in place")
	s.Contains(stored[0]["text"], text)
}

// Catches a card that costs the answer: Stream refusing the card's write leaves the episode
// without its card, and the message is answered all the same.
func (s *ChatEpisodesSuite) TestStreamRefusingTheCardStillAnswersTheMessage() {
	worker := s.worker()
	config := s.config(true, "llm-flow")
	s.chat.Fail(http.MethodPost, "/message")
	s.T().Cleanup(func() { s.chat.Heal(http.MethodPost, "/message") })

	s.Require().Equal(http.StatusOK, s.writes(config.ID, "where is my order?"))

	s.True(s.reachesWorker(worker), "the message is answered")
	omni := s.omniChannel()
	s.Equal(1, s.episodeCount(), "the episode is open")
	s.Never(func() bool { return len(s.chat.Stored(omni)) > 0 }, dropped, 20*time.Millisecond)
}

// writes delivers, as Stream Chat does, a message the test's user wrote in the test's channel,
// which names the config given, and answers the status.
func (s *ChatEpisodesSuite) writes(configID, text string) int {
	return s.delivers(&s.RouterSuite, s.written(configID, text), s.current)
}

// written is a message.new the test's user wrote in the test's channel, which names the
// config given.
func (s *ChatEpisodesSuite) written(configID, text string) string {
	return fmt.Sprintf(`{
  "type": "message.new",
  "app_id": %d,
  "channel_id": %q,
  "channel_type": "agent",
  "created_at": %q,
  "channel_custom": {%q: %q},
  "message": {"id": %q, "text": %q, "user": {"id": %q, "name": "Anya"}}
}`, suiteStreamApp, s.channelID, time.Now().UTC().Format(time.RFC3339Nano), ConfigField, configID,
		"message-"+s.utils.uuid(), text, s.user)
}

// config is a text agent of the test's app on the LLM named, with episode cards on or off.
func (s *ChatEpisodesSuite) config(cards bool, llm string) store.AgentConfig {
	config := store.AgentConfig{CustomerID: s.customerID(), Name: "chat-" + s.utils.uuid(), Mode: store.AgentModeText,
		LLM: llm, DispatchText: true, EpisodeCards: cards}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	return config
}

// worker connects a dispatch worker for the test's customer and returns what it is handed.
func (s *ChatEpisodesSuite) worker() <-chan dispatch.Message {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 4})
	s.T().Cleanup(release)
	return worker.Messages()
}

// handedTo is the next message a worker was handed.
func (s *ChatEpisodesSuite) handedTo(messages <-chan dispatch.Message) dispatch.Message {
	select {
	case message := <-messages:
		return message
	case <-time.After(settleFor):
		s.FailNow("the worker was handed nothing")
		return dispatch.Message{}
	}
}

// omniChannel is the id of the test user's omni-channel, once their message made it.
func (s *ChatEpisodesSuite) omniChannel() string {
	var cid string
	s.Require().Eventually(func() bool {
		return s.store.DB().QueryRowContext(context.Background(),
			"SELECT conversation_id FROM contact_map WHERE customer_id = ? AND kind = ? AND address = ?",
			s.customerID(), store.ContactStreamUser, s.user).Scan(&cid) == nil
	}, settleFor, 10*time.Millisecond, "no contact for %s", s.user)
	return strings.TrimPrefix(cid, "agent:")
}

// cards waits until an omni-channel holds count cards.
func (s *ChatEpisodesSuite) cards(omni string, count int) []map[string]any {
	s.Require().Eventually(func() bool { return len(s.chat.Stored(omni)) >= count }, settleFor, 10*time.Millisecond,
		"%s holds %d cards, not %d", omni, len(s.chat.Stored(omni)), count)
	return s.chat.Stored(omni)
}

// cardWrites is how many times Stream was asked to send a message into an omni-channel.
func (s *ChatEpisodesSuite) cardWrites(omni string) int {
	sent := 0
	for _, request := range s.chat.Requests(suiteStreamKey) {
		if request.Method == http.MethodPost && strings.HasSuffix(request.Path, "/agent/"+omni+"/message") {
			sent++
		}
	}
	return sent
}

// lastMessage is when the test customer's one episode last had a message, its start while
// it has had none since.
func (s *ChatEpisodesSuite) lastMessage() time.Time {
	var at time.Time
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(), `SELECT COALESCE(act.last_message_at, ep.started_at)
FROM episodes AS ep LEFT JOIN episode_activity AS act ON act.episode_id = ep.id WHERE ep.customer_id = ?`, s.customerID()).Scan(&at))
	return at
}

// episodeCount is how many episodes the test's customer has.
func (s *ChatEpisodesSuite) episodeCount() int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM episodes WHERE customer_id = ?", s.customerID()).Scan(&count))
	return count
}

// contacts is how many contact map rows the test's customer has.
func (s *ChatEpisodesSuite) contacts() int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM contact_map WHERE customer_id = ?", s.customerID()).Scan(&count))
	return count
}
