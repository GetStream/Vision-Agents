//go:build integration

package api

import (
	"context"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
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
	// configReads counts the reads of one agent config's row.
	configReads *configReads
}

// configReads counts the statements that read the agent_configs row of the config id it is
// set to, as runReads counts the scheduled task runs read.
type configReads struct {
	id    atomic.Pointer[string]
	reads atomic.Int64
}

func (*configReads) BeforeQuery(ctx context.Context, _ *bun.QueryEvent) context.Context { return ctx }

func (c *configReads) AfterQuery(_ context.Context, event *bun.QueryEvent) {
	if id := c.id.Load(); id != nil && strings.Contains(event.Query, "agent_configs") && strings.Contains(event.Query, *id) {
		c.reads.Add(1)
	}
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
	s.configReads = &configReads{}
	s.store.DB().AddQueryHook(s.configReads)
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

// Catches a card written from a channel that is not the person's alone, whose lines would
// reach her other sessions though a session reads cards only in such a channel
// (omnichannel.Between), and a channel taken as hers when what Stream sent says nothing of its
// members: the message is answered, and opens no episode. Each channel is the event's, as
// Stream sends it; PERSON is the writer and AGENT the agent's user, the channel's id.
func (s *ChatEpisodesSuite) TestAMessageInAChannelThatIsNotThePersonsAloneOpensNoEpisode() {
	for _, c := range []struct{ name, channel string }{
		{"a third member", `{"members": [{"user_id": "PERSON"}, {"user_id": "carl"}]}`},
		{"the person, the agent and a third member", `{"members": [{"user_id": "PERSON"}, {"user_id": "AGENT"}, {"user_id": "carl"}]}`},
		{"the agent alone", `{"members": [{"user_id": "AGENT"}]}`},
		{"somebody else alone", `{"members": [{"user_id": "carl"}]}`},
		{"a member with no user id", `{"members": [{"user_id": "PERSON"}, {"user": {"id": "carl"}}]}`},
		{"no members", `{"members": []}`},
		{"members null", `{"members": null}`},
		{"no members field", `{"id": "AGENT", "member_count": 1}`},
		{"members of another type", `{"members": "PERSON"}`},
		{"a user id of another type", `{"members": [{"user_id": "PERSON"}, {"user_id": 7}]}`},
		{"a channel of another type", `"PERSON"`},
		{"no channel", ``},
	} {
		s.Run(c.name, func() {
			s.channelID = "chat-" + s.utils.uuid()
			worker := s.worker()
			config := s.config(true, "llm-flow")
			channel := strings.NewReplacer("PERSON", s.user, "AGENT", s.channelID).Replace(c.channel)

			s.Require().Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.writtenIn(config.ID, "where is my order?", channel, time.Now()), s.current))

			s.True(s.reachesWorker(worker), "the message is answered")
			s.Never(func() bool { return s.contacts() > 0 || s.episodeCount() > 0 }, dropped, 20*time.Millisecond, "a contact or an episode")
		})
	}
}

// Catches the agent counted as somebody else: a person's channel the agent is a member of too
// is hers, and her message writes its card. The person alone is every other test's channel.
func (s *ChatEpisodesSuite) TestAMessageInThePersonsChannelWithTheAgentWritesACard() {
	worker := s.worker()
	config := s.config(true, "llm-flow")
	channel := fmt.Sprintf(`{"members": [{"user_id": %q}, {"user_id": %q}], "member_count": 2}`, s.user, s.channelID)

	s.Require().Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.writtenIn(config.ID, "where is my order?", channel, time.Now()), s.current))

	s.True(s.reachesWorker(worker), "the message is answered")
	custom, _ := s.cards(s.omniChannel(), 1)[0]["custom"].(map[string]any)
	s.Equal(store.EpisodeChat, custom["source"])
}

// Catches a session running on the channel passed over by the members rule: its message in a
// channel with a third member opens nothing, as a worker's does.
func (s *ChatEpisodesSuite) TestAMessageToARunningSessionInAChannelWithAThirdMemberOpensNoEpisode() {
	s.worker()
	config := s.config(true, "en-low-latency")
	s.serverClient.createSession(CreateSessionRequest{AgentId: &s.channelID, ConfigId: &config.ID})
	channel := fmt.Sprintf(`{"members": [{"user_id": %q}, {"user_id": "carl"}]}`, s.user)

	s.Require().Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.writtenIn(config.ID, "where is my order?", channel, time.Now()), s.current))

	s.Never(func() bool { return s.contacts() > 0 || s.episodeCount() > 0 }, dropped, 20*time.Millisecond, "a contact or an episode")
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
// of this delivery made no contact map row, no episode, asked Stream nothing and read the
// config's row once (ownerOf). Catches a second read of the config to learn that its cards are
// off.
func (s *ChatEpisodesSuite) TestAConfigWithoutCardsWritesAndReadsNothingNew() {
	worker := s.worker()
	config := s.config(false, "llm-flow")
	before := len(s.chat.Requests(suiteStreamKey))
	s.configReads.reads.Store(0)
	s.configReads.id.Store(&config.ID)
	s.T().Cleanup(func() { s.configReads.id.Store(nil) })

	s.Require().Equal(http.StatusOK, s.writes(config.ID, "where is my order?"))

	s.True(s.reachesWorker(worker), "the message is answered as before")
	s.Never(func() bool { return s.contacts() > 0 || s.episodeCount() > 0 }, dropped, 20*time.Millisecond, "a contact or an episode")
	s.Equal(before, len(s.chat.Requests(suiteStreamKey)), "Stream was asked something")
	s.Equal(int64(1), s.configReads.reads.Load(), "the config's row is read as many times as at base")
}

// The control in a channel a session answered before, whose calls row ownerOf finds first: a
// config without episode_cards writes nothing, and its row is read once at most. At base
// c9a61fc90 the probe of each delivery read it: naming the row's config, none; naming none or
// another config, none; with a voice recording, once (unopenedSpec). Here naming the row's
// config reads it once (heldConfigOf), as a channel with no row does at base: the one read the
// cards' gate needs. Catches a second read, a read for a channel naming no config, and a card
// written for a config the channel names other than the one that answers.
func (s *ChatEpisodesSuite) TestAConfigWithoutCardsInAChannelASessionAnsweredReadsItOnceAtMost() {
	for _, c := range []struct {
		name      string
		naming    func(row, other store.AgentConfig) string
		recording bool
		reads     int64
	}{
		{"names the row's config", func(row, _ store.AgentConfig) string { return row.ID }, false, 1},
		{"names none", func(store.AgentConfig, store.AgentConfig) string { return "" }, false, 0},
		{"names another config with cards on", func(_, other store.AgentConfig) string { return other.ID }, false, 0},
		{"holds a voice recording", func(row, _ store.AgentConfig) string { return row.ID }, true, 1},
	} {
		s.Run(c.name, func() {
			s.channelID = "chat-" + s.utils.uuid()
			worker := s.worker()
			config, other := s.config(false, "llm-flow"), s.config(true, "llm-flow")
			s.sessionAnswered(config)
			before := len(s.chat.Requests(suiteStreamKey))
			s.configReads.reads.Store(0)
			s.configReads.id.Store(&config.ID)
			defer s.configReads.id.Store(nil)
			body := s.written(c.naming(config, other), "where is my order?")
			if c.recording {
				body = strings.Replace(body, `"text":`,
					`"attachments": [{"type": "voiceRecording", "asset_url": "https://example.invalid/note.aac"}], "text":`, 1)
			}

			s.Require().Equal(http.StatusOK, s.delivers(&s.RouterSuite, body, s.current))

			s.Equal(config.ID, s.handedTo(worker).ConfigID, "the message is answered under the row's config")
			s.Never(func() bool { return s.contacts() > 0 || s.episodeCount() > 0 }, dropped, 20*time.Millisecond, "a contact or an episode")
			s.Equal(before, len(s.chat.Requests(suiteStreamKey)), "Stream was asked something")
			s.Equal(c.reads, s.configReads.reads.Load(), "the config's row is read once at most")
		})
	}
}

// Catches a chat a worker's session answered before losing its cards: the session leaves a
// calls row for the channel, which ownerOf finds first, so without heldConfigOf the next
// message keeps no episode open, and the one after the episode closed opens none.
func (s *ChatEpisodesSuite) TestAChatASessionAnsweredGoesOnWritingCards() {
	ctx := context.Background()
	worker := s.worker()
	config := s.config(true, "llm-flow")
	s.Require().Equal(http.StatusOK, s.writes(config.ID, "where is my order?"))
	s.True(s.reachesWorker(worker))
	omni := s.omniChannel()
	s.cards(omni, 1)
	first := s.lastMessage()
	s.sessionAnswered(config)

	s.Require().Equal(http.StatusOK, s.writes(config.ID, "it was due on Monday"))

	s.True(s.reachesWorker(worker), "the second message is answered")
	s.Require().Eventually(func() bool { return s.lastMessage().After(first) }, settleFor, 10*time.Millisecond,
		"the second message keeps the episode open")
	_, err := s.store.DB().ExecContext(ctx, "UPDATE episodes SET status = ?, ended_at = now() WHERE customer_id = ?",
		store.EpisodeSummarized, s.customerID())
	s.Require().NoError(err)

	s.Require().Equal(http.StatusOK, s.writes(config.ID, "and now it is late"))

	s.True(s.reachesWorker(worker), "the third message is answered")
	s.cards(omni, 2)
	s.Equal(2, s.episodeCount(), "the message after the close opens the next episode")
}

// Catches an omni-channel taken for a channel a person writes in: it is where cards are
// written, so a message in it, which a config of its own names, opens no episode of its own.
func (s *ChatEpisodesSuite) TestAMessageInAnOmniChannelOpensNoChatEpisode() {
	worker := s.worker()
	config := s.config(true, "llm-flow")
	s.channelID = "omni-" + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.writes(config.ID, "where is my order?"))

	s.True(s.reachesWorker(worker), "answered as before")
	s.Never(func() bool { return s.contacts() > 0 || s.episodeCount() > 0 }, dropped, 20*time.Millisecond)
}

// Catches a Stream user keyed by user id alone, or pinned to an app other than the one the
// message came from: the same user id in two Stream apps of one customer is two people, each
// with an omni-channel in their own app.
func (s *ChatEpisodesSuite) TestOneUserIDInTwoStreamAppsIsTwoPeople() {
	ctx := context.Background()
	config := s.config(true, "llm-flow")
	for _, app := range []int64{111, 222} {
		var event messageEvent
		event.ChannelType, event.ChannelID = "agent", fmt.Sprintf("chat-%d-%s", app, s.utils.uuid())
		event.Message.ID, event.Message.Text, event.Message.User.ID = "message-"+s.utils.uuid(), "where is my order?", s.user
		event.Channel = []byte(fmt.Sprintf(`{"members": [{"user_id": %q}]}`, s.user))
		s.router.chatEpisode(ctx, hookOrigin{app: app, queued: true}, event, session.FromConfig(config))
	}

	rows, err := s.store.DB().QueryContext(ctx, `SELECT address, stream_app_pk, conversation_id FROM contact_map
WHERE customer_id = ? AND kind = ? ORDER BY stream_app_pk`, s.customerID(), store.ContactStreamUser)
	s.Require().NoError(err)
	defer rows.Close()
	var addresses, omnis []string
	var apps []int64
	for rows.Next() {
		var address, omni string
		var app int64
		s.Require().NoError(rows.Scan(&address, &app, &omni))
		addresses, apps, omnis = append(addresses, address), append(apps, app), append(omnis, omni)
	}
	s.Require().NoError(rows.Err())
	s.Equal([]string{"111:" + s.user, "222:" + s.user}, addresses)
	s.Equal([]int64{111, 222}, apps, "each person's omni-channel is in the app they wrote in")
	s.Require().Len(omnis, 2)
	s.NotEqual(omnis[0], omnis[1])
	var pinned []int64
	s.Require().NoError(s.store.DB().NewSelect().Table("episodes").Column("stream_app_pk").
		Where("customer_id = ?", s.customerID()).Order("stream_app_pk").Scan(ctx, &pinned))
	s.Equal([]int64{111, 222}, pinned, "each episode's card is read and closed in its own app")
}

// Catches a card written off a hook's request that a draining router does not wait for: it is
// counted (ChatCardsRunning, which cmd/router's running sums) until Stream has it, and a card
// lost then is lost for good, as the next message finds its episode open.
func (s *ChatEpisodesSuite) TestADrainWaitsForAChatCardInFlight() {
	worker := s.worker()
	config := s.config(true, "llm-flow")
	waiting, release := s.chat.Hold("/message")
	released := false
	s.T().Cleanup(func() {
		if !released {
			release()
		}
	})

	s.Require().Equal(http.StatusOK, s.hooked(&s.RouterSuite, "/v1/chat/hooks/stream", s.written(config.ID, "where is my order?"), suiteStreamSecret))

	s.True(s.reachesWorker(worker), "the message is answered while its card is written")
	select {
	case <-waiting:
	case <-time.After(settleFor):
		s.FailNow("the card was never sent")
	}
	s.Equal(1, s.router.ChatCardsRunning(), "the drain counts the card in flight")
	release()
	released = true
	s.Require().Eventually(func() bool { return s.router.ChatCardsRunning() == 0 }, settleFor, 10*time.Millisecond)
	s.cards(s.omniChannel(), 1)
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

// Catches a chat episode the idle sweeper leaves open or cannot summarize, and one whose
// summary leaves out the message that opened it: Stream made that message before the router
// opened its episode, so an episode started at the open reads none of it from its start on. A
// chat episode has no session, so it closes as idle, here by its last message, its start left
// as the message made it; its channel, which the customer's app made with the config it names
// and none of the router's stamps, is read for the summary. recites answers with what it was
// handed, so the summary holds the person's line.
func (s *ChatEpisodesSuite) TestAnIdleChatEpisodeIsSummarizedFromTheMessageThatOpenedIt() {
	ctx := context.Background()
	s.worker()
	config := s.config(true, "recites/recites-model")
	bound, err := s.stream.For(ctx, s.customerID())
	s.Require().NoError(err)
	_, err = bound.Client.Chat().GetOrCreateChannel(ctx, "agent", s.channelID, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{CreatedByID: &s.user, Custom: map[string]any{ConfigField: config.ID},
			Members: []getstream.ChannelMemberRequest{{UserID: s.user}}},
	})
	s.Require().NoError(err)
	text := "I need to move my appointment to Tuesday"
	sent, err := bound.Client.Chat().SendMessage(ctx, "agent", s.channelID, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{Text: &text, UserID: &s.user},
	})
	s.Require().NoError(err)
	created := sent.Data.Message.CreatedAt.Time
	s.Require().NotNil(created)
	s.Require().Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.written(config.ID, text), s.current))
	omni := s.omniChannel()
	s.cards(omni, 1)
	var started time.Time
	s.Require().NoError(s.store.DB().QueryRowContext(ctx, "SELECT started_at FROM episodes WHERE customer_id = ?", s.customerID()).Scan(&started))
	s.True(started.Equal(created.Truncate(time.Microsecond)), "the episode starts at %s, not when its message was made (%s)", started, created)
	// Idle for two hours by its last message, past the closer's hour.
	_, err = s.store.DB().ExecContext(ctx, `INSERT INTO episode_activity (episode_id, last_message_at)
SELECT id, now() - interval '2 hours' FROM episodes WHERE customer_id = ?
ON CONFLICT (episode_id) DO UPDATE SET last_message_at = excluded.last_message_at`, s.customerID())
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

// Defect (R1.1): a worker's session that read Anya's cards answers Bob's message in her
// channel with them in its context. Her own next message is answered there, while Bob's closes
// it and is handed out anew, and the session opened for it reads none of her cards.
func (s *ChatEpisodesSuite) TestAnotherUsersMessageGetsASessionWithoutThePersonsCards() {
	worker := s.worker()
	config := s.config(true, "noted/noted-model")
	s.says(config.ID, "my address is 1 Main St", worker)
	s.cards(s.omniChannel(), 1)
	s.channelID = "chat-" + s.utils.uuid()
	hello := "hello " + s.utils.uuid()
	handed := s.says(config.ID, hello, worker)
	anyas := s.serverClient.createSession(CreateSessionRequest{AgentId: &handed.AgentID, ConfigId: &handed.ConfigID})
	s.ask(anyas.Id, hello)
	s.True(carded(s.askedWith(hello)), "Anya's own session reads her card")

	again := "anything new? " + s.utils.uuid()
	s.Require().Equal(http.StatusOK, s.inChat(config.ID, again))
	s.True(carded(s.askedWith(again)), "her own message is answered by her session")
	s.Empty(worker, "her own message was handed out again")

	s.user = "bob-" + s.utils.uuid()
	question := "what is Anya's address? " + s.utils.uuid()
	bobs := s.says(config.ID, question, worker)
	s.Equal(s.user, bobs.UserID)
	s.Eventually(func() bool {
		var read Session
		return s.serverClient.do(http.MethodGet, "/v1/agents/sessions/"+anyas.Id, nil, &read) == http.StatusOK && read.ClosedAt != nil
	}, settleFor, 10*time.Millisecond, "Anya's session still runs")
	opened := s.serverClient.createSession(CreateSessionRequest{AgentId: &bobs.AgentID, ConfigId: &bobs.ConfigID})
	s.ask(opened.Id, question)
	s.False(carded(s.askedWith(question)), "Bob's session reads Anya's card")
}

// Defect (R2.1): Bob's message closes the session holding Anya's cards at once, so her reply
// still being written after the 5 s a closing session waits (finishWithin) is cut off, and
// her message queued behind it is never answered; or the session the worker opens for Bob
// takes her conversation over, cutting them off as well. Each of her replies takes 6 s here.
// Bob's message is handed out while her first is written, and so is hers after his, never to
// the session closing; her session answers both of hers with her cards and then closes; the
// session opened for Bob waits for that, and reads none of her cards.
func (s *ChatEpisodesSuite) TestAnotherUsersMessageLetsThePersonsSessionFinishItsReplies() {
	const takes = 6 * time.Second
	worker := s.worker()
	config := s.config(true, "noted/noted-model")
	s.says(config.ID, "my address is 1 Main St", worker)
	s.cards(s.omniChannel(), 1)
	s.channelID = "chat-" + s.utils.uuid()
	anya := s.user
	hello := "hello " + s.utils.uuid()
	handed := s.says(config.ID, hello, worker)
	anyas := s.serverClient.createSession(CreateSessionRequest{AgentId: &handed.AgentID, ConfigId: &handed.ConfigID})
	s.ask(anyas.Id, hello)
	s.Require().Eventually(func() bool { return s.completed() == 1 }, settleFor, 10*time.Millisecond, "her first reply")
	s.slow(takes)

	first, second := "when is my cleaning? "+s.utils.uuid(), "and the price? "+s.utils.uuid()
	s.Require().Equal(http.StatusOK, s.inChat(config.ID, first))
	s.True(carded(s.askedWith(first)), "her first question is answered by her session")
	s.Require().Equal(http.StatusOK, s.inChat(config.ID, second))

	s.user = "bob-" + s.utils.uuid()
	question := "what is Anya's address? " + s.utils.uuid()
	bobs := s.says(config.ID, question, worker)
	s.Equal(s.user, bobs.UserID)
	s.Equal(1, s.completed(), "Bob's message waited for her reply")
	s.user = anya
	after := "one more thing " + s.utils.uuid()
	s.Equal(after, s.says(config.ID, after, worker).Text, "her message after Bob's went to the session closing")
	opened := s.serverClient.createSession(CreateSessionRequest{AgentId: &bobs.AgentID, ConfigId: &bobs.ConfigID})
	s.ask(opened.Id, question)
	s.False(carded(s.askedWith(question)), "Bob's session reads Anya's card")

	s.Eventually(func() bool {
		var read Session
		return s.serverClient.do(http.MethodGet, "/v1/agents/sessions/"+anyas.Id, nil, &read) == http.StatusOK && read.ClosedAt != nil
	}, 3*takes, 20*time.Millisecond, "Anya's session still runs once her replies are written")
	s.True(carded(s.askedWith(second)), "her queued question is answered by her session")
	all := []string{"completed", "completed", "completed", "completed"}
	var states []string
	s.Eventually(func() bool { states = s.replies(); return slices.Equal(states, all) }, settleFor, 20*time.Millisecond)
	s.Equal(all, states, "her first, both of hers and Bob's: one was cut off or never written")
}

// Defect (R1.2): the hand-out reaches another node without its author, so the worker's
// session that node opens, as the load balancer may have it, reads none of Anya's cards.
func (s *ChatEpisodesSuite) TestAWorkersSessionOnAnotherNodeReadsThePersonsCards() {
	worker := s.worker()
	config := s.config(true, "noted/noted-model")
	s.says(config.ID, "my address is 1 Main St", worker)
	s.cards(s.omniChannel(), 1)
	s.channelID = "chat-" + s.utils.uuid()
	hello := "hello " + s.utils.uuid()
	handed := s.says(config.ID, hello, worker)
	elsewhere := s.serverClient.on(s.otherNode())

	opened := elsewhere.createSession(CreateSessionRequest{AgentId: &handed.AgentID, ConfigId: &handed.ConfigID})
	s.Require().Equal(http.StatusAccepted, elsewhere.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{Text: hello, RequestId: pointerTo(s.utils.uuid())}, nil))

	s.True(carded(s.askedWith(hello)))
}

// slow has every reply of the "noted" models opened so far take d.
func (s *ChatEpisodesSuite) slow(d time.Duration) {
	s.notedMu.Lock()
	defer s.notedMu.Unlock()
	for _, model := range s.noted {
		model.mu.Lock()
		model.takes = d
		model.mu.Unlock()
	}
}

// completed is how many replies in the test's channel were written to the end.
func (s *ChatEpisodesSuite) completed() int {
	return len(slices.DeleteFunc(s.replies(), func(state string) bool { return state != "completed" }))
}

// replies is the state of each reply written in the test's channel, oldest first.
func (s *ChatEpisodesSuite) replies() []string {
	var states []string
	for _, stored := range s.chat.Stored(s.channelID) {
		custom, _ := stored["custom"].(map[string]any)
		if written, _ := custom["support_message"].(map[string]any); written != nil && written["role"] == "assistant" {
			state, _ := written["state"].(string)
			states = append(states, state)
		}
	}
	return states
}

// says writes, as the test's user, text in the test's channel, which names the config given,
// as the customer's app does, and returns what the worker was handed for it.
func (s *ChatEpisodesSuite) says(configID, text string, worker <-chan dispatch.Message) dispatch.Message {
	s.Require().Equal(http.StatusOK, s.inChat(configID, text))
	return s.handedTo(worker)
}

// inChat writes, as the test's user, text in the test's channel, which names the config given,
// and delivers it as Stream Chat does. The channel's first writer makes it, with that user its
// one member, as the router's own chat channel lists only its reader (createChatToken).
func (s *ChatEpisodesSuite) inChat(configID, text string) int {
	ctx := context.Background()
	_, err := s.chat.Client.Chat().GetOrCreateChannel(ctx, "agent", s.channelID, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{CreatedByID: &s.user, Custom: map[string]any{ConfigField: configID},
			Members: []getstream.ChannelMemberRequest{{UserID: s.user}}},
	})
	s.Require().NoError(err)
	_, err = s.chat.Client.Chat().SendMessage(ctx, "agent", s.channelID, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{Text: &text, UserID: &s.user},
	})
	s.Require().NoError(err)
	return s.writes(configID, text)
}

// ask hands a session text, as the worker does.
func (s *ChatEpisodesSuite) ask(session, text string) {
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/agents/sessions/"+session+"/responses",
		CreateResponseRequest{Text: text, RequestId: pointerTo(s.utils.uuid())}, nil))
}

// askedWith is the request a "noted" model was sent to answer text, once one was.
func (s *ChatEpisodesSuite) askedWith(text string) llm.ResponseParams {
	var found llm.ResponseParams
	s.Require().Eventually(func() bool {
		s.notedMu.Lock()
		defer s.notedMu.Unlock()
		for _, model := range s.noted {
			model.mu.Lock()
			asked := slices.Clone(model.asked)
			model.mu.Unlock()
			for _, params := range asked {
				if slices.ContainsFunc(params.Input, func(m llm.Message) bool { return m.Role == llm.User && m.Content == text }) {
					found = params
					return true
				}
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "no model was asked to answer %q", text)
	return found
}

// carded reports whether a request holds the person's episode cards, which follow their note
// (omnichannel.cardsAttribution).
func carded(params llm.ResponseParams) bool {
	return slices.ContainsFunc(params.Input, func(m llm.Message) bool {
		return m.Role == llm.System && strings.HasPrefix(m.Content, "The person's earlier episodes with this agent")
	})
}

// writes delivers, as Stream Chat does, a message the test's user wrote in the test's channel,
// which names the config given, and answers the status.
func (s *ChatEpisodesSuite) writes(configID, text string) int {
	return s.delivers(&s.RouterSuite, s.written(configID, text), s.current)
}

// written is a message.new the test's user wrote in the test's channel, which names the
// config given, or none for an empty one, as Stream sends it: with the channel's members, the
// ones the fake Chat holds, else the test's user alone, as the customer's app makes a person's
// chat; and made when the fake Chat stored the last message of that text there, else now.
func (s *ChatEpisodesSuite) written(configID, text string) string {
	members := s.chat.Members(s.channelID)
	if len(members) == 0 {
		members = []string{s.user}
	}
	var listed []string
	for _, member := range members {
		listed = append(listed, fmt.Sprintf(`{"user_id": %q, "user": {"id": %q}}`, member, member))
	}
	created := time.Now()
	for _, stored := range s.chat.Stored(s.channelID) {
		if at, ok := stored["created_at"].(int64); ok && stored["text"] == text {
			created = time.Unix(0, at)
		}
	}
	return s.writtenIn(configID, text, `{"members": [`+strings.Join(listed, ", ")+`]}`, created)
}

// writtenIn is written with the channel given, raw, as the event's channel, none for "", and
// the message made at created.
func (s *ChatEpisodesSuite) writtenIn(configID, text, channel string, created time.Time) string {
	custom := fmt.Sprintf("{%q: %q}", ConfigField, configID)
	if configID == "" {
		custom = "{}"
	}
	sent := ""
	if channel != "" {
		sent = fmt.Sprintf(`"channel": %s,`, channel)
	}
	return fmt.Sprintf(`{
  "type": "message.new",
  "app_id": %d,
  "channel_id": %q,
  "channel_type": "agent",
  "created_at": %q,
  "channel_custom": %s,
  %s
  "message": {"id": %q, "text": %q, "created_at": %q, "user": {"id": %q, "name": "Anya"}}
}`, suiteStreamApp, s.channelID, time.Now().UTC().Format(time.RFC3339Nano), custom, sent,
		"message-"+s.utils.uuid(), text, created.UTC().Format(time.RFC3339Nano), s.user)
}

// sessionAnswered leaves the calls row a worker's session answering in the test's channel
// under config leaves (session.Manager), which ended.
func (s *ChatEpisodesSuite) sessionAnswered(config store.AgentConfig) {
	started := time.Now().UTC().Add(-time.Minute)
	ended := started.Add(30 * time.Second)
	s.Require().NoError(s.store.StartCall(context.Background(), &store.Call{ID: s.utils.uuid(), CustomerID: s.customerID(),
		CallID: s.utils.callID(), AgentID: s.channelID, ConfigID: config.ID, Direction: "inbound", StartedAt: started, EndedAt: &ended}))
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
			s.customerID(), store.ContactStreamUser, fmt.Sprintf("%d:%s", suiteStreamApp, s.user)).Scan(&cid) == nil
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
