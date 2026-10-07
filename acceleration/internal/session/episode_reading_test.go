//go:build integration

package session

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// streamKey is the deployment app's key in these tests, which every Stream request is made
// with.
const streamKey = "deploy-key"

// cardsNote starts the system message the cards are handed behind.
const cardsNote = "The person's earlier episodes with this agent on other channels follow"

// EpisodeReadingSuite is a session reading the person's episode cards (T56 and T42, AI-885):
// real sessions over Postgres and the suite's Stream Chat (chattest), with a model that keeps
// every request it is sent. A text session runs on a thread channel, as the channel bridge's
// message hook opens it (api.threadSession); a voice session on a call with a SIP caller, as
// the call hook dispatches it. SMS is not on the channel bridge until T53, so an SMS thread
// is opened here as the bridge opens a Slack one: an episode, then the person's message.
type EpisodeReadingSuite struct {
	suite.Suite
	ctx     context.Context
	store   *store.Store
	chat    *chattest.Server
	apps    *streamapp.Clients
	cards   *omnichannel.Cards
	manager *Manager
	model   *recordingLLM

	customerID, configID string
	now                  time.Time
}

func TestEpisodeReadingSuite(t *testing.T) {
	suite.Run(t, new(EpisodeReadingSuite))
}

func (s *EpisodeReadingSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN is not set")
	}
	s.ctx = context.Background()
	db, err := store.Open(testenv.Database(dsn, "episode_reading"))
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.store = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })
}

func (s *EpisodeReadingSuite) SetupTest() {
	s.chat = chattest.NewServer(s.T())
	s.apps = streamapp.NewClients(streamapp.NewDeployment(streamapp.DeploymentOptions{
		APIKey: streamKey, Secret: "deploy-secret", BaseURL: s.chat.URL,
	}), streamapp.ClientsOptions{})
	var err error
	s.cards, err = omnichannel.New(omnichannel.Options{Store: s.store, Stream: s.apps})
	s.Require().NoError(err)
	s.customerID, s.configID = "cards-"+uuid.NewString(), "config-"+uuid.NewString()
	s.now = time.Now().UTC().Truncate(time.Second)
	s.model = &recordingLLM{}
	s.manager = s.managerOver()
}

// managerOver is a session manager over the suite's Postgres and Stream Chat, with stub
// speech and the recording model.
func (s *EpisodeReadingSuite) managerOver() *Manager {
	logger := slog.New(slog.DiscardHandler)
	transcription := sttrouter.NewRegistry()
	transcription.Register("stub", func(routing.Spec) (stt.STT, error) { return &stubSTT{emitter: stt.NewEmitter(64)}, nil })
	transcriber, err := sttrouter.New(sttrouter.Options{Config: stubConfig(), Registry: transcription, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(transcriber.Close)
	reasoning := llmrouter.NewRegistry()
	reasoning.Register("stub", func(routing.Spec) (llmrouter.Provider, error) { return s.model, nil })
	reasoner, err := llmrouter.New(llmrouter.Options{Config: stubConfig(), Registry: reasoning, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(reasoner.Close)
	speech := ttsrouter.NewRegistry()
	speech.Register("stub", func(routing.Spec) (tts.TTS, error) { return &stubTTS{emitter: tts.NewEmitter(64)}, nil })
	speaker, err := ttsrouter.New(ttsrouter.Options{Config: stubConfig(), Registry: speech, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(speaker.Close)

	manager, err := NewManager(ManagerOptions{
		LLM: reasoner, STT: transcriber, TTS: speaker, Store: s.store, Stream: s.apps,
		Conversations: persistent.NewForChats(persistent.StreamApps(s.apps)),
		Logger:        logger,
		Edge: func(context.Context, Spec, streamapp.Bound, *slog.Logger) (agent.Edge, error) {
			return newQuietEdge(), nil
		},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { manager.Shutdown() })
	return manager
}

// recordingLLM answers every request with one line and keeps them all, from every session.
type recordingLLM struct {
	mu    sync.Mutex
	asked []llm.ResponseParams
}

func (r *recordingLLM) Start(context.Context) error { return nil }
func (r *recordingLLM) Create(_ context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	r.mu.Lock()
	r.asked = append(r.asked, params)
	r.mu.Unlock()
	script := llmtest.New(llm.StreamOptions{ResponseID: params.ID, Provider: "stub", Model: "stub-llm"})
	script.OutputText("Noted.")
	script.Done()
	return script.Stream(), nil
}
func (r *recordingLLM) Provider() string               { return "stub" }
func (r *recordingLLM) Model() string                  { return "stub-llm" }
func (r *recordingLLM) Capabilities() llm.Capabilities { return llm.Capabilities{} }
func (r *recordingLLM) Close() error                   { return nil }

// replyTo is the request the conversation model was sent to answer text.
func (s *EpisodeReadingSuite) replyTo(text string) llm.ResponseParams {
	var found llm.ResponseParams
	s.Require().Eventually(func() bool {
		s.model.mu.Lock()
		defer s.model.mu.Unlock()
		for _, params := range s.model.asked {
			if n := len(params.Input); n > 0 && params.Input[n-1].Role == llm.User && params.Input[n-1].Content == text {
				found = params
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the model was never asked to answer %q", text)
	return found
}

// handed is the cards the model was handed before the conversation, or nil for none.
func (s *EpisodeReadingSuite) handed(params llm.ResponseParams) []map[string]any {
	for i, message := range params.Input {
		if message.Role != llm.System || !strings.HasPrefix(message.Content, cardsNote) {
			continue
		}
		s.Require().Greater(len(params.Input), i+1)
		var envelope struct {
			Episodes []map[string]any `json:"episodes"`
		}
		s.Require().NoError(json.Unmarshal([]byte(params.Input[i+1].Content), &envelope))
		return envelope.Episodes
	}
	return nil
}

// at dates every Stream Chat message written from now on, seconds after the test's start.
func (s *EpisodeReadingSuite) at(seconds int) time.Time {
	at := s.now.Add(time.Duration(seconds) * time.Second)
	s.chat.At(at)
	return at
}

// channel makes an agent channel of the customer as its creator does: a thread channel as
// the channel bridge makes it, as the author of its first message, or a call channel as
// chatlog does, as the agent.
func (s *EpisodeReadingSuite) channel(id, creator string) string {
	_, err := s.chat.Client.Chat().GetOrCreateChannel(s.ctx, "agent", id, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{CreatedByID: &creator, Custom: map[string]any{
			"agent_config_id": s.configID, persistent.CustomerField: s.customerID, "support_agent_id": id,
		}},
	})
	s.Require().NoError(err)
	return "agent:" + id
}

// thread is a new thread channel, started by author.
func (s *EpisodeReadingSuite) thread(author string) string {
	return s.channel(persistent.ThreadChannelPrefix+uuid.NewString(), author)
}

// write puts a message into a channel as user, with the custom fields given: none for a
// person on the bridge, chatlog's source for a call's lines.
func (s *EpisodeReadingSuite) write(cid, user, text string, custom map[string]any) {
	_, err := s.chat.Client.Chat().SendMessage(s.ctx, "agent", strings.TrimPrefix(cid, "agent:"), &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{Text: &text, UserID: &user, Custom: custom},
	})
	s.Require().NoError(err)
}

// said is a line of a call, as chatlog writes it: the person's transcribed speech, or the
// agent's reply.
func said(agentSpoke bool) map[string]any {
	if agentSpoke {
		return map[string]any{chatlog.SourceField: chatlog.SourceAgent, "generating": false}
	}
	return map[string]any{chatlog.SourceField: chatlog.SourceSpeech}
}

// episode opens an episode of the person and writes its card, as the bridge and the call
// path do, under the suite's agent unless config names another.
func (s *EpisodeReadingSuite) episode(episode omnichannel.Episode) omnichannel.Opened {
	if episode.CustomerID == "" {
		episode.CustomerID = s.customerID
	}
	if episode.AgentConfigID == "" {
		episode.AgentConfigID = s.configID
	}
	episode.AgentName = "Athena"
	opened, err := s.cards.Open(s.ctx, episode)
	s.Require().NoError(err)
	s.Require().NoError(s.cards.Write(s.ctx, opened))
	return opened
}

// summarize closes an episode with a summary, as T55 will: the card's text and the status.
func (s *EpisodeReadingSuite) summarize(opened omnichannel.Opened, summary string) {
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE episodes SET status = ?, ended_at = ? WHERE id = ?",
		store.EpisodeSummarized, s.now, opened.Episode.ID)
	s.Require().NoError(err)
	_, err = s.chat.Client.Chat().UpdateMessagePartial(s.ctx, opened.Episode.CardMessageID, &getstream.UpdateMessagePartialRequest{
		Set: map[string]any{"text": summary, "status": store.EpisodeSummarized},
	})
	s.Require().NoError(err)
}

// phone is the person a number is.
func (s *EpisodeReadingSuite) phone(number string) omnichannel.Person {
	person, err := omnichannel.Phone(number)
	s.Require().NoError(err)
	return person
}

// link points a Slack user at a phone's omni-channel, as account linking will.
func (s *EpisodeReadingSuite) link(slack omnichannel.Person, phone omnichannel.Opened) {
	_, err := s.store.MapContact(s.ctx, &store.ContactMapEntry{
		CustomerID: s.customerID, AgentConfigID: s.configID, Kind: slack.Kind, Address: slack.Address,
		ConversationID: phone.Contact.ConversationID,
	})
	s.Require().NoError(err)
}

// smsThread is a new SMS thread from number, as the bridge will open it: the episode, then
// the person's message in the thread channel.
func (s *EpisodeReadingSuite) smsThread(number, author, text string) (string, omnichannel.Opened) {
	thread := s.thread(author)
	opened := s.episode(omnichannel.Episode{Person: s.phone(number), Source: "sms", ThreadChannel: thread, StartedAt: s.chatNow()})
	s.write(thread, author, text, nil)
	return thread, opened
}

// chatNow is the time the suite's Stream Chat dates messages by.
func (s *EpisodeReadingSuite) chatNow() time.Time {
	return s.now
}

// textSession opens the session on a thread channel the message hook opens, and tells it
// text, as api.answerThread does.
func (s *EpisodeReadingSuite) textSession(thread string, cards bool, text string) llm.ResponseParams {
	spec := Spec{
		CustomerID: s.customerID, ConfigID: s.configID, AgentName: "Athena", EpisodeCards: cards,
		Text: true, PersistConversation: true, ConversationID: thread,
		LLMTarget: "en-low-latency", Instructions: "be brief",
	}
	created, err := s.manager.Create(persistent.RouterOpensThread(s.ctx, thread), spec)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _, _ = s.manager.Close(created.ID(), OwnerOf(created.Spec())) })
	s.Require().NoError(created.FollowUp(s.ctx, text))
	return s.replyTo(text)
}

// voiceSession opens a session on a call a SIP caller is on, as the call hook dispatches it,
// and has it answer text.
func (s *EpisodeReadingSuite) voiceSession(caller string, cards bool, text string) (llm.ResponseParams, string) {
	call := "call-" + uuid.NewString()
	s.chat.PutCall("agent", call, caller)
	created, err := s.manager.Create(s.ctx, Spec{
		CustomerID: s.customerID, ConfigID: s.configID, AgentName: "Athena", EpisodeCards: cards, CallID: call,
		LLMTarget: "en-low-latency", STTTarget: "en-low-latency", TTSTarget: "en-low-latency", Instructions: "be brief",
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _, _ = s.manager.Close(created.ID(), OwnerOf(created.Spec())) })
	_, err = created.Respond(s.ctx, text, nil)
	s.Require().NoError(err)
	return s.replyTo(text), call
}

// requests is every request made to Stream so far.
func (s *EpisodeReadingSuite) requests() []chattest.Request {
	return s.chat.Requests(streamKey)
}

// namingAny is the requests among these that name one of the ids: a channel, a call or a message.
func namingAny(requests []chattest.Request, ids ...string) []string {
	var named []string
	for _, request := range requests {
		for _, id := range ids {
			if strings.Contains(request.Path, strings.TrimPrefix(id, "agent:")) {
				named = append(named, request.Method+" "+request.Path)
			}
		}
	}
	return named
}

// threadQueries is how many times a thread channel was read whole.
func threadQueries(requests []chattest.Request, thread string) int {
	count := 0
	for _, request := range requests {
		if request.Method == http.MethodPost && strings.HasSuffix(request.Path, "/channels/agent/"+strings.TrimPrefix(thread, "agent:")+"/query") {
			count++
		}
	}
	return count
}

// conversationOf is what the model is handed, rendered for comparing two requests whole.
func conversationOf(params llm.ResponseParams) string {
	encoded, _ := json.Marshal(struct {
		Instructions string
		Input        []llm.Message
		Tools        []llm.Tool
	}{params.Instructions, params.Input, params.Tools})
	return string(encoded)
}

// T56 acceptance: a session on a new SMS thread sees the card of an earlier Slack thread of
// the same person.
func (s *EpisodeReadingSuite) TestANewSMSThreadSeesTheCardOfAnEarlierSlackThread() {
	s.at(0)
	call := s.episode(omnichannel.Episode{Person: s.phone("+15550100100"), Source: store.EpisodeCall, ThreadChannel: "agent:phone-" + uuid.NewString(),
		CallID: "call-old", SessionID: "session-" + uuid.NewString(), StartedAt: s.now.Add(-2 * time.Hour)})
	slack, err := omnichannel.SlackUser("T1", "U1")
	s.Require().NoError(err)
	s.link(slack, call)
	slackThread := s.thread("slack-author")
	earlier := s.episode(omnichannel.Episode{Person: slack, Source: store.EpisodeSlack, ThreadChannel: slackThread, StartedAt: s.now.Add(-time.Hour)})
	s.summarize(earlier, "Asked for the March invoice; the agent sent it.")
	s.at(10)
	thread, _ := s.smsThread("+15550100100", "sms-author", "and April?")

	handed := s.handed(s.textSession(thread, true, "and April?"))

	s.Require().NotEmpty(handed)
	newest := handed[len(handed)-1]
	s.Equal("slack", newest["source"])
	s.Equal("summarized", newest["status"])
	s.Equal("Asked for the March invoice; the agent sent it.", newest["summary"])
	for _, card := range handed {
		s.NotEqual("sms", card["source"], "the session's own thread is read word for word, not as a card")
	}
}

// T56 acceptance: an SMS seconds after a call reads the call's last raw lines, since the
// call's summary is not ready. A call channel named after the number rung holds every
// caller's calls, so the lines of the caller before and of the caller after are not this
// call's.
func (s *EpisodeReadingSuite) TestAnSMSSecondsAfterACallReadsTheCallsLastLines() {
	callChannel := s.channel("phone-"+uuid.NewString(), "agent-user")
	s.at(-120)
	s.write(callChannel, "sip-+15550100199", "this was somebody else's call", said(false))
	s.at(-60)
	s.episode(omnichannel.Episode{Person: s.phone("+15550100100"), Source: store.EpisodeCall, ThreadChannel: callChannel,
		CallID: "call-now", SessionID: "session-" + uuid.NewString(), StartedAt: s.now.Add(-60 * time.Second)})
	s.at(-50)
	s.write(callChannel, "sip-+15550100100", "book a cleaning for Thursday", said(false))
	s.at(-40)
	s.write(callChannel, "agent-user", "Booked: Thursday at 15:00.", said(true))
	s.episode(omnichannel.Episode{Person: s.phone("+15550100177"), Source: store.EpisodeCall, ThreadChannel: callChannel,
		CallID: "call-now", SessionID: "session-" + uuid.NewString(), StartedAt: s.now.Add(-30 * time.Second)})
	s.at(-20)
	s.write(callChannel, "sip-+15550100177", "the caller after", said(false))
	s.at(10)
	thread, _ := s.smsThread("+15550100100", "sms-author", "can I move it to 4?")

	handed := s.handed(s.textSession(thread, true, "can I move it to 4?"))

	s.Require().Len(handed, 1)
	s.Equal("call", handed[0]["source"])
	s.Equal("in_progress", handed[0]["status"])
	s.Nil(handed[0]["summary"])
	s.Equal([]any{
		map[string]any{"from": "person", "display_name": "sip-+15550100100", "text": "book a cleaning for Thursday"},
		map[string]any{"from": "agent", "text": "Booked: Thursday at 15:00."},
	}, handed[0]["lines"])
}

// Two callers on one call id at once (whether Stream does that is unverified, the episodes
// migration says), or a line written late at the window's edge: a call card's window holds a
// person's line by somebody other than its caller. Neither caller is handed the other's
// words, so the card gives no lines.
func (s *EpisodeReadingSuite) TestProbeAnotherCallersLinesInTheWindow() {
	callChannel := s.channel("phone-"+uuid.NewString(), "agent-user")
	s.at(-60)
	s.episode(omnichannel.Episode{Person: s.phone("+15550100100"), Source: store.EpisodeCall, ThreadChannel: callChannel,
		CallID: "call-shared", SessionID: "session-" + uuid.NewString(), StartedAt: s.now.Add(-60 * time.Second)})
	s.at(-50)
	s.write(callChannel, "sip-+15550100100", "I need an appointment", said(false))
	s.at(-45)
	s.write(callChannel, "sip-+15550100199", "BOB: my date of birth is 1 May 1980", said(false))
	s.at(-40)
	s.write(callChannel, "agent-user", "Booked, Bob, for 1 May.", said(true))
	s.at(10)
	thread, _ := s.smsThread("+15550100100", "sms-author", "when is it?")

	params := s.textSession(thread, true, "when is it?")

	for _, message := range params.Input {
		s.NotContains(message.Content, "1 May", "another caller's words reached this person's session")
	}
	s.Nil(s.handed(params), "a card with no line and no summary says nothing")
}

// The last lines are the last said, whatever order Stream answers a channel in: the suite's
// Stream answers in the order messages were written, here the reverse of when they are dated.
func (s *EpisodeReadingSuite) TestTheLastLinesAreTheLastSaidWhateverOrderStreamAnswersIn() {
	earlier := s.thread("sms-author")
	s.at(-1000)
	s.episode(omnichannel.Episode{Person: s.phone("+15550100100"), Source: "sms", ThreadChannel: earlier, StartedAt: s.now.Add(-1000 * time.Second)})
	for line := 24; line >= 0; line-- {
		s.at(-900 + line)
		s.write(earlier, "sms-author", fmt.Sprintf("line %d", line), nil)
	}
	s.at(0)
	thread, _ := s.smsThread("+15550100100", "sms-author", "hello")

	handed := s.handed(s.textSession(thread, true, "hello"))

	s.Require().Len(handed, 1)
	lines, _ := handed[0]["lines"].([]any)
	s.Require().Len(lines, 20)
	for i, said := range lines {
		s.Equal(fmt.Sprintf("line %d", i+5), said.(map[string]any)["text"])
	}
}

// A native speech-to-speech session reads no cards: it is handed its history as a transcript
// in its instructions, which keeps no note that the cards are not authority. A cascaded
// session of the same call reads them.
func (s *EpisodeReadingSuite) TestANativeSessionReadsNoCards() {
	s.at(-30)
	s.smsThread("+15550100100", "sms-author", "is the clinic open on Sunday?")
	call := "call-" + uuid.NewString()
	s.chat.PutCall("agent", call, "sip-+15550100100")
	bound, err := s.apps.For(s.ctx, s.customerID)
	s.Require().NoError(err)
	spec := Spec{ID: uuid.NewString(), CustomerID: s.customerID, ConfigID: s.configID, EpisodeCards: true, CallID: call, CallType: "agent"}
	asked := len(s.requests())

	native := spec
	native.STSTarget = "sts-native"
	s.Nil(s.manager.cards.read(s.ctx, native, bound))
	s.Empty(namingAny(s.requests()[asked:], "/video/call/agent/"+call), "the call is not even read")

	s.NotEmpty(s.manager.cards.read(s.ctx, spec, bound), "the same call cascaded reads the SMS card")
}

// With the cards on, a call whose Stream does not answer who is on it joins within the read's
// budget, with no cards, rather than after the Stream client's own 30 s.
func (s *EpisodeReadingSuite) TestASlowCallReadDelaysTheJoinByTheBudgetAtMost() {
	s.at(-30)
	s.smsThread("+15550100100", "sms-author", "is the clinic open on Sunday?")
	call := "call-" + uuid.NewString()
	s.chat.PutCall("agent", call, "sip-+15550100100")
	waiting, release := s.chat.Hold("/video/call/agent/" + call)
	defer release()

	started := time.Now()
	created, err := s.manager.Create(s.ctx, Spec{
		CustomerID: s.customerID, ConfigID: s.configID, AgentName: "Athena", EpisodeCards: true, CallID: call,
		LLMTarget: "en-low-latency", STTTarget: "en-low-latency", TTSTarget: "en-low-latency", Instructions: "be brief",
	})
	took := time.Since(started)

	s.Require().NoError(err)
	s.T().Cleanup(func() { _, _ = s.manager.Close(created.ID(), OwnerOf(created.Spec())) })
	<-waiting
	s.GreaterOrEqual(took, omnichannel.ReadTimeout, "the call read was held")
	s.Less(took, omnichannel.ReadTimeout+2*time.Second, "the join waited no longer than the budget")
	_, err = created.Respond(s.ctx, "hello", nil)
	s.Require().NoError(err)
	s.Nil(s.handed(s.replyTo("hello")))
}

// T42 acceptance: a call after an SMS thread starts with the SMS card in its context.
func (s *EpisodeReadingSuite) TestACallAfterAnSMSThreadStartsWithTheSMSCard() {
	s.at(-30)
	s.smsThread("+15550100100", "sms-author", "is the clinic open on Sunday?")
	s.at(-20)

	params, _ := s.voiceSession("sip-+15550100100", true, "hello")

	handed := s.handed(params)
	s.Require().Len(handed, 1)
	s.Equal("sms", handed[0]["source"])
	s.Equal([]any{map[string]any{"from": "person", "display_name": "sms-author", "text": "is the clinic open on Sunday?"}}, handed[0]["lines"])
}

// Rule 2: with episode_cards off a text session is handed exactly what it was before the
// cards existed, though the person has cards, and Stream is asked nothing of them.
func (s *EpisodeReadingSuite) TestWithTheCardsOffATextSessionIsAsBefore() {
	s.at(-60)
	callChannel := s.channel("phone-"+uuid.NewString(), "agent-user")
	call := s.episode(omnichannel.Episode{Person: s.phone("+15550100100"), Source: store.EpisodeCall, ThreadChannel: callChannel,
		CallID: "call-old", SessionID: "session-" + uuid.NewString(), StartedAt: s.now.Add(-60 * time.Second)})
	s.write(callChannel, "sip-+15550100100", "book a cleaning", said(false))
	s.at(0)
	withCards, _ := s.smsThread("+15550100100", "sms-author", "can I move it?")
	// The same thread with nobody the contact map knows: what any session read before.
	before := s.thread("sms-author")
	s.write(before, "sms-author", "can I move it?", nil)
	asked := len(s.requests())

	off := s.textSession(withCards, false, "can I move it?")
	offRequests := s.requests()[asked:]
	asked = len(s.requests())
	s.model.mu.Lock()
	s.model.asked = nil
	s.model.mu.Unlock()
	base := s.textSession(before, false, "can I move it?")
	baseRequests := s.requests()[asked:]

	s.Nil(s.handed(off))
	s.Equal(conversationOf(base), conversationOf(off), "the model is handed byte for byte what it was before")
	s.Empty(namingAny(offRequests, callChannel, call.Contact.ConversationID, call.Episode.CardMessageID, "/video/call/"))
	s.Equal(threadQueries(baseRequests, before), threadQueries(offRequests, withCards), "the thread is read as often as before")
}

// Rule 2 for a call: with episode_cards off the call is not read for its caller, no card is
// read, and the model is handed what it was before.
func (s *EpisodeReadingSuite) TestWithTheCardsOffAVoiceSessionIsAsBefore() {
	s.at(-30)
	_, sms := s.smsThread("+15550100100", "sms-author", "is the clinic open on Sunday?")
	asked := len(s.requests())

	off, call := s.voiceSession("sip-+15550100100", false, "hello")
	offRequests := s.requests()[asked:]
	s.model.mu.Lock()
	s.model.asked = nil
	s.model.mu.Unlock()
	base, _ := s.voiceSession("sip-+15550100188", false, "hello")

	s.Nil(s.handed(off))
	s.Equal(conversationOf(base), conversationOf(off))
	s.Empty(namingAny(offRequests, sms.Episode.ThreadChannel, sms.Contact.ConversationID, sms.Episode.CardMessageID, "/video/call/"+"agent/"+call))
}

// A persistent voice conversation is still refused, whatever the config says of the cards: a
// voice session reads the cards, not a conversation.
func (s *EpisodeReadingSuite) TestAPersistentVoiceConversationIsStillRefused() {
	for _, cards := range []bool{false, true} {
		_, err := s.manager.Create(s.ctx, Spec{
			CustomerID: s.customerID, ConfigID: s.configID, EpisodeCards: cards, CallID: "call-" + uuid.NewString(),
			PersistConversation: true, LLMTarget: "en-low-latency", STTTarget: "en-low-latency", TTSTarget: "en-low-latency",
		})
		s.Require().Error(err)
		s.Contains(err.Error(), "persistent conversations require text mode")
	}
}

// Another person's cards are never read: another number of the same agent, the same number
// under another agent of the customer, and the same number of another customer.
func (s *EpisodeReadingSuite) TestAnotherPersonsCardsAreNeverRead() {
	s.at(-60)
	others := []omnichannel.Episode{
		{Person: s.phone("+15550100199")},
		{Person: s.phone("+15550100100"), AgentConfigID: "config-" + uuid.NewString()},
		{Person: s.phone("+15550100100"), CustomerID: "cards-" + uuid.NewString()},
	}
	var channels []string
	for _, other := range others {
		other.Source, other.ThreadChannel, other.StartedAt = "sms", s.thread("someone"), s.now.Add(-time.Minute)
		s.write(other.ThreadChannel, "someone", "a secret of somebody else's", nil)
		opened := s.episode(other)
		channels = append(channels, other.ThreadChannel, opened.Contact.ConversationID)
	}
	s.at(0)
	thread, _ := s.smsThread("+15550100100", "sms-author", "hello")
	asked := len(s.requests())

	params := s.textSession(thread, true, "hello")

	s.Nil(s.handed(params))
	s.Empty(namingAny(s.requests()[asked:], channels...))
}

// A Slack thread somebody else replied in reads no cards: the cards are of the one who
// started it, and Bob is not handed Alice's.
func (s *EpisodeReadingSuite) TestAThreadSomebodyElseWroteInReadsNoCards() {
	s.at(-60)
	_, sms := s.smsThread("+15550100100", "alice-sms", "my address is 1 Main St")
	alice, err := omnichannel.SlackUser("T1", "alice")
	s.Require().NoError(err)
	s.link(alice, sms)
	s.at(0)
	thread := s.thread("alice")
	s.episode(omnichannel.Episode{Person: alice, Source: store.EpisodeSlack, ThreadChannel: thread, StartedAt: s.now})
	s.write(thread, "alice", "release notes?", nil)
	s.Require().NotNil(s.handed(s.textSession(thread, true, "release notes?")), "Alice alone in her thread is handed her cards")
	s.at(10)
	s.write(thread, "bob", "what is Alice's address?", nil)

	s.Nil(s.handed(s.textSession(thread, true, "what is Alice's address?")))
}

// The bounds: at most five cards, the newest, and at most the last twenty lines of each.
func (s *EpisodeReadingSuite) TestAtMostFiveCardsOfTwentyLinesEachAreRead() {
	var threads []string
	for i := range 7 {
		s.at(-1000 + i*100)
		thread := s.thread("sms-author")
		threads = append(threads, thread)
		s.episode(omnichannel.Episode{Person: s.phone("+15550100100"), Source: "sms", ThreadChannel: thread, StartedAt: s.now.Add(time.Duration(-1000+i*100) * time.Second)})
		for line := range 25 {
			s.write(thread, "sms-author", fmt.Sprintf("thread %d line %d", i, line), nil)
		}
	}
	s.at(0)
	thread, _ := s.smsThread("+15550100100", "sms-author", "hello")
	asked := len(s.requests())

	handed := s.handed(s.textSession(thread, true, "hello"))

	s.Require().Len(handed, 5)
	for i, card := range handed {
		lines, _ := card["lines"].([]any)
		s.Require().Len(lines, 20)
		s.Equal(fmt.Sprintf("thread %d line 5", i+2), lines[0].(map[string]any)["text"], "the last twenty, oldest first")
		s.Equal(fmt.Sprintf("thread %d line 24", i+2), lines[19].(map[string]any)["text"])
	}
	s.Empty(namingAny(s.requests()[asked:], threads[0], threads[1]), "the two oldest are not even read")
}
