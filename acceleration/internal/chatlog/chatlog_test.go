package chatlog

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"log/slog"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type ChatLogSuite struct {
	suite.Suite
	log *Log
}

func TestChatLogSuite(t *testing.T) {
	suite.Run(t, new(ChatLogSuite))
}

func (s *ChatLogSuite) SetupTest() {
	log, err := New(Options{
		AgentID:   "agent-1",
		Agent:     User{ID: "vision-agent", Name: "Vision Agent"},
		APIKey:    "key",
		APISecret: "secret",
		Logger:    slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.log = log
}

// useChat points the log at Chat in memory, with its channel there the way Start leaves it.
func (s *ChatLogSuite) useChat() *chattest.Server {
	chat := chattest.NewServer(s.T())
	s.log.client = chat.Client
	_, err := chat.Client.Chat().GetOrCreateChannel(context.Background(), ChannelType, s.log.channel,
		&getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{CreatedByID: &s.log.agent.ID}})
	s.Require().NoError(err)
	return chat
}

// queued returns what is waiting to be written. Nothing is started, so nothing drains it.
func (s *ChatLogSuite) queued() []message {
	var waiting []message
	for {
		select {
		case queued := <-s.log.queue:
			waiting = append(waiting, queued)
		case <-time.After(10 * time.Millisecond):
			return waiting
		}
	}
}

func (s *ChatLogSuite) TestAnAgentIdIsRequiredBecauseItNamesTheChannel() {
	_, err := New(Options{Agent: User{ID: "vision-agent"}, APIKey: "key", APISecret: "secret"})

	s.ErrorContains(err, "agent id")
}

func (s *ChatLogSuite) TestCredentialsAreRequiredBecauseTheseAreServerSideWrites() {
	// Nothing is read from the environment, which names the deployment's app rather than
	// the one a session is in.
	s.T().Setenv("STREAM_API_KEY", "deploy-key")
	s.T().Setenv("STREAM_API_SECRET", "deploy-secret")

	_, err := New(Options{AgentID: "agent-1", Agent: User{ID: "vision-agent"}})

	s.ErrorContains(err, "api key and secret")
}

func (s *ChatLogSuite) TestTheTranscriptIsStoredUnderTheAgentId() {
	s.Equal("agent-1", s.log.ChannelID())
}

func (s *ChatLogSuite) TestABoundConversationStoresTheTranscriptOnThatChannel() {
	log, err := New(Options{
		AgentID:   "agent-1",
		Channel:   "support-accafc35",
		Agent:     User{ID: "vision-agent", Name: "Vision Agent"},
		APIKey:    "key",
		APISecret: "secret",
		Logger:    slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.Equal("support-accafc35", log.ChannelID())
	s.True(log.existing)
}

func (s *ChatLogSuite) TestAParticipantIsTheAuthorOfWhatTheySaid() {
	s.log.Record(agent.Heard{
		Participant: stt.Participant{ID: "session-9", UserID: "alice", Name: "Alice"},
		Text:        "hello",
	})

	waiting := s.queued()
	s.Require().Len(waiting, 1)
	s.Equal(User{ID: "alice", Name: "Alice"}, waiting[0].author,
		"the user id identifies a speaker across calls, the session id would not")
	s.Equal("hello", waiting[0].text)
}

func (s *ChatLogSuite) TestAParticipantWithoutAUserIdFallsBackToTheirSession() {
	s.log.Record(agent.Heard{Participant: stt.Participant{ID: "session-9"}, Text: "hello"})

	waiting := s.queued()
	s.Require().Len(waiting, 1)
	s.Equal("session-9", waiting[0].author.ID)
}

func (s *ChatLogSuite) TestTheAgentIsTheAuthorOfItsOwnReplies() {
	s.log.Record(agent.Responded{TurnID: "turn-1", Text: "hi there"})

	waiting := s.queued()
	s.Require().Len(waiting, 1)
	s.Equal("vision-agent", waiting[0].author.ID)
	s.Equal("hi there", waiting[0].text)
	s.Equal(prepared, waiting[0].kind, "the model finishing is not the same as the caller hearing it")
}

func (s *ChatLogSuite) TestSpeechFinishingStoresTheSpokenReply() {
	s.log.Record(agent.Spoke{TurnID: "turn-1"})

	waiting := s.queued()
	s.Require().Len(waiting, 1)
	s.Equal(spoken, waiting[0].kind)
	s.Equal("turn-1", waiting[0].turnID)
}

func (s *ChatLogSuite) TestOnlySpeechIsStored() {
	s.log.Record(agent.Joined{At: time.Now()})
	s.log.Record(agent.Turn{TurnID: "turn-1", RoundtripMs: 120})

	s.Empty(s.queued(), "a transcript is what was said, not how the agent worked")
}

func (s *ChatLogSuite) TestAReplyIsWrittenAsItStreams() {
	s.log.Record(agent.ResponseDelta{TurnID: "turn-1", Text: "hi"})

	waiting := s.queued()
	s.Require().Len(waiting, 1, "a caller should not have to wait for the reply to finish")
	s.Equal(piece, waiting[0].kind)
	s.Equal("turn-1", waiting[0].turnID)
	s.Equal("vision-agent", waiting[0].author.ID)
}

func (s *ChatLogSuite) TestThePiecesOfAReplyAreOneMessage() {
	writer := newWriter(s.log)

	writer.handle(message{author: s.log.agent, text: "hi ", turnID: "turn-1", kind: piece})
	writer.handle(message{author: s.log.agent, text: "there", turnID: "turn-1", kind: piece})

	s.Require().Len(writer.writing, 1)
	s.Equal("hi there", writer.writing["turn-1"].text)
}

func (s *ChatLogSuite) TestARepliesPiecesAreKeptApartFromAnothers() {
	writer := newWriter(s.log)

	writer.handle(message{author: s.log.agent, text: "hi", turnID: "turn-1", kind: piece})
	writer.handle(message{author: s.log.agent, text: "bye", turnID: "turn-2", kind: piece})

	s.Equal("hi", writer.writing["turn-1"].text)
	s.Equal("bye", writer.writing["turn-2"].text)
}

func (s *ChatLogSuite) TestAnInterruptedReplyIsClosedOut() {
	s.log.Record(agent.Interrupted{TurnID: "turn-1"})

	waiting := s.queued()
	s.Require().Len(waiting, 1, "a reply nobody finished would say it was still coming forever")
	s.Equal(interrupt, waiting[0].kind)
	s.Equal("turn-1", waiting[0].turnID)
}

func (s *ChatLogSuite) TestAFinishedModelReplyStaysUnspokenUntilTheVoiceFinishes() {
	s.useChat()
	writer := newWriter(s.log)
	writer.handle(message{author: s.log.agent, text: "1, 2, 3, 4, 5", turnID: "turn-1", kind: piece})
	writer.handle(message{author: s.log.agent, text: "1, 2, 3, 4, 5", turnID: "turn-1", kind: prepared})
	s.Require().Len(writer.writing, 1)
	s.Equal("1, 2, 3, 4, 5", writer.writing["turn-1"].generated)
	writer.handle(message{author: s.log.agent, turnID: "turn-1", kind: spoken})
	s.Empty(writer.writing, "speech finishing is what stores the reply")
}

func (s *ChatLogSuite) TestASavedArtifactIsStoredEvenWithoutASpokenReply() {
	s.useChat()
	s.log.visible = []string{"save_*"}
	s.log.Record(agent.ToolRan{ID: "tool-one", TurnID: "turn-one", Tool: "save_canvas", Result: `{"schema_version":1,"status":"stored","publication":"pending","attachment":{"type":"canvas","artifact_id":"canvas-one","revision":2,"title":"Lilacs","sha256":"saved"}}`})
	queued := s.queued()
	s.Require().Len(queued, 1)
	writer := newWriter(s.log)

	writer.handle(queued[0])
	// Repeated delivery has the same stored identity rather than a second card.
	writer.handle(queued[0])

	id := fmt.Sprintf("voice-artifact-%x", sha256.Sum256([]byte(s.log.channel+"\x00turn-one\x00tool-one")))
	response, err := s.log.client.Chat().GetMessage(context.Background(), id, &getstream.GetMessageRequest{})
	s.Require().NoError(err)
	stored := response.Data.Message
	s.Equal(SourceAgent, stored.Custom[SourceField])
	s.Equal(false, stored.Custom[generatingField])
	s.Require().Len(stored.Attachments, 1)
	s.Equal("canvas", *stored.Attachments[0].Type)
	s.Equal("Lilacs", *stored.Attachments[0].Title)
	s.Equal("canvas-one", stored.Attachments[0].Custom["artifact_id"])
	s.Equal(float64(2), stored.Attachments[0].Custom["revision"])
	s.Empty(stored.Text, "the card must not duplicate the speech transcript")
}

func (s *ChatLogSuite) TestOnlyAShownToolsSuccessfulStoredReceiptMakesACard() {
	s.log.visible = []string{"save_*"}
	valid := `{"schema_version":1,"status":"stored","publication":"pending","attachment":{"type":"canvas","artifact_id":"canvas-one","revision":1,"title":"Lilacs"}}`
	for _, event := range []agent.ToolRan{
		{ID: "one", TurnID: "turn", Tool: "save_canvas", Result: valid, Err: errors.New("denied")},
		{ID: "two", TurnID: "turn", Tool: "export_crm", Result: valid},
		{ID: "three", TurnID: "turn", Tool: "save_canvas", Result: `{"status":"not_stored"}`},
		{ID: "four", Tool: "save_canvas", Result: valid},
	} {
		s.log.Record(event)
	}

	s.Empty(s.queued())
}

func (s *ChatLogSuite) TestAnInterruptedReplyIsNotStoredAsFullySpoken() {
	s.useChat()
	writer := newWriter(s.log)
	writer.handle(message{author: s.log.agent, text: "1, 2, 3, 4, 5, 6, 7, 8, 9, 10", turnID: "turn-1", kind: piece})
	writer.show()
	writing := writer.writing["turn-1"]
	s.Require().NotNil(writing)
	s.NotEmpty(writing.messageID)
	generated := "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
	writer.handle(message{author: s.log.agent, text: generated, turnID: "turn-1", kind: prepared})
	writer.handle(message{author: s.log.agent, turnID: "turn-1", kind: interrupt})
	s.Empty(writer.writing)
	response, err := s.log.client.Chat().GetMessage(context.Background(), writing.messageID, &getstream.GetMessageRequest{})
	s.Require().NoError(err)
	stored := response.Data.Message
	s.Equal(true, stored.Custom[interruptedField])
	s.Equal(false, stored.Custom[generatingField])
	s.NotEqual(generated, stored.Text, "unplayed model text must not look like a finished spoken reply")
}

func (s *ChatLogSuite) TestANativePartialAfterInterruptIsKeptAsInterrupted() {
	s.useChat()
	writer := newWriter(s.log)
	writer.handle(message{author: s.log.agent, turnID: "turn-1", kind: interrupt})
	writer.handle(message{author: s.log.agent, text: "One,", turnID: "turn-1", kind: prepared})
	s.Empty(writer.writing)
	state := true
	response, err := s.log.client.Chat().GetOrCreateChannel(context.Background(), ChannelType, s.log.channel,
		&getstream.GetOrCreateChannelRequest{State: &state})
	s.Require().NoError(err)
	s.Require().NotEmpty(response.Data.Messages)
	stored := response.Data.Messages[0]
	s.Equal("One,", stored.Text)
	s.Equal(true, stored.Custom[interruptedField])
	s.Equal(false, stored.Custom[generatingField])
}

func (s *ChatLogSuite) TestSilenceIsNotStored() {
	s.log.Record(agent.Responded{TurnID: "turn-1", Text: ""})

	s.Empty(s.queued())
}

func (s *ChatLogSuite) TestSpeechIsStoredAsSpeech() {
	// The channel is a way in as well as a record, so what the agent answered as it was
	// said has to be tellable from a question somebody has just written.
	s.log.Record(agent.Heard{
		Participant: stt.Participant{ID: "session-9", UserID: "alice"},
		Text:        "hello",
	})

	waiting := s.queued()
	s.Require().Len(waiting, 1)
	s.Equal(SourceSpeech, waiting[0].source)
}

func (s *ChatLogSuite) TestEverythingTheAgentSaysIsStoredAsTheAgents() {
	s.log.Record(agent.ResponseDelta{TurnID: "turn-1", Text: "hi"})
	s.log.Record(agent.Responded{TurnID: "turn-1", Text: "hi there"})
	s.log.Record(agent.Interrupted{TurnID: "turn-2"})

	waiting := s.queued()
	s.Require().Len(waiting, 3)
	for _, queued := range waiting {
		s.Equal(SourceAgent, queued.source, "otherwise the agent answers its own reply")
	}
}

func (s *ChatLogSuite) TestSomethingTheAgentWroteRatherThanSaidIsStillTheAgents() {
	s.log.Reply("reissuing to another company is not possible")

	waiting := s.queued()
	s.Require().Len(waiting, 1)
	s.Equal(whole, waiting[0].kind, "nothing streamed, so there is no turn to close out")
	s.Equal("vision-agent", waiting[0].author.ID)
	s.Equal(SourceAgent, waiting[0].source)
}

func (s *ChatLogSuite) TestTheTranscriptSaysWhichLinesTheAgentSaid() {
	// Speakers are user ids, and an agent can be named anything, so whoever reads the
	// transcript back cannot work out which side of the conversation a line is from.
	s.useChat()
	writer := newWriter(s.log)
	writer.handle(message{author: User{ID: "alice"}, text: "hello", kind: whole, source: SourceSpeech})
	writer.handle(message{author: s.log.agent, text: "hi there", kind: whole, source: SourceAgent})
	reader := &Reader{client: s.log.client}

	said, err := reader.Transcript(context.Background(), Read{Channel: "agent-1"})

	s.Require().NoError(err)
	s.Require().Len(said, 2)
	s.Equal("alice", said[0].Speaker)
	s.Empty(said[0].Name, "chat names a user after their id, and an id is not a name")
	s.False(said[0].Agent)
	s.Equal("vision-agent", said[1].Speaker)
	s.True(said[1].Agent)
}

// channel is what the channel holds, oldest first.
func (s *ChatLogSuite) channel() []getstream.MessageResponse {
	state := true
	response, err := s.log.client.Chat().GetOrCreateChannel(context.Background(), ChannelType, s.log.channel,
		&getstream.GetOrCreateChannelRequest{State: &state})
	s.Require().NoError(err)
	return response.Data.Messages
}

func (s *ChatLogSuite) TestWhatAParticipantIsSayingIsOneMessageThatSettles() {
	s.useChat()
	writer := newWriter(s.log)
	alice := User{ID: "alice"}

	writer.handle(message{author: alice, text: "where is", kind: hearing, source: SourceSpeech})
	writer.show()
	s.Require().Len(s.channel(), 1, "watchers see the words before the turn settles")
	s.Equal(true, s.channel()[0].Custom[generatingField])

	writer.handle(message{author: alice, text: "where is my order", kind: hearing, source: SourceSpeech})
	writer.show()
	writer.handle(message{author: alice, text: "Where is my order 1042?", kind: heard, source: SourceSpeech})

	stored := s.channel()
	s.Require().Len(stored, 1, "revisions update the message rather than adding one each")
	s.Equal("Where is my order 1042?", stored[0].Text)
	s.Equal(false, stored[0].Custom[generatingField])
	s.Equal(SourceSpeech, stored[0].Custom[SourceField])
	s.Empty(writer.listening)
}

func (s *ChatLogSuite) TestSpeechTheAgentIgnoredIsNotLeftInTheChannel() {
	s.useChat()
	writer := newWriter(s.log)
	writer.handle(message{author: User{ID: "alice"}, text: "hang on, the door", kind: hearing, source: SourceSpeech})
	writer.show()

	s.Require().Len(s.channel(), 1, "watchers saw the words while they were heard")

	writer.handle(message{author: User{ID: "alice"}, kind: ignored, source: SourceSpeech})

	s.Empty(s.channel(), "an emptied message would read as one somebody sent")
	s.Empty(writer.listening)
}

func (s *ChatLogSuite) TestSpeechThatNeverSettledIsRemovedWhenTheCallEnds() {
	s.useChat()
	writer := newWriter(s.log)
	writer.handle(message{author: User{ID: "alice"}, text: "and one more", kind: hearing, source: SourceSpeech})
	writer.show()

	writer.closeOut()

	s.Empty(s.channel(), "otherwise it says it is still being said forever, or is left empty")
}

func (s *ChatLogSuite) TestRetractingLeavesWhatWasSaidBefore() {
	s.useChat()
	writer := newWriter(s.log)
	alice := User{ID: "alice"}
	writer.handle(message{author: alice, text: "What time is it?", kind: heard, source: SourceSpeech})
	writer.handle(message{author: alice, text: "hang on", kind: hearing, source: SourceSpeech})
	writer.show()

	writer.handle(message{author: alice, kind: ignored, source: SourceSpeech})

	stored := s.channel()
	s.Require().Len(stored, 1)
	s.Equal("What time is it?", stored[0].Text)
}

func (s *ChatLogSuite) TestAnEmptyWrittenReplyIsNotStored() {
	s.log.Reply("")

	s.Empty(s.queued())
}

func (s *ChatLogSuite) TestATranscriptChannelIsStampedWithItsCustomer() {
	// Reading a transcript back checks whose channel it is, which only works if the
	// channel says so: a call row names an agent id, and an agent id is anybody's to pick.
	chat := chattest.NewServer(s.T())
	log, err := New(Options{
		AgentID: "agent-1", CustomerID: "customer-1",
		Agent: User{ID: "vision-agent"}, APIKey: "key", APISecret: "secret",
		Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	log.client = chat.Client

	s.Require().NoError(log.Start(context.Background()))
	log.Close()

	created, ok := chat.Channel("agent-1")
	s.Require().True(ok)
	s.Equal("customer-1", created["custom"].(map[string]any)[conversation.CustomerField])
}

func (s *ChatLogSuite) TestReadingATranscriptCreatesNoChannel() {
	// A get-or-create here would leave an empty channel in the app for every call that
	// was asked about and never written.
	chat := chattest.NewServer(s.T())
	reader := NewReaderFromClient(chat.Client)

	said, err := reader.Transcript(context.Background(), Read{Channel: "agent-never", Customer: "customer-1"})

	s.Require().NoError(err)
	s.Empty(said)
	_, exists := chat.Channel("agent-never")
	s.False(exists)
}

func (s *ChatLogSuite) TestATranscriptIsReadOnlyFromAChannelTheCustomerHolds() {
	chat := s.useChat()
	_, err := chat.Client.Chat().GetOrCreateChannel(context.Background(), ChannelType, s.log.channel,
		&getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{
			CreatedByID: &s.log.agent.ID, Custom: map[string]any{conversation.CustomerField: "customer-2"},
		}})
	s.Require().NoError(err)
	writer := newWriter(s.log)
	writer.handle(message{author: User{ID: "alice"}, text: "hello", kind: whole, source: SourceSpeech})
	reader := NewReaderFromClient(chat.Client)

	theirs, err := reader.Transcript(context.Background(), Read{Channel: s.log.channel, Customer: "customer-2"})
	s.Require().NoError(err)
	mine, err := reader.Transcript(context.Background(), Read{Channel: s.log.channel, Customer: "customer-1"})
	s.Require().NoError(err)

	s.Len(theirs, 1)
	s.Empty(mine, "a channel stamped for another customer is not this one's to read")
}

func (s *ChatLogSuite) TestAConversationBoundCallsTranscriptHoldsOnlyItsCall() {
	// A call bound to a conversation writes into that conversation's channel, beside what
	// was typed before and after it.
	chat := s.useChat()
	writer := newWriter(s.log)
	before := time.Date(2026, 4, 1, 9, 0, 0, 0, time.UTC)
	during := before.Add(10 * time.Minute)
	after := during.Add(10 * time.Minute)

	chat.At(before)
	s.typed(chat, "alice", "typed before the call")
	chat.At(during)
	writer.handle(message{author: User{ID: "alice"}, text: "said on the call", kind: whole, source: SourceSpeech})
	s.typed(chat, "agent-1", "the conversation's own reply, during the call")
	chat.At(after)
	s.typed(chat, "alice", "typed after the call")

	said, err := NewReaderFromClient(chat.Client).Transcript(context.Background(), Read{
		Channel: s.log.channel, Agent: "agent-1",
		From: during.Add(-time.Minute), To: during.Add(time.Minute),
	})

	s.Require().NoError(err)
	s.Require().Len(said, 2)
	s.Equal("said on the call", said[0].Text)
	s.False(said[0].Agent)
	s.Equal("the conversation's own reply, during the call", said[1].Text)
	s.True(said[1].Agent, "the conversation service writes no source; its author is the agent")
}

// typed writes a message the way the conversation service does, with no source.
func (s *ChatLogSuite) typed(chat *chattest.Server, user, text string) {
	_, err := chat.Client.Chat().SendMessage(context.Background(), ChannelType, s.log.channel,
		&getstream.SendMessageRequest{Message: getstream.MessageRequest{Text: &text, UserID: &user}})
	s.Require().NoError(err)
}

func (s *ChatLogSuite) TestATranscriptLeavesAnExistingParticipantAlone() {
	// A participant is a real person in the app. Writing what they said must not replace
	// their profile with the name the call knew them by.
	chat := s.useChat()
	chat.PutUser(map[string]any{"id": "alice", "name": "Alice Example", "image": "https://example.com/alice.png"})
	writer := newWriter(s.log)

	writer.handle(message{author: User{ID: "alice", Name: "alice"}, text: "hello", kind: whole, source: SourceSpeech})
	writer.handle(message{author: User{ID: "bob", Name: "Bob"}, text: "hi", kind: whole, source: SourceSpeech})

	alice, _ := chat.User("alice")
	s.Equal("Alice Example", alice["name"])
	s.Equal("https://example.com/alice.png", alice["image"])
	bob, created := chat.User("bob")
	s.Require().True(created, "somebody the app has never seen is still created")
	s.Equal("Bob", bob["name"])
}

// useTimings asks for timings, with Chat in memory.
func (s *ChatLogSuite) useTimings() {
	s.useChat()
	s.log.timings = true
}

// timing turns a turn into what the writer is handed for it, the way Record queues it.
func (s *ChatLogSuite) timing(turn agent.Turn) message {
	s.log.Record(turn)
	waiting := s.queued()
	s.Require().Len(waiting, 1)
	return waiting[0]
}

func (s *ChatLogSuite) TestTimingsAreOffUnlessAsked() {
	s.useChat()
	s.log.Record(agent.Turn{TurnID: "turn-1", RoundtripMs: 500, STTLatencyMs: 6})
	s.Empty(s.queued(), "a developer's option is not something every transcript carries")

	writer := newWriter(s.log)
	writer.handle(message{author: s.log.agent, text: "hi", turnID: "turn-1", kind: piece})
	writer.handle(message{author: s.log.agent, turnID: "turn-1", kind: spoken})

	s.Empty(writer.recent, "nothing is remembered for timings that were not asked for")
	stored := s.channel()
	s.Require().Len(stored, 1)
	s.Equal("hi", stored[0].Text)
	s.NotContains(stored[0].Custom, conversation.TimingsField)
}

func (s *ChatLogSuite) TestTimingsAreAnOptionOfTheLog() {
	log, err := New(Options{
		AgentID: "agent-1", Agent: User{ID: "vision-agent"}, APIKey: "key", APISecret: "secret",
		Timings: true, Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.True(log.timings)
	s.False(s.log.timings)
}

func (s *ChatLogSuite) TestATurnsTimingsAreQueuedForItsReply() {
	s.log.timings = true

	queued := s.timing(agent.Turn{TurnID: "turn-1", RoundtripMs: 500})

	s.Equal(timed, queued.kind)
	s.Equal("turn-1", queued.turnID)
	s.Equal("⏱ reply 500 ms", queued.timings.line)
}

func (s *ChatLogSuite) TestTheReplyIsSplitIntoStagesThatAddUpToIt() {
	described, ok := timingsOf(agent.Turn{
		TurnID: "turn-1", STTLatencyMs: 6.2, CadenceMs: 350, DecisionMs: 95.6, ModelToFirstTextMs: 412,
		LLMTTFTMs: 731, TextToTTSMs: 70, TTSToAudioMs: 143, TTSTTFBMs: 120, ReplyHoldMs: 300,
		RoundtripMs: 1074, SpeechEndToAudioMs: 1080, FirstFrameQueuedMs: 1080, FirstAudibleFrameMs: 1106,
		SpeechEndToAudibleMs: 1112, AudioOutMs: 2100,
	})

	s.Require().True(ok)
	s.Equal("⏱ reply 1112 ms = eou 452 + llm 412 + tts 248 · ttft 731 · ttfb 120 · hold 300", described.line)
	s.Equal(map[string]any{
		"interrupted": false,
		"reply_ms":    1112,
		"reply_heard": true,
		"eou_ms":      452,
		"stt_ms":      6,
		"wait_ms":     350,
		"eot_ms":      96,
		"llm_ms":      412,
		"tts_ms":      248,
		"llm_ttft_ms": 731,
		"tts_ttfb_ms": 120,
		"hold_ms":     300,
	}, described.fields)
	s.Equal(described.fields["reply_ms"], described.fields["eou_ms"].(int)+described.fields["llm_ms"].(int)+described.fields["tts_ms"].(int))
}

func (s *ChatLogSuite) TestAReplyReadyAtTheDecisionHasNoModelStage() {
	described, ok := timingsOf(agent.Turn{
		STTLatencyMs: 12, CadenceMs: 351, DecisionMs: 1934, TTSToAudioMs: 1040, RoundtripMs: 3325,
		SpeechEndToAudioMs: 3337, LLMTTFTMs: 850,
	})

	s.Require().True(ok)
	s.Equal("⏱ reply 3337 ms = eou 2297 + tts 1040 · ttft 850", described.line, "the model's wait was spent beside the decision")
}

func (s *ChatLogSuite) TestAnEdgeThatDoesNotSayWhenTheReplyWasHeardMeasuresToItsPublishing() {
	described, ok := timingsOf(agent.Turn{
		STTLatencyMs: 6, CadenceMs: 350, TTSToAudioMs: 143, RoundtripMs: 1074, SpeechEndToAudioMs: 1080,
	})

	s.Require().True(ok)
	s.Equal("⏱ reply 1080 ms = eou 356 + tts 143", described.line)
	s.Equal(false, described.fields["reply_heard"])
}

func (s *ChatLogSuite) TestAFigureThatDidNotHappenIsLeftOutRatherThanShownAsZero() {
	described, ok := timingsOf(agent.Turn{SpeechEndToAudioMs: 900, CadenceMs: 300, TTSToAudioMs: 80, DecisionMs: 0.2})

	s.Require().True(ok)
	s.Equal("⏱ reply 900 ms = eou 300 + tts 80", described.line)
	s.NotContains(described.fields, "stt_ms")
	s.NotContains(described.fields, "hold_ms")
	s.NotContains(described.fields, "eot_ms", "a figure that rounds to nothing did not take any time worth showing")
}

func (s *ChatLogSuite) TestTheRoundtripIsTheFigureWhenNeitherSpeechEndIsKnown() {
	described, ok := timingsOf(agent.Turn{RoundtripMs: 500, STTLatencyMs: 20})

	s.Require().True(ok)
	s.Equal("⏱ reply 500 ms", described.line, "the transcriber's settling is not in a figure that starts at the transcript")
}

func (s *ChatLogSuite) TestAnInterruptedTurnSaysSoAndKeepsWhateverWasMeasured() {
	described, ok := timingsOf(agent.Turn{Interrupted: true, STTLatencyMs: 6, CadenceMs: 300, LLMTTFTMs: 410})

	s.Require().True(ok)
	s.Equal("⏱ interrupted · eou 306 · ttft 410", described.line)
	s.Equal(true, described.fields["interrupted"])

	described, ok = timingsOf(agent.Turn{Interrupted: true})
	s.Require().True(ok)
	s.Equal("⏱ interrupted", described.line)
}

func (s *ChatLogSuite) TestATurnWithNothingToSayHasNoTimings() {
	_, ok := timingsOf(agent.Turn{TurnID: "turn-1"})

	s.False(ok)
}

func (s *ChatLogSuite) TestTimingsThatArriveBeforeTheReplyIsStoredWaitForIt() {
	s.useTimings()
	writer := newWriter(s.log)
	writer.handle(message{author: s.log.agent, text: "Hello there.", turnID: "turn-1", kind: piece})
	writer.show()
	s.Require().NotEmpty(writer.writing["turn-1"].messageID)

	writer.handle(s.timing(agent.Turn{TurnID: "turn-1", SpeechEndToAudioMs: 900, CadenceMs: 300}))

	s.Require().Len(s.channel(), 1)
	s.Equal("Hello there.", s.channel()[0].Text, "the reply is still being written")
	writer.handle(message{author: s.log.agent, text: "Hello there.", turnID: "turn-1", kind: prepared})
	writer.handle(message{author: s.log.agent, turnID: "turn-1", kind: spoken})

	stored := s.channel()
	s.Require().Len(stored, 1)
	s.Equal("Hello there.\n\n⏱ reply 900 ms = eou 300", stored[0].Text)
	s.Equal(false, stored[0].Custom[generatingField])
	s.Equal(float64(900), stored[0].Custom[conversation.TimingsField].(map[string]any)["reply_ms"])
	s.Empty(writer.recent, "the reply and its timings have met, so there is nothing to wait for")
}

func (s *ChatLogSuite) TestTimingsThatArriveAfterTheReplyIsStoredAreWrittenOntoIt() {
	s.useTimings()
	writer := newWriter(s.log)
	writer.handle(message{author: s.log.agent, text: "Hello there.", turnID: "turn-1", kind: piece})
	writer.show()
	writer.handle(message{author: s.log.agent, text: "Hello there.", turnID: "turn-1", kind: prepared})
	writer.handle(message{author: s.log.agent, turnID: "turn-1", kind: spoken})
	s.Require().Len(s.channel(), 1)
	s.Equal("Hello there.", s.channel()[0].Text)
	s.Require().Contains(writer.recent, "turn-1", "the reply is remembered for the timings that follow it")
	s.Equal("Hello there.", writer.recent["turn-1"].reply.text)

	writer.handle(s.timing(agent.Turn{TurnID: "turn-1", SpeechEndToAudioMs: 900, CadenceMs: 300}))

	stored := s.channel()
	s.Require().Len(stored, 1, "the reply is updated, not repeated")
	s.Equal("Hello there.\n\n⏱ reply 900 ms = eou 300", stored[0].Text)
	s.Equal(false, stored[0].Custom[interruptedField])
	s.Empty(writer.recent)
	s.Empty(writer.turns)
}

func (s *ChatLogSuite) TestTimingsAreWrittenOntoAReplyThatWasNeverShownWhileItStreamed() {
	s.useTimings()
	writer := newWriter(s.log)
	writer.handle(message{author: s.log.agent, text: "Hi.", turnID: "turn-1", kind: prepared})
	writer.handle(message{author: s.log.agent, turnID: "turn-1", kind: spoken})

	writer.handle(s.timing(agent.Turn{TurnID: "turn-1", RoundtripMs: 500}))

	stored := s.channel()
	s.Require().Len(stored, 1)
	s.Equal("Hi.\n\n⏱ reply 500 ms", stored[0].Text)
}

func (s *ChatLogSuite) TestAnInterruptedRepliesTimingsAreAllThatIsLeftOfIt() {
	s.useTimings()
	writer := newWriter(s.log)
	writer.handle(message{author: s.log.agent, text: "One, two, three", turnID: "turn-1", kind: piece})
	writer.show()
	writer.handle(s.timing(agent.Turn{TurnID: "turn-1", Interrupted: true, STTLatencyMs: 6}))

	writer.handle(message{author: s.log.agent, turnID: "turn-1", kind: interrupt})

	stored := s.channel()
	s.Require().Len(stored, 1)
	s.Equal("⏱ interrupted · eou 6", stored[0].Text, "the unplayed words are not put back")
	s.Equal(true, stored[0].Custom[interruptedField])
}

func (s *ChatLogSuite) TestAReplyThatLeftNoMessageGetsNoTimings() {
	s.useTimings()
	writer := newWriter(s.log)
	writer.handle(message{author: s.log.agent, turnID: "turn-1", kind: interrupt})

	writer.handle(s.timing(agent.Turn{TurnID: "turn-1", Interrupted: true, STTLatencyMs: 6}))

	s.Empty(s.channel(), "there is no message to show them on, and they are not a message of their own")
}

func (s *ChatLogSuite) TestTheTranscriptHoldsWhatTheAgentSaidWithoutItsTimings() {
	// Whoever reads the conversation back is reading what was said, and the line of timings
	// was never said.
	s.useTimings()
	writer := newWriter(s.log)
	writer.handle(message{author: User{ID: "alice"}, text: "hello", kind: whole, source: SourceSpeech})
	writer.handle(message{author: s.log.agent, text: "hi there", turnID: "turn-1", kind: prepared})
	writer.handle(message{author: s.log.agent, turnID: "turn-1", kind: spoken})
	writer.handle(s.timing(agent.Turn{TurnID: "turn-1", RoundtripMs: 500}))
	s.Require().Len(s.channel(), 2)
	s.Contains(s.channel()[1].Text, "⏱", "it is there for whoever is watching the call")

	said, err := NewReaderFromClient(s.log.client).Transcript(context.Background(), Read{Channel: s.log.channel})

	s.Require().NoError(err)
	s.Require().Len(said, 2)
	s.Equal("hello", said[0].Text)
	s.Equal("hi there", said[1].Text)
	s.True(said[1].Agent)
}

func (s *ChatLogSuite) TestWhatIsHeldForTimingsIsBounded() {
	s.useTimings()
	writer := newWriter(s.log)

	// Timings whose reply never arrives, as for a turn that was silent.
	for i := range 2 * recentTurns {
		turnID := fmt.Sprintf("turn-%d", i)
		writer.handle(message{turnID: turnID, kind: timed, timings: &timings{line: "⏱ 1 ms"}})
	}
	s.Len(writer.recent, recentTurns)
	s.Len(writer.turns, recentTurns)
	s.NotContains(writer.recent, "turn-0", "the oldest is let go first")
	s.Contains(writer.recent, fmt.Sprintf("turn-%d", 2*recentTurns-1))

	// Replies whose timings never arrive, as for an edge that does not report turns.
	for i := range 2 * recentTurns {
		turnID := fmt.Sprintf("reply-%d", i)
		writer.handle(message{author: s.log.agent, text: "hi", turnID: turnID, kind: prepared})
		writer.handle(message{author: s.log.agent, turnID: turnID, kind: spoken})
	}
	s.Len(writer.recent, recentTurns)
	s.Len(writer.turns, recentTurns)
	s.Contains(writer.recent, fmt.Sprintf("reply-%d", 2*recentTurns-1))
}
