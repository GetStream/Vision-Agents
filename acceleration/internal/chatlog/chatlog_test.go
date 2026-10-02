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
	s.T().Setenv("STREAM_API_KEY", "")
	s.T().Setenv("STREAM_API_SECRET", "")

	_, err := New(Options{AgentID: "agent-1", Agent: User{ID: "vision-agent"}})

	s.ErrorContains(err, "STREAM_API_KEY")
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
