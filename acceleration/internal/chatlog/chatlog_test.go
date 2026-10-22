package chatlog

import (
	"cmp"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
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

func (s *ChatLogSuite) TestALimitedAppIsNotAskedAgainOnEveryTick() {
	chat := s.useChat()
	writer := newWriter(s.log)
	chat.RateLimit("/message")

	writer.handle(message{author: s.log.agent, text: "hi", turnID: "turn-1", kind: piece})
	writer.show()
	writer.handle(message{author: s.log.agent, text: " there", turnID: "turn-1", kind: piece})
	writer.show()
	writer.show()

	sends := 0
	for _, asked := range chat.Requests("test") {
		if strings.HasSuffix(asked.Path, "/message") {
			sends++
		}
	}
	s.Equal(1, sends, "a 429 says when to ask again, and Stream closes the connection on each")

	chat.LiftRateLimit("/message")
	writer.handle(message{author: s.log.agent, turnID: "turn-1", kind: spoken})
	s.Equal([]string{"hi there"}, chat.Messages(s.log.channel), "the finished reply is still stored")
}

func (s *ChatLogSuite) TestASavedArtifactIsStoredEvenWithoutASpokenReply() {
	s.useChat()
	s.log.visible = []string{"save_*"}
	s.log.Record(agent.ToolRan{ID: "tool-one", TurnID: "turn-one", Tool: "save_canvas", Result: `{"schema_version":1,"status":"stored","publication":"pending","attachment":{"type":"canvas","artifact_id":"canvas-one","revision":2,"title":"Lilacs","sha256":"saved"}}`})
	queued := s.queued()
	s.Require().NotEmpty(queued)
	s.Require().Equal(artifact, queued[0].kind, "the card goes before the tool is done")
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

	for _, queued := range s.queued() {
		s.NotEqual(artifact, queued.kind, "tool call %s", queued.receiptID)
	}
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

func (s *ChatLogSuite) TestAnInterruptedReplyIsReadBackAsTalkedOver() {
	// The caller heard the start of it, and a watcher was shown it as it was written: a
	// transcript that leaves it out reads as the agent having said nothing at all.
	s.useChat()
	writer := newWriter(s.log)
	writer.handle(message{author: s.log.agent, text: "Tomorrow will be", turnID: "turn-1", kind: piece, source: SourceAgent})
	writer.show()
	writer.handle(message{author: s.log.agent, text: "Tomorrow will be sunny.", turnID: "turn-1", kind: prepared, source: SourceAgent})
	writer.handle(message{author: s.log.agent, turnID: "turn-1", kind: interrupt, source: SourceAgent})
	reader := &Reader{client: s.log.client}

	said, err := reader.Transcript(context.Background(), Read{Channel: "agent-1"})

	s.Require().NoError(err)
	s.Require().Len(said, 1)
	s.Equal("Tomorrow will be sunny.", said[0].Text)
	s.True(said[0].Interrupted)
	s.Equal("turn-1", said[0].TurnID)
}

func (s *ChatLogSuite) TestAScriptedLineIsStoredOnceItIsSaid() {
	// A greeting is said without asking the model, and was in no transcript at all.
	s.useChat()
	s.log.Record(agent.Said{TurnID: "say-1", Text: "Hi, thanks for getting in touch."})
	s.log.Record(agent.Spoke{TurnID: "say-1"})
	writer := newWriter(s.log)
	for _, queued := range s.queued() {
		writer.handle(queued)
	}

	stored := s.channel()
	s.Require().Len(stored, 1)
	s.Equal("Hi, thanks for getting in touch.", stored[0].Text)
	s.Equal(SourceAgent, stored[0].Custom[SourceField])
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
	s.useChat()
	writer := newWriter(s.log)

	s.play(writer, agent.Responded{TurnID: "turn-1", Text: ""})

	s.Empty(s.channel())
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

func (s *ChatLogSuite) TestASpokenReplyIsMarkedAsTheAssistants() {
	// Stream's AI components render a message as the assistant's only when it says so, as a
	// reply in a conversation held in writing does.
	s.useChat()
	writer := newWriter(s.log)
	writer.handle(message{author: s.log.agent, text: "Where does it", turnID: "turn-1", kind: piece, source: SourceAgent})
	writer.show()
	writer.handle(message{author: s.log.agent, text: "Where does it hurt?", turnID: "turn-1", kind: prepared, source: SourceAgent})
	writer.handle(message{author: s.log.agent, turnID: "turn-1", kind: spoken, source: SourceAgent})

	stored := s.channel()
	s.Require().Len(stored, 1)
	s.Equal("Where does it hurt?", stored[0].Text)
	s.Equal(true, stored[0].Custom[aiGeneratedField])
}

func (s *ChatLogSuite) TestWhatAParticipantSaidIsNotMarkedAsTheAssistants() {
	s.useChat()
	writer := newWriter(s.log)
	writer.handle(message{author: User{ID: "alice"}, text: "my lower back", kind: heard, source: SourceSpeech})

	stored := s.channel()
	s.Require().Len(stored, 1)
	s.NotContains(stored[0].Custom, aiGeneratedField)
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

func (s *ChatLogSuite) TestALineIsStoredUnderTheIdItWasHeardAs() {
	// The session's live frames name a line by its id, so the stored copy carries the same
	// one and a watcher never has to match the two by their words.
	s.useChat()
	writer := newWriter(s.log)
	alice := User{ID: "alice"}
	writer.handle(message{author: alice, text: "where is", lineID: "line-1", kind: hearing, source: SourceSpeech})
	writer.show()
	writer.handle(message{author: alice, text: "Where is my order?", lineID: "line-1", kind: heard, source: SourceSpeech})
	writer.handle(message{author: alice, text: "Thanks.", lineID: "line-2", kind: heard, source: SourceSpeech})

	stored := s.channel()
	s.Require().Len(stored, 2)
	s.Equal("line-1", stored[0].ID, "a line shown while it was heard keeps its id once it settles")
	s.Equal("line-2", stored[1].ID, "a line heard before it was shown is stored under its id too")
}

func (s *ChatLogSuite) TestALineHeardAgainGrownIsWrittenOverRatherThanStoredTwice() {
	s.useChat()
	writer := newWriter(s.log)
	alice := User{ID: "alice"}
	writer.handle(message{author: alice, text: "Woah.", lineID: "line-1", kind: heard, source: SourceSpeech})
	writer.handle(message{author: alice, text: "Woah. This is", lineID: "line-1", kind: hearing, source: SourceSpeech})
	writer.show()
	writer.handle(message{author: alice, text: "Woah. This is weird.", lineID: "line-1", kind: heard, source: SourceSpeech})

	stored := s.channel()
	s.Require().Len(stored, 1, "the line carried on is the same line")
	s.Equal("Woah. This is weird.", stored[0].Text)
	s.Equal(false, stored[0].Custom[generatingField])
}

func (s *ChatLogSuite) TestTwoLinesUnderOneUserAreKeptApart() {
	// The same person on two devices is two participants with one user id: each line is
	// its own message, and settling one leaves the other being said.
	s.useChat()
	writer := newWriter(s.log)
	alice := User{ID: "alice"}
	writer.handle(message{author: alice, text: "is it", lineID: "line-1", kind: hearing, source: SourceSpeech})
	writer.handle(message{author: alice, text: "hello", lineID: "line-2", kind: hearing, source: SourceSpeech})
	writer.show()
	writer.handle(message{author: alice, text: "Hello?", lineID: "line-2", kind: heard, source: SourceSpeech})
	writer.handle(message{author: alice, text: "is it raining", lineID: "line-1", kind: hearing, source: SourceSpeech})
	writer.show()
	writer.handle(message{author: alice, text: "Is it raining?", lineID: "line-1", kind: heard, source: SourceSpeech})

	// Lines shown in one tick are stored in no particular order.
	texts := map[string]string{}
	for _, stored := range s.channel() {
		texts[stored.ID] = stored.Text
	}
	s.Equal(map[string]string{"line-1": "Is it raining?", "line-2": "Hello?"}, texts)
}

func (s *ChatLogSuite) TestALineHeardAgainAndLetGoOfKeepsWhatWasHeard() {
	// "Woah." was heard; what the caller went on to say in the same line was not taken as
	// meant for the agent. That much was said, and the transcript keeps it.
	s.useChat()
	writer := newWriter(s.log)
	alice := User{ID: "alice"}
	writer.handle(message{author: alice, text: "Woah.", lineID: "line-1", kind: heard, source: SourceSpeech})
	writer.handle(message{author: alice, text: "Woah. Hang on, the door", lineID: "line-1", kind: hearing, source: SourceSpeech})
	writer.show()
	writer.handle(message{author: alice, lineID: "line-1", kind: ignored, source: SourceSpeech})

	stored := s.channel()
	s.Require().Len(stored, 1, "the line heard was removed with the words let go of")
	s.Equal("Woah.", stored[0].Text)
	s.Equal(false, stored[0].Custom[generatingField])
}

func (s *ChatLogSuite) TestALineStoredAgainUnderItsIdIsWrittenOver() {
	// Chat refuses an id it has: a line remembered no longer, said again, is still the
	// line, not lost.
	s.useChat()
	writer := newWriter(s.log)
	alice := User{ID: "alice"}
	writer.handle(message{author: alice, text: "Woah.", lineID: "line-1", kind: heard, source: SourceSpeech})
	clear(writer.heardLines)
	writer.handle(message{author: alice, text: "Woah. This is weird.", lineID: "line-1", kind: heard, source: SourceSpeech})

	stored := s.channel()
	s.Require().Len(stored, 1)
	s.Equal("Woah. This is weird.", stored[0].Text)
	s.Equal("line-1", writer.heardLines["line-1"].messageID)
}

func (s *ChatLogSuite) TestTheTranscriptSaysWhatEachLineIsAndWhetherItIsFinished() {
	s.useChat()
	writer := newWriter(s.log)
	alice := User{ID: "alice"}
	writer.handle(message{author: s.log.agent, text: "Your order ships", turnID: "turn-9", kind: piece, source: SourceAgent})
	writer.handle(message{author: alice, text: "and when", lineID: "line-3", kind: hearing, source: SourceSpeech})
	writer.show()
	reader := &Reader{client: s.log.client}

	said, err := reader.Transcript(context.Background(), Read{Channel: "agent-1"})

	s.Require().NoError(err)
	s.Require().Len(said, 2)
	s.Equal("turn-9", said[0].TurnID, "a reply says which turn it was written for")
	s.True(said[0].Generating, "a reply still being written says so")
	s.Equal("line-3", said[1].ID)
	s.True(said[1].Generating, "words still being said are stored as they first were, and say so")
}

// written runs events through a writer the way the log's own goroutine would, showing what
// is being written after each, and returns what the channel then holds.
func (s *ChatLogSuite) written(events ...agent.Event) []getstream.MessageResponse {
	s.useChat()
	writer := newWriter(s.log)
	for _, event := range events {
		s.log.Record(event)
		for _, queued := range s.queued() {
			writer.handle(queued)
		}
		writer.show()
	}
	return s.channel()
}

func (s *ChatLogSuite) TestAStreamedReplyNamesTheTurnItIsSpokenAs() {
	// The session socket's speaking and spoke name the same turn, which is how a page pairs
	// the reply it was shown with how much of it has been heard.
	s.useChat()
	writer := newWriter(s.log)
	writer.handle(message{author: s.log.agent, text: "Your order", turnID: "turn-1", kind: piece, source: SourceAgent})
	writer.show()

	live := s.channel()
	s.Require().Len(live, 1)
	s.Equal(true, live[0].Custom[generatingField])
	s.Equal("turn-1", live[0].Custom[turnField], "a watcher has the turn from the reply's first message")

	writer.handle(message{author: s.log.agent, text: "Your order ships today.", turnID: "turn-1", kind: prepared, source: SourceAgent})
	writer.handle(message{author: s.log.agent, turnID: "turn-1", kind: spoken, source: SourceAgent})

	settled := s.channel()
	s.Require().Len(settled, 1)
	s.Equal(false, settled[0].Custom[generatingField])
	s.Equal("turn-1", settled[0].Custom[turnField], "settling it keeps the turn")
}

func (s *ChatLogSuite) TestAReplyThatNeverStreamedNamesItsTurn() {
	stored := s.written(
		agent.Responded{TurnID: "turn-1", Text: "Yes."},
		agent.Spoke{TurnID: "turn-1", SynthesisID: "turn-1"},
	)

	s.Require().Len(stored, 1)
	s.Equal("turn-1", stored[0].Custom[turnField])
}

func (s *ChatLogSuite) TestALineSaidAsWrittenNamesTheTurnItIsSpokenAs() {
	// A greeting's turn is the say- id its speaking and spoke carry.
	stored := s.written(
		agent.Said{TurnID: "say-1761134400000000000", Text: "Hi there.", Purpose: agent.PurposeGreeting},
		agent.Spoke{TurnID: "say-1761134400000000000", SynthesisID: "say-1761134400000000000"},
		agent.Responded{TurnID: "back-1", Text: "Are you still there?", Purpose: agent.PurposeCheckIn},
		agent.Spoke{TurnID: "back-1", SynthesisID: "back-1"},
	)

	s.Require().Len(stored, 2)
	s.Equal("say-1761134400000000000", stored[0].Custom[turnField])
	s.Equal("back-1", stored[1].Custom[turnField])
}

func (s *ChatLogSuite) TestAReplyCutOffKeepsItsTurn() {
	stored := s.written(
		agent.ResponseDelta{TurnID: "turn-1", Text: "Tomorrow will be"},
		agent.Interrupted{TurnID: "turn-1"},
		agent.Interrupted{TurnID: "turn-2"},
		agent.Responded{TurnID: "turn-2", Text: "One,"},
	)

	s.Require().Len(stored, 2)
	s.Equal(true, stored[0].Custom[interruptedField])
	s.Equal("turn-1", stored[0].Custom[turnField], "a reply emptied when cut off")
	s.Equal(true, stored[1].Custom[interruptedField])
	s.Equal("turn-2", stored[1].Custom[turnField], "what a native model said before the cut")
}

func (s *ChatLogSuite) TestEveryMessageOfAReplySpokenSentenceBySentenceNamesItsTurn() {
	// A voice that takes a sentence at a time reports each one spoken, and one finished
	// before the model is leaves the rest of the reply to a message of its own.
	stored := s.written(
		agent.ResponseDelta{TurnID: "turn-1", Text: "One. "},
		agent.Spoke{TurnID: "turn-1", SynthesisID: "turn-1#0"},
		agent.ResponseDelta{TurnID: "turn-1", Text: "Two."},
		agent.Responded{TurnID: "turn-1", Text: "One. Two."},
		agent.Spoke{TurnID: "turn-1", SynthesisID: "turn-1#1"},
	)

	s.Require().Len(stored, 2)
	for _, message := range stored {
		s.Equal("turn-1", message.Custom[turnField])
	}
}

func (s *ChatLogSuite) TestAWrittenReplyNamesNoTurn() {
	// Nothing was spoken, so no speaking or spoke will ever name it.
	s.useChat()
	writer := newWriter(s.log)
	s.log.Reply("reissuing to another company is not possible")
	for _, queued := range s.queued() {
		writer.handle(queued)
	}

	stored := s.channel()
	s.Require().Len(stored, 1)
	s.NotContains(stored[0].Custom, turnField)
}

func (s *ChatLogSuite) TestALineSaidUnaskedIsReadBackWithWhatItWasFor() {
	s.useChat()
	s.log.Record(agent.Said{TurnID: "say-1", Text: "Hi there.", Purpose: agent.PurposeGreeting})
	s.log.Record(agent.Spoke{TurnID: "say-1"})
	s.log.Record(agent.Responded{TurnID: "back-1", Text: "Are you still there?", Purpose: agent.PurposeCheckIn})
	s.log.Record(agent.Spoke{TurnID: "back-1"})
	writer := newWriter(s.log)
	for _, queued := range s.queued() {
		writer.handle(queued)
	}
	reader := &Reader{client: s.log.client}

	said, err := reader.Transcript(context.Background(), Read{Channel: "agent-1"})

	s.Require().NoError(err)
	s.Require().Len(said, 2)
	s.Equal(agent.PurposeGreeting, said[0].Purpose)
	s.Equal(agent.PurposeCheckIn, said[1].Purpose)
}

func (s *ChatLogSuite) TestALineSaidUnaskedKeepsWhatItWasForWhenCutOff() {
	s.useChat()
	writer := newWriter(s.log)
	writer.handle(message{author: s.log.agent, text: "Are you", turnID: "back-1", kind: piece, source: SourceAgent})
	writer.show()
	writer.handle(message{author: s.log.agent, turnID: "back-1", purpose: agent.PurposeCheckIn, kind: prepared, source: SourceAgent})
	writer.handle(message{turnID: "back-1", kind: interrupt})
	writer.store(User{ID: "alice"}, "Hello?", SourceSpeech, false, identity{purpose: "whatever"})
	reader := &Reader{client: s.log.client}

	said, err := reader.Transcript(context.Background(), Read{Channel: "agent-1"})

	s.Require().NoError(err)
	s.Require().Len(said, 2)
	s.True(said[0].Interrupted)
	s.Equal(agent.PurposeCheckIn, said[0].Purpose)
	s.Empty(said[1].Purpose, "a reason no client knows is not passed on")
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
		LLMTTFTMs: 731, TextToTTSMs: 70, TTSToAudioMs: 143, TTSTTFBMs: 120,
		RoundtripMs: 1074, SpeechEndToAudioMs: 1080, FirstFrameQueuedMs: 1080, FirstAudibleFrameMs: 1106,
		SpeechEndToAudibleMs: 1112, AudioOutMs: 2100,
	})

	s.Require().True(ok)
	s.Equal("⏱ reply 1112 ms = eou 452 + llm 412 + tts 248 · ttft 731 · ttfb 120", described.line)
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

// play hands the log events as a session does, and the writer what they queued.
func (s *ChatLogSuite) play(writer *writer, events ...agent.Event) {
	for _, event := range events {
		s.log.Record(event)
	}
	for _, queued := range s.queued() {
		writer.handle(queued)
	}
}

// trail is what the writer asked Chat for, in order: "send" for a message stored, "update"
// for one changed, and each AI indicator as its state, or "clear", then the turn whose reply
// it names. Every update names one, which Stream's typed clients require.
func (s *ChatLogSuite) trail(chat *chattest.Server, turns ...string) []string {
	named := map[string]string{}
	for _, turn := range turns {
		named[replyID(s.log.channel, turn)] = turn
	}
	events := chat.Events(s.log.channel)
	var trail []string
	for _, asked := range chat.Requests("test") {
		switch {
		case strings.HasSuffix(asked.Path, "/event"):
			event := events[0]
			events = events[1:]
			s.Equal(s.log.agent.ID, event["user_id"], "the indicator is the agent's")
			custom, _ := event["custom"].(map[string]any)
			id, _ := custom["message_id"].(string)
			step := "clear"
			if event["type"] == "ai_indicator.update" {
				step, _ = custom["ai_state"].(string)
				s.Require().NotEmpty(id, "an update names its message")
			}
			if id != "" {
				step += " " + cmp.Or(named[id], id)
			}
			trail = append(trail, step)
		case strings.HasSuffix(asked.Path, "/message"):
			trail = append(trail, "send")
		case asked.Method == http.MethodPut && strings.Contains(asked.Path, "/messages/"):
			trail = append(trail, "update")
		}
	}
	return trail
}

// clock stops the writer's time at a moment the test moves on by hand.
func (s *ChatLogSuite) clock(writer *writer) *time.Time {
	now := time.Now()
	writer.now = func() time.Time { return now }
	return &now
}

func (s *ChatLogSuite) TestAStreamedReplyIsShownThinkingThenGeneratingUntilItIsHeard() {
	// Stream's AI components show what a reply is doing from the indicator, as they do for a
	// conversation in writing. It clears only once the reply is stored as heard.
	chat := s.useChat()
	writer := newWriter(s.log)

	s.play(writer, agent.Responding{TurnID: "turn-1"}, agent.ResponseDelta{TurnID: "turn-1", Text: "Tomorrow "})
	writer.show()
	s.play(writer,
		agent.ResponseDelta{TurnID: "turn-1", Text: "is sunny."},
		agent.Responded{TurnID: "turn-1", Text: "Tomorrow is sunny."},
		agent.Spoke{TurnID: "turn-1"})

	s.Equal([]string{"AI_STATE_THINKING turn-1", "send", "AI_STATE_GENERATING turn-1", "update", "clear turn-1"}, s.trail(chat, "turn-1"))
	stored := s.channel()
	s.Require().Len(stored, 1)
	s.Equal(replyID(s.log.channel, "turn-1"), stored[0].ID, "the reply is the message its indicator named while thinking")
	s.Equal("Tomorrow is sunny.", stored[0].Text)
}

func (s *ChatLogSuite) TestAnInterruptedReplyClearsItsIndicator() {
	chat := s.useChat()
	writer := newWriter(s.log)

	s.play(writer, agent.Responding{TurnID: "turn-1"}, agent.ResponseDelta{TurnID: "turn-1", Text: "Tomorrow "})
	writer.show()
	s.play(writer, agent.Interrupted{TurnID: "turn-1"})

	s.Equal([]string{"AI_STATE_THINKING turn-1", "send", "AI_STATE_GENERATING turn-1", "update", "clear turn-1"}, s.trail(chat, "turn-1"))
}

func (s *ChatLogSuite) TestAReplyInterruptedBeforeAWordClearsItsIndicator() {
	chat := s.useChat()
	writer := newWriter(s.log)

	s.play(writer, agent.Responding{TurnID: "turn-1"}, agent.Interrupted{TurnID: "turn-1"})

	s.Equal([]string{"AI_STATE_THINKING turn-1", "clear turn-1"}, s.trail(chat, "turn-1"))
}

func (s *ChatLogSuite) TestAReplyThatNeverStreamedClearsItsIndicatorOnceItIsStored() {
	chat := s.useChat()
	writer := newWriter(s.log)

	s.play(writer,
		agent.Responding{TurnID: "turn-1"},
		agent.Responded{TurnID: "turn-1", Text: "Yes."},
		agent.Spoke{TurnID: "turn-1"})

	s.Equal([]string{"AI_STATE_THINKING turn-1", "send", "clear turn-1"}, s.trail(chat, "turn-1"))
	s.Equal(replyID(s.log.channel, "turn-1"), s.channel()[0].ID)
}

func (s *ChatLogSuite) TestAReplyWithNothingToSayClearsItsIndicator() {
	// Nothing is spoken, so no Spoke would ever clear it.
	chat := s.useChat()
	writer := newWriter(s.log)

	s.play(writer, agent.Responding{TurnID: "turn-1"}, agent.Responded{TurnID: "turn-1"})

	s.Equal([]string{"AI_STATE_THINKING turn-1", "clear turn-1"}, s.trail(chat, "turn-1"))
}

func (s *ChatLogSuite) TestSpeechTheAgentIgnoresShowsNoIndicator() {
	chat := s.useChat()
	writer := newWriter(s.log)
	alice := stt.Participant{ID: "session-9", UserID: "alice"}

	s.play(writer, agent.Hearing{Participant: alice, Text: "not you, sorry", LineID: "line-1"})
	writer.show()
	s.play(writer, agent.Decided{Kind: string(agent.ActIgnore), Participant: alice, LineID: "line-1"})

	s.Empty(chat.Events(s.log.channel), "the agent never started on a reply")
}

func (s *ChatLogSuite) TestASearchIsShownAsCheckingExternalSources() {
	chat := s.useChat()
	writer := newWriter(s.log)

	s.play(writer,
		agent.Responding{TurnID: "turn-1"},
		agent.Responded{TurnID: "turn-1", PendingWork: true},
		agent.ToolStarted{ID: "call-1", TurnID: "turn-1", Tool: "web_search"},
		agent.ToolRan{ID: "call-1", TurnID: "turn-1", Tool: "web_search", Result: "{}"},
		agent.Responding{TurnID: "turn-2", Continues: "turn-1"},
		agent.ResponseDelta{TurnID: "turn-2", Text: "It is sunny."})
	writer.show()
	s.play(writer, agent.Responded{TurnID: "turn-2", Text: "It is sunny."}, agent.Spoke{TurnID: "turn-2"})

	s.Equal([]string{
		"AI_STATE_THINKING turn-1", "AI_STATE_EXTERNAL_SOURCES turn-1", "clear turn-1",
		"AI_STATE_THINKING turn-2", "send", "AI_STATE_GENERATING turn-2", "update", "clear turn-2",
	}, s.trail(chat, "turn-1", "turn-2"))
}

func (s *ChatLogSuite) TestTheAgentIsShownThinkingWhileAToolRunsAfterItsWordsAreHeard() {
	// "One moment" is heard in full while the booking is still being made.
	chat := s.useChat()
	writer := newWriter(s.log)

	s.play(writer, agent.Responding{TurnID: "turn-1"}, agent.ResponseDelta{TurnID: "turn-1", Text: "One moment."})
	writer.show()
	s.play(writer,
		agent.Responded{TurnID: "turn-1", Text: "One moment.", PendingWork: true},
		agent.ToolStarted{ID: "call-1", TurnID: "turn-1", Tool: "book_table"},
		agent.Spoke{TurnID: "turn-1"})
	s.Equal([]string{"AI_STATE_THINKING turn-1", "send", "AI_STATE_GENERATING turn-1", "update", "AI_STATE_THINKING turn-1"},
		s.trail(chat, "turn-1"))

	s.play(writer, agent.ToolRan{ID: "call-1", TurnID: "turn-1", Tool: "book_table", Result: "{}"})

	s.Equal("clear turn-1", s.trail(chat, "turn-1")[5])
}

func (s *ChatLogSuite) TestAReplyOfToolsAloneClearsOnceTheLastComesBack() {
	// Pressing digits at a menu has nothing to say and no reply after it.
	chat := s.useChat()
	writer := newWriter(s.log)

	s.play(writer,
		agent.Responding{TurnID: "turn-1"},
		agent.Responded{TurnID: "turn-1", PendingWork: true},
		agent.ToolStarted{ID: "call-1", TurnID: "turn-1", Tool: "press_digits"},
		agent.ToolStarted{ID: "call-2", TurnID: "turn-1", Tool: "press_digits"},
		agent.ToolRan{ID: "call-1", TurnID: "turn-1", Tool: "press_digits"})
	s.Equal([]string{"AI_STATE_THINKING turn-1"}, s.trail(chat, "turn-1"))

	s.play(writer, agent.ToolRan{ID: "call-2", TurnID: "turn-1", Tool: "press_digits"})

	s.Equal([]string{"AI_STATE_THINKING turn-1", "clear turn-1"}, s.trail(chat, "turn-1"))
}

func (s *ChatLogSuite) TestAnIndicatorNothingEndsIsClearedOnceStale() {
	// Work the model was told is running can come to nothing without a word.
	chat := s.useChat()
	writer := newWriter(s.log)
	now := s.clock(writer)
	s.play(writer, agent.Responding{TurnID: "turn-1"}, agent.Responded{TurnID: "turn-1", PendingWork: true})

	*now = now.Add(staleIndicator - time.Second)
	writer.show()
	s.Equal([]string{"AI_STATE_THINKING turn-1"}, s.trail(chat, "turn-1"))

	*now = now.Add(time.Second)
	writer.show()
	s.Equal([]string{"AI_STATE_THINKING turn-1", "clear turn-1"}, s.trail(chat, "turn-1"))
}

func (s *ChatLogSuite) TestARunningToolKeepsTheIndicatorFromGoingStale() {
	chat := s.useChat()
	writer := newWriter(s.log)
	now := s.clock(writer)
	s.play(writer,
		agent.Responding{TurnID: "turn-1"},
		agent.Responded{TurnID: "turn-1", PendingWork: true},
		agent.ToolStarted{ID: "call-1", TurnID: "turn-1", Tool: "run_code"})

	*now = now.Add(2 * staleIndicator)
	writer.show()

	s.Equal([]string{"AI_STATE_THINKING turn-1"}, s.trail(chat, "turn-1"), "the agent is still working")
}

func (s *ChatLogSuite) TestAnEarlierReplyEndingDoesNotClearTheTurnAfterIt() {
	chat := s.useChat()
	writer := newWriter(s.log)

	s.play(writer, agent.Responding{TurnID: "turn-1"}, agent.ResponseDelta{TurnID: "turn-1", Text: "Tomorrow "})
	writer.show()
	s.play(writer, agent.Responding{TurnID: "turn-2"}, agent.Spoke{TurnID: "turn-1"})

	s.Equal([]string{
		"AI_STATE_THINKING turn-1", "send", "AI_STATE_GENERATING turn-1",
		"clear turn-1", "AI_STATE_THINKING turn-2", "update",
	}, s.trail(chat, "turn-1", "turn-2"))
}

func (s *ChatLogSuite) TestASearchWhoseEndWasLostIsNotShownForTheNextTurn() {
	// The queue drops rather than blocks, so a tool's end can be lost.
	chat := s.useChat()
	writer := newWriter(s.log)

	s.play(writer,
		agent.Responding{TurnID: "turn-1"},
		agent.Responded{TurnID: "turn-1", PendingWork: true},
		agent.ToolStarted{ID: "call-1", TurnID: "turn-1", Tool: "web_search"},
		agent.Responding{TurnID: "turn-2"})

	s.Equal([]string{
		"AI_STATE_THINKING turn-1", "AI_STATE_EXTERNAL_SOURCES turn-1", "clear turn-1", "AI_STATE_THINKING turn-2",
	}, s.trail(chat, "turn-1", "turn-2"))
}

func (s *ChatLogSuite) TestClosingClearsAnIndicatorLeftUp() {
	// A call can end while the agent is still thinking, and nothing would clear it after.
	chat := s.useChat()
	s.Require().NoError(s.log.Start(context.Background()))

	s.log.Record(agent.Responding{TurnID: "turn-1"})
	s.log.Close()

	s.Equal([]string{"AI_STATE_THINKING turn-1", "clear turn-1"}, s.trail(chat, "turn-1"))
}

func (s *ChatLogSuite) TestClosingWhileAReplyIsWrittenClearsIt() {
	chat := s.useChat()
	writer := newWriter(s.log)
	s.play(writer, agent.Responding{TurnID: "turn-1"}, agent.ResponseDelta{TurnID: "turn-1", Text: "Tomorrow "})
	writer.show()

	writer.closeOut()

	s.Equal([]string{"AI_STATE_THINKING turn-1", "send", "AI_STATE_GENERATING turn-1", "update", "clear turn-1"}, s.trail(chat, "turn-1"))
}

func (s *ChatLogSuite) TestNoIndicatorIsSentWhileStreamHoldsOff() {
	// Stream answers a 429 by closing the connection, so a limited app is not asked again
	// until it says to.
	chat := s.useChat()
	writer := newWriter(s.log)
	now := s.clock(writer)
	writer.quietUntil = now.Add(time.Minute)

	s.play(writer, agent.Responding{TurnID: "turn-1"})
	s.Empty(chat.Events(s.log.channel))

	*now = now.Add(time.Minute)
	writer.show()
	s.Equal([]string{"AI_STATE_THINKING turn-1"}, s.trail(chat, "turn-1"), "what is true by then goes out")

	writer.quietUntil = now.Add(time.Minute)
	s.play(writer, agent.Interrupted{TurnID: "turn-1"})
	s.Len(chat.Events(s.log.channel), 1)
	writer.closeOut()
	s.Equal([]string{"AI_STATE_THINKING turn-1", "clear turn-1"}, s.trail(chat, "turn-1"), "nothing would clear it after the call")
}

func (s *ChatLogSuite) TestAReplyStoredByAStoreThatTimedOutIsWrittenOver() {
	// The first store timed out here but landed in Chat, so the reply's id is already taken.
	chat := s.useChat()
	writer := newWriter(s.log)
	s.play(writer, agent.Responding{TurnID: "turn-1"})
	id, text := replyID(s.log.channel, "turn-1"), "Tomorrow"
	_, err := chat.Client.Chat().SendMessage(context.Background(), ChannelType, s.log.channel,
		&getstream.SendMessageRequest{Message: getstream.MessageRequest{ID: &id, Text: &text, UserID: &s.log.agent.ID}})
	s.Require().NoError(err)

	s.play(writer, agent.ResponseDelta{TurnID: "turn-1", Text: "Tomorrow is"})
	writer.show()
	s.play(writer, agent.Responded{TurnID: "turn-1", Text: "Tomorrow is sunny."}, agent.Spoke{TurnID: "turn-1"})

	s.Equal([]string{"Tomorrow is sunny."}, chat.Messages(s.log.channel))
	s.Equal(false, s.channel()[0].Custom[generatingField])
}

func (s *ChatLogSuite) TestAReplySpokenInPiecesKeepsEveryPiece() {
	// A voice that speaks a sentence at a time reports each as spoken, and words after the
	// first are a message of their own rather than written over it.
	chat := s.useChat()
	writer := newWriter(s.log)

	s.play(writer, agent.Responding{TurnID: "turn-1"}, agent.ResponseDelta{TurnID: "turn-1", Text: "One."})
	writer.show()
	s.play(writer, agent.Spoke{TurnID: "turn-1"}, agent.ResponseDelta{TurnID: "turn-1", Text: " Two."})
	writer.show()
	s.play(writer, agent.Spoke{TurnID: "turn-1"})

	s.Equal([]string{"One.", " Two."}, chat.Messages(s.log.channel))
}
