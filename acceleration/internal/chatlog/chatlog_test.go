package chatlog

import (
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
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
	s.Equal(end, waiting[0].kind)
	s.Equal("turn-1", waiting[0].turnID)
}

func (s *ChatLogSuite) TestSilenceIsNotStored() {
	s.log.Record(agent.Responded{TurnID: "turn-1", Text: ""})

	s.Empty(s.queued())
}
