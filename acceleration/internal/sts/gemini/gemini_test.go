package gemini

import (
	"encoding/base64"
	"encoding/json"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type GeminiSuite struct {
	suite.Suite
}

func TestGeminiSuite(t *testing.T) {
	suite.Run(t, new(GeminiSuite))
}

// newProvider returns a provider that is wired up but never connected, so the event
// mapping can be exercised without touching the network.
func (s *GeminiSuite) newProvider() *STS {
	provider, err := New(Options{APIKey: "test-key", Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	return provider
}

// drain collects the events emitted so far without blocking on an empty channel.
func (s *GeminiSuite) drain(provider *STS) []sts.Event {
	var events []sts.Event
	for {
		select {
		case event := <-provider.Events():
			events = append(events, event)
		default:
			return events
		}
	}
}

func heard(text string) serverMessage {
	return serverMessage{ServerContent: &serverContent{InputTranscription: &transcription{Text: text}}}
}

func spoke(samples int) serverMessage {
	return serverMessage{ServerContent: &serverContent{ModelTurn: &content{Role: "model", Parts: []part{{
		InlineData: &blob{
			Data:     base64.StdEncoding.EncodeToString(make([]byte, samples*2)),
			MimeType: "audio/pcm;rate=24000",
		},
	}}}}}
}

func turnComplete() serverMessage {
	return serverMessage{ServerContent: &serverContent{TurnComplete: true}}
}

func (s *GeminiSuite) TestNewRequiresAnAPIKey() {
	s.T().Setenv("GOOGLE_API_KEY", "")
	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *GeminiSuite) TestWhatIsTakenAtSetupCannotChangeAfterwards() {
	provider := s.newProvider()
	s.ErrorIs(provider.SetInstructions("x"), sts.ErrInstructionsFixed)
	s.ErrorIs(provider.SetTools([]llm.Tool{{Name: "x"}}), sts.ErrToolsFixed)
	s.False(provider.Capabilities().InstructionsMidSession)
	s.True(provider.Capabilities().Resumable)
	s.Equal(maxDuration, provider.Capabilities().MaxDuration)
	s.Equal(OutputSampleRate, provider.SampleRate())
}

func (s *GeminiSuite) TestTheCallerIsTakenToHaveStoppedAtTheLastWordHeard() {
	provider := s.newProvider()
	provider.participant = sts.Participant{ID: "alice"}

	provider.handleMessage(heard("hello "))
	provider.handleMessage(heard("world"))
	time.Sleep(20 * time.Millisecond)
	provider.handleMessage(spoke(2400))
	provider.handleMessage(turnComplete())

	events := s.drain(provider)
	s.Require().Len(events, 8)

	started, ok := events[0].(sts.SpeechStarted)
	s.Require().True(ok, "the first word heard is the only sign the caller began")
	s.Equal("alice", started.Participant.ID)
	first, ok := events[1].(sts.InputTranscript)
	s.Require().True(ok)
	s.Equal(stt.ModeDelta, first.Mode)
	s.Equal("hello ", first.Text)
	_, ok = events[2].(sts.InputTranscript)
	s.Require().True(ok, "the second piece is another delta, not another start")

	settled, ok := events[3].(sts.InputTranscript)
	s.Require().True(ok)
	s.Equal(stt.ModeFinal, settled.Mode)
	s.Equal("hello world", settled.Text, "the model beginning to speak is what settles what it heard")
	_, ok = events[4].(sts.SpeechStopped)
	s.Require().True(ok)

	begun, ok := events[5].(sts.ResponseStarted)
	s.Require().True(ok)
	s.Equal("r-1", begun.ResponseID, "this API has no response ids, so the reply is numbered")
	chunk, ok := events[6].(sts.AudioChunk)
	s.Require().True(ok)
	s.Equal(OutputSampleRate, chunk.Audio.SampleRate)
	s.Len(chunk.Audio.Samples, 2400)

	complete, ok := events[7].(sts.ResponseComplete)
	s.Require().True(ok)
	s.False(complete.Interrupted)
	s.GreaterOrEqual(complete.TimeToFirstByteMs, 20.0, "the wait is timed from the last word heard")
	s.InDelta(100, complete.AudioDurationMs, 0.001)
}

func (s *GeminiSuite) TestTheServerInterruptingSettlesTheReplyAsCutOff() {
	provider := s.newProvider()

	provider.handleMessage(spoke(240))
	provider.handleMessage(serverMessage{ServerContent: &serverContent{Interrupted: true}})
	provider.handleMessage(spoke(240))

	events := s.drain(provider)
	var completes []sts.ResponseComplete
	var starts int
	for _, event := range events {
		switch typed := event.(type) {
		case sts.ResponseComplete:
			completes = append(completes, typed)
		case sts.ResponseStarted:
			starts++
		}
	}
	s.Require().Len(completes, 1)
	s.True(completes[0].Interrupted)
	s.Equal(2, starts, "audio after the interruption is the next reply")
}

func (s *GeminiSuite) TestALocalInterruptMutesTheRestOfTheTurn() {
	provider := s.newProvider()

	provider.handleMessage(spoke(240))
	s.Require().NoError(provider.Interrupt(0))
	// The model has not heard about it and keeps talking; none of this may get through.
	provider.handleMessage(spoke(240))
	provider.handleMessage(spoke(240))
	provider.handleMessage(turnComplete())
	// The next turn is a new reply and is heard.
	provider.handleMessage(spoke(240))

	events := s.drain(provider)
	var chunks []sts.AudioChunk
	var completes []sts.ResponseComplete
	for _, event := range events {
		switch typed := event.(type) {
		case sts.AudioChunk:
			chunks = append(chunks, typed)
		case sts.ResponseComplete:
			completes = append(completes, typed)
		}
	}
	s.Require().Len(completes, 1)
	s.True(completes[0].Interrupted)
	s.Require().Len(chunks, 2, "one before the interrupt, one from the next reply")
	s.Equal(1, chunks[0].Generation)
	s.Equal(2, chunks[1].Generation)
}

func (s *GeminiSuite) TestWhatTheModelSaidIsForwardedAsDeltas() {
	provider := s.newProvider()

	provider.handleMessage(serverMessage{ServerContent: &serverContent{OutputTranscription: &transcription{Text: "Hello "}}})
	provider.handleMessage(serverMessage{ServerContent: &serverContent{OutputTranscription: &transcription{Text: "there."}}})

	events := s.drain(provider)
	s.Require().Len(events, 3, "a reply is opened by the first thing the model says about it")
	_, ok := events[0].(sts.ResponseStarted)
	s.True(ok)
	said, ok := events[1].(sts.OutputTranscript)
	s.Require().True(ok)
	s.Equal(stt.ModeDelta, said.Mode)
	s.Equal("Hello ", said.Text)
	s.Equal("r-1", said.ResponseID)
}

func (s *GeminiSuite) TestAToolCallRemembersTheNameItsAnswerNeeds() {
	provider := s.newProvider()

	provider.handleMessage(serverMessage{ToolCall: &toolCall{FunctionCalls: []functionCall{
		{ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{"city":"Paris"}`)},
		{ID: "call_2", Name: "get_time"},
	}}})

	events := s.drain(provider)
	var calls []sts.ToolCall
	for _, event := range events {
		if typed, ok := event.(sts.ToolCall); ok {
			calls = append(calls, typed)
		}
	}
	s.Require().Len(calls, 2)
	s.Equal("call_1", calls[0].CallID)
	s.Equal("get_weather", calls[0].Name)
	s.JSONEq(`{"city":"Paris"}`, calls[0].Arguments)
	s.Equal("{}", calls[1].Arguments, "a call with no arguments still carries an object")
	s.Equal("get_weather", provider.calls["call_1"], "the answer has to carry the name back")

	s.ErrorContains(provider.Answer("call_9", "x", nil), "no tool call")

	provider.handleMessage(serverMessage{ToolCallCancellation: &toolCallCancellation{IDs: []string{"call_2"}}})
	cancelled := s.drain(provider)
	s.Require().Len(cancelled, 1)
	s.Equal([]string{"call_2"}, cancelled[0].(sts.ToolCancel).CallIDs)
	s.NotContains(provider.calls, "call_2")
}

func (s *GeminiSuite) TestUsageIsAttachedToTheReplyItBelongsTo() {
	provider := s.newProvider()

	provider.handleMessage(spoke(240))
	provider.handleMessage(serverMessage{UsageMetadata: &usageMetadata{
		PromptTokenCount:      120,
		ResponseTokenCount:    60,
		PromptTokensDetails:   []modalityTokens{{Modality: "AUDIO", TokenCount: 100}, {Modality: "TEXT", TokenCount: 20}},
		ResponseTokensDetails: []modalityTokens{{Modality: "AUDIO", TokenCount: 60}},
	}})
	provider.handleMessage(turnComplete())

	events := s.drain(provider)
	complete, ok := events[len(events)-1].(sts.ResponseComplete)
	s.Require().True(ok)
	s.Equal(sts.Usage{InputTokens: 120, OutputTokens: 60, InputAudioTokens: 100, OutputAudioTokens: 60}, complete.Usage)
}

func (s *GeminiSuite) TestTheServerWarnsBeforeItHangsUpAndLeavesAHandle() {
	provider := s.newProvider()

	provider.handleMessage(serverMessage{SessionResumptionUpdate: &resumptionUpdate{NewHandle: "h-1", Resumable: true}})
	provider.handleMessage(serverMessage{SessionResumptionUpdate: &resumptionUpdate{NewHandle: "h-2", Resumable: false}})
	provider.handleMessage(serverMessage{GoAway: &goAway{TimeLeft: "30s"}})

	s.Equal("h-1", provider.handle, "a handle that cannot be resumed from is not worth keeping")
	events := s.drain(provider)
	s.Require().Len(events, 1)
	s.Equal(30*time.Second, events[0].(sts.SessionExpiring).TimeLeft)
}

func (s *GeminiSuite) TestCloseSettlesAReplyInFlight() {
	provider := s.newProvider()
	provider.handleMessage(spoke(240))

	s.Require().NoError(provider.Close())

	var events []sts.Event
	for event := range provider.Events() {
		events = append(events, event)
	}
	complete, ok := events[len(events)-1].(sts.ResponseComplete)
	s.Require().True(ok)
	s.True(complete.Interrupted)
}
