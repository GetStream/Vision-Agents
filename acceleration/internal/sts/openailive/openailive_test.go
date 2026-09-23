package openailive

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

// replyGap is short so a test can wait out the quiet that settles a reply.
const replyGap = 30 * time.Millisecond

type OpenAILiveSuite struct {
	suite.Suite
}

func TestOpenAILiveSuite(t *testing.T) {
	suite.Run(t, new(OpenAILiveSuite))
}

// newProvider returns a provider that is wired up but never connected, so the event
// mapping can be exercised without touching the network.
func (s *OpenAILiveSuite) newProvider() *STS {
	provider, err := New(Options{APIKey: "test-key", ReplyGap: replyGap, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = provider.Close() })
	return provider
}

// frame decodes a server frame the way it arrives on the wire.
func (s *OpenAILiveSuite) frame(raw string) serverEvent {
	var event serverEvent
	s.Require().NoError(json.Unmarshal([]byte(raw), &event))
	return event
}

func heard(text string) string {
	return `{"type":"session.input_transcript.delta","delta":"` + text + `","start_ms":600,"end_ms":800}`
}

func said(text string) string {
	return `{"type":"session.output_transcript.delta","delta":"` + text + `","start_ms":900,"end_ms":1100}`
}

func spoke(samples int) string {
	return `{"type":"session.output_audio.delta","delta":"` + base64.StdEncoding.EncodeToString(make([]byte, samples*2)) + `"}`
}

// drain collects the events emitted so far without blocking on an empty channel.
func (s *OpenAILiveSuite) drain(provider *STS) []sts.Event {
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

// settled waits out the quiet that ends a reply, then collects what was emitted.
func (s *OpenAILiveSuite) settled(provider *STS) []sts.Event {
	time.Sleep(5 * replyGap)
	return s.drain(provider)
}

func completes(events []sts.Event) []sts.ResponseComplete {
	var found []sts.ResponseComplete
	for _, event := range events {
		if typed, ok := event.(sts.ResponseComplete); ok {
			found = append(found, typed)
		}
	}
	return found
}

func (s *OpenAILiveSuite) TestNewRequiresAnAPIKey() {
	s.T().Setenv("OPENAI_API_KEY", "")
	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *OpenAILiveSuite) TestNewRefusesARealtimeModel() {
	_, err := New(Options{APIKey: "test-key", Model: "gpt-realtime-2"})
	s.ErrorContains(err, "not a GPT-Live model")
	s.True(Serves("gpt-live-1"))
	s.False(Serves("gpt-realtime-2"))
}

func (s *OpenAILiveSuite) TestWhatTheModelCannotDoItRefuses() {
	provider := s.newProvider()
	s.ErrorIs(provider.SendFrame(llm.ImagePart{MIME: "image/jpeg", Data: []byte{1}}), sts.ErrNoImages)
	s.ErrorIs(provider.SetInstructions("x"), sts.ErrInstructionsFixed)

	capabilities := provider.Capabilities()
	s.False(capabilities.Accepts("image"))
	s.False(capabilities.SemanticTurns, "there is no detector to ask for")
	s.False(capabilities.Endpointing, "there is no silence timer to tune")
	s.False(capabilities.InstructionsMidSession)
	s.True(capabilities.ToolsMidSession)
	s.True(capabilities.Tools)
	s.True(capabilities.Text)
	s.Equal(SampleRate, provider.SampleRate())
}

func (s *OpenAILiveSuite) TestTheCallerIsTakenToHaveStoppedAtTheLastWordHeard() {
	provider := s.newProvider()
	provider.participant = sts.Participant{ID: "alice"}

	provider.handleMessage(s.frame(heard("hello ")))
	provider.handleMessage(s.frame(heard("world")))
	time.Sleep(20 * time.Millisecond)
	provider.handleMessage(s.frame(spoke(2400)))

	events := s.settled(provider)
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
	s.Equal(SampleRate, chunk.Audio.SampleRate)
	s.Len(chunk.Audio.Samples, 2400)

	complete, ok := events[7].(sts.ResponseComplete)
	s.Require().True(ok, "the quiet after the audio is what ends the reply")
	s.False(complete.Interrupted)
	s.GreaterOrEqual(complete.TimeToFirstByteMs, 20.0, "the wait is timed from the last word heard")
	s.InDelta(100, complete.AudioDurationMs, 0.001)
}

func (s *OpenAILiveSuite) TestAudioWithoutAPauseIsOneReply() {
	provider := s.newProvider()

	provider.handleMessage(s.frame(spoke(240)))
	provider.handleMessage(s.frame(said("Hello ")))
	provider.handleMessage(s.frame(spoke(240)))
	provider.handleMessage(s.frame(said("there.")))

	events := s.settled(provider)
	var starts, chunks int
	for _, event := range events {
		switch typed := event.(type) {
		case sts.ResponseStarted:
			starts++
		case sts.AudioChunk:
			chunks++
			s.Equal(1, typed.Generation)
		case sts.OutputTranscript:
			s.Equal(stt.ModeDelta, typed.Mode)
			s.Equal("r-1", typed.ResponseID)
		}
	}
	s.Equal(1, starts)
	s.Equal(2, chunks)
	s.Len(completes(events), 1)
}

func (s *OpenAILiveSuite) TestAPauseLongerThanTheGapStartsTheNextReply() {
	provider := s.newProvider()

	provider.handleMessage(s.frame(spoke(240)))
	first := s.settled(provider)
	provider.handleMessage(s.frame(spoke(240)))
	second := s.settled(provider)

	s.Require().Len(completes(first), 1)
	s.Equal(1, completes(first)[0].Generation)
	s.Require().Len(completes(second), 1)
	s.Equal(2, completes(second)[0].Generation)
}

func (s *OpenAILiveSuite) TestAReplyTheCallerWasHeardOverIsSettledAsInterrupted() {
	provider := s.newProvider()

	provider.handleMessage(s.frame(spoke(240)))
	provider.handleMessage(s.frame(heard("wait")))

	found := completes(s.settled(provider))
	s.Require().Len(found, 1)
	s.True(found[0].Interrupted, "the model stopping while the caller spoke is a barge-in")
}

func (s *OpenAILiveSuite) TestALocalInterruptMutesTheRestOfTheReply() {
	provider := s.newProvider()

	provider.handleMessage(s.frame(spoke(240)))
	s.Require().NoError(provider.Interrupt(0))
	// The model has not heard about it and keeps talking; none of this may get through.
	provider.handleMessage(s.frame(spoke(240)))
	provider.handleMessage(s.frame(spoke(240)))
	cut := s.settled(provider)
	// The quiet ended the mute, so the next reply is heard.
	provider.handleMessage(s.frame(spoke(240)))
	next := s.settled(provider)

	var chunks []sts.AudioChunk
	for _, event := range append(cut, next...) {
		if chunk, ok := event.(sts.AudioChunk); ok {
			chunks = append(chunks, chunk)
		}
	}
	s.Require().Len(completes(cut), 1)
	s.True(completes(cut)[0].Interrupted)
	s.Require().Len(chunks, 2, "one before the interrupt, one from the next reply")
	s.Equal(1, chunks[0].Generation)
	s.Equal(2, chunks[1].Generation)
}

func (s *OpenAILiveSuite) TestAFinishedBackendFunctionCallIsAToolCall() {
	provider := s.newProvider()

	provider.handleMessage(s.frame(`{"type":"response.event","delegation_id":"item_1","event":{"type":"response.function_call_arguments.done","arguments":"{\"city\":\"Paris\"}"}}`))
	provider.handleMessage(s.frame(`{"type":"response.event","delegation_id":"item_1","event":{"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}}`))
	provider.handleMessage(s.frame(`{"type":"response.event","delegation_id":"item_1","event":{"type":"response.output_item.done","item":{"type":"message"}}}`))
	provider.handleMessage(s.frame(`{"type":"response.event","delegation_id":"item_2","event":{"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_2","name":"get_time"}}}`))

	events := s.drain(provider)
	s.Require().Len(events, 2, "only a finished function call names the call; nothing else is one")
	call, ok := events[0].(sts.ToolCall)
	s.Require().True(ok)
	s.Equal("call_1", call.CallID)
	s.Equal("get_weather", call.Name)
	s.JSONEq(`{"city":"Paris"}`, call.Arguments)
	s.Equal("{}", events[1].(sts.ToolCall).Arguments, "a call with no arguments still carries an object")
}

func (s *OpenAILiveSuite) TestFailuresAreReportedWithoutEndingTheSession() {
	provider := s.newProvider()

	provider.handleMessage(s.frame(`{"type":"error","error":{"type":"invalid_request_error","code":"invalid_audio","message":"PCM16 audio must contain an even number of bytes","param":"audio"}}`))
	provider.handleMessage(s.frame(`{"type":"response.event","delegation_id":"item_1","event":{"type":"response.failed","response":{"error":{"code":"server_error"}}}}`))

	events := s.drain(provider)
	s.Require().Len(events, 2)
	command := events[0].(sts.Error)
	s.False(command.Fatal, "a rejected command does not close a running session")
	s.ErrorContains(command, "invalid_audio")
	backend := events[1].(sts.Error)
	s.False(backend.Fatal)
	s.Equal("delegation", backend.Context)
}

func (s *OpenAILiveSuite) TestTheServerClosingSettlesTheReplyAndSaysWhy() {
	provider := s.newProvider()

	provider.handleMessage(s.frame(spoke(240)))
	provider.handleMessage(s.frame(`{"type":"session.closed","reason":"expired","usage":{"seconds":128}}`))
	provider.handleMessage(s.frame(`{"type":"session.closed","reason":"connection_lost"}`))

	events := s.drain(provider)
	found := completes(events)
	s.Require().Len(found, 1)
	s.True(found[0].Interrupted)

	var closes []sts.Disconnected
	for _, event := range events {
		if typed, ok := event.(sts.Disconnected); ok {
			closes = append(closes, typed)
		}
	}
	s.Require().Len(closes, 2)
	s.Equal("expired", closes[0].Reason)
	s.True(closes[0].Clean)
	s.False(closes[1].Clean, "a lost connection is nobody's decision")
}

func (s *OpenAILiveSuite) TestCloseSettlesAReplyInFlight() {
	provider, err := New(Options{APIKey: "test-key", ReplyGap: time.Minute, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	provider.handleMessage(s.frame(spoke(240)))

	s.Require().NoError(provider.Close())

	var events []sts.Event
	for event := range provider.Events() {
		events = append(events, event)
	}
	complete, ok := events[len(events)-1].(sts.ResponseComplete)
	s.Require().True(ok)
	s.True(complete.Interrupted)
}
