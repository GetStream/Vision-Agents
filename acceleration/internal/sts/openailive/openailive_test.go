package openailive

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
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

// spoke is a piece of the model's speech: every sample at 1000, well above audio.AudibleSample.
func spoke(samples int) string {
	return `{"type":"session.output_audio.delta","delta":"` + base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{0xe8, 0x03}, samples)) + `"}`
}

// quiet is a piece of the digital silence the model streams between replies, with the most
// a sample can be and still not be heard.
func quiet(samples int) string {
	pcm := audio.PcmData{Samples: make([]int16, samples)}
	pcm.Samples[0] = audio.AudibleSample - 1
	return `{"type":"session.output_audio.delta","delta":"` + base64.StdEncoding.EncodeToString(pcm.Bytes()) + `"}`
}

// streamQuiet sends silence the way the model does between replies, a 100 ms piece every
// 10 ms, for long enough that any reply or mute it kept alive would outlast the test's gap.
func (s *OpenAILiveSuite) streamQuiet(provider *STS) {
	for range 15 {
		provider.handleMessage(s.frame(quiet(2400)))
		time.Sleep(replyGap / 3)
	}
}

// drain collects the events emitted so far without blocking on an empty channel.
func (s *OpenAILiveSuite) drain(provider *STS) []sts.Event {
	var events []sts.Event
	for {
		select {
		case event, ok := <-provider.Events():
			if !ok {
				return events
			}
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
	s.NotEmpty(begun.ResponseID, "this API has no response ids, so the reply gets a generated ID")
	chunk, ok := events[6].(sts.AudioChunk)
	s.Require().True(ok)
	s.Equal(SampleRate, chunk.Audio.SampleRate)
	s.Len(chunk.Audio.Samples, 2400)

	complete, ok := events[7].(sts.ResponseComplete)
	s.Require().True(ok, "the quiet after the audio is what ends the reply")
	s.Equal(begun.ResponseID, complete.ResponseID)
	s.False(complete.Interrupted)
	s.GreaterOrEqual(complete.TimeToFirstByteMs, 20.0, "the wait is timed from the last word heard")
	s.InDelta(100, complete.AudioDurationMs, 0.001)
}

// The words of a reply can come before its audio. They still find the caller's turn
// settled first, so each reply's wait starts at its own caller's last word.
func (s *OpenAILiveSuite) TestAReplyWhoseWordsComeFirstStillFollowsTheCallerStopping() {
	provider := s.newProvider()
	provider.participant = sts.Participant{ID: "alice"}

	for _, words := range []string{"my favorite color is blue", "and green"} {
		provider.handleMessage(s.frame(heard(words)))
		time.Sleep(20 * time.Millisecond)
		provider.handleMessage(s.frame(said("Nice.")))
		provider.handleMessage(s.frame(spoke(2400)))

		events := s.settled(provider)
		var kinds []string
		for _, event := range events {
			kinds = append(kinds, fmt.Sprintf("%T", event))
		}
		s.Require().Equal([]string{
			"sts.SpeechStarted", "sts.InputTranscript", "sts.InputTranscript", "sts.SpeechStopped",
			"sts.ResponseStarted", "sts.OutputTranscript", "sts.AudioChunk", "sts.ResponseComplete",
		}, kinds, "the caller stops before the reply to them starts")

		settled := events[2].(sts.InputTranscript)
		s.Equal(stt.ModeFinal, settled.Mode)
		s.Equal(words, settled.Text, "what the reply answers is known when it starts")
		stopped := events[3].(sts.SpeechStopped)
		begun := events[4].(sts.ResponseStarted)
		s.False(begun.At.Before(stopped.At))

		complete := events[7].(sts.ResponseComplete)
		s.GreaterOrEqual(complete.TimeToFirstByteMs, 20.0, "the wait is timed from this caller's last word")
		s.Less(complete.TimeToFirstByteMs, 100.0, "not from the turn before")
	}
}

func (s *OpenAILiveSuite) TestAudioWithoutAPauseIsOneReply() {
	provider := s.newProvider()

	provider.handleMessage(s.frame(spoke(240)))
	provider.handleMessage(s.frame(said("Hello ")))
	provider.handleMessage(s.frame(spoke(240)))
	provider.handleMessage(s.frame(said("there.")))

	events := s.settled(provider)
	var starts, chunks int
	var responseID string
	for _, event := range events {
		switch typed := event.(type) {
		case sts.ResponseStarted:
			starts++
			responseID = typed.ResponseID
		case sts.AudioChunk:
			chunks++
			s.Equal(1, typed.Generation)
		case sts.OutputTranscript:
			s.Equal(stt.ModeDelta, typed.Mode)
			s.Equal(responseID, typed.ResponseID)
		}
	}
	s.Equal(1, starts)
	s.Equal(2, chunks)
	s.NotEmpty(responseID)
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

// The model streams silence for the whole session. Silence opens no reply, and the silence
// after one does not keep it open, so each reply is settled before the next.
func (s *OpenAILiveSuite) TestSilenceBetweenRepliesOpensNoReplyAndEndsTheOneBefore() {
	provider := s.newProvider()

	s.streamQuiet(provider)
	s.Empty(s.drain(provider), "the silence when the session starts is no reply")

	provider.handleMessage(s.frame(spoke(240)))
	s.streamQuiet(provider)
	first := s.drain(provider)
	provider.handleMessage(s.frame(spoke(240)))
	s.streamQuiet(provider)
	second := s.drain(provider)

	var starts int
	for _, event := range append(first, second...) {
		if _, ok := event.(sts.ResponseStarted); ok {
			starts++
		}
	}
	s.Equal(2, starts)
	s.Require().Len(completes(first), 1, "the silence after a reply settles it")
	s.Equal(1, completes(first)[0].Generation)
	s.False(completes(first)[0].Interrupted)
	s.Require().Len(completes(second), 1)
	s.Equal(2, completes(second)[0].Generation)
}

// A pause between words is part of the reply: it goes out with it, and speech after it
// within the gap is the same reply.
func (s *OpenAILiveSuite) TestSilenceInsideAReplyGoesOutWithIt() {
	provider := s.newProvider()

	provider.handleMessage(s.frame(spoke(240)))
	provider.handleMessage(s.frame(quiet(240)))
	provider.handleMessage(s.frame(spoke(240)))

	events := s.settled(provider)
	var chunks int
	for _, event := range events {
		if chunk, ok := event.(sts.AudioChunk); ok {
			chunks++
			s.Equal(1, chunk.Generation)
		}
	}
	s.Equal(3, chunks)
	s.Require().Len(completes(events), 1)
	s.InDelta(30, completes(events)[0].AudioDurationMs, 0.001, "the pause is part of how long the reply played")
}

// Hearing the caller during a reply says nothing about whether the reply was cut off. The
// API transcribes the caller asynchronously, so the deltas for the utterance being answered
// normally land after the answer has started. Only Interrupt settles a reply as cut off.
func (s *OpenAILiveSuite) TestAReplyTheCallerWasHeardOverStillSettlesAsFinished() {
	provider := s.newProvider()

	provider.handleMessage(s.frame(spoke(240)))
	provider.handleMessage(s.frame(heard("wait")))

	found := completes(s.settled(provider))
	s.Require().Len(found, 1)
	s.False(found[0].Interrupted, "a late transcript of the caller is not a barge-in")
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

// A mute ends once the model falls silent, even though its silence keeps streaming.
func (s *OpenAILiveSuite) TestAMuteEndsWhileTheModelStreamsSilence() {
	provider := s.newProvider()

	provider.handleMessage(s.frame(spoke(240)))
	s.Require().NoError(provider.Interrupt(0))
	provider.handleMessage(s.frame(spoke(240)))
	s.streamQuiet(provider)
	s.drain(provider)
	provider.handleMessage(s.frame(spoke(240)))

	var chunks []sts.AudioChunk
	for _, event := range s.settled(provider) {
		if chunk, ok := event.(sts.AudioChunk); ok {
			chunks = append(chunks, chunk)
		}
	}
	s.Require().Len(chunks, 1, "the next reply is heard")
	s.Equal(2, chunks[0].Generation)
}

// A caller who barges in is still being heard when the rest of the cut-off reply's words
// arrive. They may not settle their turn or open a reply.
func (s *OpenAILiveSuite) TestWhatRemainsOfACutOffReplyDoesNotEndTheBargeIn() {
	provider := s.newProvider()
	provider.participant = sts.Participant{ID: "alice"}

	provider.handleMessage(s.frame(spoke(240)))
	s.Require().NoError(provider.Interrupt(0))
	s.drain(provider)
	provider.handleMessage(s.frame(heard("wait ")))
	provider.handleMessage(s.frame(said("and the rest")))

	var kinds []string
	for _, event := range s.drain(provider) {
		kinds = append(kinds, fmt.Sprintf("%T", event))
	}
	s.Equal([]string{"sts.SpeechStarted", "sts.InputTranscript"}, kinds, "the caller is still speaking")
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
	s.Require().Len(closes, 1)
	s.Equal("expired", closes[0].Reason)
	s.True(closes[0].Clean)
	select {
	case _, ok := <-provider.Events():
		s.False(ok, "a terminal session must close its event stream after its events are drained")
	default:
		s.Fail("the terminal session left its event stream open")
	}
}

func (s *OpenAILiveSuite) TestAConnectionLostSessionEndsUncleanlyAndClosesItsEventStream() {
	provider := s.newProvider()

	provider.handleMessage(s.frame(`{"type":"session.closed","reason":"connection_lost"}`))
	events := s.drain(provider)

	s.Require().Len(events, 1)
	disconnected, ok := events[0].(sts.Disconnected)
	s.Require().True(ok)
	s.Equal("connection_lost", disconnected.Reason)
	s.False(disconnected.Clean, "a lost connection is nobody's decision")
	select {
	case _, ok := <-provider.Events():
		s.False(ok, "a terminal session must close its event stream after its events are drained")
	default:
		s.Fail("the terminal session left its event stream open")
	}
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
