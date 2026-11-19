package phonic

import (
	"bufio"
	"bytes"
	_ "embed"
	"encoding/json"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// The frames Phonic sent on live calls, with the audio replaced by a short tone.
var (
	//go:embed testdata/spoken_turn.jsonl
	spokenTurn []byte
	//go:embed testdata/barge_in.jsonl
	bargeIn []byte
	//go:embed testdata/tool_call_first.jsonl
	toolCallFirst []byte
)

type PhonicSuite struct {
	suite.Suite
}

func TestPhonicSuite(t *testing.T) {
	suite.Run(t, new(PhonicSuite))
}

// newProvider returns a provider that is wired up but never connected, so the frame mapping
// can be exercised without touching the network.
func (s *PhonicSuite) newProvider() *STS {
	provider, err := New(Options{APIKey: "test-key", Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = provider.Close() })
	return provider
}

// feed hands the provider every frame of a recording, in order.
func (s *PhonicSuite) feed(provider *STS, recording []byte) {
	lines := bufio.NewScanner(bytes.NewReader(recording))
	for lines.Scan() {
		s.feedOne(provider, lines.Text())
	}
	s.Require().NoError(lines.Err())
}

func (s *PhonicSuite) feedOne(provider *STS, raw string) {
	var frame serverFrame
	s.Require().NoError(json.Unmarshal([]byte(raw), &frame))
	provider.handleMessage(frame)
}

// drain collects the events emitted so far without blocking on an empty channel.
func (s *PhonicSuite) drain(provider *STS) []sts.Event {
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

// settled waits out the grace in which a finished reply may still turn out cut off.
func (s *PhonicSuite) settled(provider *STS) []sts.Event {
	time.Sleep(interruptGrace + 100*time.Millisecond)
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

func chunks(events []sts.Event) []sts.AudioChunk {
	var found []sts.AudioChunk
	for _, event := range events {
		if typed, ok := event.(sts.AudioChunk); ok {
			found = append(found, typed)
		}
	}
	return found
}

func (s *PhonicSuite) TestNewRequiresAnAPIKey() {
	s.T().Setenv(apiKeyEnvVar, "")
	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *PhonicSuite) TestWhatTheModelCannotDoItRefuses() {
	provider := s.newProvider()
	s.ErrorIs(provider.SendFrame(llm.ImagePart{MIME: "image/jpeg", Data: []byte{1}}), sts.ErrNoImages)

	capabilities := provider.Capabilities()
	s.False(capabilities.Accepts("image"))
	s.False(capabilities.FullDuplex, "it stops speaking when the caller does")
	s.False(capabilities.Endpointing, "no silence timer is known to end its turns")
	s.False(capabilities.Usage, "it reports no tokens")
	s.True(capabilities.SessionBilled)
	s.True(capabilities.InstructionsMidSession)
	s.True(capabilities.ToolsMidSession)
	s.Equal(SampleRate, provider.SampleRate())
}

func (s *PhonicSuite) TestASpokenTurnIsWrittenDownAndAnsweredOnce() {
	provider := s.newProvider()
	s.feed(provider, spokenTurn)
	events := s.settled(provider)

	var final string
	var stopped, started bool
	for _, event := range events {
		switch typed := event.(type) {
		case sts.InputTranscript:
			if typed.Mode == stt.ModeFinal {
				s.True(stopped, "the turn is settled after the caller stopped")
				final = typed.Text
			}
		case sts.SpeechStopped:
			stopped = true
		case sts.ResponseStarted:
			started = true
			s.NotEmpty(final, "the caller's turn is settled before the reply opens")
		}
	}
	s.True(started)
	s.Equal("in a quiet village where the sky brushes the fields in hues of gold, young Mear discovered a map leading to forgotten treasures.", final,
		"the words are joined into the sentence, the one after the stop included")

	settled := completes(events)
	s.Require().Len(settled, 1)
	s.False(settled[0].Interrupted)
	s.Positive(settled[0].TimeToFirstByteMs, "the caller was waiting for this reply")
	spoken := chunks(events)
	s.Len(spoken, 6)
	s.Equal(SampleRate, spoken[0].Audio.SampleRate)
	s.InDelta(6*640*1000.0/SampleRate, settled[0].AudioDurationMs, 0.01)

	var said string
	for _, event := range events {
		if transcript, ok := event.(sts.OutputTranscript); ok {
			s.Equal(settled[0].ResponseID, transcript.ResponseID)
			said += transcript.Text
		}
	}
	s.Equal("Ooh, what happens next?", said)
}

func (s *PhonicSuite) TestAReplyIsHeldUntilItIsKnownNotToHaveBeenCutOff() {
	provider := s.newProvider()
	s.feed(provider, spokenTurn)
	s.Empty(completes(s.drain(provider)), "a barge-in is reported only after the reply's end")
	s.Len(completes(s.settled(provider)), 1)
}

func (s *PhonicSuite) TestABargeInSettlesTheReplyAsCutOffOnce() {
	provider := s.newProvider()
	s.feed(provider, bargeIn)
	settled := completes(s.settled(provider))

	s.Require().Len(settled, 1)
	s.True(settled[0].Interrupted)
}

func (s *PhonicSuite) TestAToolCallBeforeAnyWordsBelongsToTheReplyThatFollows() {
	provider := s.newProvider()
	s.feed(provider, toolCallFirst)
	events := s.drain(provider)

	var call sts.ToolCall
	var opened int
	for _, event := range events {
		switch typed := event.(type) {
		case sts.ToolCall:
			call = typed
		case sts.ResponseStarted:
			opened++
		}
	}
	s.Equal(1, opened, "the speech after the call is the same reply")
	s.Equal("get_weather", call.Name)
	s.Equal("tool_call_c18f3bc0-8b02-46cb-aff6-d11432d9222b", call.CallID)
	s.JSONEq(`{"city":"Paris"}`, call.Arguments)
	spoken := chunks(events)
	s.Require().NotEmpty(spoken)
	s.Equal(call.ResponseID, spoken[0].ResponseID)
}

func (s *PhonicSuite) TestAToolCallPhonicGaveUpOnIsCancelled() {
	provider := s.newProvider()
	s.feedOne(provider, `{"type":"tool_call_interrupted","tool_call_id":"tool_call_1","tool_name":"get_weather"}`)

	events := s.drain(provider)
	s.Require().Len(events, 1)
	s.Equal(sts.ToolCancel{CallIDs: []string{"tool_call_1"}}, events[0])
}

func (s *PhonicSuite) TestInterruptDropsTheRestOfTheReplyUntilTheNextOne() {
	provider := s.newProvider()
	s.feed(provider, toolCallFirst)
	s.drain(provider)

	s.Require().NoError(provider.Interrupt(0))
	settled := completes(s.drain(provider))
	s.Require().Len(settled, 1)
	s.True(settled[0].Interrupted)

	s.feedOne(provider, `{"type":"audio_chunk","text":"more ","audio":"6APoAw=="}`)
	s.feedOne(provider, `{"type":"assistant_finished_speaking"}`)
	s.Empty(s.settled(provider), "nothing of the reply that was cut off is forwarded or settled again")

	s.feedOne(provider, `{"type":"assistant_started_speaking"}`)
	s.feedOne(provider, `{"type":"audio_chunk","text":"Hello.","audio":"6APoAw=="}`)
	next := chunks(s.drain(provider))
	s.Require().Len(next, 1)
	s.Equal(settled[0].Generation+1, next[0].Generation)
}

func (s *PhonicSuite) TestSpeechPhonicCouldNotTranscribeLeavesNoTranscript() {
	provider := s.newProvider()
	s.feedOne(provider, `{"type":"user_started_speaking"}`)
	s.feedOne(provider, `{"type":"input_cancelled"}`)
	s.feedOne(provider, `{"type":"assistant_started_speaking"}`)

	for _, event := range s.drain(provider) {
		_, transcribed := event.(sts.InputTranscript)
		s.False(transcribed)
	}
}

func (s *PhonicSuite) TestAReplyNobodyWaitedForReportsNoWait() {
	provider := s.newProvider()
	s.feedOne(provider, `{"type":"assistant_started_speaking"}`)
	s.feedOne(provider, `{"type":"audio_chunk","text":"Hi!","audio":"6APoAw=="}`)
	s.feedOne(provider, `{"type":"assistant_finished_speaking"}`)

	settled := completes(s.settled(provider))
	s.Require().Len(settled, 1)
	s.Zero(settled[0].TimeToFirstByteMs)
}
