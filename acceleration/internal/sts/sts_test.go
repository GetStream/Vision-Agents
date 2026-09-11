package sts

import (
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

type STSSuite struct {
	suite.Suite
}

func TestSTSSuite(t *testing.T) {
	suite.Run(t, new(STSSuite))
}

func (s *STSSuite) TestCapabilitiesExpressOnlyWhatTheyDeclare() {
	all := Capabilities{
		Text: true, Tools: true, InputTranscript: true, OutputTranscript: true,
		SemanticTurns: true, ManualTurns: true, Endpointing: true,
	}
	for _, term := range []options.Term{
		options.SemanticTurns, options.ManualTurns, options.Endpointing, options.Tools,
		options.TextInput, options.InputTranscript, options.OutputTranscript,
	} {
		s.True(all.Expresses(term), "%s should be expressible when declared", term)
		s.False(Capabilities{}.Expresses(term), "%s should be refused when not declared", term)
	}
	s.False(all.Expresses(options.Diarize), "a transcriber's term means nothing to a speech-to-speech model")
}

func (s *STSSuite) TestAcceptsTreatsAudioAndTextAsImplicit() {
	sees := Capabilities{InputModalities: []string{options.ModalityImage}}
	s.True(sees.Accepts("audio"))
	s.True(sees.Accepts("text"))
	s.True(sees.Accepts(options.ModalityImage))
	s.False(Capabilities{}.Accepts(options.ModalityImage))
}

func (s *STSSuite) TestATurnMeasuresTheWaitFromWhenTheCallerStopped() {
	heardAt := time.Now().Add(-200 * time.Millisecond)
	turn := NewTurn("resp_1", 3, heardAt)

	chunk := turn.Chunk(audio.PcmData{Samples: make([]int16, 2400), SampleRate: 24_000, Channels: 1})
	s.Equal("resp_1", chunk.ResponseID)
	s.Equal(3, chunk.Generation)
	s.Equal(0, chunk.Index)

	second := turn.Chunk(audio.PcmData{Samples: make([]int16, 2400), SampleRate: 24_000, Channels: 1})
	s.Equal(1, second.Index)
	s.Equal(200, turn.SentMs())

	complete := turn.Complete("openai", "gpt-realtime-2", false, Usage{InputTokens: 10, OutputTokens: 20})
	s.Equal("resp_1", complete.ResponseID)
	s.Equal(3, complete.Generation)
	s.InDelta(200, complete.AudioDurationMs, 0.001)
	s.GreaterOrEqual(complete.TimeToFirstByteMs, 200.0, "the wait started when the caller stopped, not when the reply began")
	s.Equal(int64(10), complete.Usage.InputTokens)
	s.False(complete.Interrupted)
}

func (s *STSSuite) TestAReplyNobodyWaitedForReportsNoWait() {
	turn := NewTurn("", 1, time.Time{})
	s.Equal("r-1", turn.ID, "a vendor without response ids gets one made up from the generation")

	turn.Chunk(audio.PcmData{Samples: make([]int16, 240), SampleRate: 24_000, Channels: 1})
	complete := turn.Complete("gemini", "live", true, Usage{})
	s.Zero(complete.TimeToFirstByteMs)
	s.True(complete.Interrupted)
}
