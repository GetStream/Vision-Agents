package streamedge

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
	"github.com/thesyncim/gopus"
	"github.com/thesyncim/gopus/container/ogg"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/echocancel"
)

// EchoSuite is the agent's voice going out on its track and coming back on a caller's line.
type EchoSuite struct {
	suite.Suite
	ctx context.Context
	// reply is the agent's voice as a voice provider hands it over, at 24 kHz.
	reply audio.PcmData
}

func TestEchoSuite(t *testing.T) {
	suite.Run(t, new(EchoSuite))
}

func (s *EchoSuite) SetupSuite() {
	s.ctx = context.Background()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "audiofloor", "reply.ogg"))
	s.Require().NoError(err)
	reader, err := ogg.NewReader(bytes.NewReader(raw))
	s.Require().NoError(err)
	decoder, err := gopus.NewDecoder(gopus.DefaultDecoderConfig(echocancel.SampleRate, 1))
	s.Require().NoError(err)
	pcm := make([]int16, 5760)
	var voice []int16
	for {
		packet, _, err := reader.ReadPacket()
		if errors.Is(err, io.EOF) {
			break
		}
		s.Require().NoError(err)
		if bytes.HasPrefix(packet, []byte("OpusHead")) || bytes.HasPrefix(packet, []byte("OpusTags")) {
			continue
		}
		n, err := decoder.DecodeInt16(packet, pcm)
		s.Require().NoError(err)
		voice = append(voice, pcm[:n]...)
	}
	s.reply = audio.Resample(audio.PcmData{Samples: voice, SampleRate: echocancel.SampleRate, Channels: 1}, 24_000, 1)
}

// phone is a caller's phone playing what the track sent: it comes back 300 ms later, 10 dB
// quieter, with two reflections off the walls.
type phone struct {
	played []float64
}

const phoneDelay = 300 * time.Millisecond

func newPhone() *phone {
	return &phone{played: make([]float64, int(phoneDelay.Seconds()*echocancel.SampleRate)+200)}
}

// back is what the phone's microphone hears of 20 ms it played.
func (p *phone) back(played []int16) []int16 {
	for _, v := range played {
		p.played = append(p.played, float64(v))
	}
	delay := int(phoneDelay.Seconds() * echocancel.SampleRate)
	heard := make([]int16, len(played))
	for i := range heard {
		at := len(p.played) - len(played) + i - delay
		v := 0.3*p.played[at] + 0.1*p.played[at-37] - 0.05*p.played[at-160]
		heard[i] = int16(max(math.MinInt16, min(math.MaxInt16, math.Round(v))))
	}
	p.played = p.played[len(played):]
	return heard
}

// call plays the reply over and over for seconds through a speaker that tells echo of what it
// sends, as a provider streams it, the way the track takes it, and returns what came back on
// the caller's line and what was left of it once cleaned.
func (s *EchoSuite) call(echo *echocancel.Call, seconds int) (back, cleaned []int16) {
	talker := newSpeaker(slog.New(slog.DiscardHandler), echo)
	s.T().Cleanup(func() { _ = talker.Close() })
	line, err := echo.Listen()
	s.Require().NoError(err)
	s.T().Cleanup(func() { line.Close() })
	decoder, err := gopus.NewDecoder(gopus.DefaultDecoderConfig(echocancel.SampleRate, 1))
	s.Require().NoError(err)
	caller := newPhone()

	// Pieces of a provider's stream are seldom whole frames, and resampled to 16 kHz each one
	// rounds to a whole number of samples.
	piece := s.reply.SampleRate/50 + 7
	pause := 0
	at := 0
	frame := make([]int16, 5760)
	for range seconds * 50 {
		// The provider keeps a frame or two ahead of the track while it speaks, and pauses a
		// second between replies.
		if pause > 0 {
			pause--
		} else if len(talker.frames) < 2 {
			end := min(at+piece, len(s.reply.Samples))
			s.Require().NoError(talker.Write(audio.PcmData{Samples: s.reply.Samples[at:end],
				SampleRate: s.reply.SampleRate, Channels: 1}))
			if at = end; at == len(s.reply.Samples) {
				at, pause = 0, 50
			}
		}
		sample, err := talker.NextSample(s.ctx)
		s.Require().NoError(err)
		played := make([]int16, echoFrameSamples)
		if sample.Data != nil {
			n, err := decoder.DecodeInt16(sample.Data, frame)
			s.Require().NoError(err)
			copy(played, frame[:n])
		}
		heard := caller.back(played)
		back = append(back, heard...)
		cleaned = append(cleaned, line.Clean(append([]int16(nil), heard...))...)
	}
	return back, cleaned
}

// loudness is the energy of samples over the 10 ms frames from from seconds in that hold
// what loud says they do.
func loudness(samples []int16, from int, loud []int16) float64 {
	var sum float64
	for at := from * echocancel.SampleRate; at+echocancel.FrameSamples <= len(samples); at += echocancel.FrameSamples {
		if !audio.Audible(loud[at : at+echocancel.FrameSamples]) {
			continue
		}
		for _, v := range samples[at : at+echocancel.FrameSamples] {
			sum += float64(v) * float64(v)
		}
	}
	return sum
}

func (s *EchoSuite) TestTheVoiceTheTrackSendsIsCancelledFromWhatTheCallerSendsBack() {
	echo, err := echocancel.NewCall(echocancel.Config{})
	s.Require().NoError(err)

	back, cleaned := s.call(echo, 30)

	gone := 10 * math.Log10(loudness(back, 5, back)/loudness(cleaned, 5, back))
	s.T().Logf("the agent's voice coming back %s late, through the track's Opus: %.1f dB quieter from 5 s on",
		phoneDelay, gone)
	s.Greater(gone, 20.0)
}

func (s *EchoSuite) TestSpeechDroppedOnAnInterruptionIsNotTakenForVoiceSent() {
	echo, err := echocancel.NewCall(echocancel.Config{})
	s.Require().NoError(err)
	line, err := echo.Listen()
	s.Require().NoError(err)
	s.T().Cleanup(func() { line.Close() })
	talker := newSpeaker(slog.New(slog.DiscardHandler), echo)
	s.T().Cleanup(func() { _ = talker.Close() })

	s.Require().NoError(talker.Write(audio.PcmData{Samples: s.reply.Samples[:s.reply.SampleRate/5],
		SampleRate: s.reply.SampleRate, Channels: 1}))
	s.NotEmpty(talker.voice[talker.voiceAt:])
	talker.drop()
	s.Empty(talker.voice[talker.voiceAt:])

	_, err = talker.NextSample(s.ctx)
	s.Require().NoError(err)
	s.Equal(make([]int16, echoFrameSamples), talker.sent[:], "the frame after the drop is silence")
}

func (s *EchoSuite) TestACallWithNobodyToCancelFromKeepsNoVoice() {
	echo, err := echocancel.NewCall(echocancel.Config{})
	s.Require().NoError(err)
	talker := newSpeaker(slog.New(slog.DiscardHandler), echo)
	s.T().Cleanup(func() { _ = talker.Close() })

	s.Require().NoError(talker.Write(audio.PcmData{Samples: s.reply.Samples[:s.reply.SampleRate/5],
		SampleRate: s.reply.SampleRate, Channels: 1}))

	s.Empty(talker.voice)
}

func (s *EchoSuite) TestALineThatStartsMidUtteranceGetsTheVoiceFromTheNextOne() {
	echo, err := echocancel.NewCall(echocancel.Config{})
	s.Require().NoError(err)
	talker := newSpeaker(slog.New(slog.DiscardHandler), echo)
	s.T().Cleanup(func() { _ = talker.Close() })
	utterance := audio.PcmData{Samples: s.reply.Samples[:s.reply.SampleRate/5], SampleRate: s.reply.SampleRate, Channels: 1}

	s.Require().NoError(talker.Write(utterance))
	line, err := echo.Listen()
	s.Require().NoError(err)
	s.T().Cleanup(func() { line.Close() })
	s.Require().NoError(talker.Write(utterance))
	s.Empty(talker.voice, "voice kept from partway through would not line up with its frames")

	for talker.pending() || len(talker.frames) > 0 {
		_, err := talker.NextSample(s.ctx)
		s.Require().NoError(err)
	}
	_, err = talker.NextSample(s.ctx)
	s.Require().NoError(err)
	s.Require().NoError(talker.Write(utterance))
	s.NotEmpty(talker.voice)
}
