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
	"slices"
	"testing"
	"time"

	webrtcaudio "github.com/GetStream/getstream-go-webrtc/audio"
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
	// reply is the agent's voice as a voice provider hands it over, at 24 kHz, and caller a
	// caller's, at 16 kHz.
	reply  audio.PcmData
	caller []int16
}

func TestEchoSuite(t *testing.T) {
	suite.Run(t, new(EchoSuite))
}

func (s *EchoSuite) SetupSuite() {
	s.ctx = context.Background()
	s.reply = audio.Resample(audio.PcmData{Samples: s.fixture("reply"), SampleRate: echocancel.SampleRate, Channels: 1},
		24_000, 1)
	s.caller = s.fixture("long")
}

// fixture is one of the agent's recorded voices, ../testdata/audiofloor/name.ogg, at 16 kHz.
func (s *EchoSuite) fixture(name string) []int16 {
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "audiofloor", name+".ogg"))
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
	return voice
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

// speaking is how a voice provider streams the reply in a test call.
type speaking struct {
	// piece is how many of the reply's samples go in each write. writeOnEmpty writes only
	// once the track has taken everything queued; otherwise the provider keeps a frame or
	// two ahead of the track.
	piece        int
	writeOnEmpty bool
	// lineAt is the frame of the track the caller's line opens at, the caller having been in
	// the call from the start.
	lineAt int
}

// call plays the reply over and over for seconds through a speaker that tells echo of what it
// sends, as a provider streams it, the way the track takes it, and returns what came back on
// the caller's line and what was left of it once cleaned: what came back itself before the
// line opened.
func (s *EchoSuite) call(echo *echocancel.Call, seconds int, how speaking) (back, cleaned []int16) {
	talker := newSpeaker(slog.New(slog.DiscardHandler), echo)
	s.T().Cleanup(func() { _ = talker.Close() })
	echo.Expect(1)
	var line *echocancel.Line
	decoder, err := gopus.NewDecoder(gopus.DefaultDecoderConfig(echocancel.SampleRate, 1))
	s.Require().NoError(err)
	caller := newPhone()

	pause := 0
	at := 0
	frame := make([]int16, 5760)
	for i := range seconds * 50 {
		if i == how.lineAt {
			line, err = echo.Listen()
			s.Require().NoError(err)
			s.T().Cleanup(func() { line.Close() })
		}
		// The provider pauses a second between replies.
		if pause > 0 {
			pause--
		} else if len(talker.frames) == 0 || (!how.writeOnEmpty && len(talker.frames) < 2) {
			end := min(at+how.piece, len(s.reply.Samples))
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
		if line == nil {
			cleaned = append(cleaned, heard...)
			continue
		}
		cleaned = append(cleaned, line.Clean(append([]int16(nil), heard...))...)
	}
	return back, cleaned
}

// gone is how much quieter cleaned is than back, in dB, from..to seconds in.
func gone(back, cleaned []int16, from, to float64) float64 {
	return 10 * math.Log10(loudness(back, from, to, back)/loudness(cleaned, from, to, back))
}

// loudness is the energy of samples over the 10 ms frames from..to seconds in that hold what
// loud says they do.
func loudness(samples []int16, from, to float64, loud []int16) float64 {
	var sum float64
	end := min(len(samples), int(to*echocancel.SampleRate))
	for at := int(from * echocancel.SampleRate); at+echocancel.FrameSamples <= end; at += echocancel.FrameSamples {
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

	// Pieces of a provider's stream are seldom whole frames, and resampled to 16 kHz each one
	// rounds to a whole number of samples.
	back, cleaned := s.call(echo, 15, speaking{piece: s.reply.SampleRate/50 + 7})

	quieter := gone(back, cleaned, 5, 15)
	s.T().Logf("the agent's voice coming back %s late, through the track's Opus: %.1f dB quieter from 5 s on",
		phoneDelay, quieter)
	s.Greater(quieter, 20.0)
}

// A provider slower than the track lets the queue run dry between its pieces, and each piece
// is then taken as soon as it is written, before the resampler has given up its end.
func (s *EchoSuite) TestTheVoiceWrittenOnlyOnceTheQueueRunsDryIsCancelled() {
	echo, err := echocancel.NewCall(echocancel.Config{})
	s.Require().NoError(err)

	back, cleaned := s.call(echo, 15, speaking{piece: s.reply.SampleRate / 10, writeOnEmpty: true})

	quieter := gone(back, cleaned, 5, 15)
	s.T().Logf("100 ms pieces written on an empty queue: %.1f dB quieter from 5 s on", quieter)
	s.Greater(quieter, 50.0)
}

// A phone that is answered while the agent greets it, or a track the SFU sends again, opens
// the caller's line partway through a reply.
func (s *EchoSuite) TestALineThatOpensPartwayThroughAReplyIsCancelledFromThere() {
	echo, err := echocancel.NewCall(echocancel.Config{})
	s.Require().NoError(err)
	reply := float64(len(s.reply.Samples)) / float64(s.reply.SampleRate)

	back, cleaned := s.call(echo, 12, speaking{piece: s.reply.SampleRate/50 + 7, lineAt: 50})

	quieter := gone(back, cleaned, 4, reply)
	s.T().Logf("a line opened 1 s into a %.1f s reply: %.1f dB quieter from 4 s to its end", reply, quieter)
	s.Greater(quieter, 20.0)
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

// Once an utterance has gone out, the track sends silence, and so does the voice it reports.
func (s *EchoSuite) TestTheVoiceSentStopsWithTheUtterance() {
	echo, err := echocancel.NewCall(echocancel.Config{})
	s.Require().NoError(err)
	echo.Expect(1)
	talker := newSpeaker(slog.New(slog.DiscardHandler), echo)
	s.T().Cleanup(func() { _ = talker.Close() })

	// A fifth of a second from the middle of a word, so its last frame is loud.
	s.Require().NoError(talker.Write(audio.PcmData{Samples: s.reply.Samples[s.reply.SampleRate : s.reply.SampleRate*6/5],
		SampleRate: s.reply.SampleRate, Channels: 1}))
	for len(talker.frames) > 0 {
		_, err := talker.NextSample(s.ctx)
		s.Require().NoError(err)
	}
	s.Require().NotEqual(make([]int16, echoFrameSamples), talker.sent[:], "the utterance sent no voice")
	_, err = talker.NextSample(s.ctx)
	s.Require().NoError(err)

	s.Equal(make([]int16, echoFrameSamples), talker.sent[:])
}

// The voice kept for an utterance, once the encoder is flushed, is the whole utterance
// resampled, the resampler's end included, behind the resampler's lead.
func (s *EchoSuite) TestTheVoiceKeptOnceFlushedIsTheWholeUtterance() {
	echo, err := echocancel.NewCall(echocancel.Config{})
	s.Require().NoError(err)
	echo.Expect(1)
	talker := newSpeaker(slog.New(slog.DiscardHandler), echo)
	s.T().Cleanup(func() { _ = talker.Close() })
	utterance := audio.PcmData{Samples: s.reply.Samples[:s.reply.SampleRate*21/100], SampleRate: s.reply.SampleRate,
		Channels: 1}

	s.Require().NoError(talker.Write(utterance))
	talker.mu.Lock()
	talker.flush()
	kept := slices.Clone(talker.voice[talker.voiceAt:])
	talker.mu.Unlock()

	once := webrtcaudio.NewStreamResampler(webrtcaudio.FormatS16, echocancel.SampleRate, 1, 0)
	want := make([]int16, resamplerLead(utterance.SampleRate))
	for _, write := range [][]webrtcaudio.PCM{s.resampled(once.Write(webrtcaudio.FromInt16(utterance.Samples,
		utterance.SampleRate, 1))), s.resampled(once.Flush())} {
		for _, pcm := range write {
			want = append(want, pcm.ToInt16().Int16()...)
		}
	}
	s.Equal(want, kept)
}

// resampled is what a resampler returned, failing the test on its error.
func (s *EchoSuite) resampled(pcm []webrtcaudio.PCM, err error) []webrtcaudio.PCM {
	s.Require().NoError(err)
	return pcm
}
