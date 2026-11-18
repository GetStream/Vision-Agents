package echocancel

import (
	"bytes"
	"errors"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
	"time"

	webrtcaudio "github.com/GetStream/getstream-go-webrtc/audio"
	"github.com/stretchr/testify/suite"
	"github.com/thesyncim/gopus"
	"github.com/thesyncim/gopus/container/ogg"
)

// chunk is what a call carries per packet: 20 ms.
const chunk = 2 * FrameSamples

// CallSuite holds a synthesized phone call: the agent's voice goes out, and the caller's line
// brings it back late, quieter and through a room, with the caller's own voice on top.
type CallSuite struct {
	suite.Suite
	// agent is the agent's voice and caller the caller's, recorded by two voices of macOS's say.
	agent, caller []int16
}

func TestCall(t *testing.T) {
	suite.Run(t, new(CallSuite))
}

func (s *CallSuite) SetupSuite() {
	s.agent = s.fixture("reply")
	s.caller = s.fixture("long")
}

// fixture is one of the agent's recorded voices, internal/agent/testdata/audiofloor/name.ogg,
// 16 kHz mono.
func (s *CallSuite) fixture(name string) []int16 {
	raw, err := os.ReadFile(filepath.Join("..", "agent", "testdata", "audiofloor", name+".ogg"))
	s.Require().NoError(err)
	reader, err := ogg.NewReader(bytes.NewReader(raw))
	s.Require().NoError(err)
	decoder, err := gopus.NewDecoder(gopus.DefaultDecoderConfig(SampleRate, 1))
	s.Require().NoError(err)
	pcm := make([]int16, 5760)
	var out []int16
	var granule uint64
	for {
		packet, position, err := reader.ReadPacket()
		if errors.Is(err, io.EOF) {
			break
		}
		s.Require().NoError(err)
		if bytes.HasPrefix(packet, []byte("OpusHead")) || bytes.HasPrefix(packet, []byte("OpusTags")) {
			continue
		}
		n, err := decoder.DecodeInt16(packet, pcm)
		s.Require().NoError(err)
		out = append(out, pcm[:n]...)
		granule = max(granule, position)
	}
	// The granule position and pre-skip count 48 kHz samples.
	return out[int(reader.PreSkip())/3 : min(len(out), int(granule)/3)]
}

// phoneRate is the rate a phone line carries, G.711's 8 kHz.
const phoneRate = 8_000

// line is a caller's phone and the line to it: the agent's voice reaches the phone as G.711
// μ-law at 8 kHz, plays in a small room and comes back to the microphone delay later and loss
// dB quieter, with the caller's voice and the room's hiss, and goes back the way it came, up
// to the 16 kHz the agent hears at.
type line struct {
	delay int
	room  []float64
	// played is what the phone played, still on its way back.
	played []float64
	noise  *rand.Rand
	// down is the gateway's resampling to the line, and up the resampling from it, one for
	// what comes back and one each for the echo and the caller's voice alone in it.
	down, downVoice, up, upEcho, upVoice *webrtcaudio.StreamResampler
}

func newLine(delay time.Duration, loss float64) *line {
	random := rand.New(rand.NewPCG(1, 2))
	// 30 ms of reverberation, decaying by 60 dB, under a direct path.
	room := make([]float64, phoneRate*30/1000)
	gain := math.Pow(10, -loss/20)
	var energy float64
	for i := range room {
		decay := math.Pow(10, -3*float64(i)/float64(len(room)))
		room[i] = random.NormFloat64() * decay * 0.3
		energy += room[i] * room[i]
	}
	room[0] = 1
	energy++
	for i := range room {
		room[i] *= gain / math.Sqrt(energy)
	}
	toLine := func() *webrtcaudio.StreamResampler {
		return webrtcaudio.NewStreamResampler(webrtcaudio.FormatS16, phoneRate, 1, chunk/4)
	}
	fromLine := func() *webrtcaudio.StreamResampler {
		return webrtcaudio.NewStreamResampler(webrtcaudio.FormatS16, SampleRate, 1, chunk)
	}
	samples := int(delay.Seconds() * phoneRate)
	return &line{
		delay: samples, room: room, played: make([]float64, samples+len(room)), noise: random,
		down: toLine(), downVoice: toLine(), up: fromLine(), upEcho: fromLine(), upVoice: fromLine(),
	}
}

// carry returns what comes back to the agent for one chunk it sent and one chunk of the
// caller's voice, and the echo and the caller's voice alone in it, each as the line carries it.
func (l *line) carry(sent, voice []int16) (back, echo, said []int16) {
	toPhone := muLaw(resample(l.down, sent, SampleRate))
	spoken := resample(l.downVoice, voice, SampleRate)
	for _, sample := range toPhone {
		l.played = append(l.played, float64(sample))
	}
	mic := make([]int16, len(toPhone))
	heard := make([]int16, len(toPhone))
	for i := range toPhone {
		// The oldest sample of this chunk's echo is the one delay samples back.
		at := len(l.played) - len(toPhone) + i - l.delay
		var v float64
		for k, tap := range l.room {
			v += tap * l.played[at-k]
		}
		heard[i] = clip(v)
		mic[i] = clip(v + float64(spoken[i]) + l.noise.NormFloat64()*8)
	}
	l.played = l.played[len(toPhone):]
	return resample(l.up, muLaw(mic), phoneRate), resample(l.upEcho, heard, phoneRate),
		resample(l.upVoice, muLaw(spoken), phoneRate)
}

// resample is samples at rate through a stream resampler, whatever it has ready.
func resample(r *webrtcaudio.StreamResampler, samples []int16, rate int) []int16 {
	out, err := r.Write(webrtcaudio.FromInt16(samples, rate, 1))
	if err != nil {
		panic(err)
	}
	var joined []int16
	for _, pcm := range out {
		joined = append(joined, pcm.ToInt16().Int16()...)
	}
	return joined
}

func clip(v float64) int16 { return int16(max(math.MinInt16, min(math.MaxInt16, math.Round(v)))) }

// muLaw is 8 kHz samples coded as G.711 μ-law and back.
func muLaw(samples []int16) []int16 {
	coded, err := webrtcaudio.FromInt16(samples, phoneRate, 1).ToG711(webrtcaudio.MuLaw)
	if err != nil {
		panic(err)
	}
	decoded, err := webrtcaudio.FromG711(coded, webrtcaudio.MuLaw, phoneRate, 1)
	if err != nil {
		panic(err)
	}
	return decoded.ToInt16().Int16()
}

// talk is a call's audio, 16 kHz mono: the caller's voice and the echo of the agent's as the
// agent hears them, what came back to the agent, with both in it, and what the canceller made
// of that.
type talk struct {
	voice, echo, back, cleaned []int16
	// lost says which 10 ms frames never reached the agent, which stand in cleaned as silence.
	lost []bool
}

// converse runs a call of seconds through a phone line as a call carries it, 20 ms at a time,
// with the agent's voice at agentAt and the caller's at callerAt, cancelled as cfg says.
func (s *CallSuite) converse(cfg Config, phone *line, seconds float64, agentAt, callerAt []time.Duration) talk {
	return s.converseLosing(cfg, phone, seconds, agentAt, callerAt, nil)
}

// converseLosing is converse with the 20 ms chunks lost says never reaching the agent, from the
// chunk's index and what the phone sent in it. The line is told how much went missing, by the
// timestamps of the chunk that comes after, as the edge tells it.
func (s *CallSuite) converseLosing(cfg Config, phone *line, seconds float64, agentAt, callerAt []time.Duration,
	lost func(index int, back []int16) bool) talk {
	length := int(seconds*SampleRate) / chunk * chunk
	sent, voice := make([]int16, length), make([]int16, length)
	lay := func(dst, clip []int16, at []time.Duration) {
		for _, start := range at {
			copy(dst[int(start.Seconds()*SampleRate):], clip)
		}
	}
	lay(sent, s.agent, agentAt)
	lay(voice, s.caller, callerAt)

	call, err := NewCall(cfg)
	s.Require().NoError(err)
	listening, err := call.Listen()
	s.Require().NoError(err)
	defer listening.Close()

	var t talk
	var missing time.Duration
	for at := 0; at < length; at += chunk {
		call.Played(sent[at : at+chunk])
		back, echo, said := phone.carry(sent[at:at+chunk], voice[at:at+chunk])
		t.back, t.echo, t.voice = append(t.back, back...), append(t.echo, echo...), append(t.voice, said...)
		gone := lost != nil && lost(at/chunk, back)
		t.lost = append(t.lost, gone, gone)
		if gone {
			missing += 2 * FrameDuration
			t.cleaned = append(t.cleaned, make([]int16, len(back))...)
			continue
		}
		listening.Missed(missing)
		missing = 0
		t.cleaned = append(t.cleaned, listening.Clean(back)...)
	}
	return t
}

// energy is the sum of the squares of samples from..to seconds in, over the frames mask says.
func energy(samples []int16, from, to float64, mask func(frame int) bool) float64 {
	var sum float64
	for frame := int(from * 100); frame < int(to*100); frame++ {
		if !mask(frame) {
			continue
		}
		for _, v := range samples[frame*FrameSamples : (frame+1)*FrameSamples] {
			sum += float64(v) * float64(v)
		}
	}
	return sum
}

// loud says which 10 ms frames of samples hold sound above -50 dBFS.
func loud(samples []int16) func(frame int) bool {
	return func(frame int) bool {
		return energy(samples, float64(frame)/100, float64(frame+1)/100, func(int) bool { return true }) >
			FrameSamples*100*100
	}
}

func db(ratio float64) float64 { return 10 * math.Log10(ratio) }

// echoGone is how much quieter the canceller made the echo, in dB, over the frames from..to
// seconds in that hold echo and none of the caller's voice.
func echoGone(t talk, from, to float64) float64 {
	echoing, speaking := loud(t.echo), loud(t.voice)
	only := func(frame int) bool {
		return echoing(frame) && !speaking(frame) && (frame >= len(t.lost) || !t.lost[frame])
	}
	return db(energy(t.back, from, to, only) / energy(t.cleaned, from, to, only))
}

// voiceKept is how loud the caller's voice still is, in dB against what they said, over the
// frames from..to seconds in that hold it.
func voiceKept(t talk, from, to float64) float64 {
	speaking := loud(t.voice)
	return db(energy(t.cleaned, from, to, speaking) / energy(t.voice, from, to, speaking))
}

// replies are the agent's reply said over and over until seconds in, with a pause after each.
func (s *CallSuite) replies(until time.Duration) []time.Duration {
	reply := time.Duration(len(s.agent)) * time.Second / SampleRate
	var at []time.Duration
	for start := time.Duration(0); start < until; start += reply + 700*time.Millisecond {
		at = append(at, start)
	}
	return at
}

func (s *CallSuite) TestTheAgentsVoiceComingBackAQuarterSecondLateIsCancelled() {
	t := s.converse(Config{}, newLine(250*time.Millisecond, 10), 40, s.replies(40*time.Second), nil)

	first, rest := echoGone(t, 0, 3), echoGone(t, 10, 39)
	s.T().Logf("echo 250 ms late: %.1f dB quieter in the first 3 s, %.1f dB from 10 s on", first, rest)
	s.Greater(rest, 20.0)
}

func (s *CallSuite) TestTheAgentsVoiceComingBackSevenHundredMillisecondsLateIsCancelled() {
	t := s.converse(Config{}, newLine(700*time.Millisecond, 10), 40, s.replies(40*time.Second), nil)

	first, rest := echoGone(t, 0, 3), echoGone(t, 10, 39)
	s.T().Logf("echo 700 ms late: %.1f dB quieter in the first 3 s, %.1f dB from 10 s on", first, rest)
	s.Greater(rest, 20.0)
}

func (s *CallSuite) TestAnEchoLaterThanTheLongestDelayLookedForIsLeft() {
	t := s.converse(Config{MaxDelay: 300 * time.Millisecond}, newLine(700*time.Millisecond, 10), 40,
		s.replies(40*time.Second), nil)

	rest := echoGone(t, 10, 39)
	s.T().Logf("echo 700 ms late, looked for up to 300 ms: %.1f dB quieter from 10 s on", rest)
	s.Less(rest, 10.0)
}

func (s *CallSuite) TestTheCallersVoiceIsKeptOnceTheEchoIsCancelled() {
	t := s.converse(Config{}, newLine(250*time.Millisecond, 10), 50, s.replies(25*time.Second),
		[]time.Duration{30 * time.Second})

	kept := voiceKept(t, 30, 49)
	s.T().Logf("caller alone after 25 s of echo: %.1f dB", kept)
	s.InDelta(0, kept, 1)
}

func (s *CallSuite) TestTheCallersVoiceOverTheAgentIsHeard() {
	t := s.converse(Config{}, newLine(250*time.Millisecond, 10), 45, s.replies(45*time.Second),
		[]time.Duration{25 * time.Second})

	kept := voiceKept(t, 25, 44)
	s.T().Logf("caller over the agent: %.1f dB, echo before them %.1f dB quieter", kept, echoGone(t, 10, 25))
	s.Greater(kept, -6.0)
}

// losing loses the 20 ms chunks from at for d.
func losing(at, d time.Duration) func(int, []int16) bool {
	from, to := int(at/(2*FrameDuration)), int((at+d)/(2*FrameDuration))
	return func(index int, _ []int16) bool { return index >= from && index < to }
}

// dtx is a phone that stops sending while it hears nothing, as Opus DTX does: once its
// microphone has been quiet for hangover, it sends one chunk every 400 ms until it hears
// something again.
func dtx(hangover time.Duration) func(int, []int16) bool {
	quiet := 0
	return func(index int, back []int16) bool {
		var sum float64
		for _, v := range back {
			sum += float64(v) * float64(v)
		}
		if sum/float64(len(back)) > 100*100 {
			quiet = 0
			return false
		}
		quiet++
		return time.Duration(quiet)*2*FrameDuration > hangover && quiet%20 != 0
	}
}

func (s *CallSuite) TestTheEchoStaysCancelledAcrossAChunkLost() {
	for _, gap := range []time.Duration{20 * time.Millisecond, 100 * time.Millisecond, 600 * time.Millisecond} {
		at := 15 * time.Second
		t := s.converseLosing(Config{}, newLine(250*time.Millisecond, 10), 25, s.replies(25*time.Second), nil,
			losing(at, gap))

		after := (at + gap).Seconds()
		gone := echoGone(t, after, after+2)
		s.T().Logf("%s of the line lost at 15 s: %.1f dB quieter in the 2 s after it", gap, gone)
		s.Greater(gone, 20.0, "after %s lost", gap)
	}
}

func (s *CallSuite) TestTheEchoStaysCancelledFromAPhoneThatStopsSendingWhileQuiet() {
	for _, hangover := range []time.Duration{100 * time.Millisecond, 400 * time.Millisecond} {
		t := s.converseLosing(Config{}, newLine(250*time.Millisecond, 10), 40, s.replies(40*time.Second), nil,
			dtx(hangover))

		gone := echoGone(t, 10, 39)
		lost := 0
		for _, l := range t.lost {
			if l {
				lost++
			}
		}
		s.T().Logf("a phone that stops sending %s after it goes quiet (%d%% of the line never sent): %.1f dB quieter from 10 s on",
			hangover, lost*100/len(t.lost), gone)
		s.Greater(gone, 20.0, "with a hangover of %s", hangover)
	}
}

func (s *CallSuite) TestALineSilentForLongerThanTheVoiceHeldIsCancelledAfresh() {
	at := 15 * time.Second
	t := s.converseLosing(Config{}, newLine(250*time.Millisecond, 10), 30, s.replies(30*time.Second), nil,
		losing(at, 3*time.Second))

	gone := echoGone(t, 19, 29)
	s.T().Logf("3 s of the line lost at 15 s: %.1f dB quieter from 19 s on", gone)
	s.Greater(gone, 20.0)
}
