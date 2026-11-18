package echocancel

import (
	"bytes"
	"errors"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	webrtcaudio "github.com/GetStream/getstream-go-webrtc/audio"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/suite"
	"github.com/thesyncim/gopus"
	"github.com/thesyncim/gopus/container/ogg"

	"github.com/GetStream/Vision-Agents/acceleration/internal/echocancel/echodelay"
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
	// err is why the call could not be cancelled at all.
	err error
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

	// Calls run side by side (alongside), so a failure is returned for the test's own
	// goroutine to fail on.
	call, err := NewCall(cfg)
	if err != nil {
		return talk{err: err}
	}
	listening, err := call.Listen()
	if err != nil {
		return talk{err: err}
	}
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
		var sum float64
		for _, v := range samples[frame*FrameSamples : (frame+1)*FrameSamples] {
			sum += float64(v) * float64(v)
		}
		return sum > FrameSamples*100*100
	}
}

// cleanedLag is how far behind the line what AEC3 cleaned of it runs once it is cancelling:
// two 4 ms blocks, one more than while it has heard no voice to cancel.
const cleanedLag = 2 * aec3Lag

// voiceHeard is how much louder the caller's voice is than everything else left of the line
// once cleaned, in dB (0 for a line muted where both speak), over the 10 ms frames from..to seconds in where the caller and the
// echo are both loud, and how many such frames there are. What is left is the cleaned line
// less the caller's voice, cleanedLag late: the echo it left and the voice it took away.
func voiceHeard(t talk, from, to float64) (float64, int) {
	echoing, speaking := loud(t.echo), loud(t.voice)
	var said, left float64
	frames := 0
	for frame := int(from * 100); frame < int(to*100); frame++ {
		if !echoing(frame) || !speaking(frame) {
			continue
		}
		frames++
		for i := frame * FrameSamples; i < (frame+1)*FrameSamples; i++ {
			voice := float64(t.voice[i-cleanedLag])
			said += voice * voice
			off := float64(t.cleaned[i]) - voice
			left += off * off
		}
	}
	return db(said / left), frames
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

// alongside runs calls side by side, each on a goroutine of its own, and returns what each
// came to, failing the test on any that could not be cancelled: a call is seconds of audio
// cancelled 10 ms at a time, and one after the other they would take minutes under the race
// detector.
func (s *CallSuite) alongside(calls ...func() talk) []talk {
	talks := make([]talk, len(calls))
	var wg sync.WaitGroup
	for i, call := range calls {
		wg.Go(func() { talks[i] = call() })
	}
	wg.Wait()
	for i, t := range talks {
		s.Require().NoError(t.err, "call %d", i)
	}
	return talks
}

func (s *CallSuite) TestTheAgentsVoiceComingBackAQuarterAndSevenTenthsOfASecondLateIsCancelled() {
	delays := []time.Duration{250 * time.Millisecond, 700 * time.Millisecond}
	talks := s.alongside(
		func() talk { return s.converse(Config{}, newLine(delays[0], 10), 20, s.replies(20*time.Second), nil) },
		func() talk { return s.converse(Config{}, newLine(delays[1], 10), 20, s.replies(20*time.Second), nil) },
	)

	for i, t := range talks {
		first, rest := echoGone(t, 0, 3), echoGone(t, 8, 19)
		s.T().Logf("echo %s late: %.1f dB quieter in the first 3 s, %.1f dB from 8 s on", delays[i], first, rest)
		s.Greater(rest, 35.0, "echo %s late", delays[i])
	}
}

func (s *CallSuite) TestAnEchoLaterThanTheLongestDelayLookedForIsLeft() {
	t := s.alongside(func() talk {
		return s.converse(Config{MaxDelay: 300 * time.Millisecond}, newLine(700*time.Millisecond, 10), 20,
			s.replies(20*time.Second), nil)
	})[0]

	rest := echoGone(t, 8, 19)
	s.T().Logf("echo 700 ms late, looked for up to 300 ms: %.1f dB quieter from 8 s on", rest)
	s.Less(rest, 10.0)
}

func (s *CallSuite) TestTheCallersVoiceIsKeptOnceTheEchoIsCancelledAndHeardOverTheAgent() {
	talks := s.alongside(
		// The agent says its replies for 10 s, and the caller speaks once it has stopped.
		func() talk {
			return s.converse(Config{}, newLine(250*time.Millisecond, 10), 24, s.replies(10*time.Second),
				[]time.Duration{12 * time.Second})
		},
		// The caller speaks over the agent from 12 s on.
		func() talk {
			return s.converse(Config{}, newLine(250*time.Millisecond, 10), 24, s.replies(24*time.Second),
				[]time.Duration{12 * time.Second})
		},
	)

	kept := voiceKept(talks[0], 12, 23.9)
	heard, overlapping := voiceHeard(talks[1], 12, 23.9)
	s.T().Logf("caller alone after 10 s of echo: %.1f dB; caller over the agent, over %d frames of both: "+
		"%.1f dB above what else is left, echo before them %.1f dB quieter", kept, overlapping, heard,
		echoGone(talks[1], 6, 12))
	s.InDelta(0, kept, 1)
	s.Require().Positive(overlapping, "the caller never spoke over the echo")
	// Muting the line over the echo leaves 0 dB; the echo, 10 dB under the caller, was
	// measured on its own above.
	s.Greater(heard, 5.0)
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

// gapAt is when the line's audio stops in the calls with a gap: after the canceller has found
// the echo, partway through the agent's second reply.
const gapAt = 10 * time.Second

// gapCall is a call whose line, its echo delay late, loses gap of its audio at gapAt, running
// on until 1.5 s after it.
func (s *CallSuite) gapCall(cfg Config, delay, gap time.Duration) func() talk {
	return func() talk {
		end := gapAt + gap + 1500*time.Millisecond
		return s.converseLosing(cfg, newLine(delay, 10), end.Seconds(), s.replies(end), nil, losing(gapAt, gap))
	}
}

// The echo of what was said during the gap comes back after it, while the canceller is
// catching up with the voice that went out meanwhile: the second after the gap is the hardest.
func (s *CallSuite) TestTheEchoStaysCancelledInTheSecondAfterTheLineGoesMissing() {
	type gapped struct{ delay, gap time.Duration }
	var cases []gapped
	var calls []func() talk
	for _, delay := range []time.Duration{250 * time.Millisecond, 700 * time.Millisecond, 900 * time.Millisecond} {
		for _, gap := range []time.Duration{20 * time.Millisecond, 100 * time.Millisecond, 500 * time.Millisecond,
			600 * time.Millisecond, 800 * time.Millisecond, 980 * time.Millisecond} {
			cases = append(cases, gapped{delay, gap})
			calls = append(calls, s.gapCall(Config{}, delay, gap))
		}
	}

	for i, t := range s.alongside(calls...) {
		after := (gapAt + cases[i].gap).Seconds()
		quieter := echoGone(t, after, after+1)
		s.T().Logf("echo %s late, %s of the line lost: %.1f dB quieter in the second after", cases[i].delay,
			cases[i].gap, quieter)
		s.Greater(quieter, 30.0, "echo %s late, %s lost", cases[i].delay, cases[i].gap)
	}
}

// The voice a line holds is only what went out since the line's audio last came: what went
// out before it was handed to AEC3 already, which keeps as much as the longest delay looked
// for. So an echo later than the second a line holds is cancelled across a gap too.
func (s *CallSuite) TestAnEchoLaterThanTheVoiceALineHoldsStaysCancelledAcrossAGap() {
	t := s.alongside(s.gapCall(Config{MaxDelay: echodelay.Max}, 1100*time.Millisecond, 500*time.Millisecond))[0]

	after := (gapAt + 500*time.Millisecond).Seconds()
	quieter := echoGone(t, after, after+1)
	s.T().Logf("echo 1.1 s late, 500 ms of the line lost: %.1f dB quieter in the second after", quieter)
	s.Greater(quieter, 30.0)
}

func (s *CallSuite) TestTheEchoStaysCancelledFromAPhoneThatStopsSendingWhileQuiet() {
	hangovers := []time.Duration{100 * time.Millisecond, 400 * time.Millisecond}
	var calls []func() talk
	for _, hangover := range hangovers {
		calls = append(calls, func() talk {
			return s.converseLosing(Config{}, newLine(250*time.Millisecond, 10), 20, s.replies(20*time.Second), nil,
				dtx(hangover))
		})
	}

	for i, t := range s.alongside(calls...) {
		// The phone stopped sending, and started again, after 8 s, where it is measured.
		gaps := 0
		for frame := 800; frame < len(t.lost)-1; frame++ {
			if t.lost[frame] && !t.lost[frame+1] {
				gaps++
			}
		}
		lost := 0
		for _, l := range t.lost {
			if l {
				lost++
			}
		}
		gone := echoGone(t, 8, 19)
		s.T().Logf("a phone that stops sending %s after it goes quiet (%d%% of the line never sent, %d gaps after 8 s): "+
			"%.1f dB quieter from 8 s on", hangovers[i], lost*100/len(t.lost), gaps, gone)
		s.Positive(gaps, "with a hangover of %s the phone never stopped sending", hangovers[i])
		s.Greater(gone, 35.0, "with a hangover of %s", hangovers[i])
	}
}

func (s *CallSuite) TestALineSilentForLongerThanTheVoiceHeldIsCancelledAfreshOnce() {
	resets, dropped := testutil.ToFloat64(cancellerResets), testutil.ToFloat64(voiceDropped)
	const gap = 3 * time.Second
	end := gapAt + gap + 6*time.Second
	callerAt := gapAt + gap + 2*time.Second

	t := s.alongside(func() talk {
		return s.converseLosing(Config{}, newLine(250*time.Millisecond, 10), end.Seconds(), s.replies(end),
			[]time.Duration{callerAt}, losing(gapAt, gap))
	})[0]

	quieter := echoGone(t, (gapAt + gap + time.Second).Seconds(), end.Seconds())
	heard, overlapping := voiceHeard(t, callerAt.Seconds(), end.Seconds())
	s.T().Logf("%s of the line lost at %s: %.1f dB quieter from 1 s after it; the caller after it, over %d frames: "+
		"%.1f dB above what else is left", gap, gapAt, quieter, overlapping, heard)
	s.Greater(quieter, 35.0)
	s.Require().Positive(overlapping)
	s.Greater(heard, 5.0, "the caller's voice after the gap was not kept")
	s.Equal(1.0, testutil.ToFloat64(cancellerResets)-resets, "the canceller was not replaced exactly once")
	// The line holds a second of voice; what went out over the rest of the gap, and with the
	// 20 ms that ended it, was let go.
	s.Equal(float64(int(gap/FrameDuration)+2-playedFrames), testutil.ToFloat64(voiceDropped)-dropped)
}

func (s *CallSuite) TestAGapIsCountedFrameByFrame() {
	frames := testutil.ToFloat64(gapFrames)

	s.alongside(s.gapCall(Config{}, 250*time.Millisecond, 500*time.Millisecond))

	s.Equal(50.0, testutil.ToFloat64(gapFrames)-frames)
}
