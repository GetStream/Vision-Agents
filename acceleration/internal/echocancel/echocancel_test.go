package echocancel

import (
	"math/rand/v2"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/echocancel/echodelay"
)

// LineSuite is one call with one line listened to.
type LineSuite struct {
	suite.Suite
	call *Call
	line *Line
}

func TestLine(t *testing.T) {
	suite.Run(t, new(LineSuite))
}

func (s *LineSuite) SetupTest() {
	call, err := NewCall(Config{})
	s.Require().NoError(err)
	line, err := call.Listen()
	s.Require().NoError(err)
	s.call, s.line = call, line
}

func (s *LineSuite) TearDownTest() { s.line.Close() }

// speech is n samples of noise loud enough to be a voice, the same for the same seed.
func speech(n int, seed uint64) []int16 {
	random := rand.New(rand.NewPCG(seed, seed))
	samples := make([]int16, n)
	for i := range samples {
		samples[i] = int16(random.NormFloat64() * 3000)
	}
	return samples
}

// aec3Lag is how far behind what it is given AEC3's output runs: one 4 ms block, which its
// 10 ms frames are cut into and put back together from.
const aec3Lag = 64

func (s *LineSuite) TestALineWhileTheAgentHasSaidNothingComesThroughUnchanged() {
	heard := speech(50*chunk, 1)
	var back []int16
	for at := 0; at < len(heard); at += chunk {
		back = append(back, s.line.Clean(append([]int16(nil), heard[at:at+chunk]...))...)
	}
	s.Equal(make([]int16, aec3Lag), back[:aec3Lag])
	s.Equal(heard[:len(heard)-aec3Lag], back[aec3Lag:])
}

func (s *LineSuite) TestChunksOfAnyLengthComeBackInOrderAsWholeFrames() {
	heard := speech(20*chunk, 2)
	var back []int16
	at := 0
	for _, n := range []int{1, 7, 152, 159, 160, 161, 0, 333, 480, 2, 319, 320, 1000} {
		out := s.line.Clean(append([]int16(nil), heard[at:at+n]...))
		s.Zero(len(out)%FrameSamples, "a chunk of %d came back as %d samples", n, len(out))
		back = append(back, out...)
		at += n
	}
	s.Equal(at/FrameSamples*FrameSamples, len(back))
	s.Equal(heard[:len(back)-aec3Lag], back[aec3Lag:])
}

func (s *LineSuite) TestAChunkOfWholeFramesIsCleanedWhereItIs() {
	s.call.Played(speech(chunk, 3))
	in := speech(chunk, 4)
	out := s.line.Clean(in)
	s.Require().Len(out, chunk)
	s.Same(&in[0], &out[0])
}

func (s *LineSuite) TestPlayingAndCleaningAllocateNothing() {
	voice, heard := speech(chunk, 5), speech(chunk, 6)
	in := make([]int16, chunk)
	round := func() {
		s.call.Played(voice)
		copy(in, heard)
		s.line.Clean(in)
	}
	// AEC3 sizes some of its buffers on the first frames it is given.
	for range 200 {
		round()
	}
	s.Zero(testing.AllocsPerRun(200, round))
}

func (s *LineSuite) TestALineWhoseAudioStopsHoldsOnlyTheLastSecondOfTheVoice() {
	for i := range 150 {
		frame := make([]int16, FrameSamples)
		frame[0] = int16(i + 1)
		s.call.Played(frame)
	}
	taken, dropped := s.line.take()
	s.Equal(playedFrames, taken)
	s.True(dropped, "the frames let go are not said to be")
	s.Equal(int16(51), s.line.taking[0][0], "the oldest frame held is not the first of the last second")
	s.Equal(int16(150), s.line.taking[playedFrames-1][0])
}

func (s *LineSuite) TestAPartialFrameOfVoiceIsHeldUntilItIsWhole() {
	s.call.Played(speech(FrameSamples-1, 7))
	s.Zero(s.taken(s.line))
	s.call.Played(speech(FrameSamples+1, 8))
	s.Equal(2, s.taken(s.line))
}

func (s *LineSuite) TestAClosedLineIsPlayedNoMoreVoice() {
	other, err := s.call.Listen()
	s.Require().NoError(err)
	other.Close()
	s.call.Played(speech(chunk, 9))
	s.Zero(s.taken(other))
	s.Equal(2, s.taken(s.line))
}

func (s *LineSuite) TestLinesCleanedAlongsideTheVoicePlayedAndLinesComingAndGoingDoNotRace() {
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Go(func() {
		voice := speech(chunk, 10)
		for {
			select {
			case <-stop:
				return
			default:
				s.call.Played(voice)
			}
		}
	})
	var churned error
	wg.Go(func() {
		for range 50 {
			line, err := s.call.Listen()
			if err != nil {
				churned = err
				return
			}
			line.Clean(speech(chunk, 11))
			line.Close()
		}
	})
	for range 200 {
		s.line.Clean(speech(chunk, 12))
	}
	close(stop)
	wg.Wait()
	s.NoError(churned, "lines did not come and go")
}

// ConfigSuite is how far an echo is looked for.
type ConfigSuite struct {
	suite.Suite
}

func TestConfig(t *testing.T) {
	suite.Run(t, new(ConfigSuite))
}

func (s *ConfigSuite) TestHalfASecondIsWebRTCsOwnFiveFilters() {
	s.Equal(5, filters(512*time.Millisecond))
}

func (s *ConfigSuite) TestTheDefaultLooksForAnEchoASecondLate() {
	config, err := aec3Config(Config{})
	s.Require().NoError(err)
	s.Equal(11, config.Delay.NumFilters)
	s.GreaterOrEqual(filterWindow+time.Duration(config.Delay.NumFilters-1)*filterShift, time.Second)
}

func (s *ConfigSuite) TestALongestDelayOutsideItsBoundsIsRefused() {
	s.ErrorIs(Check(Config{MaxDelay: 100 * time.Millisecond}), echodelay.ErrOutOfBounds)
	s.ErrorIs(Check(Config{MaxDelay: 2 * time.Second}), echodelay.ErrOutOfBounds)
	s.NoError(Check(Config{MaxDelay: echodelay.Max}))
	s.NoError(Check(Config{MaxDelay: echodelay.Min}))
}

func (s *ConfigSuite) TestEveryLongestDelayInItsBoundsMakesACanceller() {
	// Every count of matched filters from one to the 13 echodelay.Max takes.
	for delay := echodelay.Min; delay <= echodelay.Max; delay += filterShift {
		call, err := NewCall(Config{MaxDelay: delay})
		s.Require().NoError(err)
		line, err := call.Listen()
		s.Require().NoError(err, "a longest delay of %s", delay)
		line.Close()
	}
}

// taken is how many frames of voice line holds, which are taken from it.
func (s *LineSuite) taken(line *Line) int {
	n, _ := line.take()
	return n
}

// A line whose audio stops for longer than the voice it holds gets a new canceller, which has
// estimated nothing of the echo yet when the line closes soon after.
func (s *LineSuite) TestALineClosedJustAfterItsCancellerWasReplacedReportsWhatTheOldOneFound() {
	voice := speech(10*SampleRate, 13)
	// The echo comes back 100 ms late and 10 dB down.
	echo := make([]int16, len(voice))
	for i := 1600; i < len(voice); i++ {
		echo[i] = voice[i-1600] / 3
	}
	for at := 0; at < len(voice); at += chunk {
		s.call.Played(voice[at : at+chunk])
		s.line.Clean(append([]int16(nil), echo[at:at+chunk]...))
	}
	found := s.line.Metrics()
	s.Require().Greater(found.ERLE, 3.0, "the canceller found no echo to report")

	s.line.Missed(3 * time.Second)
	s.call.Played(make([]int16, chunk))
	s.line.Clean(make([]int16, chunk))

	s.NotEqual(found, s.line.Metrics(), "the canceller was not replaced")
	s.Equal(found, s.line.Close())
}

// measured is how many lines have had their echo return loss enhancement observed.
func measured() uint64 {
	var metric dto.Metric
	if err := lineERLE.Write(&metric); err != nil {
		panic(err)
	}
	return metric.GetHistogram().GetSampleCount()
}

// lines is how many lines have been counted, by whether the agent's voice went out on them.
func lines(voiced bool) float64 {
	return testutil.ToFloat64(linesCancelled.WithLabelValues(strconv.FormatBool(voiced)))
}

func (s *LineSuite) TestALineTheAgentSpokeOnIsCountedAndMeasuredOnceHoweverOftenItCloses() {
	counted, observed := lines(true), measured()
	s.call.Played(speech(chunk, 20))
	s.line.Clean(speech(chunk, 21))

	s.line.Close()
	s.line.Close()

	s.Equal(1.0, lines(true)-counted)
	s.Equal(uint64(1), measured()-observed)
}

func (s *LineSuite) TestALineTheAgentNeverSpokeOnIsCountedButNotMeasured() {
	counted, observed := lines(false), measured()
	s.call.Played(make([]int16, chunk))
	s.line.Clean(speech(chunk, 22))

	s.line.Close()

	s.Equal(1.0, lines(false)-counted)
	s.Equal(uint64(0), measured()-observed)
}

// AEC3 that was handed the agent's voice but never cleaned the line has estimated nothing.
func (s *LineSuite) TestALineWhoseAudioNeverCameAfterTheAgentSpokeIsCountedButNotMeasured() {
	counted, observed := lines(true), measured()
	s.call.Played(speech(chunk, 23))

	s.line.Close()

	s.Equal(1.0, lines(true)-counted)
	s.Equal(uint64(0), measured()-observed)
}

func (s *LineSuite) TestALineWhoseVoiceOverflowedGetsANewCanceller() {
	resets := testutil.ToFloat64(cancellerResets)
	for range playedFrames + 50 {
		s.call.Played(speech(FrameSamples, 24))
	}

	s.line.Clean(speech(chunk, 25))

	s.Equal(1.0, testutil.ToFloat64(cancellerResets)-resets)
}

func (s *LineSuite) TestAGapLongerThanTheVoiceHeldGetsANewCancellerWithoutCleaningItAsSilence() {
	resets, frames := testutil.ToFloat64(cancellerResets), testutil.ToFloat64(gapFrames)
	started := time.Now()

	s.line.Missed(time.Minute)
	s.line.Clean(speech(chunk, 26))

	s.Less(time.Since(started), 500*time.Millisecond, "a minute of silence was cleaned frame by frame")
	s.Equal(1.0, testutil.ToFloat64(cancellerResets)-resets)
	s.Equal(0.0, testutil.ToFloat64(gapFrames)-frames)
}

func (s *LineSuite) TestTheSilenceAGapIsCleanedAsStaysSilence() {
	for range 10 {
		s.call.Played(speech(FrameSamples, 27))
	}
	s.line.Clean(speech(chunk, 28))

	s.line.Missed(100 * time.Millisecond)

	s.Equal([FrameSamples]int16{}, s.line.silence)
}

// The start of a frame held from before a gap is finished with silence, not with what comes
// after the gap. With no voice played the line comes through as it is, a block late.
func (s *LineSuite) TestAFrameHeldFromBeforeAGapIsNotFinishedWithWhatComesAfterIt() {
	before, after := speech(FrameSamples/2, 29), speech(FrameSamples/2, 30)
	s.Empty(s.line.Clean(append([]int16(nil), before...)))

	s.line.Missed(30 * time.Millisecond)
	out := s.line.Clean(append([]int16(nil), after...))

	s.Require().Len(out, FrameSamples, "the frame from before the gap was not handed back")
	want := append(append(make([]int16, aec3Lag), before...), make([]int16, FrameSamples)...)
	s.Equal(want[:FrameSamples], out)
}

func (s *LineSuite) TestALineClosedLeavesNothingOfItHeldByTheCall() {
	s.line.Close()

	s.Zero(len(s.call.lines))
	for _, held := range s.call.lines[:cap(s.call.lines)] {
		s.Nil(held)
	}
}

// What the agent says to a phone caller before their audio arrives comes back as echo once it
// does: their line starts with the last second of it.
func (s *LineSuite) TestALineThatOpensAfterItsCallerArrivedStartsWithTheVoiceSaidToThem() {
	call, err := NewCall(Config{})
	s.Require().NoError(err)
	call.Expect(1)
	for i := range playedFrames + 20 {
		frame := make([]int16, FrameSamples)
		frame[0] = int16(i + 1)
		call.Played(frame)
	}

	line, err := call.Listen()
	s.Require().NoError(err)
	defer line.Close()

	s.Require().Equal(playedFrames, s.taken(line))
	s.Equal(int16(21), line.taking[0][0], "the oldest frame kept is not the first of the last second")
	s.Equal(int16(playedFrames+20), line.taking[playedFrames-1][0])
}

func (s *LineSuite) TestNoVoiceIsKeptForALineNobodyIsExpectedOn() {
	call, err := NewCall(Config{})
	s.Require().NoError(err)
	call.Expect(1)
	call.Expect(-1)
	call.Played(speech(chunk, 31))

	line, err := call.Listen()
	s.Require().NoError(err)
	defer line.Close()

	s.Zero(s.taken(line))
}

// The router checks the config once at start; a call builds no canceller until a line opens.
func (s *LineSuite) TestACallBuildsNoCancellerUntilALineOpens() {
	allocations := testing.AllocsPerRun(5, func() {
		if _, err := NewCall(Config{}); err != nil {
			panic(err)
		}
	})

	s.Less(allocations, 10.0)
}
