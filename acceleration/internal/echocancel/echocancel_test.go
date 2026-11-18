package echocancel

import (
	"math/rand/v2"
	"sync"
	"testing"
	"time"

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
	wg.Go(func() {
		for range 50 {
			line, err := s.call.Listen()
			if err != nil {
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
	for delay := echodelay.Min; delay <= echodelay.Max; delay += 50 * time.Millisecond {
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

	s.Equal(found, s.line.Close())
}
