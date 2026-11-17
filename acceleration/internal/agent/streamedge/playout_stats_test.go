package streamedge

import (
	"bytes"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
)

// sent takes n samples from the track and returns what each carried, silence included.
func (s *StreamEdgeSuite) sent(talker *speaker, n int) [][]byte {
	var frames [][]byte
	for range n {
		sample, err := talker.NextSample(s.ctx)
		s.Require().NoError(err)
		frames = append(frames, sample.Data)
	}
	return frames
}

// scenario plays one run through a speaker and returns what it sent. When counted, it reads
// the stats between the steps, as the agent does around an utterance.
func (s *StreamEdgeSuite) scenario(counted bool) [][]byte {
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() { _ = talker.Close() })
	read := func() {
		if counted {
			talker.stats()
		}
	}
	read()
	frames := s.sent(talker, 1)
	// A lead of silence takes the marked path, 210 ms leaves a part-frame for the flush, and
	// the drop throws away the rest of the first reply and what the encoder held of it.
	lead := silence(24_000, 30)
	lead.Samples = append(lead.Samples, speech(24_000, 210).Samples...)
	s.Require().NoError(talker.WriteMarked(lead, new(playoutRecord)))
	read()
	frames = append(frames, s.sent(talker, 1)...)
	read()
	frames = append(frames, s.sent(talker, 2)...)
	talker.drop()
	read()
	s.Require().NoError(talker.Write(speech(opusSampleRate, 130)))
	frames = append(frames, s.sent(talker, 6)...)
	read()
	frames = append(frames, s.sent(talker, 6)...)
	read()
	return frames
}

// The Opus bytes differ between platforms, so the control compares two speakers in one run
// rather than a digest probed elsewhere.
func (s *StreamEdgeSuite) TestCountingThePlayoutLeavesWhatIsSentAsItWas() {
	asked := s.scenario(true)
	never := s.scenario(false)

	s.Len(asked, 16)
	s.Equal(never, asked, "reading the stats changed the frames sent")
	silent := func(frames [][]byte) []bool {
		var out []bool
		for _, frame := range frames {
			out = append(out, bytes.Equal(frame, silenceFrame))
		}
		return out
	}
	s.Equal(silent(never), silent(asked))
	s.Contains(silent(asked), true, "the run has silence in it")
	s.Contains(silent(asked), false, "the run has speech in it")
}

func (s *StreamEdgeSuite) TestTheEdgeCountsItsQueue() {
	edge, err := New(Options{CallID: "demo", User: User{ID: "agent"}, APIKey: "key", APISecret: "secret"})
	s.Require().NoError(err)
	s.Require().NoError(edge.PublishAudio(speech(opusSampleRate, 100)))

	// The agent finds the counts by asking for them, so an edge that lost the method would
	// report nothing rather than fail to build.
	var counter agent.PlayoutCounter = edge

	s.EqualValues(5, counter.PlayoutStats().Queued)
}

func (s *StreamEdgeSuite) TestTheQueueCountsWhatWentInWhatWentOutAndWhatWasDropped() {
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() { _ = talker.Close() })
	s.Require().NoError(talker.Write(speech(opusSampleRate, 200)))
	s.sent(talker, 3)

	talker.drop()

	s.Equal(agent.PlayoutStats{Queued: 10, Sent: 3, Dropped: 7, Peak: 200 * time.Millisecond}, talker.stats())
}

func (s *StreamEdgeSuite) TestSpeechStillQueuedIsWaiting() {
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() { _ = talker.Close() })
	s.Require().NoError(talker.Write(speech(opusSampleRate, 200)))
	s.sent(talker, 6)

	stats := talker.stats()

	s.Equal(80*time.Millisecond, stats.Waiting)
	s.Equal(200*time.Millisecond, stats.Peak)
	s.EqualValues(6, stats.Sent, "the four frames still waiting have not gone")
}

func (s *StreamEdgeSuite) TestChunksQueuedBeforeAnyIsTakenStackInThePeak() {
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() { _ = talker.Close() })
	s.Require().NoError(talker.Write(speech(opusSampleRate, 100)))
	s.Require().NoError(talker.Write(speech(opusSampleRate, 100)))

	s.Equal(200*time.Millisecond, talker.stats().Peak, "the queue held both chunks, not the larger one")
}

func (s *StreamEdgeSuite) TestReadingTheStatsStartsTheNextPeakFromWhatIsQueued() {
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() { _ = talker.Close() })
	s.Require().NoError(talker.Write(speech(opusSampleRate, 200)))
	s.sent(talker, 6)
	talker.stats()

	s.Equal(80*time.Millisecond, talker.stats().Peak, "the four frames left are the deepest since")
}

func (s *StreamEdgeSuite) TestTheTailFlushedOutOfTheEncoderCountsTowardsThePeak() {
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() { _ = talker.Close() })
	// Less than a frame: the encoder holds all of it until the track finds the queue empty.
	s.Require().NoError(talker.Write(speech(opusSampleRate, 10)))
	s.Require().Zero(talker.stats().Peak)

	s.sent(talker, 1)

	stats := talker.stats()
	s.Equal(opusFrameDuration, stats.Peak)
	s.EqualValues(1, stats.Sent)
}

func (s *StreamEdgeSuite) TestSpeechQueuedWhenTheCallIsLeftIsCountedAsDropped() {
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.Require().NoError(talker.Write(speech(opusSampleRate, 100)))

	s.Require().NoError(talker.Close())

	stats := talker.stats()
	s.EqualValues(5, stats.Dropped)
	s.Zero(stats.Sent)
}
