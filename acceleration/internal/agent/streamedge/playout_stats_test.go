package streamedge

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
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

func (s *StreamEdgeSuite) TestCountingThePlayoutLeavesWhatIsSentAsItWas() {
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() { _ = talker.Close() })
	frames := s.sent(talker, 1)
	// A lead of silence takes the marked path, 210 ms leaves a part-frame for the flush, and
	// the drop throws away the rest of the first reply and what the encoder held of it.
	lead := silence(24_000, 30)
	lead.Samples = append(lead.Samples, speech(24_000, 210).Samples...)
	s.Require().NoError(talker.WriteMarked(lead, new(playoutRecord)))
	frames = append(frames, s.sent(talker, 3)...)
	talker.drop()
	s.Require().NoError(talker.Write(speech(opusSampleRate, 130)))
	frames = append(frames, s.sent(talker, 12)...)

	digest := sha256.New()
	for _, frame := range frames {
		_ = binary.Write(digest, binary.BigEndian, uint32(len(frame)))
		digest.Write(frame)
	}
	s.Len(frames, 16)
	// Probed on 3a172535f, before the queue was counted.
	s.Equal("71943fda5a73181989bcab5677322169206e42b0ff98adfdd7ee6b2470d10648", hex.EncodeToString(digest.Sum(nil)))
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
