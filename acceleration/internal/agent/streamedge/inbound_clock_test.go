package streamedge

import (
	"bytes"
	"io"
	"math"
	"sync"
	"testing"
	"time"

	sdkaudio "github.com/GetStream/getstream-go-webrtc/audio"
	"github.com/GetStream/getstream-go-webrtc/audio/opus"
	audiortc "github.com/GetStream/getstream-go-webrtc/audio/rtc"
	"github.com/pion/interceptor"
	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/emit"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const (
	testRTPFrameTicks = 960
	testRTPSSRC       = 0x10203040
)

type packetSource struct {
	packets []*rtp.Packet
	index   int
}

func (s *packetSource) ReadRTP() (*rtp.Packet, interceptor.Attributes, error) {
	if s.index >= len(s.packets) {
		return nil, nil, io.EOF
	}
	pkt := s.packets[s.index]
	s.index++
	return pkt, nil, nil
}

func encodedTestPackets(t *testing.T, count int) []*rtp.Packet {
	t.Helper()
	encoder, err := opus.NewEncoder(opus.Config{SampleRate: 48000, Channels: 1})
	require.NoError(t, err)
	samples := make([]float32, count*testRTPFrameTicks)
	for i := range samples {
		if i >= testRTPFrameTicks {
			samples[i] = float32(0.1 * math.Sin(2*math.Pi*440*float64(i)/48000))
		}
	}
	payloads, err := encoder.Encode(sdkaudio.FromFloat32(samples, 48000, 1))
	require.NoError(t, err)
	require.Len(t, payloads, count)

	packets := make([]*rtp.Packet, count)
	for i, payload := range payloads {
		packets[i] = &rtp.Packet{
			Header: rtp.Header{
				Version:        2,
				PayloadType:    111,
				SequenceNumber: uint16(100 + i),
				Timestamp:      uint32(50000 + i*testRTPFrameTicks),
				SSRC:           testRTPSSRC,
			},
			Payload: payload,
		}
	}
	return packets
}

func testOpusCodec() webrtc.RTPCodecCapability {
	return webrtc.RTPCodecCapability{
		MimeType:  webrtc.MimeTypeOpus,
		ClockRate: 48000,
		Channels:  1,
	}
}

func testReader(t *testing.T, source audiortc.RTPSource, codec webrtc.RTPCodecCapability) *audiortc.TrackReader {
	t.Helper()
	reader, err := audiortc.NewRTPReader(source, codec,
		audiortc.ReaderConfig{Opus: opus.Config{SampleRate: 16000}})
	require.NoError(t, err)
	return reader
}

func cloneRTPPackets(packets []*rtp.Packet) []*rtp.Packet {
	clones := make([]*rtp.Packet, len(packets))
	for i, packet := range packets {
		clone := *packet
		clone.Payload = bytes.Clone(packet.Payload)
		clones[i] = &clone
	}
	return clones
}

func decodeFrames(t *testing.T, reader *audiortc.TrackReader) []sdkaudio.PCM {
	t.Helper()
	defer reader.Close()
	var frames []sdkaudio.PCM
	for pcm, err := range reader.Frames() {
		require.NoError(t, err)
		frames = append(frames, pcm)
	}
	return frames
}

func TestInboundClockTracksTimestampGapsAcrossRTPWraps(t *testing.T) {
	clock := newInboundClock(testOpusCodec())
	firstAt := time.Now()
	clock.observePacket(&rtp.Packet{Header: rtp.Header{
		SequenceNumber: 65535,
		Timestamp:      0xfffffc40,
		SSRC:           testRTPSSRC,
	}}, firstAt)
	first := clock.timingForPCM(0, 20*time.Millisecond)
	require.True(t, first.Valid, "zero is a valid PTS")
	require.Zero(t, first.PTS)
	require.Equal(t, firstAt, first.ReceivedAt)

	// Both raw clocks wrap. The timestamp advances 30 ms while the next
	// decoded buffer is 20 ms, so the unexplained decoded PTS gap is 10 ms.
	clock.observePacket(&rtp.Packet{Header: rtp.Header{
		SequenceNumber: 0,
		Timestamp:      0x1e0,
		SSRC:           testRTPSSRC,
	}}, firstAt.Add(time.Millisecond))
	second := clock.timingForPCM(30*time.Millisecond, 20*time.Millisecond)

	require.Equal(t, 10*time.Millisecond, second.TimestampGap)
	require.Equal(t, 10*time.Millisecond, second.TimestampOnlyGap)
	require.Zero(t, second.SequenceLoss)
	require.Zero(t, second.ClockResets)
	require.Zero(t, second.AmbiguousGaps)
	require.Equal(t, first.Epoch, second.Epoch)
}

func TestInboundClockMarksREDGapsAmbiguous(t *testing.T) {
	codec := testOpusCodec()
	codec.MimeType = "audio/red"
	clock := newInboundClock(codec)
	at := time.Now()
	clock.observePacket(&rtp.Packet{Header: rtp.Header{SequenceNumber: 1, Timestamp: 0, SSRC: testRTPSSRC}}, at)
	clock.timingForPCM(0, 20*time.Millisecond)
	clock.observePacket(&rtp.Packet{Header: rtp.Header{SequenceNumber: 2, Timestamp: testRTPFrameTicks, SSRC: testRTPSSRC}}, at.Add(time.Millisecond))
	got := clock.timingForPCM(40*time.Millisecond, 20*time.Millisecond)

	require.Equal(t, 20*time.Millisecond, got.TimestampGap)
	require.Zero(t, got.TimestampOnlyGap)
	require.Equal(t, uint64(1), got.AmbiguousGaps)
}

func TestInboundClockReportsResetsAndOverlap(t *testing.T) {
	clock := newInboundClock(testOpusCodec())
	at := time.Now()
	clock.observePacket(&rtp.Packet{Header: rtp.Header{SequenceNumber: 7, Timestamp: 10000, SSRC: testRTPSSRC}}, at)
	first := clock.timingForPCM(0, 20*time.Millisecond)

	clock.observePacket(&rtp.Packet{Header: rtp.Header{SequenceNumber: 8, Timestamp: 100, SSRC: testRTPSSRC}}, at.Add(time.Millisecond))
	reset := clock.timingForPCM(40*time.Millisecond, 20*time.Millisecond)
	require.Equal(t, uint64(1), reset.ClockResets)
	require.Equal(t, first.Epoch, reset.Epoch, "Epoch identifies the track, while reset is counted separately")
	require.Zero(t, reset.TimestampGap, "PTS from different clock epochs are incomparable")
	require.Zero(t, reset.TimestampOnlyGap)
	require.Zero(t, reset.AmbiguousGaps)

	clock.observePacket(&rtp.Packet{Header: rtp.Header{SequenceNumber: 1, Timestamp: 500, SSRC: testRTPSSRC + 1}}, at.Add(2*time.Millisecond))
	swapped := clock.timingForPCM(50*time.Millisecond, 20*time.Millisecond)
	require.Equal(t, uint64(2), swapped.ClockResets)
	require.Equal(t, reset.Epoch, swapped.Epoch)
	require.Equal(t, reset.SequenceLoss, swapped.SequenceLoss, "a new SSRC starts a new sequence space")
	require.Zero(t, swapped.Overlap, "PTS from different clock epochs are incomparable")

	clock.observePacket(&rtp.Packet{Header: rtp.Header{SequenceNumber: 2, Timestamp: 500 + testRTPFrameTicks, SSRC: testRTPSSRC + 1}}, at.Add(3*time.Millisecond))
	withinEpoch := clock.timingForPCM(60*time.Millisecond, 20*time.Millisecond)
	require.Equal(t, 10*time.Millisecond, withinEpoch.Overlap)
	require.Equal(t, swapped.Epoch, withinEpoch.Epoch)
}

func TestInboundClockExtendsDecodedPTSAtFullRTPClockRollover(t *testing.T) {
	clock := newInboundClock(testOpusCodec())
	at := time.Now()
	cycle := time.Duration(uint64(1)<<32) * time.Second / 48000
	clock.observePacket(&rtp.Packet{Header: rtp.Header{SequenceNumber: 1, Timestamp: 100, SSRC: testRTPSSRC}}, at)
	first := clock.timingForPCM(cycle-20*time.Millisecond, 20*time.Millisecond)
	clock.observePacket(&rtp.Packet{Header: rtp.Header{SequenceNumber: 2, Timestamp: 100 + testRTPFrameTicks, SSRC: testRTPSSRC}}, at.Add(time.Millisecond))
	wrapped := clock.timingForPCM(0, 20*time.Millisecond)

	require.True(t, wrapped.Valid)
	require.Zero(t, wrapped.PTS, "the decoder's zero PTS remains visible")
	require.Equal(t, first.Epoch, wrapped.Epoch)
	require.Zero(t, wrapped.TimestampGap)
	require.Zero(t, wrapped.Overlap)
	require.Zero(t, wrapped.ClockResets)
}

func TestInboundClockEpochsAreUniqueAcrossTracksAndResets(t *testing.T) {
	first := newInboundClock(testOpusCodec())
	second := newInboundClock(testOpusCodec())
	require.NotEqual(t, first.epoch, second.epoch)

	at := time.Now()
	first.observePacket(&rtp.Packet{Header: rtp.Header{SequenceNumber: 1, Timestamp: 10000, SSRC: testRTPSSRC}}, at)
	initial := first.timingForPCM(0, 20*time.Millisecond)
	first.observePacket(&rtp.Packet{Header: rtp.Header{SequenceNumber: 2, Timestamp: 1, SSRC: testRTPSSRC}}, at.Add(time.Millisecond))
	reset := first.timingForPCM(0, 20*time.Millisecond)

	require.Equal(t, initial.Epoch, reset.Epoch)
	require.NotEqual(t, second.epoch, reset.Epoch)
	require.Equal(t, uint64(1), reset.ClockResets)
}

func TestInboundClockReaderPreservesPTSAndAccountsForPLCOnce(t *testing.T) {
	packets := encodedTestPackets(t, 2)
	packets[0].SequenceNumber = 10
	packets[0].Timestamp = 0
	packets[1].SequenceNumber = 12
	packets[1].Timestamp = 2 * testRTPFrameTicks

	clock := newInboundClock(testOpusCodec())
	reader := testReader(t, observedRTPSource{source: &packetSource{packets: packets}, clock: clock}, testOpusCodec())
	defer reader.Close()

	var timings []agent.AudioTiming
	for pcm, err := range reader.Frames() {
		require.NoError(t, err)
		timings = append(timings, clock.timingForPCM(pcm.PTS, pcm.Duration()))
	}
	require.Len(t, timings, 3, "one received packet yields a PLC frame and the decoded frame")
	require.True(t, timings[0].Valid)
	require.Zero(t, timings[0].PTS)
	require.Equal(t, 20*time.Millisecond, timings[1].PTS)
	require.Equal(t, 40*time.Millisecond, timings[2].PTS)
	require.Equal(t, uint64(0), timings[0].SequenceLoss)
	require.Equal(t, uint64(1), timings[1].SequenceLoss)
	require.Equal(t, uint64(1), timings[2].SequenceLoss)
	require.Zero(t, timings[2].TimestampGap, "the PLC frame fills the packet-loss interval")
	require.Equal(t, timings[1].ReceivedAt, timings[2].ReceivedAt, "PLC and decoded frames share the raw packet arrival")
}

func TestObservedRTPSourceDoesNotChangeDecodedAudio(t *testing.T) {
	packets := encodedTestPackets(t, 2)
	packets[0].SequenceNumber = 10
	packets[0].Timestamp = 0
	packets[1].SequenceNumber = 12
	packets[1].Timestamp = 2 * testRTPFrameTicks
	codec := testOpusCodec()

	plain := decodeFrames(t, testReader(t, &packetSource{packets: cloneRTPPackets(packets)}, codec))
	clock := newInboundClock(codec)
	reader := testReader(t, observedRTPSource{
		source: &packetSource{packets: cloneRTPPackets(packets)},
		clock:  clock,
	}, codec)
	edge := &Edge{inbound: emit.New[agent.InboundAudio](audioBuffer)}
	var done sync.WaitGroup
	done.Add(1)
	go func() {
		defer done.Done()
		edge.hear(reader, clock, stt.Participant{ID: "pcm-parity"}, make(chan struct{}))
	}()
	done.Wait()
	edge.inbound.Close()
	var observed []agent.InboundAudio
	for frame := range edge.Audio() {
		observed = append(observed, frame)
	}

	require.Len(t, plain, 3, "the fixture contains silence followed by one PLC frame")
	require.Len(t, observed, len(plain))
	for i := range plain {
		require.Equal(t, plain[i].PTS, observed[i].Timing.PTS)
		require.Equal(t, plain[i].Duration(), time.Duration(len(observed[i].Audio.Samples))*time.Second/time.Duration(observed[i].Audio.SampleRate))
		require.Equal(t, plain[i].SampleRate, observed[i].Audio.SampleRate)
		require.Equal(t, plain[i].Channels, observed[i].Audio.Channels)
		require.Equal(t, plain[i].ToInt16().Int16(), observed[i].Audio.Samples, "frame %d samples are unchanged", i)
	}
	for _, sample := range plain[0].ToInt16().Int16() {
		require.Zero(t, sample, "the first fixture packet is encoded from silence")
	}
}

func TestInboundClockTreatsLatePacketAsAmbiguityWithoutReset(t *testing.T) {
	packets := encodedTestPackets(t, 3)
	packets[0].SequenceNumber = 10
	packets[0].Timestamp = 1000
	packets[1].SequenceNumber = 9
	packets[1].Timestamp = 0
	packets[2].SequenceNumber = 11
	packets[2].Timestamp = 1000 + 30*48

	clock := newInboundClock(testOpusCodec())
	reader := testReader(t, observedRTPSource{source: &packetSource{packets: packets}, clock: clock}, testOpusCodec())
	defer reader.Close()

	var timings []agent.AudioTiming
	for pcm, err := range reader.Frames() {
		require.NoError(t, err)
		timings = append(timings, clock.timingForPCM(pcm.PTS, pcm.Duration()))
	}
	require.Len(t, timings, 2, "the SDK reader ignores the late raw packet")
	require.Equal(t, time.Duration(0), timings[0].TimestampGap)
	require.Equal(t, 10*time.Millisecond, timings[1].TimestampGap)
	require.Zero(t, timings[1].TimestampOnlyGap, "the late packet makes the interval ambiguous")
	require.Equal(t, uint64(1), timings[1].AmbiguousGaps)
	require.Zero(t, timings[1].ClockResets, "a late old timestamp must not reset the accepted clock")
	require.Zero(t, timings[1].SequenceLoss)
	require.Equal(t, timings[0].Epoch, timings[1].Epoch)
}

func TestInboundClockLeavesLongLossGapAmbiguous(t *testing.T) {
	packets := encodedTestPackets(t, 2)
	packets[0].SequenceNumber = 1
	packets[0].Timestamp = 0
	packets[1].SequenceNumber = 40
	packets[1].Timestamp = 39 * testRTPFrameTicks

	clock := newInboundClock(testOpusCodec())
	reader := testReader(t, observedRTPSource{source: &packetSource{packets: packets}, clock: clock}, testOpusCodec())
	defer reader.Close()

	var last agent.AudioTiming
	var firstGap agent.AudioTiming
	for pcm, err := range reader.Frames() {
		require.NoError(t, err)
		last = clock.timingForPCM(pcm.PTS, pcm.Duration())
		if firstGap.Valid == false && last.TimestampGap > 0 {
			firstGap = last
		}
	}
	require.Equal(t, uint64(38), last.SequenceLoss)
	require.Equal(t, 260*time.Millisecond, firstGap.TimestampGap)
	require.Zero(t, firstGap.TimestampOnlyGap)
	require.Equal(t, uint64(1), firstGap.AmbiguousGaps)
	require.Equal(t, 260*time.Millisecond, last.TimestampGap)
}

func TestInboundClockBackpressureDoesNotMoveRawArrivalTime(t *testing.T) {
	packets := encodedTestPackets(t, audioBuffer+2)
	source := &packetSource{packets: packets}
	clock := newInboundClock(testOpusCodec())
	observedAt := make(chan time.Time, len(packets))
	reader := testReader(t, observedRTPSource{
		source:       source,
		clock:        clock,
		afterObserve: func(at time.Time) { observedAt <- at },
	}, testOpusCodec())
	edge := &Edge{inbound: emit.New[agent.InboundAudio](audioBuffer)}
	var done sync.WaitGroup
	done.Add(1)
	go func() {
		defer done.Done()
		edge.hear(reader, clock, sttParticipantForTimingTest(), make(chan struct{}))
	}()

	var eleventhArrival time.Time
	for i := 0; i < audioBuffer+1; i++ {
		select {
		case eleventhArrival = <-observedAt:
		case <-time.After(2 * time.Second):
			t.Fatal("reader did not pull the packet that meets inbound backpressure")
		}
	}
	deadline := time.Now().Add(2 * time.Second)
	for len(edge.inbound.Events()) < audioBuffer && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	require.Equal(t, audioBuffer, len(edge.inbound.Events()))
	time.Sleep(2 * time.Millisecond)
	unblockedAt := time.Now()
	<-edge.inbound.Events()
	for i := 0; i < audioBuffer-1; i++ {
		<-edge.inbound.Events()
	}
	eleventh := <-edge.inbound.Events()
	require.True(t, eleventh.Timing.Valid)
	require.Equal(t, eleventhArrival, eleventh.Timing.ReceivedAt)
	require.True(t, eleventh.Timing.ReceivedAt.Before(unblockedAt), "raw arrival is stamped before a blocked emitter send")

	edge.inbound.Close()
	done.Wait()
}

func sttParticipantForTimingTest() stt.Participant { return stt.Participant{ID: "timing-test"} }
