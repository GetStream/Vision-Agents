package streamedge

import (
	"bytes"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/GetStream/getstream-go-webrtc/audio/opus"
	audiortc "github.com/GetStream/getstream-go-webrtc/audio/rtc"
	"github.com/pion/interceptor"
	"github.com/pion/rtcp"
	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// The SSRCs a test's reports are about: the agent's own audio, zero, which is a valid SSRC (RFC 3550
// §8 picks them at random) and also what a struct starts with, and another stream.
const (
	agentSSRC = 0
	otherSSRC = 2
)

type MediaQualitySuite struct {
	suite.Suite
	// frame is the payload of every packet a test sends.
	frame []byte
}

func TestMediaQualitySuite(t *testing.T) {
	suite.Run(t, new(MediaQualitySuite))
}

func (s *MediaQualitySuite) SetupSuite() {
	frame, err := silentOpusFrame()
	s.Require().NoError(err)
	s.frame = frame
}

// silentOpusFrame is one Opus packet of silence, which the SDK's decoder decodes like any other.
func silentOpusFrame() ([]byte, error) {
	encoder, err := opus.NewEncoder(opus.Config{SampleRate: opusSampleRate, FrameDuration: opusFrameDuration})
	if err != nil {
		return nil, err
	}
	return encoder.EncodeSilence()
}

// recorded is a track's packets in the order they arrive, then its end.
type recorded struct{ packets []*rtp.Packet }

func (r *recorded) ReadRTP() (*rtp.Packet, interceptor.Attributes, error) {
	if len(r.packets) == 0 {
		return nil, nil, io.EOF
	}
	packet := r.packets[0]
	r.packets = r.packets[1:]
	return packet, nil, nil
}

// packets are the Opus packets numbered seqs, each stamped as sent 20 ms after the one before it.
func (s *MediaQualitySuite) packets(seqs ...uint16) []*rtp.Packet {
	var packets []*rtp.Packet
	for _, seq := range seqs {
		packets = append(packets, &rtp.Packet{Header: rtp.Header{Version: 2, PayloadType: 111,
			SequenceNumber: seq, Timestamp: uint32(seq) * 960, SSRC: agentSSRC}, Payload: s.frame})
	}
	return packets
}

// run is the sequence numbers from first to last.
func run(first, last uint16) []uint16 {
	var seqs []uint16
	for seq := first; ; seq++ {
		seqs = append(seqs, seq)
		if seq == last {
			return seqs
		}
	}
}

// onTime is each packet arriving exactly as long after the first as it was sent after it.
func onTime(packets []*rtp.Packet) []time.Time {
	start := time.Date(2026, 11, 10, 12, 0, 0, 0, time.UTC)
	var arrivals []time.Time
	for _, packet := range packets {
		arrivals = append(arrivals, start.Add(time.Duration(packet.Timestamp)*time.Second/opusSampleRate))
	}
	return arrivals
}

// heard reads the packets through the edge's counting and the SDK's decoder, each packet arriving
// at its time in arrivals, and returns the session's summary.
func (s *MediaQualitySuite) heard(packets []*rtp.Packet, arrivals []time.Time) mediaSummary {
	quality := newMediaQuality()
	next := 0
	quality.now = func() time.Time {
		next++
		return arrivals[next-1]
	}
	stream := quality.inbound(&recorded{packets: packets}, opusSampleRate)
	quality.listening(stream)
	reader, err := audiortc.NewRTPReader(stream,
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: opusSampleRate, Channels: 2},
		audiortc.ReaderConfig{Opus: opus.Config{SampleRate: stt.SampleRate}})
	s.Require().NoError(err)
	for _, err := range reader.Frames() {
		s.Require().NoError(err)
		stream.decoded()
	}
	quality.stopped(stream)
	return quality.summary(nil)
}

func (s *MediaQualitySuite) value(measured *float64) float64 {
	s.Require().NotNil(measured, "the session measured it")
	return *measured
}

func (s *MediaQualitySuite) TestPacketsMissingFromTheSequenceAreLost() {
	// Catches loss counted from what came alone, or expected counted one short or long.
	packets := s.packets(append(run(0, 9), run(20, 99)...)...)
	summary := s.heard(packets, onTime(packets))
	s.InDelta(0.1, s.value(summary.inboundLoss), 1e-9, "ten of a hundred never came")
}

func (s *MediaQualitySuite) TestALatePacketIsNeitherLostNorDecoded() {
	// Catches a packet that came after a newer one counted as lost, or as one the decoder decoded:
	// the decoder made up audio for it when the newer one came, and drops it.
	packets := s.packets(0, 1, 2, 4, 3, 5, 6, 7, 8, 9)
	summary := s.heard(packets, onTime(packets))
	s.InDelta(0, s.value(summary.inboundLoss), 1e-9, "every packet came")
	s.InDelta(0.1, s.value(summary.concealed), 1e-9, "one of ten buffers was made up for the late one")
}

func (s *MediaQualitySuite) TestADuplicateIsNoNegativeLossAndIsNotDecoded() {
	// Catches the RFC 3550 count going below zero when more packets come than were sent, and a
	// duplicate, which the decoder drops, counted as decoded, hiding a buffer made up elsewhere.
	packets := s.packets(append(run(0, 5), append(run(5, 9), run(12, 19)...)...)...)
	summary := s.heard(packets, onTime(packets))
	s.InDelta(0.05, s.value(summary.inboundLoss), 1e-9, "two of 20 never came, less the duplicate (RFC 3550 A.3)")
	s.InDelta(0.1, s.value(summary.concealed), 1e-9, "two of twenty buffers were made up")

	packets = s.packets(0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9)
	s.InDelta(0, s.value(s.heard(packets, onTime(packets)).inboundLoss), 1e-9)
}

func (s *MediaQualitySuite) TestAPacketReadBeforeItsBufferIsTakenIsNoNegativeConcealment() {
	// Catches a summary taken mid-session, between the decoder reading a packet and handing out
	// its buffer, reading as less than nothing made up.
	quality := newMediaQuality()
	stream := quality.inbound(&recorded{packets: s.packets(0, 1)}, opusSampleRate)
	quality.listening(stream)
	for range 2 {
		_, _, err := stream.ReadRTP()
		s.Require().NoError(err)
	}
	stream.decoded()
	s.InDelta(0, s.value(quality.summary(nil).concealed), 1e-9)
}

func (s *MediaQualitySuite) TestSequenceNumbersWrapAroundWithTheirLossKept() {
	// Catches the count of a sequence that wraps at 65535 starting again, or counted backwards.
	packets := s.packets(65533, 65534, 65535, 1, 2)
	summary := s.heard(packets, onTime(packets))
	s.InDelta(1.0/6, s.value(summary.inboundLoss), 1e-9, "0 never came, between 65535 and 1")
}

func (s *MediaQualitySuite) TestAJumpOfMaxDropoutIsARestartNotLoss() {
	// Catches a stream that started again far ahead (RFC 3550 A.1's MAX_DROPOUT, 3000) counted
	// as losing every packet in between.
	packets := s.packets(append(run(0, 9), run(3009, 3018)...)...)
	summary := s.heard(packets, onTime(packets))
	s.InDelta(0, s.value(summary.inboundLoss), 1e-9)
}

func (s *MediaQualitySuite) TestAJumpJustUnderMaxDropoutIsLoss() {
	// Catches the restart taken a packet early: 2999 ahead is a gap of 2998 lost.
	packets := s.packets(append(run(0, 9), run(3008, 3017)...)...)
	summary := s.heard(packets, onTime(packets))
	s.InDelta(2998.0/3018, s.value(summary.inboundLoss), 1e-9)
}

func (s *MediaQualitySuite) TestAGapTheDecoderFillsIsConcealed() {
	// Catches concealment counted as zero, or from the buffers the decoder decoded.
	packets := s.packets(append(run(0, 9), run(13, 19)...)...)
	summary := s.heard(packets, onTime(packets))
	s.InDelta(3.0/20, s.value(summary.concealed), 1e-9, "three of twenty buffers were made up")
}

func (s *MediaQualitySuite) TestConcealmentIsWhatTheDecoderMadeUpNotWhatWasLost() {
	// Catches concealment taken from the gap: the SDK's decoder makes up at most 500 ms, 25
	// buffers, of a longer one.
	packets := s.packets(append(run(0, 9), run(110, 119)...)...)
	summary := s.heard(packets, onTime(packets))
	s.InDelta(25.0/45, s.value(summary.concealed), 1e-9)
	s.InDelta(100.0/120, s.value(summary.inboundLoss), 1e-9)
}

func (s *MediaQualitySuite) TestPacketsArrivingAsTheyWereSentHaveNoJitter() {
	// Catches jitter taken from the arrival gaps alone, which are 20 ms here, not from how they
	// differ from the send gaps.
	packets := s.packets(run(0, 199)...)
	summary := s.heard(packets, onTime(packets))
	s.InDelta(0, s.value(summary.inboundJitter), 1e-12)
}

func (s *MediaQualitySuite) TestUnevenArrivalsAreJitter() {
	// Catches the RFC 3550 A.8 estimate taken without the absolute value, with another gain, or in
	// RTP units rather than seconds. Packets 20 ms apart arrive 10 and 30 ms apart in turn, so
	// every transit difference is 10 ms and the estimate after packet k is 10 ms (1 - (15/16)^k);
	// its mean over k = 1..199 is 10 ms (1 - 15 (1 - (15/16)^199) / 199).
	packets := s.packets(run(0, 199)...)
	arrivals := onTime(packets)
	for i := range arrivals {
		if i%2 == 1 {
			arrivals[i] = arrivals[i].Add(-10 * time.Millisecond)
		}
	}
	summary := s.heard(packets, arrivals)
	s.InDelta(0.010*(1-15.0/199), s.value(summary.inboundJitter), 1e-6)
}

func (s *MediaQualitySuite) TestARestartIsNoJitter() {
	// Catches the timestamps' jump at a restart taken as a packet arriving that late.
	packets := s.packets(append(run(0, 9), run(5000, 5009)...)...)
	arrivals := onTime(packets[:10])
	last := arrivals[9]
	for i := range 10 {
		arrivals = append(arrivals, last.Add(time.Duration(i+1)*opusFrameDuration))
	}
	summary := s.heard(packets, arrivals)
	s.InDelta(0, s.value(summary.inboundJitter), 1e-12)
}

func (s *MediaQualitySuite) TestAStreamThatStoppedStillCounts() {
	// Catches the totals of a track that ended, folded into the session, getting lost.
	packets := s.packets(append(run(0, 9), run(20, 29)...)...)
	quality := newMediaQuality()
	stream := quality.inbound(&recorded{packets: packets}, opusSampleRate)
	quality.listening(stream)
	for range packets {
		_, _, err := stream.ReadRTP()
		s.Require().NoError(err)
	}
	quality.stopped(stream)
	s.InDelta(10.0/30, s.value(quality.summary(nil).inboundLoss), 1e-9, "10 to 19 never came")
}

// report is a receiver report from the SFU with one block.
func report(ssrc, highest, lost, jitter uint32) *rtcp.ReceiverReport {
	return &rtcp.ReceiverReport{SSRC: 9, Reports: []rtcp.ReceptionReport{
		{SSRC: ssrc, LastSequenceNumber: highest, TotalLost: lost, Jitter: jitter}}}
}

func (s *MediaQualitySuite) TestTheSFUsReportsGiveTheOutboundLossAndJitter() {
	// Catches loss counted from the start of the stream, which the first report already covers,
	// or jitter left in 48 kHz RTP units.
	quality := newMediaQuality()
	quality.reported(report(agentSSRC, 100, 4, 480), agentSSRC)
	quality.reported(report(agentSSRC, 300, 24, 1440), agentSSRC)
	summary := quality.summary(nil)
	s.InDelta(0.1, s.value(summary.outboundLoss), 1e-9, "20 lost of the 200 sent between the reports")
	s.InDelta(0.020, s.value(summary.outboundJitter), 1e-9, "the mean of 10 and 30 ms")
}

func (s *MediaQualitySuite) TestOneReportGivesTheJitterButNoLoss() {
	// Catches a loss made up from a single report, which says nothing of what was sent since.
	quality := newMediaQuality()
	quality.reported(report(agentSSRC, 100, 4, 480), agentSSRC)
	summary := quality.summary(nil)
	s.Nil(summary.outboundLoss)
	s.InDelta(0.010, s.value(summary.outboundJitter), 1e-9)
}

func (s *MediaQualitySuite) TestReportsOnAnotherStreamAreLeftOut() {
	// Catches a block about another SSRC counted as the agent's audio.
	quality := newMediaQuality()
	quality.reported(report(agentSSRC, 100, 0, 480), agentSSRC)
	quality.reported(report(otherSSRC, 5000, 2000, 48000), agentSSRC)
	quality.reported(report(agentSSRC, 200, 10, 480), agentSSRC)
	summary := quality.summary(nil)
	s.InDelta(0.1, s.value(summary.outboundLoss), 1e-9)
	s.InDelta(0.010, s.value(summary.outboundJitter), 1e-9)
}

func (s *MediaQualitySuite) TestAnotherSSRCStartsANewRunOfReports() {
	// Catches the counts of a republished track, under a new SSRC, taken as going on from the old
	// one's.
	quality := newMediaQuality()
	quality.reported(report(agentSSRC, 100, 0, 0), agentSSRC)
	quality.reported(report(agentSSRC, 200, 10, 0), agentSSRC)
	quality.reported(report(otherSSRC, 1000, 0, 0), otherSSRC)
	quality.reported(report(otherSSRC, 1100, 5, 0), otherSSRC)
	s.InDelta(15.0/200, s.value(quality.summary(nil).outboundLoss), 1e-9)
}

func (s *MediaQualitySuite) TestASequenceThatGoesBackStartsANewRunOfReports() {
	// Catches the counts of a stream the SFU started counting again taken as going on.
	quality := newMediaQuality()
	quality.reported(report(agentSSRC, 1000, 30, 0), agentSSRC)
	quality.reported(report(agentSSRC, 1100, 40, 0), agentSSRC)
	quality.reported(report(agentSSRC, 50, 0, 0), agentSSRC)
	quality.reported(report(agentSSRC, 150, 5, 0), agentSSRC)
	s.InDelta(15.0/200, s.value(quality.summary(nil).outboundLoss), 1e-9)
}

func (s *MediaQualitySuite) TestAReportWithTheSameHighestGoesOnFromTheSameRun() {
	// Catches a report on no new packets, whose lost count fell as a late packet came, starting a
	// new run: the run's loss is the change in the cumulative count from its first report.
	quality := newMediaQuality()
	quality.reported(report(agentSSRC, 100, 10, 0), agentSSRC)
	quality.reported(report(agentSSRC, 100, 9, 0), agentSSRC)
	quality.reported(report(agentSSRC, 200, 15, 0), agentSSRC)
	s.InDelta(0.05, s.value(quality.summary(nil).outboundLoss), 1e-9)
}

func (s *MediaQualitySuite) TestALostCountThatFallsIsNoNegativeLoss() {
	// Catches a cumulative lost count that went down, as duplicates make it (RFC 3550 §6.4.1),
	// wrapping round to a huge loss.
	quality := newMediaQuality()
	quality.reported(report(agentSSRC, 100, 10, 0), agentSSRC)
	quality.reported(report(agentSSRC, 200, 8, 0), agentSSRC)
	s.InDelta(0, s.value(quality.summary(nil).outboundLoss), 1e-9)
}

func (s *MediaQualitySuite) TestANegativeCumulativeLossIsNoLoss() {
	// Catches the signed 24-bit lost count (RFC 3550 §6.4.1) read unsigned: -1 arrives from pion
	// as 0xFFFFFF and counted as sixteen million lost packets.
	quality := newMediaQuality()
	quality.reported(report(agentSSRC, 100, 0, 0), agentSSRC)
	quality.reported(report(agentSSRC, 300, 0xFFFFFF, 0), agentSSRC)
	s.InDelta(0, s.value(quality.summary(nil).outboundLoss), 1e-9)
}

func (s *MediaQualitySuite) TestACountOfLossBeyondThePacketsExpectedIsAllLost() {
	// Catches a loss ratio above 1 when the lost count grows by more than the sequence did.
	quality := newMediaQuality()
	quality.reported(report(agentSSRC, 100, 0, 0), agentSSRC)
	quality.reported(report(agentSSRC, 110, 50, 0), agentSSRC)
	s.InDelta(1, s.value(quality.summary(nil).outboundLoss), 1e-9)
}

func (s *MediaQualitySuite) TestRTCPOtherThanAReceiverReportIsLeftOut() {
	// Catches a sender report's or a NACK's fields read as the SFU's report on the agent's audio.
	quality := newMediaQuality()
	quality.reported(&rtcp.SenderReport{SSRC: 9, Reports: []rtcp.ReceptionReport{
		{SSRC: agentSSRC, LastSequenceNumber: 100, Jitter: 480}}}, agentSSRC)
	quality.reported(&rtcp.TransportLayerNack{MediaSSRC: agentSSRC}, agentSSRC)
	summary := quality.summary(nil)
	s.Nil(summary.outboundJitter)
	s.Nil(summary.outboundLoss)
}

func (s *MediaQualitySuite) TestASessionWithNoMediaObservesNothing() {
	// Catches an empty session observed as perfect media, or as NaN, or logged as a line with
	// nothing in it.
	before := mediaSeries()
	var logged bytes.Buffer
	newMediaQuality().summary(nil).observe(slog.New(slog.NewTextHandler(&logged, nil)))
	s.Equal(before, mediaSeries())
	s.Empty(logged.String())
}

func (s *MediaQualitySuite) TestASessionIsObservedOnceWithItsValues() {
	// Catches a value put in another direction's or another number's distribution.
	inboundLoss, inboundJitter, concealed := 0.01, 0.002, 0.03
	outboundLoss, outboundJitter, rtt := 0.04, 0.005, 0.06
	summary := mediaSummary{inboundLoss: &inboundLoss, inboundJitter: &inboundJitter, concealed: &concealed,
		outboundLoss: &outboundLoss, outboundJitter: &outboundJitter, rtt: &rtt}
	before := mediaSeries()
	var logged bytes.Buffer
	summary.observe(slog.New(slog.NewTextHandler(&logged, nil)))
	after := mediaSeries()
	s.Contains(logged.String(), "msg=\"media quality\" inbound_loss=0.01 inbound_jitter_s=0.002 "+
		"inbound_concealed=0.03 outbound_loss=0.04 outbound_jitter_s=0.005 rtt_s=0.06",
		"one line a call can be looked up by")
	for _, observed := range []struct {
		series string
		value  float64
	}{
		{"router_agent_media_packet_loss_ratio/inbound", inboundLoss},
		{"router_agent_media_jitter_seconds/inbound", inboundJitter},
		{"router_agent_media_concealed_ratio/", concealed},
		{"router_agent_media_packet_loss_ratio/outbound", outboundLoss},
		{"router_agent_media_jitter_seconds/outbound", outboundJitter},
		{"router_agent_media_rtt_seconds/", rtt},
	} {
		s.Equal(before[observed.series].count+1, after[observed.series].count, observed.series)
		s.InDelta(observed.value, after[observed.series].sum-before[observed.series].sum, 1e-12, observed.series)
	}
}

func (s *MediaQualitySuite) TestNoConnectionHasNoRoundTrip() {
	// Catches a connection that never came up read as a zero round trip.
	s.Nil(iceRoundTrip(nil))
	pc, err := webrtc.NewPeerConnection(webrtc.Configuration{})
	s.Require().NoError(err)
	defer pc.Close()
	s.Nil(iceRoundTrip(pc))
}

// histogramSeries is one series' sample count and sum.
type histogramSeries struct {
	count uint64
	sum   float64
}

// mediaSeries are the router_agent_media_* histograms in the registry /metrics serves, by name
// and direction ("name/direction", the direction empty for those without one).
func mediaSeries() map[string]histogramSeries {
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		panic(err)
	}
	series := map[string]histogramSeries{}
	for _, family := range families {
		switch family.GetName() {
		case "router_agent_media_packet_loss_ratio", "router_agent_media_jitter_seconds",
			"router_agent_media_concealed_ratio", "router_agent_media_rtt_seconds":
		default:
			continue
		}
		for _, metric := range family.GetMetric() {
			direction := ""
			for _, pair := range metric.GetLabel() {
				if pair.GetName() == "direction" {
					direction = pair.GetValue()
				}
			}
			series[family.GetName()+"/"+direction] = histogramSeries{
				count: metric.GetHistogram().GetSampleCount(), sum: metric.GetHistogram().GetSampleSum()}
		}
	}
	return series
}
