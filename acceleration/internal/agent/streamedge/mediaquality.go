package streamedge

import (
	"log/slog"
	"sync"
	"time"

	audiortc "github.com/GetStream/getstream-go-webrtc/audio/rtc"
	"github.com/pion/interceptor"
	"github.com/pion/rtcp"
	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// The numbers here describe one leg of a call, the agent's own: between the router and the SFU.
// A caller's leg, between their device and the SFU, is not in them. pion's GetStats has no RTP
// stream stats for the agent's connections (the SDK keeps its stats interceptor's getter to
// itself), so the edge counts what it can see: the packets its decoder reads, the receiver
// reports the SFU sends about the agent's own audio, and the ICE pair's round trips.

const (
	// directionInbound is the participants' audio as it reached the agent from the SFU.
	directionInbound = "inbound"
	// directionOutbound is the agent's audio as it reached the SFU, in the SFU's own reports.
	directionOutbound = "outbound"
)

// lossBuckets bracket the 5% of packets lost past which Twilio Voice Insights flags a call
// (https://www.twilio.com/docs/voice/voice-insights/api/call/details-call-tags): finer below it,
// to tell a clean call from one near it, coarser above.
var lossBuckets = []float64{0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2}

// jitterBuckets bracket Twilio's 5 ms average and 30 ms peak jitter flags (same page).
var jitterBuckets = []float64{0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1}

// rttBuckets run past 300 ms, twice the 150 ms one-way delay ITU-T G.114 §1 holds most
// conversations unaffected by (https://www.itu.int/rec/T-REC-G.114-200305-I/en).
var rttBuckets = []float64{0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1}

// Every series is one value per voice session, so a bucket counts sessions. direction is a fixed
// pair; nothing names a call or a customer.
var (
	mediaLossRatio = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "agent_media_packet_loss_ratio",
		Help:      "Share of RTP audio packets lost on the leg between the agent and the SFU over a voice session, by direction: inbound counted by the agent per RFC 3550 A.3, outbound from the SFU's receiver reports",
		Buckets:   lossBuckets,
	}, []string{"direction"})
	mediaJitterSeconds = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "agent_media_jitter_seconds",
		Help:      "Mean RTP interarrival jitter (RFC 3550 A.8) of audio on the leg between the agent and the SFU over a voice session, by direction: inbound averaged over the packets the agent read, outbound over the SFU's receiver reports",
		Buckets:   jitterBuckets,
	}, []string{"direction"})
	mediaRTTSeconds = promauto.NewHistogram(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "agent_media_rtt_seconds",
		Help:      "Mean round trip between the agent and the SFU over a voice session, from the ICE consent checks on the agent's outgoing connection",
		Buckets:   rttBuckets,
	})
	mediaConcealedRatio = promauto.NewHistogram(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "agent_media_concealed_ratio",
		Help:      "Share of the participants' decoded audio the agent's Opus decoder made up to cover packets that never arrived, over a voice session",
		Buckets:   lossBuckets,
	})
)

// rtpMaxDropout is the jump ahead in sequence numbers past which a stream is taken to have
// restarted, not to have lost the packets between: MAX_DROPOUT in RFC 3550 Appendix A.1.
const rtpMaxDropout = 3000

// rtpSeqHalf splits a 16-bit sequence number difference into ahead (below it) and behind.
const rtpSeqHalf = 0x8000

// jitterGain is the 1/16 RFC 3550 A.8 smooths the interarrival jitter estimate with.
const jitterGain = 16

// inboundTotals are what one or more inbound streams counted.
type inboundTotals struct {
	// expected is how many packets the sequence numbers say were sent, received how many came,
	// lost the shortfall of a stream (RFC 3550 A.3). received is one stream's; add sums lost.
	expected, received, lost uint64
	// inOrder are the packets that moved the sequence on, the ones the decoder decodes; frames
	// are the buffers the decoder handed out. Each in-order packet is one buffer, so the rest
	// are buffers the decoder made up for packets that never came.
	inOrder, frames uint64
	// jitterSum is the jitter estimate in seconds after each packet, summed over jitterSamples.
	jitterSum     float64
	jitterSamples uint64
}

func (t *inboundTotals) add(other inboundTotals) {
	t.expected += other.expected
	t.lost += other.lost
	t.inOrder += other.inOrder
	t.frames += other.frames
	t.jitterSum += other.jitterSum
	t.jitterSamples += other.jitterSamples
}

// inboundStream is one participant's track as the decoder reads it, counted per RFC 3550
// Appendix A on the way. It is read on the decoding goroutine and totalled on another.
type inboundStream struct {
	source    audiortc.RTPSource
	clockRate float64
	now       func() time.Time

	mu      sync.Mutex
	totals  inboundTotals
	started bool
	// base and highest are the window's first and newest extended sequence numbers. A restart
	// opens a new window; the packets the old one expected are already in totals.
	base, highest uint64
	// lastArrival and lastTimestamp are the previous packet's, for its transit difference.
	lastArrival   time.Time
	lastTimestamp uint32
	jitter        float64
}

// ReadRTP reads the next packet and counts it, satisfying audiortc.RTPSource.
func (s *inboundStream) ReadRTP() (*rtp.Packet, interceptor.Attributes, error) {
	packet, attributes, err := s.source.ReadRTP()
	if err == nil {
		s.arrived(packet.SequenceNumber, packet.Timestamp, s.now())
	}
	return packet, attributes, err
}

func (s *inboundStream) arrived(seq uint16, timestamp uint32, at time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.totals.received++

	ahead := seq - uint16(s.highest)
	switch {
	case !s.started || (ahead >= rtpMaxDropout && ahead < rtpSeqHalf):
		s.restart(seq)
	case ahead == 0 || ahead >= rtpSeqHalf:
		// A duplicate or a packet older than the newest: counted already as lost or as come,
		// and the decoder drops it.
		s.sampleJitter(timestamp, at)
		return
	default:
		s.highest += uint64(ahead)
	}
	s.totals.inOrder++
	s.sampleJitter(timestamp, at)
}

// restart opens a new window at seq, keeping what the old one expected.
func (s *inboundStream) restart(seq uint16) {
	if s.started {
		s.totals.expected += s.highest - s.base + 1
		// The timestamps jumped with the sequence numbers, so the gap is not jitter.
		s.lastArrival = time.Time{}
	}
	s.started = true
	s.base, s.highest = uint64(seq), uint64(seq)
}

// sampleJitter updates the interarrival jitter (RFC 3550 A.8) from the transit difference to
// the previous packet, and adds it to the session's mean.
func (s *inboundStream) sampleJitter(timestamp uint32, at time.Time) {
	if !s.lastArrival.IsZero() {
		arrived := at.Sub(s.lastArrival).Seconds() * s.clockRate
		sent := float64(int32(timestamp - s.lastTimestamp))
		difference := arrived - sent
		if difference < 0 {
			difference = -difference
		}
		s.jitter += (difference - s.jitter) / jitterGain
		s.totals.jitterSum += s.jitter / s.clockRate
		s.totals.jitterSamples++
	}
	s.lastArrival, s.lastTimestamp = at, timestamp
}

// decoded counts a buffer the decoder handed out.
func (s *inboundStream) decoded() {
	s.mu.Lock()
	s.totals.frames++
	s.mu.Unlock()
}

func (s *inboundStream) snapshot() inboundTotals {
	s.mu.Lock()
	defer s.mu.Unlock()
	totals := s.totals
	if s.started {
		totals.expected += s.highest - s.base + 1
	}
	// Duplicates can make more packets come than were expected (RFC 3550 A.3). That cancels no
	// loss on another stream, so the shortfall is taken here, per stream, not from the sums.
	if totals.expected > totals.received {
		totals.lost = totals.expected - totals.received
	}
	return totals
}

// outboundWindow is the run of receiver reports about one SSRC whose cumulative counts go on
// from each other.
type outboundWindow struct {
	ssrc                      uint32
	firstHighest, lastHighest uint32
	firstLost, lastLost       int32
}

// mediaQuality gathers one voice session's media quality and observes it once, when it ends.
type mediaQuality struct {
	now func() time.Time

	mu    sync.Mutex
	live  map[*inboundStream]struct{}
	ended inboundTotals
	// window is the current run of the SFU's reports; opened says there is one. expected and
	// lost are what the windows before it counted.
	window         outboundWindow
	opened         bool
	expected, lost uint64
	// jitterSum is the reports' jitter in seconds, over jitterReports.
	jitterSum     float64
	jitterReports uint64
}

func newMediaQuality() *mediaQuality {
	return &mediaQuality{now: time.Now, live: map[*inboundStream]struct{}{}}
}

// inbound wraps a track's packets so they are counted as the decoder reads them. The stream
// counts towards the session once listening says so.
func (q *mediaQuality) inbound(source audiortc.RTPSource, clockRate uint32) *inboundStream {
	return &inboundStream{source: source, clockRate: float64(clockRate), now: q.now}
}

func (q *mediaQuality) listening(stream *inboundStream) {
	q.mu.Lock()
	q.live[stream] = struct{}{}
	q.mu.Unlock()
}

// stopped folds a stream that is no longer read into the session's totals, so a session holds
// one stream per track it is listening to, however many come and go.
func (q *mediaQuality) stopped(stream *inboundStream) {
	totals := stream.snapshot()
	q.mu.Lock()
	defer q.mu.Unlock()
	delete(q.live, stream)
	q.ended.add(totals)
}

// reported counts the report blocks about ssrc, the agent's own audio, in an RTCP packet from
// the SFU. The SFU sends the agent no media on this connection, so its reports are receiver
// reports (RFC 3550 §6.4.2).
func (q *mediaQuality) reported(packet rtcp.Packet, ssrc uint32) {
	report, ok := packet.(*rtcp.ReceiverReport)
	if !ok {
		return
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	for _, block := range report.Reports {
		if block.SSRC != ssrc {
			continue
		}
		q.jitterSum += float64(block.Jitter) / opusSampleRate
		q.jitterReports++
		// The cumulative lost count is a signed 24-bit number (RFC 3550 §6.4.1), negative when
		// duplicates outnumber the losses; pion parses it unsigned, so -1 reads as 0xFFFFFF.
		totalLost := int32(block.TotalLost<<8) >> 8
		// Another SSRC, or a sequence that went back, is a stream the counts do not go on from.
		if !q.opened || block.SSRC != q.window.ssrc || block.LastSequenceNumber < q.window.lastHighest {
			q.closeWindow()
			q.window = outboundWindow{ssrc: block.SSRC, firstHighest: block.LastSequenceNumber,
				firstLost: totalLost, lastHighest: block.LastSequenceNumber, lastLost: totalLost}
			q.opened = true
			continue
		}
		q.window.lastHighest, q.window.lastLost = block.LastSequenceNumber, totalLost
	}
}

// closeWindow adds the current window's counts to the session's. Before the first report the
// window is empty and counts nothing.
func (q *mediaQuality) closeWindow() {
	expected, lost := q.window.counts()
	q.expected += expected
	q.lost += lost
}

// counts are the packets the window's reports expected and lost. The first report's counts take
// in the packets before it, from a first sequence number the edge does not keep, so a window
// counts from its first report on.
func (w outboundWindow) counts() (expected, lost uint64) {
	expected = uint64(w.lastHighest - w.firstHighest)
	if w.lastLost > w.firstLost {
		// Never more than the window expected: a count can only grow so far in a window.
		lost = min(uint64(w.lastLost-w.firstLost), expected)
	}
	return expected, lost
}

// mediaSummary is one session's media quality; a field is set only when something measured it.
type mediaSummary struct {
	inboundLoss, inboundJitter, concealed *float64
	outboundLoss, outboundJitter, rtt     *float64
}

// summary totals the session. rtt is the ICE round trip, when there was one.
func (q *mediaQuality) summary(rtt *float64) mediaSummary {
	q.mu.Lock()
	defer q.mu.Unlock()
	totals := q.ended
	for stream := range q.live {
		totals.add(stream.snapshot())
	}
	summary := mediaSummary{rtt: rtt}
	if totals.expected > 0 {
		summary.inboundLoss = ratio(float64(totals.lost), float64(totals.expected))
	}
	if totals.jitterSamples > 0 {
		summary.inboundJitter = ratio(totals.jitterSum, float64(totals.jitterSamples))
	}
	if totals.frames > 0 {
		// A packet is counted in before its buffer is counted out, so a summary between the two
		// sees more packets than buffers.
		concealed := float64(totals.frames) - float64(totals.inOrder)
		summary.concealed = ratio(max(concealed, 0), float64(totals.frames))
	}
	expected, lost := q.expected, q.lost
	if q.opened {
		windowExpected, windowLost := q.window.counts()
		expected += windowExpected
		lost += windowLost
	}
	if expected > 0 {
		summary.outboundLoss = ratio(float64(lost), float64(expected))
	}
	if q.jitterReports > 0 {
		summary.outboundJitter = ratio(q.jitterSum, float64(q.jitterReports))
	}
	return summary
}

func ratio(part, whole float64) *float64 {
	value := part / whole
	return &value
}

// observe puts the session's summary in the distributions and the log, so one call's media can
// be looked up by its call id.
func (s mediaSummary) observe(logger *slog.Logger) {
	var fields []any
	observe := func(value *float64, histogram prometheus.Observer, field string) {
		if value == nil {
			return
		}
		histogram.Observe(*value)
		fields = append(fields, field, *value)
	}
	observe(s.inboundLoss, mediaLossRatio.WithLabelValues(directionInbound), "inbound_loss")
	observe(s.inboundJitter, mediaJitterSeconds.WithLabelValues(directionInbound), "inbound_jitter_s")
	observe(s.concealed, mediaConcealedRatio, "inbound_concealed")
	observe(s.outboundLoss, mediaLossRatio.WithLabelValues(directionOutbound), "outbound_loss")
	observe(s.outboundJitter, mediaJitterSeconds.WithLabelValues(directionOutbound), "outbound_jitter_s")
	observe(s.rtt, mediaRTTSeconds, "rtt_s")
	if len(fields) > 0 {
		logger.Info("media quality", fields...)
	}
}

// iceRoundTrip is the mean round trip of the ICE consent checks on pc's selected pair
// (totalRoundTripTime over responsesReceived, https://www.w3.org/TR/webrtc-stats/#candidatepair-dict*),
// read once: pion keeps the totals, so nothing is polled.
func iceRoundTrip(pc *webrtc.PeerConnection) *float64 {
	if pc == nil {
		return nil
	}
	// pion makes the DTLS and ICE transports with the connection, so neither is nil. No selected
	// pair reads as one with no responses, and so does a pair the controlled side selected on the
	// remote's nomination before any check of its own came back.
	pair, _ := pc.SCTP().Transport().ICETransport().GetSelectedCandidatePairStats()
	if pair.ResponsesReceived == 0 {
		return nil
	}
	return ratio(pair.TotalRoundTripTime, float64(pair.ResponsesReceived))
}
