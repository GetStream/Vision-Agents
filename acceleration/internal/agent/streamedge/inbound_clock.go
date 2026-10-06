package streamedge

import (
	"strings"
	"sync/atomic"
	"time"

	"github.com/GetStream/getstream-go-webrtc/audio/rtc"
	"github.com/pion/interceptor"
	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
)

var inboundEpochCounter atomic.Uint64

// observedRTPSource records when each raw packet returned from the SDK. It
// passes the packet and interceptor attributes through unchanged so the audio
// reader remains responsible for RED, decoding, and packet loss concealment.
type observedRTPSource struct {
	source       rtc.RTPSource
	clock        *inboundClock
	afterObserve func(time.Time)
}

func (s observedRTPSource) ReadRTP() (*rtp.Packet, interceptor.Attributes, error) {
	pkt, attributes, err := s.source.ReadRTP()
	if err == nil {
		receivedAt := time.Now()
		s.clock.observePacket(pkt, receivedAt)
		if s.afterObserve != nil {
			s.afterObserve(receivedAt)
		}
	}
	return pkt, attributes, err
}

type inboundPacketTiming struct {
	sequence   uint16
	ssrc       uint32
	id         uint64
	epoch      uint64
	clockEpoch uint64
	red        bool
	loss       bool
	reset      bool
	duplicate  bool
	receivedAt time.Time
}

// inboundClock tracks raw RTP sequence and clock continuity alongside the
// decoder's output timestamps. One reader goroutine owns it for the life of a
// track; timingForPCM is called before the corresponding frame can block in
// the inbound emitter.
type inboundClock struct {
	epoch      uint64
	clockEpoch uint64
	isRED      bool
	cycle      time.Duration
	haveRaw    bool
	ssrc       uint32
	expected   uint16
	timestamp  uint32

	packetID uint64
	current  inboundPacketTiming

	sequenceLoss  uint64
	clockResets   uint64
	ambiguousGaps uint64

	// An anomaly on a raw packet that has not yielded decoded PCM yet makes the
	// next PTS gap ambiguous. The packet-local flags remain set while the
	// reader drains PLC or RED frames made from that same RTP read.
	ambiguousSinceFrame bool

	haveDecoded bool
	previousEnd time.Duration
	previousPTS time.Duration
	ptsOffset   time.Duration
	previous    inboundPacketTiming

	timestampGap     time.Duration
	timestampOnlyGap time.Duration
	overlap          time.Duration
}

func newInboundClock(codec webrtc.RTPCodecCapability) *inboundClock {
	clockRate := codec.ClockRate
	if clockRate == 0 {
		clockRate = 48000
	}
	return &inboundClock{
		epoch: inboundEpochCounter.Add(1),
		isRED: strings.EqualFold(codec.MimeType, "audio/red"),
		cycle: time.Duration(uint64(1)<<32) * time.Second / time.Duration(clockRate),
	}
}

func (c *inboundClock) observePacket(pkt *rtp.Packet, receivedAt time.Time) {
	c.packetID++
	packet := inboundPacketTiming{
		sequence:   pkt.SequenceNumber,
		ssrc:       pkt.SSRC,
		id:         c.packetID,
		receivedAt: receivedAt,
		red:        c.isRED,
	}

	if !c.haveRaw {
		c.haveRaw = true
		c.ssrc = pkt.SSRC
		c.expected = pkt.SequenceNumber + 1
		c.timestamp = pkt.Timestamp
	} else if pkt.SSRC != c.ssrc {
		// SSRC changes start a new source clock. Sequence numbers from the old
		// source cannot be compared to this one.
		c.resetClock()
		packet.reset = true
		c.ssrc = pkt.SSRC
		c.expected = pkt.SequenceNumber + 1
		c.timestamp = pkt.Timestamp
	} else {
		sequenceGap := pkt.SequenceNumber - c.expected
		if sequenceGap > 0x8000 {
			// A late or duplicate packet is ignored by TrackReader too. Keep the
			// accepted sequence/timestamp baseline, but make a later gap
			// conservative.
			packet.duplicate = true
			c.ambiguousSinceFrame = true
		} else {
			if sequenceGap != 0 {
				c.sequenceLoss += uint64(sequenceGap)
				packet.loss = true
				c.ambiguousSinceFrame = true
			}

			timestampDelta := pkt.Timestamp - c.timestamp
			if timestampDelta > 1<<31 {
				// A large backward jump is a clock reset. Small modular deltas
				// across uint32 wrap remain ordinary forward progress.
				c.resetClock()
				packet.reset = true
				c.ambiguousSinceFrame = true
			}
			c.expected = pkt.SequenceNumber + 1
			c.timestamp = pkt.Timestamp
		}
	}

	packet.epoch = c.epoch
	packet.clockEpoch = c.clockEpoch
	c.current = packet
}

func (c *inboundClock) resetClock() {
	c.clockResets++
	c.clockEpoch++
}

func (c *inboundClock) timingForPCM(pts, duration time.Duration) agent.AudioTiming {
	packet := c.current
	effectivePTS := pts + c.ptsOffset
	sameEpoch := c.haveDecoded && packet.clockEpoch == c.previous.clockEpoch
	if sameEpoch && pts < c.previousPTS && c.previousPTS-pts > c.cycle/2 {
		// TrackReader's PTS is a duration from the first RTP timestamp using
		// uint32 modular subtraction. Extend that duration when it completes a
		// full RTP timestamp cycle, without changing the public decoder PTS.
		c.ptsOffset += c.cycle
		effectivePTS = pts + c.ptsOffset
	} else if c.haveDecoded && !sameEpoch {
		// PTS values from separate SSRC/timestamp epochs are incomparable. The
		// decoded PTS itself remains untouched in metadata; only gap accounting
		// rebases at the first frame of this epoch.
		c.ptsOffset = 0
		effectivePTS = pts
	}
	if sameEpoch {
		gap := effectivePTS - c.previousEnd
		switch {
		case gap > 0:
			c.timestampGap += gap
			if c.timestampOnlyGapIsConfirmed(packet) {
				c.timestampOnlyGap += gap
			} else {
				c.ambiguousGaps++
			}
		case gap < 0:
			c.overlap -= gap
		}
	}

	c.haveDecoded = true
	c.previousEnd = effectivePTS + duration
	c.previousPTS = pts
	c.previous = packet
	c.ambiguousSinceFrame = false
	return agent.AudioTiming{
		Valid:            true,
		PTS:              pts,
		ReceivedAt:       packet.receivedAt,
		Epoch:            packet.epoch,
		TimestampGap:     c.timestampGap,
		TimestampOnlyGap: c.timestampOnlyGap,
		SequenceLoss:     c.sequenceLoss,
		ClockResets:      c.clockResets,
		AmbiguousGaps:    c.ambiguousGaps,
		Overlap:          c.overlap,
	}
}

func (c *inboundClock) timestampOnlyGapIsConfirmed(packet inboundPacketTiming) bool {
	return !c.ambiguousSinceFrame &&
		!packet.red && !packet.loss && !packet.reset && !packet.duplicate &&
		!c.previous.red &&
		packet.id != c.previous.id &&
		packet.ssrc == c.previous.ssrc &&
		packet.clockEpoch == c.previous.clockEpoch &&
		packet.sequence == c.previous.sequence+1
}
