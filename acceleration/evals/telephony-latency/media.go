package main

import (
	"context"
	"errors"
	"log/slog"
	"math/rand/v2"
	"net"
	"net/netip"
	"sync"
	"time"

	"github.com/pion/rtp"
)

// packetSamples is 20 ms of audio, the packetisation every carrier uses for G.711.
const packetSamples = 160

// packetInterval is how often a packet goes out.
const packetInterval = 20 * time.Millisecond

// payloadPCMU is G.711 μ-law's static RTP payload type.
const payloadPCMU = 0

// restPackets is the silence kept after a stamp before the next one starts, longer than the
// quiet the detector needs to tell one stamp's onset from the end of the last.
const restPackets = 5

// maxGapSamples caps how much lost audio is filled with silence: a second. A longer gap is
// a stall rather than loss, and filling it would only delay reading what comes after.
const maxGapSamples = sampleRate

// Media is the audio of one leg: RTP to and from the carrier on a socket of its own.
//
// It sends continuously, silence when there is nothing to say, because a carrier that sees
// no RTP for a while may decide the call is dead, and because a stamp sent after silence
// then meets the same jitter buffer a word in a conversation would.
type Media struct {
	conn *net.UDPConn
	log  *slog.Logger

	mu sync.Mutex
	// remote is where RTP goes. It starts as the address in the carrier's SDP and moves to
	// wherever the carrier's RTP actually comes from, which is what a NAT in between needs.
	remote netip.AddrPort
	// stamps waiting to be sent.
	stamps int
	// sent is every stamp sent on this leg, so our own stamp echoed back on the same leg is
	// not counted as the other side's.
	sent     map[uint32]bool
	detector Detector

	heard chan Heard
}

// NewMedia starts the audio of a leg on a socket already bound to a local port. It runs
// until the socket is closed or the context ends.
func NewMedia(ctx context.Context, conn *net.UDPConn, remote netip.AddrPort, log *slog.Logger) *Media {
	m := &Media{
		conn:   conn,
		log:    log,
		remote: remote,
		sent:   map[uint32]bool{},
		heard:  make(chan Heard, 64),
	}
	go m.send(ctx)
	go m.receive()
	return m
}

// Say queues a stamp. The millisecond in it is the moment its first packet goes out, so
// waiting in the queue is not counted against the network.
func (m *Media) Say() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stamps++
}

// Heard is the stamps that arrived on this leg from the other side.
func (m *Media) Heard() <-chan Heard { return m.heard }

// Remote is where the carrier's media is, as last seen.
func (m *Media) Remote() netip.AddrPort {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.remote
}

// Garbled is how many stamps started on this leg but did not decode.
func (m *Media) Garbled() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.detector.Garbled()
}

// Close stops the audio. The socket goes with it.
func (m *Media) Close() error { return m.conn.Close() }

func (m *Media) send(ctx context.Context) {
	packet := rtp.Packet{Header: rtp.Header{
		Version:        2,
		PayloadType:    payloadPCMU,
		SequenceNumber: uint16(rand.Uint32()),
		Timestamp:      rand.Uint32(),
		SSRC:           rand.Uint32(),
	}}
	var speaking []int16
	rest := 0
	started := time.Now()
	for n := 1; ; n++ {
		if speaking == nil && rest == 0 {
			m.mu.Lock()
			if m.stamps > 0 {
				m.stamps--
				millis := Millis(time.Now())
				m.sent[millis] = true
				speaking = Stamp(millis)
			}
			m.mu.Unlock()
		}

		frame := make([]int16, packetSamples)
		taken := copy(frame, speaking)
		switch {
		case taken > 0 && taken == len(speaking):
			speaking = nil
			rest = restPackets
		case taken > 0:
			speaking = speaking[taken:]
		case rest > 0:
			rest--
		}
		packet.Payload = EncodeULaw(frame)
		raw, err := packet.Marshal()
		if err != nil {
			m.log.Error("telephony-latency: rtp marshal", "err", err)
			return
		}
		if _, err := m.conn.WriteToUDPAddrPort(raw, m.Remote()); err != nil {
			if !errors.Is(err, net.ErrClosed) {
				m.log.Error("telephony-latency: rtp send", "err", err)
			}
			return
		}
		packet.SequenceNumber++
		packet.Timestamp += packetSamples

		select {
		case <-ctx.Done():
			return
		case <-time.After(time.Until(started.Add(time.Duration(n) * packetInterval))):
		}
	}
}

func (m *Media) receive() {
	defer close(m.heard)
	buffer := make([]byte, 1500)
	var next, ssrc uint32
	first := true
	for {
		n, from, err := m.conn.ReadFromUDPAddrPort(buffer)
		if err != nil {
			if !errors.Is(err, net.ErrClosed) {
				m.log.Error("telephony-latency: rtp receive", "err", err)
			}
			return
		}
		at := time.Now()

		var packet rtp.Packet
		if err := packet.Unmarshal(buffer[:n]); err != nil || packet.PayloadType != payloadPCMU {
			// Comfort noise and telephone events carry no audio to read.
			continue
		}

		samples := DecodeULaw(packet.Payload)
		// A new source or a jump of more than a second is a carrier re-anchoring the media,
		// e.g. on a bridge, and the timeline starts again from this packet.
		gap := int32(packet.Timestamp - next)
		if first || packet.SSRC != ssrc || gap > maxGapSamples || gap < -maxGapSamples {
			gap = 0
		}
		if gap < 0 {
			// A late packet whose audio was already filled with silence.
			continue
		}
		// Lost audio is filled with silence, so the tones after it stay where they belong;
		// the first sample of the filled packet was due as long before this one as the gap.
		samples = append(make([]int16, gap), samples...)
		at = at.Add(-time.Duration(gap) * time.Second / sampleRate)
		first = false
		ssrc = packet.SSRC
		next = packet.Timestamp + uint32(len(packet.Payload))

		m.mu.Lock()
		m.remote = netip.AddrPortFrom(from.Addr().Unmap(), from.Port())
		heard := m.detector.Push(samples, at)
		var fromOther []Heard
		for _, h := range heard {
			if !m.sent[h.Sent] {
				fromOther = append(fromOther, h)
			}
		}
		m.mu.Unlock()

		for _, h := range fromOther {
			select {
			case m.heard <- h:
			default:
			}
		}
	}
}
