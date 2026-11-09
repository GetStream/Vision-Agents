package audiofloor

import (
	"encoding/binary"
	"fmt"
	"sync"
	"time"

	"github.com/thesyncim/gopus"
	"github.com/thesyncim/gopus/container/ogg"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

const (
	// frame is how much audio one Opus packet carries, and frameSamples its samples.
	frame        = 20 * time.Millisecond
	frameSamples = SampleRate * int(frame/time.Millisecond) / 1000
	// bitrate is the Opus encoder's target. At 16 kHz Opus codes speech with SILK, and
	// 32 kb/s is where it reaches its full 8 kHz of bandwidth.
	bitrate = 32000
	// granulePerFrame is a frame in Ogg Opus granule positions, which count at 48 kHz.
	granulePerFrame = 48000 * int(frame/time.Millisecond) / 1000
)

// Format is how a stream sends its audio.
type Format string

const (
	// Opus encodes each frame once, as it arrives, and sends the frames as Ogg Opus.
	Opus Format = "opus"
	// PCM sends 16-bit PCM, 8 to 20 times the bytes, for telling the codec apart from the
	// model when debugging a decision.
	PCM Format = "pcm"
)

const pcmType = "audio/pcm;rate=16000;channels=1;format=s16le"

// Stream is one participant's audio, kept as it arrives in 20 ms frames, each encoded once.
// Last is what the server is sent: the frames of the last moments, without encoding any of
// them again. Frames are whole, so the end of the audio is up to a frame behind what was
// written, and with Opus also by the encoder's lookahead, 6.5 ms.
type Stream struct {
	mu      sync.Mutex
	encoder *gopus.Encoder // nil sends PCM
	packet  []byte
	// pending is the start of the next frame.
	pending [frameSamples]int16
	filled  int
	// frames is a ring of the last frames, next the slot the next one goes in.
	frames [][]byte
	next   int
	heard  int64
}

func newStream(format Format, keep time.Duration) (*Stream, error) {
	s := &Stream{frames: make([][]byte, keep/frame)}
	if format == PCM {
		return s, nil
	}
	encoder, err := gopus.NewEncoder(gopus.EncoderConfig{SampleRate: SampleRate, Channels: 1, Application: gopus.ApplicationVoIP})
	if err != nil {
		return nil, stack.Wrap(err)
	}
	if err := encoder.SetBitrate(bitrate); err != nil {
		return nil, stack.Wrap(err)
	}
	s.encoder, s.packet = encoder, make([]byte, 1500)
	return s, nil
}

// Write adds samples, 16 kHz mono, in pieces of any length.
func (s *Stream) Write(samples []int16) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for len(samples) > 0 {
		n := copy(s.pending[s.filled:], samples)
		samples, s.filled = samples[n:], s.filled+n
		if s.filled < frameSamples {
			return nil
		}
		s.filled = 0
		slot := s.frames[s.next][:0]
		if s.encoder == nil {
			for _, sample := range s.pending {
				slot = binary.LittleEndian.AppendUint16(slot, uint16(sample))
			}
		} else {
			size, err := s.encoder.EncodeInt16(s.pending[:], s.packet)
			if err != nil {
				return stack.Wrap(fmt.Errorf("audiofloor: encode: %w", err))
			}
			slot = append(slot, s.packet[:size]...)
		}
		s.frames[s.next] = slot
		s.next = (s.next + 1) % len(s.frames)
		s.heard++
	}
	return nil
}

// Frames is how many frames the stream has been written.
func (s *Stream) Frames() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.heard
}

// Last is the frames of the last d, or as many as the stream keeps, as the server takes them.
func (s *Stream) Last(d time.Duration) harness.Audio {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := int(min(int64(d/frame), s.heard, int64(len(s.frames))))
	if n <= 0 {
		return harness.Audio{}
	}
	frames := make([][]byte, 0, n)
	for i := range n {
		frames = append(frames, s.frames[(s.next-n+i+len(s.frames))%len(s.frames)])
	}
	audio := harness.Audio{Type: pcmType, Duration: time.Duration(n) * frame}
	if s.encoder == nil {
		audio.Data = make([]byte, 0, n*2*frameSamples)
		for _, f := range frames {
			audio.Data = append(audio.Data, f...)
		}
		return audio
	}
	audio.Type, audio.Data = "audio/ogg", oggOpus(frames, uint16(3*s.encoder.Lookahead()))
	return audio
}

// oggOpus is packets, 20 ms each, as an Ogg Opus stream of their own (RFC 7845): OpusHead
// and OpusTags, then as many packets to a page as its lacing values hold, with granule
// positions counted from the first. A decoder that starts on a packet from the middle of an
// encoder's output is a few milliseconds from converging, and drops preSkip samples at
// 48 kHz, the encoder's lookahead, so that the audio starts with the first frame's.
func oggOpus(packets [][]byte, preSkip uint16) []byte {
	head := ogg.DefaultOpusHead(SampleRate, 1)
	head.PreSkip = preSkip
	pages := []ogg.Page{
		{HeaderType: ogg.PageFlagBOS, Payload: head.Encode()},
		{Payload: ogg.DefaultOpusTags().Encode()},
	}
	for i := range pages {
		pages[i].Segments = ogg.BuildSegmentTable(len(pages[i].Payload))
	}
	// The pages' lacing values and payloads are slices of one of each, the page going on
	// from first and at.
	size := 0
	for _, packet := range packets {
		size += len(packet)
	}
	lacing, payload := make([]byte, 0, size/255+len(packets)), make([]byte, 0, size)
	first, at := 0, 0
	for i, packet := range packets {
		if len(lacing)-first+len(packet)/255+1 > 255 {
			pages = append(pages, ogg.Page{Segments: lacing[first:], Payload: payload[at:], GranulePos: uint64(i * granulePerFrame)})
			first, at = len(lacing), len(payload)
		}
		for range len(packet) / 255 {
			lacing = append(lacing, 255)
		}
		lacing, payload = append(lacing, byte(len(packet)%255)), append(payload, packet...)
	}
	pages = append(pages, ogg.Page{HeaderType: ogg.PageFlagEOS, Segments: lacing[first:], Payload: payload[at:],
		GranulePos: uint64(len(packets) * granulePerFrame)})
	out := make([]byte, 0, len(lacing)+len(payload)+len(pages)*27+64)
	for i := range pages {
		// One stream to a body, so its serial number need tell it from no other.
		pages[i].SerialNumber, pages[i].PageSequence = 1, uint32(i)
		out = append(out, pages[i].Encode()...)
	}
	return out
}
