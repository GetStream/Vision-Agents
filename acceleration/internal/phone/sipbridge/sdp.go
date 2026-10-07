package sipbridge

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"strconv"
	"strings"

	"github.com/pion/sdp/v3"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

const (
	DirSendRecv = "sendrecv"
	DirInactive = "inactive"
)

// The placeholder is the address in flow B's first offer to Stream, before the customer
// has answered. The default is a documentation address (RFC 5737) that routes nowhere.
const (
	defaultPlaceholderAddr = "192.0.2.1"
	defaultPlaceholderPort = 20000
)

// rejectAnswer's address is unrelated to the flow B placeholder: the stream is rejected, so
// nothing is ever sent to it.
const rejectAddr = "192.0.2.1"

// Codec is one RTP payload format.
type Codec struct {
	Name        string
	PayloadType uint8
	ClockRate   uint32
}

// staticCodecs are the audio codecs both Stream and this package accept, keyed by name.
// G722's clock rate is 8000 in SDP for historical reasons (RFC 3551).
var staticCodecs = map[string]Codec{
	"PCMU": {Name: "PCMU", PayloadType: 0, ClockRate: 8000},
	"PCMA": {Name: "PCMA", PayloadType: 8, ClockRate: 8000},
	"G722": {Name: "G722", PayloadType: 9, ClockRate: 8000},
}

var dtmfCodec = Codec{Name: "telephone-event", PayloadType: 101, ClockRate: 8000}

func codecsFromNames(names []string) ([]Codec, error) {
	if len(names) == 0 {
		return nil, stack.Wrap(fmt.Errorf("at least one codec is required"))
	}
	codecs := make([]Codec, 0, len(names))
	for _, name := range names {
		codec, ok := staticCodecs[name]
		if !ok {
			return nil, stack.Wrap(fmt.Errorf("unsupported codec %q, use PCMU, PCMA or G722", name))
		}
		codecs = append(codecs, codec)
	}
	return codecs, nil
}

// CheckCodecs reports whether every name is a codec Dial can offer, so a caller can refuse a
// trunk when it is saved rather than when it is first dialled.
func CheckCodecs(names []string) error {
	_, err := codecsFromNames(names)
	return err
}

func staticCodecByPayloadType(pt uint8) (Codec, bool) {
	for _, codec := range staticCodecs {
		if codec.PayloadType == pt {
			return codec, true
		}
	}
	return Codec{}, false
}

// Media is the first audio stream of an SDP body: where to send RTP and in which formats.
type Media struct {
	Addr      string
	Port      int
	Codecs    []Codec
	Direction string
}

func (m Media) audioCodec() (Codec, bool) {
	for _, codec := range m.Codecs {
		if codec.Name != dtmfCodec.Name {
			return codec, true
		}
	}
	return Codec{}, false
}

func (m Media) hasDTMF() bool {
	for _, codec := range m.Codecs {
		if codec.Name == dtmfCodec.Name {
			return true
		}
	}
	return false
}

// ParseMedia reads the first m= line of body.
func ParseMedia(body []byte) (Media, error) {
	var sd sdp.SessionDescription
	if err := sd.Unmarshal(body); err != nil {
		return Media{}, stack.Wrap(fmt.Errorf("parse sdp: %w", err))
	}
	if len(sd.MediaDescriptions) == 0 {
		return Media{}, stack.Wrap(errors.New("sdp has no media"))
	}
	md := sd.MediaDescriptions[0]

	m := Media{Port: md.MediaName.Port.Value, Direction: DirSendRecv}
	switch {
	case md.ConnectionInformation != nil && md.ConnectionInformation.Address != nil:
		m.Addr = md.ConnectionInformation.Address.Address
	case sd.ConnectionInformation != nil && sd.ConnectionInformation.Address != nil:
		m.Addr = sd.ConnectionInformation.Address.Address
	default:
		return Media{}, stack.Wrap(errors.New("sdp has no connection address"))
	}

	for _, format := range md.MediaName.Formats {
		pt, err := strconv.ParseUint(format, 10, 8)
		if err != nil {
			return Media{}, stack.Wrap(fmt.Errorf("sdp payload type %q: %w", format, err))
		}
		if codec, err := sd.GetCodecForPayloadType(uint8(pt)); err == nil {
			name := codec.Name
			// Normalize codec names: telephone-event variants -> "telephone-event", others -> uppercase
			if strings.EqualFold(name, "telephone-event") {
				name = dtmfCodec.Name
			} else {
				name = strings.ToUpper(name)
			}
			m.Codecs = append(m.Codecs, Codec{Name: name, PayloadType: uint8(pt), ClockRate: codec.ClockRate})
			continue
		}
		if codec, ok := staticCodecByPayloadType(uint8(pt)); ok {
			m.Codecs = append(m.Codecs, codec)
		}
	}

	for _, attr := range md.Attributes {
		switch attr.Key {
		case "sendrecv", "sendonly", "recvonly", "inactive":
			m.Direction = attr.Key
		}
	}
	return m, nil
}

func buildSDP(m Media, sessionID, version uint64) []byte {
	pts := make([]string, 0, len(m.Codecs))
	for _, codec := range m.Codecs {
		pts = append(pts, strconv.Itoa(int(codec.PayloadType)))
	}

	var b strings.Builder
	b.WriteString("v=0\r\n")
	fmt.Fprintf(&b, "o=- %d %d IN IP4 %s\r\n", sessionID, version, m.Addr)
	b.WriteString("s=sipbridge\r\n")
	fmt.Fprintf(&b, "c=IN IP4 %s\r\n", m.Addr)
	b.WriteString("t=0 0\r\n")
	fmt.Fprintf(&b, "m=audio %d RTP/AVP %s\r\n", m.Port, strings.Join(pts, " "))
	for _, codec := range m.Codecs {
		fmt.Fprintf(&b, "a=rtpmap:%d %s/%d\r\n", codec.PayloadType, codec.Name, codec.ClockRate)
		if codec.Name == dtmfCodec.Name {
			fmt.Fprintf(&b, "a=fmtp:%d 0-16\r\n", codec.PayloadType)
		}
	}
	b.WriteString("a=ptime:20\r\n")
	fmt.Fprintf(&b, "a=%s\r\n", m.Direction)
	return []byte(b.String())
}

// sdpSession is our side of one SDP session. RFC 3264 wants the o= session id to stay the
// same across re-INVITEs and the version to go up on every new offer.
type sdpSession struct {
	id      uint64
	version uint64
}

func newSDPSession() *sdpSession {
	return &sdpSession{id: rand.Uint64N(1 << 62)}
}

func (s *sdpSession) build(m Media) []byte {
	s.version++
	return buildSDP(m, s.id, s.version)
}

// placeholderOffer opens a session in Stream before we know where the customer's RTP is.
func placeholderOffer(addr string, port int, codecs []Codec) Media {
	all := append(append([]Codec{}, codecs...), dtmfCodec)
	return Media{Addr: addr, Port: port, Codecs: all, Direction: DirSendRecv}
}

// holdOffer puts Stream on hold while the phone rings.
func holdOffer(offer Media) Media {
	offer.Direction = DirInactive
	return offer
}

// customerOffer is what the customer is offered: Stream's RTP address and only the codec Stream
// picked.
func customerOffer(stream Media) (Media, error) {
	audio, ok := stream.audioCodec()
	if !ok {
		return Media{}, stack.Wrap(errors.New("stream answered with no audio codec"))
	}
	codecs := []Codec{audio}
	if stream.hasDTMF() {
		codecs = append(codecs, dtmfCodec)
	}
	return Media{Addr: stream.Addr, Port: stream.Port, Codecs: codecs, Direction: DirSendRecv}, nil
}

// retargetOffer points Stream's RTP at the customer, keeping the codecs Stream already uses.
func retargetOffer(customer, offered Media) (Media, error) {
	if customer.Port == 0 {
		return Media{}, stack.Wrap(errors.New("customer rejected the audio stream"))
	}
	want, ok := offered.audioCodec()
	if !ok {
		return Media{}, stack.Wrap(errors.New("offer to the customer had no audio codec"))
	}
	got, ok := customer.audioCodec()
	if !ok || got.Name != want.Name {
		return Media{}, stack.Wrap(fmt.Errorf("customer answered with %v, but only %s was offered", customer.Codecs, want.Name))
	}
	return Media{Addr: customer.Addr, Port: customer.Port, Codecs: offered.Codecs, Direction: DirSendRecv}, nil
}

// rejectAnswer answers an offer we cannot use. Port 0 rejects the stream (RFC 3264 §6); the
// m= line still has to name at least one format to be valid.
func rejectAnswer(offer Media) Media {
	codecs := offer.Codecs
	if len(codecs) == 0 {
		codecs = []Codec{staticCodecs["PCMU"]}
	}
	return Media{Addr: rejectAddr, Port: 0, Codecs: codecs, Direction: DirInactive}
}
