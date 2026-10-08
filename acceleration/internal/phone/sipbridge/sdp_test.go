package sipbridge

import (
	"testing"

	"github.com/stretchr/testify/require"
)

var (
	pcmu = staticCodecs["PCMU"]
	pcma = staticCodecs["PCMA"]
)

func TestBuildSDPWritesTheWholeBody(t *testing.T) {
	m := Media{Addr: "192.0.2.1", Port: 20000, Codecs: []Codec{pcmu, pcma, dtmfCodec}, Direction: DirSendRecv}

	got := string(buildSDP(m, 42, 1))

	want := "v=0\r\n" +
		"o=- 42 1 IN IP4 192.0.2.1\r\n" +
		"s=sipbridge\r\n" +
		"c=IN IP4 192.0.2.1\r\n" +
		"t=0 0\r\n" +
		"m=audio 20000 RTP/AVP 0 8 101\r\n" +
		"a=rtpmap:0 PCMU/8000\r\n" +
		"a=rtpmap:8 PCMA/8000\r\n" +
		"a=rtpmap:101 telephone-event/8000\r\n" +
		"a=fmtp:101 0-16\r\n" +
		"a=ptime:20\r\n" +
		"a=sendrecv\r\n"
	require.Equal(t, want, got)
}

func TestParseMediaReadsWhatBuildSDPWrote(t *testing.T) {
	m := Media{Addr: "203.0.113.10", Port: 20000, Codecs: []Codec{pcma, dtmfCodec}, Direction: DirInactive}

	got, err := ParseMedia(buildSDP(m, 7, 3))

	require.NoError(t, err)
	require.Equal(t, m, got)
}

func TestParseMediaPrefersTheMediaLevelAddress(t *testing.T) {
	body := "v=0\r\no=- 1 1 IN IP4 10.0.0.1\r\ns=-\r\nc=IN IP4 10.0.0.1\r\nt=0 0\r\n" +
		"m=audio 30000 RTP/AVP 0\r\nc=IN IP4 198.51.100.7\r\n"

	got, err := ParseMedia([]byte(body))

	require.NoError(t, err)
	require.Equal(t, Media{Addr: "198.51.100.7", Port: 30000, Codecs: []Codec{pcmu}, Direction: DirSendRecv}, got)
}

func TestParseMediaKnowsStaticCodecsWithoutRtpmap(t *testing.T) {
	body := "v=0\r\no=- 1 1 IN IP4 198.51.100.7\r\ns=-\r\nc=IN IP4 198.51.100.7\r\nt=0 0\r\n" +
		"m=audio 30000 RTP/AVP 9 8\r\n"

	got, err := ParseMedia([]byte(body))

	require.NoError(t, err)
	require.Equal(t, []Codec{staticCodecs["G722"], pcma}, got.Codecs)
}

func TestParseMediaRefusesABodyWithoutAudio(t *testing.T) {
	_, err := ParseMedia([]byte("v=0\r\no=- 1 1 IN IP4 203.0.113.9\r\ns=-\r\nt=0 0\r\n"))

	require.ErrorContains(t, err, "no media")
}

func TestSDPSessionKeepsItsIDAndBumpsTheVersion(t *testing.T) {
	s := &sdpSession{id: 99}
	m := Media{Addr: "192.0.2.1", Port: 20000, Codecs: []Codec{pcmu}, Direction: DirSendRecv}

	first, second := string(s.build(m)), string(s.build(m))

	require.Contains(t, first, "o=- 99 1 IN IP4 192.0.2.1\r\n")
	require.Contains(t, second, "o=- 99 2 IN IP4 192.0.2.1\r\n")
}

func TestPlaceholderOfferPointsNowhere(t *testing.T) {
	got := placeholderOffer("192.0.2.1", 20000, []Codec{pcmu, pcma})

	require.Equal(t, Media{Addr: "192.0.2.1", Port: 20000, Codecs: []Codec{pcmu, pcma, dtmfCodec}, Direction: DirSendRecv}, got)
}

func TestHoldOfferOnlyChangesDirection(t *testing.T) {
	offer := placeholderOffer("192.0.2.1", 20000, []Codec{pcmu})

	got := holdOffer(offer)

	want := offer
	want.Direction = DirInactive
	require.Equal(t, want, got)
	require.Equal(t, DirSendRecv, offer.Direction)
}

func TestCustomerOfferUsesStreamsAddressAndOnlyItsCodec(t *testing.T) {
	stream := Media{Addr: "203.0.113.10", Port: 20000, Codecs: []Codec{pcma, dtmfCodec}, Direction: DirSendRecv}

	got, err := customerOffer(stream)

	require.NoError(t, err)
	require.Equal(t, Media{Addr: "203.0.113.10", Port: 20000, Codecs: []Codec{pcma, dtmfCodec}, Direction: DirSendRecv}, got)
}

func TestCustomerOfferDropsDTMFStreamDidNotAccept(t *testing.T) {
	stream := Media{Addr: "203.0.113.10", Port: 20000, Codecs: []Codec{pcmu}, Direction: DirSendRecv}

	got, err := customerOffer(stream)

	require.NoError(t, err)
	require.Equal(t, []Codec{pcmu}, got.Codecs)
}

func TestCustomerOfferRefusesAnAnswerWithoutAudio(t *testing.T) {
	_, err := customerOffer(Media{Addr: "203.0.113.10", Port: 20000, Codecs: []Codec{dtmfCodec}})

	require.ErrorContains(t, err, "no audio codec")
}

func TestRetargetPointsStreamAtTheCustomer(t *testing.T) {
	offered := Media{Addr: "203.0.113.10", Port: 20000, Codecs: []Codec{pcma, dtmfCodec}, Direction: DirSendRecv}
	customer := Media{Addr: "198.51.100.7", Port: 30000, Codecs: []Codec{pcma}, Direction: DirSendRecv}

	got, err := retargetOffer(customer, offered)

	require.NoError(t, err)
	require.Equal(t, Media{Addr: "198.51.100.7", Port: 30000, Codecs: []Codec{pcma, dtmfCodec}, Direction: DirSendRecv}, got)
}

func TestRetargetRefusesAnotherCodec(t *testing.T) {
	offered := Media{Addr: "203.0.113.10", Port: 20000, Codecs: []Codec{pcma}, Direction: DirSendRecv}
	customer := Media{Addr: "198.51.100.7", Port: 30000, Codecs: []Codec{pcmu}, Direction: DirSendRecv}

	_, err := retargetOffer(customer, offered)

	require.ErrorContains(t, err, "PCMA")
}

func TestRetargetRefusesRejectedAudio(t *testing.T) {
	offered := Media{Addr: "203.0.113.10", Port: 20000, Codecs: []Codec{pcma}, Direction: DirSendRecv}
	customer := Media{Addr: "198.51.100.7", Port: 0, Codecs: []Codec{pcma}, Direction: DirSendRecv}

	_, err := retargetOffer(customer, offered)

	require.ErrorContains(t, err, "rejected the audio")
}

func TestRejectAnswerClosesTheStream(t *testing.T) {
	offer := Media{Addr: "198.51.100.7", Port: 30000, Codecs: []Codec{pcmu, dtmfCodec}, Direction: DirSendRecv}

	got := rejectAnswer(offer)

	require.Equal(t, Media{Addr: "192.0.2.1", Port: 0, Codecs: []Codec{pcmu, dtmfCodec}, Direction: DirInactive}, got)
}

func TestRejectAnswerStillNamesAFormatWhenTheOfferHadNone(t *testing.T) {
	got := rejectAnswer(Media{})

	require.Equal(t, []Codec{pcmu}, got.Codecs)
}

func TestParseMediaNormalisesCodecNames(t *testing.T) {
	body := "v=0\r\no=- 1 1 IN IP4 192.0.2.1\r\ns=-\r\nc=IN IP4 192.0.2.1\r\nt=0 0\r\n" +
		"m=audio 30000 RTP/AVP 0 101\r\n" +
		"a=rtpmap:0 pcmu/8000\r\n" +
		"a=rtpmap:101 Telephone-Event/8000\r\n"

	got, err := ParseMedia([]byte(body))

	require.NoError(t, err)
	require.Equal(t, Media{Addr: "192.0.2.1", Port: 30000, Codecs: []Codec{pcmu, dtmfCodec}, Direction: DirSendRecv}, got)
}

func TestRetargetSucceedsWithLowercaseCodecInAnswer(t *testing.T) {
	offered := Media{Addr: "203.0.113.10", Port: 20000, Codecs: []Codec{pcmu, dtmfCodec}, Direction: DirSendRecv}
	// Customer answer parsed from SDP with lowercase "pcmu"
	customerSDP := "v=0\r\no=- 1 1 IN IP4 198.51.100.7\r\ns=-\r\nc=IN IP4 198.51.100.7\r\nt=0 0\r\n" +
		"m=audio 30000 RTP/AVP 0\r\n" +
		"a=rtpmap:0 pcmu/8000\r\n"
	customer, err := ParseMedia([]byte(customerSDP))
	require.NoError(t, err)

	got, err := retargetOffer(customer, offered)

	require.NoError(t, err)
	require.Equal(t, Media{Addr: "198.51.100.7", Port: 30000, Codecs: []Codec{pcmu, dtmfCodec}, Direction: DirSendRecv}, got)
}

func TestCheckCodecsAcceptsTheSupportedNamesAndRefusesOthers(t *testing.T) {
	require.NoError(t, CheckCodecs([]string{"PCMU", "PCMA", "G722"}))
	require.EqualError(t, CheckCodecs(nil), "at least one codec is required")
	require.EqualError(t, CheckCodecs([]string{"PCMU", "OPUS"}), `unsupported codec "OPUS", use PCMU, PCMA or G722`)
}
