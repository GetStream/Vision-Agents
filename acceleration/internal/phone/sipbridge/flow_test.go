package sipbridge

import (
	"context"
	"errors"
	"log/slog"
	"regexp"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func flowConfig() Config {
	cfg := validConfig()
	cfg.Logger = slog.New(slog.DiscardHandler)
	cfg.Call.RingTimeout = 50 * time.Millisecond
	cfg.CustomerTrunk.Codecs = []string{"PCMA", "PCMU"}
	return cfg
}

var (
	customerOfferSDP    = buildSDP(Media{Addr: "198.51.100.7", Port: 30000, Codecs: []Codec{pcmu, dtmfCodec}, Direction: DirSendRecv}, 1, 1)
	streamAnswerSDP     = buildSDP(Media{Addr: "203.0.113.10", Port: 20000, Codecs: []Codec{pcmu, dtmfCodec}, Direction: DirSendRecv}, 2, 1)
	streamAnswerPCMASDP = buildSDP(Media{Addr: "203.0.113.10", Port: 20000, Codecs: []Codec{pcma, dtmfCodec}, Direction: DirSendRecv}, 3, 1)
	customerPCMA        = buildSDP(Media{Addr: "198.51.100.7", Port: 30000, Codecs: []Codec{pcma, dtmfCodec}, Direction: DirSendRecv}, 4, 1)
	customerPCMU        = buildSDP(Media{Addr: "198.51.100.7", Port: 30000, Codecs: []Codec{pcmu}, Direction: DirSendRecv}, 5, 1)
	errBusy             = errors.New("486 Busy Here")
)

func mediaOf(t *testing.T, body []byte) Media {
	t.Helper()
	m, err := ParseMedia(body)
	require.NoError(t, err)
	return m
}

// Flow A

func TestFlowAConnectsCustomerOfferToStream(t *testing.T) {
	customer := &fakeLeg{inviteAnswer: customerOfferSDP}
	stream := &fakeLeg{inviteAnswer: streamAnswerSDP}

	require.NoError(t, flowA(t.Context(), flowConfig(), customer, stream))

	require.Equal(t, []string{"invite", "ack"}, customer.calls())
	require.Nil(t, customer.body(0))
	require.Equal(t, streamAnswerSDP, customer.body(1))
	require.Equal(t, []string{"invite", "ack"}, stream.calls())
	require.Equal(t, customerOfferSDP, stream.body(0))
}

func TestFlowACustomerNotAnsweringNeverRingsStream(t *testing.T) {
	customer := &fakeLeg{inviteErr: errBusy}
	stream := &fakeLeg{}

	err := flowA(t.Context(), flowConfig(), customer, stream)

	require.ErrorIs(t, err, errBusy)
	require.Empty(t, stream.calls())
}

func TestFlowARingTimeoutCancelsCustomer(t *testing.T) {
	customer := &fakeLeg{inviteBlocks: true}
	stream := &fakeLeg{}

	err := flowA(t.Context(), flowConfig(), customer, stream)

	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.Equal(t, []string{"invite"}, customer.calls())
	require.Empty(t, stream.calls())
}

func TestFlowAStreamRejectingHangsUpAnsweredCustomer(t *testing.T) {
	customer := &fakeLeg{inviteAnswer: customerOfferSDP}
	stream := &fakeLeg{inviteErr: errors.New("404 Not Found")}

	err := flowA(t.Context(), flowConfig(), customer, stream)

	require.ErrorContains(t, err, "404 Not Found")
	require.Equal(t, []string{"invite", "ack", "bye"}, customer.calls())
	ack := mediaOf(t, customer.body(1))
	require.Equal(t, 0, ack.Port)
	require.Equal(t, []Codec{pcmu, dtmfCodec}, ack.Codecs)
}

func TestFlowAFailedStreamAckHangsUpBoth(t *testing.T) {
	customer := &fakeLeg{inviteAnswer: customerOfferSDP}
	stream := &fakeLeg{inviteAnswer: streamAnswerSDP, ackErr: errors.New("broken pipe")}

	err := flowA(t.Context(), flowConfig(), customer, stream)

	require.ErrorContains(t, err, "broken pipe")
	require.Equal(t, []string{"invite", "ack", "bye"}, customer.calls())
	ack := mediaOf(t, customer.body(1))
	require.Equal(t, 0, ack.Port)
	require.Equal(t, []Codec{pcmu, dtmfCodec}, ack.Codecs)
	require.Equal(t, "bye", stream.last())
}

// Flow B

func TestFlowBSetsUpStreamFirstThenRetargets(t *testing.T) {
	stream := &fakeLeg{inviteAnswer: streamAnswerPCMASDP}
	customer := &fakeLeg{inviteAnswer: customerPCMA}

	require.NoError(t, flowB(t.Context(), flowConfig(), customer, stream))

	require.Equal(t, []string{"invite", "ack", "reinvite", "reinvite"}, stream.calls())
	require.Equal(t, []string{"invite", "ack"}, customer.calls())

	require.Equal(t,
		Media{Addr: "192.0.2.1", Port: 20000, Codecs: []Codec{pcma, pcmu, dtmfCodec}, Direction: DirSendRecv},
		mediaOf(t, stream.body(0)))
	require.Equal(t,
		Media{Addr: "192.0.2.1", Port: 20000, Codecs: []Codec{pcma, pcmu, dtmfCodec}, Direction: DirInactive},
		mediaOf(t, stream.body(2)))
	require.Equal(t,
		Media{Addr: "203.0.113.10", Port: 20000, Codecs: []Codec{pcma, dtmfCodec}, Direction: DirSendRecv},
		mediaOf(t, customer.body(0)))
	require.Equal(t,
		Media{Addr: "198.51.100.7", Port: 30000, Codecs: []Codec{pcma, dtmfCodec}, Direction: DirSendRecv},
		mediaOf(t, stream.body(3)))
}

func TestFlowBFirstStreamInviteUsesTheConfiguredPlaceholder(t *testing.T) {
	cfg := flowConfig()
	cfg.FlowB.PlaceholderAddr = "0.0.0.0"
	cfg.FlowB.PlaceholderPort = 4000
	stream := &fakeLeg{inviteAnswer: streamAnswerPCMASDP}
	customer := &fakeLeg{inviteAnswer: customerPCMA}

	require.NoError(t, flowB(t.Context(), cfg, customer, stream))

	first := mediaOf(t, stream.body(0))
	require.Equal(t, "0.0.0.0", first.Addr)
	require.Equal(t, 4000, first.Port)
}

func TestFlowBOffersToStreamShareOneSessionAndBumpTheVersion(t *testing.T) {
	stream := &fakeLeg{inviteAnswer: streamAnswerPCMASDP}
	customer := &fakeLeg{inviteAnswer: customerPCMA}

	require.NoError(t, flowB(t.Context(), flowConfig(), customer, stream))

	origin := regexp.MustCompile(`(?m)^o=- (\d+) (\d+) IN IP4 `)
	var ids, versions []string
	for _, i := range []int{0, 2, 3} {
		match := origin.FindSubmatch(stream.body(i))
		require.NotNil(t, match, "no o= line in %q", stream.body(i))
		ids = append(ids, string(match[1]))
		versions = append(versions, string(match[2]))
	}
	require.Equal(t, []string{ids[0], ids[0], ids[0]}, ids)
	require.Equal(t, []string{"1", "2", "3"}, versions)
}

func TestFlowBUnansweredCustomerLeavesNothingInStream(t *testing.T) {
	stream := &fakeLeg{inviteAnswer: streamAnswerPCMASDP}
	customer := &fakeLeg{inviteBlocks: true}

	err := flowB(t.Context(), flowConfig(), customer, stream)

	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.Equal(t, "bye", stream.last())
	require.Equal(t, []string{"invite"}, customer.calls())
}

func TestFlowBCustomerBusyHangsUpStream(t *testing.T) {
	stream := &fakeLeg{inviteAnswer: streamAnswerPCMASDP}
	customer := &fakeLeg{inviteErr: errBusy}

	err := flowB(t.Context(), flowConfig(), customer, stream)

	require.ErrorIs(t, err, errBusy)
	require.Equal(t, "bye", stream.last())
}

func TestFlowBCustomerAnsweringWithAnotherCodecHangsUpBoth(t *testing.T) {
	stream := &fakeLeg{inviteAnswer: streamAnswerPCMASDP}
	customer := &fakeLeg{inviteAnswer: customerPCMU}

	err := flowB(t.Context(), flowConfig(), customer, stream)

	require.ErrorContains(t, err, "only PCMA was offered")
	require.Equal(t, "bye", customer.last())
	require.Equal(t, "bye", stream.last())
}

func TestFlowBStreamRejectingStopsBeforeRinging(t *testing.T) {
	stream := &fakeLeg{inviteErr: errors.New("403 Forbidden")}
	customer := &fakeLeg{}

	err := flowB(t.Context(), flowConfig(), customer, stream)

	require.ErrorContains(t, err, "403 Forbidden")
	require.Equal(t, []string{"invite"}, stream.calls())
	require.Empty(t, customer.calls())
}

func TestFlowBHoldRefusedHangsUpStreamBeforeRinging(t *testing.T) {
	stream := &fakeLeg{inviteAnswer: streamAnswerPCMASDP, reinviteErr: errors.New("400 Bad Request")}
	customer := &fakeLeg{}

	err := flowB(t.Context(), flowConfig(), customer, stream)

	require.ErrorContains(t, err, "400 Bad Request")
	require.Equal(t, "bye", stream.last())
	require.Empty(t, customer.calls())
}

func TestFlowBStreamRejectingRetargetHangsUpBoth(t *testing.T) {
	stream := &fakeLeg{inviteAnswer: streamAnswerPCMASDP, reinviteErrs: []error{nil, errors.New("488 Not Acceptable Here")}}
	customer := &fakeLeg{inviteAnswer: customerPCMA}

	err := flowB(t.Context(), flowConfig(), customer, stream)

	require.ErrorContains(t, err, "488 Not Acceptable Here")
	require.Equal(t, "bye", customer.last())
	require.Equal(t, "bye", stream.last())
}

func TestFlowBFailedCustomerAckHangsUpBoth(t *testing.T) {
	stream := &fakeLeg{inviteAnswer: streamAnswerPCMASDP}
	customer := &fakeLeg{inviteAnswer: customerPCMA, ackErr: errors.New("network error")}

	err := flowB(t.Context(), flowConfig(), customer, stream)

	require.ErrorContains(t, err, "network error")
	require.Equal(t, "bye", customer.last())
	require.Equal(t, "bye", stream.last())
}

func TestFlowAFailedCustomerAckHangsUpBoth(t *testing.T) {
	customer := &fakeLeg{inviteAnswer: customerOfferSDP, ackErr: errors.New("network error")}
	stream := &fakeLeg{inviteAnswer: streamAnswerSDP}

	err := flowA(t.Context(), flowConfig(), customer, stream)

	require.ErrorContains(t, err, "network error")
	require.Equal(t, "bye", customer.last())
	require.Equal(t, "bye", stream.last())
}

func TestFlowBCancelWhileRingingStillHangsUpStream(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	stream := &fakeLeg{inviteAnswer: streamAnswerPCMASDP}
	customer := &fakeLeg{inviteBlocks: true}

	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	err := flowB(ctx, flowConfig(), customer, stream)

	require.ErrorIs(t, err, context.Canceled)
	require.Equal(t, "bye", stream.last())
	// The BYE went out with a live cleanup context (nil error)
	require.Len(t, stream.byeCtxErrs, 1)
	require.Nil(t, stream.byeCtxErrs[0])
}
