package sipbridge

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"testing"

	"github.com/emiago/sipgo"
	"github.com/emiago/sipgo/sip"
	"github.com/stretchr/testify/require"
)

func TestDialRefusesAnInvalidConfigBeforeSendingAnything(t *testing.T) {
	_, err := Dial(t.Context(), Config{})

	require.ErrorContains(t, err, "customer_trunk.host is required")
}

func TestRegistryFindsALegByCallID(t *testing.T) {
	r := newRegistry()
	l := &sipLeg{side: streamSide}
	r.add("call-1", l)

	req := sip.NewRequest(sip.BYE, sip.Uri{Host: "x"})
	callID := sip.CallIDHeader("call-1")
	req.AppendHeader(&callID)
	got, ok := r.lookup(req)

	require.True(t, ok)
	require.Same(t, l, got)
}

func TestRegistryDoesNotKnowOtherCalls(t *testing.T) {
	r := newRegistry()

	req := sip.NewRequest(sip.BYE, sip.Uri{Host: "x"})
	callID := sip.CallIDHeader("someone-else")
	req.AppendHeader(&callID)
	_, ok := r.lookup(req)

	require.False(t, ok)
}

func TestFinishSetupHangsUpBothLegsWhenTheCallEndedDuringSetup(t *testing.T) {
	customer, stream := &fakeLeg{}, &fakeLeg{}
	call := newCall(customer, stream, slog.New(slog.DiscardHandler))
	call.onBye(t.Context(), customerSide)

	err := finishSetup(t.Context(), call, customer, stream)

	require.ErrorContains(t, err, "call ended during setup")
	require.Equal(t, []string{"bye"}, customer.calls())
	require.Equal(t, []string{"bye", "bye"}, stream.calls())
}

func TestFinishSetupLeavesALiveCallAlone(t *testing.T) {
	customer, stream := &fakeLeg{}, &fakeLeg{}
	call := newCall(customer, stream, slog.New(slog.DiscardHandler))

	require.NoError(t, finishSetup(t.Context(), call, customer, stream))
	require.Empty(t, customer.calls())
}

func reinviteFrom(body []byte) *sip.Request {
	req := sip.NewRequest(sip.INVITE, sip.Uri{User: "sipbridge", Host: "10.0.0.5"})
	callID := sip.CallIDHeader("call-1")
	req.AppendHeader(&callID)
	req.AppendHeader(&sip.CSeqHeader{SeqNo: 3, MethodName: sip.INVITE})
	req.AppendHeader(sip.NewHeader("Session-Expires", "1800;refresher=uac"))
	if body != nil {
		req.SetBody(body)
		req.AppendHeader(sip.NewHeader("Content-Type", "application/sdp"))
	}
	return req
}

func reinviteLeg() *sipLeg {
	ua := &sipgo.DialogUA{ContactHDR: sip.ContactHeader{Address: sip.Uri{User: "sipbridge", Host: "198.51.100.20", Port: 61234}}}
	return &sipLeg{ua: ua, log: slog.New(slog.DiscardHandler), lastSDP: []byte("v=0 ours\r\n")}
}

func TestAnswerReinviteWithoutSDPRepeatsOurLastSDPAndForwardsNothing(t *testing.T) {
	l := reinviteLeg()
	forwarded := false
	forward := func(context.Context, []byte) ([]byte, error) { forwarded = true; return nil, nil }

	res := answerReinvite(t.Context(), l, reinviteFrom(nil), forward)

	require.False(t, forwarded)
	require.Equal(t, 200, res.StatusCode)
	require.Equal(t, []byte("v=0 ours\r\n"), res.Body())
	require.Equal(t, "application/sdp", res.GetHeader("Content-Type").Value())
	require.Equal(t, "sip:sipbridge@198.51.100.20:61234", res.Contact().Address.String())
}

func TestAnswerReinviteWithSDPForwardsItAndRemembersTheAnswer(t *testing.T) {
	l := reinviteLeg()
	var offer []byte
	forward := func(_ context.Context, b []byte) ([]byte, error) { offer = b; return []byte("v=0 theirs\r\n"), nil }

	res := answerReinvite(t.Context(), l, reinviteFrom([]byte("v=0 hold\r\n")), forward)

	require.Equal(t, []byte("v=0 hold\r\n"), offer)
	require.Equal(t, 200, res.StatusCode)
	require.Equal(t, []byte("v=0 theirs\r\n"), res.Body())
	require.Equal(t, []byte("v=0 theirs\r\n"), l.lastSDP)
}

func TestAnswerReinviteRejectsWhenTheOtherLegFails(t *testing.T) {
	l := reinviteLeg()
	forward := func(context.Context, []byte) ([]byte, error) { return nil, errors.New("400 Bad Request") }

	res := answerReinvite(t.Context(), l, reinviteFrom([]byte("v=0 hold\r\n")), forward)

	require.Equal(t, 488, res.StatusCode)
	require.Equal(t, []byte("v=0 ours\r\n"), l.lastSDP)
}

func TestInfoStatusPassesTheOtherLegsRejectionBack(t *testing.T) {
	err := fmt.Errorf("passing on: %w", &statusError{method: sip.INFO, code: 415, reason: "Unsupported Media Type"})

	code, reason := infoStatus(err)

	require.Equal(t, 415, code)
	require.Equal(t, "Unsupported Media Type", reason)
}

func TestInfoStatusIs500WhenTheOtherLegGaveNoAnswer(t *testing.T) {
	code, _ := infoStatus(errors.New("transaction timed out"))

	require.Equal(t, 500, code)
}

func TestInfoStatusIs200WithoutAnError(t *testing.T) {
	code, reason := infoStatus(nil)

	require.Equal(t, 200, code)
	require.Equal(t, "OK", reason)
}
