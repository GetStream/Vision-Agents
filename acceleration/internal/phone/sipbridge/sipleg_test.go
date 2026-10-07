package sipbridge

import (
	"strings"
	"testing"

	"github.com/emiago/sipgo"
	"github.com/emiago/sipgo/sip"
	"github.com/stretchr/testify/require"
)

func testInvite() *sip.Request {
	req := sip.NewRequest(sip.INVITE, sip.Uri{User: "+15550001111", Host: "sip.carrier.test", Port: 5060})
	from := &sip.FromHeader{Address: sip.Uri{User: "+15550002222", Host: "sip.carrier.test"}, Params: sip.NewParams()}
	from.Params.Add("tag", "fromtag")
	req.AppendHeader(from)
	req.AppendHeader(&sip.ToHeader{Address: sip.Uri{User: "+15550001111", Host: "sip.carrier.test"}, Params: sip.NewParams()})
	callID := sip.CallIDHeader("call-1")
	req.AppendHeader(&callID)
	req.AppendHeader(&sip.CSeqHeader{SeqNo: 2, MethodName: sip.INVITE})
	req.AppendHeader(&sip.ContactHeader{Address: sip.Uri{User: "sipbridge", Host: "10.0.0.5", Port: 51000}})
	req.AppendHeader(sip.NewHeader("Via", "SIP/2.0/TCP 10.0.0.5:51000;branch=z9hG4bK1"))
	return req
}

func TestNewACKCarriesTheAnswerToTheRemoteTarget(t *testing.T) {
	invite := testInvite()
	res := sip.NewResponseFromRequest(invite, 200, "OK", nil)
	res.To().Params.Add("tag", "totag")
	res.AppendHeader(&sip.ContactHeader{Address: sip.Uri{User: "carrier", Host: "198.51.100.7", Port: 5060}})

	ack := newACK(invite, res, []byte("v=0\r\n"))

	require.Equal(t, sip.ACK, ack.Method)
	require.Equal(t, "sip:carrier@198.51.100.7:5060", ack.Recipient.String())
	require.Equal(t, uint32(2), ack.CSeq().SeqNo)
	require.Equal(t, sip.ACK, ack.CSeq().MethodName)
	require.Equal(t, "call-1", ack.CallID().Value())
	totag, _ := ack.To().Params.Get("tag")
	require.Equal(t, "totag", totag)
	require.Equal(t, []byte("v=0\r\n"), ack.Body())
	require.Equal(t, "application/sdp", ack.GetHeader("Content-Type").Value())
}

func TestNewACKWithoutBodyHasNoContentType(t *testing.T) {
	invite := testInvite()
	res := sip.NewResponseFromRequest(invite, 200, "OK", nil)

	ack := newACK(invite, res, nil)

	require.Nil(t, ack.GetHeader("Content-Type"))
	require.Equal(t, "sip:+15550001111@sip.carrier.test:5060", ack.Recipient.String())
}

func TestAuthorizeAnswersAWWWAuthenticateChallenge(t *testing.T) {
	req := testInvite()
	res := sip.NewResponseFromRequest(req, 401, "Unauthorized", nil)
	res.AppendHeader(sip.NewHeader("WWW-Authenticate", `Digest realm="sip.test", nonce="abc", algorithm=md5`))

	require.NoError(t, authorize(req, res, "alice", "secret"))

	h := req.GetHeader("Authorization")
	require.NotNil(t, h)
	require.True(t, strings.HasPrefix(h.Value(), "Digest "))
	require.Contains(t, h.Value(), `username="alice"`)
	require.Contains(t, h.Value(), `uri="sip:+15550001111@sip.carrier.test:5060"`)
}

func TestAuthorizeUsesProxyHeadersFor407(t *testing.T) {
	req := testInvite()
	res := sip.NewResponseFromRequest(req, 407, "Proxy Authentication Required", nil)
	res.AppendHeader(sip.NewHeader("Proxy-Authenticate", `Digest realm="sip.test", nonce="abc"`))

	require.NoError(t, authorize(req, res, "alice", "secret"))

	require.NotNil(t, req.GetHeader("Proxy-Authorization"))
	require.Nil(t, req.GetHeader("Authorization"))
}

func TestAuthorizeFailsWithoutAChallenge(t *testing.T) {
	req := testInvite()
	res := sip.NewResponseFromRequest(req, 401, "Unauthorized", nil)

	require.ErrorContains(t, authorize(req, res, "alice", "secret"), "WWW-Authenticate")
}

func TestNewACKDoesNotCopyRoutes(t *testing.T) {
	req := testInvite()
	req.AppendHeader(sip.NewHeader("Route", "<sip:proxy1.test;lr>"))
	req.AppendHeader(sip.NewHeader("Route", "<sip:proxy2.test;lr>"))
	res := sip.NewResponseFromRequest(req, 200, "OK", nil)

	require.Empty(t, newACK(req, res, nil).GetHeaders("Route"))
}

var _ leg = (*sipLeg)(nil)

// answeredLeg is a leg whose INVITE was answered by a server that put its own address
// in Contact and whose 2xx arrived from a different address, like a proxy in front of it.
func answeredLeg() *sipLeg {
	target := sip.Uri{User: "+15550002222", Host: "bridge.sip.test", UriParams: sip.NewParams()}
	invite := sip.NewRequest(sip.INVITE, target)
	from := &sip.FromHeader{Address: sip.Uri{User: "+15550001111", Host: "bridge.sip.test"}, Params: sip.NewParams()}
	from.Params.Add("tag", "fromtag")
	invite.AppendHeader(from)
	invite.AppendHeader(&sip.ToHeader{Address: target, Params: sip.NewParams()})
	callID := sip.CallIDHeader("call-1")
	invite.AppendHeader(&callID)
	invite.AppendHeader(&sip.CSeqHeader{SeqNo: 2, MethodName: sip.INVITE})
	invite.AppendHeader(&sip.ContactHeader{Address: sip.Uri{User: "sipbridge", Host: "10.0.0.5", Port: 51000}})
	invite.AppendHeader(sip.NewHeader("Via", "SIP/2.0/TCP 10.0.0.5:51000;branch=z9hG4bK1"))

	res := sip.NewResponseFromRequest(invite, 200, "OK", nil)
	res.To().Params.Add("tag", "totag")
	res.AppendHeader(&sip.ContactHeader{Address: sip.Uri{User: "peer", Host: "203.0.113.10", Port: 5060}})
	res.SetSource("203.0.113.9:5060")

	l := &sipLeg{target: target}
	l.dlg = &sipgo.DialogClientSession{Dialog: sipgo.Dialog{InviteRequest: invite, InviteResponse: res}}
	l.answer = res
	return l
}

func TestInDialogRequestsGoToTheRemoteContact(t *testing.T) {
	l := answeredLeg()

	for _, method := range []sip.RequestMethod{sip.INVITE, sip.INFO, sip.OPTIONS, sip.BYE} {
		req := l.inDialog(method, "", nil)

		require.Equal(t, "sip:peer@203.0.113.10:5060", req.Recipient.String(), method)
		require.Equal(t, "203.0.113.10:5060", req.Destination(), method)
		require.Empty(t, req.GetHeaders("Route"), method)
	}
}

// ipContactLeg is answeredLeg whose 2xx Contact has an IP as its host and a URI parameter,
// so the tests can show which parts of it a re-INVITE keeps.
func ipContactLeg(reinviteHost string) *sipLeg {
	l := answeredLeg()
	contact := l.dlg.InviteResponse.Contact()
	contact.Address = sip.Uri{User: "peer", Host: "198.51.100.36", Port: 5060, UriParams: sip.NewParams()}
	contact.Address.UriParams.Add("x-hint", "abc")
	l.reinviteHost = reinviteHost
	return l
}

func TestReinviteTakesTheTrunkHostAndKeepsTheRestOfTheContact(t *testing.T) {
	l := ipContactLeg("bridge.sip.example.com")

	req := l.inDialog(sip.INVITE, "application/sdp", []byte("v=0\r\n"))

	require.Equal(t, "sip:peer@bridge.sip.example.com:5060;x-hint=abc", req.Recipient.String())
}

func TestReinviteWithoutTheOptionTargetsTheContact(t *testing.T) {
	l := ipContactLeg("")

	req := l.inDialog(sip.INVITE, "application/sdp", []byte("v=0\r\n"))

	require.Equal(t, "sip:peer@198.51.100.36:5060;x-hint=abc", req.Recipient.String())
}

func TestOtherInDialogRequestsKeepTheContactEvenWithTheOption(t *testing.T) {
	l := ipContactLeg("bridge.sip.example.com")

	for _, method := range []sip.RequestMethod{sip.BYE, sip.OPTIONS, sip.INFO} {
		req := l.inDialog(method, "", nil)

		require.Equal(t, "sip:peer@198.51.100.36:5060;x-hint=abc", req.Recipient.String(), method)
	}
	require.Equal(t, "sip:peer@198.51.100.36:5060;x-hint=abc", l.ack(nil).Recipient.String())
	require.Equal(t, "198.51.100.36", l.dlg.InviteResponse.Contact().Address.Host)
}

func TestACKGoesToTheRemoteContact(t *testing.T) {
	l := answeredLeg()

	for _, ack := range []*sip.Request{
		l.ack(nil),
		newACK(l.inDialog(sip.INVITE, "application/sdp", []byte("v=0\r\n")), l.dlg.InviteResponse, nil),
	} {
		require.Equal(t, "sip:peer@203.0.113.10:5060", ack.Recipient.String())
		require.Equal(t, "203.0.113.10:5060", ack.Destination())
	}
}

func TestNATContactTakesTheAddressTheServerSaw(t *testing.T) {
	contact := &sip.ContactHeader{Address: sip.Uri{User: "sipbridge", Host: "10.0.0.5", Port: 51000}}
	res := sip.NewResponse(401, "Unauthorized")
	res.AppendHeader(sip.NewHeader("Via", "SIP/2.0/TCP 10.0.0.5:51000;branch=z9hG4bK1;rport=61234;received=198.51.100.20"))

	natContact(contact, res)

	require.Equal(t, "sip:sipbridge@198.51.100.20:61234", contact.Address.String())
}

func TestNATContactLeavesTheContactWithoutReceivedOrRport(t *testing.T) {
	contact := &sip.ContactHeader{Address: sip.Uri{User: "sipbridge", Host: "10.0.0.5", Port: 51000}}
	res := sip.NewResponse(401, "Unauthorized")
	res.AppendHeader(sip.NewHeader("Via", "SIP/2.0/TCP 10.0.0.5:51000;branch=z9hG4bK1;rport"))

	natContact(contact, res)

	require.Equal(t, "sip:sipbridge@10.0.0.5:51000", contact.Address.String())
}

func TestARingingLegHasNoDialogToHangUpOrReport(t *testing.T) {
	l := answeredLeg()
	ringing := sip.NewResponseFromRequest(l.dlg.InviteRequest, 180, "Ringing", nil)
	ringing.To().Params.Add("tag", "totag")
	l.dlg.InviteResponse, l.answer = ringing, nil

	require.NoError(t, l.Bye(t.Context()))
	require.Equal(t, LegState{}, l.State())
}

func TestAFailedInviteThatWasNeverAnsweredNeedsNoACK(t *testing.T) {
	invite := testInvite()
	for _, res := range []*sip.Response{
		nil,
		sip.NewResponseFromRequest(invite, 180, "Ringing", nil),
		sip.NewResponseFromRequest(invite, 487, "Request Terminated", nil),
	} {
		_, answered := ackForLateAnswer([]byte("v=0\r\n"), res)

		require.False(t, answered, res)
	}
}

func TestA2xxCrossingOurCancelIsACKedWithoutABodyWhenWeMadeTheOffer(t *testing.T) {
	res := sip.NewResponseFromRequest(testInvite(), 200, "OK", []byte("v=0 theirs\r\n"))

	ack, answered := ackForLateAnswer([]byte("v=0 ours\r\n"), res)

	require.True(t, answered)
	require.Nil(t, ack)
}

func TestA2xxCrossingOurCancelWithAnOfferIsACKedWithAnAnswerThatRejectsIt(t *testing.T) {
	offer := "v=0\r\n" +
		"o=- 7 7 IN IP4 198.51.100.7\r\n" +
		"s=-\r\n" +
		"c=IN IP4 198.51.100.7\r\n" +
		"t=0 0\r\n" +
		"m=audio 40000 RTP/AVP 8\r\n" +
		"a=rtpmap:8 PCMA/8000\r\n"
	res := sip.NewResponseFromRequest(testInvite(), 200, "OK", []byte(offer))

	ack, answered := ackForLateAnswer(nil, res)

	require.True(t, answered)
	require.Equal(t, "v=0\r\n"+
		"o=- 1 1 IN IP4 192.0.2.1\r\n"+
		"s=sipbridge\r\n"+
		"c=IN IP4 192.0.2.1\r\n"+
		"t=0 0\r\n"+
		"m=audio 0 RTP/AVP 8\r\n"+
		"a=rtpmap:8 PCMA/8000\r\n"+
		"a=ptime:20\r\n"+
		"a=inactive\r\n", string(ack))
}
