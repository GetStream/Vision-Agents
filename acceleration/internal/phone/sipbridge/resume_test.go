package sipbridge

import (
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/netip"
	"strconv"
	"testing"
	"time"

	"github.com/emiago/sipgo/sip"
	"github.com/stretchr/testify/suite"
)

const (
	customerCallID = "customer-call"
	streamCallID   = "stream-call"
)

// ResumeSuite continues a call whose dialogs another bridge set up: a customer trunk and
// Stream on loopback, and the state that bridge stored.
type ResumeSuite struct {
	suite.Suite
	customer, stream *peer
	cfg              ResumeConfig
	state            CallState
}

func TestResumeSuite(t *testing.T) {
	suite.Run(t, new(ResumeSuite))
}

func (s *ResumeSuite) SetupTest() {
	s.setup(peerOptions{}, peerOptions{})
}

func (s *ResumeSuite) setup(customer, stream peerOptions) {
	s.customer = startPeer(s.T(), customer)
	s.stream = startPeer(s.T(), stream)
	s.state = CallState{
		Customer: LegState{
			CallID: customerCallID, LocalTag: "ct-local", RemoteTag: "ct-remote",
			From:         "<sip:+15550001@127.0.0.1>;tag=ct-local",
			To:           "<sip:+15550002@127.0.0.1>;tag=ct-remote",
			Transport:    "tcp",
			LocalCSeq:    17340,
			RemoteTarget: fmt.Sprintf("sip:peer@127.0.0.1:%d;transport=tcp", s.customer.port),
			LastSDP:      []byte("v=0 to customer\r\n"),
		},
		Stream: LegState{
			CallID: streamCallID, LocalTag: "st-local", RemoteTag: "st-remote",
			From:         "<sip:+15550002@127.0.0.1>;tag=st-local",
			To:           "<sip:+15550001@127.0.0.1>;tag=st-remote",
			Transport:    "tcp",
			ReinviteHost: "127.0.0.1",
			LocalCSeq:    22180,
			RemoteTarget: "sip:stream-sip@10.9.9.9:5060;alias=10.21.119.123~5060~1",
			RouteSet:     []string{fmt.Sprintf("<sip:127.0.0.1:%d;transport=tcp;lr;did=1.2>", s.stream.port)},
			LastSDP:      []byte("v=0 to stream\r\n"),
		},
	}
	s.cfg = ResumeConfig{
		CustomerTrunk: CustomerTrunk{Host: "127.0.0.1", Port: s.customer.port, Transport: "tcp"},
		Stream: StreamTrunk{
			URI:      fmt.Sprintf("sip:+15550001@127.0.0.1:%d", s.stream.port),
			Username: "bridge", Password: "secret", Transport: "tcp",
		},
		Logger: slog.New(slog.DiscardHandler),
		guard:  guard{allow: func(ip netip.Addr) bool { return ip.IsLoopback() }},
	}
}

func (s *ResumeSuite) resume() *Call {
	call, err := Resume(s.T().Context(), s.cfg, s.state)
	s.Require().NoError(err)
	s.T().Cleanup(call.Release)
	return call
}

func (s *ResumeSuite) requireDone(call *Call) {
	select {
	case <-call.Done():
	case <-time.After(2 * time.Second):
		s.FailNow("call has not ended")
	}
}

// requireReinvite checks the re-INVITE p got continues the dialog in state, and its ACK.
func (s *ResumeSuite) requireReinvite(p *peer, state LegState, cseq uint32) {
	invites := p.requests(sip.INVITE, state.CallID)
	s.Require().Len(invites, 1)
	inv := invites[0]
	s.Equal(state.LocalTag, inv.From().Params.GetOr("tag", ""))
	s.Equal(state.RemoteTag, inv.To().Params.GetOr("tag", ""))
	s.Equal(cseq, inv.CSeq().SeqNo)
	s.Equal(state.LastSDP, inv.Body())
	contact := inv.Contact()
	s.Require().NotNil(contact)
	s.Equal(inv.Source(), net.JoinHostPort(contact.Address.Host, strconv.Itoa(contact.Address.Port)),
		"Contact names the address the request came from")

	s.Eventually(func() bool { return len(p.requests(sip.ACK, state.CallID)) == 1 }, 2*time.Second, 10*time.Millisecond)
	s.Equal(cseq, p.requests(sip.ACK, state.CallID)[0].CSeq().SeqNo)
}

func (s *ResumeSuite) TestResumeSendsEachSideAReinviteInItsOldDialog() {
	s.resume()

	s.requireReinvite(s.customer, s.state.Customer, 18340)
	s.requireReinvite(s.stream, s.state.Stream, 23180)
}

func (s *ResumeSuite) TestResumeKeepsTheStreamTargetParameters() {
	s.resume()

	invites := s.stream.requests(sip.INVITE, streamCallID)
	s.Require().Len(invites, 1)
	s.Equal("sip:stream-sip@127.0.0.1:5060;alias=10.21.119.123~5060~1", invites[0].Recipient.String())
	route := invites[0].GetHeader("Route")
	s.Require().NotNil(route)
	s.Equal(s.state.Stream.RouteSet[0], route.Value())
}

func (s *ResumeSuite) TestResumeAnswersAChallengeWithTheTrunkCredentials() {
	s.setup(peerOptions{}, peerOptions{challengeOnce: true})

	s.resume()

	invites := s.stream.requests(sip.INVITE, streamCallID)
	s.Require().Len(invites, 2)
	s.Nil(invites[0].GetHeader("Authorization"))
	s.NotNil(invites[1].GetHeader("Authorization"))
	s.Equal(invites[0].CSeq().SeqNo+1, invites[1].CSeq().SeqNo)
}

func (s *ResumeSuite) TestAByeFromOneSideAfterResumeEndsTheOther() {
	call := s.resume()

	s.customer.sendBye(customerCallID)

	s.Eventually(func() bool { return len(s.stream.requests(sip.BYE, streamCallID)) == 1 }, 2*time.Second, 10*time.Millisecond)
	s.requireDone(call)
	s.NoError(call.Err())
}

func (s *ResumeSuite) TestAResumedCallKeepsBothLegsAlive() {
	s.cfg.KeepaliveInterval = 50 * time.Millisecond

	s.resume()

	for _, side := range []struct {
		p     *peer
		state LegState
	}{{s.customer, s.state.Customer}, {s.stream, s.state.Stream}} {
		invites := side.p.requests(sip.INVITE, side.state.CallID)
		s.Require().Len(invites, 1)
		reinvite := invites[0].CSeq().SeqNo
		s.Eventually(func() bool {
			for _, opt := range side.p.requests(sip.OPTIONS, side.state.CallID) {
				if opt.CSeq().SeqNo > reinvite {
					return true
				}
			}
			return false
		}, time.Second, 10*time.Millisecond, "no in-dialog OPTIONS on %s", side.state.CallID)
	}
}

func (s *ResumeSuite) TestResumeEndsBothSidesWhenOneSideRefuses() {
	s.setup(peerOptions{reinviteStatus: sip.StatusCallTransactionDoesNotExists}, peerOptions{})

	call, err := Resume(s.T().Context(), s.cfg, s.state)

	s.Nil(call)
	se, ok := errors.AsType[*statusError](err)
	s.Require().True(ok, "%v", err)
	s.Equal(481, se.code)
	s.Eventually(func() bool { return len(s.stream.requests(sip.BYE, streamCallID)) == 1 }, 2*time.Second, 10*time.Millisecond)
}

func (s *ResumeSuite) TestResumeEndsTheCustomerWhenTheStreamStateIsUnreadable() {
	s.state.Stream.RemoteTarget = "sip:stream-sip@[::1"

	call, err := Resume(s.T().Context(), s.cfg, s.state)

	s.Nil(call)
	s.ErrorContains(err, "stream leg remote target")
	s.Eventually(func() bool { return len(s.customer.requests(sip.BYE, customerCallID)) == 1 }, 2*time.Second, 10*time.Millisecond)
}

func (s *ResumeSuite) TestReleaseOfAResumedCallSendsNoBye() {
	call, err := Resume(s.T().Context(), s.cfg, s.state)
	s.Require().NoError(err)

	call.Release()

	s.requireDone(call)
	time.Sleep(300 * time.Millisecond)
	s.Empty(s.customer.requests(sip.BYE, customerCallID))
	s.Empty(s.stream.requests(sip.BYE, streamCallID))
}

func (s *ResumeSuite) TestStateOfAResumedCallCarriesItsDialog() {
	call := s.resume()

	state := call.State()

	s.Equal(customerCallID, state.Customer.CallID)
	s.Equal("ct-local", state.Customer.LocalTag)
	s.Equal("ct-remote", state.Customer.RemoteTag)
	s.Equal(streamCallID, state.Stream.CallID)
	s.Equal("st-local", state.Stream.LocalTag)
	s.Equal("st-remote", state.Stream.RemoteTag)
	s.Equal("sip:stream-sip@10.9.9.9:5060;alias=10.21.119.123~5060~1", state.Stream.RemoteTarget)
	s.Equal(s.state.Stream.RouteSet, state.Stream.RouteSet)
	s.GreaterOrEqual(state.Stream.LocalCSeq, uint32(23180))
}
