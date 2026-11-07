//go:build integration

package session

import (
	"log/slog"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/outcome"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// Whose omni-channel a call's card goes to on a call the router placed, and on a call whose
// phone caller joins after the agent.

const (
	// line is the customer's own number a call is placed from or rings, callee who a placed
	// call rang, and lateCaller who rings in.
	line       = "+15550100150"
	callee     = "+15550100160"
	lateCaller = "+15550100170"
)

// withTelephony gives the suite's manager telephony, which a session with a number needs.
func (s *EpisodeReadingSuite) withTelephony() {
	config, err := phone.DefaultConfig()
	s.Require().NoError(err)
	s.telephony, err = phone.NewService(phone.ServiceOptions{
		Registry: phone.NewRegistry(config), Store: s.store, Stream: phone.NewStreamFromClient(s.chat.Client),
		Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.manager = s.managerOver()
}

// placedCall keeps a call the router placed from line to callee in the deployment's own Stream
// app, as phone.Service.Call does, and returns its call id, which is the session's.
func (s *EpisodeReadingSuite) placedCall(callType string) string {
	return s.placedCallIn(0, callType)
}

// placedCallIn keeps a call the router placed into a Stream app, dialled under the vendor's
// call id the session names (placedSpec).
func (s *EpisodeReadingSuite) placedCallIn(streamApp int64, callType string) string {
	call := uuid.NewString()
	keeper := outcome.NewKeeper(s.store, time.Now)
	token, err := keeper.Placed(s.ctx, outcome.Call{
		Key:    outcome.Key{CustomerID: s.customerID, StreamAppPK: streamApp, CallType: callType, CallID: call, Local: line, Remote: callee},
		Vendor: "twilio",
	})
	s.Require().NoError(err)
	s.Require().NoError(keeper.Dialed(s.ctx, token, vendorCallID(call)))
	return call
}

// vendorCallID is the vendor's id for the leg of a placed call.
func vendorCallID(call string) string {
	return "vendor-" + call
}

// placedSpec is the session a customer's backend opens on a call it placed through
// POST /v1/phone/calls: the session id the call was placed into, the line and the vendor's
// call id in phone, as the SDKs send them, and no PhoneSpec.To, which the API cannot set
// (api.specOf).
func (s *EpisodeReadingSuite) placedSpec(call string, cards bool) Spec {
	return Spec{
		ID: call, CustomerID: s.customerID, ConfigID: s.configID, AgentName: "Athena", EpisodeCards: cards,
		CallerKind: auth.KindServer, Phone: &PhoneSpec{Number: line, VendorCallID: vendorCallID(call)},
		LLMTarget: "en-low-latency", STTTarget: "en-low-latency", TTSTarget: "en-low-latency", Instructions: "be brief",
	}
}

// inboundSpec is the session the router opens on a call rung to line (api.answerWithAgent).
func (s *EpisodeReadingSuite) inboundSpec(call string, cards bool) Spec {
	return Spec{
		ID: call, CustomerID: s.customerID, ConfigID: s.configID, AgentName: "Athena", EpisodeCards: cards,
		Phone:     &PhoneSpec{Number: line},
		LLMTarget: "en-low-latency", STTTarget: "en-low-latency", TTSTarget: "en-low-latency", Instructions: "be brief",
	}
}

// opens creates a session and closes it when the test ends.
func (s *EpisodeReadingSuite) opens(spec Spec) *Session {
	if s.telephony == nil {
		s.withTelephony()
	}
	created, err := s.manager.Create(s.ctx, spec)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _, _ = s.manager.Close(created.ID(), OwnerOf(created.Spec())) })
	return created
}

// attended has every session join on one edge that says who comes and goes, as Stream's does.
func (s *EpisodeReadingSuite) attended() *quietEdge {
	edge := newQuietEdge()
	s.edge = attendedEdge{edge}
	return edge
}

// startsWithin sets how long a phone session waits for anybody to join, until the test ends.
func (s *EpisodeReadingSuite) startsWithin(wait time.Duration) {
	inbound, outbound := inboundStartTimeout, outboundStartTimeout
	inboundStartTimeout, outboundStartTimeout = wait, wait
	s.T().Cleanup(func() { inboundStartTimeout, outboundStartTimeout = inbound, outbound })
}

// cardWritersDone is closed once no card is being written or waited for.
func (s *EpisodeReadingSuite) cardWritersDone() <-chan struct{} {
	done := make(chan struct{})
	cards := s.manager.cards
	go func() {
		cards.running.Wait()
		close(done)
	}()
	return done
}

// Defect: a call placed through POST /v1/phone/calls carries no PhoneSpec.To, so its card went
// to the phone on the call, the vendor's leg under the line's own number, and every person the
// line rang shared that omni-channel.
func (s *EpisodeReadingSuite) TestACallPlacedThroughTheAPIFilesItsCardUnderWhoItRang() {
	call := s.placedCall("agent")
	s.chat.PutCall("agent", call, "sip-"+line)

	created := s.opens(s.placedSpec(call, true))

	db, callees, lines := s.store, s.contactRows(callee), s.contactRows(line)
	s.Require().Eventually(func() bool { return len(callEpisodesIn(db, created.ID())) == 1 }, settleFor, 10*time.Millisecond)
	s.Equal(1, callees(), "the card is in the omni-channel of who the call rang")
	s.Zero(lines(), "the line's own number is nobody's omni-channel")
}

// Defect: the same call read the cards of the line's number, every callee's, rather than those
// of who it rang.
func (s *EpisodeReadingSuite) TestACallPlacedThroughTheAPIReadsTheCardsOfWhoItRang() {
	s.at(-30)
	s.smsThread(callee, "sms-author", "is the clinic open on Sunday?")
	call := s.placedCall("agent")
	s.chat.PutCall("agent", call, "sip-"+line)
	bound, err := s.apps.For(s.ctx, s.customerID)
	s.Require().NoError(err)
	spec := s.placedSpec(call, true)
	spec.CallID, spec.CallType = call, "agent"

	cards, _ := s.manager.cards.read(s.ctx, spec, bound)

	var handed string
	for _, card := range cards {
		handed += card.Content
	}
	s.Contains(handed, "is the clinic open on Sunday?")
}

// Defect: an end user's device could open a session under the id of a call its customer
// placed and be handed the cards of who that call rang.
func (s *EpisodeReadingSuite) TestAnEndUsersSessionOnAPlacedCallReadsNoneOfWhoItRang() {
	s.at(-30)
	s.smsThread(callee, "sms-author", "is the clinic open on Sunday?")
	call := s.placedCall("agent")
	s.chat.PutCall("agent", call, "sip-"+line)
	bound, err := s.apps.For(s.ctx, s.customerID)
	s.Require().NoError(err)
	spec := s.placedSpec(call, true)
	spec.CallID, spec.CallType = call, "agent"
	spec.CallerKind, spec.Caller = auth.KindAuthenticated, routing.Caller{UserID: "anya_123", Verified: true}

	cards, _ := s.manager.cards.read(s.ctx, spec, bound)

	s.Nil(cards)
}

// Defect: a backend's session with no phone could name the id of a call the router placed and
// be handed the cards of who that call rang, though no phone is on it.
func (s *EpisodeReadingSuite) TestASessionWithNoPhoneOnAPlacedCallReadsNoneOfWhoItRang() {
	s.at(-30)
	s.smsThread(callee, "sms-author", "is the clinic open on Sunday?")
	call := s.placedCall("agent")
	s.chat.PutCall("agent", call, "browser-user")
	bound, err := s.apps.For(s.ctx, s.customerID)
	s.Require().NoError(err)
	spec := s.placedSpec(call, true)
	spec.Phone = nil
	spec.CallID, spec.CallType = call, "agent"

	cards, _ := s.manager.cards.read(s.ctx, spec, bound)

	s.Nil(cards)
}

// Defect: the same session filed its card under who the placed call rang.
func (s *EpisodeReadingSuite) TestASessionWithNoPhoneOnAPlacedCallFilesNoCard() {
	call := s.placedCall("agent")
	s.chat.PutCall("agent", call, "browser-user")
	spec := s.placedSpec(call, true)
	spec.Phone = nil

	created := s.opens(spec)

	select {
	case <-s.cardWritersDone():
	case <-time.After(settleFor):
		s.FailNow("the card writer never stopped")
	}
	s.Empty(callEpisodesIn(s.store, created.ID()))
	s.Zero(s.contactRows(callee)(), "nobody the session is not with gets its card")
}

// Defect: a call the router placed in another Stream app of the same customer, with the
// session's call type and id, is another call, whose callee is nobody on this one.
func (s *EpisodeReadingSuite) TestACallPlacedInAnotherStreamAppNamesNobodyOnTheSessionsCall() {
	s.at(-30)
	s.smsThread(callee, "sms-author", "is the clinic open on Sunday?")
	bound, err := s.apps.For(s.ctx, s.customerID)
	s.Require().NoError(err)
	call := s.placedCallIn(bound.Identity.StreamApp+7, "agent")
	s.chat.PutCall("agent", call, "sip-"+line)
	spec := s.placedSpec(call, true)
	spec.CallID, spec.CallType, spec.StreamApp = call, "agent", bound.Identity.StreamApp

	cards, _ := s.manager.cards.read(s.ctx, spec, bound)

	s.Nil(cards)
}

// Defect: a session naming another vendor call than the one placed into its call id is not
// that call's, so who it rang is nobody on it.
func (s *EpisodeReadingSuite) TestACallPlacedUnderAnotherVendorCallNamesNobodyOnTheSessionsCall() {
	s.at(-30)
	s.smsThread(callee, "sms-author", "is the clinic open on Sunday?")
	call := s.placedCall("agent")
	s.chat.PutCall("agent", call, "sip-"+line)
	bound, err := s.apps.For(s.ctx, s.customerID)
	s.Require().NoError(err)
	spec := s.placedSpec(call, true)
	spec.CallID, spec.CallType = call, "agent"
	spec.Phone.VendorCallID = vendorCallID(uuid.NewString())

	cards, _ := s.manager.cards.read(s.ctx, spec, bound)

	s.Nil(cards)
}

// Defect: a call still being dialled (or whose vendor call id failed to save) has no vendor call
// id yet, so it is not the call a session naming a vendor call is on; its callee is nobody on it.
func (s *EpisodeReadingSuite) TestADialingCallWithNoVendorCallIDYetNamesNobodyOnTheSessionsCall() {
	s.at(-30)
	s.smsThread(callee, "sms-author", "is the clinic open on Sunday?")
	call := uuid.NewString()
	_, err := outcome.NewKeeper(s.store, time.Now).Placed(s.ctx, outcome.Call{
		Key:    outcome.Key{CustomerID: s.customerID, CallType: "agent", CallID: call, Local: line, Remote: callee},
		Vendor: "twilio",
	})
	s.Require().NoError(err)
	s.chat.PutCall("agent", call, "sip-"+line)
	bound, err := s.apps.For(s.ctx, s.customerID)
	s.Require().NoError(err)
	spec := s.placedSpec(call, true)
	spec.CallID, spec.CallType = call, "agent"

	cards, _ := s.manager.cards.read(s.ctx, spec, bound)

	s.Nil(cards)
}

// Defect: a backend whose SDK sends no vendor call id would find nobody the call it placed
// rang, so its card went to nobody.
func (s *EpisodeReadingSuite) TestAPlacedCallsSessionNamingNoVendorCallReadsWhoItRang() {
	s.at(-30)
	s.smsThread(callee, "sms-author", "is the clinic open on Sunday?")
	call := s.placedCall("agent")
	s.chat.PutCall("agent", call, "sip-"+line)
	bound, err := s.apps.For(s.ctx, s.customerID)
	s.Require().NoError(err)
	spec := s.placedSpec(call, true)
	spec.CallID, spec.CallType = call, "agent"
	spec.Phone.VendorCallID = ""

	cards, _ := s.manager.cards.read(s.ctx, spec, bound)

	var handed string
	for _, card := range cards {
		handed += card.Content
	}
	s.Contains(handed, "is the clinic open on Sunday?")
}

// Defect: a call of another type that has the session's call id is another call, whose callee
// is nobody on this one.
func (s *EpisodeReadingSuite) TestACallPlacedIntoAnotherCallTypeNamesNobodyOnTheSessionsCall() {
	s.at(-30)
	s.smsThread(callee, "sms-author", "is the clinic open on Sunday?")
	call := s.placedCall("default")
	s.chat.PutCall("agent", call)
	bound, err := s.apps.For(s.ctx, s.customerID)
	s.Require().NoError(err)
	spec := s.placedSpec(call, true)
	spec.CallID, spec.CallType = call, "agent"

	cards, _ := s.manager.cards.read(s.ctx, spec, bound)

	s.Nil(cards)
}

// Defect: a call placed with no record kept of who it rang (its session names another id)
// filed its card under the line's own number, the phone on the call.
func (s *EpisodeReadingSuite) TestACallWhoseCalleeIsNotKnownFilesNoCardUnderItsOwnLine() {
	s.startsWithin(300 * time.Millisecond)
	call := uuid.NewString()
	s.chat.PutCall("agent", call, "sip-00"+strings.TrimPrefix(line, "+"))

	created := s.opens(s.placedSpec(call, true))

	select {
	case <-s.cardWritersDone():
	case <-time.After(settleFor):
		s.FailNow("the card writer never stopped")
	}
	s.Empty(callEpisodesIn(s.store, created.ID()))
	s.Zero(s.contactRows(line)(), "the line's own number is nobody's omni-channel")
}

// Defect: a session a backend opened on a call the router did not place, a worker answering
// an arriving call, found no placed call and so wrote no card for its caller.
func (s *EpisodeReadingSuite) TestAWorkersSessionOnAnArrivingCallFilesTheCallersCard() {
	call := uuid.NewString()
	s.chat.PutCall("agent", call, "sip-"+lateCaller)
	spec := s.inboundSpec(call, true)
	spec.CallerKind = auth.KindServer

	created := s.opens(spec)

	db, callers := s.store, s.contactRows(lateCaller)
	s.Require().Eventually(func() bool { return len(callEpisodesIn(db, created.ID())) == 1 }, settleFor, 10*time.Millisecond)
	s.Equal(1, callers())
}

// Defect: the card read the call once, when the agent joined, so a caller who joined after the
// agent got no card.
func (s *EpisodeReadingSuite) TestAnInboundCallerWhoJoinsAfterTheAgentGetsTheirCard() {
	edge := s.attended()
	call := uuid.NewString()
	s.chat.PutCall("agent", call)
	created := s.opens(s.inboundSpec(call, true))
	db, callers := s.store, s.contactRows(lateCaller)
	s.Never(func() bool { return len(callEpisodesIn(db, created.ID())) > 0 }, 200*time.Millisecond, 20*time.Millisecond,
		"no card before anybody rings in")

	s.chat.PutCall("agent", call, "sip-"+lateCaller)
	edge.arrives(stt.Participant{ID: "caller-" + uuid.NewString(), UserID: "sip-" + lateCaller})

	s.Require().Eventually(func() bool { return len(callEpisodesIn(db, created.ID())) == 1 }, settleFor, 10*time.Millisecond)
	s.Equal(1, callers())

	// A leg that drops comes back under a new connection id: the same call, the same card.
	rejoined := stt.Participant{ID: "caller-" + uuid.NewString(), UserID: "sip-" + lateCaller}
	watched, detach := created.Watch()
	defer detach()
	edge.arrives(rejoined)
	s.Require().Eventually(func() bool {
		for {
			select {
			case event := <-watched:
				if joined, ok := event.(agent.ParticipantJoined); ok && joined.Participant.ID == rejoined.ID {
					return true
				}
			default:
				return false
			}
		}
	}, settleFor, 10*time.Millisecond, "the session took the rejoin")
	s.Equal(Live, created.State())
	s.Len(callEpisodesIn(db, created.ID()), 1)
}

// Defect: the vendor's leg of a placed call can be the first phone to join it, under the
// line's own number, written as the carrier writes it; a card waiting for the caller took it
// as the person rather than the caller who joined next.
func (s *EpisodeReadingSuite) TestTheLineJoiningItsOwnCallIsNotTheCaller() {
	edge := s.attended()
	call := uuid.NewString()
	s.chat.PutCall("agent", call)
	created := s.opens(s.inboundSpec(call, true))

	edge.arrives(stt.Participant{ID: "leg-" + uuid.NewString(), UserID: "sip-00" + strings.TrimPrefix(line, "+")})
	edge.arrives(stt.Participant{ID: "caller-" + uuid.NewString(), UserID: "sip-" + lateCaller})

	db, lines, callers := s.store, s.contactRows(line), s.contactRows(lateCaller)
	s.Require().Eventually(func() bool { return len(callEpisodesIn(db, created.ID())) == 1 }, settleFor, 10*time.Millisecond)
	s.Equal(1, callers())
	s.Zero(lines(), "the line's own number is nobody's omni-channel")
}

// Defect: a card that waits for the caller with no end holds a goroutine for as long as the
// router runs, for every phone call nobody joined. It waits as long as the session waits for
// anybody.
func (s *EpisodeReadingSuite) TestACardStopsWaitingForTheCallerAtTheSessionsStartTimeout() {
	wait := 500 * time.Millisecond
	s.startsWithin(wait)
	s.attended()
	call := uuid.NewString()
	s.chat.PutCall("agent", call)
	started := time.Now()
	created := s.opens(s.inboundSpec(call, true))

	select {
	case <-s.cardWritersDone():
	case <-time.After(wait + settleFor):
		s.FailNow("the card writer still waits past the session's start timeout")
	}
	s.GreaterOrEqual(time.Since(started), wait, "it waited for the caller")
	s.Empty(callEpisodesIn(s.store, created.ID()))
}

// Defect: a session that closed before its caller joined kept its card writer, and the session
// with it, for the whole start timeout (90 s for a placed call).
func (s *EpisodeReadingSuite) TestACardStopsWaitingForTheCallerWhenTheSessionCloses() {
	s.startsWithin(time.Minute)
	s.attended()
	call := uuid.NewString()
	s.chat.PutCall("agent", call)
	created := s.opens(s.inboundSpec(call, true))

	_, err := s.manager.Close(created.ID(), OwnerOf(created.Spec()))
	s.Require().NoError(err)

	select {
	case <-s.cardWritersDone():
	case <-time.After(settleFor):
		s.FailNow("the card writer still waits for a caller after the session closed")
	}
	s.Empty(callEpisodesIn(s.store, created.ID()))
}

// Defect: a call no phone is on (a device's, a browser's) would wait for a phone caller who is
// never coming. It writes no card at once, as before.
func (s *EpisodeReadingSuite) TestACallWithNoPhoneWaitsForNoCaller() {
	s.startsWithin(settleFor)
	call := "call-" + uuid.NewString()
	s.chat.PutCall("agent", call, "browser-user")

	created := s.joinedCallOn(call)

	select {
	case <-s.cardWritersDone():
	case <-time.After(settleFor / 2):
		s.FailNow("the card writer waits for a phone on a call with none")
	}
	s.Empty(callEpisodesIn(s.store, created.ID()))
}

// joinedCallOn is a cascaded session with the cards on, on a call that is no phone session's.
func (s *EpisodeReadingSuite) joinedCallOn(call string) *Session {
	return s.opens(Spec{
		CustomerID: s.customerID, ConfigID: s.configID, AgentName: "Athena", EpisodeCards: true, CallID: call,
		LLMTarget: "en-low-latency", STTTarget: "en-low-latency", TTSTarget: "en-low-latency", Instructions: "be brief",
	})
}

// Control, its expectation from the base: under a config without episode_cards, a placed call
// and a caller who joins late write nothing, no contact row and no episode.
func (s *EpisodeReadingSuite) TestWithTheCardsOffAPlacedCallAndALateCallerWriteNothing() {
	placed := s.placedCall("agent")
	s.chat.PutCall("agent", placed, "sip-"+line)
	rung := uuid.NewString()
	s.chat.PutCall("agent", rung)

	outbound := s.opens(s.placedSpec(placed, false))
	edge := s.attended()
	inbound := s.opens(s.inboundSpec(rung, false))
	s.chat.PutCall("agent", rung, "sip-"+lateCaller)
	edge.arrives(stt.Participant{ID: "caller-" + uuid.NewString(), UserID: "sip-" + lateCaller})

	db := s.store
	rows := []func() int{s.contactRows(callee), s.contactRows(line), s.contactRows(lateCaller)}
	s.Never(func() bool {
		return len(callEpisodesIn(db, outbound.ID(), inbound.ID())) > 0 || rows[0]() > 0 || rows[1]() > 0 || rows[2]() > 0
	}, 500*time.Millisecond, 20*time.Millisecond)
}
