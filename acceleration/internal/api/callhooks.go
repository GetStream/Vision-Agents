package api

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// signatureHeader carries the HMAC Stream signs a delivery with.
const signatureHeader = "X-Signature"

// callEvent is the part of a call event this hook acts on.
//
// The SDK's own event structs are not used to decode, only to verify: its Timestamp reads
// epoch nanoseconds and writes RFC 3339, so a delivery carrying one format fails to parse
// as the other and would be refused as though it were unsigned. Nothing here needs a
// timestamp off the wire — a call event arrives while the phone is still ringing, so when it
// was received is when it started — and the fields that are needed are a few strings.
type callEvent struct {
	CallCid string `json:"call_cid"`
	// SessionID is Stream's own session of the call, which its call stats are kept under.
	SessionID string `json:"session_id"`
	Call      struct {
		// Custom is whatever was put on the Stream call. Stream takes arbitrary JSON here.
		Custom map[string]any `json:"custom"`
		// Session lists who is in the call, which is where the caller's number is.
		Session *struct {
			Participants []callParticipant `json:"participants"`
		} `json:"session"`
	} `json:"call"`
	// Participant is who joined or left, on a participant event.
	Participant *callParticipant `json:"participant"`
}

// callParticipant is somebody in a call: a phone is a user named sip-<number>, and its user
// session is what names it on the event saying it left.
type callParticipant struct {
	UserSessionID string `json:"user_session_id"`
	User          struct {
		ID string `json:"id"`
	} `json:"user"`
}

// phone reports whether the participant is a phone, on a call over SIP.
func (p callParticipant) phone() bool {
	_, ok := phone.CallerNumber(p.User.ID)
	return ok
}

// receiveCallEvent takes the call events Stream sends and turns an arriving phone call into
// a call a worker is asked to answer.
//
// This is what makes inbound calling possible at all: a caller reaches a Stream call by
// themselves, over SIP, and nothing here knows about it until this arrives. Like the vendor
// answer host it carries no customer header, because Stream is not a customer. It is
// authenticated by signature instead.
//
// Every outcome short of something that did not come from Stream is a 200. Stream retries a
// non-2xx, and none of the reasons this cannot place a call — no worker waiting, a call
// belonging to nobody — get better on a second delivery, while the caller is on the line for
// the whole of it.
func (s *Server) receiveCallEvent(w http.ResponseWriter, r *http.Request) {
	if !s.hooksConfigured() {
		// Refusing is the only safe answer: without the secret there is no way to tell
		// Stream from anyone who found the URL, and this path starts agents.
		writeError(w, notFound("call events are not configured"))
		return
	}

	payload, ok := readHook(w, r, "call event")
	if !ok {
		s.logger.Warn("rejected a call event before reading its signature")
		return
	}
	origin, ok := s.verifyHook(w, r, payload, "call event")
	if !ok {
		return
	}
	// What the event starts here, an ended call's episode summaries, no key started.
	r = r.WithContext(routing.WithAttribution(r.Context(), routing.Attribution{APIKey: routing.OriginHookVideo}))

	eventType := getstream.GetEventType(payload)
	if eventType == "" {
		writeError(w, invalidRequest("could not read that call event"))
		return
	}

	switch eventType {
	case getstream.EventTypeCallSessionStarted:
		var event callEvent
		if err := json.Unmarshal(payload, &event); err != nil {
			writeError(w, invalidRequest("could not read that call event"))
			return
		}
		s.dispatchArrivingCall(r, origin, event, payload)

	case getstream.EventTypeCallSessionEnded:
		var event callEvent
		if err := json.Unmarshal(payload, &event); err != nil {
			writeError(w, invalidRequest("could not read that call event"))
			return
		}
		// Asked before the release, which deletes what marks a placed call.
		ended, phoneCall := s.endedPhoneCall(r, origin, event)
		s.releaseEndedCall(r, origin, event)
		s.endCallEpisodes(r, origin, event)
		if phoneCall {
			s.logCallSites(ended)
		}

	case getstream.EventTypeCallSessionParticipantJoined, getstream.EventTypeCallSessionParticipantLeft:
		var event callEvent
		if err := json.Unmarshal(payload, &event); err != nil {
			writeError(w, invalidRequest("could not read that call event"))
			return
		}
		s.phoneJoinedOrLeft(r, origin, eventType, event)

	default:
		s.logger.Debug("ignoring a call event", "type", eventType)
	}
	w.WriteHeader(http.StatusOK)
}

// dispatchArrivingCall works out whose call it is and hands it to one of their workers.
func (s *Server) dispatchArrivingCall(r *http.Request, origin hookOrigin, event callEvent, payload []byte) {
	callType, callID, split := strings.Cut(event.CallCid, ":")
	if !split {
		s.logger.Debug("a call event named no call", "cid", event.CallCid)
		return
	}
	if s.store == nil || s.dispatch == nil {
		return
	}

	// Only a call one of the numbers reaches is a phone call. Every video call in the app
	// arrives here too, and there is nothing to answer on those.
	// And only a number attached in the app the hook came from: a call of the same name in
	// another app rings somebody else's phone.
	// A number attached today names each caller's call for the session answering it and
	// puts itself on the call; one attached before that names a call of its own.
	custom := customOf(event.Call.Custom)
	rung, fresh := custom[phone.NumberKey]
	var number store.PhoneNumber
	var err error
	if fresh {
		number, err = s.store.NumberRungInApp(r.Context(), origin.scope(), rung)
	} else {
		number, err = s.store.NumberByCallInApp(r.Context(), origin.scope(), callType, callID)
	}
	if err == nil && !origin.deployment && number.CustomerID != origin.customer {
		err = store.ErrAmbiguousHook
	}
	if err == nil && !s.mayWrite(r.Context(), origin, number.CustomerID) {
		err = streamapp.ErrReadOnly
	}
	if err != nil {
		s.logger.Debug("no number reaches an arriving call",
			"call", event.CallCid, "stream_app", origin.app, "error", err)
		return
	}
	// The caller is on the line from here, whether an agent answers or the call is refused
	// and ended below, and the vendor bills every second of it. No key placed it, so its
	// minutes are the app's, and priced by the country of the number rung. One leg per Stream
	// session, however often or late this is delivered; a delivery another node handed over
	// was opened there.
	if s.phone != nil && !node.Forwarded(r.Context()) {
		caller := phoneCallerOf(event)
		s.phone.OpenLeg(r.Context(), routing.Owner{
			CustomerID:  number.CustomerID,
			Attribution: routing.Attribution{APIKey: routing.OriginPhoneInbound},
			Tags:        number.Tags,
		}, store.CallLeg{
			StreamAppPK: number.StreamAppPK, CallType: callType, CallID: callID,
			StreamSessionID: event.SessionID, ParticipantSession: caller.UserSessionID, Vendor: number.Vendor,
			Direction: store.Inbound, LocalNumber: number.E164, RemoteNumber: caller.number,
			Country: cmp.Or(number.Country, phone.Country(number.E164)),
		}, true)
	}
	// A suspension that cannot be read refuses the call too: it is there to stop fraud that
	// costs money.
	if _, err := s.store.TelephonySuspension(r.Context(), number.CustomerID); !errors.Is(err, store.ErrNotSuspended) {
		s.refuseArrivingCall(r, number, callType, callID, "refused an arriving call to a suspended app",
			"call", event.CallCid, "customer", number.CustomerID, "error", err)
		return
	}
	// The vendor bills the minutes of a call an agent answers, and the agent is billed work,
	// so an organization that may not pay has nobody answer.
	if err := s.gate.Admit(r.Context(), number.CustomerID); err != nil {
		answer, _ := moneyError(err)
		s.refuseArrivingCall(r, number, callType, callID, "refused an arriving call to an organization that may not pay for it",
			"call", event.CallCid, "customer", number.CustomerID, "code", answer.Code)
		return
	}
	// Only a call that rings a number is worth recording as delivered: every video call in
	// the app arrives here too.
	if !s.acting(r.Context(), origin, getstream.EventTypeCallSessionStarted, payload) {
		return
	}
	if s.forwardedToWorkers(r, payload, number.CustomerID) {
		return
	}

	call := dispatch.Call{
		CallID:       callID,
		CallType:     callType,
		CalledNumber: number.E164,
		CallerNumber: callerOf(event),
		Custom:       custom,
		At:           time.Now().UTC(),
	}
	if fresh {
		call.SessionID = callID
	}

	worker, err := s.dispatch.Assign(number.CustomerID, call)
	if err != nil {
		// Somebody is listening to a ringing phone that nothing is going to answer, which
		// is the most useful error this service can report.
		s.logger.Error("nobody could answer an arriving call",
			"call", event.CallCid, "customer", number.CustomerID,
			"number", number.E164, "error", err)
		return
	}
	s.pinHook(origin, number.CustomerID, event.CallCid)
	s.logger.Info("handed an arriving call to a worker",
		"call", event.CallCid, "customer", number.CustomerID,
		"number", number.E164, "caller", call.CallerNumber, "worker", worker.ID)
	arrived := []any{"worker", worker.ID, "stream_sip_host", phone.BridgeHost(number.BridgeURI)}
	if call.SessionID != "" {
		arrived = append(arrived, "session_id", call.SessionID)
	}
	s.logger.Info("arriving call sites", arrived...)
}

// releaseEndedCall tears down the per-call SIP trunks a placed call or transfer created,
// now that the call is over. Everything short of a malformed event is a 200: Stream
// retries a non-2xx, and nothing here is worth a redelivery: the caller is not waiting on
// it, and cleanup is best-effort. ReleaseCall removes the record before deleting at Stream,
// so a failed Stream delete is logged and the trunk leaks; a later delivery finds no record
// to retry, and there is no sweeper.
func (s *Server) releaseEndedCall(r *http.Request, origin hookOrigin, event callEvent) {
	if s.phone == nil {
		return
	}
	callType, callID, split := strings.Cut(event.CallCid, ":")
	if !split {
		s.logger.Debug("a call event named no call", "cid", event.CallCid)
		return
	}
	// The event is about a call in the app that signed it, and only what was made there is
	// released: a call's id is only unique within its app.
	if err := s.phone.ReleaseCall(r.Context(), origin.scope(), callType, callID); err != nil {
		s.logger.Error("could not release an ended call's resources", "call", event.CallCid, "error", err)
	}
	// Every leg of the call ends with it, and is billed for its minutes. A leg whose end is
	// lost here is ended by the sweep, at the longest a call is billed for.
	if err := s.phone.EndLegs(r.Context(), origin.scope(), callType, callID); err != nil {
		s.logger.Error("could not end an ended call's legs", "call", event.CallCid, "error", err)
	}
}

// endRefusedTimeout bounds ending a refused call at Stream, which runs after the hook answers.
const endRefusedTimeout = 10 * time.Second

// refuseArrivingCall dispatches no agent to a call and ends it at Stream at once. Stream's SIP
// bridge has answered the caller by the time the call's session starts (the caller is its
// first participant), so a call left alone would run until the caller hung up, and the vendor
// bills its seconds; it is ended off the request, so the hook still answers at once.
func (s *Server) refuseArrivingCall(r *http.Request, number store.PhoneNumber, callType, callID, why string, attributes ...any) {
	s.logger.Info(why+"; ending it", attributes...)
	if s.phone == nil {
		return
	}
	ctx := context.WithoutCancel(r.Context())
	go func() {
		ctx, cancel := context.WithTimeout(ctx, endRefusedTimeout)
		defer cancel()
		if err := s.phone.EndCall(ctx, number.CustomerID, number.StreamAppPK, callType, callID); err != nil {
			s.logger.Warn("could not end a refused arriving call; it runs until the caller hangs up",
				append(attributes, "end_error", err)...)
		}
	}()
}

// endCallEpisodes closes the episodes of an ended call and has them summarized off the request
// (omnichannel.Closer.EndCall, T55). Only a call under an agent config with episode_cards on
// has one; any other call matches no row, so nothing is written and no Stream is reached, and
// the answer is the 200 it was before. A failure is logged and answered 200 as well: Stream
// retries a non-2xx, and the episode is read by its last lines until it closes.
func (s *Server) endCallEpisodes(r *http.Request, origin hookOrigin, event callEvent) {
	if s.episodes == nil {
		return
	}
	_, callID, split := strings.Cut(event.CallCid, ":")
	if !split {
		return
	}
	// The call id is unique only in the app that signed the event, as releaseEndedCall's is.
	if err := s.episodes.EndCall(r.Context(), origin.scope(), callID); err != nil {
		s.logger.Error("could not close an ended call's episodes", "call", event.CallCid, "error", err)
	}
}

// callerOf reads the calling number off the SIP participant the routing rule named.
//
// It is empty when the caller has not been added to the session yet, which is normal: the
// call is dispatched the moment it starts, and the agent joining it will see the participant
// whether or not this did.
func callerOf(event callEvent) string {
	return phoneCallerOf(event).number
}

// phoneCaller is the phone in a call, and its user session there.
type phoneCaller struct {
	number        string
	UserSessionID string
}

// phoneCallerOf is the first phone among the participants an event lists, or nobody.
func phoneCallerOf(event callEvent) phoneCaller {
	if event.Call.Session == nil {
		return phoneCaller{}
	}
	for _, participant := range event.Call.Session.Participants {
		if caller, ok := phone.CallerNumber(participant.User.ID); ok {
			return phoneCaller{number: caller, UserSessionID: participant.UserSessionID}
		}
	}
	return phoneCaller{}
}

// phoneJoinedOrLeft starts the leg of a phone that joined a call, which is when a placed call
// or a transfer was answered, and ends and bills the leg of one that left. Anybody else coming
// and going, as every participant of every video call in the app does, changes nothing.
func (s *Server) phoneJoinedOrLeft(r *http.Request, origin hookOrigin, eventType string, event callEvent) {
	if s.phone == nil || event.Participant == nil || !event.Participant.phone() {
		return
	}
	callType, callID, split := strings.Cut(event.CallCid, ":")
	if !split {
		return
	}
	participant := event.Participant.UserSessionID
	var err error
	if eventType == getstream.EventTypeCallSessionParticipantJoined {
		err = s.phone.Connect(r.Context(), origin.scope(), callType, callID, event.SessionID, participant)
	} else {
		err = s.phone.Left(r.Context(), origin.scope(), callType, callID, participant)
	}
	if err != nil {
		s.logger.Error("could not record a phone joining or leaving a call", "type", eventType,
			"call", event.CallCid, "error", err)
	}
}

// customOf narrows the call's custom data to the strings a worker can read.
//
// Stream takes arbitrary JSON there, but everything this service puts on a call is a string,
// and passing nested objects through the socket would make the field's type depend on who
// set it.
func customOf(custom map[string]any) map[string]string {
	if len(custom) == 0 {
		return nil
	}
	narrowed := make(map[string]string, len(custom))
	for key, value := range custom {
		if text, ok := value.(string); ok {
			narrowed[key] = text
		}
	}
	return narrowed
}
