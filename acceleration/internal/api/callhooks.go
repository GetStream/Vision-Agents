package api

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
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
// was received is when it started — and the fields that are needed are three strings.
type callEvent struct {
	CallCid string `json:"call_cid"`
	Call    struct {
		// Custom is whatever was put on the Stream call. Stream takes arbitrary JSON here.
		Custom map[string]any `json:"custom"`
		// Session lists who is in the call, which is where the caller's number is.
		Session *struct {
			Participants []struct {
				User struct {
					ID string `json:"id"`
				} `json:"user"`
			} `json:"participants"`
		} `json:"session"`
	} `json:"call"`
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
		s.releaseEndedCall(r, origin, event)

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
	number, err := s.store.NumberByCallInApp(r.Context(), origin.scope(), callType, callID)
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
	// Only a call that rings a number is worth recording as delivered: every video call in
	// the app arrives here too.
	if !s.acting(r.Context(), origin, getstream.EventTypeCallSessionStarted, payload) {
		return
	}

	call := dispatch.Call{
		CallID:       callID,
		CallType:     callType,
		CalledNumber: number.E164,
		CallerNumber: callerOf(event),
		Custom:       customOf(event.Call.Custom),
		At:           time.Now().UTC(),
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
}

// callerOf reads the calling number off the SIP participant the routing rule named.
//
// It is empty when the caller has not been added to the session yet, which is normal: the
// call is dispatched the moment it starts, and the agent joining it will see the participant
// whether or not this did.
func callerOf(event callEvent) string {
	if event.Call.Session == nil {
		return ""
	}
	for _, participant := range event.Call.Session.Participants {
		if caller, ok := phone.CallerNumber(participant.User.ID); ok {
			return caller
		}
	}
	return ""
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
