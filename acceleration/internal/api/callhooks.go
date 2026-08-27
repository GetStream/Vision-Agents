package api

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
)

// callerPrefix is what the inbound routing rule names a SIP caller, so the number they are
// calling from can be read back off the participant. It has to match the template the rule
// is created with.
const callerPrefix = "sip-"

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
	if s.streamSecret == "" {
		// Refusing is the only safe answer: without the secret there is no way to tell
		// Stream from anyone who found the URL, and this path starts agents.
		http.Error(w, "call events are not configured", http.StatusNotFound)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "could not read that call event", http.StatusBadRequest)
		return
	}
	// Deliveries may be compressed, and the signature is over what is inside.
	payload, err := getstream.GunzipPayload(body)
	if err != nil {
		s.logger.Warn("rejected a call event", "error", err)
		http.Error(w, "that is not a call event from Stream", http.StatusUnauthorized)
		return
	}
	if !getstream.VerifySignature(payload, r.Header.Get(signatureHeader), s.streamSecret) {
		s.logger.Warn("rejected a call event with a bad signature", "bytes", len(payload))
		http.Error(w, "that is not a call event from Stream", http.StatusUnauthorized)
		return
	}

	eventType := getstream.GetEventType(payload)
	if eventType == "" {
		http.Error(w, "could not read that call event", http.StatusBadRequest)
		return
	}

	switch eventType {
	case getstream.EventTypeCallSessionStarted:
		var event callEvent
		if err := json.Unmarshal(payload, &event); err != nil {
			http.Error(w, "could not read that call event", http.StatusBadRequest)
			return
		}
		s.dispatchArrivingCall(r, event)

	case getstream.EventTypeCallSessionEnded:
		// Nothing to do: the worker is in the call and finds out from the call itself,
		// which is sooner and more reliable than a webhook round trip.
		s.logger.Debug("a call session ended")

	default:
		s.logger.Debug("ignoring a call event", "type", eventType)
	}
	w.WriteHeader(http.StatusOK)
}

// dispatchArrivingCall works out whose call it is and hands it to one of their workers.
func (s *Server) dispatchArrivingCall(r *http.Request, event callEvent) {
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
	number, err := s.store.NumberByCall(r.Context(), callType, callID)
	if err != nil {
		s.logger.Debug("no number reaches an arriving call",
			"call", event.CallCid, "error", err)
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
	s.logger.Info("handed an arriving call to a worker",
		"call", event.CallCid, "customer", number.CustomerID,
		"number", number.E164, "caller", call.CallerNumber, "worker", worker.ID)
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
		if caller, ok := strings.CutPrefix(participant.User.ID, callerPrefix); ok {
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
