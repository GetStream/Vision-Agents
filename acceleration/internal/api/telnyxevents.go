package api

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/telnyxsig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/vendorevent"
)

// telnyxEventsHookPath is where the Telnyx account's notification channel sends account
// events: port-outs, port-ins, number orders.
const telnyxEventsHookPath = telnyxCallHookPath + "/events"

// telnyxEvent is the part of a Telnyx event the router reads.
type telnyxEvent struct {
	Data struct {
		EventType string `json:"event_type"`
		ID        string `json:"id"`
		Payload   struct {
			// Status is any because only a number order's is read, and another event's
			// status need not be a string.
			Status any `json:"status"`
		} `json:"payload"`
	} `json:"data"`
}

// receiveTelnyxEvent counts and logs an account event Telnyx signed.
//
// Telnyx retries a delivery it saw no answer to, so one event id may be counted twice. That
// is accepted: the monitor pages on the counter's sum being above 0, which a second count
// does not change.
func (s *Server) receiveTelnyxEvent(w http.ResponseWriter, r *http.Request) {
	if s.telnyxPublicKey == nil {
		// Without the key anyone could raise the counter a monitor pages on.
		writeError(w, notFound("telnyx events are not configured"))
		return
	}

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, telnyxCallLimit))
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		writeError(w, payloadTooLarge("that event is too large"))
		return
	}
	if err != nil {
		writeError(w, invalidRequest("could not read that event"))
		return
	}
	err = telnyxsig.Verify(s.telnyxPublicKey,
		r.Header.Get(telnyxsig.TimestampHeader), r.Header.Get(telnyxsig.SignatureHeader), body, time.Now())
	if err != nil {
		s.logger.Warn("refused a telnyx account event", "reason", err.Error())
		writeError(w, unauthenticated("that is not an event from Telnyx"))
		return
	}

	var event telnyxEvent
	if err := json.Unmarshal(body, &event); err != nil || event.Data.EventType == "" {
		writeError(w, invalidRequest("that is not a Telnyx event"))
		return
	}
	status, _ := event.Data.Payload.Status.(string)
	name, level := vendorevent.Telnyx(event.Data.EventType, status)
	vendorevent.Count(vendorevent.VendorTelnyx, name, level)
	s.logger.Info("received a telnyx account event", "vendor", vendorevent.VendorTelnyx, "event", name,
		"level", string(level), "event_type", event.Data.EventType, "id", event.Data.ID)
	w.WriteHeader(http.StatusNoContent)
}
