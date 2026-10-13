package api

import (
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	phonetelnyx "github.com/GetStream/Vision-Agents/acceleration/internal/phone/telnyx"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/telnyxsig"
)

// telnyxCallHookPath is the voice URL of the Telnyx TeXML application every attached
// Telnyx number is on. Telnyx asks it what to do with each call to one of them.
const telnyxCallHookPath = "/v1/phone/hooks/telnyx"

// telnyxCallLimit caps the form Telnyx posts, which is a few hundred bytes.
const telnyxCallLimit = 64 << 10

// receiveTelnyxCall answers a call to one of the numbers on the TeXML application with
// TeXML that dials that number's Stream trunk.
//
// Once the request is known to be Telnyx's, every failure is a hangup rather than an HTTP
// error: Telnyx would otherwise play its own error to the caller, and a hangup is also
// what stops one number's call from ever being sent anywhere but its own trunk.
func (s *Server) receiveTelnyxCall(w http.ResponseWriter, r *http.Request) {
	if s.telnyxPublicKey == nil {
		// Without the key anyone could ask this path for a number's trunk credentials.
		writeError(w, notFound("telnyx calls are not configured"))
		return
	}

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, telnyxCallLimit))
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		writeError(w, payloadTooLarge("that call is too large"))
		return
	}
	if err != nil {
		writeError(w, invalidRequest("could not read that call"))
		return
	}
	err = telnyxsig.Verify(s.telnyxPublicKey,
		r.Header.Get(telnyxsig.TimestampHeader), r.Header.Get(telnyxsig.SignatureHeader), body, time.Now())
	if err != nil {
		s.logger.Warn("refused a telnyx call webhook", "reason", err.Error())
		writeError(w, unauthenticated("that is not a call from Telnyx"))
		return
	}

	form, err := url.ParseQuery(string(body))
	if err != nil {
		hangUp(w, s.logger, "hung up a telnyx call whose form could not be read")
		return
	}
	logger := s.logger.With("call_sid", form.Get("CallSid"))
	if direction := form.Get("Direction"); direction != "inbound" {
		hangUp(w, logger, "hung up a telnyx call that is not inbound", "direction", direction)
		return
	}
	if s.telnyxConnectionID == "" {
		// Otherwise a signed call that names no application would match.
		hangUp(w, logger, "hung up a telnyx call: TELNYX_CONNECTION_ID is not set")
		return
	}
	// Dev and hosted routers share one Telnyx account, so a call for the other router's
	// application is signed with the same key. Its numbers are not in this database, but the
	// check keeps that from being the only thing that stops it.
	if connection := form.Get("ConnectionId"); connection != s.telnyxConnectionID {
		hangUp(w, logger, "hung up a telnyx call on another TeXML application than TELNYX_CONNECTION_ID",
			"connection_id", connection)
		return
	}
	to := form.Get("To")
	if to == "" {
		hangUp(w, logger, "hung up a telnyx call that names no number")
		return
	}
	if s.phone == nil {
		hangUp(w, logger, "hung up a telnyx call: telephony is not configured", "to", to)
		return
	}

	bridge, err := s.phone.InboundBridge(r.Context(), "telnyx", to)
	if errors.Is(err, phone.ErrNoSealer) {
		hangUp(w, logger, "hung up a telnyx call: this router has no key to open trunk passwords: "+
			"set ROUTER_AUTH_KEK, then attach the number again", "to", to)
		return
	}
	if errors.Is(err, store.ErrNoNumber) {
		hangUp(w, logger, "hung up a telnyx call to a number that is unknown, released or not attached", "to", to)
		return
	}
	if errors.Is(err, store.ErrNoStoredBridge) {
		hangUp(w, logger, "hung up a telnyx call to a number attached before its trunk was kept: "+
			"attach the number again", "to", to)
		return
	}
	if err != nil {
		hangUp(w, logger, "hung up a telnyx call: could not find its trunk", "to", to, "error", err)
		return
	}
	plan, err := phonetelnyx.RenderInbound(bridge)
	if err != nil {
		hangUp(w, logger, "hung up a telnyx call: could not render its trunk", "to", to, "error", err)
		return
	}
	writePlan(w, logger, plan)
}

func hangUp(w http.ResponseWriter, logger *slog.Logger, why string, attributes ...any) {
	logger.Warn(why, attributes...)
	writePlan(w, logger, phonetelnyx.RenderHangup())
}

func writePlan(w http.ResponseWriter, logger *slog.Logger, plan phone.Plan) {
	w.Header().Set("Content-Type", plan.ContentType)
	if _, err := w.Write(plan.Body); err != nil {
		logger.Error("could not serve a call plan", "error", err)
	}
}
