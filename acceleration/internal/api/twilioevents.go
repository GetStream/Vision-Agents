package api

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/twiliosig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/vendorevent"
)

// twilioEventsHookPath is where the Twilio account sends its Monitor alerts (the Console
// Debugger webhook) and its usage triggers' callbacks.
const twilioEventsHookPath = "/v1/phone/hooks/twilio/events"

// twilioEventLimit is the largest event read. Twilio documents no size, and an alert's Payload
// carries the request and response of the webhook that failed
// (https://www.twilio.com/docs/usage/troubleshooting/debugging-event-webhooks), so it is the
// limit of a Stream delivery rather than the 64 KiB of a Telnyx call.
const twilioEventLimit = hookBodyLimit

// twilioAlertPayload is the part of a Monitor alert's Payload the router reads. The docs give
// error_code no type and their sample sends a string, so a number is read as well.
type twilioAlertPayload struct {
	ErrorCode any `json:"error_code"`
}

// receiveTwilioEvent counts and logs a Monitor alert or a usage-trigger callback Twilio
// signed. Both are form posts signed with X-Twilio-Signature:
// https://www.twilio.com/docs/usage/troubleshooting/debugging-event-webhooks and
// https://www.twilio.com/docs/usage/api/usage-trigger (CallbackUrl request parameters).
//
// Twilio may send an alert again, and each copy is counted. That is accepted: the monitor
// pages on the counter's sum being above 0, which a second count does not change.
func (s *Server) receiveTwilioEvent(w http.ResponseWriter, r *http.Request) {
	public := strings.TrimRight(s.publicURL, "/")
	if s.twilioAuthToken == "" || public == "" {
		// Without the token anyone could raise the counter a monitor pages on, and without the
		// public URL no signature Twilio made over it can match.
		writeError(w, notFound("twilio events are not configured"))
		return
	}

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, twilioEventLimit))
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		writeError(w, payloadTooLarge("that event is too large"))
		return
	}
	if err != nil {
		writeError(w, invalidRequest("could not read that event"))
		return
	}
	form, err := url.ParseQuery(string(body))
	if err != nil {
		writeError(w, invalidRequest("that is not a form"))
		return
	}
	// The router is behind a proxy, so the URL Twilio called and signed is the public one.
	called := public + twilioEventsHookPath
	if r.URL.RawQuery != "" {
		called += "?" + r.URL.RawQuery
	}
	err = twiliosig.Verify(s.twilioAuthToken, called, form, r.Header.Get(twiliosig.SignatureHeader))
	if err != nil {
		s.logger.Warn("refused a twilio account event", "reason", err.Error())
		writeError(w, unauthenticated("that is not an event from Twilio"))
		return
	}

	switch {
	case form.Get("UsageTriggerSid") != "":
		name, level := vendorevent.TwilioUsageTrigger(form.Get("UsageCategory"))
		vendorevent.Count(vendorevent.VendorTwilio, name, level)
		s.logger.Info("received a twilio account event", "vendor", vendorevent.VendorTwilio, "event", name,
			"level", string(level), "usage_trigger_sid", form.Get("UsageTriggerSid"),
			"usage_category", form.Get("UsageCategory"), "trigger_by", form.Get("TriggerBy"),
			"trigger_value", form.Get("TriggerValue"), "current_value", form.Get("CurrentValue"))
	case form.Get("Sid") != "" && form.Get("Payload") != "":
		code := twilioErrorCode(form.Get("Payload"))
		name, level := vendorevent.TwilioAlert(code)
		vendorevent.Count(vendorevent.VendorTwilio, name, level)
		// Twilio's docs write Level both as Error and as ERROR, so it is logged in lower case.
		s.logger.Info("received a twilio account event", "vendor", vendorevent.VendorTwilio, "event", name,
			"level", string(level), "sid", form.Get("Sid"), "error_code", code,
			"twilio_level", strings.ToLower(form.Get("Level")))
	default:
		// The form is not logged: it may hold numbers.
		s.logger.Warn("refused a signed twilio account event that is no event")
		writeError(w, invalidRequest("that is not a Twilio event"))
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// twilioErrorCode is error_code from a Monitor alert's Payload, empty when there is none. An
// alert Twilio signed is counted either way, as error.other.
func twilioErrorCode(payload string) string {
	var fields twilioAlertPayload
	if json.Unmarshal([]byte(payload), &fields) != nil {
		return ""
	}
	switch code := fields.ErrorCode.(type) {
	case string:
		return code
	case float64:
		return strconv.FormatFloat(code, 'f', -1, 64)
	}
	return ""
}
