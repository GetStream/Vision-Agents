package api

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatdelivery"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// chatDeliveryPath is where Stream Chat delivers the messages written in agent channels
// itself, rather than through the app's webhook, which the customer can overwrite.
const chatDeliveryPath = "/v1/chat/internal/messages"

// chatDelivery is what Stream Chat delivers: the message.new event the webhook carries, and
// the app it was written in. The app is read from the signed body and never from a header,
// so nobody can name another app on a delivery Stream signed.
type chatDelivery struct {
	AppID int64  `json:"app_id"`
	Type  string `json:"type"`
	messageEvent
}

// receiveChatDelivery takes a message Stream Chat delivers and answers it as the message
// webhook does (receiveMessageEvent).
//
// Only Stream can sign one: it holds the private key, and the router only the public ones.
// Nothing an app's customer holds, its API secret, a token or its key, is enough. Every
// outcome short of something Stream did not sign, or an app this router does not serve, is a
// 200, for the reason the webhook gives.
func (s *Server) receiveChatDelivery(w http.ResponseWriter, r *http.Request) {
	if len(s.chatDeliveryKeys) == 0 {
		// Without a key there is no telling Stream from anyone who found the URL.
		writeError(w, notFound("Stream Chat deliveries are not configured"))
		return
	}

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, hookBodyLimit))
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		writeError(w, payloadTooLarge("that delivery is too large"))
		return
	}
	if err != nil {
		writeError(w, invalidRequest("could not read that delivery"))
		return
	}
	// Checked before anything reads the body, so what a forger sends is never parsed.
	if err := chatdelivery.Verify(s.chatDeliveryKeys, r.Header, body, time.Now()); err != nil {
		s.logger.Warn("rejected a Stream Chat delivery", "bytes", len(body), "error", err)
		writeError(w, unauthenticated("that is not a delivery from Stream Chat"))
		return
	}

	var delivery chatDelivery
	if err := json.Unmarshal(body, &delivery); err != nil || delivery.AppID <= 0 {
		writeError(w, invalidRequest("could not read that delivery: it is a message.new event with the app_id it was written in"))
		return
	}
	origin, ok := s.chatDeliveryOrigin(w, delivery.AppID)
	if !ok {
		return
	}
	if delivery.Type != getstream.EventTypeMessageNew {
		s.logger.Debug("ignoring a Stream Chat delivery", "type", delivery.Type)
		w.WriteHeader(http.StatusOK)
		return
	}
	if addressed(delivery.messageEvent) && s.acting(r.Context(), origin, delivery.Type, body) {
		s.routeArrivingMessage(r, origin, body, delivery.messageEvent)
	}
	w.WriteHeader(http.StatusOK)
}

// chatDeliveryOrigin is the app a delivery is from, as the webhook's would be for the same
// app (verifyHook): in deployment mode the deployment's own app only, and in app mode the
// deployment's app or a customer's own. It answers the request itself when the app is not
// one to act on.
func (s *Server) chatDeliveryOrigin(w http.ResponseWriter, app int64) (hookOrigin, bool) {
	var own int64
	if s.stream != nil {
		own = s.stream.DeploymentApp()
	}
	if own == 0 && s.stream != nil && s.stream.DeploymentAppKnowable() {
		// Stream retries, and by then the deployment will know which app is its own.
		writeStreamWaiting(w)
		return hookOrigin{}, false
	}
	if app == own {
		return hookOrigin{app: own, deployment: true}, true
	}
	if s.stream == nil || !s.stream.PerApp() {
		// Deployment mode acts in its own app alone, so another app's message is not this
		// router's to answer.
		writeError(w, notFound("that app is not served here"))
		return hookOrigin{}, false
	}
	return hookOrigin{customer: streamapp.CustomerOf(app), app: app}, true
}
