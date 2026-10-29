package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatdelivery"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// chatDeliveryPath is where Stream Chat delivers the messages written in agent channels
// itself, rather than through the app's webhook, which the customer can overwrite.
const chatDeliveryPath = "/v1/chat/internal/messages"

// chatDeliveryPeerPath is where a router's queue worker hands a queued delivery to the router
// answering in its channel. It is served by hand and named in no spec: only a peer calls it,
// through the peer forwarder, which is what node.Forwarded(ctx) attests.
const chatDeliveryPeerPath = "/internal/chat-deliveries/answer"

// errChatDeliveryUnsettled is a queued delivery nothing took this time: the reason is logged
// where it happened, and the worker tries again.
var errChatDeliveryUnsettled = errors.New("api: nothing took the queued chat delivery")

// chatDeliveryReleaseTimeout bounds queueing a delivery again that nothing took: one update by
// key. The chat delivery worker's recordTimeout, which bounds its own row writes.
const chatDeliveryReleaseTimeout = 10 * time.Second

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
		origin.handOver = s.replayed(r, body)
		s.routeArrivingMessage(r.Context(), origin, delivery.messageEvent)
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

// queuedChatDelivery is a queued delivery as one router hands it to another: the origin the
// endpoint verified, and the event.
type queuedChatDelivery struct {
	Seq        int64  `json:"seq"`
	StreamApp  int64  `json:"stream_app"`
	MessageID  string `json:"message_id"`
	CustomerID string `json:"customer_id"`
	Deployment bool   `json:"deployment"`
	APIKey     string `json:"api_key"`
	Body       []byte `json:"body"`
}

// origin is where the delivery came from, as the endpoint verified it.
func (q queuedChatDelivery) origin() hookOrigin {
	return hookOrigin{customer: q.CustomerID, app: q.StreamApp, deployment: q.Deployment, apiKey: q.APIKey,
		queuedMessage: q.MessageID}
}

// AnswerChatDelivery answers a delivery the endpoint queued, as it would have answered it on
// the request, and returns once the answer has ended: the worker's lease holds the delivery
// until then (chatdelivery.Lease). An answer from a session or a dispatch worker, here or on a
// peer, settles the row as it starts (answerOnce), so a try after that answers nothing. The
// endpoint already checked it is an addressed new message, first delivered and fresh, so none
// of that is checked again: a retry ten minutes on would read as stale.
func (s *Server) AnswerChatDelivery(ctx context.Context, delivery store.ChatDelivery) error {
	queued := queuedChatDelivery{Seq: delivery.Seq, StreamApp: delivery.StreamApp, MessageID: delivery.MessageID,
		CustomerID: delivery.CustomerID, Deployment: delivery.Deployment, APIKey: delivery.APIKey, Body: delivery.Body}
	var event chatDelivery
	if err := json.Unmarshal(delivery.Body, &event); err != nil {
		return stack.Wrap(fmt.Errorf("api: read queued chat delivery: %w", err))
	}
	origin := queued.origin()
	origin.queued = true
	origin.handOver = func(ctx context.Context, address string) (*node.ForwardResponse, error) {
		return s.handChatDeliveryToPeer(ctx, address, queued)
	}
	if !s.routeArrivingMessage(ctx, origin, event.messageEvent) {
		return errChatDeliveryUnsettled
	}
	return nil
}

// handChatDeliveryToPeer gives a queued delivery to the node at address. The signed request is
// gone by now, so the peer takes this router's word for its origin, which only a node holding
// the deployment's credential can give (answerChatDeliveryForPeer).
func (s *Server) handChatDeliveryToPeer(ctx context.Context, address string, queued queuedChatDelivery) (*node.ForwardResponse, error) {
	body, err := json.Marshal(queued)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, chatDeliveryPeerPath, bytes.NewReader(body))
	if err != nil {
		return nil, stack.Wrap(err)
	}
	s.logger.Info("handing a queued chat delivery to another node", "seq", queued.Seq, "node", address)
	return s.forwarder.Ask(ctx, address, request, body)
}

// answerChatDeliveryForPeer is the peer hand-off's other end: the message is answered here, off
// the request, as the webhook's is. Anything not handed over by a peer is answered as no such
// route, because the payload names its app and customer outright, which is only a peer's to say.
func (s *Server) answerChatDeliveryForPeer(w http.ResponseWriter, r *http.Request) {
	if !node.Forwarded(r.Context()) {
		writeError(w, notFound("no such route"))
		return
	}
	var queued queuedChatDelivery
	var event chatDelivery
	if json.NewDecoder(r.Body).Decode(&queued) != nil || json.Unmarshal(queued.Body, &event) != nil {
		writeError(w, invalidRequest("the queued delivery could not be read"))
		return
	}
	s.logger.Info("answering a queued chat delivery another node handed over", "seq", queued.Seq, "stream_app", queued.StreamApp)
	if !s.routeArrivingMessage(r.Context(), queued.origin(), event.messageEvent) {
		writeError(w, unavailable("nothing took the queued delivery"))
		return
	}
	w.WriteHeader(http.StatusOK)
}

// answerOnce settles a queued delivery's row as its answer starts, so a try of it after that, on
// this router or another, answers nothing (store.StartChatDeliveryAnswer). answer is false when
// the message is not to be answered now, and settled then says whether that holds: its answer
// already started, or the row could not be written and a later try may. A hook is never tried
// again, so it settles nothing.
func (s *Server) answerOnce(ctx context.Context, origin hookOrigin, event messageEvent) (answer, settled bool) {
	if origin.queuedMessage == "" {
		return true, true
	}
	started, err := s.store.StartChatDeliveryAnswer(ctx, origin.app, origin.queuedMessage, time.Now())
	if err != nil {
		s.logger.Error("could not settle a queued chat delivery before answering it",
			"channel", event.ChannelID, "message", origin.queuedMessage, "stream_app", origin.app, "error", err)
		return false, false
	}
	if !started {
		s.logger.Info("not answering a queued chat delivery whose answer already started",
			"channel", event.ChannelID, "message", origin.queuedMessage, "stream_app", origin.app)
	}
	return started, true
}

// unanswered queues again a delivery answerOnce settled and nothing then took, so a later try
// answers it. Written past a cancelled ctx: Close cancels the worker's, and a row left settled
// is a message nobody answers.
func (s *Server) unanswered(ctx context.Context, origin hookOrigin) {
	if origin.queuedMessage == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), chatDeliveryReleaseTimeout)
	defer cancel()
	if err := s.store.ReleaseChatDeliveryAnswer(ctx, origin.app, origin.queuedMessage); err != nil {
		s.logger.Error("a queued chat delivery nothing took could not be queued again, and is left unanswered",
			"message", origin.queuedMessage, "stream_app", origin.app, "error", err)
	}
}
