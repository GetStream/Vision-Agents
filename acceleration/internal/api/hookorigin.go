package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// hookStaleAfter is how old a signed event may be and still be acted on. An older one is a
// replay, or a delivery Stream gave up retrying long ago.
const hookStaleAfter = 10 * time.Minute

// hookOrigin is the Stream app a verified hook came from, which is the only thing that says
// whose its work is: a hook carries no customer, and what it names is only acted on within
// its app.
type hookOrigin struct {
	// customer is the registered app's customer, empty for a hook signed with the
	// deployment's own secret.
	customer string
	// app is the app's id, zero for the deployment's in deployment mode while that id is
	// not known.
	app int64
	// deployment is a hook signed with the deployment's own secret.
	deployment bool
	// apiKey is the key whose secret signed it.
	apiKey string
	// handOver hands what arrived to the node at an address: a hook's signed request replayed
	// whole (replayed), or a queued delivery over the peer path (queuedHandOver). Nil on the
	// node a peer handed it to, which answers it itself.
	handOver func(ctx context.Context, address string) (*node.ForwardResponse, error)
	// queued is a delivery the queue's worker holds a lease on, whose answer runs in line, so
	// the worker records it once it has ended.
	queued bool
}

// answer runs an answer: in line for a queued delivery, and off the request for a hook,
// because a model call takes seconds and Stream is waiting on the delivery.
func (o hookOrigin) answer(answer func()) {
	if o.queued {
		answer()
		return
	}
	go answer()
}

// replayed hands a hook to a peer whole, signature and all, because the node answering it
// verifies the delivery for itself rather than taking this node's word that it came from
// Stream. A hook a peer handed here goes nowhere further: two nodes each believing the other
// answers would pass it back and forth.
func (s *Server) replayed(r *http.Request, body []byte) func(ctx context.Context, address string) (*node.ForwardResponse, error) {
	if r.Header.Get(node.ForwardedHeader) != "" {
		return nil
	}
	return func(ctx context.Context, address string) (*node.ForwardResponse, error) {
		return s.forwarder.Ask(ctx, address, r, body)
	}
}

// scope is what rows a hook from this origin may act on: its app's, and for the
// deployment's own app, the rows written there before rows named an app.
func (o hookOrigin) scope() store.AppScope {
	return store.AppScope{App: o.app, Unpinned: o.deployment}
}

// owns reports whether work for a customer, pinned to an app, is this origin's to act on.
func (o hookOrigin) owns(customer string, app int64) bool {
	if o.deployment {
		return app == 0 || app == o.app
	}
	return customer == o.customer && app == o.app
}

// verifyHook checks a hook's signature with the secrets its app holds, and says which app
// it came from. It answers the request itself when the hook is not one to act on.
func (s *Server) verifyHook(w http.ResponseWriter, r *http.Request, payload []byte, what string) (hookOrigin, bool) {
	signature := r.Header.Get(signatureHeader)
	var pathApp int64
	if named := r.PathValue("app"); named != "" {
		app, ok := streamapp.ParseAppID(named)
		if !ok {
			http.NotFound(w, r)
			return hookOrigin{}, false
		}
		pathApp = app
	}

	if s.stream == nil || !s.stream.PerApp() {
		return s.verifyDeploymentHook(w, r, payload, signature, pathApp, what)
	}
	if origin, vouched := s.vouchedHook(r, pathApp); vouched {
		return origin, true
	}

	verifiers, err := s.stream.Verifiers(r.Context(), strings.TrimSpace(r.Header.Get(auth.APIKeyHeader)), pathApp)
	if errors.Is(err, streamapp.ErrDeploymentAppUnknown) {
		// Stream retries, and by then the deployment will know which app is its own.
		writeStreamWaiting(w)
		return hookOrigin{}, false
	}
	if err != nil {
		s.logger.Error("could not find the keys to check a "+what+" with", "error", err)
		writeError(w, unavailable("could not check that "+what))
		return hookOrigin{}, false
	}
	for _, verifier := range verifiers {
		if !getstream.VerifySignature(payload, signature, verifier.Secret.Reveal()) {
			continue
		}
		return hookOrigin{app: verifier.StreamApp, deployment: verifier.Deployment,
			apiKey: verifier.APIKey}, true
	}
	s.logger.Warn("rejected a "+what+" with a bad signature", "bytes", len(payload), "app", pathApp)
	writeError(w, unauthenticated("that is not a "+what+" from Stream"))
	return hookOrigin{}, false
}

// vouchedHook is a hook the proxy checked the signature of with its app's secret, which the
// router acting with forwarded tokens holds none of. The token the proxy forwarded beside it
// is kept, so the hook's work is done with it. The deployment's own app is still checked
// here, with the secret the router holds for it.
func (s *Server) vouchedHook(r *http.Request, pathApp int64) (hookOrigin, bool) {
	if _, forwarded := s.stream.Forwarded(); !forwarded || s.authMode != auth.Proxy || !s.proxyDeclaresKind {
		return hookOrigin{}, false
	}
	app, ok := streamapp.ParseAppID(strings.TrimSpace(r.Header.Get(auth.HookAppHeader)))
	if !ok || (pathApp != 0 && pathApp != app) || app == s.stream.DeploymentApp() {
		return hookOrigin{}, false
	}
	customer := streamapp.CustomerOf(app)
	apiKey := strings.TrimSpace(r.Header.Get(auth.ServerTokenKeyHeader))
	if err := s.stream.KeepToken(r.Context(), customer, apiKey, strings.TrimSpace(r.Header.Get(auth.ServerTokenHeader))); err != nil {
		s.logger.Warn("stream: could not keep the server token the proxy forwarded with a hook", "stream_app", app, "error", err)
	}
	return hookOrigin{customer: customer, app: app, apiKey: apiKey}, true
}

// verifyDeploymentHook checks a hook in deployment mode, where every hook is the
// deployment's own app's and is signed with its secret.
func (s *Server) verifyDeploymentHook(w http.ResponseWriter, r *http.Request, payload []byte, signature string, pathApp int64, what string) (hookOrigin, bool) {
	var own int64
	if s.stream != nil {
		own = s.stream.DeploymentApp()
	}
	if pathApp != 0 && pathApp != own {
		http.NotFound(w, r)
		return hookOrigin{}, false
	}
	if !getstream.VerifySignature(payload, signature, s.hookSecret) {
		s.logger.Warn("rejected a "+what+" with a bad signature", "bytes", len(payload))
		writeError(w, unauthenticated("that is not a "+what+" from Stream"))
		return hookOrigin{}, false
	}
	return hookOrigin{app: own, deployment: true}, true
}

// hookStamp is what makes one delivery the same as another, and how old it is.
type hookStamp struct {
	CreatedAt json.RawMessage `json:"created_at"`
	SessionID string          `json:"session_id"`
	CallCid   string          `json:"call_cid"`
	Message   struct {
		ID string `json:"id"`
	} `json:"message"`
}

// fresh reports whether an event was made recently enough to act on. One that says nothing
// readable about when it was made is taken as made now.
func (h hookStamp) fresh(now time.Time) bool {
	at := eventTime(h.CreatedAt)
	return at.IsZero() || now.Sub(at) <= hookStaleAfter
}

// eventTime is when Stream made an event, from its created_at: RFC 3339, or an epoch number in
// seconds, milliseconds, microseconds or nanoseconds, told apart by its size (any time from 1973
// to 5138 reads as one of them). Zero when it says nothing readable.
func eventTime(createdAt json.RawMessage) time.Time {
	if len(createdAt) == 0 {
		return time.Time{}
	}
	var text string
	if json.Unmarshal(createdAt, &text) == nil {
		at, err := time.Parse(time.RFC3339Nano, text)
		if err != nil {
			return time.Time{}
		}
		return at
	}
	var number int64
	if json.Unmarshal(createdAt, &number) != nil || number <= 0 {
		return time.Time{}
	}
	switch {
	case number < 1e11:
		return time.Unix(number, 0)
	case number < 1e14:
		return time.UnixMilli(number)
	case number < 1e17:
		return time.UnixMicro(number)
	default:
		return time.Unix(0, number)
	}
}

// firstDelivery reports whether a hook is the first delivery of its event, which is the one
// acted on: Stream delivers again when it does not hear back in time.
func (s *Server) firstDelivery(ctx context.Context, origin hookOrigin, eventType string, stamp hookStamp) bool {
	if s.store == nil {
		return true
	}
	var key string
	switch {
	case eventType == getstream.EventTypeMessageNew && stamp.Message.ID != "":
		key = "message:" + stamp.Message.ID
	case stamp.CallCid != "":
		key = "call:" + stamp.CallCid + ":" + stamp.SessionID + ":" + eventType
	default:
		return true
	}
	first, err := s.store.FirstDelivery(ctx, strconv.FormatInt(origin.app, 10)+":"+key)
	if err != nil {
		// Answering twice is the lesser harm than answering never.
		s.logger.Warn("could not record a hook delivery", "error", err)
		return true
	}
	return first
}

// acting reports whether a verified hook is one to act on: made recently, and not a
// delivery already acted on. Deployment mode acts on every delivery, as it always has.
func (s *Server) acting(ctx context.Context, origin hookOrigin, eventType string, payload []byte) bool {
	if s.stream == nil || !s.stream.PerApp() {
		return true
	}
	var stamp hookStamp
	if err := json.Unmarshal(payload, &stamp); err != nil {
		return true
	}
	if !stamp.fresh(time.Now()) {
		s.logger.Info("ignoring a stale hook", "type", eventType)
		return false
	}
	// The node that handed it over recorded the delivery, so this one is not the second.
	if node.Forwarded(ctx) {
		return true
	}
	if !s.firstDelivery(ctx, origin, eventType, stamp) {
		s.logger.Debug("ignoring a hook delivered again", "type", eventType)
		return false
	}
	return true
}

// hooksConfigured reports whether hooks can be told apart from anybody who found the URL:
// in app mode by every app's keys and the deployment's, otherwise by the deployment's
// secret.
func (s *Server) hooksConfigured() bool {
	return s.hookSecret != "" || (s.stream != nil && s.stream.PerApp())
}

// pinHook says which app a worker's session for what a hook started acts in: the hook's,
// for a registered app, whatever app its customer acts in by the time the session starts.
func (s *Server) pinHook(origin hookOrigin, customer, cid string) {
	if origin.deployment || s.sessions == nil {
		return
	}
	s.sessions.PinHook(customer, cid, origin.app)
}

// mayWrite reports whether what a hook found for a customer may be acted on: in app mode, a
// hook from the deployment's own app starts work only for a customer still writing there,
// never for one whose work there is only to be read.
func (s *Server) mayWrite(ctx context.Context, origin hookOrigin, customer string) bool {
	if !origin.deployment || s.stream == nil || !s.stream.PerApp() {
		return true
	}
	_, err := s.stream.ForApp(ctx, customer, origin.app)
	return err == nil
}
