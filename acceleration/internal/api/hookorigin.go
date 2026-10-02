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
		app, err := strconv.ParseInt(named, 10, 64)
		if err != nil || app <= 0 || strconv.FormatInt(app, 10) != named {
			http.NotFound(w, r)
			return hookOrigin{}, false
		}
		pathApp = app
	}

	if s.stream == nil || !s.stream.PerApp() {
		return s.verifyDeploymentHook(w, r, payload, signature, pathApp, what)
	}

	verifiers, err := s.stream.Verifiers(r.Context(), strings.TrimSpace(r.Header.Get(auth.APIKeyHeader)), pathApp)
	if errors.Is(err, streamapp.ErrDeploymentAppUnknown) {
		// Stream retries, and by then the deployment will know which app is its own.
		w.Header().Set("Retry-After", strconv.Itoa(int(streamRetryAfter.Seconds())))
		http.Error(w, streamUnknown, http.StatusServiceUnavailable)
		return hookOrigin{}, false
	}
	if err != nil {
		s.logger.Error("could not find the keys to check a "+what+" with", "error", err)
		http.Error(w, "could not check that "+what, http.StatusServiceUnavailable)
		return hookOrigin{}, false
	}
	for _, verifier := range verifiers {
		if !getstream.VerifySignature(payload, signature, verifier.Secret.Reveal()) {
			continue
		}
		if !verifier.Deployment && s.store != nil {
			if err := s.store.TouchStreamAppWebhook(r.Context(), verifier.APIKey, time.Now()); err != nil {
				s.logger.Warn("could not record a hook a key signed", "api_key", verifier.APIKey, "error", err)
			}
		}
		return hookOrigin{customer: verifier.CustomerID, app: verifier.StreamApp, deployment: verifier.Deployment,
			apiKey: verifier.APIKey}, true
	}
	s.logger.Warn("rejected a "+what+" with a bad signature", "bytes", len(payload), "app", pathApp)
	http.Error(w, "that is not a "+what+" from Stream", http.StatusUnauthorized)
	return hookOrigin{}, false
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
		http.Error(w, "that is not a "+what+" from Stream", http.StatusUnauthorized)
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
	if len(h.CreatedAt) == 0 {
		return true
	}
	var text string
	if json.Unmarshal(h.CreatedAt, &text) == nil {
		at, err := time.Parse(time.RFC3339Nano, text)
		return err != nil || now.Sub(at) <= hookStaleAfter
	}
	var number int64
	if json.Unmarshal(h.CreatedAt, &number) == nil && number > 0 {
		at := time.Unix(0, number)
		if number < 1e15 {
			at = time.UnixMilli(number)
		}
		return now.Sub(at) <= hookStaleAfter
	}
	return true
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
