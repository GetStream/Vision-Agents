package session

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"slices"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ConnectorScopeRequired is a connector tool call the provider refused because the grant
// lacks access it needs (core.OutcomeScopeRequired), and the step-up consent begun for it on
// the caller's own connection (AI-854). A client opens LaunchURL in a popup and posts it
// HandoffToken, as for a consent the backend begins. The old grant keeps working until that
// consent succeeds.
type ConnectorScopeRequired struct {
	// Name is the binding's alias.
	Name         string
	ConnectorID  string
	ConnectionID string
	// Scopes are the scopes the provider asked for, empty for a claims challenge.
	Scopes          []string
	AuthorizationID string
	LaunchURL       string
	HandoffToken    string
	ExpiresAt       time.Time
}

// scopeRequired is the status the model reads while the person is shown a step-up to press.
const scopeRequired = "scope_required"

// stepUps are the step-up consents a dispatcher has begun, one open at a time per
// connection, so a provider that refuses every call for want of a scope asks the person once
// per reply rather than once per call.
type stepUps struct {
	consents Consents
	// notify sends an event to the session's watchers (Session.broadcast).
	notify func(Event)
	logger *slog.Logger

	mu sync.Mutex
	// open is the newest step-up begun, by connection id.
	open map[string]openStepUp
}

// openStepUp is one step-up begun, what the provider asked for when it was, and the reply
// (llm.ToolCall.TurnID) it was begun in.
type openStepUp struct {
	asked   core.Outcome
	consent Consent
	turnID  string
}

// newStepUps is the step-ups of a session whose deployment can begin a consent, nil when it
// cannot (Connectors.Consents is nil): a call the provider refuses for want of a scope then
// fails as it did before.
func newStepUps(consents Consents, notify func(Event), logger *slog.Logger) *stepUps {
	if consents == nil {
		return nil
	}
	return &stepUps{consents: consents, notify: notify, logger: logger, open: map[string]openStepUp{}}
}

// stepUp asks the caller for the access the provider asked for on a call of r, and returns
// what the model reads instead of the provider's refusal. It reports false when nobody can be
// asked here, and the call then fails as it did before: the deployment cannot begin a
// consent, or the connection is not the caller's own. The app's connections are the app's
// backend to reconnect (createAuthorization), not an end user's.
//
// A step-up still open for the connection that asks for at least as much is reused by a
// later call in the reply it was begun in (turnID): no second attempt and no second event.
// One that was used or expired is not, and neither is one begun in an earlier reply, as
// askToLogIn reuses a login only while the reply being written shows it: the person may have
// closed its popup, whose launch page cannot be handed off again, or missed its event, which
// is not replayed to a watcher that attaches later.
func (d *dispatcher) stepUp(ctx context.Context, r route, turnID string, asked core.Outcome) (string, bool) {
	s := d.stepUps
	if s == nil || d.spec.Caller.UserID == "" || r.connection.OwnerType != store.OwnerUser ||
		r.connection.OwnerID != d.spec.Caller.UserID {
		return "", false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if open, found := s.open[r.connection.ID]; found && turnID != "" && open.turnID == turnID &&
		covers(open.asked, asked) && d.attemptOpen(ctx, open.consent.AuthorizationID) {
		return stepUpRequired(open.consent.Name), true
	}
	consent, err := s.consents(ctx, ConsentRequest{
		CustomerID: d.spec.CustomerID, ConnectorID: r.binding.ConnectorID,
		UserID: d.spec.Caller.UserID, ConnectionID: r.connection.ID, StepUp: &asked,
	})
	if err != nil {
		s.logger.Warn("could not begin a step-up consent", "connector", r.binding.Name, "error", err)
		return "", false
	}
	s.open[r.connection.ID] = openStepUp{asked: asked, consent: consent, turnID: turnID}
	s.notify(ConnectorScopeRequired{
		Name: r.binding.Name, ConnectorID: r.binding.ConnectorID, ConnectionID: consent.ConnectionID,
		Scopes: slices.Clone(asked.Scopes), AuthorizationID: consent.AuthorizationID,
		LaunchURL: consent.LaunchURL, HandoffToken: consent.HandoffToken, ExpiresAt: consent.ExpiresAt,
	})
	return stepUpRequired(consent.Name), true
}

// attemptOpen reports whether the attempt is still open: not used, not expired, and on a
// connection that was not deleted (store.ConnectorAuthorizationAttemptByID).
func (d *dispatcher) attemptOpen(ctx context.Context, id string) bool {
	_, err := d.store.ConnectorAuthorizationAttemptByID(ctx, id)
	return err == nil
}

// covers reports whether a step-up begun for open also asks for what asked asks for: the
// same claims challenge, and every scope asked among the ones open asked for.
func covers(open, asked core.Outcome) bool {
	if open.Claims != asked.Claims {
		return false
	}
	for _, scope := range asked.Scopes {
		if !slices.Contains(open.Scopes, scope) {
			return false
		}
	}
	return true
}

// stepUpRequired is what the model reads while the person is shown a step-up to press. It
// names no URL and no token: those are for the person.
func stepUpRequired(name string) string {
	raw, _ := json.Marshal(struct {
		Status  string `json:"status"`
		Message string `json:"message"`
	}{
		Status: scopeRequired,
		Message: fmt.Sprintf("%s needs access the user has not granted yet. They have been shown a button "+
			"to grant it. Tell them to press it, and call the tool again once they have.", name),
	})
	return string(raw)
}
