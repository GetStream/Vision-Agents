package api

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// connectorEventsPath is where a provider delivers the events of a built-in connector,
// followed by its id: the per-connector route the architecture doc decided on October 5
// («AI-816: keep, change, add» → Add, item 4 on connectors/planning). The per-provider-app
// route (T38) is a second route into receiveConnectorEvent.
const connectorEventsPath = "/v1/agents/connectors/events/"

// maxConnectorEventBytes caps the body read before it is verified. Slack's pages name no
// largest event (the Events API and request verification pages, opened October 6, 2026), so
// this is the cap the router's other provider webhooks already use, channels.MaxDeliveryBytes
// and plugins.MaxEventBytes (256 KiB). Unverified against Slack's largest event.
const maxConnectorEventBytes = 256 << 10

// signingSecretSuffix follows client.env in the variable that holds the operator's signing
// secret. The operator's client id and secret are <client.env>_MCP_CLIENT_ID and _SECRET
// (oauth2code.EnvClients), and the signing secret belongs to the same provider app, so it
// sits under the same prefix. A new name: the prototype read no provider events.
const signingSecretSuffix = "_MCP_SIGNING_SECRET"

// errNoConnectorEvents is the one answer for a route that takes no events: no such built-in,
// one without a channel block, a verifier or a secret this deployment lacks, or a secret only
// a provider app's own route has. One answer, so a probe learns nothing of which.
var errNoConnectorEvents = notFound("this connector takes no events here")

// errUnsigned is the answer to a request the verifier did not prove the provider sent.
var errUnsigned = unauthenticated("the request is not signed by the provider")

// EventSecretLookup finds the secret a connector's events are verified with on a connector's
// route. found is false when this deployment has none. The route of a provider app (T38)
// finds the customer's own secret by its path instead, so it needs no lookup of this kind.
type EventSecretLookup func(m core.Manifest) (secret []byte, found bool)

// ConnectorEventSecrets reads the operator's signing secret from the environment, as
// <client.env>_MCP_SIGNING_SECRET, for a manifest whose channel.verifier.secret is operator.
// A provider_app secret is a customer's, which this route has no customer to read for.
func ConnectorEventSecrets(getenv func(string) string) EventSecretLookup {
	return func(m core.Manifest) ([]byte, bool) {
		if m.Channel == nil || m.Channel.Verifier.Secret != core.SecretOperator || m.Client.Env == "" {
			return nil, false
		}
		secret := getenv(m.Client.Env + signingSecretSuffix)
		return []byte(secret), secret != ""
	}
}

// ChannelBridge takes the messages a verified inbound request carried, for the thread
// channels they belong to (T57). An error makes the endpoint answer 500, so the provider
// delivers again; a bridge drops a retried message by its ProviderMessageID.
type ChannelBridge interface {
	Deliver(ctx context.Context, messages []core.InboundMessage) error
}

// droppingBridge is the bridge until T57's lands: it logs that messages came and drops them.
// It logs no text and no author, which are a person's.
type droppingBridge struct {
	logger *slog.Logger
}

func (b droppingBridge) Deliver(_ context.Context, messages []core.InboundMessage) error {
	if len(messages) > 0 {
		b.logger.Info("dropped inbound connector messages: no channel bridge yet",
			"connector", messages[0].ConnectorID, "messages", len(messages))
	}
	return nil
}

// receiveConnectorEvent is the one inbound handler for a built-in connector's provider
// events. It is unauthenticated: the provider is no customer, and what proves a request is
// the provider's is its verifier, with the operator's secret. Nothing is acted on before
// the request verifies.
//
//	read the body, at most maxConnectorEventBytes   413 past it
//	the latest built-in definition and its channel  404 errNoConnectorEvents
//	verify with the operator's secret               401, nothing changed
//	a challenge                                     200 text/plain, the challenge
//	each signal -> the connections of its account -> Resolver.Revoke
//	the messages -> ChannelBridge.Deliver
//	                                                200, or 500 so the provider retries
func (s *Server) receiveConnectorEvent(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxConnectorEventBytes))
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		writeError(w, payloadTooLarge("an event is at most 256 KiB"))
		return
	}
	if err != nil {
		writeError(w, invalidRequest("the event could not be read"))
		return
	}
	if s.store == nil || s.connectorResolver == nil || s.eventSecrets == nil {
		writeError(w, errNoConnectorEvents)
		return
	}
	connectorID := r.PathValue("connector_id")
	definition, err := s.store.LatestBuiltinConnectorDefinition(r.Context(), connectorID)
	if errors.Is(err, store.ErrNoConnectorDefinition) {
		writeError(w, errNoConnectorEvents)
		return
	}
	if err != nil {
		writeFailure(w, r, err)
		return
	}
	manifest := definition.Manifest
	if manifest.Channel == nil {
		writeError(w, errNoConnectorEvents)
		return
	}
	verifier, registered := s.connectors.Verifiers[string(manifest.Channel.Verifier.Kind)]
	secret, found := s.eventSecrets(manifest)
	if !registered || !found {
		s.logger.Warn("refused a connector event this deployment cannot verify",
			"connector", connectorID, "verifier", manifest.Channel.Verifier.Kind,
			"verifier_registered", registered, "secret_set", found)
		writeError(w, errNoConnectorEvents)
		return
	}

	event, err := verifier.Verify(r, body, manifest, secret)
	if err != nil {
		// The reason names no signature or secret (hmacheader.ErrUnsigned, ErrStale).
		s.logger.Info("refused an unverified connector event", "connector", connectorID, "reason", err.Error())
		writeError(w, errUnsigned)
		return
	}
	if event.Challenge != "" {
		// Slack takes the challenge back as text/plain (url_verification,
		// https://docs.slack.dev/reference/events/url_verification, opened October 6, 2026).
		// nosniff keeps a browser from reading the echoed value as anything else.
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, event.Challenge)
		return
	}
	for _, signal := range event.Signals {
		if err := s.revokeAccount(r.Context(), manifest, signal); err != nil {
			writeFailure(w, r, err)
			return
		}
	}
	if len(event.Messages) > 0 {
		if err := s.channelBridge.Deliver(r.Context(), event.Messages); err != nil {
			writeFailure(w, r, err)
			return
		}
	}
	w.WriteHeader(http.StatusOK)
}

// revokeAccount moves every connection of the account a signal names to
// needs_reauthorization, through the resolver, so the next Resolve on any router fails. Each
// identity part is matched where the manifest keeps it: an input, or a captured value.
func (s *Server) revokeAccount(ctx context.Context, m core.Manifest, signal core.Signal) error {
	inputs, metadata := map[string]string{}, map[string]string{}
	for name, value := range signal.Identity {
		if slices.ContainsFunc(m.Inputs, func(in core.Input) bool { return in.Name == name }) {
			inputs[name] = value
		} else {
			metadata[name] = value
		}
	}
	refs, err := s.store.ConnectionsByIdentity(ctx, m.ID, inputs, metadata)
	if err != nil {
		return err
	}
	for _, ref := range refs {
		// A connection deleted since the lookup has no grant left to end, so it counts as
		// revoked and the rest of the account's connections are still revoked on this
		// delivery. Revoke itself leaves one already needs_reauthorization as it is.
		err := s.connectorResolver.Revoke(ctx, ref, signal.Kind, signal.At)
		if err != nil && !errors.Is(err, store.ErrNoConnectorConnection) {
			return err
		}
	}
	if len(refs) > 0 {
		s.logger.Info("a provider ended connector grants", "connector", m.ID, "signal", signal.Kind, "connections", len(refs))
	}
	return nil
}
