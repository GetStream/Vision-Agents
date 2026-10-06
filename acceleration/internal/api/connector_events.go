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
// route (T38) is a second route into the same handling, receiveEvent.
const connectorEventsPath = "/v1/agents/connectors/events/"

// providerAppEventsPath is where a provider delivers the events of one customer's provider app,
// followed by the connector id and the app's id, such as a customer's Slack app's Request URL
// (T38, AI-878; channels.md on connectors/planning, «Integration modes: one Slack app for each
// customer»). The URL names the app, and with it the customer and the signing secret, so no
// lookup by account picks the customer. It names the connector too, because an app id is
// unique only among one connector's records (connector_oauth_clients_provider_app,
// 20261006195500).
const providerAppEventsPath = "/v1/connectors/events/"

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

// ChannelBridge moves messages between external threads and their thread channels (T57,
// internal/channelbridge). Deliver takes the messages a verified inbound request carried, for
// the provider app the request's route named: the zero record on a connector's own route,
// which names no customer. An error makes the endpoint answer 500, so the provider delivers
// again; a bridge drops a retried message by its ProviderMessageID. Reply takes the agent's
// reply in a thread channel linked to an external thread, which the message hook hands over,
// and sends it there; it answers at once, and a reply it cannot send is its own to report.
type ChannelBridge interface {
	Deliver(ctx context.Context, app store.ConnectorOAuthClient, messages []core.InboundMessage) error
	Reply(ctx context.Context, thread store.ChannelThread, text string)
}

// droppingBridge is the bridge of a deployment without one: it logs that messages came and
// drops them, and sends no reply. It logs no text and no author, which are a person's.
type droppingBridge struct {
	logger *slog.Logger
}

func (b droppingBridge) Deliver(_ context.Context, _ store.ConnectorOAuthClient, messages []core.InboundMessage) error {
	if len(messages) > 0 {
		b.logger.Info("dropped inbound connector messages: no channel bridge",
			"connector", messages[0].ConnectorID, "messages", len(messages))
	}
	return nil
}

func (b droppingBridge) Reply(_ context.Context, thread store.ChannelThread, _ string) {
	b.logger.Info("dropped a reply to an external thread: no channel bridge",
		"connector", thread.ConnectorID, "channel", thread.ChannelID)
}

// receiveConnectorEvent is the inbound handler for a built-in connector's provider events,
// verified with the operator's secret. It is unauthenticated: the provider is no customer, and
// what proves a request is the provider's is its verifier. Nothing is acted on before the
// request verifies.
//
//	read the body, at most maxConnectorEventBytes   413 past it
//	the latest built-in definition and its channel  404 errNoConnectorEvents
//	verify with the operator's secret               401, nothing changed
//	then receiveEvent's steps, for every customer
func (s *Server) receiveConnectorEvent(w http.ResponseWriter, r *http.Request) {
	body, ok := readConnectorEvent(w, r)
	if !ok {
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
	secret, found := s.eventSecrets(manifest)
	if !found {
		s.logger.Warn("refused a connector event this deployment has no secret for", "connector", connectorID)
		writeError(w, errNoConnectorEvents)
		return
	}
	s.receiveEvent(w, r, body, manifest, secret, store.ConnectorOAuthClient{})
}

// receiveProviderAppEvent is the inbound handler for the events of one customer's provider
// app (T38), verified with that app's own signing secret, which api.ProviderApp opens. A
// request signed with another app's secret fails the verifier, so a customer's events act only
// on that customer's connections and threads.
//
//	read the body, at most maxConnectorEventBytes   413 past it
//	the provider app's record and signing secret    404 errNoConnectorEvents
//	the customer's latest definition and channel,
//	  verified with the provider app's secret       404 errNoConnectorEvents
//	verify with the app's secret                    401, nothing changed
//	then receiveEvent's steps, for the app's customer alone
func (s *Server) receiveProviderAppEvent(w http.ResponseWriter, r *http.Request) {
	body, ok := readConnectorEvent(w, r)
	if !ok {
		return
	}
	if s.store == nil || s.connectorResolver == nil || s.connectorSecrets == nil {
		writeError(w, errNoConnectorEvents)
		return
	}
	connectorID, appID := r.PathValue("connector_id"), r.PathValue("provider_app_id")
	app, secret, err := ProviderApp(r.Context(), s.store, s.connectorSecrets, connectorID, appID)
	if errors.Is(err, store.ErrNoConnectorOAuthClient) {
		writeError(w, errNoConnectorEvents)
		return
	}
	if err != nil {
		writeFailure(w, r, err)
		return
	}
	definition, err := s.store.LatestConnectorDefinition(r.Context(), app.CustomerID, connectorID)
	if errors.Is(err, store.ErrNoConnectorDefinition) {
		writeError(w, errNoConnectorEvents)
		return
	}
	if err != nil {
		writeFailure(w, r, err)
		return
	}
	manifest := definition.Manifest
	if manifest.Channel == nil || manifest.Channel.Verifier.Secret != core.SecretProviderApp {
		writeError(w, errNoConnectorEvents)
		return
	}
	s.receiveEvent(w, r, body, manifest, []byte(secret), app)
}

// readConnectorEvent reads an event's body, at most maxConnectorEventBytes. false is an
// answer already written.
func readConnectorEvent(w http.ResponseWriter, r *http.Request) ([]byte, bool) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxConnectorEventBytes))
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		writeError(w, payloadTooLarge("an event is at most 256 KiB"))
		return nil, false
	}
	if err != nil {
		writeError(w, invalidRequest("the event could not be read"))
		return nil, false
	}
	return body, true
}

// receiveEvent verifies one event with secret and acts on it: the steps both routes share.
// app is the provider app the route named, or the zero record on a connector's route, where a
// signal acts on every customer's connections of the account, since the operator's app serves
// them all.
//
//	the manifest's verifier                         404 errNoConnectorEvents when not registered
//	verify                                          401, nothing changed
//	a challenge                                     200 text/plain, the challenge
//	each signal -> the connections of its account -> Resolver.Revoke, with when it ended
//	the messages -> ChannelBridge.Deliver, with app
//	                                                200, or 500 so the provider retries
func (s *Server) receiveEvent(w http.ResponseWriter, r *http.Request, body []byte, manifest core.Manifest, secret []byte, app store.ConnectorOAuthClient) {
	verifier, registered := s.connectors.Verifiers[string(manifest.Channel.Verifier.Kind)]
	if !registered {
		s.logger.Warn("refused a connector event this deployment cannot verify",
			"connector", manifest.ID, "verifier", manifest.Channel.Verifier.Kind)
		writeError(w, errNoConnectorEvents)
		return
	}
	event, err := verifier.Verify(r, body, manifest, secret)
	if err != nil {
		// The reason names no signature or secret (hmacheader.ErrUnsigned, ErrStale).
		s.logger.Info("refused an unverified connector event", "connector", manifest.ID, "reason", err.Error())
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
		if err := s.revokeAccount(r.Context(), manifest, signal, app.CustomerID); err != nil {
			writeFailure(w, r, err)
			return
		}
	}
	if len(event.Messages) > 0 {
		if err := s.channelBridge.Deliver(r.Context(), app, event.Messages); err != nil {
			writeFailure(w, r, err)
			return
		}
	}
	w.WriteHeader(http.StatusOK)
}

// revokeAccount moves every connection of the account a signal names to
// needs_reauthorization, through the resolver, so the next Resolve on any router fails. Each
// identity part is matched where the manifest keeps it: an input, or a captured value. A
// non-empty customerID keeps it to that customer's connections: a provider app's event is
// signed with that customer's secret, and says nothing about another customer's grants in
// the same account. A connection a consent connected after the signal's time is left alone
// (core.Resolver.Revoke): a delivery Slack retried after the account reconnected.
func (s *Server) revokeAccount(ctx context.Context, m core.Manifest, signal core.Signal, customerID string) error {
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
	if customerID != "" {
		refs = slices.DeleteFunc(refs, func(ref core.ConnectionRef) bool { return ref.CustomerID != customerID })
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
