package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/slackapps"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// configTokenMargin is how long before its expiry a configuration token is rotated rather
// than used. A PUT spends it on up to two Slack calls after the rotation check (create, then
// update), each bounded by the egress client's timeout (connectorHTTPTimeout, 10 s, in
// cmd/router), so a token with a minute left outlives them. Not a Slack value: Slack only
// recommends rotating «before it expires, rather than waiting for it to expire»
// (https://docs.slack.dev/app-manifests/configuring-apps-with-app-manifests#config-tokens).
const configTokenMargin = time.Minute

// operatorAppIDSuffix follows client.env in the variable that holds the operator's provider
// app id, beside <client.env>_MCP_CLIENT_ID, _SECRET and _MCP_SIGNING_SECRET, which belong to
// the same app. A new name, as signingSecretSuffix is.
const operatorAppIDSuffix = "_MCP_APP_ID"

const noManagedApp = "the app has no provider app the router created for this connector"

// ConnectorProviderApp is the customer's app at a connector's provider, as the router keeps
// it. Its secrets are never shown.
type ConnectorProviderApp struct {
	ConnectorID   string                            `json:"connector_id" readOnly:"true"`
	Registration  ConnectorClientRegistrationMethod `json:"registration" readOnly:"true" doc:"managed: the router created the app in the customer's workspace. operator: Stream's own app, which serves Stream's own agents."`
	ProviderAppID string                            `json:"provider_app_id" readOnly:"true" doc:"The provider's id for the app, such as a Slack app id."`
	ClientID      string                            `json:"client_id" readOnly:"true" doc:"The app's OAuth client, which every consent of the connector's connections uses."`
	CreatedAt     time.Time                         `json:"created_at" readOnly:"true"`
	UpdatedAt     time.Time                         `json:"updated_at" readOnly:"true"`
}

func (*ConnectorProviderApp) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The customer's app at a connector's provider, such as a Slack app. Its client " +
		"secret and signing secret are kept sealed and never returned."
	return schema
}

// ConnectorProviderAppRequest is what the customer decides about the app the router creates.
// The scopes and the events come from the connector.
type ConnectorProviderAppRequest struct {
	// Slack's display_information.name: «Maximum length is 35 characters»
	// (https://docs.slack.dev/reference/app-manifest).
	Name string `json:"name" minLength:"1" maxLength:"35" doc:"The app's name in the customer's workspace."`
	// The refresh token tooling.tokens.rotate takes
	// (https://docs.slack.dev/reference/methods/tooling.tokens.rotate).
	ConfigRefreshToken string `json:"config_refresh_token,omitempty" writeOnly:"true" doc:"The refresh token of an app configuration token a workspace admin generated in Slack's app settings. Required the first time; the router rotates it and keeps the result sealed. Sent again, it replaces the one kept. Never returned."`
	// settings.allowed_ip_address_ranges: «Maximum 10 items» (https://docs.slack.dev/reference/app-manifest).
	AllowedIPAddressRanges []string `json:"allowed_ip_address_ranges,omitempty" maxItems:"10" doc:"IP addresses or CIDR ranges the app's tokens work from, at most 10. Left out, they work from anywhere."`
}

func (*ConnectorProviderAppRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The app the router creates and keeps in the customer's workspace. An unknown " +
		"field is refused rather than ignored."
	schema.AdditionalProperties = false
	return schema
}

type providerAppRequest struct {
	ID   string `path:"id" doc:"The connector, such as slack."`
	Body ConnectorProviderAppRequest
}

type deleteProviderAppRequest struct {
	ID string `path:"id" doc:"The connector, such as slack."`
}

type operatorProviderAppRequest struct {
	CustomerID string `path:"customer_id" doc:"The customer Stream's app serves."`
	ID         string `path:"id" doc:"The built-in connector, such as slack."`
}

type providerAppResponse struct {
	Status int
	Body   ConnectorProviderApp
}

// OperatorProviderApp is the operator's own provider app for one connector, as this
// deployment's environment holds it. Its secrets never print.
type OperatorProviderApp struct {
	AppID         string
	ClientID      string
	ClientSecret  string
	SigningSecret string
}

// String names the app and hides both secrets.
func (a OperatorProviderApp) String() string {
	return fmt.Sprintf("api.OperatorProviderApp{AppID: %s, ClientID: %s, ClientSecret: [redacted], SigningSecret: [redacted]}", a.AppID, a.ClientID)
}

// GoString is String, so %#v hides the secrets too.
func (a OperatorProviderApp) GoString() string { return a.String() }

// LogValue keeps the secrets out of slog.
func (a OperatorProviderApp) LogValue() slog.Value {
	return slog.GroupValue(slog.String("app_id", a.AppID), slog.String("client_id", a.ClientID))
}

// OperatorAppLookup finds the operator's provider app for a connector. found is false when
// this deployment lacks any part of it.
type OperatorAppLookup func(m core.Manifest) (app OperatorProviderApp, found bool)

// ConnectorOperatorApps reads the operator's provider app from the environment, as
// <client.env>_MCP_APP_ID, _MCP_CLIENT_ID, _MCP_CLIENT_SECRET and _MCP_SIGNING_SECRET: the
// client the router already reads there (oauth2code.EnvClients), the secret its events are
// verified with (ConnectorEventSecrets) and the id of the app all three belong to.
func ConnectorOperatorApps(getenv func(string) string) OperatorAppLookup {
	return func(m core.Manifest) (OperatorProviderApp, bool) {
		if m.Client.Env == "" {
			return OperatorProviderApp{}, false
		}
		app := OperatorProviderApp{
			AppID:         getenv(m.Client.Env + operatorAppIDSuffix),
			ClientID:      getenv(m.Client.Env + "_MCP_CLIENT_ID"),
			ClientSecret:  getenv(m.Client.Env + "_MCP_CLIENT_SECRET"),
			SigningSecret: getenv(m.Client.Env + signingSecretSuffix),
		}
		return app, app.AppID != "" && app.ClientID != "" && app.ClientSecret != "" && app.SigningSecret != ""
	}
}

// registerConnectorProviderApps declares the operations on a customer's provider app: the
// app the router creates and deletes in the customer's workspace (managed, server-side only),
// and Stream's own app set for one customer by Stream staff (operator).
func (s *Server) registerConnectorProviderApps(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "setConnectorProviderApp",
		Method:      http.MethodPut,
		Path:        "/v1/agents/connectors/{id}/provider-app",
		Summary:     "Create or update the app the router keeps at the connector's provider",
		Description: "Creates the customer's own Slack app in its workspace with Slack's " +
			"apps.manifest.create, from the connector's scopes and events, the name given and " +
			"this router's callback and events URLs, with token rotation on. The app's client " +
			"is what every later consent of the connector's connections uses. It needs an app " +
			"configuration token's refresh token the first time, which a workspace admin " +
			"generates in Slack's app settings; the router rotates it before it expires and " +
			"keeps it sealed. Putting it again changes nothing at Slack but the app's manifest: " +
			"there is one app per customer and connector, never a second. A connector that does " +
			"not authorize at Slack, or whose client.registration does not list managed, " +
			"refuses it. No response carries a token or a secret.\n\n" +
			messageHookNote + "\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		// Huma describes the body of the default status (200) alone, so 201 names it here.
		Responses: map[string]*huma.Response{
			"200": {Description: "The app exists; its manifest was applied again"},
			"201": {Description: "The app was created", Content: map[string]*huma.MediaType{
				"application/json": {Schema: &huma.Schema{Ref: "#/components/schemas/ConnectorProviderApp"}},
			}},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound, http.StatusConflict, http.StatusTooManyRequests, http.StatusServiceUnavailable},
	}, s.setConnectorProviderApp)
	huma.Register(api, huma.Operation{
		OperationID:   "deleteConnectorProviderApp",
		Method:        http.MethodDelete,
		Path:          "/v1/agents/connectors/{id}/provider-app",
		Summary:       "Delete the app the router keeps at the connector's provider",
		DefaultStatus: http.StatusNoContent,
		Description: "Deletes the customer's Slack app with Slack's apps.manifest.delete, then its " +
			"record and the configuration token the router kept. An app already deleted in " +
			"Slack is removed here too. Connections consented with it need a reconnect.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"204": {Description: "The app is deleted"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound, http.StatusConflict, http.StatusTooManyRequests, http.StatusServiceUnavailable},
	}, s.deleteConnectorProviderApp)
	staffErrors := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound, http.StatusConflict}
	huma.Register(api, huma.Operation{
		OperationID: "setOperatorProviderApp",
		Method:      http.MethodPut,
		Path:        "/v1/ops/customers/{customer_id}/connectors/{id}/provider-app",
		Summary:     "Make Stream's own app a customer's provider app",
		Description: "Records this deployment's own app for a built-in connector, as its environment " +
			"holds it (<client.env>_MCP_APP_ID, _MCP_CLIENT_ID, _MCP_CLIENT_SECRET and " +
			"_MCP_SIGNING_SECRET), as the customer's provider app, so its events reach that " +
			"customer. One customer per app: another customer's record of it is a conflict.\n\n" +
			messageHookNote + "\n\n" +
			"Stream staff only: it needs the ops key.",
		Security: opsSecurity,
		Responses: map[string]*huma.Response{
			"200": {Description: "The record was replaced"},
			"201": {Description: "The record was stored", Content: map[string]*huma.MediaType{
				"application/json": {Schema: &huma.Schema{Ref: "#/components/schemas/ConnectorProviderApp"}},
			}},
		},
		Errors: append(slices.Clone(staffErrors), http.StatusServiceUnavailable),
	}, s.setOperatorProviderApp)
	huma.Register(api, huma.Operation{
		OperationID:   "deleteOperatorProviderApp",
		Method:        http.MethodDelete,
		Path:          "/v1/ops/customers/{customer_id}/connectors/{id}/provider-app",
		Summary:       "Stop Stream's own app being a customer's provider app",
		DefaultStatus: http.StatusNoContent,
		Description:   "Removes the record setOperatorProviderApp made. The app itself is Stream's and stays.\n\nStream staff only: it needs the ops key.",
		Security:      opsSecurity,
		Responses:     map[string]*huma.Response{"204": {Description: "The record is removed"}},
		Errors:        staffErrors,
	}, s.deleteOperatorProviderApp)
}

// setConnectorProviderApp creates the customer's managed app, or applies its manifest again.
// Everything that spends the configuration token runs under the provider app lock, so of two
// PUTs for one customer the second finds the app the first created.
//
//	the connector authorizes at Slack and lists managed     400 otherwise
//	lock (customer, connector)
//	  configuration token: the one sent, rotated; else the
//	  one kept, rotated when it expires within the margin
//	  a managed record   -> apps.manifest.update            200
//	  none -> apps.manifest.create, no request URL
//	       -> record: client, sealed secrets, app id, pin
//	       -> apps.manifest.update with the request URL     201
func (s *Server) setConnectorProviderApp(ctx context.Context, request *providerAppRequest) (*providerAppResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConnectors
	}
	if s.connectorSecrets == nil || s.slackApps == nil {
		return nil, notConfigured("provider apps cannot be created: connectors are not enabled on this deployment")
	}
	public := strings.TrimRight(s.publicURL, "/")
	if !strings.HasPrefix(public, "https://") {
		return nil, notConfigured("provider apps cannot be created: ROUTER_PUBLIC_URL is not an https URL, and Slack sends consents and events only to one")
	}
	manifest, err := s.managedConnector(ctx, customerID, request.ID)
	if err != nil {
		return nil, err
	}
	template := slackapps.Template{
		Name:                   request.Body.Name,
		RedirectURL:            connectorCallbackURL(public),
		AllowedIPAddressRanges: request.Body.AllowedIPAddressRanges,
	}
	initial, err := slackapps.ManifestFor(manifest, template)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	// The pin of a new record, read before anything is created at Slack.
	var pin int64
	if s.stream != nil {
		if pin, err = s.stream.Pin(ctx, customerID); err != nil {
			return nil, err
		}
	}

	var answer *providerAppResponse
	var pinned int64
	err = s.store.WithConnectorProviderAppLock(ctx, customerID, manifest.ID, func() error {
		token, err := s.configToken(ctx, customerID, manifest.ID, request.Body.ConfigRefreshToken)
		if err != nil {
			return err
		}
		existing, err := s.store.ConnectorOAuthClient(ctx, customerID, manifest.ID)
		switch {
		case err == nil && existing.Registration != core.ClientManaged:
			return conflict(fmt.Sprintf("the app's OAuth client for %s is of registration %s, not one the router created; remove it first", manifest.ID, existing.Registration))
		case err == nil:
			if err := s.applyProviderAppManifest(ctx, manifest, template, token, existing.ProviderAppID); err != nil {
				return err
			}
			answer, pinned = &providerAppResponse{Status: http.StatusOK, Body: providerAppOf(existing)}, existing.StreamAppPK
			return nil
		case !errors.Is(err, store.ErrNoConnectorOAuthClient):
			return err
		}

		created, err := s.slackApps.Create(ctx, token, initial)
		if err != nil {
			return slackFailure(err)
		}
		record, err := s.managedRecord(customerID, manifest.ID, pin, created)
		if err != nil {
			return s.deleteOrphan(ctx, token, created.AppID, err)
		}
		// Detached: an app Slack made is recorded, or deleted below, whether or not the caller
		// is still waiting.
		_, err = s.store.PutConnectorOAuthClient(context.WithoutCancel(ctx), record)
		if err != nil {
			return s.deleteOrphan(ctx, token, created.AppID, providerAppPutFailure(err))
		}
		if err := s.applyProviderAppManifest(ctx, manifest, template, token, created.AppID); err != nil {
			return err
		}
		answer, pinned = &providerAppResponse{Status: http.StatusCreated, Body: providerAppOf(*record)}, record.StreamAppPK
		return nil
	})
	if err != nil {
		return nil, err
	}
	if err := s.pointMessageHook(ctx, customerID, pinned); err != nil {
		return nil, err
	}
	return answer, nil
}

// deleteConnectorProviderApp deletes the customer's managed app at Slack, then its record and
// its configuration token.
func (s *Server) deleteConnectorProviderApp(ctx context.Context, request *deleteProviderAppRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConnectors
	}
	if s.connectorSecrets == nil || s.slackApps == nil {
		return nil, notConfigured("provider apps cannot be deleted: connectors are not enabled on this deployment")
	}
	err := s.store.WithConnectorProviderAppLock(ctx, customerID, request.ID, func() error {
		record, err := s.store.ConnectorOAuthClient(ctx, customerID, request.ID)
		if errors.Is(err, store.ErrNoConnectorOAuthClient) || err == nil && record.Registration != core.ClientManaged {
			return notFound(noManagedApp)
		}
		if err != nil {
			return err
		}
		token, err := s.configToken(ctx, customerID, request.ID, "")
		if err != nil {
			return err
		}
		// An app deleted in Slack's own settings is already what this asks for.
		if err := s.slackApps.Delete(ctx, token, record.ProviderAppID); err != nil && !errors.Is(err, slackapps.ErrAppNotFound) {
			return slackFailure(err)
		}
		// Detached: the app is gone at Slack, so its record goes whether or not the caller
		// is still waiting.
		detached := context.WithoutCancel(ctx)
		if err := s.store.DeleteConnectorOAuthClient(detached, customerID, request.ID, core.ClientManaged); err != nil {
			return err
		}
		return s.store.DeleteConnectorConfigToken(detached, customerID, request.ID)
	})
	if err != nil {
		return nil, err
	}
	return nil, nil
}

// setOperatorProviderApp records this deployment's own app as one customer's provider app.
func (s *Server) setOperatorProviderApp(ctx context.Context, request *operatorProviderAppRequest) (*providerAppResponse, error) {
	if s.store == nil {
		return nil, errNoConnectors
	}
	if s.connectorSecrets == nil || s.operatorApps == nil {
		return nil, notConfigured("operator apps cannot be set: connectors are not enabled on this deployment")
	}
	definition, err := s.store.LatestBuiltinConnectorDefinition(ctx, request.ID)
	if errors.Is(err, store.ErrNoConnectorDefinition) {
		return nil, notFound("no such built-in connector")
	}
	if err != nil {
		return nil, err
	}
	manifest := definition.Manifest
	if !slices.Contains(manifest.Client.Registration, core.ClientOperator) {
		return nil, invalidRequest(fmt.Sprintf("%s takes no operator's app: its client.registration is %v, which does not list operator",
			manifest.ID, manifest.Client.Registration))
	}
	app, found := s.operatorApps(manifest)
	if !found {
		return nil, notConfigured(fmt.Sprintf("this deployment has no app of its own for %s: set %s%s, %s_MCP_CLIENT_ID, %s_MCP_CLIENT_SECRET and %s%s",
			manifest.ID, manifest.Client.Env, operatorAppIDSuffix, manifest.Client.Env, manifest.Client.Env, manifest.Client.Env, signingSecretSuffix))
	}
	record := &store.ConnectorOAuthClient{
		CustomerID:    request.CustomerID,
		ConnectorID:   manifest.ID,
		Registration:  core.ClientOperator,
		ClientID:      app.ClientID,
		ProviderAppID: app.AppID,
	}
	if s.stream != nil {
		if record.StreamAppPK, err = s.stream.Pin(ctx, request.CustomerID); err != nil {
			return nil, err
		}
	}
	if err := s.sealProviderApp(record, app.ClientSecret, app.SigningSecret); err != nil {
		return nil, err
	}
	created, err := s.store.PutConnectorOAuthClient(ctx, record)
	if err != nil {
		return nil, providerAppPutFailure(err)
	}
	if err := s.pointMessageHook(ctx, request.CustomerID, record.StreamAppPK); err != nil {
		return nil, err
	}
	status := http.StatusOK
	if created {
		status = http.StatusCreated
	}
	return &providerAppResponse{Status: status, Body: providerAppOf(*record)}, nil
}

// deleteOperatorProviderApp removes a customer's record of this deployment's own app.
func (s *Server) deleteOperatorProviderApp(ctx context.Context, request *operatorProviderAppRequest) (*struct{}, error) {
	if s.store == nil {
		return nil, errNoConnectors
	}
	err := s.store.DeleteConnectorOAuthClient(ctx, request.CustomerID, request.ID, core.ClientOperator)
	if errors.Is(err, store.ErrNoConnectorOAuthClient) {
		return nil, notFound("the customer has no record of this deployment's app for this connector")
	}
	if err != nil {
		return nil, err
	}
	return nil, nil
}

// messageHookNote is what the PUTs that make a provider app (both provider app PUTs and the
// oauth-client PUT) say of the message hook they point.
const messageHookNote = "When the provider app is pinned to a Stream app the customer registered, the " +
	"router then points that app's message hook at itself, at ROUTER_PUBLIC_URL" + chat.MessageHookPath +
	"/{stream app id}: it adds the hook, or updates the one already there, so the messages written " +
	"in the app's thread channels reach the router. A router without ROUTER_PUBLIC_URL points " +
	"none and logs a warning. When Stream refuses, the provider app is kept, the answer is a 503, " +
	"and putting it again points the hook again."

// pointMessageHook makes the customer's Stream app the provider app is pinned to deliver new
// messages to this router (T48, AI-887): ROUTER_PUBLIC_URL, chat.MessageHookPath, then the
// app's id, the route receiveMessageEvent serves and checks with that app's own keys
// (verifyHook), as `router phone hooks -url <public> -app <id>` points it by hand. Only the
// PUTs that make a provider app call it: both provider app PUTs and the oauth-client PUT of a
// connector whose channel reads the app's events (readsProviderAppEvents). They are served
// only with connectors on: cmd/router sets ConnectorSecrets, SlackApps and
// OperatorProviderApps only then.
//
// A pin of zero, or the deployment's own app, names no app of the customer's: the
// deployment's hooks are the operator's to point, so nothing is asked of Stream. A router
// without ROUTER_PUBLIC_URL has no URL to point at; it warns and the PUT goes on. Stream
// refusing fails the PUT, as it fails `router phone hooks` (cmd/phone/main.go), with the
// record kept: a PUT again points the hook again. Two routers pointing one app converge: the
// hook is matched by its URL (chat.PointMessageHook).
func (s *Server) pointMessageHook(ctx context.Context, customerID string, pin int64) error {
	if s.stream == nil || pin == 0 || pin == s.stream.DeploymentApp() {
		return nil
	}
	public := strings.TrimRight(s.publicURL, "/")
	if public == "" {
		s.logger.Warn("no message hook was pointed at this router: ROUTER_PUBLIC_URL is not set",
			"customer", customerID, "stream_app", pin)
		return nil
	}
	bound, err := s.stream.ForApp(ctx, customerID, pin)
	if err == nil {
		_, err = chat.StreamOf(bound.Client).PointMessageHook(ctx, public+chat.MessageHookPath+"/"+strconv.FormatInt(pin, 10))
	}
	if err != nil {
		s.logger.Error("could not point the message hook of a customer's Stream app", "customer", customerID, "stream_app", pin, "error", err)
		return unavailable("the provider app is kept, but its Stream app's message hook could not be pointed at this router: put it again")
	}
	return nil
}

// WarnWithoutMessageHooks says once, at startup, for each Stream app a provider app is pinned
// to, when that app sends its new messages nowhere this router answers them: the messages the
// bridge writes into its thread channels are then stored and never answered, and nothing else
// on this router says why (AI-990 F19 for the deployment's own app; this is app mode's).
//
// It runs only in app mode with connectors on and ROUTER_PUBLIC_URL set: in deployment mode
// every provider app is pinned to the deployment's own app, which cmd/router's
// warnWithoutMessageHook checks. It only reads, one GET of each app's settings, with the
// check the startup one uses (chat.Stream.DeliversMessagesTo): pointMessageHook points the
// hook when a provider app is put, and an app pointed elsewhere since, or before that PUT
// pointed it, is what is left to see. A record of a connector whose channel does not read the
// app's events (readsProviderAppEvents) is not checked, nor one pinned to the deployment's
// own app, which pointMessageHook leaves to the operator (PinnedProviderApps has no pin of
// zero). A check that fails is a warning, never a reason not to start.
func (s *Server) WarnWithoutMessageHooks(ctx context.Context) {
	public := strings.TrimRight(s.publicURL, "/")
	if s.store == nil || s.connectorSecrets == nil || s.stream == nil || !s.stream.PerApp() || public == "" {
		return
	}
	pinned, err := s.store.PinnedProviderApps(ctx)
	if err != nil {
		s.logger.Warn("stream: could not list the provider apps whose message hooks to check", "error", err)
		return
	}
	hook := public + chat.MessageHookPath
	checked := map[int64]bool{}
	for _, app := range pinned {
		// The deployment's own app's hooks are the operator's, as pointMessageHook leaves them.
		if checked[app.StreamAppPK] || app.StreamAppPK == s.stream.DeploymentApp() {
			continue
		}
		definition, err := s.store.LatestConnectorDefinition(ctx, app.CustomerID, app.ConnectorID)
		if err != nil || !readsProviderAppEvents(definition.Manifest) {
			continue
		}
		checked[app.StreamAppPK] = true
		bound, err := s.stream.ForApp(ctx, app.CustomerID, app.StreamAppPK)
		pointed := false
		if err == nil {
			pointed, err = chat.StreamOf(bound.Client).DeliversMessagesTo(ctx, hook)
		}
		switch {
		case err != nil:
			s.logger.Warn("stream: could not read a provider app's Stream app hooks, so whether its messages reach this router is unknown",
				"customer", app.CustomerID, "connector", app.ConnectorID, "provider_app", app.ProviderAppID,
				"stream_app", app.StreamAppPK, "hook", hook, "error", err)
		case !pointed:
			s.logger.Warn("stream: a provider app's Stream app sends no new message to this router, so a message written "+
				"in its thread channels is stored and never answered; put the provider app again to point the hook",
				"customer", app.CustomerID, "connector", app.ConnectorID, "provider_app", app.ProviderAppID,
				"stream_app", app.StreamAppPK, "hook", hook+"/"+strconv.FormatInt(app.StreamAppPK, 10))
		}
	}
}

// managedConnector is the customer's latest definition of a connector the router can create a
// Slack app for, or why it cannot.
func (s *Server) managedConnector(ctx context.Context, customerID, id string) (core.Manifest, error) {
	definition, err := s.store.LatestConnectorDefinition(ctx, customerID, id)
	if errors.Is(err, store.ErrNoConnectorDefinition) {
		return core.Manifest{}, notFound("no such connector")
	}
	if err != nil {
		return core.Manifest{}, err
	}
	manifest := definition.Manifest
	if !slackapps.Serves(manifest) {
		return core.Manifest{}, invalidRequest(fmt.Sprintf("%s does not authorize at Slack, so the router creates no app for it", manifest.ID))
	}
	// The store refuses such a record too (store.PutConnectorOAuthClient); checked here so no
	// app is created at Slack that could never be recorded.
	if !slices.Contains(manifest.Client.Registration, core.ClientManaged) {
		return core.Manifest{}, invalidRequest(fmt.Sprintf("%s takes no app the router creates: its client.registration is %v, which does not list managed",
			manifest.ID, manifest.Client.Registration))
	}
	// The app's events reach the provider app's own route (receiveProviderAppEvent), which
	// verifies them with the app's signing secret only for a channel whose secret is
	// provider_app. Any other channel's events would never be read there.
	if manifest.Channel != nil && manifest.Channel.Verifier.Secret != core.SecretProviderApp {
		return core.Manifest{}, invalidRequest(fmt.Sprintf("%s verifies its events with the %s secret, not the app's own (provider_app), so an app the router creates could not deliver them",
			manifest.ID, manifest.Channel.Verifier.Secret))
	}
	return manifest, nil
}

// managedRecord is the record of an app Slack just created for the customer, its secrets
// sealed.
func (s *Server) managedRecord(customerID, connectorID string, pin int64, created slackapps.Credentials) (*store.ConnectorOAuthClient, error) {
	record := &store.ConnectorOAuthClient{
		CustomerID:    customerID,
		ConnectorID:   connectorID,
		Registration:  core.ClientManaged,
		ClientID:      created.ClientID,
		ProviderAppID: created.AppID,
		StreamAppPK:   pin,
	}
	return record, s.sealProviderApp(record, created.ClientSecret, created.SigningSecret)
}

// sealProviderApp seals a provider app's client secret and signing secret onto its record,
// bound to the record as openOAuthClient and ProviderApp open them.
func (s *Server) sealProviderApp(record *store.ConnectorOAuthClient, clientSecret, signingSecret string) error {
	var err error
	record.SecretSealed, err = s.connectorSecrets.SealWithAAD(clientSecret, oauthClientAAD(record.CustomerID, record.ConnectorID))
	if err != nil {
		return err
	}
	record.SigningSecretSealed, err = s.connectorSecrets.SealWithAAD(signingSecret,
		providerAppAAD(record.CustomerID, record.ConnectorID, record.ProviderAppID))
	if err != nil {
		return err
	}
	record.KEKVersion, record.SigningKEKVersion = s.connectorSecrets.CurrentVersion(), s.connectorSecrets.CurrentVersion()
	return nil
}

// applyProviderAppManifest sets the app's manifest with its events URL, which names the app
// id apps.manifest.create returned: the connector's own scopes and events at its latest
// revision, and the template's name and ranges.
func (s *Server) applyProviderAppManifest(ctx context.Context, manifest core.Manifest, template slackapps.Template, token, appID string) error {
	// The route receiveProviderAppEvent serves, POST providerAppEventsPath{connector_id}/{provider_app_id}.
	// A connector without a channel reads no events, so its app gets no request URL.
	if manifest.Channel != nil {
		template.RequestURL = strings.TrimRight(s.publicURL, "/") + providerAppEventsPath + manifest.ID + "/" + appID
	}
	applied, err := slackapps.ManifestFor(manifest, template)
	if err != nil {
		return invalidRequest(err.Error())
	}
	err = s.slackApps.Update(ctx, token, appID, applied)
	if errors.Is(err, slackapps.ErrAppNotFound) {
		return conflict(fmt.Sprintf("Slack no longer has app %s: delete it here, then put it again to create a new one", appID))
	}
	if err != nil {
		return slackFailure(err)
	}
	return nil
}

// deleteOrphan deletes an app Slack created that could not be recorded, so no app is left in
// the customer's workspace that nothing here manages, and returns why it was not recorded. A
// delete that fails too is joined to it, with the app id, for whoever has to delete the app
// by hand.
func (s *Server) deleteOrphan(ctx context.Context, token, appID string, why error) error {
	if err := s.slackApps.Delete(context.WithoutCancel(ctx), token, appID); err != nil {
		return errors.Join(why, stack.Wrap(fmt.Errorf("api: Slack app %s was created but not recorded, and could not be deleted: %w", appID, err)))
	}
	return why
}

// configTokenBlob is what a configuration token's sealed blob holds.
type configTokenBlob struct {
	Token        string `json:"token"`
	RefreshToken string `json:"refresh_token"`
}

// configToken is a configuration token good for configTokenMargin more: sent, the refresh
// token the caller sent, rotated; else the one kept, rotated when it is due. A rotated token
// is saved before it is used. The caller holds the provider app lock, so no other router
// spends the same refresh token. A rotation and its save run detached from the caller: once
// Slack has rotated, the refresh token just spent may no longer work, so the new one is kept
// whether or not the caller is still waiting.
func (s *Server) configToken(ctx context.Context, customerID, connectorID, sent string) (string, error) {
	refresh := sent
	if refresh == "" {
		stored, err := s.store.ConnectorConfigToken(ctx, customerID, connectorID)
		if errors.Is(err, store.ErrNoConnectorConfigToken) {
			return "", invalidRequest("no app configuration token is kept for this connector: send config_refresh_token")
		}
		if err != nil {
			return "", err
		}
		raw, err := s.connectorSecrets.OpenWithAADVersion(stored.TokensSealed, configTokenAAD(customerID, connectorID), stored.KEKVersion)
		if err != nil {
			return "", stack.Wrap(fmt.Errorf("api: the configuration token of %s does not open: %w", connectorID, err))
		}
		var opened configTokenBlob
		if err := json.Unmarshal([]byte(raw), &opened); err != nil {
			return "", stack.Wrap(fmt.Errorf("api: the configuration token of %s is not one this router sealed: %w", connectorID, err))
		}
		if time.Now().Add(configTokenMargin).Before(stored.ExpiresAt) {
			return opened.Token, nil
		}
		refresh = opened.RefreshToken
	}
	detached := context.WithoutCancel(ctx)
	rotated, err := s.slackApps.Rotate(detached, refresh)
	if errors.Is(err, slackapps.ErrInvalidRefreshToken) {
		if sent != "" {
			return "", invalidRequest("Slack refused config_refresh_token: send the refresh token of an app configuration token generated in Slack's app settings")
		}
		return "", conflict("Slack refused the kept app configuration token: send a new config_refresh_token with a PUT")
	}
	if err != nil {
		return "", slackFailure(err)
	}
	blob, err := json.Marshal(configTokenBlob{Token: rotated.Token, RefreshToken: rotated.RefreshToken})
	if err != nil {
		return "", stack.Wrap(err)
	}
	sealed, err := s.connectorSecrets.SealWithAAD(string(blob), configTokenAAD(customerID, connectorID))
	if err != nil {
		return "", err
	}
	err = s.store.PutConnectorConfigToken(detached, &store.ConnectorConfigToken{
		CustomerID: customerID, ConnectorID: connectorID,
		TokensSealed: sealed, KEKVersion: s.connectorSecrets.CurrentVersion(), ExpiresAt: rotated.ExpiresAt,
	})
	if err != nil {
		return "", err
	}
	return rotated.Token, nil
}

// slackFailure is the answer to a Slack call that failed: its own code when Slack refused,
// the error as it is otherwise.
func slackFailure(err error) error {
	var refused *slackapps.Error
	switch {
	case errors.Is(err, slackapps.ErrRateLimited):
		// The app manifest methods are Tier 1, «1+ per minute»
		// (https://docs.slack.dev/apis/web-api/rate-limits).
		return rateLimited("Slack is rate limiting its app manifest API, which allows about one call a minute: try again in a minute")
	case errors.Is(err, slackapps.ErrTokenExpired), errors.Is(err, slackapps.ErrInvalidAuth):
		return conflict("Slack refused the kept app configuration token: send a new config_refresh_token with a PUT")
	case errors.As(err, &refused):
		return unavailable(fmt.Sprintf("Slack refused %s: %s", refused.Method, refused.Code))
	}
	return err
}

// providerAppPutFailure is the answer to a provider app record the store refused.
func providerAppPutFailure(err error) error {
	switch {
	case errors.Is(err, store.ErrOAuthClientRegistrationNotListed):
		return invalidRequest(err.Error())
	// 409: the request conflicts with the state of the resource (RFC 9110 section 15.5.10).
	case errors.Is(err, store.ErrOAuthClientRegistration):
		return conflict("the customer's OAuth client for this connector is of another registration; remove it first")
	case errors.Is(err, store.ErrProviderAppTaken):
		return errProviderAppTaken
	}
	return err
}

// configTokenAAD binds a sealed configuration token to its customer and connector, the row's
// key, so a blob copied onto another row does not open. Laid out as oauthClientAAD, with its
// own prefix. v1 changes with the layout.
func configTokenAAD(customerID, connectorID string) []byte {
	return fmt.Appendf(nil, "accelerate:connector-config-token:v1:%d:%s:%d:%s",
		len(customerID), customerID, len(connectorID), connectorID)
}

func providerAppOf(record store.ConnectorOAuthClient) ConnectorProviderApp {
	return ConnectorProviderApp{
		ConnectorID:   record.ConnectorID,
		Registration:  ConnectorClientRegistrationMethod(record.Registration),
		ProviderAppID: record.ProviderAppID,
		ClientID:      record.ClientID,
		CreatedAt:     record.CreatedAt,
		UpdatedAt:     record.UpdatedAt,
	}
}
