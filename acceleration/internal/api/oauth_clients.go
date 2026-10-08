package api

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"slices"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const noOAuthClient = "the app has no OAuth client of its own for this connector"

// errProviderAppTaken is the answer to a record naming a provider app another customer's
// record of the connector already names (store.ErrProviderAppTaken). 409: the request
// conflicts with the state of the resource (RFC 9110 section 15.5.10).
var errProviderAppTaken = conflict("another customer's record already names this provider app: an app serves one customer")

// ConnectorOAuthClient is the OAuth client an app registered with a provider itself, as the
// router keeps it for one connector. Its secret is never shown.
type ConnectorOAuthClient struct {
	ConnectorID  string                            `json:"connector_id" readOnly:"true"`
	Registration ConnectorClientRegistrationMethod `json:"registration" readOnly:"true" doc:"customer: the app's own client, which every consent and refresh of the connector's connections then uses."`
	ClientID     string                            `json:"client_id"`
	AuthMethod   ConnectorOAuthClientAuthMethod    `json:"auth_method,omitempty"`
	// Left out when the record names no app, so the answer to a put without one is what it
	// was before provider apps.
	ProviderAppID string    `json:"provider_app_id,omitempty" doc:"The provider's id for the app the client belongs to, as put. Absent when none was put."`
	CreatedAt     time.Time `json:"created_at" readOnly:"true"`
	UpdatedAt     time.Time `json:"updated_at" readOnly:"true" doc:"When the client, its secrets or its method last changed."`
}

func (*ConnectorOAuthClient) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The OAuth client the app registered with a connector's provider itself. " +
		"The client secret and the signing secret are write-only: no response carries them."
	return schema
}

// ConnectorOAuthClientRequest is the app's own OAuth client for one connector.
//
// RFC 6749 Appendix A.1 and A.2 make client_id and client_secret *VSCHAR, %x20-7E, which the
// pattern holds them to. Section 2.2 leaves the client identifier's size undefined; 2048 is
// the bound CustomConnectorRequest.Endpoint already uses here, since a CIMD client_id is a
// URL, and is not measured, nor is the same bound on the secret.
//
// ProviderAppID is one path segment of the app's events URL, so it is held to the store's RFC
// 3986 section 2.3 unreserved characters (store.providerAppIDPattern). Slack publishes no
// grammar for an app id, only examples such as A012ABCD0A0
// (https://docs.slack.dev/reference/methods/apps.manifest.create), so no Slack-only pattern is
// applied. No grammar is shared by the providers' signing secrets either; the three this repo
// verifies are printable ASCII: Slack's is hex
// (https://docs.slack.dev/authentication/verifying-requests-from-slack), Linq's whsec_ and
// base64 (internal/channels/linq.go), Telnyx's a base64 Ed25519 public key
// (internal/channels/telnyx.go). Both are bounded at client_id's 2048, which is not measured
// for them either.
type ConnectorOAuthClientRequest struct {
	ClientID      string                         `json:"client_id" minLength:"1" maxLength:"2048" pattern:"^[ -~]+$" patternDescription:"printable ASCII, RFC 6749 Appendix A.1"`
	ClientSecret  string                         `json:"client_secret,omitempty" maxLength:"2048" pattern:"^[ -~]+$" patternDescription:"printable ASCII, RFC 6749 Appendix A.2" writeOnly:"true" doc:"Sealed at rest and never returned. Left out for a public client (auth_method none)."`
	AuthMethod    ConnectorOAuthClientAuthMethod `json:"auth_method,omitempty" doc:"Overrides the connector's own client.auth_method. Left out, the connector's applies, and failing that the consent picks: none without a secret, else client_secret_basic where the provider accepts it."`
	ProviderAppID string                         `json:"provider_app_id,omitempty" maxLength:"2048" pattern:"^[A-Za-z0-9._~-]+$" patternDescription:"RFC 3986 section 2.3 unreserved characters, not . or .." doc:"The provider's id for the app the client belongs to, such as a Slack app id (A012ABCD0A0). The app's events then reach POST /v1/connectors/events/{id}/{provider_app_id}. An app serves one customer: another customer's record naming it is a 409."`
	SigningSecret string                         `json:"signing_secret,omitempty" maxLength:"2048" pattern:"^[ -~]+$" patternDescription:"printable ASCII" writeOnly:"true" doc:"The secret the provider signs the app's events with, such as a Slack app's signing secret. Needs provider_app_id, and a connector whose events are verified with the app's own secret (channel.verifier.secret provider_app). Sealed at rest and never returned. Putting the client again without it removes it, as it does client_secret."`
}

func (*ConnectorOAuthClientRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The OAuth client the app registered with the connector's provider. An " +
		"unknown field is refused rather than ignored."
	schema.AdditionalProperties = false
	return schema
}

// ConnectorOAuthClientAuthMethod is how an app's own OAuth client authenticates at the token
// endpoint: the three token_endpoint_auth_method values RFC 7591 section 2 defines, which are
// the ones oauth2_code implements (supportedMethods in
// internal/connectors/schemes/oauth2code/client.go). private_key_jwt and tls_client_auth
// need a key or a certificate this record does not hold.
type ConnectorOAuthClientAuthMethod string

func (ConnectorOAuthClientAuthMethod) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ConnectorOAuthClientAuthMethod",
		"How the app's own OAuth client authenticates at the token endpoint (RFC 7591 section 2): "+
			"none for a public client, which has no secret, client_secret_basic or client_secret_post.",
		string(core.AuthNone), string(core.AuthClientSecretBasic), string(core.AuthClientSecretPost))
}

type oauthClientRequest struct {
	ID   string `path:"id" doc:"The connector, such as github or custom_crm."`
	Body ConnectorOAuthClientRequest
}

type deleteOAuthClientRequest struct {
	ID string `path:"id" doc:"The connector, such as github or custom_crm."`
}

type oauthClientResponse struct {
	Status int
	Body   ConnectorOAuthClient
}

// registerOAuthClients declares the operations on an app's own OAuth client. Both are
// server-side only: the client secret is the app's backend's to hold, as every connector
// operation is (registerConnectors).
func (s *Server) registerOAuthClients(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "setConnectorOAuthClient",
		Method:      http.MethodPut,
		Path:        "/v1/agents/connectors/{id}/oauth-client",
		Summary:     "Set the app's own OAuth client for a connector",
		Description: "Stores the OAuth client the app registered with the connector's provider, " +
			"for every consent and refresh of the app's connections to it. Putting it again " +
			"replaces it: a rotated secret is used from the next refresh of each connection. A " +
			"new client_id makes the connections consented with the old one need a reconnect, " +
			"since a refresh token is bound to the client it was issued to (RFC 6749 section 6). " +
			"A connector whose client.registration does not list customer refuses it. With " +
			"provider_app_id and signing_secret it is also the app's own provider app: the " +
			"provider's events for the app, posted to /v1/connectors/events/{id}/{provider_app_id}, " +
			"are verified with that secret and reach the app alone. Both secrets are sealed and " +
			"never returned.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		// Huma describes the body of the default status (200) alone, so 201 names it here.
		Responses: map[string]*huma.Response{
			"200": {Description: "The client was replaced"},
			"201": {Description: "The client was stored", Content: map[string]*huma.MediaType{
				"application/json": {Schema: &huma.Schema{Ref: "#/components/schemas/ConnectorOAuthClient"}},
			}},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound, http.StatusConflict},
	}, s.setConnectorOAuthClient)
	huma.Register(api, huma.Operation{
		OperationID:   "deleteConnectorOAuthClient",
		Method:        http.MethodDelete,
		Path:          "/v1/agents/connectors/{id}/oauth-client",
		Summary:       "Remove the app's own OAuth client for a connector",
		DefaultStatus: http.StatusNoContent,
		Description: "Drops the client and its secret. Connections consented with it stop " +
			"refreshing and need a reconnect with another client.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"204": {Description: "The client is removed"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.deleteConnectorOAuthClient)
}

// setConnectorOAuthClient stores the caller's own client for a connector that takes one.
func (s *Server) setConnectorOAuthClient(ctx context.Context, request *oauthClientRequest) (*oauthClientResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConnectors
	}
	if s.connectorSecrets == nil {
		return nil, notConfigured("OAuth clients cannot be stored: connectors are not enabled on this deployment, so there is no key to seal the secret with")
	}
	definition, err := s.store.LatestConnectorDefinition(ctx, customerID, request.ID)
	if errors.Is(err, store.ErrNoConnectorDefinition) {
		return nil, notFound("no such connector")
	}
	if err != nil {
		return nil, err
	}
	sent := request.Body
	if err := checkOAuthClient(definition.Manifest, sent); err != nil {
		return nil, invalidRequest(err.Error())
	}

	record := &store.ConnectorOAuthClient{
		CustomerID:    customerID,
		ConnectorID:   definition.ID,
		Registration:  core.ClientCustomer,
		ClientID:      sent.ClientID,
		AuthMethod:    core.ClientAuthMethod(sent.AuthMethod),
		ProviderAppID: sent.ProviderAppID,
	}
	// The app the customer acts in now is the record's pin when this put creates it; a put
	// that replaces the record keeps the pin it has (store.PutConnectorOAuthClient).
	if s.stream != nil {
		record.StreamAppPK, err = s.stream.Pin(ctx, customerID)
		if err != nil {
			return nil, err
		}
	}
	if sent.ClientSecret != "" {
		record.SecretSealed, err = s.connectorSecrets.SealWithAAD(sent.ClientSecret, oauthClientAAD(customerID, definition.ID))
		if err != nil {
			return nil, err
		}
		record.KEKVersion = s.connectorSecrets.CurrentVersion()
	}
	// Bound to the app it signs for, as sealProviderApp seals it and ProviderApp opens it.
	if sent.SigningSecret != "" {
		record.SigningSecretSealed, err = s.connectorSecrets.SealWithAAD(sent.SigningSecret,
			providerAppAAD(customerID, definition.ID, sent.ProviderAppID))
		if err != nil {
			return nil, err
		}
		record.SigningKEKVersion = s.connectorSecrets.CurrentVersion()
	}
	created, err := s.store.PutConnectorOAuthClient(ctx, record)
	// 409: the request conflicts with the state of the resource (RFC 9110 section 15.5.10).
	if errors.Is(err, store.ErrOAuthClientRegistration) {
		return nil, conflict("the connector's OAuth client for the app is one this deployment's operator registered or the router created, and it cannot be replaced through the API")
	}
	if errors.Is(err, store.ErrProviderAppTaken) {
		return nil, errProviderAppTaken
	}
	if err != nil {
		return nil, err
	}
	// RFC 9110 section 9.3.4: a PUT that creates the resource MUST answer 201, one that
	// replaces it 200 or 204.
	status := http.StatusOK
	if created {
		status = http.StatusCreated
	}
	return &oauthClientResponse{Status: status, Body: oauthClientOf(*record)}, nil
}

// deleteConnectorOAuthClient removes the caller's own client for a connector.
func (s *Server) deleteConnectorOAuthClient(ctx context.Context, request *deleteOAuthClientRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConnectors
	}
	err := s.store.DeleteConnectorOAuthClient(ctx, customerID, request.ID, core.ClientCustomer)
	if errors.Is(err, store.ErrNoConnectorOAuthClient) {
		return nil, notFound(noOAuthClient)
	}
	if err != nil {
		return nil, err
	}
	return nil, nil
}

// checkOAuthClient is why the connector cannot use the client sent, or nil. The method checked
// is the one a consent would use: the client's, else the manifest's (preregisteredMethod in
// oauth2code/client.go).
func checkOAuthClient(manifest core.Manifest, sent ConnectorOAuthClientRequest) error {
	if !slices.Contains(manifest.Client.Registration, core.ClientCustomer) {
		return stack.Wrap(fmt.Errorf("%s does not take an app's own OAuth client: its client.registration is %v, which does not list customer",
			manifest.ID, manifest.Client.Registration))
	}
	method := core.ClientAuthMethod(sent.AuthMethod)
	if method == "" {
		method = manifest.Client.AuthMethod
	}
	switch method {
	case core.AuthNone:
		// RFC 7591 section 2: none is «a public client ... and does not have a client secret».
		if sent.ClientSecret != "" {
			return stack.Wrap(errors.New("auth_method none is a public client, which has no secret (RFC 7591 section 2): leave client_secret out"))
		}
	case core.AuthClientSecretBasic, core.AuthClientSecretPost:
		if sent.ClientSecret == "" {
			return stack.Wrap(fmt.Errorf("auth_method %s needs client_secret", method))
		}
	case "":
	default:
		return stack.Wrap(fmt.Errorf("the connector's client.auth_method %s needs a key or a certificate this client cannot hold: "+
			"set auth_method to none, client_secret_basic or client_secret_post", method))
	}
	// RFC 3986 section 3.3: a dot segment would be removed from the events URL it is in. The
	// store refuses it too (store.checkOAuthClient); checked here so it is a 400.
	if sent.ProviderAppID == "." || sent.ProviderAppID == ".." {
		return stack.Wrap(errors.New("provider_app_id is a dot segment, which an events URL cannot carry (RFC 3986 section 3.3)"))
	}
	if sent.SigningSecret == "" {
		return nil
	}
	// The events route finds a signing secret by the app it signs for (ProviderApp).
	if sent.ProviderAppID == "" {
		return stack.Wrap(errors.New("signing_secret needs provider_app_id: an app's events are found, and verified, by the app their URL names"))
	}
	// Only a channel verified with the app's own secret reads it (receiveProviderAppEvent).
	if manifest.Channel == nil || manifest.Channel.Verifier.Secret != core.SecretProviderApp {
		return stack.Wrap(fmt.Errorf("%s does not verify its events with the app's own secret (channel.verifier.secret provider_app), so nothing would read signing_secret: leave it out",
			manifest.ID))
	}
	return nil
}

// ConnectorClients is how oauth2_code finds a client registered in advance (oauth2code.Config.Clients):
// the app's own from its record (setConnectorOAuthClient), the one the router created for the
// app (managed, T54) from its record, and the operator's from a record
// this deployment keeps for the app, else from the environment, <client.env>_MCP_CLIENT_ID and
// _SECRET (oauth2code.EnvClients). It is called at every exchange, refresh and revocation, so
// a secret put again is the one the next refresh sends. Without a store or a keyring no record
// is read.
func ConnectorClients(records *store.Store, secrets *auth.Sealer, getenv func(string) string) oauth2code.ClientLookup {
	return connectorClients(records, secrets, getenv, true)
}

// ConnectorClientsReadOnly is ConnectorClients that never seals a secret again: a record sealed
// under an older key version is opened and left as it is. For router plugins migrate, whose dry
// run writes nothing, and whose keyring may be newer than the routers' that read the record.
func ConnectorClientsReadOnly(records *store.Store, secrets *auth.Sealer, getenv func(string) string) oauth2code.ClientLookup {
	return connectorClients(records, secrets, getenv, false)
}

func connectorClients(records *store.Store, secrets *auth.Sealer, getenv func(string) string, rewrap bool) oauth2code.ClientLookup {
	environment := oauth2code.EnvClients(getenv)
	return func(ctx context.Context, ref core.ConnectionRef, m core.ResolvedManifest, registration core.ClientRegistrationMethod) (oauth2code.Client, bool, error) {
		if records != nil && secrets != nil {
			record, err := records.ConnectorOAuthClient(ctx, ref.CustomerID, m.ConnectorID)
			if err != nil && !errors.Is(err, store.ErrNoConnectorOAuthClient) {
				return oauth2code.Client{}, false, err
			}
			if err == nil && record.Registration == registration {
				return openOAuthClient(ctx, records, secrets, record, rewrap)
			}
		}
		if registration == core.ClientOperator {
			return environment(ctx, ref, m, registration)
		}
		return oauth2code.Client{}, false, nil
	}
}

// openOAuthClient is a record as oauth2code takes it, its secret opened. A secret sealed under
// an older key version is sealed again under the current one, as pgsealed does for a
// connection's credentials, so an old key can be removed once every record was used
// (auth.NewSealerWithKeyring). A managed or operator record has no put through the API to do it.
// Without rewrap it is left as it is.
func openOAuthClient(ctx context.Context, records *store.Store, secrets *auth.Sealer, record store.ConnectorOAuthClient, rewrap bool) (oauth2code.Client, bool, error) {
	client := oauth2code.Client{ID: record.ClientID, AuthMethod: record.AuthMethod}
	if len(record.SecretSealed) == 0 {
		return client, true, nil
	}
	secret, err := secrets.OpenWithAADVersion(record.SecretSealed, oauthClientAAD(record.CustomerID, record.ConnectorID), record.KEKVersion)
	if err != nil {
		// Not the opener's error alone: it would not say whose client failed to open.
		return oauth2code.Client{}, false, stack.Wrap(fmt.Errorf("api: the %s OAuth client of %s does not open: %w", record.Registration, record.ConnectorID, err))
	}
	client.Secret = secret
	if rewrap && record.KEKVersion != secrets.CurrentVersion() {
		rewrapOAuthClient(ctx, record, "client secret", secret, oauthClientAAD(record.CustomerID, record.ConnectorID), secrets,
			func(sealed []byte, version int) (bool, error) {
				return records.RewrapConnectorOAuthClientSecret(ctx, record.CustomerID, record.ConnectorID, record.SecretSealed, sealed, version)
			})
	}
	return client, true, nil
}

// ProviderApp is the record of the provider app an inbound request names, with the secret the
// provider signs that app's requests with, opened: what the events route for one provider app
// (T38) hands the verifier (secret source provider_app, core.SecretProviderApp). The record
// says whose app it is, and its StreamAppPK the Stream app the app's events are written into.
// A signing secret sealed under an older key version is sealed again under the current one, as
// openOAuthClient does for a client secret. A record without a signing secret is
// store.ErrNoConnectorOAuthClient, as a missing one is: an HMAC under an empty key is one
// anybody can compute.
func ProviderApp(ctx context.Context, records *store.Store, secrets *auth.Sealer, connectorID, providerAppID string) (store.ConnectorOAuthClient, string, error) {
	record, err := records.ConnectorOAuthClientByProviderApp(ctx, connectorID, providerAppID)
	if err != nil {
		return store.ConnectorOAuthClient{}, "", err
	}
	if len(record.SigningSecretSealed) == 0 {
		return store.ConnectorOAuthClient{}, "", stack.Wrap(fmt.Errorf("%w: provider app %s of %s has no signing secret",
			store.ErrNoConnectorOAuthClient, providerAppID, connectorID))
	}
	secret, err := secrets.OpenWithAADVersion(record.SigningSecretSealed,
		providerAppAAD(record.CustomerID, record.ConnectorID, record.ProviderAppID), record.SigningKEKVersion)
	if err != nil {
		return store.ConnectorOAuthClient{}, "", stack.Wrap(fmt.Errorf("api: the signing secret of provider app %s of %s does not open: %w",
			providerAppID, connectorID, err))
	}
	if record.SigningKEKVersion != secrets.CurrentVersion() {
		rewrapOAuthClient(ctx, record, "signing secret", secret, providerAppAAD(record.CustomerID, record.ConnectorID, record.ProviderAppID), secrets,
			func(sealed []byte, version int) (bool, error) {
				return records.RewrapConnectorOAuthClientSigningSecret(ctx, record.CustomerID, record.ConnectorID, record.SigningSecretSealed, sealed, version)
			})
	}
	return record, secret, nil
}

// rewrapOAuthClient seals an opened secret of a record again under the keyring's current version and
// writes it with write, which leaves a secret replaced meanwhile as it is. A chore, not a
// condition of using the secret: the old seal still opens, so a failure is logged and the
// next use tries again.
func rewrapOAuthClient(ctx context.Context, record store.ConnectorOAuthClient, what, secret string, aad []byte, secrets *auth.Sealer,
	write func(sealed []byte, version int) (bool, error)) {
	sealed, err := secrets.SealWithAAD(secret, aad)
	if err == nil {
		_, err = write(sealed, secrets.CurrentVersion())
	}
	if err != nil {
		slog.WarnContext(ctx, "connectors: could not seal an OAuth client's secret again under the current key version",
			"customer_id", record.CustomerID, "connector_id", record.ConnectorID, "secret", what, "error", err)
	}
}

// providerAppAAD binds a sealed signing secret to its customer, its connector and the provider
// app it signs for, so a blob copied onto another app's row, or a row whose app id was changed,
// does not open. Laid out as oauthClientAAD, with its own prefix, so a client secret's blob
// never opens as a signing secret. v1 changes with the layout.
func providerAppAAD(customerID, connectorID, providerAppID string) []byte {
	return fmt.Appendf(nil, "accelerate:connector-provider-app-signing-secret:v1:%d:%s:%d:%s:%d:%s",
		len(customerID), customerID, len(connectorID), connectorID, len(providerAppID), providerAppID)
}

// oauthClientAAD binds a sealed client secret to its customer and connector, the row's key,
// so a blob copied onto another app's or another connector's row does not open. Each part is
// length-prefixed, as attemptAAD and pgsealed's credentialsAAD are, so no two pairs give the
// same bytes. v1 changes with the layout.
func oauthClientAAD(customerID, connectorID string) []byte {
	return fmt.Appendf(nil, "accelerate:connector-oauth-client:v1:%d:%s:%d:%s",
		len(customerID), customerID, len(connectorID), connectorID)
}

// SealConnectorOAuthClientSecret seals a client secret for the customer's record of the
// connector as a put through the API does, under the keyring's current version, so a record
// router plugins migrate writes opens where the API's do.
func SealConnectorOAuthClientSecret(secrets *auth.Sealer, customerID, connectorID, secret string) ([]byte, int, error) {
	sealed, err := secrets.SealWithAAD(secret, oauthClientAAD(customerID, connectorID))
	if err != nil {
		return nil, 0, err
	}
	return sealed, secrets.CurrentVersion(), nil
}

func oauthClientOf(record store.ConnectorOAuthClient) ConnectorOAuthClient {
	return ConnectorOAuthClient{
		ConnectorID:   record.ConnectorID,
		Registration:  ConnectorClientRegistrationMethod(record.Registration),
		ClientID:      record.ClientID,
		AuthMethod:    ConnectorOAuthClientAuthMethod(record.AuthMethod),
		ProviderAppID: record.ProviderAppID,
		CreatedAt:     record.CreatedAt,
		UpdatedAt:     record.UpdatedAt,
	}
}
