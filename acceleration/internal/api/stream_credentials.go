package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// streamCredentialsPath is where an app's Stream keys are written. Nothing under it is
// handed to Sentry, which would copy the body, and the body is the secrets.
const streamCredentialsPath = "/v1/settings/app/stream/"

// maxSecretLength bounds a secret, so a body cannot carry anything much else in its place.
const maxSecretLength = 256

// StreamCredentials is every key the calling app's Stream app is now acted in with.
type StreamCredentials struct {
	Keys             []StreamKeyInput `json:"keys" maxItems:"8" doc:"Every key the router may act in the app with, which replaces those it held. Each is checked with Stream. Empty disconnects the app, which needs a proof."`
	PrimaryKey       string           `json:"primary_key,omitempty" doc:"The key tokens are minted with. Left out is the first."`
	ExpectedRevision int64            `json:"expected_revision" doc:"The revision last read, 0 for an app never registered. A write made against an older one is a 409."`
	AllowGuests      *bool            `json:"allow_guests,omitempty" doc:"Whether guests may be made in the app. Left out keeps what was set, which starts off false."`
	Proof            *StreamKeyInput  `json:"proof,omitempty" doc:"A key and secret of the app, which disconnecting needs. It is checked with Stream and kept nowhere."`
}

func (*StreamCredentials) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The keys the router acts in the calling app's own Stream app with. " +
		"Secrets are written and never read back: no answer carries one."
	return schema
}

// StreamKeyInput is a key and its secret.
type StreamKeyInput struct {
	APIKey    string      `json:"api_key" minLength:"1" maxLength:"128"`
	APISecret secretValue `json:"api_secret" required:"true"`
	CreatedAt *time.Time  `json:"created_at,omitempty" doc:"When Stream made the key, which says which key is the oldest and so signs the app's webhooks."`
}

// secretValue is a secret as it was sent, read in the handler rather than by the decoder,
// because a decoding error names the value it could not decode.
type secretValue struct {
	raw json.RawMessage
}

func (v *secretValue) UnmarshalJSON(data []byte) error {
	v.raw = append(v.raw[:0], data...)
	return nil
}

func (secretValue) Schema(huma.Registry) *huma.Schema {
	longest := maxSecretLength
	return &huma.Schema{Type: huma.TypeString, WriteOnly: true, MaxLength: &longest,
		Description: "The key's secret. It is sealed and never read back."}
}

// reveal is the secret as a string, refused without repeating what was sent.
func (v secretValue) reveal(apiKey string) (streamapp.Secret, error) {
	var secret string
	if err := json.Unmarshal(v.raw, &secret); err != nil || secret == "" || len(secret) > maxSecretLength {
		return streamapp.Secret{}, huma.Error400BadRequest("the secret of key " + apiKey +
			" must be a string of at most " + strconv.Itoa(maxSecretLength) + " characters")
	}
	return streamapp.NewSecret(secret), nil
}

// StreamCheck is what checking the app found beyond its settings.
type StreamCheck struct {
	Settings AppSettings `json:"settings"`
	// Reattach are numbers whose lines were made in another app than the registered one,
	// so callers still land there until each is attached again.
	Reattach []string `json:"reattach"`
}

type streamCredentialsRequest struct {
	Body StreamCredentials
}

type streamCheckResponse struct {
	Body StreamCheck
}

func (s *Server) registerStreamCredentials(api huma.API) {
	errs := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
		http.StatusConflict, http.StatusServiceUnavailable}
	huma.Register(api, huma.Operation{
		OperationID: "updateAppStreamCredentials",
		Method:      http.MethodPut,
		Path:        streamCredentialsPath + "credentials",
		Summary:     "Register the calling app's own Stream app",
		Description: "The keys the router acts in the calling app's own Stream app with, from now " +
			"on, for every conversation, transcript, call and phone line. Each key is checked " +
			"with Stream: it has to belong to the calling app, and the app may be neither " +
			"suspended nor taking requests without checking their tokens. Keys left out are " +
			"dropped, and the sessions acting in the app end. Needs stream.tenancy=app.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device. Behind a proxy, the proxy has to declare the caller a server.",
		Responses: map[string]*huma.Response{"200": {Description: "The app's settings as they now are"}},
		Errors:    errs,
	}, s.updateAppStreamCredentials)
	huma.Register(api, huma.Operation{
		OperationID: "checkAppStreamCredentials",
		Method:      http.MethodPost,
		Path:        streamCredentialsPath + "check",
		Summary:     "Check the calling app's own Stream app",
		Description: "Asks Stream again about the registered app: whether it stands, whether it " +
			"holds the agent channel and call types, and which numbers still have their lines " +
			"in another app. Needs stream.tenancy=app.\n\nServer-side only.",
		Responses: map[string]*huma.Response{"200": {Description: "What the check found"}},
		Errors:    errs,
	}, s.checkAppStreamCredentials)
}

// registrar is app mode's source, once the caller may write the app's credentials at all.
func (s *Server) registrar(ctx context.Context) (string, *streamapp.Stored, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return "", nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	var stored *streamapp.Stored
	if s.stream != nil {
		stored, ok = s.stream.Stored()
	}
	if !ok {
		return "", nil, huma.Error400BadRequest("an app registers its own Stream app only when the " +
			"router runs with stream.tenancy=app")
	}
	// A proxy that does not say what kind of caller it vouched for passes every caller
	// as a backend, and a page must never be able to hand the router somebody's keys.
	if s.authMode == auth.Proxy && (!s.proxyDeclaresKind || KindFrom(ctx) != auth.KindServer) {
		return "", nil, huma.Error403Forbidden("registering a Stream app needs a proxy that declares " +
			"the caller a server, with auth.proxy_declares_kind on")
	}
	if slices.Contains(s.denyRegistration, customerID) {
		return "", nil, huma.Error403Forbidden("this app may not register a Stream app of its own")
	}
	return customerID, stored, nil
}

func (s *Server) updateAppStreamCredentials(ctx context.Context, request *streamCredentialsRequest) (*appSettingsResponse, error) {
	customerID, stored, err := s.registrar(ctx)
	if err != nil {
		return nil, err
	}
	sent := request.Body
	expected := sent.ExpectedRevision
	before, err := s.store.StreamApp(ctx, customerID)
	if err != nil && !errors.Is(err, store.ErrNoStreamApp) {
		return nil, err
	}

	if len(sent.Keys) == 0 {
		if sent.Proof == nil {
			return nil, huma.Error400BadRequest("disconnecting needs a proof: a key and secret of the app")
		}
		secret, err := sent.Proof.APISecret.reveal(sent.Proof.APIKey)
		if err != nil {
			return nil, err
		}
		app, err := stored.Disconnect(ctx, customerID, streamapp.Key{APIKey: sent.Proof.APIKey, Secret: secret}, &expected, updatedBy(ctx))
		if err != nil {
			return nil, registrationFailure(err)
		}
		s.stopActingIn(customerID, app.StreamAppPK)
		return s.getAppSettings(ctx, nil)
	}

	keys := make([]streamapp.Key, 0, len(sent.Keys))
	for _, key := range sent.Keys {
		secret, err := key.APISecret.reveal(key.APIKey)
		if err != nil {
			return nil, err
		}
		one := streamapp.Key{APIKey: key.APIKey, Secret: secret}
		if key.CreatedAt != nil {
			one.CreatedAt = *key.CreatedAt
		}
		keys = append(keys, one)
	}
	allowGuests := before.AllowGuests
	if sent.AllowGuests != nil {
		allowGuests = *sent.AllowGuests
	}
	registered, err := stored.Register(ctx, streamapp.Registration{
		CustomerID: customerID, OrganizationID: OrganizationFrom(ctx), Keys: keys, PrimaryKey: sent.PrimaryKey,
		AllowGuests: allowGuests, ExpectedRevision: &expected, UpdatedBy: updatedBy(ctx),
	})
	if err != nil {
		return nil, registrationFailure(err)
	}
	s.stream.Invalidate(customerID)
	// What was acting with a key that is gone, or in an app that is no longer this one's,
	// stops rather than going on with a credential the app took back.
	if len(registered.Dropped) > 0 {
		s.stopActingIn(customerID, registered.Previous)
	}
	if registered.Previous != 0 && registered.Previous != registered.App.StreamAppPK {
		s.stopActingIn(customerID, registered.Previous)
	}
	return s.getAppSettings(ctx, nil)
}

func (s *Server) checkAppStreamCredentials(ctx context.Context, _ *struct{}) (*streamCheckResponse, error) {
	customerID, stored, err := s.registrar(ctx)
	if err != nil {
		return nil, err
	}
	err = stored.CheckApp(ctx, s.stream, customerID, func(customer string, app int64) { s.stopActingIn(customer, app) })
	if errors.Is(err, store.ErrNoStreamApp) {
		return nil, huma.Error400BadRequest("this app has not registered a Stream app of its own")
	}
	if err != nil && !errors.Is(err, streamapp.ErrStreamAppDisconnected) {
		return nil, err
	}
	s.stream.Invalidate(customerID)
	settings, err := s.getAppSettings(ctx, nil)
	if err != nil {
		return nil, err
	}
	app, err := s.store.StreamApp(ctx, customerID)
	if err != nil {
		return nil, err
	}
	numbers, err := s.store.CustomerNumbers(ctx, customerID, false)
	if err != nil {
		return nil, err
	}
	reattach := []string{}
	for _, number := range numbers {
		if number.StreamTrunkID != "" && number.StreamAppPK != app.StreamAppPK {
			reattach = append(reattach, number.E164)
		}
	}
	return &streamCheckResponse{Body: StreamCheck{Settings: settings.Body, Reattach: reattach}}, nil
}

// stopActingIn ends what the router holds in an app it no longer acts in for a customer.
func (s *Server) stopActingIn(customerID string, app int64) {
	s.stream.Invalidate(customerID)
	if s.sessions != nil && app != 0 {
		s.sessions.EndPinned(customerID, app)
	}
}

// registrationFailure answers what registering or disconnecting could not do, naming keys
// and never secrets.
func registrationFailure(err error) error {
	switch {
	case errors.Is(err, streamapp.ErrKeyRefused):
		return huma.Error400BadRequest(strings.TrimPrefix(err.Error(), "streamapp: "))
	case errors.Is(err, store.ErrStreamAppChanged):
		return huma.Error409Conflict("the app's Stream credentials changed since they were read: read them again")
	case errors.Is(err, store.ErrStreamAppTaken):
		return huma.Error409Conflict("that Stream app is registered by another app")
	case errors.Is(err, store.ErrStreamAppKeyTaken):
		return huma.Error409Conflict("one of those keys belongs to another app's Stream app")
	case errors.Is(err, store.ErrNoStreamApp):
		return huma.Error400BadRequest("this app has not registered a Stream app of its own")
	}
	return huma.Error503ServiceUnavailable("Stream could not be asked about the keys: try again")
}

// updatedBy is who to record as having written an app's credentials.
func updatedBy(ctx context.Context) string {
	if user := CallerFrom(ctx).UserID; user != "" {
		return user
	}
	return string(KindFrom(ctx))
}
