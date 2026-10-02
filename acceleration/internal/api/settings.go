package api

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// AppSettings is what the router does for the calling app, as far as the app can see it.
type AppSettings struct {
	Stream StreamSettings `json:"stream"`
}

func (*AppSettings) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What the router does for the calling app. It never carries a secret."
	return schema
}

// StreamSettings is which Stream app the router writes the app's conversations and calls
// into, and whether that app holds what they need.
type StreamSettings struct {
	Tenancy     StreamTenancy    `json:"tenancy"`
	WritesInto  StreamWritesInto `json:"writes_into"`
	ChannelType StreamTypeState  `json:"channel_type" doc:"Whether the Stream app holds the agent channel type conversations are kept in. unsafe is one whose grants let somebody other than the app's backend make, change or join a conversation's channel, or read one they are not in."`
	CallType    StreamTypeState  `json:"call_type" doc:"Whether the Stream app holds the agent call type calls are made with. Stream reports no grants for it here, so it is present or missing."`
	CheckedAt   *time.Time       `json:"checked_at,omitempty" doc:"When Stream was asked. Absent when it could not be, and the types are then unknown. Answers are reused for a minute."`

	// What follows is app mode's, and absent in deployment mode.
	Revision    *int64           `json:"revision,omitempty" doc:"The registration's revision, 0 for an app that registered none. A write names the one it read."`
	State       *StreamAppState  `json:"state,omitempty" doc:"Whether the router acts in the registered app. Absent for an app that registered none."`
	StateReason string           `json:"state_reason,omitempty" doc:"Why the router stopped acting in the app, for one that is blocked."`
	StreamAppID *int64           `json:"stream_app_id,omitempty" doc:"The registered app's own id."`
	PrimaryKey  string           `json:"primary_key,omitempty" doc:"The key tokens are minted with."`
	AllowGuests *bool            `json:"allow_guests,omitempty" doc:"Whether guests may be made in the registered app."`
	Keys        []StreamKeyState `json:"keys,omitempty" doc:"The registered app's keys, oldest first. No secret is ever read back."`
}

// StreamKeyState is one key the router holds for the calling app, and how it stands.
type StreamKeyState struct {
	APIKey        string     `json:"api_key"`
	SecretLast4   string     `json:"secret_last4,omitempty" doc:"The end of the secret, enough to tell two apart."`
	CreatedAt     *time.Time `json:"created_at,omitempty" doc:"When Stream made the key."`
	Status        string     `json:"status" enum:"active,rejected" doc:"rejected is a key Stream stopped accepting, which the router no longer uses."`
	VerifiedAt    *time.Time `json:"verified_at,omitempty"`
	LastWebhookAt *time.Time `json:"last_webhook_at,omitempty" doc:"When Stream last signed a hook with this key."`
	SignsWebhooks string     `json:"signs_webhooks" enum:"yes,no,unknown" doc:"Whether Stream signs the app's hooks with this key, which is its oldest. yes is one a hook arrived signed with."`
}

// StreamAppState is whether the router acts in a registered app.
type StreamAppState string

func (StreamAppState) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "StreamAppState", "Whether the router acts in a registered app. "+
		"disconnected is one the app took back, and blocked one Stream suspended or that stopped "+
		"checking tokens. Neither is ever written into the router's own app instead.",
		string(store.StreamAppConnected), string(store.StreamAppDisconnected), string(store.StreamAppBlocked))
}

func (*StreamSettings) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Which Stream app the router writes the calling app's conversations, " +
		"transcripts, calls and phone lines into, and whether that app holds the types they need."
	return schema
}

// StreamTenancy is whose Stream app the router acts in.
type StreamTenancy string

func (StreamTenancy) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "StreamTenancy", "Whose Stream app the router acts in. deployment "+
		"is one app, the router's own, for every app it serves; app is each app's own.",
		config.TenancyDeployment, config.TenancyApp)
}

// StreamWritesInto is which Stream app the calling app's work is written into.
type StreamWritesInto string

const (
	WritesIntoThisApp       StreamWritesInto = "this_app"
	WritesIntoDeploymentApp StreamWritesInto = "deployment_app"
	WritesIntoNowhere       StreamWritesInto = "nowhere"
)

func (StreamWritesInto) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "StreamWritesInto", "Which Stream app the calling app's work is "+
		"written into: this_app is its own, deployment_app is the router's own app, shared with "+
		"every app it serves that has none, and nowhere is no app at all, so conversations are "+
		"not kept and calls cannot be made.",
		string(WritesIntoThisApp), string(WritesIntoDeploymentApp), string(WritesIntoNowhere))
}

// StreamTypeState is whether a Stream app holds a type the router needs.
type StreamTypeState string

func (StreamTypeState) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "StreamTypeState", "Whether a Stream app holds a type the router "+
		"needs. unknown is a type Stream could not be asked about.",
		string(streamapp.TypePresent), string(streamapp.TypeMissing),
		string(streamapp.TypeUnsafe), string(streamapp.TypeUnknown))
}

type appSettingsResponse struct {
	Body AppSettings
}

func (s *Server) registerSettings(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "getAppSettings",
		Method:      http.MethodGet,
		Path:        "/v1/settings/app",
		Summary:     "What the router does for the calling app",
		Description: "Which Stream app the router writes the calling app's conversations and " +
			"calls into, and whether that app holds the agent channel and call types. Stream is " +
			"asked at most once a minute.\n\nServer-side only: it needs a server-side token, so " +
			"it cannot be reached from an end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The app's settings"}},
		Errors:    []int{http.StatusUnauthorized, http.StatusForbidden, http.StatusServiceUnavailable},
	}, s.getAppSettings)
}

// getAppSettings reports which Stream app the calling app's work goes into, and whether it
// is ready for it. A failure to reach Stream is reported, not answered with: the settings
// are worth reading most when Stream is the trouble.
func (s *Server) getAppSettings(ctx context.Context, _ *struct{}) (*appSettingsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	settings := StreamSettings{
		Tenancy:     StreamTenancy(s.streamTenancy),
		WritesInto:  WritesIntoNowhere,
		ChannelType: StreamTypeState(streamapp.TypeUnknown),
		CallType:    StreamTypeState(streamapp.TypeUnknown),
	}
	if err := s.describeRegistration(ctx, customerID, &settings); err != nil {
		return nil, err
	}
	bound, found, err := s.streamFor(ctx, customerID)
	if errors.Is(err, streamapp.ErrStreamAppDisconnected) || (err == nil && !found) {
		return &appSettingsResponse{Body: AppSettings{Stream: settings}}, nil
	}
	if errors.Is(err, streamapp.ErrDeploymentAppUnknown) {
		return nil, streamWaiting()
	}
	if err != nil {
		return nil, huma.Error503ServiceUnavailable("which Stream app this app acts in could not be read")
	}
	settings.WritesInto = s.writesInto(customerID, bound.Identity)
	if readiness, err := s.stream.Readiness(ctx, bound); err == nil {
		settings.ChannelType = StreamTypeState(readiness.ChannelType)
		settings.CallType = StreamTypeState(readiness.CallType)
		settings.CheckedAt = &readiness.CheckedAt
	} else {
		s.logger.Warn("stream: could not read what an app holds", "customer_id", customerID, "error", err)
	}
	return &appSettingsResponse{Body: AppSettings{Stream: settings}}, nil
}

// writesInto names the app an identity acts in, as the calling app should read it. The
// router's own app is the caller's own only when the caller is that app; to anybody else
// it is shared, and which app it is is not theirs to know.
func (s *Server) writesInto(customerID string, identity streamapp.Identity) StreamWritesInto {
	deployment := s.stream.DeploymentApp()
	if identity.StreamApp != 0 && identity.StreamApp != deployment {
		return WritesIntoThisApp
	}
	if deployment != 0 && customerID == strconv.FormatInt(deployment, 10) {
		return WritesIntoThisApp
	}
	return WritesIntoDeploymentApp
}

// describeRegistration fills in the Stream app the calling app registered, in app mode.
func (s *Server) describeRegistration(ctx context.Context, customerID string, settings *StreamSettings) error {
	if s.stream == nil || !s.stream.PerApp() || s.store == nil {
		return nil
	}
	app, err := s.store.StreamApp(ctx, customerID)
	if errors.Is(err, store.ErrNoStreamApp) {
		settings.Revision = new(int64)
		return nil
	}
	if err != nil {
		return err
	}
	state := StreamAppState(app.State)
	settings.Revision, settings.State, settings.StateReason = &app.Revision, &state, app.StateReason
	settings.StreamAppID, settings.PrimaryKey, settings.AllowGuests = &app.StreamAppPK, app.PrimaryKey, &app.AllowGuests
	oldest := oldestKey(app.Keys)
	for _, key := range app.Keys {
		signs := "no"
		switch {
		case key.LastWebhookAt != nil:
			signs = "yes"
		case key.APIKey == oldest || oldest == "":
			signs = "unknown"
		}
		settings.Keys = append(settings.Keys, StreamKeyState{
			APIKey: key.APIKey, SecretLast4: key.Last4, CreatedAt: key.KeyCreatedAt, Status: string(key.Status),
			VerifiedAt: key.VerifiedAt, LastWebhookAt: key.LastWebhookAt, SignsWebhooks: signs,
		})
	}
	return nil
}

// oldestKey is the key Stream made first, which is the one it signs an app's hooks with,
// or empty when Stream's dates for the keys are not all known.
func oldestKey(keys []store.StreamAppKey) string {
	var oldest store.StreamAppKey
	for _, key := range keys {
		if key.KeyCreatedAt == nil {
			return ""
		}
		if oldest.KeyCreatedAt == nil || key.KeyCreatedAt.Before(*oldest.KeyCreatedAt) {
			oldest = key
		}
	}
	return oldest.APIKey
}
