package api

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
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
	bound, found, err := s.streamFor(ctx, customerID)
	if errors.Is(err, streamapp.ErrStreamAppDisconnected) || (err == nil && !found) {
		return &appSettingsResponse{Body: AppSettings{Stream: settings}}, nil
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
