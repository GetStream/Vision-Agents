package main

import (
	"context"
	"errors"
	"log/slog"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/agent/streamedge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// errNoStreamApp is a session asked to do something in Stream when the router has no app
// to do it in.
var errNoStreamApp = errors.New("router: no Stream app is configured for this customer")

// edgeFor joins a session's call in the Stream app the session is pinned to, with that
// app's own credential. Nothing is read from the environment: the identity is the whole
// answer, so a customer's call is never joined in whichever app the deployment names.
func edgeFor(clients *streamapp.Clients) session.EdgeFactory {
	return func(_ context.Context, spec session.Spec, stream streamapp.Bound, logger *slog.Logger) (agent.Edge, error) {
		identity := stream.Identity
		if identity.APIKey == "" {
			return nil, errNoStreamApp
		}
		return streamedge.New(streamedge.Options{
			CallID:     spec.CallID,
			CallType:   spec.CallType,
			User:       streamedge.User{ID: spec.UserID, Name: spec.UserName},
			APIKey:     identity.APIKey,
			APISecret:  identity.Secret.Reveal(),
			UserToken:  identity.UserToken,
			BaseURL:    identity.BaseURL,
			HTTPClient: clients.HTTPClient(),
			Logger:     logger,
		})
	}
}

// transcriptFor writes a session's transcript in the Stream app it is pinned to, with the
// client already held for that app.
func transcriptFor() session.TranscriptFactory {
	return func(_ context.Context, spec session.Spec, stream streamapp.Bound, logger *slog.Logger) (session.Transcript, error) {
		if stream.Client == nil {
			return nil, errNoStreamApp
		}
		return chatlog.New(chatlog.Options{
			AgentID: spec.AgentID,
			// The same reading the session's episode card names (Spec.TranscriptChannel).
			Channel:      spec.ConversationChannel(),
			CustomerID:   spec.CustomerID,
			Agent:        chatlog.User{ID: spec.UserID, Name: spec.UserName},
			VisibleTools: spec.VisibleTools,
			Client:       stream.Client,
			Logger:       logger,
		})
	}
}

// phoneApps makes a customer's phone lines in the Stream app it acts in, and finishes the
// lines already made in the app they were made in.
type phoneApps struct {
	clients *streamapp.Clients
}

func (a phoneApps) For(ctx context.Context, customer string) (*phone.Stream, int64, error) {
	bound, err := a.clients.For(ctx, customer)
	if err != nil {
		return nil, 0, err
	}
	return phone.NewStreamFromClient(bound.Client), bound.Identity.StreamApp, nil
}

func (a phoneApps) ForApp(ctx context.Context, customer string, app int64) (*phone.Stream, error) {
	bound, err := a.clients.ForApp(ctx, customer, app)
	if err != nil {
		return nil, err
	}
	return phone.NewStreamFromClient(bound.Client), nil
}

func (a phoneApps) ForAppRemoving(ctx context.Context, customer string, app int64) (*phone.Stream, error) {
	bound, err := a.clients.ForAppReading(ctx, customer, app)
	if err != nil {
		return nil, err
	}
	return phone.NewStreamFromClient(bound.Client), nil
}
