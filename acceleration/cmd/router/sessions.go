package main

import (
	"context"
	"errors"
	"log/slog"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/agent/streamedge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
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
			Explicit:   true,
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
		channel := strings.TrimPrefix(spec.ConversationID, "agent:")
		if channel == spec.ConversationID {
			channel = ""
		}
		return chatlog.New(chatlog.Options{
			AgentID:      spec.AgentID,
			Channel:      channel,
			CustomerID:   spec.CustomerID,
			Agent:        chatlog.User{ID: spec.UserID, Name: spec.UserName},
			VisibleTools: spec.VisibleTools,
			Client:       stream.Client,
			Logger:       logger,
		})
	}
}
