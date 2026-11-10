package main

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/agent/streamedge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// errNoStreamApp is a session asked to do something in Stream when the router has no app
// to do it in.
var errNoStreamApp = errors.New("router: no Stream app is configured for this customer")

// agentTokenValidity is how long the token an agent joins a call with lasts when the
// gateway signs it: longer than any call, and within the day the gateway allows.
const agentTokenValidity = 12 * time.Hour

// edgeFor joins a session's call in the Stream app the session is pinned to, with that
// app's own credential. Nothing is read from the environment: the identity is the whole
// answer, so a customer's call is never joined in whichever app the deployment names.
func edgeFor(clients *streamapp.Clients) session.EdgeFactory {
	return func(ctx context.Context, spec session.Spec, stream streamapp.Bound, logger *slog.Logger) (agent.Edge, error) {
		identity := stream.Identity
		if identity.APIKey == "" {
			return nil, errNoStreamApp
		}
		userToken := identity.UserToken
		if identity.Secret.Empty() && userToken == "" {
			// An app acted in with a forwarded token: the agent joins with a user token
			// the gateway signs, since the router holds no secret to sign one with.
			// The agent joins with the identity's own key, so its token is minted for that key.
			minted, _, err := clients.MintUserToken(ctx, stream, "", spec.UserID, agentTokenValidity)
			if err != nil {
				return nil, err
			}
			userToken = minted
		}
		return streamedge.New(streamedge.Options{
			CallID:     spec.CallID,
			CallType:   spec.CallType,
			User:       streamedge.User{ID: spec.UserID, Name: spec.UserName},
			APIKey:     identity.APIKey,
			APISecret:  identity.Secret.Reveal(),
			UserToken:  userToken,
			BaseURL:    identity.BaseURL,
			HTTPClient: clients.HTTPClient(),
			Logger:     logger,
		})
	}
}

// transcriptFor writes a session's transcript in the Stream app it is pinned to, with the
// client already held for that app. timings has each reply carry how long its turn took, which
// is for a developer and not for a customer.
func transcriptFor(timings bool) session.TranscriptFactory {
	return func(_ context.Context, spec session.Spec, stream streamapp.Bound, logger *slog.Logger) (session.Transcript, error) {
		if stream.Client == nil {
			return nil, errNoStreamApp
		}
		// The same reading the session's episode card names (Spec.TranscriptChannel).
		channelType, channel := spec.ConversationChannel()
		return chatlog.New(chatlog.Options{
			AgentID:      spec.AgentID,
			Channel:      channel,
			ChannelType:  channelType,
			Opened:       spec.PersistConversation,
			CustomerID:   spec.CustomerID,
			Team:         spec.ChannelTeam,
			Agent:        chatlog.User{ID: spec.UserID, Name: spec.UserName},
			VisibleTools: spec.VisibleTools,
			Timings:      timings,
			Client:       stream.Client,
			Logger:       logger,
		})
	}
}
