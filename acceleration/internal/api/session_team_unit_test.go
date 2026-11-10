package api

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

func backendAsking(categories ...string) context.Context {
	ctx := context.WithValue(context.Background(), serverSideContextKey{}, true)
	return routing.WithAttribution(ctx, routing.Attribution{UsageCategories: categories})
}

func TestADeviceMayNotNameASessionsTeam(t *testing.T) {
	failure, refused := refuseSessionRequest(context.Background(), CreateSessionRequest{Team: new("finance")})

	require.True(t, refused)
	require.Equal(t, errDeviceTeam, failure)
}

func TestABackendNamesATeamOrTheOneItsTokenWasSignedFor(t *testing.T) {
	_, refused := refuseSessionRequest(backendAsking(), CreateSessionRequest{Team: new("finance")})
	require.False(t, refused)
	_, refused = refuseSessionRequest(backendAsking("team:finance"), CreateSessionRequest{Team: new("finance")})
	require.False(t, refused, "the team its token names")

	failure, refused := refuseSessionRequest(backendAsking("team:sales"), CreateSessionRequest{Team: new("finance")})
	require.True(t, refused)
	require.Contains(t, failure.Message, `the team "sales" the token was signed for`)

	failure, refused = refuseSessionRequest(backendAsking(), CreateSessionRequest{Team: new("fin:ance")})
	require.True(t, refused)
	require.Equal(t, ErrorTypeInvalidRequest, failure.Type)
}

func TestASessionsTeamIsItsRequestsOverItsConfigsTag(t *testing.T) {
	config := &store.AgentConfig{ID: "c", Tags: map[string]string{"team": "support"}}

	require.Equal(t, "support", specOf(CreateSessionRequest{}, "acme", config).Team)
	require.Equal(t, "finance", specOf(CreateSessionRequest{Team: new("finance")}, "acme", config).Team)
	require.Empty(t, specOf(CreateSessionRequest{Tags: &map[string]string{"team": "sales"}}, "acme", nil).Team,
		"a session's own tags, which a device may send, name no team")
}

func TestAConfigsTeamTagIsATeamsName(t *testing.T) {
	_, ok := teamTagComplaint(map[string]string{"team": "support"}, nil)
	require.True(t, ok)
	_, ok = teamTagComplaint(map[string]string{"env": "prod"}, nil)
	require.True(t, ok, "no team tag")

	message, ok := teamTagComplaint(map[string]string{"team": "customer support"}, nil)
	require.False(t, ok)
	require.Contains(t, message, `tags.team "customer support" is not a team`)

	_, ok = teamTagComplaint(map[string]string{"team": "customer support"}, map[string]string{"team": "customer support"})
	require.True(t, ok, "a tag the config already held is let through")
}
