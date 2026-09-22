package api

import (
	"context"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/stretchr/testify/require"
)

func TestPersonalSessionAccessUsesVerifiedCaller(t *testing.T) {
	personal := session.Spec{PersistConversation: true, Caller: routing.Caller{UserID: "owner"}}
	for _, caller := range []string{"owner", "other", ""} {
		ctx := context.WithValue(context.Background(), callerContextKey{}, routing.Caller{UserID: caller})
		require.Equal(t, caller == "owner", canReadSession(ctx, personal))
	}
	legacy := session.Spec{PersistConversation: true}
	require.True(t, canReadSession(context.Background(), legacy))
	employee := context.WithValue(context.Background(), callerContextKey{}, routing.Caller{UserID: "owner"})
	require.False(t, canReadSession(employee, legacy), "an employee must not inherit a shared demo channel")
}
