package api

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A change is saved before its task is rendered, so a store that cannot say
// when the task last ran leaves last_run_at out instead of failing the
// answer to a change that stands.
func TestATaskIsRenderedWithoutItsLastRunWhenThatCannotBeRead(t *testing.T) {
	unreachable, err := store.Open("postgres://postgres:postgres@127.0.0.1:1/none?sslmode=disable")
	require.NoError(t, err)
	t.Cleanup(func() { _ = unreachable.Close() })
	server := &Server{store: unreachable, logger: slog.New(slog.DiscardHandler)}
	next := time.Date(2026, 10, 20, 9, 0, 0, 0, time.UTC)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	rendered := server.scheduledTaskOf(ctx, store.ScheduledTask{
		ID: "task", CustomerID: "acme", Title: "Stretch", Timezone: "UTC",
		StartLocal: next, NextRunAt: &next, State: store.ScheduledTaskActive,
	})

	require.Equal(t, "task", rendered.Id)
	require.Equal(t, "Stretch", rendered.Title)
	require.Nil(t, rendered.LastRunAt)
}
