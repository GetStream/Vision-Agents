//go:build integration

package turbopuffer

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// TestEveryCallReportsWhatItBilled makes each call against turbopuffer itself, in a scratch
// namespace deleted after, and checks each read back what turbopuffer billed it. Its name has
// no customer in it, so no storage run bills it.
func TestEveryCallReportsWhatItBilled(t *testing.T) {
	if os.Getenv(apiKeyEnvVar) == "" {
		t.Skip(apiKeyEnvVar + " not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	store, err := New(Options{Cost: listCost, Timeout: 30 * time.Second, Logger: slog.New(slog.DiscardHandler)})
	require.NoError(t, err)
	namespace := fmt.Sprintf("acceleration-test-billing-%d", time.Now().UnixNano())
	t.Cleanup(func() {
		var deleted map[string]any
		_, err := store.call(context.Background(), http.MethodDelete, "/v2/namespaces/"+namespace, nil, &deleted)
		require.NoError(t, err)
	})

	written, err := store.Upsert(ctx, namespace, []knowledge.Document{
		{ID: "shipping.md#0", Text: "Delivery is free over $50.", Source: "shipping.md"},
		{ID: "shipping.md#1", Text: "Next-day delivery costs $9.", Source: "shipping.md"},
	})
	require.NoError(t, err)
	require.EqualValues(t, 1, written.Requests)
	require.Positive(t, written.BytesWritten)
	require.Positive(t, written.CostNanos)

	found, err := store.Search(ctx, knowledge.Query{Namespace: namespace, Text: "delivery"})
	require.NoError(t, err)
	require.NotEmpty(t, found.Documents)
	require.GreaterOrEqual(t, found.BytesQueried, int64(1_280_000_000), "a small namespace is billed the minimum")
	require.Positive(t, found.BytesReturned)
	require.EqualValues(t, len(found.Documents), found.Results)

	fetched, err := store.Fetch(ctx, namespace, []string{"shipping.md#0"})
	require.NoError(t, err)
	require.Len(t, fetched.Documents, 1)
	require.Positive(t, fetched.BytesQueried)

	stored, err := store.Stored(ctx, namespace)
	require.NoError(t, err)
	require.EqualValues(t, 1, stored.Requests)

	names, err := store.Namespaces(ctx, namespace)
	require.NoError(t, err)
	require.Equal(t, []string{namespace}, names)

	deleted, err := store.Delete(ctx, namespace, []string{"shipping.md#0", "shipping.md#1"})
	require.NoError(t, err)
	require.EqualValues(t, 1, deleted.Requests)
	require.Positive(t, deleted.CostNanos, "a delete is billed at least the write minimum")
}
