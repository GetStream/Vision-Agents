//go:build integration

package routing

import (
	"context"
	"log/slog"
	"os"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// told is the usage ids a writer's callback was told of.
type told struct {
	mu  sync.Mutex
	ids []uuid.UUID
}

func (t *told) note(rows []Acked) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for _, row := range rows {
		t.ids = append(t.ids, row.UsageID)
	}
}

func (t *told) has(id uuid.UUID) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return slices.Contains(t.ids, id)
}

func TestAShardedWriterNeverWaitsOnCountingGrants(t *testing.T) {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("ROUTER_POSTGRES_DSN not set")
	}
	ctx := context.Background()
	shared, shard := testenv.Database(dsn, "grantcount_shared"), testenv.Database(dsn, "grantcount_shard")
	for _, each := range []string{shared, shard} {
		db, err := store.Open(each)
		require.NoError(t, err)
		require.NoError(t, db.Migrate(ctx))
		require.NoError(t, db.Close())
	}
	db, err := store.OpenShards(store.Options{DSN: shared, ShardDSN: shard, Shards: map[string]string{"here": shard}})
	require.NoError(t, err)
	t.Cleanup(func() { _ = db.Close() })
	grant := store.LimitGrant{OrganizationID: "org-" + uuid.NewString(), Subject: usage.SubjectOrganization,
		AmountMicros: 1_000_000, Reason: "test", Actor: "test"}
	require.NoError(t, db.AddGrant(ctx, &grant, store.LimitAudit{CustomerID: grant.OrganizationID, Source: store.AuditSourceAPI}))

	// The shared database cannot count anything: limit_grants is held by another session.
	locked, err := db.DB().BeginTx(ctx, nil)
	require.NoError(t, err)
	defer locked.Rollback() //nolint:errcheck // nothing was written
	_, err = locked.ExecContext(ctx, "LOCK TABLE limit_grants IN ACCESS EXCLUSIVE MODE")
	require.NoError(t, err)

	written, counted := &told{}, &told{}
	writer := NewWriter(WriterOptions{Store: db, Logger: slog.New(slog.DiscardHandler), Written: written.note, Counted: counted.note,
		WriteTimeout: 200 * time.Millisecond})
	customer := uuid.NewString()
	row := func() queued {
		return queued{request: store.Request{
			UsageID: uuid.Must(uuid.NewV7()), Modality: "llm", CustomerID: customer, OrganizationID: grant.OrganizationID,
			Provider: "openai", Model: "gpt", StartedAt: time.Now().UTC(), CostMicros: 25, Success: true, Billable: true,
			LimitGrants: []int64{grant.ID},
		}}
	}
	first, second := row(), row()
	require.True(t, writer.enqueue(first))
	require.Eventually(t, func() bool { return written.has(first.request.UsageID) }, 5*time.Second, 5*time.Millisecond,
		"the shard took the row while the shared database cannot count it")
	time.Sleep(500 * time.Millisecond)
	require.True(t, writer.enqueue(second))
	require.Eventually(t, func() bool { return written.has(second.request.UsageID) }, time.Second, 5*time.Millisecond,
		"the next batch is not held up by the count the first is waiting on")
	require.False(t, counted.has(first.request.UsageID), "nothing is told counted before it is")

	require.NoError(t, locked.Rollback())
	require.Eventually(t, func() bool { return counted.has(first.request.UsageID) && counted.has(second.request.UsageID) },
		10*time.Second, 10*time.Millisecond, "counted once the shared database answers")
	read, err := db.Grant(ctx, "", grant.ID)
	require.NoError(t, err)
	require.EqualValues(t, 50, read.UsedMicros)
	closing, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	writer.Close(closing)
}
