package store

import (
	"context"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// SandboxUsed is what this shard recorded of one sandbox's time, its meters summed, and whose
// the rows were.
type SandboxUsed struct {
	SandboxID       string `bun:"sandbox_id"`
	VCPUSeconds     int64  `bun:"vcpu_seconds"`
	MemoryGBSeconds int64  `bun:"memory_gb_seconds"`
	DiskGBSeconds   int64  `bun:"disk_gb_seconds"`
	Rows            int64  `bun:"rows"`
	CustomerID      string `bun:"customer_id"`
	OrganizationID  string `bun:"organization_id"`
	SessionID       string `bun:"session_id"`
	// Class is the sort of sandbox the rows say it was.
	Class string `bun:"class"`
	// Region is where the rows say it ran, empty on rows recorded before regions.
	Region string `bun:"region"`
}

// SandboxUsage is what this shard recorded of each of ids' time started from from until to, by
// the sandbox_id attribute its rows carry, counting only rows whose deployment attribute is
// deployment. A sandbox it recorded nothing of is not in the map.
func (s *Store) SandboxUsage(ctx context.Context, deployment string, ids []string, from, to time.Time) (map[string]SandboxUsed, error) {
	used := map[string]SandboxUsed{}
	if len(ids) == 0 {
		return used, nil
	}
	var rows []SandboxUsed
	err := s.shard.NewSelect().
		TableExpr("requests").
		ColumnExpr("attributes->>'sandbox_id' AS sandbox_id").
		ColumnExpr("COALESCE(SUM((meters->>'vcpu_seconds')::bigint), 0) AS vcpu_seconds").
		ColumnExpr("COALESCE(SUM((meters->>'memory_gb_seconds')::bigint), 0) AS memory_gb_seconds").
		ColumnExpr("COALESCE(SUM((meters->>'disk_gb_seconds')::bigint), 0) AS disk_gb_seconds").
		ColumnExpr("COUNT(*) AS rows").
		ColumnExpr("MAX(customer_id) AS customer_id").
		ColumnExpr("MAX(organization_id) AS organization_id").
		ColumnExpr("MAX(session_id) AS session_id").
		ColumnExpr("COALESCE(MAX(attributes->>'class'), '') AS class").
		ColumnExpr("COALESCE(MAX(attributes->>'region'), '') AS region").
		Where("modality = ?", SandboxModality).
		Where("success").
		Where("attributes->>'deployment' = ?", deployment).
		Where("attributes->>'sandbox_id' IN (?)", bun.In(ids)).
		Where("started_at >= ?", from).
		Where("started_at < ?", to).
		GroupExpr("attributes->>'sandbox_id'").
		Scan(ctx, &rows)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: sandbox usage: %w", err))
	}
	for _, row := range rows {
		used[row.SandboxID] = row
	}
	return used, nil
}

// SessionEnd is what this shard knows of whether a session is over.
type SessionEnd struct {
	// Known says the session has a row; ClosedAt is when it closed, nil while open.
	Known    bool
	ClosedAt *time.Time
	// Running says it recorded runtime since the time asked about: what tells a session with
	// no row, such as an incognito one, from one that is gone.
	Running bool
}

// Ended reports whether the session is over: closed, or unknown here and silent.
func (e SessionEnd) Ended() bool {
	return e.ClosedAt != nil || !e.Known && !e.Running
}

// SessionEnds is what this shard knows of whether each of ids is over, a session with no row
// counting as running while it has recorded runtime since since.
func (s *Store) SessionEnds(ctx context.Context, ids []string, since time.Time) (map[string]SessionEnd, error) {
	ends := map[string]SessionEnd{}
	if len(ids) == 0 {
		return ends, nil
	}
	var sessions []struct {
		ID       string     `bun:"id"`
		ClosedAt *time.Time `bun:"closed_at"`
	}
	if err := s.shard.NewSelect().TableExpr("agent_sessions").Column("id", "closed_at").
		Where("id IN (?)", bun.In(ids)).Scan(ctx, &sessions); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: session ends: %w", err))
	}
	for _, session := range sessions {
		ends[session.ID] = SessionEnd{Known: true, ClosedAt: session.ClosedAt}
	}
	var unknown []string
	for _, id := range ids {
		if _, known := ends[id]; !known {
			unknown = append(unknown, id)
		}
	}
	if len(unknown) == 0 {
		return ends, nil
	}
	var running []string
	err := s.shard.NewSelect().TableExpr("requests").ColumnExpr("DISTINCT session_id").
		Where("modality = ?", RuntimeModality).
		Where("session_id IN (?)", bun.In(unknown)).
		Where("started_at >= ?", since).
		Scan(ctx, &running)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: session ends: %w", err))
	}
	for _, id := range unknown {
		ends[id] = SessionEnd{}
	}
	for _, id := range running {
		ends[id] = SessionEnd{Running: true}
	}
	return ends, nil
}
