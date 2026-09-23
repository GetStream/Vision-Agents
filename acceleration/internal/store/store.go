// Package store persists request rows in Postgres and aggregates them into hourly and
// daily rollups. Uptime, latency percentiles, billable usage and cost all come from the
// same request rows, so there is no separate health-probe pipeline to keep in sync.
//
// Rows from every modality share one table, distinguished by a modality column, because
// every question worth asking about usage is asked the same way whatever the modality is.
package store

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/pressly/goose/v3"
	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/pgdialect"
	"github.com/uptrace/bun/driver/pgdriver"

	"github.com/GetStream/Vision-Agents/acceleration/migrations"
)

// Store reads and writes router statistics.
type Store struct {
	db       *bun.DB
	LogDrops atomic.Int64
}

// Open connects to Postgres using a pgdriver DSN, for example
// postgres://user:pass@localhost:5432/router?sslmode=disable.
func Open(dsn string) (*Store, error) {
	if dsn == "" {
		return nil, errors.New("store: dsn is required")
	}

	sqldb := sql.OpenDB(pgdriver.NewConnector(pgdriver.WithDSN(dsn)))
	return &Store{db: bun.NewDB(sqldb, pgdialect.New())}, nil
}

// DB exposes the bun handle so callers can run queries this store does not wrap.
func (s *Store) DB() *bun.DB { return s.db }

// Close releases the connection pool.
func (s *Store) Close() error { return s.db.Close() }

// Ping verifies the connection is usable.
func (s *Store) Ping(ctx context.Context) error { return s.db.PingContext(ctx) }

// Migrate applies every pending migration.
func (s *Store) Migrate(ctx context.Context) error {
	goose.SetBaseFS(migrations.FS)
	if err := goose.SetDialect("postgres"); err != nil {
		return fmt.Errorf("store: set dialect: %w", err)
	}
	if err := goose.UpContext(ctx, s.db.DB, "."); err != nil {
		return fmt.Errorf("store: migrate: %w", err)
	}
	return nil
}

// RecordRequest stores one request. Latency is optional because a request that failed
// before reaching the provider has none.
func (s *Store) RecordRequest(ctx context.Context, request *Request) error {
	request.ErrorMessage = SafeLogText(request.ErrorMessage)
	if request.CustomerID == "" {
		return errors.New("store: customer id is required")
	}
	if request.Modality == "" {
		return errors.New("store: modality is required")
	}
	if request.StartedAt.IsZero() {
		request.StartedAt = time.Now().UTC()
	}

	if _, err := s.db.NewInsert().Model(request).Exec(ctx); err != nil {
		return fmt.Errorf("store: record request: %w", err)
	}
	return nil
}

// Rollup aggregates the requests in [from, to) into the rollup tables for the given
// granularity, both the provider breakdown and the cost-tag breakdown, and returns how
// many buckets were written across the two. It is idempotent: re-running it recomputes
// the buckets it touches, so a missed run is fixed by running it again over the same
// window.
func (s *Store) Rollup(ctx context.Context, granularity Granularity, from, to time.Time) (int64, error) {
	if !granularity.Valid() {
		return 0, fmt.Errorf("store: unknown granularity %q", granularity)
	}
	if !to.After(from) {
		return 0, fmt.Errorf("store: rollup window must be non-empty, got %s to %s", from, to)
	}

	providers, err := s.rollupProviders(ctx, granularity, from, to)
	if err != nil {
		return 0, err
	}
	tags, err := s.rollupTags(ctx, granularity, from, to)
	if err != nil {
		return 0, err
	}
	turns, err := s.rollupTurns(ctx, granularity, from, to)
	if err != nil {
		return 0, err
	}
	return providers + tags + turns, nil
}

func (s *Store) rollupProviders(ctx context.Context, granularity Granularity, from, to time.Time) (int64, error) {
	query := fmt.Sprintf(`
INSERT INTO %s (
    modality, customer_id, provider, model, bucket,
    audio_ms_total, characters_total,
    input_tokens_total, cached_input_tokens_total, output_tokens_total,
    cost_micros_total,
    request_count, error_count,
    latency_p50_ms, latency_p95_ms
)
SELECT
    modality,
    customer_id,
    provider,
    model,
    date_trunc('%s', started_at) AS bucket,
    COALESCE(SUM(audio_ms), 0),
    COALESCE(SUM(characters), 0),
    COALESCE(SUM(input_tokens), 0),
    COALESCE(SUM(cached_input_tokens), 0),
    COALESCE(SUM(output_tokens), 0),
    COALESCE(SUM(cost_micros), 0),
    COUNT(*),
    COUNT(*) FILTER (WHERE NOT success),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms),
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)
FROM requests
WHERE started_at >= ? AND started_at < ?
GROUP BY modality, customer_id, provider, model, bucket
ON CONFLICT (modality, customer_id, provider, model, bucket) DO UPDATE SET
    audio_ms_total = EXCLUDED.audio_ms_total,
    characters_total = EXCLUDED.characters_total,
    input_tokens_total = EXCLUDED.input_tokens_total,
    cached_input_tokens_total = EXCLUDED.cached_input_tokens_total,
    output_tokens_total = EXCLUDED.output_tokens_total,
    cost_micros_total = EXCLUDED.cost_micros_total,
    request_count = EXCLUDED.request_count,
    error_count = EXCLUDED.error_count,
    latency_p50_ms = EXCLUDED.latency_p50_ms,
    latency_p95_ms = EXCLUDED.latency_p95_ms`,
		granularity.table(), granularity.truncateUnit())

	result, err := s.db.ExecContext(ctx, query, from, to)
	if err != nil {
		return 0, fmt.Errorf("store: rollup %s: %w", granularity, err)
	}

	affected, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("store: rollup %s: %w", granularity, err)
	}
	return affected, nil
}

// rollupTags aggregates the same window by cost tag. Each request is unrolled into one
// row per label it carries, so a request tagged with a project and an environment counts
// towards both breakdowns.
func (s *Store) rollupTags(ctx context.Context, granularity Granularity, from, to time.Time) (int64, error) {
	query := fmt.Sprintf(`
INSERT INTO %s (
    modality, customer_id, tag_key, tag_value, bucket,
    audio_ms_total, characters_total,
    input_tokens_total, cached_input_tokens_total, output_tokens_total,
    cost_micros_total,
    request_count, error_count,
    latency_p50_ms, latency_p95_ms
)
SELECT
    r.modality,
    r.customer_id,
    tag.key,
    tag.value,
    date_trunc('%s', r.started_at) AS bucket,
    COALESCE(SUM(r.audio_ms), 0),
    COALESCE(SUM(r.characters), 0),
    COALESCE(SUM(r.input_tokens), 0),
    COALESCE(SUM(r.cached_input_tokens), 0),
    COALESCE(SUM(r.output_tokens), 0),
    COALESCE(SUM(r.cost_micros), 0),
    COUNT(*),
    COUNT(*) FILTER (WHERE NOT r.success),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY r.latency_ms),
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY r.latency_ms)
FROM requests AS r
CROSS JOIN LATERAL jsonb_each_text(r.tags) AS tag(key, value)
WHERE r.started_at >= ? AND r.started_at < ?
GROUP BY r.modality, r.customer_id, tag.key, tag.value, bucket
ON CONFLICT (modality, customer_id, tag_key, tag_value, bucket) DO UPDATE SET
    audio_ms_total = EXCLUDED.audio_ms_total,
    characters_total = EXCLUDED.characters_total,
    input_tokens_total = EXCLUDED.input_tokens_total,
    cached_input_tokens_total = EXCLUDED.cached_input_tokens_total,
    output_tokens_total = EXCLUDED.output_tokens_total,
    cost_micros_total = EXCLUDED.cost_micros_total,
    request_count = EXCLUDED.request_count,
    error_count = EXCLUDED.error_count,
    latency_p50_ms = EXCLUDED.latency_p50_ms,
    latency_p95_ms = EXCLUDED.latency_p95_ms`,
		granularity.tagTable(), granularity.truncateUnit())

	result, err := s.db.ExecContext(ctx, query, from, to)
	if err != nil {
		return 0, fmt.Errorf("store: rollup %s tags: %w", granularity, err)
	}

	affected, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("store: rollup %s tags: %w", granularity, err)
	}
	return affected, nil
}

// RecordTurn stores one conversational turn.
func (s *Store) RecordTurn(ctx context.Context, turn *Turn) error {
	if turn.CustomerID == "" {
		return errors.New("store: customer id is required")
	}
	if turn.AgentID == "" {
		return errors.New("store: agent id is required")
	}
	if turn.TurnID == "" {
		return errors.New("store: turn id is required")
	}
	if turn.StartedAt.IsZero() {
		turn.StartedAt = time.Now().UTC()
	}

	// A turn is recorded once, but a retry after a write that did land must not double
	// count it, so the agent's own turn id settles which row wins.
	_, err := s.db.NewInsert().Model(turn).
		On("CONFLICT (agent_id, turn_id) DO NOTHING").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: record turn: %w", err)
	}
	return nil
}

// rollupTurns aggregates conversational turns into per-agent latency percentiles. p99 is
// carried for the roundtrip only, since that is the number a conversation is judged on.
func (s *Store) rollupTurns(ctx context.Context, granularity Granularity, from, to time.Time) (int64, error) {
	query := fmt.Sprintf(`
INSERT INTO %s (
    customer_id, agent_id, bucket,
    turn_count, interrupted_count, audio_out_ms_total,
    stt_latency_p50_ms, stt_latency_p95_ms,
    llm_ttft_p50_ms, llm_ttft_p95_ms,
    tts_ttfb_p50_ms, tts_ttfb_p95_ms,
    roundtrip_p50_ms, roundtrip_p95_ms, roundtrip_p99_ms
)
SELECT
    customer_id,
    agent_id,
    date_trunc('%s', started_at) AS bucket,
    COUNT(*),
    COUNT(*) FILTER (WHERE interrupted),
    COALESCE(SUM(audio_out_ms), 0),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY stt_latency_ms),
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY stt_latency_ms),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY llm_ttft_ms),
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY llm_ttft_ms),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tts_ttfb_ms),
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY tts_ttfb_ms),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY roundtrip_ms),
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY roundtrip_ms),
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY roundtrip_ms)
FROM turns
WHERE started_at >= ? AND started_at < ?
GROUP BY customer_id, agent_id, bucket
ON CONFLICT (customer_id, agent_id, bucket) DO UPDATE SET
    turn_count = EXCLUDED.turn_count,
    interrupted_count = EXCLUDED.interrupted_count,
    audio_out_ms_total = EXCLUDED.audio_out_ms_total,
    stt_latency_p50_ms = EXCLUDED.stt_latency_p50_ms,
    stt_latency_p95_ms = EXCLUDED.stt_latency_p95_ms,
    llm_ttft_p50_ms = EXCLUDED.llm_ttft_p50_ms,
    llm_ttft_p95_ms = EXCLUDED.llm_ttft_p95_ms,
    tts_ttfb_p50_ms = EXCLUDED.tts_ttfb_p50_ms,
    tts_ttfb_p95_ms = EXCLUDED.tts_ttfb_p95_ms,
    roundtrip_p50_ms = EXCLUDED.roundtrip_p50_ms,
    roundtrip_p95_ms = EXCLUDED.roundtrip_p95_ms,
    roundtrip_p99_ms = EXCLUDED.roundtrip_p99_ms`,
		granularity.turnTable(), granularity.truncateUnit())

	result, err := s.db.ExecContext(ctx, query, from, to)
	if err != nil {
		return 0, fmt.Errorf("store: rollup %s turns: %w", granularity, err)
	}

	affected, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("store: rollup %s turns: %w", granularity, err)
	}
	return affected, nil
}

// CustomerTurnStats returns the conversational latency buckets for one customer, oldest
// first. An agent id narrows it to one agent.
func (s *Store) CustomerTurnStats(
	ctx context.Context,
	customerID, agentID string,
	granularity Granularity,
	from, to time.Time,
) ([]TurnBucket, error) {
	if !granularity.Valid() {
		return nil, fmt.Errorf("store: unknown granularity %q", granularity)
	}
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}

	query := s.db.NewSelect().
		Table(granularity.turnTable()).
		Where("customer_id = ?", customerID).
		Where("bucket >= ?", from).
		Where("bucket < ?", to)
	if agentID != "" {
		query = query.Where("agent_id = ?", agentID)
	}

	var buckets []TurnBucket
	if err := query.Order("bucket ASC", "agent_id ASC").Scan(ctx, &buckets); err != nil {
		return nil, fmt.Errorf("store: customer turn stats: %w", err)
	}
	return buckets, nil
}

// CustomerStats returns the buckets for one customer and modality in [from, to), oldest
// first.
//
// Reads recorded requests directly so recent activity is visible before rollups run.
func (s *Store) CustomerStats(
	ctx context.Context,
	modality, customerID string,
	granularity Granularity,
	from, to time.Time,
	tags map[string]string,
) ([]Bucket, error) {
	if !granularity.Valid() {
		return nil, fmt.Errorf("store: unknown granularity %q", granularity)
	}
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}
	if modality == "" {
		return nil, errors.New("store: modality is required")
	}

	if tags == nil {
		tags = map[string]string{}
	}
	return s.taggedStats(ctx, modality, customerID, granularity, from, to, tags)
}

// taggedStats aggregates raw request rows carrying every one of the given labels.
func (s *Store) taggedStats(
	ctx context.Context,
	modality, customerID string,
	granularity Granularity,
	from, to time.Time,
	tags map[string]string,
) ([]Bucket, error) {
	filter, err := json.Marshal(tags)
	if err != nil {
		return nil, fmt.Errorf("store: customer stats: encode tag filter: %w", err)
	}

	query := fmt.Sprintf(`
SELECT
    modality,
    customer_id,
    provider,
    model,
    date_trunc('%s', started_at) AS bucket,
    COALESCE(SUM(audio_ms), 0) AS audio_ms_total,
    COALESCE(SUM(characters), 0) AS characters_total,
    COALESCE(SUM(input_tokens), 0) AS input_tokens_total,
    COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens_total,
    COALESCE(SUM(output_tokens), 0) AS output_tokens_total,
    COALESCE(SUM(cost_micros), 0) AS cost_micros_total,
    COUNT(*) AS request_count,
    COUNT(*) FILTER (WHERE NOT success) AS error_count,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) AS latency_p50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS latency_p95_ms,
    (COUNT(*) - COUNT(*) FILTER (WHERE NOT success))::double precision
        / NULLIF(COUNT(*), 0) AS uptime
FROM requests
WHERE modality = ? AND customer_id = ?
  AND started_at >= ? AND started_at < ?
  AND tags @> ?::jsonb
GROUP BY modality, customer_id, provider, model, bucket
ORDER BY bucket ASC, provider ASC, model ASC`, granularity.truncateUnit())

	var buckets []Bucket
	if err := s.db.NewRaw(query, modality, customerID, from, to, string(filter)).Scan(ctx, &buckets); err != nil {
		return nil, fmt.Errorf("store: customer stats by tag: %w", err)
	}
	return buckets, nil
}

// CustomerTagStats returns what each value of one tag key cost, oldest bucket first. It
// is the "what drives our spend" query: group by project, or by environment, or by
// whichever label the customer bills on.
func (s *Store) CustomerTagStats(
	ctx context.Context,
	modality, customerID, tagKey string,
	granularity Granularity,
	from, to time.Time,
) ([]TagBucket, error) {
	if !granularity.Valid() {
		return nil, fmt.Errorf("store: unknown granularity %q", granularity)
	}
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}
	if modality == "" {
		return nil, errors.New("store: modality is required")
	}
	if tagKey == "" {
		return nil, errors.New("store: tag key is required")
	}

	var buckets []TagBucket
	err := s.db.NewSelect().
		Table(granularity.tagTable()).
		Where("modality = ?", modality).
		Where("customer_id = ?", customerID).
		Where("tag_key = ?", tagKey).
		Where("bucket >= ?", from).
		Where("bucket < ?", to).
		Order("bucket ASC", "tag_value ASC").
		Scan(ctx, &buckets)
	if err != nil {
		return nil, fmt.Errorf("store: customer tag stats: %w", err)
	}
	return buckets, nil
}

// spendGroupByModality is the group_by that means "where the money went" rather than
// "what it was spent on", and the only one that is not a cost label key.
const spendGroupByModality = "modality"

// spendOther is the value every group outside the biggest few is summed into.
const spendOther = "other"

// CustomerSpend returns what one customer spent per bucket and group, oldest bucket first.
// It is the whole bill rather than one modality's share of it, which is what a spend trend
// is read as.
//
// groupBy is either "modality" or a cost label key. Only the limit biggest values over the
// whole window keep a group of their own: a label such as customer_id has as many values as
// the customer has customers, and a chart of all of them says nothing. The rest are summed
// into "other", and requests carrying no such label into the empty value, so the rows still
// add up to the total.
//
// Reads the request rows rather than the rollups, so today's spend is there before a
// rollup has run.
func (s *Store) CustomerSpend(
	ctx context.Context,
	customerID, groupBy string,
	granularity Granularity,
	from, to time.Time,
	limit int,
	tags map[string]string,
) ([]SpendBucket, error) {
	if !granularity.Valid() {
		return nil, fmt.Errorf("store: unknown granularity %q", granularity)
	}
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}
	if limit < 1 {
		return nil, errors.New("store: limit must be at least 1")
	}
	if tags == nil {
		tags = map[string]string{}
	}
	filter, err := json.Marshal(tags)
	if err != nil {
		return nil, fmt.Errorf("store: customer spend: encode tag filter: %w", err)
	}

	valueExpr := "modality"
	args := []any{}
	if groupBy != spendGroupByModality {
		valueExpr = "COALESCE(tags->>?, '')"
		args = append(args, groupBy)
	}
	args = append(args, customerID, from, to, string(filter), limit)

	query := fmt.Sprintf(`
WITH grouped AS (
    SELECT
        date_trunc('%s', started_at) AS bucket,
        %s AS value,
        cost_micros
    FROM requests
    WHERE customer_id = ?
      AND started_at >= ? AND started_at < ?
      AND tags @> ?::jsonb
),
biggest AS (
    SELECT
        value,
        ROW_NUMBER() OVER (
            ORDER BY SUM(cost_micros) DESC, COUNT(*) DESC, value ASC
        ) AS rank
    FROM grouped
    WHERE value <> ''
    GROUP BY value
),
folded AS (
    SELECT
        g.bucket AS bucket,
        CASE
            WHEN g.value = '' THEN ''
            WHEN b.rank > ? THEN '%s'
            ELSE g.value
        END AS value,
        g.cost_micros AS cost_micros
    FROM grouped AS g
    LEFT JOIN biggest AS b ON b.value = g.value
)
SELECT
    bucket,
    value,
    COALESCE(SUM(cost_micros), 0) AS cost_micros_total,
    COUNT(*) AS request_count
FROM folded
GROUP BY bucket, value
ORDER BY bucket ASC, cost_micros_total DESC, value ASC`,
		granularity.truncateUnit(), valueExpr, spendOther)

	var buckets []SpendBucket
	if err := s.db.NewRaw(query, args...).Scan(ctx, &buckets); err != nil {
		return nil, fmt.Errorf("store: customer spend: %w", err)
	}
	return buckets, nil
}

// tagKeyRow is one cost label key paired with one of its largest values, which is how the
// two levels come back from a single query.
type tagKeyRow struct {
	Key             string `bun:"key"`
	ValueCount      int64  `bun:"value_count"`
	CostMicrosTotal int64  `bun:"cost_micros_total"`
	RequestCount    int64  `bun:"request_count"`
	Value           string `bun:"value"`
	ValueCost       int64  `bun:"value_cost_micros_total"`
	ValueRequests   int64  `bun:"value_request_count"`
}

// topTagValues is how many values of a key come back with it. Enough to see what drives the
// key's spend, few enough that a key naming an end customer does not return a database.
const topTagValues = 10

// CustomerTagKeys returns which cost label keys one customer's spend carries, biggest spend
// first, each with its ten largest values.
//
// Cost labels are the customer's own, so nothing here knows in advance whether spend is
// broken down by product, by environment or by the end customer it was incurred for. What
// tells them apart is how many values a key was used with and how much of the traffic
// carries it, which is what this reports.
func (s *Store) CustomerTagKeys(
	ctx context.Context,
	customerID string,
	from, to time.Time,
	tags map[string]string,
) ([]TagKeySummary, error) {
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}
	if tags == nil {
		tags = map[string]string{}
	}
	filter, err := json.Marshal(tags)
	if err != nil {
		return nil, fmt.Errorf("store: customer tag keys: encode tag filter: %w", err)
	}

	// Coverage is measured against every request in the window, labelled or not, because a
	// key on half the traffic breaks down half the bill and reading it as the whole of it
	// is the mistake this number exists to prevent.
	var total int64
	err = s.db.NewSelect().
		Table("requests").
		ColumnExpr("COUNT(*)").
		Where("customer_id = ?", customerID).
		Where("started_at >= ?", from).
		Where("started_at < ?", to).
		Where("tags @> ?::jsonb", string(filter)).
		Scan(ctx, &total)
	if err != nil {
		return nil, fmt.Errorf("store: customer tag keys: %w", err)
	}
	if total == 0 {
		return nil, nil
	}

	query := `
WITH labelled AS (
    SELECT tag.key AS key, tag.value AS value, r.cost_micros AS cost_micros
    FROM requests AS r
    CROSS JOIN LATERAL jsonb_each_text(r.tags) AS tag(key, value)
    WHERE r.customer_id = ?
      AND r.started_at >= ? AND r.started_at < ?
      AND r.tags @> ?::jsonb
),
per_value AS (
    SELECT
        key,
        value,
        SUM(cost_micros) AS cost_micros_total,
        COUNT(*) AS request_count
    FROM labelled
    GROUP BY key, value
),
ranked AS (
    SELECT
        key, value, cost_micros_total, request_count,
        ROW_NUMBER() OVER (
            PARTITION BY key
            ORDER BY cost_micros_total DESC, request_count DESC, value ASC
        ) AS rank
    FROM per_value
),
per_key AS (
    SELECT
        key,
        COUNT(*) AS value_count,
        SUM(cost_micros_total) AS cost_micros_total,
        SUM(request_count) AS request_count
    FROM per_value
    GROUP BY key
)
SELECT
    k.key AS key,
    k.value_count AS value_count,
    k.cost_micros_total AS cost_micros_total,
    k.request_count AS request_count,
    r.value AS value,
    r.cost_micros_total AS value_cost_micros_total,
    r.request_count AS value_request_count
FROM per_key AS k
JOIN ranked AS r ON r.key = k.key AND r.rank <= ?
ORDER BY k.cost_micros_total DESC, k.request_count DESC, k.key ASC, r.rank ASC`

	var rows []tagKeyRow
	err = s.db.NewRaw(query, customerID, from, to, string(filter), topTagValues).Scan(ctx, &rows)
	if err != nil {
		return nil, fmt.Errorf("store: customer tag keys: %w", err)
	}

	var keys []TagKeySummary
	for _, row := range rows {
		if len(keys) == 0 || keys[len(keys)-1].Key != row.Key {
			keys = append(keys, TagKeySummary{
				Key:             row.Key,
				ValueCount:      row.ValueCount,
				CostMicrosTotal: row.CostMicrosTotal,
				RequestCount:    row.RequestCount,
				Coverage:        float64(row.RequestCount) / float64(total),
			})
		}
		key := &keys[len(keys)-1]
		// A deployment with no prices configured records every row at zero, so a share of
		// spend would be a division by nothing. Requests are what is left to rank by.
		share := float64(row.ValueRequests) / float64(key.RequestCount)
		if key.CostMicrosTotal > 0 {
			share = float64(row.ValueCost) / float64(key.CostMicrosTotal)
		}
		key.TopValues = append(key.TopValues, TagValueSummary{
			Value:           row.Value,
			CostMicrosTotal: row.ValueCost,
			RequestCount:    row.ValueRequests,
			Share:           share,
		})
	}
	return keys, nil
}

// CustomerActivity returns how much one customer's agents were used per bucket, oldest
// first, and by how many distinct people.
//
// A bucket with nothing in it is left out rather than returned as zeroes, the same way the
// stats paths leave out a bucket nobody used.
func (s *Store) CustomerActivity(
	ctx context.Context,
	customerID string,
	granularity ActivityGranularity,
	from, to time.Time,
) ([]ActivityBucket, error) {
	if !granularity.Valid() {
		return nil, fmt.Errorf("store: unknown activity granularity %q", granularity)
	}
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}

	unit := granularity.truncateUnit()
	query := fmt.Sprintf(`
WITH seen AS (
    SELECT
        date_trunc('%[1]s', s.created_at) AS bucket,
        COALESCE(NULLIF(g.claimed_by, ''), s.user_id) AS user_id,
        s.caller_kind AS caller_kind
    FROM agent_sessions AS s
    LEFT JOIN guest_users AS g ON g.id = s.user_id AND g.customer_id = s.customer_id
    WHERE s.customer_id = ? AND s.created_at >= ? AND s.created_at < ?
    UNION ALL
    SELECT
        date_trunc('%[1]s', a.created_at) AS bucket,
        COALESCE(NULLIF(g.claimed_by, ''), s.user_id) AS user_id,
        s.caller_kind AS caller_kind
    FROM agent_responses AS a
    JOIN agent_sessions AS s ON s.id = a.session_id
    LEFT JOIN guest_users AS g ON g.id = s.user_id AND g.customer_id = s.customer_id
    WHERE a.customer_id = ? AND a.created_at >= ? AND a.created_at < ?
),
user_counts AS (
    SELECT bucket, COUNT(*) AS n
    FROM (SELECT DISTINCT bucket, user_id FROM seen WHERE user_id <> '' AND caller_kind <> 'anonymous') AS people
    GROUP BY bucket
),
session_counts AS (
    SELECT date_trunc('%[1]s', created_at) AS bucket, COUNT(*) AS n
    FROM agent_sessions
    WHERE customer_id = ? AND created_at >= ? AND created_at < ?
    GROUP BY bucket
),
message_counts AS (
    SELECT date_trunc('%[1]s', created_at) AS bucket, COUNT(*) AS n
    FROM agent_responses
    WHERE customer_id = ? AND created_at >= ? AND created_at < ?
    GROUP BY bucket
),
call_counts AS (
    SELECT
        date_trunc('%[1]s', started_at) AS bucket,
        COUNT(*) AS n,
        SUM(EXTRACT(EPOCH FROM (COALESCE(ended_at, now()) - started_at)))::double precision / 60
            AS voice_minutes,
        COALESCE(SUM(EXTRACT(EPOCH FROM (COALESCE(ended_at, now()) - started_at)))
            FILTER (WHERE from_number IS NOT NULL OR to_number IS NOT NULL), 0)::double precision / 60
            AS phone_minutes
    FROM calls
    WHERE customer_id = ? AND started_at >= ? AND started_at < ?
    GROUP BY bucket
),
buckets AS (
    SELECT bucket FROM user_counts
    UNION SELECT bucket FROM session_counts
    UNION SELECT bucket FROM message_counts
    UNION SELECT bucket FROM call_counts
)
SELECT
    b.bucket AS bucket,
    COALESCE(u.n, 0) AS active_users,
    COALESCE(s.n, 0) AS sessions,
    COALESCE(m.n, 0) AS messages,
    COALESCE(c.n, 0) AS calls,
    COALESCE(c.voice_minutes, 0) AS voice_minutes,
    COALESCE(c.phone_minutes, 0) AS phone_minutes
FROM buckets AS b
LEFT JOIN user_counts AS u ON u.bucket = b.bucket
LEFT JOIN session_counts AS s ON s.bucket = b.bucket
LEFT JOIN message_counts AS m ON m.bucket = b.bucket
LEFT JOIN call_counts AS c ON c.bucket = b.bucket
ORDER BY b.bucket ASC`, unit)

	var buckets []ActivityBucket
	err := s.db.NewRaw(query,
		customerID, from, to,
		customerID, from, to,
		customerID, from, to,
		customerID, from, to,
		customerID, from, to,
	).Scan(ctx, &buckets)
	if err != nil {
		return nil, fmt.Errorf("store: customer activity: %w", err)
	}
	return buckets, nil
}

// ModelRequests returns how many requests each "provider/model" served for a modality
// since a time, across every customer. It is what makes a model popular, so it counts
// calls rather than spend, and is read from the raw requests so it needs no rollup.
func (s *Store) ModelRequests(ctx context.Context, modality string, since time.Time) (map[string]int64, error) {
	if modality == "" {
		return nil, errors.New("store: modality is required")
	}

	var rows []struct {
		Provider string `bun:"provider"`
		Model    string `bun:"model"`
		Requests int64  `bun:"requests"`
	}
	err := s.db.NewSelect().
		Table("requests").
		Column("provider", "model").
		ColumnExpr("COUNT(*) AS requests").
		Where("modality = ?", modality).
		Where("started_at >= ?", since).
		Group("provider", "model").
		Scan(ctx, &rows)
	if err != nil {
		return nil, fmt.Errorf("store: model requests: %w", err)
	}

	counts := make(map[string]int64, len(rows))
	for _, row := range rows {
		counts[row.Provider+"/"+row.Model] = row.Requests
	}
	return counts, nil
}

// RecordNumber stores a number a customer now holds.
func (s *Store) RecordNumber(ctx context.Context, number *PhoneNumber) error {
	if number.E164 == "" {
		return errors.New("store: a number is required")
	}
	if number.Vendor == "" {
		return errors.New("store: vendor is required")
	}
	if number.CustomerID == "" {
		return errors.New("store: customer id is required")
	}
	if number.PurchasedAt.IsZero() {
		number.PurchasedAt = time.Now().UTC()
	}
	// A vendor that says nothing about what a number carries is not the same as a null
	// column, and the array is not nullable.
	if number.Capabilities == nil {
		number.Capabilities = []string{}
	}

	if _, err := s.db.NewInsert().Model(number).Exec(ctx); err != nil {
		return fmt.Errorf("store: record number: %w", err)
	}
	return nil
}

// ReleaseNumber marks a number as given back. The row stays, because what it cost while
// it was held is still part of that month's bill.
func (s *Store) ReleaseNumber(ctx context.Context, customerID, e164 string, at time.Time) error {
	if customerID == "" || e164 == "" {
		return errors.New("store: a customer and a number are required")
	}
	if at.IsZero() {
		at = time.Now().UTC()
	}

	result, err := s.db.NewUpdate().Model((*PhoneNumber)(nil)).
		Set("released_at = ?", at).
		Where("customer_id = ?", customerID).
		Where("e164 = ?", e164).
		Where("released_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: release number: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: release number: %w", err)
	}
	if affected == 0 {
		return fmt.Errorf("store: %s is not a number %s holds", e164, customerID)
	}
	return nil
}

// AttachNumber records which SIP trunk calls to a number arrive on and which Stream call
// they land in.
//
// The call is recorded as well as the trunk because an inbound call arrives over a webhook
// that names the call, so without it there is nothing to attribute the call to.
func (s *Store) AttachNumber(ctx context.Context, customerID, e164, trunkID, callType, callID string) error {
	if customerID == "" || e164 == "" {
		return errors.New("store: a customer and a number are required")
	}
	if trunkID == "" {
		return errors.New("store: a trunk id is required")
	}

	result, err := s.db.NewUpdate().Model((*PhoneNumber)(nil)).
		Set("stream_trunk_id = ?", trunkID).
		Set("stream_call_id = ?", callID).
		Set("stream_call_type = ?", callType).
		Where("customer_id = ?", customerID).
		Where("e164 = ?", e164).
		Where("released_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: attach number: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: attach number: %w", err)
	}
	if affected == 0 {
		return fmt.Errorf("store: %s is not a number %s holds", e164, customerID)
	}
	return nil
}

// CustomerNumbers returns the numbers a customer holds, newest first. Released numbers
// are left out unless asked for, since what is normally wanted is what can be called.
func (s *Store) CustomerNumbers(ctx context.Context, customerID string, includeReleased bool) ([]PhoneNumber, error) {
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}

	query := s.db.NewSelect().Model((*PhoneNumber)(nil)).
		Where("customer_id = ?", customerID).
		Order("purchased_at DESC")
	if !includeReleased {
		query = query.Where("released_at IS NULL")
	}

	var numbers []PhoneNumber
	if err := query.Scan(ctx, &numbers); err != nil {
		return nil, fmt.Errorf("store: customer numbers: %w", err)
	}
	return numbers, nil
}

// Number returns one number a customer holds.
func (s *Store) Number(ctx context.Context, customerID, e164 string) (PhoneNumber, error) {
	if customerID == "" || e164 == "" {
		return PhoneNumber{}, errors.New("store: a customer and a number are required")
	}

	var number PhoneNumber
	err := s.db.NewSelect().Model(&number).
		Where("customer_id = ?", customerID).
		Where("e164 = ?", e164).
		Where("released_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return PhoneNumber{}, fmt.Errorf("store: %s is not a number %s holds", e164, customerID)
	}
	if err != nil {
		return PhoneNumber{}, fmt.Errorf("store: number: %w", err)
	}
	return number, nil
}

// NumberByCall returns the number whose callers land in a Stream call.
//
// This is the way back from an arriving call to the customer whose call it is: the webhook
// that reports one is app-wide and names the call rather than the number or the customer.
//
// A number attached before the call was recorded is found by the "phone-<e164>" the default
// routing rule names, which is derivable rather than stored. Without that fallback every
// number already in service would have to be attached again to answer a call.
func (s *Store) NumberByCall(ctx context.Context, callType, callID string) (PhoneNumber, error) {
	if callID == "" {
		return PhoneNumber{}, errors.New("store: a call id is required")
	}
	if callType == "" {
		callType = "agent"
	}

	var number PhoneNumber
	err := s.db.NewSelect().Model(&number).
		Where("stream_call_id = ?", callID).
		Where("stream_call_type = ?", callType).
		Where("released_at IS NULL").
		Limit(1).
		Scan(ctx)
	if err == nil {
		return number, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return PhoneNumber{}, fmt.Errorf("store: number by call: %w", err)
	}

	e164, named := strings.CutPrefix(callID, "phone-")
	if !named {
		return PhoneNumber{}, fmt.Errorf("store: no number reaches call %s:%s", callType, callID)
	}
	err = s.db.NewSelect().Model(&number).
		Where("e164 = ?", e164).
		Where("stream_trunk_id IS NOT NULL").
		Where("released_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return PhoneNumber{}, fmt.Errorf("store: no number reaches call %s:%s", callType, callID)
	}
	if err != nil {
		return PhoneNumber{}, fmt.Errorf("store: number by call: %w", err)
	}
	return number, nil
}

// newID is the handle a caller holds a row by. It is random rather than sequential because
// it is the only thing standing between two customers who both guessed at an id.
func newID() string {
	raw := make([]byte, 16)
	// rand.Read on crypto/rand never returns an error, which is why the result is not
	// checked: the alternative would be a row that could not be created.
	_, _ = rand.Read(raw)
	return hex.EncodeToString(raw)
}

// NewID is newID for callers that have to name something before the row holding it exists,
// as an uploaded file must be given an object key before it can be recorded.
func NewID() string { return newID() }
