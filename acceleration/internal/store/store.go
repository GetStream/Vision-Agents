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
	"time"

	"github.com/pressly/goose/v3"
	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/pgdialect"
	"github.com/uptrace/bun/driver/pgdriver"

	"github.com/GetStream/Vision-Agents/acceleration/migrations"
)

// Store reads and writes router statistics.
type Store struct {
	db *bun.DB
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
// Without a tag filter this reads the rollup table. With one it aggregates the matching
// request rows directly, because a rollup bucket has already lost which labels its
// requests carried and only the raw rows can answer "and also tagged with".
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

	if len(tags) > 0 {
		return s.taggedStats(ctx, modality, customerID, granularity, from, to, tags)
	}

	var buckets []Bucket
	err := s.db.NewSelect().
		Table(granularity.table()).
		Where("modality = ?", modality).
		Where("customer_id = ?", customerID).
		Where("bucket >= ?", from).
		Where("bucket < ?", to).
		Order("bucket ASC", "provider ASC", "model ASC").
		Scan(ctx, &buckets)
	if err != nil {
		return nil, fmt.Errorf("store: customer stats: %w", err)
	}
	return buckets, nil
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
