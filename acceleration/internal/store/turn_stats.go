package store

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// latencyGrowth is how much wider each histogram bin is than the one before it, and so how
// far a percentile read off the bins can be from the true one.
const latencyGrowth = 1.05

// The legs a turn's latency is counted for, as turn_latency_hourly names them.
const (
	legSTTLatency = "stt_latency"
	legLLMTTFT    = "llm_ttft"
	legTTSTTFB    = "tts_ttfb"
	legRoundtrip  = "roundtrip"
)

// RecordTurnStats counts one finished turn into its hour, in one statement: the turn
// itself, and each leg it had into that leg's histogram. A turn counted twice is counted
// twice, so the caller writes each one once.
func (s *Store) RecordTurnStats(ctx context.Context, turn TurnSample) error {
	if turn.CustomerID == "" || turn.AgentID == "" {
		return errors.New("store: a customer and an agent id are required")
	}
	if turn.StartedAt.IsZero() {
		turn.StartedAt = time.Now()
	}
	hour := turn.StartedAt.UTC().Truncate(time.Hour)
	interrupted := 0
	if turn.Interrupted {
		interrupted = 1
	}

	counted := `
INSERT INTO turn_stats_hourly (customer_id, agent_id, bucket, turn_count, interrupted_count, audio_out_ms_total)
VALUES (?, ?, ?, 1, ?, ?)
ON CONFLICT (customer_id, agent_id, bucket) DO UPDATE SET
    turn_count = turn_stats_hourly.turn_count + 1,
    interrupted_count = turn_stats_hourly.interrupted_count + EXCLUDED.interrupted_count,
    audio_out_ms_total = turn_stats_hourly.audio_out_ms_total + EXCLUDED.audio_out_ms_total`
	args := []any{turn.CustomerID, turn.AgentID, hour, interrupted, turn.AudioOutMs}

	var rows []string
	for _, leg := range []struct {
		name string
		ms   *float64
	}{
		{legSTTLatency, turn.STTLatencyMs},
		{legLLMTTFT, turn.LLMTTFTMs},
		{legTTSTTFB, turn.TTSTTFBMs},
		{legRoundtrip, turn.RoundtripMs},
	} {
		if leg.ms == nil {
			continue
		}
		rows = append(rows, "(?, ?, ?, ?, ?, 1)")
		args = append(args, turn.CustomerID, turn.AgentID, hour, leg.name, latencyBin(*leg.ms))
	}

	query := counted
	if len(rows) > 0 {
		query = "WITH counted AS (" + counted + `)
INSERT INTO turn_latency_hourly (customer_id, agent_id, bucket, leg, bin, count)
VALUES ` + strings.Join(rows, ", ") + `
ON CONFLICT (customer_id, agent_id, bucket, leg, bin) DO UPDATE SET
    count = turn_latency_hourly.count + 1`
	}
	if _, err := s.shard.ExecContext(ctx, query, args...); err != nil {
		return fmt.Errorf("store: record turn stats: %w", err)
	}
	return nil
}

// CustomerTurnStats returns the conversational latency buckets for one customer, oldest
// first. An agent id narrows it to one agent. A day is the hours in it from from to to.
func (s *Store) CustomerTurnStats(
	ctx context.Context,
	customerID, agentID string,
	granularity Granularity,
	from, to time.Time,
) ([]TurnBucket, error) {
	if !granularity.Valid() {
		return nil, stack.Wrap(fmt.Errorf("store: unknown granularity %q", granularity))
	}
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}
	bucket := fmt.Sprintf("date_trunc('%s', bucket)", granularity.truncateUnit())

	type binRow struct {
		AgentID string    `bun:"agent_id"`
		Leg     string    `bun:"leg"`
		Bin     int       `bun:"bin"`
		Bucket  time.Time `bun:"bucket"`
		Count   int64     `bun:"count"`
	}
	type shardRows struct {
		buckets []TurnBucket
		bins    []binRow
	}
	answers, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) (shardRows, error) {
		counts := db.NewSelect().
			Table("turn_stats_hourly").
			ColumnExpr("customer_id, agent_id").
			ColumnExpr(bucket+" AS bucket").
			ColumnExpr("SUM(turn_count) AS turn_count").
			ColumnExpr("SUM(interrupted_count) AS interrupted_count").
			ColumnExpr("SUM(audio_out_ms_total) AS audio_out_ms_total").
			Where("customer_id = ?", customerID).
			Where("bucket >= ?", from).
			Where("bucket < ?", to).
			GroupExpr("customer_id, agent_id, " + bucket).
			OrderExpr(bucket + " ASC, agent_id ASC")
		bins := db.NewSelect().
			Table("turn_latency_hourly").
			ColumnExpr("agent_id, leg, bin").
			ColumnExpr(bucket+" AS bucket").
			ColumnExpr("SUM(count) AS count").
			Where("customer_id = ?", customerID).
			Where("bucket >= ?", from).
			Where("bucket < ?", to).
			GroupExpr("agent_id, leg, bin, " + bucket).
			OrderExpr("bin ASC")
		if agentID != "" {
			counts = counts.Where("agent_id = ?", agentID)
			bins = bins.Where("agent_id = ?", agentID)
		}
		var read shardRows
		if err := counts.Scan(ctx, &read.buckets); err != nil {
			return read, fmt.Errorf("store: customer turn stats: %w", err)
		}
		if err := bins.Scan(ctx, &read.bins); err != nil {
			return read, fmt.Errorf("store: customer turn latency: %w", err)
		}
		return read, nil
	})
	if err != nil {
		return nil, stack.Wrap(err)
	}

	// Each shard's buckets are summed into one per agent and bucket, and its bins kept in
	// ascending order, where a bin two shards both counted is two entries side by side.
	buckets := answers[0].buckets
	counted := answers[0].bins
	if len(answers) > 1 {
		type bucketKey struct {
			agentID string
			bucket  int64
		}
		sums := map[bucketKey]*TurnBucket{}
		buckets, counted = nil, nil
		for _, answer := range answers {
			for _, b := range answer.buckets {
				key := bucketKey{b.AgentID, b.Bucket.Unix()}
				if sum := sums[key]; sum != nil {
					sum.TurnCount += b.TurnCount
					sum.InterruptedCount += b.InterruptedCount
					sum.AudioOutMsTotal += b.AudioOutMsTotal
					continue
				}
				sums[key] = &b
			}
			counted = append(counted, answer.bins...)
		}
		for _, sum := range sums {
			buckets = append(buckets, *sum)
		}
		slices.SortFunc(buckets, func(a, b TurnBucket) int {
			return cmp.Or(a.Bucket.Compare(b.Bucket), strings.Compare(a.AgentID, b.AgentID))
		})
		slices.SortStableFunc(counted, func(a, b binRow) int { return cmp.Compare(a.Bin, b.Bin) })
	}

	type histogramKey struct {
		agentID, leg string
		bucket       int64
	}
	histograms := map[histogramKey]*histogram{}
	for _, row := range counted {
		key := histogramKey{row.AgentID, row.Leg, row.Bucket.Unix()}
		if histograms[key] == nil {
			histograms[key] = &histogram{}
		}
		histograms[key].add(row.Bin, row.Count)
	}
	for i := range buckets {
		of := func(leg string) *histogram {
			return histograms[histogramKey{buckets[i].AgentID, leg, buckets[i].Bucket.Unix()}]
		}
		stt, llm, tts, roundtrip := of(legSTTLatency), of(legLLMTTFT), of(legTTSTTFB), of(legRoundtrip)
		buckets[i].STTLatencyP50Ms, buckets[i].STTLatencyP95Ms = stt.percentile(0.5), stt.percentile(0.95)
		buckets[i].LLMTTFTP50Ms, buckets[i].LLMTTFTP95Ms = llm.percentile(0.5), llm.percentile(0.95)
		buckets[i].TTSTTFBP50Ms, buckets[i].TTSTTFBP95Ms = tts.percentile(0.5), tts.percentile(0.95)
		buckets[i].RoundtripP50Ms = roundtrip.percentile(0.5)
		buckets[i].RoundtripP95Ms = roundtrip.percentile(0.95)
		buckets[i].RoundtripP99Ms = roundtrip.percentile(0.99)
	}
	return buckets, nil
}

// latencyBin is the histogram bin a latency falls in: bin b holds (growth^(b-1), growth^b]
// ms, and anything up to a millisecond is bin 0.
func latencyBin(ms float64) int {
	if ms <= 1 {
		return 0
	}
	return int(math.Ceil(math.Log(ms) / math.Log(latencyGrowth)))
}

// histogram is one leg's counted bins, in ascending order.
type histogram struct {
	bins   []int
	counts []int64
	total  int64
}

func (h *histogram) add(bin int, count int64) {
	h.bins = append(h.bins, bin)
	h.counts = append(h.counts, count)
	h.total += count
}

// percentile is the nearest-rank p-th percentile, as the middle of the bin it falls in, or
// nil for a leg no turn had.
func (h *histogram) percentile(p float64) *float64 {
	if h == nil || h.total == 0 {
		return nil
	}
	rank := int64(math.Ceil(p * float64(h.total)))
	var seen int64
	for i, count := range h.counts {
		seen += count
		if seen >= rank {
			ms := math.Pow(latencyGrowth, float64(h.bins[i])-0.5)
			return &ms
		}
	}
	return nil
}
