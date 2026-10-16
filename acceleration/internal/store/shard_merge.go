package store

import (
	"cmp"
	"slices"
	"strings"
	"time"
)

// What several shards answered for one aggregate, put together as one shard's answer would
// read. Sums and counts add up exactly. A percentile does not: two shards' medians do not make
// the median of both, so a merged one is each shard's weighted by the requests behind it, an
// estimate that is exact while a customer's traffic for the bucket ran in one shard.

// weighted is the mean of values weighted by counts, nil when none has a value.
func weighted(values []*float64, counts []int64) *float64 {
	var sum float64
	var total int64
	for i, value := range values {
		if value != nil && counts[i] > 0 {
			sum += *value * float64(counts[i])
			total += counts[i]
		}
	}
	if total == 0 {
		return nil
	}
	mean := sum / float64(total)
	return &mean
}

// uptimeOf is the share of requests that did not fail, nil for none.
func uptimeOf(requests, errors int64) *float64 {
	if requests == 0 {
		return nil
	}
	uptime := float64(requests-errors) / float64(requests)
	return &uptime
}

// mergeBuckets adds up the provider buckets several shards returned, oldest bucket first.
func mergeBuckets(lists [][]Bucket) []Bucket {
	if len(lists) == 1 {
		return lists[0]
	}
	type key struct {
		modality, customer, provider, model string
		bucket                              time.Time
	}
	var order []key
	sums := map[key]*Bucket{}
	p50s, p95s, counts := map[key][]*float64{}, map[key][]*float64{}, map[key][]int64{}
	for _, list := range lists {
		for _, b := range list {
			k := key{b.Modality, b.CustomerID, b.Provider, b.Model, b.Bucket.UTC()}
			sum, seen := sums[k]
			if !seen {
				sum = &Bucket{Modality: b.Modality, CustomerID: b.CustomerID, Provider: b.Provider, Model: b.Model, Bucket: b.Bucket}
				sums[k] = sum
				order = append(order, k)
			}
			sum.AudioMsTotal += b.AudioMsTotal
			sum.CharactersTotal += b.CharactersTotal
			sum.InputTokensTotal += b.InputTokensTotal
			sum.CachedInputTokensTotal += b.CachedInputTokensTotal
			sum.OutputTokensTotal += b.OutputTokensTotal
			sum.ImagesTotal += b.ImagesTotal
			sum.CostMicrosTotal += b.CostMicrosTotal
			sum.RequestCount += b.RequestCount
			sum.ErrorCount += b.ErrorCount
			p50s[k] = append(p50s[k], b.LatencyP50Ms)
			p95s[k] = append(p95s[k], b.LatencyP95Ms)
			counts[k] = append(counts[k], b.RequestCount)
		}
	}
	var out []Bucket
	for _, k := range order {
		b := *sums[k]
		b.LatencyP50Ms, b.LatencyP95Ms = weighted(p50s[k], counts[k]), weighted(p95s[k], counts[k])
		b.Uptime = uptimeOf(b.RequestCount, b.ErrorCount)
		out = append(out, b)
	}
	slices.SortStableFunc(out, func(a, b Bucket) int {
		return cmp.Or(a.Bucket.Compare(b.Bucket), strings.Compare(a.Provider, b.Provider), strings.Compare(a.Model, b.Model))
	})
	return out
}

// mergeTagBuckets adds up the cost tag buckets several shards returned, oldest bucket first.
func mergeTagBuckets(lists [][]TagBucket) []TagBucket {
	if len(lists) == 1 {
		return lists[0]
	}
	type key struct {
		modality, customer, tag, value string
		bucket                         time.Time
	}
	var order []key
	sums := map[key]*TagBucket{}
	p50s, p95s, counts := map[key][]*float64{}, map[key][]*float64{}, map[key][]int64{}
	for _, list := range lists {
		for _, b := range list {
			k := key{b.Modality, b.CustomerID, b.TagKey, b.TagValue, b.Bucket.UTC()}
			sum, seen := sums[k]
			if !seen {
				sum = &TagBucket{Modality: b.Modality, CustomerID: b.CustomerID, TagKey: b.TagKey, TagValue: b.TagValue, Bucket: b.Bucket}
				sums[k] = sum
				order = append(order, k)
			}
			sum.AudioMsTotal += b.AudioMsTotal
			sum.CharactersTotal += b.CharactersTotal
			sum.InputTokensTotal += b.InputTokensTotal
			sum.CachedInputTokensTotal += b.CachedInputTokensTotal
			sum.OutputTokensTotal += b.OutputTokensTotal
			sum.ImagesTotal += b.ImagesTotal
			sum.CostMicrosTotal += b.CostMicrosTotal
			sum.RequestCount += b.RequestCount
			sum.ErrorCount += b.ErrorCount
			p50s[k] = append(p50s[k], b.LatencyP50Ms)
			p95s[k] = append(p95s[k], b.LatencyP95Ms)
			counts[k] = append(counts[k], b.RequestCount)
		}
	}
	var out []TagBucket
	for _, k := range order {
		b := *sums[k]
		b.LatencyP50Ms, b.LatencyP95Ms = weighted(p50s[k], counts[k]), weighted(p95s[k], counts[k])
		b.Uptime = uptimeOf(b.RequestCount, b.ErrorCount)
		out = append(out, b)
	}
	slices.SortStableFunc(out, func(a, b TagBucket) int {
		return cmp.Or(a.Bucket.Compare(b.Bucket), strings.Compare(a.TagValue, b.TagValue))
	})
	return out
}

// spendRow is one group's spend in one bucket, before the smallest groups are folded.
type spendRow struct {
	Bucket          time.Time `bun:"bucket"`
	Value           string    `bun:"value"`
	CostMicrosTotal int64     `bun:"cost_micros_total"`
	RequestCount    int64     `bun:"request_count"`
}

// foldSpend keeps the limit biggest groups over the whole window and sums the rest into
// spendOther, the way CustomerSpend's query does for one shard: rows with no value stay
// their own group. It reads every shard's unfolded rows, so the folding is exact.
func foldSpend(lists [][]spendRow, limit int) []SpendBucket {
	type total struct {
		cost, requests int64
	}
	totals := map[string]*total{}
	for _, list := range lists {
		for _, row := range list {
			if row.Value == "" {
				continue
			}
			if totals[row.Value] == nil {
				totals[row.Value] = &total{}
			}
			totals[row.Value].cost += row.CostMicrosTotal
			totals[row.Value].requests += row.RequestCount
		}
	}
	ranked := slices.SortedFunc(func(yield func(string) bool) {
		for value := range totals {
			if !yield(value) {
				return
			}
		}
	}, func(a, b string) int {
		return cmp.Or(cmp.Compare(totals[b].cost, totals[a].cost), cmp.Compare(totals[b].requests, totals[a].requests), strings.Compare(a, b))
	})
	kept := map[string]bool{}
	for _, value := range ranked[:min(limit, len(ranked))] {
		kept[value] = true
	}

	type key struct {
		bucket time.Time
		value  string
	}
	var order []key
	sums := map[key]*SpendBucket{}
	for _, list := range lists {
		for _, row := range list {
			value := row.Value
			if value != "" && !kept[value] {
				value = spendOther
			}
			k := key{row.Bucket.UTC(), value}
			sum, seen := sums[k]
			if !seen {
				sum = &SpendBucket{Bucket: row.Bucket, Value: value}
				sums[k] = sum
				order = append(order, k)
			}
			sum.CostMicrosTotal += row.CostMicrosTotal
			sum.RequestCount += row.RequestCount
		}
	}
	var out []SpendBucket
	for _, k := range order {
		out = append(out, *sums[k])
	}
	slices.SortStableFunc(out, func(a, b SpendBucket) int {
		return cmp.Or(a.Bucket.Compare(b.Bucket), cmp.Compare(b.CostMicrosTotal, a.CostMicrosTotal), strings.Compare(a.Value, b.Value))
	})
	return out
}
