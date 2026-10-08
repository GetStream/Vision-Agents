package api

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	// popularityWindow is how far back a model's share of requests is counted.
	popularityWindow = 7 * 24 * time.Hour
	// popularityTTL is how long one count is reused. It is the same answer for every
	// customer and moves slowly, so it is not worth a scan of every request per page load.
	popularityTTL = 10 * time.Minute
)

// popularity is each model's share of its modality's requests across every customer.
type popularity struct {
	store  *store.Store
	logger *slog.Logger

	mu     sync.Mutex
	counts map[string]popularityCount
}

type popularityCount struct {
	at     time.Time
	shares map[string]float64
}

func newPopularity(pgStore *store.Store, logger *slog.Logger) *popularity {
	return &popularity{store: pgStore, logger: logger, counts: map[string]popularityCount{}}
}

// shares returns each "provider/model"'s share of a modality's requests, from 0 to 1. It
// is empty when the deployment keeps no statistics. A failed count keeps the last one,
// since a picker that cannot sort by popularity is still a picker.
func (p *popularity) shares(ctx context.Context, modality string) map[string]float64 {
	if p.store == nil {
		return nil
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	cached, ok := p.counts[modality]
	if ok && time.Since(cached.at) < popularityTTL {
		return cached.shares
	}

	requests, err := p.store.ModelRequests(ctx, modality, time.Now().Add(-popularityWindow))
	if err != nil {
		p.logger.Warn("count model popularity", "modality", modality, "error", err)
		return cached.shares
	}
	var total int64
	for _, count := range requests {
		total += count
	}
	shares := make(map[string]float64, len(requests))
	for name, count := range requests {
		shares[name] = float64(count) / float64(total)
	}
	p.counts[modality] = popularityCount{at: time.Now(), shares: shares}
	return shares
}
