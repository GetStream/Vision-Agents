package searchrouter

import (
	"context"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
)

// Session is a selected search provider attached to one customer. It answers questions and
// records a stat row per question on the way past.
//
// There is no event stream to forward, unlike the three model modalities: a search is one
// request and one answer, so the row is written where the answer arrives.
type Session struct {
	provider search.Provider
	// config is the routing identity of the provider. Stats and health are keyed by it,
	// so a provider registered under a different name still aggregates coherently.
	config   routing.ProviderConfig
	owner    routing.Owner
	recorder *routing.Recorder

	closeOnce sync.Once
}

func newSession(
	provider search.Provider,
	config routing.ProviderConfig,
	owner routing.Owner,
	recorder *routing.Recorder,
) *Session {
	return &Session{provider: provider, config: config, owner: owner, recorder: recorder}
}

// Search asks the selected provider one question.
func (s *Session) Search(ctx context.Context, query search.Query) (search.Result, error) {
	started := time.Now()
	found, err := s.provider.Search(ctx, query)
	s.record(started, err)
	if err != nil {
		return search.Result{}, err
	}
	return found, nil
}

// Provider is the provider serving this session.
func (s *Session) Provider() string { return s.config.Provider }

// Model is the model serving this session.
func (s *Session) Model() string { return s.config.Model }

// Close ends the session.
func (s *Session) Close() error {
	var err error
	s.closeOnce.Do(func() { err = s.provider.Close() })
	return err
}

// record files one search as a request row. A search API bills by the call rather than by
// what it read, so the cost is set outright from the configured rate instead of being
// multiplied out of a usage count nothing here collects.
func (s *Session) record(started time.Time, err error) {
	if s.recorder == nil {
		return
	}

	stat := routing.Stat{
		Owner:      s.owner,
		StartedAt:  started.UTC(),
		LatencyMs:  routing.MsSince(started),
		CostMicros: s.config.Price.RequestMicros(),
		Success:    err == nil,
	}
	if err != nil {
		stat.ErrorCode = "search_failed"
		// A search that failed is not billed, so the row records what was waited for
		// without charging for it.
		stat.CostMicros = 0
	}
	s.recorder.Record(s.config, stat)
}
