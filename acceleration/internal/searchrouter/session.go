package searchrouter

import (
	"context"
	"errors"
	"fmt"
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
	mu       sync.Mutex
	provider search.Provider
	// config is the routing identity of the provider. Stats and health are keyed by it,
	// so a provider registered under a different name still aggregates coherently.
	config   routing.ProviderConfig
	owner    routing.Owner
	recorder *routing.Recorder
	closed   bool
	// fallback answers a search the serving provider failed, from the next candidate that
	// will have it, and hands back that candidate's session so this one can move there:
	// the provider that just failed is not the one to ask next time. Nil, which is every
	// session that was not given a priority list, reports the failure instead.
	fallback func(ctx context.Context, query search.Query, failed routing.ProviderConfig) (*Session, search.Result, error)
}

func newSession(
	provider search.Provider,
	config routing.ProviderConfig,
	owner routing.Owner,
	recorder *routing.Recorder,
) *Session {
	return &Session{provider: provider, config: config, owner: owner, recorder: recorder}
}

// Search asks the serving provider one question, and the next candidate if it fails.
func (s *Session) Search(ctx context.Context, query search.Query) (search.Result, error) {
	s.mu.Lock()
	provider, config := s.provider, s.config
	s.mu.Unlock()

	found, err := s.ask(ctx, provider, config, query)
	if err == nil || s.fallback == nil || ctx.Err() != nil {
		return found, err
	}
	next, found, fallbackErr := s.fallback(ctx, query, config)
	if fallbackErr != nil {
		return search.Result{}, errors.Join(fmt.Errorf("%s: %w", config.Name(), err), fallbackErr)
	}
	s.moveTo(config, next)
	return found, nil
}

// Provider is the provider serving this session.
func (s *Session) Provider() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.config.Provider
}

// Model is the model serving this session.
func (s *Session) Model() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.config.Model
}

// Close ends the session.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	return s.provider.Close()
}

// ask puts one question to one provider and records what it cost.
func (s *Session) ask(ctx context.Context, provider search.Provider, config routing.ProviderConfig, query search.Query) (search.Result, error) {
	started := time.Now()
	found, err := provider.Search(ctx, query)
	s.record(config, started, err)
	if err != nil {
		return search.Result{}, err
	}
	return found, nil
}

// moveTo makes next's provider the one this session asks. Another search may have moved
// it on already, or it may have closed, and then next was wanted for one answer only.
func (s *Session) moveTo(failed routing.ProviderConfig, next *Session) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.config.Name() != failed.Name() {
		_ = next.provider.Close()
		return
	}
	_ = s.provider.Close()
	s.provider, s.config = next.provider, next.config
}

// record files one search as a request row. A search API bills by the call rather than by
// what it read, so the cost is set outright from the configured rate instead of being
// multiplied out of a usage count nothing here collects.
func (s *Session) record(config routing.ProviderConfig, started time.Time, err error) {
	if s.recorder == nil {
		return
	}

	stat := routing.Stat{
		Owner:      s.owner,
		StartedAt:  started.UTC(),
		LatencyMs:  routing.MsSince(started),
		CostMicros: config.Price.RequestMicros(),
		Success:    err == nil,
	}
	if err != nil {
		stat.ErrorCode = "search_failed"
		stat.ErrorMessage = err.Error()
		// A search that failed is not billed, so the row records what was waited for
		// without charging for it.
		stat.CostMicros = 0
	}
	s.recorder.Record(config, stat)
}
