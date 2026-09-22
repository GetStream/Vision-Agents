package lcmrouter

import (
	"context"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/lcm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// Session is a selected classifier attached to one customer. It answers questions and
// records a stat row per request on the way past.
//
// There is no event stream to forward, unlike the three model modalities: a judgement is
// one request and one answer, so the row is written where the answer arrives.
type Session struct {
	provider lcm.Provider
	// config is the routing identity of the provider. Stats and health are keyed by it,
	// so a provider registered under a different name still aggregates coherently.
	config   routing.ProviderConfig
	owner    routing.Owner
	recorder *routing.Recorder

	closeOnce sync.Once
}

func newSession(
	provider lcm.Provider,
	config routing.ProviderConfig,
	owner routing.Owner,
	recorder *routing.Recorder,
) *Session {
	return &Session{provider: provider, config: config, owner: owner, recorder: recorder}
}

// Classify puts one request's questions to the selected classifier.
func (s *Session) Classify(
	ctx context.Context, request lcm.Request,
) (lcm.Result, error) {
	started := time.Now()
	answered, err := s.provider.Classify(ctx, request)
	s.record(started, answered.Usage, err)
	if err != nil {
		return lcm.Result{}, err
	}
	return answered, nil
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

// record files one judgement.
//
// The usage is reported and the cost left to the configured rates, rather than being worked
// out here: a classifier is billed by what it read, and only the config knows what this
// deployment pays for it. A request that failed reports no usage, since a judgement that
// did not arrive is not one to charge for, and the row is kept so the wait still shows up.
func (s *Session) record(started time.Time, used lcm.Usage, err error) {
	if s.recorder == nil {
		return
	}

	stat := routing.Stat{
		Owner:     s.owner,
		StartedAt: started.UTC(),
		LatencyMs: routing.MsSince(started),
		Usage: routing.Usage{
			InputTokens:  used.InputTokens,
			OutputTokens: used.OutputTokens,
		},
		Success: err == nil,
	}
	if err != nil {
		stat.ErrorCode = "classify_failed"
		stat.ErrorMessage = err.Error()
		stat.Usage = routing.Usage{}
	}
	s.recorder.Record(s.config, stat)
}
