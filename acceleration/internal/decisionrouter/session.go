package decisionrouter

import (
	"context"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// Session is a selected classifier attached to one customer. It answers questions and
// records a stat row per request on the way past.
//
// There is no event stream to forward, unlike the three model modalities: a judgement is
// one request and one answer, so the row is written where the answer arrives.
type Session struct {
	provider decisionmodel.Provider
	// config is the routing identity of the provider. Stats and health are keyed by it,
	// so a provider registered under a different name still aggregates coherently.
	config   routing.ProviderConfig
	owner    routing.Owner
	recorder *routing.Recorder

	closeOnce sync.Once
}

func newSession(
	provider decisionmodel.Provider,
	config routing.ProviderConfig,
	owner routing.Owner,
	recorder *routing.Recorder,
) *Session {
	return &Session{provider: provider, config: config, owner: owner, recorder: recorder}
}

// Classify puts one request's questions to the selected classifier.
//
// The usage is reported and the cost left to the price list rather than worked out here: a
// classifier is billed by what it read, and only the list knows what this deployment pays for
// it. A request that failed is not charged, and its row is kept so the wait still shows up.
func (s *Session) Classify(
	ctx context.Context, request decisionmodel.Request,
) (decisionmodel.Result, error) {
	answered, err := routing.Call(ctx, routing.Meter{
		Recorder: s.recorder, Config: s.config, Owner: s.owner,
		Code: func(error) string { return "classify_failed" },
	}, func(ctx context.Context) (decisionmodel.Result, error) {
		return s.provider.Classify(ctx, request)
	})
	if err != nil {
		return decisionmodel.Result{}, err
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
