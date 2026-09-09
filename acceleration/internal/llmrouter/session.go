package llmrouter

import (
	"context"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// Session is a live model attached to one customer. It hands out the provider's streams
// untouched apart from recording a stat row per response on the way past.
type Session struct {
	provider Provider
	// config is the routing identity of the provider. Stats and health are keyed by it,
	// so a provider registered under a different name still aggregates coherently.
	config   routing.ProviderConfig
	owner    routing.Owner
	recorder *routing.Recorder
}

func newSession(
	provider Provider,
	config routing.ProviderConfig,
	owner routing.Owner,
	recorder *routing.Recorder,
) *Session {
	return &Session{provider: provider, config: config, owner: owner, recorder: recorder}
}

// Create asks the selected provider for a response. The stream it returns records what the
// response cost as it is drained, which is why a caller must drain it even after closing
// it: an abandoned response still generated tokens.
func (s *Session) Create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	startedAt := time.Now().UTC()

	stream, err := s.provider.Create(ctx, params)
	if err != nil {
		s.recorder.Record(s.config, routing.Stat{
			Owner:     s.owner,
			StartedAt: startedAt,
			Success:   false,
			ErrorCode: "create_failed",
		})
		return nil, err
	}
	return stream.Observe(func(event llm.Event) { s.observe(startedAt, event) }), nil
}

// Provider is the provider serving this session.
func (s *Session) Provider() string { return s.config.Provider }

// Model is the model serving this session.
func (s *Session) Model() string { return s.config.Model }

// Capabilities is what the model serving this session accepts.
func (s *Session) Capabilities() llm.Capabilities { return s.provider.Capabilities() }

// Price is what this session's provider charges, so a caller can report a cost without
// reaching for the router's config.
func (s *Session) Price() routing.Price { return s.config.Price }

// LLM exposes the underlying provider so callers can reach provider-specific features.
func (s *Session) LLM() llm.LLM { return s.provider }

// Close ends the session, abandoning anything still in flight.
func (s *Session) Close() error { return s.provider.Close() }

// observe records statistics as a response settles.
//
// One response is one unit of billable work, the way one synthesis is for text-to-speech,
// and it is recorded once: a failure is carried on the response that failed rather than
// written as a row of its own, so one turn stays one row.
func (s *Session) observe(startedAt time.Time, event llm.Event) {
	completed, settled := event.(llm.ResponseCompleted)
	if !settled {
		return
	}

	response := completed.Response
	s.recorder.Record(s.config, routing.Stat{
		Owner:     s.owner,
		StartedAt: startedAt,
		Usage: routing.Usage{
			InputTokens:       response.Usage.InputTokens,
			CachedInputTokens: response.Usage.InputTokensDetails.CachedTokens,
			OutputTokens:      response.Usage.OutputTokens,
		},
		// Time to first token is what the caller actually waited for; the rest of the
		// answer arrives while they are already reading or hearing it.
		LatencyMs: response.TimeToFirstTokenMs,
		Success:   response.Status != llm.StatusFailed,
		ErrorCode: errorCode(response),
	})
}

// errorCode says why a response failed, or nothing for one that did not.
func errorCode(response llm.Response) string {
	if response.Status != llm.StatusFailed {
		return ""
	}
	return "provider_error"
}
