//go:build integration

package llmrouter

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// LLMQuotaSuite covers what a session does about a daily limit. It needs a Redis to count
// in but no provider credentials: the stub answers every turn, so what is under test is the
// session's own decision rather than a model's.
type LLMQuotaSuite struct {
	suite.Suite
	ctx    context.Context
	redis  rueidis.Client
	caller routing.Caller
}

func TestLLMQuotaSuite(t *testing.T) {
	suite.Run(t, new(LLMQuotaSuite))
}

func (s *LLMQuotaSuite) SetupSuite() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR must be set")
	}
	client, err := rueidis.NewClient(rueidis.ClientOption{
		InitAddress:  []string{address},
		DisableCache: true,
	})
	s.Require().NoError(err)
	s.redis = client
	s.ctx = context.Background()
}

func (s *LLMQuotaSuite) TearDownSuite() {
	if s.redis != nil {
		s.redis.Close()
	}
}

func (s *LLMQuotaSuite) SetupTest() {
	// A fresh user per test, so no test spends another's allowance.
	s.caller = routing.Caller{UserID: fmt.Sprintf("user-%d", time.Now().UnixNano())}
}

// session returns a session over the stub provider, limited to the given allowance and
// owned by the caller under test.
func (s *LLMQuotaSuite) session(messages, tokens int64, caller routing.Caller) (*Session, *stubLLM) {
	limiter, err := quota.New(s.redis,
		quota.Limits{MessagesPerDay: messages, TokensPerDay: tokens}, slog.Default())
	s.Require().NoError(err)

	provider := newStubLLM()
	recorder := routing.NewRecorder(routing.LLM, nil, nil, slog.Default())
	config := routing.ProviderConfig{Provider: "stub", Model: "stub-model"}
	session := newSession(provider, config,
		routing.Owner{CustomerID: "acme", Caller: caller}, recorder, limiter)

	s.T().Cleanup(func() {
		_ = session.Close()
		recorder.Close()
	})
	return session, provider
}

// answer drives the nth response to completion with the token usage given, which is what
// makes the session debit it.
func (s *LLMQuotaSuite) answer(provider *stubLLM, stream *llm.Stream, n int, input, output int64) {
	provider.script(n).OutputText("an answer")
	provider.script(n).Usage(llm.Usage{InputTokens: input, OutputTokens: output})
	provider.script(n).Done()
	drain(stream)
}

func (s *LLMQuotaSuite) TestASocketStopsBeingAnsweredOnceTheDayIsSpent() {
	// The hole an admission check alone leaves open: one socket, many turns.
	session, provider := s.session(2, 0, s.caller)

	for turn := range 2 {
		stream, err := session.Create(s.ctx, llm.ResponseParams{
			ID: fmt.Sprintf("c%d", turn), Input: prompt(),
		})
		s.Require().NoError(err, "turn %d was within the allowance", turn)
		s.answer(provider, stream, turn, 10, 5)
	}

	_, err := session.Create(s.ctx, llm.ResponseParams{ID: "c3", Input: prompt()})

	s.ErrorIs(err, quota.ErrExhausted)
}

func (s *LLMQuotaSuite) TestWhatAResponseCostIsCountedOnceItHasSettled() {
	// A limit counted after the fact still stops the turn after the expensive one.
	session, provider := s.session(0, 100, s.caller)

	stream, err := session.Create(s.ctx, llm.ResponseParams{ID: "c1", Input: prompt()})
	s.Require().NoError(err)
	s.answer(provider, stream, 0, 90, 20)

	_, err = session.Create(s.ctx, llm.ResponseParams{ID: "c2", Input: prompt()})

	s.ErrorIs(err, quota.ErrExhausted)
}

func (s *LLMQuotaSuite) TestABackendIsNotLimited() {
	// No caller to count against, which is what a customer's own backend has.
	session, provider := s.session(1, 1, routing.Caller{})

	for turn := range 3 {
		stream, err := session.Create(s.ctx, llm.ResponseParams{
			ID: fmt.Sprintf("c%d", turn), Input: prompt(),
		})
		s.Require().NoError(err, "a backend should not be refused on turn %d", turn)
		s.answer(provider, stream, turn, 1_000, 1_000)
	}
}

func (s *LLMQuotaSuite) TestARefusedTurnAsksTheProviderForNothing() {
	session, provider := s.session(1, 0, s.caller)

	stream, err := session.Create(s.ctx, llm.ResponseParams{ID: "c1", Input: prompt()})
	s.Require().NoError(err)
	s.answer(provider, stream, 0, 10, 5)

	_, err = session.Create(s.ctx, llm.ResponseParams{ID: "c2", Input: prompt()})

	s.Require().ErrorIs(err, quota.ErrExhausted)
	s.Len(provider.asked, 1, "a refused turn should never reach the provider")
}
