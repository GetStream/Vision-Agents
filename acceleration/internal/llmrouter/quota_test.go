//go:build integration

package llmrouter

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/mock"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv/redisready"
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
	client, err := redisready.NewClient(rueidis.ClientOption{
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
	return s.sessionAs("stub", messages, tokens, caller)
}

// sessionAs is session with the stub routed under the provider name given.
func (s *LLMQuotaSuite) sessionAs(name string, messages, tokens int64, caller routing.Caller) (*Session, *stubLLM) {
	limiter, err := quota.New(s.redis,
		quota.Limits{MessagesPerDay: messages, TokensPerDay: tokens}, nil, slog.Default())
	s.Require().NoError(err)

	provider := newStubLLM()
	recorder := routing.NewRecorder(routing.LLM, nil, nil, nil, slog.Default())
	config := routing.ProviderConfig{Provider: name, Model: "stub-model"}
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

// spentFor is how long a debit may take to be counted once the limiter has stopped waiting
// for it: the Redis client gives up on a write that long after sending it.
const spentFor = 15 * time.Second

// slowWrites is a Redis whose writes take a while to land, as a loaded one's can.
type slowWrites struct {
	rueidis.Client
	by time.Duration
}

func (r slowWrites) DoMulti(ctx context.Context, commands ...rueidis.Completed) []rueidis.RedisResult {
	if len(commands) > 0 && commands[0].Commands()[0] == "HINCRBY" {
		time.Sleep(r.by)
	}
	return r.Client.DoMulti(ctx, commands...)
}

// spent waits until the session refuses the turn turn asks for, which is once what it
// answered has been counted. The limiter stops waiting for a debit after two seconds, so
// when this process stalls the debit goes on after the answer has ended.
func (s *LLMQuotaSuite) spent(session *Session, turn context.Context) {
	s.Require().Eventually(func() bool {
		return errors.Is(session.allow(turn), quota.ErrExhausted)
	}, spentFor, 10*time.Millisecond, "what the session answered was never counted")
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
	s.spent(session, s.ctx)

	_, err := session.Create(s.ctx, llm.ResponseParams{ID: "c3", Input: prompt()})

	s.ErrorIs(err, quota.ErrExhausted)
}

func (s *LLMQuotaSuite) TestWhatAResponseCostIsCountedOnceItHasSettled() {
	// A limit counted after the fact still stops the turn after the expensive one.
	session, provider := s.session(0, 100, s.caller)

	stream, err := session.Create(s.ctx, llm.ResponseParams{ID: "c1", Input: prompt()})
	s.Require().NoError(err)
	s.answer(provider, stream, 0, 90, 20)
	s.spent(session, s.ctx)

	_, err = session.Create(s.ctx, llm.ResponseParams{ID: "c2", Input: prompt()})

	s.ErrorIs(err, quota.ErrExhausted)
}

func (s *LLMQuotaSuite) TestAResponseIsCountedBeforeItsStreamEnds() {
	// The next turn is admitted against the day as it stands when this one ends, so a debit
	// left to land after the stream lets a caller who is quick enough in over the limit.
	const landing = 300 * time.Millisecond
	session, provider := s.session(1, 0, s.caller)
	limiter, err := quota.New(slowWrites{s.redis, landing}, quota.Limits{MessagesPerDay: 1}, nil, slog.Default())
	s.Require().NoError(err)
	session.quota = limiter
	stream, err := session.Create(s.ctx, llm.ResponseParams{ID: "c1", Input: prompt()})
	s.Require().NoError(err)

	start := time.Now()
	s.answer(provider, stream, 0, 10, 5)

	s.GreaterOrEqual(time.Since(start), landing, "the stream ended before its debit had landed")
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
	s.spent(session, s.ctx)

	_, err = session.Create(s.ctx, llm.ResponseParams{ID: "c2", Input: prompt()})

	s.Require().ErrorIs(err, quota.ErrExhausted)
	s.Len(provider.asked, 1, "a refused turn should never reach the provider")
}

func (s *LLMQuotaSuite) TestASessionABackendOpenedForAUserIsNeverRefused() {
	// The backend names the user it acts for, and is trusted with that spend all the same.
	backend := s.caller
	backend.ServerSide = true
	session, provider := s.session(1, 1, backend)

	for turn := range 3 {
		stream, err := session.Create(s.ctx, llm.ResponseParams{
			ID: fmt.Sprintf("c%d", turn), Input: prompt(),
		})
		s.Require().NoError(err, "a backend's session should not be refused on turn %d", turn)
		s.answer(provider, stream, turn, 1_000, 1_000)
	}
	own, _ := s.session(1, 1, s.caller)
	_, err := own.Create(s.ctx, llm.ResponseParams{ID: "own", Input: prompt()})
	s.NoError(err, "what the backend spent was not charged to the user's own day")
}

func (s *LLMQuotaSuite) TestAScheduledTurnIsTheCustomersSpend() {
	// A turn a scheduled task fired is refused nothing and spends nothing of
	// the user's day: they agreed to the task once, not to each time it runs.
	session, provider := s.session(1, 0, s.caller)
	scheduled := routing.WithScheduledTurn(s.ctx)

	stream, err := session.Create(scheduled, llm.ResponseParams{ID: "s1", Input: prompt()})
	s.Require().NoError(err, "a scheduled turn is not refused by the user's day")
	s.answer(provider, stream, 0, 10, 5)

	stream, err = session.Create(s.ctx, llm.ResponseParams{ID: "u1", Input: prompt()})
	s.Require().NoError(err, "the scheduled turn spent none of the user's day")
	s.answer(provider, stream, 1, 10, 5)
	s.spent(session, s.ctx)

	_, err = session.Create(s.ctx, llm.ResponseParams{ID: "u2", Input: prompt()})
	s.Require().ErrorIs(err, quota.ErrExhausted, "the user's own turn spent the day")

	stream, err = session.Create(scheduled, llm.ResponseParams{ID: "s2", Input: prompt()})
	s.Require().NoError(err, "a scheduled turn is answered even once the day is spent")
	s.answer(provider, stream, 2, 10, 5)
}

func (s *LLMQuotaSuite) TestAMockAnswerIsNeitherCheckedNorCounted() {
	mocked, mocker := s.sessionAs(mock.ProviderName, 1, 0, s.caller)
	counted, counter := s.session(1, 0, s.caller)

	for turn := range 3 {
		stream, err := mocked.Create(s.ctx, llm.ResponseParams{
			ID: fmt.Sprintf("m%d", turn), Input: prompt(),
		})
		s.Require().NoError(err, "a mock answer should not be refused on turn %d", turn)
		s.answer(mocker, stream, turn, 1_000, 1_000)
	}

	stream, err := counted.Create(s.ctx, llm.ResponseParams{ID: "real", Input: prompt()})
	s.Require().NoError(err, "the mock answers were not charged to the user's day")
	s.answer(counter, stream, 0, 1_000, 1_000)
	s.spent(counted, s.ctx)

	_, err = counted.Create(s.ctx, llm.ResponseParams{ID: "over", Input: prompt()})
	s.Require().ErrorIs(err, quota.ErrExhausted, "the user's day is spent")

	stream, err = mocked.Create(s.ctx, llm.ResponseParams{ID: "m3", Input: prompt()})
	s.Require().NoError(err, "a mock answer is not refused once the day is spent")
	s.answer(mocker, stream, 3, 1_000, 1_000)
}

func (s *LLMQuotaSuite) TestAUsersOwnTurnInASessionABackendOpenedForThemIsCountedOnTheirDay() {
	backend := s.caller
	backend.ServerSide = true
	session, provider := s.session(1, 0, backend)
	sentByUser := routing.WithCaller(s.ctx, s.caller)

	stream, err := session.Create(sentByUser, llm.ResponseParams{ID: "user", Input: prompt()})
	s.Require().NoError(err)
	s.answer(provider, stream, 0, 10, 5)
	s.spent(session, sentByUser)

	_, err = session.Create(sentByUser, llm.ResponseParams{ID: "over", Input: prompt()})
	s.Require().ErrorIs(err, quota.ErrExhausted, "the user's next turn is refused")
	stream, err = session.Create(routing.WithCaller(s.ctx, backend), llm.ResponseParams{ID: "backend", Input: prompt()})
	s.Require().NoError(err, "the backend's own turn is not counted")
	s.answer(provider, stream, 1, 10, 5)
}

func (s *LLMQuotaSuite) TestAUserOpenedSessionIsCountedOnWhoeverOpenedIt() {
	// A turn in it is the user's whoever sends it, and from wherever: the backend passing on
	// what the user wrote, or the user from another address, spends the day of the address
	// the session was opened from, as it always has.
	opener := routing.Caller{IP: fmt.Sprintf("opener-%d", time.Now().UnixNano())}
	user := s.caller
	user.IP = opener.IP
	session, provider := s.session(2, 0, user)
	elsewhere := user
	elsewhere.IP = "elsewhere-" + opener.IP

	for turn, sender := range []routing.Caller{{UserID: user.UserID, ServerSide: true}, elsewhere} {
		stream, err := session.Create(routing.WithCaller(s.ctx, sender), llm.ResponseParams{
			ID: fmt.Sprintf("c%d", turn), Input: prompt(),
		})
		s.Require().NoError(err)
		s.answer(provider, stream, turn, 10, 5)
	}

	sameAddress, _ := s.session(2, 0, opener)
	s.spent(sameAddress, s.ctx)
	_, err := sameAddress.Create(s.ctx, llm.ResponseParams{ID: "address", Input: prompt()})
	s.ErrorIs(err, quota.ErrExhausted, "both turns were counted on the address the session was opened from")
}
