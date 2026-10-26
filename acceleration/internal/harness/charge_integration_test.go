//go:build integration

package harness

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
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// messagesPerDay is the user's allowance, far more than any test spends.
const messagesPerDay = 100

// userSet is a quota source where the customer set every user's own quota.
type userSet struct{}

func (userSet) Quota(_ context.Context, _, userID string, _ quota.Limits) quota.Quota {
	return quota.Quota{Messages: quota.Limit{PerDay: messagesPerDay, By: quota.Setter{Level: quota.LevelUser, ID: userID}}}
}

// HarnessChargeSuite covers whose day delegated work is counted on. It counts in a real
// Redis, and what it reads back is the user's day as the limiter keeps it.
type HarnessChargeSuite struct {
	suite.Suite
	ctx   context.Context
	redis rueidis.Client

	user    routing.Caller
	limiter *quota.Limiter
	fast    *stubLLM
	slow    *stubLLM
	// asked holds the subagent's answer back until the turn that asked for it has ended.
	asked   chan struct{}
	harness *Harness
	events  *collector
}

func TestHarnessChargeSuite(t *testing.T) {
	suite.Run(t, new(HarnessChargeSuite))
}

func (s *HarnessChargeSuite) SetupSuite() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR must be set")
	}
	client, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.redis = client
	s.ctx = context.Background()
}

func (s *HarnessChargeSuite) TearDownSuite() {
	if s.redis != nil {
		s.redis.Close()
	}
}

func (s *HarnessChargeSuite) SetupTest() {
	// A fresh user per test, so no test reads another's day.
	s.user = routing.Caller{UserID: fmt.Sprintf("user-%d", time.Now().UnixNano())}
	limiter, err := quota.New(s.redis, quota.Limits{MessagesPerDay: messagesPerDay}, nil, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	s.limiter = limiter
}

// build returns a harness whose conversation and subagent are both opened for owner, as
// the agent opens them for the session's owner.
func (s *HarnessChargeSuite) build(owner routing.Caller) {
	s.fast, s.slow = newStubLLM(), newStubLLM()
	s.fast.automatic = "Let me check."
	s.slow.automatic = "It is 12.63."
	s.asked = make(chan struct{})
	s.slow.holdCreate = s.asked
	harness, err := New(Options{
		Model:    s.session(s.fast, owner),
		Subagent: s.session(s.slow, owner),
		Skills:   testSkills(),
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.harness = harness
	s.events = collect(harness)
	s.T().Cleanup(func() { <-s.events.done })
	s.T().Cleanup(func() { _ = harness.Close() })
}

// session starts a session over a stub provider, counted on the limiter under test.
func (s *HarnessChargeSuite) session(provider *stubLLM, owner routing.Caller) *llmrouter.Session {
	registry := llmrouter.NewRegistry()
	registry.Register("stub", func(routing.Spec) (llmrouter.Provider, error) { return provider, nil })
	router, err := llmrouter.New(llmrouter.Options{
		Config: stubConfig(), Registry: registry, Quota: s.limiter, Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	session, err := router.Start(s.ctx, llmrouter.Request{CustomerID: "acme", Caller: owner, Target: "en-low-latency"})
	s.Require().NoError(err)
	return session
}

// reply answers a turn asked on ctx and passes what the model wrote through the filter,
// as the agent does. The turn's ctx ends with its reply, as the agent's does.
func (s *HarnessChargeSuite) reply(ctx context.Context, turnID, written string) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stream, err := s.harness.Respond(ctx, Turn{
		ID: turnID, Instructions: "be brief",
		History: []llm.Message{{Role: llm.User, Content: "what is 15% of 84.20"}},
	})
	s.Require().NoError(err)
	for stream.Next() {
	}
	s.harness.Filter(turnID, written)
}

// settled lets the subagent answer, waits for one task to settle, and returns how it did.
func (s *HarnessChargeSuite) settled() Result {
	close(s.asked)
	s.Require().Eventually(func() bool { return len(settledIn(s.events.seen())) == 1 },
		settleFor, 10*time.Millisecond, "the delegated work never settled")
	return settledIn(s.events.seen())[0].Result
}

// day is how many answers are counted on the user's day: what the limiter, asked about
// one more, says is gone of it.
func (s *HarnessChargeSuite) day() int64 {
	status, err := s.limiter.Allow(s.ctx, "acme", s.user)
	s.Require().NoError(err)
	return messagesPerDay - status.Remaining
}

func (s *HarnessChargeSuite) TestWorkAScheduledTurnDelegatesIsNotCountedOnTheUsersDay() {
	s.build(s.user)

	s.reply(routing.WithScheduledTurn(s.ctx), "turn-1", `<ask skill="think">15% of 84.20</ask>`)

	result := s.settled()
	s.Equal(Done, result.State, "the work outlives the turn that asked for it")
	s.Equal("It is 12.63.", result.Text)
	s.Zero(s.day(), "neither the scheduled turn nor the work it delegated is the user's spend")
}

func (s *HarnessChargeSuite) TestWorkTheUsersOwnTurnDelegatesIsCountedOnTheirDay() {
	s.build(s.user)

	s.reply(s.ctx, "turn-1", `<ask skill="think">15% of 84.20</ask>`)

	s.Equal(Done, s.settled().State)
	s.EqualValues(2, s.day(), "the reply and the work it delegated are both the user's")
}

func (s *HarnessChargeSuite) TestWorkAUserDelegatesInASessionABackendOpenedStaysOnTheBackend() {
	// The backend opened both sessions; the user typed the turn. The reply is the user's,
	// and the work it delegated is counted on whoever opened the subagent, the backend.
	s.build(routing.Caller{UserID: s.user.UserID, ServerSide: true})

	s.reply(routing.WithCaller(s.ctx, s.user), "turn-1", `<ask skill="think">15% of 84.20</ask>`)

	s.Equal(Done, s.settled().State)
	s.EqualValues(1, s.day(), "only the reply is the user's")
}

func (s *HarnessChargeSuite) TestWorkAHeardTurnDelegatesInASessionABackendOpenedIsCountedOnTheOwner() {
	// A heard turn names no sender, so it and its work are counted on whoever opened the
	// session. The customer set the user's own quota, which holds the owner too.
	limiter, err := quota.New(s.redis, quota.Limits{}, userSet{}, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	s.limiter = limiter
	owner := routing.Caller{UserID: s.user.UserID, ServerSide: true}
	s.build(owner)

	s.reply(s.ctx, "turn-1", `<ask skill="think">15% of 84.20</ask>`)

	s.Equal(Done, s.settled().State)
	status, err := s.limiter.Allow(s.ctx, "acme", owner)
	s.Require().NoError(err)
	s.EqualValues(2, messagesPerDay-status.Remaining, "the reply and the work it delegated are both the owner's")
}

func (s *HarnessChargeSuite) TestWorkTheUsersOwnTurnDelegatesAfterAScheduledTurnIsCountedOnTheirDay() {
	// What a turn is counted as is the turn's own: the mark of the scheduled turn before it
	// is not kept.
	s.build(s.user)

	s.reply(routing.WithScheduledTurn(s.ctx), "turn-1", "")
	s.reply(s.ctx, "turn-2", `<ask skill="think">15% of 84.20</ask>`)

	s.Equal(Done, s.settled().State)
	s.EqualValues(2, s.day(), "the user's own reply and the work it delegated are both the user's")
}

func (s *HarnessChargeSuite) TestWorkAScheduledTurnHandsOverIsNotCountedOnTheUsersDay() {
	s.build(s.user)
	turn, cancel := context.WithCancel(routing.WithScheduledTurn(s.ctx))

	_, err := s.harness.Delegate(turn, "think", "15% of 84.20", "turn-1", nil, nil)
	s.Require().NoError(err)
	s.Require().Eventually(func() bool { return len(s.slow.requests()) == 1 },
		settleFor, time.Millisecond, "the subagent was never asked")
	cancel()

	s.Never(func() bool { return len(settledIn(s.events.seen())) > 0 }, 100*time.Millisecond, 5*time.Millisecond,
		"the work ended with the turn that handed it over")
	s.Equal(Done, s.settled().State, "the work outlives the turn that handed it over")
	s.Zero(s.day(), "the work a scheduled turn handed over is not the user's spend")
}

func (s *HarnessChargeSuite) TestAConversationAScheduledTurnCompactsIsNotCountedOnTheUsersDay() {
	s.build(s.user)
	s.reply(routing.WithScheduledTurn(s.ctx), "turn-1", "")

	started, err := s.harness.MaybeCompact(longHistory(), compactionMinTokens, 0)
	s.Require().NoError(err)
	s.Require().True(started)
	close(s.asked)

	s.Require().Eventually(func() bool { return len(compactedIn(s.events.seen())) == 1 },
		settleFor, 10*time.Millisecond, "the conversation was never compacted")
	s.Zero(s.day(), "the summary a scheduled turn set off is not the user's spend")
}

func (s *HarnessChargeSuite) TestAConversationTheUsersOwnTurnCompactsIsCountedOnTheirDay() {
	s.build(s.user)
	s.reply(s.ctx, "turn-1", "")

	started, err := s.harness.MaybeCompact(longHistory(), compactionMinTokens, 0)
	s.Require().NoError(err)
	s.Require().True(started)
	close(s.asked)

	s.Require().Eventually(func() bool { return len(compactedIn(s.events.seen())) == 1 },
		settleFor, 10*time.Millisecond, "the conversation was never compacted")
	s.EqualValues(2, s.day(), "the reply and the summary it set off are both the user's")
}
