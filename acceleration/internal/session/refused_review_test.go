//go:build integration

package session

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// RefusedReviewSuite runs a call with a turn the prompt injection screen refused through to
// the review that lands on its row when it stops, as staging does.
type RefusedReviewSuite struct {
	suite.Suite
	ctx   context.Context
	store *store.Store
}

func TestRefusedReviewSuite(t *testing.T) {
	suite.Run(t, new(RefusedReviewSuite))
}

func (s *RefusedReviewSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN is not set")
	}
	s.ctx = context.Background()
	db, err := store.Open(testenv.Database(dsn, "refused_review"))
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.store = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })
}

func (s *RefusedReviewSuite) TestACallWithARefusedTurnIsReviewedWhenItStops() {
	logs := &lockedBuffer{}
	// One model answers the call and reviews it, behind the screen that refuses injected.
	manager, err := NewManager(ManagerOptions{
		LLM:    screenedBy(&s.Suite, stubConfig(), &llmtest.Thinker{Answer: reviewAnswer}),
		Store:  s.store,
		Logger: slog.New(slog.NewTextHandler(logs, nil)),
		Edge: func(context.Context, Spec, streamapp.Bound, *slog.Logger) (agent.Edge, error) {
			return newQuietEdge(), nil
		},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { manager.Shutdown() })
	customer := "refused-" + uuid.NewString()
	created, err := manager.Create(s.ctx, Spec{Text: true, CustomerID: customer, LLMTarget: "en-low-latency"})
	s.Require().NoError(err)

	_, err = created.Respond(s.ctx, "Where is order 3825?", nil, options.LLM{})
	s.Require().NoError(err)
	s.Require().Eventually(func() bool { return len(created.conversation()) == 2 }, settleFor, 10*time.Millisecond,
		"the first turn was never answered")
	_, err = created.Respond(s.ctx, injected, nil, options.LLM{})
	s.Require().NoError(err)
	s.Require().Eventually(func() bool {
		return strings.Contains(logs.String(), "a reply was refused as a prompt injection")
	}, settleFor, 10*time.Millisecond, "the screen never refused the second turn")
	s.Require().NoError(created.Close())

	s.Eventually(func() bool {
		found, err := s.store.Call(s.ctx, customer, created.ID())
		return err == nil && found.ReviewScore != nil
	}, settleFor, 20*time.Millisecond, "the call was never reviewed")
	// The suite's usage rows carry no price or key, which logs errors of its own: only the
	// review's is this test's.
	s.NotContains(logs.String(), "could not review the call")
	row, err := s.store.Call(s.ctx, customer, created.ID())
	s.Require().NoError(err)
	s.Require().NotNil(row.ReviewScore)
	s.Equal(5, *row.ReviewScore)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}
