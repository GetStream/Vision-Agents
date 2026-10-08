//go:build integration

package mem0

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// extractionGrace is how long mem0 is given to turn a handed-over conversation into
// memories, since it does that on its own side rather than while the call waits.
const extractionGrace = 30 * time.Second

type Mem0IntegrationSuite struct {
	suite.Suite
	store *Store
	scope memory.Scope
}

func TestMem0IntegrationSuite(t *testing.T) {
	suite.Run(t, new(Mem0IntegrationSuite))
}

func (s *Mem0IntegrationSuite) SetupSuite() {
	if os.Getenv(apiKeyEnvVar) == "" {
		s.T().Skip(apiKeyEnvVar + " not set")
	}

	store, err := New(Options{})
	s.Require().NoError(err)
	s.store = store

	// A fresh user per run, so one run's memories are not another's.
	s.scope = memory.Scope{
		AppID:   "acceleration-test",
		UserID:  fmt.Sprintf("test-%d", time.Now().UnixNano()),
		AgentID: "acceleration-test-agent",
		RunID:   fmt.Sprintf("run-%d", time.Now().UnixNano()),
		Extra:   map[string]string{"company_id": "12312"},
	}
}

func (s *Mem0IntegrationSuite) TestWhatIsToldToMem0CanBeRecalledAfterwards() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	err := s.store.Remember(ctx, s.scope, []llm.Message{
		{Role: llm.User, Content: "I am allergic to peanuts and I live in Amsterdam."},
		{Role: llm.Assistant, Content: "Noted, no peanuts."},
	})
	s.Require().NoError(err)

	// Extraction is asynchronous, so this polls rather than asserting straight away.
	deadline := time.Now().Add(extractionGrace)
	var recalled []memory.Memory
	for time.Now().Before(deadline) {
		recalled, err = s.store.Recall(ctx, memory.Query{
			Scope: s.scope,
			Text:  "what should I not feed them",
			Limit: 5,
		})
		s.Require().NoError(err)
		if len(recalled) > 0 {
			break
		}
		time.Sleep(2 * time.Second)
	}

	s.Require().NotEmpty(recalled, "mem0 never produced a memory from the conversation")
	for _, remembered := range recalled {
		s.NotEmpty(remembered.Text)
		s.NotEmpty(remembered.ID, "a memory has to be identifiable to be corrected later")
	}

	// A later session is a different run and still knows what this one learned.
	later := s.scope
	later.RunID += "-later"
	recalled, err = s.store.Recall(ctx, memory.Query{Scope: later, Text: "what should I not feed them"})
	s.Require().NoError(err)
	s.NotEmpty(recalled, "memories are carried from one session into the next")

	otherCompany := s.scope
	otherCompany.Extra = map[string]string{"company_id": "99999"}
	recalled, err = s.store.Recall(ctx, memory.Query{Scope: otherCompany, Text: "what should I not feed them"})
	s.Require().NoError(err)
	s.Empty(recalled, "a label narrows recall to what was written under it")
}

func (s *Mem0IntegrationSuite) TestForgettingASessionKeepsTheOthersAndTruncatingKeepsNone() {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	user := s.scope
	user.UserID += "-forgets"
	first, second := user, user
	first.RunID += "-first"
	second.RunID += "-second"
	s.Require().NoError(s.store.Remember(ctx, first, []llm.Message{
		{Role: llm.User, Content: "I am allergic to peanuts."}, {Role: llm.Assistant, Content: "Noted."},
	}))
	s.Require().NoError(s.store.Remember(ctx, second, []llm.Message{
		{Role: llm.User, Content: "My dog is called Rex."}, {Role: llm.Assistant, Content: "Noted."},
	}))
	recalled := s.recallsAtLeast(ctx, user, 2)

	s.Require().NoError(s.store.ForgetRun(ctx, user.AppID, first.RunID))
	recalled, err := s.store.Recall(ctx, memory.Query{Scope: user, Text: "what do you know about me", Limit: 10})
	s.Require().NoError(err)
	s.Require().NotEmpty(recalled, "the other session's memories are kept")
	for _, remembered := range recalled {
		s.NotContains(remembered.Text, "peanut", "the forgotten session's memories are gone")
	}

	s.Require().NoError(s.store.Truncate(ctx, user.AppID, user.UserID))
	recalled, err = s.store.Recall(ctx, memory.Query{Scope: user, Text: "what do you know about me", Limit: 10})
	s.Require().NoError(err)
	s.Empty(recalled, "nothing is left about a truncated user")
}

// recallsAtLeast waits for mem0 to have extracted at least n memories about the scope.
func (s *Mem0IntegrationSuite) recallsAtLeast(ctx context.Context, scope memory.Scope, n int) []memory.Memory {
	deadline := time.Now().Add(extractionGrace)
	var recalled []memory.Memory
	for time.Now().Before(deadline) {
		var err error
		recalled, err = s.store.Recall(ctx, memory.Query{Scope: scope, Text: "what do you know about me", Limit: 10})
		s.Require().NoError(err)
		if len(recalled) >= n {
			return recalled
		}
		time.Sleep(2 * time.Second)
	}
	s.Require().GreaterOrEqual(len(recalled), n, "mem0 never produced the memories")
	return recalled
}

func (s *Mem0IntegrationSuite) TestAnotherUsersMemoriesAreNotRecalled() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	stranger := s.scope
	stranger.UserID += "-stranger"

	recalled, err := s.store.Recall(ctx, memory.Query{Scope: stranger, Text: "peanuts"})
	s.Require().NoError(err)

	s.Empty(recalled, "memories are personal")
}

func (s *Mem0IntegrationSuite) TestAScopeWithoutAUserIsRefusedBeforeTheNetwork() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := s.store.Recall(ctx, memory.Query{Scope: memory.Scope{AppID: "acceleration-test"}})

	s.ErrorContains(err, "user id is required")
}
