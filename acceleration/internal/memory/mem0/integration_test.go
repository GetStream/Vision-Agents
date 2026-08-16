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
		AppID:  "acceleration-test",
		UserID: fmt.Sprintf("test-%d", time.Now().UnixNano()),
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
}

func (s *Mem0IntegrationSuite) TestAnotherUsersMemoriesAreNotRecalled() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	stranger := memory.Scope{AppID: s.scope.AppID, UserID: s.scope.UserID + "-stranger"}

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
