package agent

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory/mem0"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// MemoryWriterSuite runs the writer against the real mem0 store and a mem0 that is out of
// quota, so what reaches the log is what a deployment would see.
type MemoryWriterSuite struct {
	suite.Suite
	logs   *bytes.Buffer
	writer *memoryWriter
}

func TestMemoryWriterSuite(t *testing.T) { suite.Run(t, new(MemoryWriterSuite)) }

func (s *MemoryWriterSuite) SetupTest() {
	quota := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"Usage quota exceeded for this billing period."}`))
	}))
	s.T().Cleanup(quota.Close)

	s.logs = &bytes.Buffer{}
	logger := slog.New(slog.NewTextHandler(s.logs, &slog.HandlerOptions{Level: slog.LevelDebug}))
	store, err := mem0.New(mem0.Options{APIKey: "test-key", BaseURL: quota.URL, Logger: logger})
	s.Require().NoError(err)
	scope := memory.Scope{AppID: "router", UserID: "acme", AgentID: "support", RunID: "session-1"}
	s.writer = newMemoryWriter(store, scope, false, routing.Owner{}, nil, logger)
	s.T().Cleanup(s.writer.Close)
}

// failures is every ERROR line logged.
func (s *MemoryWriterSuite) failures() []string {
	var logged []string
	for _, line := range strings.Split(s.logs.String(), "\n") {
		if strings.Contains(line, "level=ERROR") {
			logged = append(logged, line)
		}
	}
	return logged
}

func (s *MemoryWriterSuite) TestARecallMem0RefusesIsOneErrorNamingTheQuota() {
	s.Nil(s.writer.Recall(context.Background(), 5))

	logged := s.failures()
	s.Require().Len(logged, 1)
	s.Contains(logged[0], "429 Too Many Requests (mem0 quota or rate limit)")
	s.Contains(logged[0], "Usage quota exceeded")
}

func (s *MemoryWriterSuite) TestAnExchangeMem0RefusesIsOneErrorNamingTheQuota() {
	s.writer.Remember([]llm.Message{{Role: llm.User, Content: "I moved to Austin"}})
	s.writer.Close()

	logged := s.failures()
	s.Require().Len(logged, 1)
	s.Contains(logged[0], "429 Too Many Requests (mem0 quota or rate limit)")
}
