package agent

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory/mem0"
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
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
	s.writer = newMemoryWriter(metered.NewMemory(store, nil).For(routing.Owner{}), scope, false, logger)
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

// TestCloseGivesUpOnAMem0ThatNeverAnswers is the teardown of a call whose memory store is
// stuck: every queued exchange used to get its own memoryTimeout, so closing waited for
// all of them. Now it waits for one deadline and drops what is left, counted once.
func (s *MemoryWriterSuite) TestCloseGivesUpOnAMem0ThatNeverAnswers() {
	const queued = 20
	var attempts atomic.Int64
	release := make(chan struct{})
	stuck := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		attempts.Add(1)
		<-release
	}))
	s.T().Cleanup(stuck.Close)
	s.T().Cleanup(func() { close(release) })

	var logs bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&logs, nil))
	store, err := mem0.New(mem0.Options{APIKey: "test-key", BaseURL: stuck.URL, Logger: logger})
	s.Require().NoError(err)
	writer := newMemoryWriter(metered.NewMemory(store, nil).For(routing.Owner{}), memory.Scope{AppID: "router", UserID: "acme", AgentID: "support", RunID: "session-1"}, false, logger)
	writer.drainTimeout = 200 * time.Millisecond
	for i := 0; i < queued; i++ {
		writer.Remember([]llm.Message{{Role: llm.User, Content: "I moved to Austin"}})
	}

	closed := make(chan struct{})
	started := time.Now()
	go func() {
		writer.Close()
		close(closed)
	}()
	select {
	case <-closed:
	case <-time.After(3 * time.Second):
		s.FailNow("Close was still waiting on a mem0 that never answers")
	}

	s.GreaterOrEqual(time.Since(started), 200*time.Millisecond, "closing does not drop exchanges before the deadline")
	s.EqualValues(queued, writer.dropped.Load(), "the exchange in flight and every queued one are counted")
	s.EqualValues(1, attempts.Load(), "no exchange is tried once the deadline has passed")
	s.Equal(1, strings.Count(logs.String(), "dropped exchanges"), "the drops are logged once")
	s.Contains(logs.String(), "count=20")
}

func (s *MemoryWriterSuite) TestTheDrainDeadlineIsTwoWrites() {
	s.Equal(2*memoryTimeout, s.writer.drainTimeout)
}
