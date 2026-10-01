package conversation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/suite"
)

// LiveSuite covers what watchers of a persistent reply see while it works: its live
// (ephemeral) updates, the reasoning windows they carry, and their pace against Stream's
// throttle.
type LiveSuite struct {
	suite.Suite
	db      *chatStore
	service *Service
	c       *Conversation
	id      string
}

func TestLiveSuite(t *testing.T) { suite.Run(t, new(LiveSuite)) }

// SetupTest begins a reply and waits for Stream to have it, so live updates can go out.
func (s *LiveSuite) SetupTest() {
	db, client := newChat(s.T())
	service, err := newService(s.T().TempDir(), client)
	s.Require().NoError(err)
	c, _, _, err := service.Open(context.Background(), "customer", "support-agent", "")
	s.Require().NoError(err)
	s.Require().NoError(c.Begin("question"))
	s.db, s.service, s.c, s.id = db, service, c, current(c).ID
	s.Require().Eventually(func() bool { c.mu.Lock(); defer c.mu.Unlock(); return c.created[s.id] }, 3*time.Second, 20*time.Millisecond)
}

func (s *LiveSuite) TearDownTest() {
	s.c.Release()
	s.service.Close()
}

// windows are the reasoning windows Stream accepted, in order.
func (s *LiveSuite) windows() []reasoningWindow {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	var all []reasoningWindow
	for _, patch := range s.db.patches {
		if raw, ok := patch["reasoning"]; ok {
			var w reasoningWindow
			b, _ := json.Marshal(raw)
			s.Require().NoError(json.Unmarshal(b, &w))
			all = append(all, w)
		}
	}
	return all
}

// TestProgressIsLiveUntilTheReplySettles covers what a watcher sees while a reply works:
// tool steps and the model's thinking arrive as ephemeral updates, the only stored write
// is the finished reply with its steps, and of the thinking only a round's opening is
// ever stored.
func (s *LiveSuite) TestProgressIsLiveUntilTheReplySettles() {
	s.c.ShowTools([]string{"athena_*"})
	thinking := "Weighing the two options. " + strings.Repeat("Considering more. ", 40) + "PRIVATE TAIL"
	s.c.Observe(agent.ReasoningDelta{Text: thinking})
	s.c.Observe(agent.ToolStarted{ID: "one", Tool: "athena_start_task", StartedAt: time.Now().UTC()})
	s.c.Progress("one", "searching")
	s.c.Observe(agent.ToolRan{ID: "one", Result: `{}`})
	s.Require().Eventually(func() bool {
		s.db.mu.Lock()
		defer s.db.mu.Unlock()
		raw, _ := json.Marshal(s.db.patches)
		return strings.Contains(string(raw), "PRIVATE TAIL") && strings.Contains(string(raw), `"status":"completed"`)
	}, 3*time.Second, 20*time.Millisecond)
	s.db.mu.Lock()
	s.Zero(s.db.updates, "tool progress was stored before the reply settled")
	s.db.mu.Unlock()

	// The local ledger still holds the progress a restart recovers from.
	raw, err := os.ReadFile(filepath.Join(s.c.dir(), "state.json"))
	s.Require().NoError(err)
	s.Contains(string(raw), "athena_start_task")
	s.Contains(string(raw), `"summary":"Weighing the two options."`)
	s.NotContains(string(raw), "PRIVATE TAIL")

	s.c.Observe(agent.ResponseDelta{Text: "The answer."})
	s.c.Observe(agent.Responded{})
	saved(s.T(), s.c)
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	s.Equal(1, s.db.updates)
	stored, _ := json.Marshal(s.db.messages[s.id])
	s.Contains(string(stored), "athena_start_task")
	s.Contains(string(stored), `"type":"ai_reasoning"`)
	s.Contains(string(stored), `"summary":"Weighing the two options."`)
	s.NotContains(string(stored), "PRIVATE TAIL")
}

// TestThinkingStreamsInWindows covers the cost of showing thinking live: it is sent in
// windows a watcher appends, thinking alone goes out at the gentler pace, the reply is
// not republished for it, and the last thoughts still arrive once the reply settles.
func (s *LiveSuite) TestThinkingStreamsInWindows() {
	sequence := current(s.c).Sequence

	var thinking strings.Builder
	begun := time.Now()
	for i := 0; time.Since(begun) < time.Second; i++ {
		piece := fmt.Sprintf("thought %d. ", i)
		thinking.WriteString(piece)
		s.c.Observe(agent.ReasoningDelta{Text: piece})
		time.Sleep(5 * time.Millisecond)
	}
	s.Equal(sequence+1, current(s.c).Sequence, "only opening the reasoning step changes the reply")
	s.c.Observe(agent.ReasoningDelta{Text: "Last thought."})
	thinking.WriteString("Last thought.")
	s.c.Observe(agent.ResponseDelta{Text: "The answer."})
	s.c.Observe(agent.Responded{})
	saved(s.T(), s.c)

	s.Require().Eventually(func() bool {
		all := s.windows()
		return len(all) > 0 && all[len(all)-1].Length == utf8.RuneCountInString(thinking.String())
	}, 3*time.Second, 20*time.Millisecond, "the last thoughts never arrived")

	var w watcher
	sent := 0
	for _, window := range s.windows() {
		w.apply(window)
		sent += len(window.Text)
	}
	s.Equal(thinking.String(), w.text)
	// About five updates a second while only thinking, plus the settled reply's last one.
	s.LessOrEqual(len(s.windows()), 9, "thinking alone was sent at the answer's pace")
	s.Less(sent, 2*thinking.Len(), "windows repeated thinking already sent")

	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	// The settled reply keeps the round's opening as its step, and none of the rest.
	stored, _ := json.Marshal(s.db.messages[s.id])
	s.Contains(string(stored), `"summary":"thought 0."`)
	s.NotContains(string(stored), "Last thought.")
	s.NotContains(string(stored), "thought 99.")
}

// TestTheAnswerStaysWithinStreamsThrottle covers the answer's live pace: Stream throttles
// message.updated to 10 a second per channel, and the answer goes out at most every
// answerEvery so the channel's other updates fit too.
func (s *LiveSuite) TestTheAnswerStaysWithinStreamsThrottle() {
	s.db.mu.Lock()
	before := len(s.db.patches)
	s.db.mu.Unlock()
	begun := time.Now()
	for time.Since(begun) < time.Second {
		s.c.Observe(agent.ResponseDelta{Text: "word "})
		time.Sleep(5 * time.Millisecond)
	}
	s.db.mu.Lock()
	sent := len(s.db.patches) - before
	s.db.mu.Unlock()
	s.GreaterOrEqual(sent, 3, "the answer stopped streaming")
	s.LessOrEqual(sent, int(time.Second/answerEvery)+1, "the answer outpaced answerEvery")
	s.c.Observe(agent.Responded{})
	saved(s.T(), s.c)
}

// TestLiveUpdatesWaitOutARateLimit covers Stream refusing live updates: the runtime waits
// for its Retry-After instead of trying again every tick, then resumes with nothing lost,
// and the stored reply is unaffected.
func (s *LiveSuite) TestLiveUpdatesWaitOutARateLimit() {
	s.db.mu.Lock()
	s.db.rateLimited = true
	s.db.liveAttempts = 0
	s.db.mu.Unlock()
	var thinking strings.Builder
	begun := time.Now()
	for time.Since(begun) < 1500*time.Millisecond {
		piece := "thinking it over. "
		thinking.WriteString(piece)
		s.c.Observe(agent.ReasoningDelta{Text: piece})
		time.Sleep(10 * time.Millisecond)
	}
	s.db.mu.Lock()
	attempts := s.db.liveAttempts
	s.db.rateLimited = false
	s.db.mu.Unlock()
	// One refused, then a second after Retry-After, rather than one every tick.
	s.LessOrEqual(attempts, 3, "live updates kept hammering a rate-limited Stream")
	s.GreaterOrEqual(attempts, 1)

	s.Require().Eventually(func() bool {
		all := s.windows()
		return len(all) > 0 && all[len(all)-1].Length == utf8.RuneCountInString(thinking.String())
	}, 5*time.Second, 20*time.Millisecond, "the thinking held back never arrived")
	var w watcher
	for _, window := range s.windows() {
		w.apply(window)
	}
	s.Equal(thinking.String(), w.text, "thinking was lost while Stream refused updates")

	s.c.Observe(agent.ResponseDelta{Text: "The answer."})
	s.c.Observe(agent.Responded{})
	saved(s.T(), s.c)
}

func (s *LiveSuite) TestLiveBackoffFollowsStream() {
	now := time.Unix(1_790_000_000, 0)
	limited := func(retryAfter time.Duration, reset int64) error {
		return &getstream.StreamError{StatusCode: http.StatusTooManyRequests, RetryAfter: retryAfter,
			RateLimit: &getstream.RateLimitInfo{Reset: reset}}
	}
	s.Equal(3*time.Second, liveBackoff(limited(3*time.Second, 0), 1, now), "Retry-After first")
	s.Equal(20*time.Second, liveBackoff(limited(0, now.Add(20*time.Second).Unix()), 1, now), "then the window's reset")
	s.Equal(maxLivePause, liveBackoff(limited(0, now.Add(10*time.Minute).Unix()), 1, now), "never past a minute")
	other := errors.New("connection reset")
	s.Equal(time.Second, liveBackoff(other, 1, now))
	s.Equal(2*time.Second, liveBackoff(other, 2, now))
	s.Equal(8*time.Second, liveBackoff(other, 4, now))
	s.Equal(8*time.Second, liveBackoff(other, 12, now))
}
