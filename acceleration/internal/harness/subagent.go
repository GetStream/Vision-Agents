package harness

import (
	"context"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
)

// CaptureRequest selects evidence for one task. Zero Frames uses the session default.
type CaptureRequest struct {
	TaskID string
	At     time.Time
	Source string
	Frames int
}

// opening is the subagent's session, which may still be starting: it is opened in the
// background so the call does not wait on a model it may never ask anything.
type opening struct {
	ready   chan struct{}
	session *llmrouter.Session
	err     error
}

// open starts the subagent's session in the background. One it replaces stays open for
// the tasks already running on it, and closes with the manager.
func (m *manager) open(start func(context.Context) (*llmrouter.Session, error)) {
	opened := &opening{ready: make(chan struct{})}
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return
	}
	if m.subagent != nil {
		m.retired = append(m.retired, m.subagent)
	}
	m.subagent = opened
	m.warming.Add(1)
	m.mu.Unlock()

	go func() {
		defer m.warming.Done()
		defer close(opened.ready)
		opened.session, opened.err = start(m.ctx)
	}()
}

// current is the subagent new work is asked of.
func (m *manager) current() *opening {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.subagent
}

func (m *manager) model(ctx context.Context) (*llmrouter.Session, error) {
	current := m.current()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-current.ready:
		return current.session, current.err
	}
}
