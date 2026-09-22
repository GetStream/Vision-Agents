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

func (m *manager) open(start func(context.Context) (*llmrouter.Session, error)) {
	m.subagent = &opening{ready: make(chan struct{})}
	m.warming.Add(1)
	go func() {
		defer m.warming.Done()
		defer close(m.subagent.ready)
		m.subagent.session, m.subagent.err = start(m.ctx)
	}()
}

func (m *manager) model(ctx context.Context) (*llmrouter.Session, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-m.subagent.ready:
		return m.subagent.session, m.subagent.err
	}
}
