package harness

import (
	"context"
	"fmt"
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

type worker struct {
	ready   chan struct{}
	session *llmrouter.Session
	err     error
}

func (m *manager) prepare(openers map[string]func(context.Context) (*llmrouter.Session, error)) {
	for name, open := range openers {
		w := &worker{ready: make(chan struct{})}
		m.workers[name] = w
		m.warming.Add(1)
		go func() {
			defer m.warming.Done()
			defer close(w.ready)
			w.session, w.err = open(m.ctx)
		}()
	}
}

func (m *manager) model(ctx context.Context, name string) (*llmrouter.Session, error) {
	if name == "" {
		name = "default"
	}
	w, ok := m.workers[name]
	if !ok {
		return nil, fmt.Errorf("harness: no worker called %q", name)
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-w.ready:
		return w.session, w.err
	}
}
