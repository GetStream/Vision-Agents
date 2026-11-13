//go:build integration

package main

import (
	"context"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagejobs"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagerouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

// slowDrawer draws a picture once it is let go, as a provider taking its minute does.
type slowDrawer struct {
	entered chan struct{}
	letGo   chan struct{}
}

func (d *slowDrawer) Generate(ctx context.Context, _ imagegen.Request) (imagegen.Result, error) {
	d.entered <- struct{}{}
	select {
	case <-d.letGo:
	case <-ctx.Done():
		return imagegen.Result{}, imagegen.Fail(imagegen.Timeout, true, ctx.Err())
	}
	return imagegen.Result{Images: []imagegen.Image{{Data: []byte("png"), MediaType: "image/png", Width: 64, Height: 64}}}, nil
}

func (d *slowDrawer) Start(context.Context) error { return nil }
func (d *slowDrawer) Close() error                { return nil }
func (d *slowDrawer) Provider() string            { return "stub" }
func (d *slowDrawer) Model() string               { return "draw" }

// postedTexts is a conversation that keeps what is posted into it.
type postedTexts struct {
	mu    sync.Mutex
	texts []string
}

func (p *postedTexts) UploadTo(context.Context, string, int64, string, string, sandbox.File) (sandbox.Attachment, error) {
	return sandbox.Attachment{Name: "picture.png", MIME: "image/png", URL: "https://cdn.example/picture.png", Size: 3}, nil
}

func (p *postedTexts) Post(_ context.Context, _ string, _ int64, _, _, _ string, _ time.Time, text string, _ []sandbox.Attachment) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.texts = append(p.texts, text)
	return nil
}

func (p *postedTexts) posted() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.texts...)
}

// A router told to stop while it draws a picture holds its drain until the picture is posted:
// cut short, the person was told a picture was on its way and another router would have to
// draw it again.
func (s *DrainSuite) TestTheDrainWaitsForAPictureBeingDrawn() {
	drawer := &slowDrawer{entered: make(chan struct{}, 1), letGo: make(chan struct{})}
	registry := imagerouter.NewRegistry()
	registry.Register("stub", func(routing.Spec) (imagegen.Provider, error) { return drawer, nil })
	images, err := imagerouter.New(imagerouter.Options{
		Config: routing.ModalityConfig{Providers: []routing.ProviderConfig{{
			Provider: "stub", Model: "draw", Languages: []string{"en"}, Price: routing.Price{PerImage: 0.04},
		}}},
		Registry: registry,
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(images.Close)
	drawings, err := imagejobs.New(imagejobs.Options{Store: s.db, Images: images, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	signal, stop := context.WithCancel(s.ctx)
	defer stop()
	conversation := &postedTexts{}
	drawings.Start(signal, conversation)
	s.T().Cleanup(drawings.Close)
	s.Require().NoError(drawings.Queue(s.ctx, imagejobs.Request{
		CustomerID: "customer-" + uuid.NewString(), CID: "agent:" + uuid.NewString(), AgentUser: "illustrator",
		SessionID: "session-" + uuid.NewString(), ToolCallID: "draw-1", Target: "stub/draw", Prompt: "a field",
	}))
	select {
	case <-drawer.entered:
	case <-time.After(10 * time.Second):
		s.FailNow("the picture was never drawn")
	}

	stop()
	stopped := make(chan error, 1)
	go func() {
		stopped <- stopServing(stopping{
			public:    &http.Server{ReadHeaderTimeout: time.Second},
			drainWork: func() {},
			held:      func() int { return running(nil, nil, nil, nil, nil, nil, drawings) },
			telephony: idlePhone(s.T()),
			drain:     time.Minute,
			grace:     testGrace,
			logger:    slog.New(slog.DiscardHandler),
		})
	}()

	select {
	case <-stopped:
		s.FailNow("the drain ended while the picture was being drawn")
	case <-time.After(500 * time.Millisecond):
	}
	close(drawer.letGo)
	select {
	case err := <-stopped:
		s.Require().NoError(err)
	case <-time.After(10 * time.Second):
		s.FailNow("the drain never ended once the picture was posted")
	}
	s.Equal([]string{"Here is the picture."}, conversation.posted(), "posted before the router let go")
}
