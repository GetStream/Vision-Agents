//go:build integration

package imagejobs_test

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagejobs"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagerouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// drawer is an image provider with no model behind it. It draws a fixed picture, fails as the
// test says, or holds each draw until the test lets it go, and counts what it was asked.
type drawer struct {
	mu    sync.Mutex
	err   error
	asked int
	// held, when set, holds every draw until it is closed or the draw's context ends; entered
	// is told each time a draw starts.
	held    chan struct{}
	entered chan struct{}
}

func (d *drawer) Generate(ctx context.Context, _ imagegen.Request) (imagegen.Result, error) {
	d.mu.Lock()
	d.asked++
	held, entered, err := d.held, d.entered, d.err
	d.mu.Unlock()
	if entered != nil {
		entered <- struct{}{}
	}
	if held != nil {
		select {
		case <-held:
		case <-ctx.Done():
			return imagegen.Result{}, imagegen.Fail(imagegen.Timeout, true, ctx.Err())
		}
	}
	if err != nil {
		return imagegen.Result{}, err
	}
	return imagegen.Result{Images: []imagegen.Image{{Data: []byte("png"), MediaType: "image/png", Width: 1024, Height: 1024}}}, nil
}

func (d *drawer) Start(context.Context) error { return nil }
func (d *drawer) Close() error                { return nil }
func (d *drawer) Provider() string            { return "stub" }
func (d *drawer) Model() string               { return "draw" }

func (d *drawer) times() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.asked
}

// losingPoster writes each post and then, the first time, says it failed, as a router does
// that stopped after the message was written and before it recorded so.
type losingPoster struct {
	imagejobs.Poster
	mu   sync.Mutex
	lost bool
}

func (p *losingPoster) Post(ctx context.Context, customer string, app int64, cid, agent, id string, at time.Time,
	text string, files []sandbox.Attachment) error {
	if err := p.Poster.Post(ctx, customer, app, cid, agent, id, at, text, files); err != nil {
		return err
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.lost {
		p.lost = true
		return errors.New("the answer was lost")
	}
	return nil
}

// ImageJobsSuite queues pictures into Postgres and draws them with workers that post into a
// Stream Chat held in memory, through the real conversation service and image router.
type ImageJobsSuite struct {
	suite.Suite
	ctx      context.Context
	store    *store.Store
	chat     *chattest.Server
	posts    *conversation.Service
	drawer   *drawer
	images   *imagerouter.Router
	customer string
	cid      string
}

func TestImageJobsSuite(t *testing.T) {
	suite.Run(t, new(ImageJobsSuite))
}

func (s *ImageJobsSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN must be set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "imagejobs"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
}

func (s *ImageJobsSuite) SetupTest() {
	_, err := s.store.DB().ExecContext(s.ctx, "TRUNCATE image_jobs")
	s.Require().NoError(err)
	s.chat = chattest.NewServer(s.T())
	s.posts = conversation.NewForChat(s.chat.Client)
	s.T().Cleanup(s.posts.Close)
	s.drawer = &drawer{}
	var images *imagerouter.Router
	registry := imagerouter.NewRegistry()
	registry.Register("stub", func(routing.Spec) (imagegen.Provider, error) { return s.drawer, nil })
	images, err = imagerouter.New(imagerouter.Options{
		Config: routing.ModalityConfig{Providers: []routing.ProviderConfig{{
			Provider: "stub", Model: "draw", Languages: []string{"en"}, Tier: routing.LowLatency,
			Price: routing.Price{PerImage: 0.04},
		}}},
		Registry: registry,
		Store:    s.store,
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(images.Close)
	s.images = images
	s.customer = "customer-" + uuid.NewString()
	// The session that asked opened the conversation before the picture was queued.
	opened, _, _, err := s.posts.OpenForCaller(s.ctx, s.customer, "illustrator", "", "ana")
	s.Require().NoError(err)
	s.cid = opened.CID()
	opened.Release()
}

// worker is a router's worker, with a lease and a post retry short enough to wait out.
func (s *ImageJobsSuite) worker() *imagejobs.Service {
	worker, err := imagejobs.New(imagejobs.Options{
		Store: s.store, Images: s.images, Lease: 300 * time.Millisecond, Poll: 20 * time.Millisecond,
		PostRetry: 50 * time.Millisecond, Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	return worker
}

// queue asks worker for a picture, as generate_image does once its turn ended.
func (s *ImageJobsSuite) queue(worker *imagejobs.Service) {
	s.Require().NoError(worker.Queue(s.ctx, imagejobs.Request{
		CustomerID: s.customer, CID: s.cid, AgentUser: "illustrator", ToolCallID: "draw-1",
		AgentID: "illustrator", Target: "stub/draw", Prompt: "a yellow watering can",
		Attribution: routing.Attribution{APIKey: "key-1", UserID: "ana", SessionID: "session-1",
			UsageCategories: []string{"team:design"}},
	}))
}

// posted are the agent's finished messages in the conversation.
func (s *ImageJobsSuite) posted() []map[string]any {
	var posted []map[string]any
	for _, stored := range s.chat.Stored(s.cid) {
		custom, _ := stored["custom"].(map[string]any)
		message, _ := custom["support_message"].(map[string]any)
		if message["role"] == "assistant" && message["state"] == "completed" {
			posted = append(posted, stored)
		}
	}
	return posted
}

// awaitPosted waits for the conversation to hold n of the agent's messages, and returns them.
func (s *ImageJobsSuite) awaitPosted(n int) []map[string]any {
	s.Require().Eventually(func() bool { return len(s.posted()) >= n }, 10*time.Second, 20*time.Millisecond,
		"the conversation never got its message")
	return s.posted()
}

// job is the one row queued for the conversation, once it left queued.
func (s *ImageJobsSuite) job() store.ImageJob {
	var job store.ImageJob
	s.Require().Eventually(func() bool {
		err := s.store.DB().NewSelect().Model(&job).Where("cid = ?", s.cid).Scan(s.ctx)
		return err == nil && job.State != store.ImageJobQueued
	}, 10*time.Second, 20*time.Millisecond, "the job never finished")
	return job
}

func (s *ImageJobsSuite) TestAPictureIsPostedIntoTheConversationOnceDrawn() {
	worker := s.worker()
	worker.Start(s.ctx, s.posts)
	s.T().Cleanup(worker.Close)

	s.queue(worker)

	posted := s.awaitPosted(1)
	s.Require().Len(posted, 1)
	s.Equal("illustrator", posted[0]["user_id"], "posted as the agent")
	attachments, _ := posted[0]["attachments"].([]any)
	s.Require().Len(attachments, 1, "the picture is attached")
	uploads := s.chat.Uploads()
	s.Require().Len(uploads, 1)
	s.JSONEq(`{"id":"illustrator"}`, uploads[0].User, "uploaded as the agent")
	job := s.job()
	s.Equal(store.ImageJobDone, job.State)
	s.Equal(1, job.Attempts)
	s.Equal(1, s.drawer.times())

	page, err := s.posts.HistoryForCaller(s.ctx, s.customer, "illustrator", s.cid, "", "ana")
	s.Require().NoError(err)
	s.Require().Len(page.Messages, 1, "the next turn reads the picture back with the conversation")
	s.Equal("assistant", page.Messages[0].Role)
	s.Len(page.Messages[0].Files, 1)
}

func (s *ImageJobsSuite) TestADrawIsBilledUnderTheSessionThatAskedWhenItIsDrawn() {
	// The row is what spend limits sum: one missing the user, team or session is spent by
	// nobody those limits count.
	worker := s.worker()
	worker.Start(s.ctx, s.posts)
	s.T().Cleanup(worker.Close)

	s.queue(worker)
	s.awaitPosted(1)

	var rows []store.Request
	s.Require().Eventually(func() bool {
		rows = nil
		err := s.store.DB().NewSelect().Model(&rows).
			Where("customer_id = ? AND modality = ?", s.customer, string(routing.Image)).Scan(s.ctx)
		return err == nil && len(rows) == 1
	}, 10*time.Second, 20*time.Millisecond, "the draw was never recorded")
	row := rows[0]
	s.True(row.Success)
	s.Equal("illustrator", row.AgentID)
	s.Equal("key-1", row.APIKey)
	s.Equal("ana", row.UserID)
	s.Equal("session-1", row.SessionID)
	s.Equal([]string{"team:design"}, row.UsageCategories)
	s.EqualValues(1, row.Quantities()["images"])
}

func (s *ImageJobsSuite) TestAPictureTheProviderFailedPostsWhyAndCostsNothing() {
	// Never silent: the person was told a picture was on its way.
	s.drawer.err = imagegen.Fail(imagegen.ProviderFailed, true, errors.New("upstream 500"))
	worker := s.worker()
	worker.Start(s.ctx, s.posts)
	s.T().Cleanup(worker.Close)

	s.queue(worker)

	posted := s.awaitPosted(1)
	s.Contains(posted[0]["text"], "could not draw that picture")
	s.Empty(posted[0]["attachments"])
	s.Empty(s.chat.Uploads())
	job := s.job()
	s.Equal(store.ImageJobFailed, job.State)
	s.Contains(job.Outcome, "could not draw")
}

func (s *ImageJobsSuite) TestAPictureTheSafetyFilterRefusedSaysSo() {
	s.drawer.err = imagegen.Fail(imagegen.ContentFiltered, true, errors.New("flagged"))
	worker := s.worker()
	worker.Start(s.ctx, s.posts)
	s.T().Cleanup(worker.Close)

	s.queue(worker)

	posted := s.awaitPosted(1)
	s.Contains(posted[0]["text"], "safety filter refused it")
	s.Equal(1, s.drawer.times(), "not asked again with the same prompt")
}

func (s *ImageJobsSuite) TestARouterThatStoppedMidDrawLeavesThePictureToTheNext() {
	// A deploy rolls the routers, and one can be killed in the middle of a draw.
	s.drawer.held, s.drawer.entered = make(chan struct{}), make(chan struct{}, 4)
	stopped := s.worker()
	stopped.Start(s.ctx, s.posts)
	s.queue(stopped)
	select {
	case <-s.drawer.entered:
	case <-time.After(10 * time.Second):
		s.FailNow("the first router never started drawing")
	}

	stopped.Close()
	s.Empty(s.posted(), "the stopped router posted nothing")
	s.drawer.mu.Lock()
	s.drawer.held = nil
	s.drawer.mu.Unlock()
	next := s.worker()
	next.Start(s.ctx, s.posts)
	s.T().Cleanup(next.Close)

	posted := s.awaitPosted(1)
	s.Require().Len(posted, 1)
	attachments, _ := posted[0]["attachments"].([]any)
	s.Len(attachments, 1)
	job := s.job()
	s.Equal(store.ImageJobDone, job.State)
	s.Equal(2, job.Attempts, "claimed again once the stopped router's lease ran out")
}

func (s *ImageJobsSuite) TestAPostWhoseAnswerWasLostIsNotPostedTwice() {
	worker := s.worker()
	worker.Start(s.ctx, &losingPoster{Poster: s.posts})
	s.T().Cleanup(worker.Close)

	s.queue(worker)

	job := s.job()
	s.Equal(store.ImageJobDone, job.State)
	s.Equal(2, job.Attempts, "the lost answer made the job due again")
	s.Len(s.posted(), 1, "the second post found the message written")
	s.Equal(1, s.drawer.times(), "the kept picture was posted, not drawn again")
}

func (s *ImageJobsSuite) TestAJobClaimedMoreTimesThanItMayBeDrawnPostsItsFailure() {
	// Each claim that never finished may be a draw that kills routers: it is not tried forever.
	worker := s.worker()
	s.queue(worker)
	_, err := s.store.DB().NewUpdate().Model((*store.ImageJob)(nil)).Set("attempts = 2").Where("cid = ?", s.cid).Exec(s.ctx)
	s.Require().NoError(err)

	worker.Start(s.ctx, s.posts)
	s.T().Cleanup(worker.Close)

	posted := s.awaitPosted(1)
	s.Contains(posted[0]["text"], "could not draw that picture")
	s.Zero(s.drawer.times())
	s.Equal(store.ImageJobFailed, s.job().State)
}

func (s *ImageJobsSuite) TestACallRunAgainQueuesNoSecondPicture() {
	worker := s.worker()
	s.queue(worker)
	s.queue(worker)

	var count int
	s.Require().NoError(s.store.DB().NewSelect().Model((*store.ImageJob)(nil)).ColumnExpr("count(*)").
		Where("cid = ?", s.cid).Scan(s.ctx, &count))
	s.Equal(1, count)
}
