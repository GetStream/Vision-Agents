//go:build integration

package imagejobs_test

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
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

// drawer is an image provider with no model behind it. It draws a fixed picture, fails its
// next draws as the test says, or holds each draw until the test lets it go, and counts what it
// was asked and how many it drew at once.
type drawer struct {
	mu sync.Mutex
	// failing are what the next draws fail with, in turn; err what every later one fails with.
	failing []error
	err     error
	asked   int
	running int
	most    int
	// held, when set, holds every draw until it is closed or the draw's context ends; entered
	// is told each time a draw starts.
	held    chan struct{}
	entered chan struct{}
	// takes is how long each draw takes.
	takes time.Duration
	// outlasts, when set, has a draw finish only once its context ended, and finish drawn: a
	// picture that came back just as the router stopped.
	outlasts bool
}

func (d *drawer) Generate(ctx context.Context, _ imagegen.Request) (imagegen.Result, error) {
	d.mu.Lock()
	d.asked++
	d.running++
	d.most = max(d.most, d.running)
	held, entered, err, takes, outlasts := d.held, d.entered, d.err, d.takes, d.outlasts
	if len(d.failing) > 0 {
		err, d.failing = d.failing[0], d.failing[1:]
	}
	d.mu.Unlock()
	defer func() {
		d.mu.Lock()
		d.running--
		d.mu.Unlock()
	}()
	if entered != nil {
		entered <- struct{}{}
	}
	if outlasts {
		<-ctx.Done()
		return imagegen.Result{Images: []imagegen.Image{{Data: []byte("png"), MediaType: "image/png", Width: 1024, Height: 1024}}}, nil
	}
	if held != nil {
		select {
		case <-held:
		case <-ctx.Done():
			return imagegen.Result{}, imagegen.Fail(imagegen.Timeout, true, ctx.Err())
		}
	}
	select {
	case <-time.After(takes):
	case <-ctx.Done():
		return imagegen.Result{}, imagegen.Fail(imagegen.Timeout, true, ctx.Err())
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

func (d *drawer) mostAtOnce() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.most
}

func (d *drawer) letGo() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.held = nil
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
	session  string
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
	registry := imagerouter.NewRegistry()
	registry.Register("stub", func(routing.Spec) (imagegen.Provider, error) { return s.drawer, nil })
	images, err := imagerouter.New(imagerouter.Options{
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
	s.session = "session-" + uuid.NewString()
	// The session that asked opened the conversation before the picture was queued.
	opened, _, _, err := s.posts.OpenForCaller(s.ctx, s.customer, "illustrator", "", "ana")
	s.Require().NoError(err)
	s.cid = opened.CID()
	opened.Release()
}

// worker is a router's worker, with a lease and waits short enough to wait out.
func (s *ImageJobsSuite) worker() *imagejobs.Service {
	worker, err := imagejobs.New(imagejobs.Options{
		Store: s.store, Images: s.images, Lease: 300 * time.Millisecond, Poll: 20 * time.Millisecond,
		PostBackoff: []time.Duration{50 * time.Millisecond, 50 * time.Millisecond, 50 * time.Millisecond},
		UntakenWait: 50 * time.Millisecond, Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	return worker
}

// started is a worker posting into the conversation through poster, closed when the test ends.
func (s *ImageJobsSuite) started(poster imagejobs.Poster) *imagejobs.Service {
	worker := s.worker()
	worker.Start(s.ctx, poster)
	s.T().Cleanup(worker.Close)
	return worker
}

// queue asks worker for a picture, as a call of the session's first turn does once it ended.
func (s *ImageJobsSuite) queue(worker *imagejobs.Service, call string) {
	s.queueIn(worker, "turn-1", call, "a yellow watering can")
}

func (s *ImageJobsSuite) queueIn(worker *imagejobs.Service, turn, call, prompt string) {
	s.Require().NoError(worker.Queue(s.ctx, imagejobs.Request{
		CustomerID: s.customer, CID: s.cid, AgentUser: "illustrator", SessionID: s.session, TurnID: turn, ToolCallID: call,
		AgentID: "illustrator", Target: "stub/draw", Prompt: prompt,
		Attribution: routing.Attribution{APIKey: "key-1", UserID: "ana", SessionID: s.session,
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
		"the conversation never got its messages")
	return s.posted()
}

func pictures(posted map[string]any) int {
	attachments, _ := posted["attachments"].([]any)
	return len(attachments)
}

// jobs are the conversation's rows once none is queued any more.
func (s *ImageJobsSuite) jobs() []store.ImageJob {
	var jobs []store.ImageJob
	s.Require().Eventually(func() bool {
		jobs = nil
		err := s.store.DB().NewSelect().Model(&jobs).Where("cid = ?", s.cid).Order("tool_call_id").Scan(s.ctx)
		if err != nil || len(jobs) == 0 {
			return false
		}
		for _, job := range jobs {
			if job.State == store.ImageJobQueued {
				return false
			}
		}
		return true
	}, 10*time.Second, 20*time.Millisecond, "the jobs never finished")
	return jobs
}

func (s *ImageJobsSuite) job() store.ImageJob {
	jobs := s.jobs()
	s.Require().Len(jobs, 1)
	return jobs[0]
}

// rows are the usage rows the conversation's draws wrote.
func (s *ImageJobsSuite) rows(n int) []store.Request {
	var rows []store.Request
	s.Require().Eventually(func() bool {
		rows = nil
		err := s.store.DB().NewSelect().Model(&rows).
			Where("customer_id = ? AND modality = ?", s.customer, string(routing.Image)).Scan(s.ctx)
		return err == nil && len(rows) == n
	}, 10*time.Second, 20*time.Millisecond, "the draws were never recorded")
	return rows
}

func (s *ImageJobsSuite) TestAPictureIsPostedIntoTheConversationOnceDrawn() {
	worker := s.started(s.posts)

	s.queue(worker, "draw-1")

	posted := s.awaitPosted(1)
	s.Require().Len(posted, 1)
	s.Equal("illustrator", posted[0]["user_id"], "posted as the agent")
	s.Equal(1, pictures(posted[0]), "the picture is attached")
	uploads := s.chat.Uploads()
	s.Require().Len(uploads, 1)
	s.JSONEq(`{"id":"illustrator"}`, uploads[0].User, "uploaded as the agent")
	job := s.job()
	s.Equal(store.ImageJobDone, job.State)
	s.Equal(1, job.Draws)
	s.Nil(job.Drawn, "the drawn bytes are let go of once posted")

	page, err := s.posts.HistoryForCaller(s.ctx, s.customer, "illustrator", s.cid, "", "ana")
	s.Require().NoError(err)
	s.Require().Len(page.Messages, 1, "the next turn reads the picture back with the conversation")
	s.Equal("assistant", page.Messages[0].Role)
	s.Len(page.Messages[0].Files, 1)
}

func (s *ImageJobsSuite) TestADrawIsBilledUnderTheSessionThatAskedWhenItIsDrawn() {
	// The row is what spend limits sum: one missing the user, team or session is spent by
	// nobody those limits count.
	worker := s.started(s.posts)

	s.queue(worker, "draw-1")
	s.awaitPosted(1)

	row := s.rows(1)[0]
	s.True(row.Success)
	s.Equal("illustrator", row.AgentID)
	s.Equal("key-1", row.APIKey)
	s.Equal("ana", row.UserID)
	s.Equal(s.session, row.SessionID)
	s.Equal([]string{"team:design"}, row.UsageCategories)
	s.EqualValues(1, row.Quantities()["images"])
	s.EqualValues(40_000, row.CostMicros, "four cents a picture")
	s.Equal(routing.UsageID("image_job", s.job().ID), row.UsageID,
		"the job's own id, so a picture drawn again after a stopped router settles on this row")
}

func (s *ImageJobsSuite) TestAPictureTheProviderFailedPostsWhyAndCostsNothing() {
	// Never silent: the person was told a picture was on its way.
	s.drawer.err = imagegen.Fail(imagegen.ProviderFailed, true, errors.New("upstream 500"))
	worker := s.started(s.posts)

	s.queue(worker, "draw-1")

	posted := s.awaitPosted(1)
	s.Contains(posted[0]["text"], "could not draw that picture")
	s.Zero(pictures(posted[0]))
	s.Empty(s.chat.Uploads())
	job := s.job()
	s.Equal(store.ImageJobFailed, job.State)
	s.Contains(job.Outcome, "could not draw")
	row := s.rows(1)[0]
	s.False(row.Success)
	s.Zero(row.CostMicros, "a draw that failed is not billed")
	s.Zero(row.Quantities()["images"])
}

func (s *ImageJobsSuite) TestAPictureTheSafetyFilterRefusedSaysSo() {
	s.drawer.err = imagegen.Fail(imagegen.ContentFiltered, true, errors.New("flagged"))
	worker := s.started(s.posts)

	s.queue(worker, "draw-1")

	posted := s.awaitPosted(1)
	s.Contains(posted[0]["text"], "safety filter refused it")
	s.Equal(1, s.drawer.times(), "not asked again with the same prompt")
}

func (s *ImageJobsSuite) TestFourPicturesAtOnceAreAllDrawnTwoAtATime() {
	// One reply may ask for four. Drawn all at once, the third and fourth would wait out their
	// deadline for FAL's two slots and be lost while another router's slots sat idle.
	s.drawer.takes = 100 * time.Millisecond
	worker := s.started(s.posts)

	for n := 1; n <= 4; n++ {
		s.queue(worker, fmt.Sprintf("draw-%d", n))
	}

	posted := s.awaitPosted(4)
	s.Len(posted, 4)
	for _, one := range posted {
		s.Equal(1, pictures(one))
	}
	s.LessOrEqual(s.drawer.mostAtOnce(), 2, "no more draws at once than the provider runs")
	s.Equal(4, s.drawer.times())
}

func (s *ImageJobsSuite) TestADrawNoProviderTookInTimeIsTriedAgainUncounted() {
	// Such as one that waited out its deadline for a slot the endpoint's own requests held: it
	// cost nothing and drew nothing, so it is neither a failure nor one of the job's draws.
	s.drawer.failing = []error{imagegen.Fail(imagegen.Timeout, false, fmt.Errorf("%w: %w", imagegen.ErrNotStarted, context.DeadlineExceeded))}
	worker := s.started(s.posts)

	s.queue(worker, "draw-1")

	posted := s.awaitPosted(1)
	s.Equal(1, pictures(posted[0]))
	job := s.job()
	s.Equal(store.ImageJobDone, job.State)
	s.Equal(2, s.drawer.times())
	s.Equal(1, job.Draws, "the draw no provider took is not counted")
}

func (s *ImageJobsSuite) TestADrawSentThatTimedOutIsCountedAndSaysSo() {
	// Gemini's call that ran out of time, or a FAL submit that timed out once FAL took it: the
	// vendor may have drawn, and billed, it, so it is not sent again as if it never left.
	s.drawer.err = imagegen.Fail(imagegen.Timeout, false, context.DeadlineExceeded)
	worker := s.started(s.posts)

	s.queue(worker, "draw-1")

	posted := s.awaitPosted(1)
	s.Contains(posted[0]["text"], "took too long")
	s.Equal(1, s.drawer.times(), "not sent again")
	job := s.job()
	s.Equal(store.ImageJobFailed, job.State)
	s.Equal(1, job.Draws)
}

func (s *ImageJobsSuite) TestADrawNeverStartedForHalfAnHourPostsThatItTookTooLong() {
	// A provider whose slots stay taken would otherwise keep the person waiting for ever.
	s.drawer.err = imagegen.Fail(imagegen.Timeout, false, fmt.Errorf("%w: %w", imagegen.ErrNotStarted, context.DeadlineExceeded))
	worker := s.worker()
	s.queue(worker, "draw-1")
	_, err := s.store.DB().NewUpdate().Model((*store.ImageJob)(nil)).
		Set("created_at = now() - interval '31 minutes'").Where("cid = ?", s.cid).Exec(s.ctx)
	s.Require().NoError(err)

	worker.Start(s.ctx, s.posts)
	s.T().Cleanup(worker.Close)

	posted := s.awaitPosted(1)
	s.Contains(posted[0]["text"], "took too long")
	s.Equal(1, s.drawer.times(), "given up, not tried again")
	s.Equal(store.ImageJobFailed, s.job().State)
}

func (s *ImageJobsSuite) TestADrawThatFinishesAsTheRouterStopsIsKeptNotDrawnAgain() {
	// The picture is drawn and billed; thrown away, the next router would draw it again.
	s.drawer.outlasts, s.drawer.entered = true, make(chan struct{}, 4)
	stopping := s.worker()
	stopping.Start(s.ctx, s.posts)
	s.queue(stopping, "draw-1")
	select {
	case <-s.drawer.entered:
	case <-time.After(10 * time.Second):
		s.FailNow("the router never started drawing")
	}

	stopping.Close()
	var kept store.ImageJob
	s.Require().NoError(s.store.DB().NewSelect().Model(&kept).Where("cid = ?", s.cid).Scan(s.ctx))
	s.Equal("Here is the picture.", kept.Message, "the drawn picture is kept with its message")
	s.Equal([]byte("png"), kept.Drawn)
	s.Equal(1, kept.Draws)

	s.drawer.mu.Lock()
	s.drawer.outlasts = false
	s.drawer.mu.Unlock()
	s.started(s.posts)
	posted := s.awaitPosted(1)
	s.Equal(1, pictures(posted[0]))
	s.Equal(1, s.drawer.times(), "posted, not drawn again")
}

func (s *ImageJobsSuite) TestARouterThatStopsMidDrawGivesThePictureBackAtOnce() {
	// A router stopping without a drain cuts the draw: the job is due again now, not after its
	// lease, and the cut draw does not count against the job.
	s.drawer.held, s.drawer.entered = make(chan struct{}), make(chan struct{}, 4)
	stopped := s.worker()
	stopped.Start(s.ctx, s.posts)
	s.queue(stopped, "draw-1")
	select {
	case <-s.drawer.entered:
	case <-time.After(10 * time.Second):
		s.FailNow("the first router never started drawing")
	}

	stopped.Close()
	var given store.ImageJob
	s.Require().NoError(s.store.DB().NewSelect().Model(&given).Where("cid = ?", s.cid).Scan(s.ctx))
	s.Equal(store.ImageJobQueued, given.State)
	s.Zero(given.Draws, "the cut draw is not counted")
	s.False(given.NextAttemptAt.After(time.Now()), "due again now, not when the lease runs out")
	s.Empty(s.posted(), "the stopped router posted nothing")

	s.drawer.letGo()
	s.started(s.posts)
	posted := s.awaitPosted(1)
	s.Require().Len(posted, 1)
	s.Equal(1, pictures(posted[0]))
	job := s.job()
	s.Equal(store.ImageJobDone, job.State)
	s.Equal(1, job.Draws)
}

func (s *ImageJobsSuite) TestARouterKilledMidDrawLeavesThePictureToTheNextOnceItsLeaseRunsOut() {
	// Killed, a router writes nothing: the row is left counted and claimed until the lease ends.
	worker := s.worker()
	s.queue(worker, "draw-1")
	_, err := s.store.DB().NewUpdate().Model((*store.ImageJob)(nil)).
		Set("attempts = 1, draws = 1, next_attempt_at = now() + interval '200 milliseconds'").
		Where("cid = ?", s.cid).Exec(s.ctx)
	s.Require().NoError(err)

	worker.Start(s.ctx, s.posts)
	s.T().Cleanup(worker.Close)

	posted := s.awaitPosted(1)
	s.Equal(1, pictures(posted[0]))
	job := s.job()
	s.Equal(2, job.Draws, "drawn once more, within the limit")
	s.Equal(2, job.Attempts)
}

func (s *ImageJobsSuite) TestAJobDrawnAsOftenAsItMayBePostsItsFailure() {
	// Each draw that never finished may be one that ends routers: it is not tried forever.
	worker := s.worker()
	s.queue(worker, "draw-1")
	_, err := s.store.DB().NewUpdate().Model((*store.ImageJob)(nil)).Set("draws = 2").Where("cid = ?", s.cid).Exec(s.ctx)
	s.Require().NoError(err)

	worker.Start(s.ctx, s.posts)
	s.T().Cleanup(worker.Close)

	posted := s.awaitPosted(1)
	s.Contains(posted[0]["text"], "could not draw that picture")
	s.Zero(s.drawer.times())
	s.Equal(store.ImageJobFailed, s.job().State)
}

func (s *ImageJobsSuite) TestAPostWhoseAnswerWasLostIsNotPostedTwice() {
	s.started(&losingPoster{Poster: s.posts})
	worker := s.worker()

	s.queue(worker, "draw-1")

	job := s.job()
	s.Equal(store.ImageJobDone, job.State)
	s.Equal(1, job.Posts, "the lost answer made the job due again")
	s.Len(s.posted(), 1, "the second post found the message written")
	s.Equal(1, s.drawer.times(), "the kept picture was posted, not drawn again")
	s.rows(1)
}

func (s *ImageJobsSuite) TestAFailurePostedButNotRecordedIsNotFollowedByAPaidDraw() {
	// The person was shown the failure: a later claim posts the same failure, never a picture
	// drawn and billed behind it that the conversation would then take as already written.
	s.drawer.failing = []error{imagegen.Fail(imagegen.ProviderFailed, true, errors.New("upstream 500"))}
	s.started(&losingPoster{Poster: s.posts})
	worker := s.worker()

	s.queue(worker, "draw-1")

	job := s.job()
	s.Equal(store.ImageJobFailed, job.State)
	s.Equal(1, s.drawer.times(), "not drawn again")
	posted := s.posted()
	s.Require().Len(posted, 1)
	s.Contains(posted[0]["text"], "could not draw that picture")
}

func (s *ImageJobsSuite) TestAnUploadStreamChatFailedIsTriedAgainWithoutDrawing() {
	s.chat.Fail(http.MethodPost, "/image")
	worker := s.started(s.posts)

	s.queue(worker, "draw-1")
	s.Require().Eventually(func() bool {
		var job store.ImageJob
		err := s.store.DB().NewSelect().Model(&job).Where("cid = ?", s.cid).Scan(s.ctx)
		return err == nil && job.Posts >= 1
	}, 10*time.Second, 20*time.Millisecond, "the upload was never tried")
	s.chat.Heal(http.MethodPost, "/image")

	posted := s.awaitPosted(1)
	s.Equal(1, pictures(posted[0]))
	s.Equal(1, s.drawer.times(), "drawn once, uploaded twice")
	s.rows(1)
}

func (s *ImageJobsSuite) TestAStreamChatRequestTimeoutIsTriedAgain() {
	// Stream Chat's RequestTimeoutError is a 408: the next try may well pass.
	s.chat.FailWith(http.MethodPost, "/image", http.StatusRequestTimeout)
	worker := s.started(s.posts)

	s.queue(worker, "draw-1")
	s.Require().Eventually(func() bool {
		var job store.ImageJob
		err := s.store.DB().NewSelect().Model(&job).Where("cid = ?", s.cid).Scan(s.ctx)
		return err == nil && job.Posts >= 1
	}, 10*time.Second, 20*time.Millisecond, "the upload was never tried")
	s.chat.Heal(http.MethodPost, "/image")

	posted := s.awaitPosted(1)
	s.Equal(1, pictures(posted[0]))
	s.Equal(1, s.drawer.times())
	s.Equal(store.ImageJobDone, s.job().State)
}

func (s *ImageJobsSuite) TestAnUploadStreamChatRefusedForGoodPostsThatItCouldNotBePosted() {
	// Such as a file the app's upload settings refuse: uploading it again gets the same answer,
	// and drawing it again would bill the same picture twice.
	s.chat.FailWith(http.MethodPost, "/image", http.StatusForbidden)
	worker := s.started(s.posts)

	s.queue(worker, "draw-1")

	posted := s.awaitPosted(1)
	s.Contains(posted[0]["text"], "could not be posted here")
	s.Zero(pictures(posted[0]))
	s.Equal(1, s.drawer.times())
	job := s.job()
	s.Equal(store.ImageJobFailed, job.State)
	s.rows(1)
}

func (s *ImageJobsSuite) TestAFailureWhosePostFailedIsPostedLater() {
	s.drawer.err = imagegen.Fail(imagegen.ProviderFailed, true, errors.New("upstream 500"))
	s.chat.Fail(http.MethodPost, "/message")
	worker := s.started(s.posts)

	s.queue(worker, "draw-1")
	s.Require().Eventually(func() bool {
		var job store.ImageJob
		err := s.store.DB().NewSelect().Model(&job).Where("cid = ?", s.cid).Scan(s.ctx)
		return err == nil && job.Posts >= 1
	}, 10*time.Second, 20*time.Millisecond, "the post was never tried")
	s.chat.Heal(http.MethodPost, "/message")

	posted := s.awaitPosted(1)
	s.Contains(posted[0]["text"], "could not draw that picture")
	s.Equal(1, s.drawer.times(), "the decided failure is posted, the picture not drawn again")
	s.Equal(store.ImageJobFailed, s.job().State)
}

func (s *ImageJobsSuite) TestAPostThatNeverGetsThroughIsGivenUpOnceItsWaitsRanOut() {
	s.chat.Fail(http.MethodPost, "/message")
	worker := s.started(s.posts)

	s.queue(worker, "draw-1")

	job := s.job()
	s.Equal(store.ImageJobFailed, job.State)
	s.Contains(job.Outcome, "not posted")
	s.Equal(3, job.Posts, "one post for each wait")
	s.Equal(1, s.drawer.times(), "never drawn again")
}

func (s *ImageJobsSuite) TestTwoRoutersDrawEachPictureOnce() {
	s.drawer.takes = 50 * time.Millisecond
	one := s.started(s.posts)
	s.started(s.posts)

	for n := 1; n <= 6; n++ {
		s.queue(one, fmt.Sprintf("draw-%d", n))
	}

	jobs := s.jobs()
	s.Len(jobs, 6)
	for _, job := range jobs {
		s.Equal(store.ImageJobDone, job.State)
		s.Equal(1, job.Draws)
	}
	s.Len(s.posted(), 6)
	s.Equal(6, s.drawer.times())
}

func (s *ImageJobsSuite) TestACallRunAgainQueuesNoSecondPicture() {
	worker := s.worker()
	s.queue(worker, "draw-1")
	s.queue(worker, "draw-1")

	s.Len(s.queued(), 1)
}

func (s *ImageJobsSuite) TestTheSameCallIdInAnotherTurnIsAnotherPicture() {
	// Calls parsed from a reply's tags are numbered ask-1, ask-2 again by each harness, and a
	// session reopened after it ended starts a new one under the same session id.
	worker := s.worker()
	s.queueIn(worker, "turn-1", "ask-1", "a yellow watering can")
	s.queueIn(worker, "turn-2", "ask-1", "a yellow watering can")

	s.Len(s.queued(), 2)
}

func (s *ImageJobsSuite) TestTheSameCallAskingForAnotherPictureIsAnotherJob() {
	// Never dropped: a call id reused for another prompt is another picture to draw.
	worker := s.worker()
	s.queueIn(worker, "turn-1", "ask-1", "a yellow watering can")
	s.queueIn(worker, "turn-1", "ask-1", "a red kite")

	s.Len(s.queued(), 2)
}

func (s *ImageJobsSuite) queued() []store.ImageJob {
	var jobs []store.ImageJob
	s.Require().NoError(s.store.DB().NewSelect().Model(&jobs).Where("cid = ?", s.cid).Scan(s.ctx))
	return jobs
}
