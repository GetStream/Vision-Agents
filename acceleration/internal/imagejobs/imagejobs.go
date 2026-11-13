// Package imagejobs draws the pictures agents are asked for after the turn that asked has
// ended, and posts each into its conversation when it is ready.
//
// A picture takes one to two minutes, longer than a turn should wait. generate_image queues a
// job (Queue) and answers at once; a worker on every router claims the jobs due from Postgres
// (image_jobs), draws each through the image router, under the attribution of the session that
// asked, and posts the picture, or a short failure, as an agent message in the conversation.
//
//	generate_image            Queue: image_jobs row, Kick
//	worker (every router)     ClaimImageJobs: due rows, lease, attempts + 1
//	  picture kept            post it                 (a claim after a crash mid-post)
//	  attempts > maxDraws     post the failure        (claims that never finished)
//	  otherwise               draw, upload, keep it, post it; a refusal or failure posts why
//	  posted                  done or failed          a post that failed: due again, no new draw
//
// A router that stops between a claim and its outcome leaves the row queued under its lease;
// once the lease runs out another router claims it again. A draw already billed is not drawn
// again once its picture was kept, and a message already written is found written rather than
// posted twice (conversation.Service.Post).
package imagejobs

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagerouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Lease is how far a claim pushes a job's next attempt: the longest draw, imagerouter.Deadline,
// and a minute to upload and post the picture. A router that stopped mid-job holds it no
// longer than that.
const Lease = imagerouter.Deadline + time.Minute

// maxDraws is how many claims may draw a job. The second is for a router that stopped in the
// middle of the first; a third would say the draw itself ends routers, and the job posts its
// failure instead.
const maxDraws = 2

// maxClaims bounds the claims of a job whose picture was kept but could not be posted, such as
// while Stream Chat is down: after them it is recorded failed and logged, there being nowhere
// to say so.
const maxClaims = 6

// retryPost is how long a job whose post failed waits before the next claim. A choice: Stream
// Chat failing a write is usually over within a minute.
const retryPost = 30 * time.Second

// look is the longest wait between looks for due jobs, so a job a stopped router held is
// claimed within about a lease and a look. A choice, chatinbound's.
const look = time.Minute

// defaultPoll is the shortest wait between looks while jobs are queued.
const defaultPoll = time.Second

// inFlight is how many jobs one router draws at once. A choice: FAL holds two per process
// (imagegen/fal slots), and the rest of these wait for its slot inside their own deadline.
const inFlight = 16

// pruneEvery is how often the worker deletes the rows past their keep.
const pruneEvery = time.Hour

// recordTimeout bounds writing what came of a claim.
const recordTimeout = 10 * time.Second

// What a job posts. The model is gone by the time the picture is drawn, so these are written
// to the person.
const (
	pictureText  = "Here is the picture."
	failedText   = "I could not draw that picture: something went wrong. Ask me again to try once more."
	filteredText = "I could not draw that picture: the safety filter refused it."
	spentText    = "I could not draw that picture: the spending limit for pictures has been reached."
	shapeText    = "I could not draw that picture in that shape."
	slowText     = "I could not draw that picture: it took too long. Ask me again to try once more."
)

// Poster writes into a conversation outside any turn: conversation.Service.
type Poster interface {
	UploadTo(ctx context.Context, customer string, app int64, cid, uploader string, file sandbox.File) (sandbox.Attachment, error)
	Post(ctx context.Context, customer string, app int64, cid, agent, id string, startedAt time.Time,
		text string, files []sandbox.Attachment) error
}

// Request is a picture an agent was asked to draw, and the conversation it is posted into.
type Request struct {
	// CustomerID, StreamApp, CID and AgentUser are the conversation: who pays, the Stream app
	// it is kept in, its channel and the agent's Stream user.
	CustomerID string
	StreamApp  int64
	CID        string
	AgentUser  string
	// ToolCallID is the call that asked. A call in a channel queues one job, however often it runs.
	ToolCallID string
	// AgentID, CallID, Region, Target, Tags and Attribution are what the draw is routed and
	// billed under, as the turn's own work is.
	AgentID     string
	CallID      string
	Region      string
	Target      string
	Tags        routing.Tags
	Attribution routing.Attribution
	Prompt      string
	AspectRatio string
}

// Options configures a Service. Store and Images are required to queue; Poster to draw.
type Options struct {
	Store  *store.Store
	Images *imagerouter.Router
	// Lease is how far a claim pushes a job's next attempt. Zero is Lease. Tests shorten it.
	Lease time.Duration
	// Poll is the shortest wait between looks while jobs are queued. Zero is defaultPoll.
	Poll time.Duration
	// PostRetry is how long a job whose post failed waits. Zero is retryPost. Tests shorten it.
	PostRetry time.Duration
	Logger    *slog.Logger
}

// Service queues the pictures agents are asked for and, once started, draws and posts them.
type Service struct {
	store     *store.Store
	images    *imagerouter.Router
	lease     time.Duration
	poll      time.Duration
	postRetry time.Duration
	logger    *slog.Logger

	// kick wakes the loop: a job was queued, or a slot freed.
	kick chan struct{}
	// slots holds one token for each job running.
	slots   chan struct{}
	poster  Poster
	pruned  time.Time
	ctx     context.Context
	cancel  context.CancelFunc
	working sync.WaitGroup
}

// New validates the options and returns a Service. It draws nothing until Start.
func New(options Options) (*Service, error) {
	if options.Store == nil || options.Images == nil {
		return nil, stack.Wrap(errors.New("imagejobs: a store and an image router are required"))
	}
	s := &Service{
		store: options.Store, images: options.Images, lease: options.Lease, poll: options.Poll,
		postRetry: options.PostRetry, logger: options.Logger, kick: make(chan struct{}, 1), slots: make(chan struct{}, inFlight),
	}
	if s.lease <= 0 {
		s.lease = Lease
	}
	if s.poll <= 0 {
		s.poll = defaultPoll
	}
	if s.postRetry <= 0 {
		s.postRetry = retryPost
	}
	if s.logger == nil {
		s.logger = slog.Default()
	}
	s.ctx, s.cancel = context.WithCancel(context.Background())
	return s, nil
}

// Queue writes the job for a worker to draw, on this router or another, and wakes this one's.
func (s *Service) Queue(ctx context.Context, request Request) error {
	attribution, err := json.Marshal(request.Attribution)
	if err != nil {
		return stack.Wrap(err)
	}
	job := store.ImageJob{
		ID: uuid.NewString(), CustomerID: request.CustomerID, StreamApp: request.StreamApp, CID: request.CID,
		AgentUser: request.AgentUser, ToolCallID: request.ToolCallID, AgentID: request.AgentID, CallID: request.CallID,
		Region: request.Region, Target: request.Target, Tags: request.Tags, Attribution: attribution,
		Prompt: request.Prompt, AspectRatio: request.AspectRatio,
	}
	if _, err := s.store.QueueImageJob(ctx, &job); err != nil {
		return err
	}
	s.Kick()
	return nil
}

// Start draws the jobs due until ctx ends or Close, posting each into its conversation with
// poster. It looks at start, for the jobs a router left when it stopped, then whenever a job is
// queued or ends, and at least once a look.
//
// ctx is the router's stop signal: a draining router claims no more jobs and the ones queued
// wait for another router. The jobs already running go on; the drain waits for them (Running).
func (s *Service) Start(ctx context.Context, poster Poster) {
	s.poster = poster
	ctx, cancel := context.WithCancel(ctx)
	stop := context.AfterFunc(s.ctx, cancel)
	s.working.Add(1)
	go func() {
		defer s.working.Done()
		defer stop()
		defer cancel()
		timer := time.NewTimer(0)
		defer timer.Stop()
		for {
			wait := s.drawDue(ctx)
			timer.Stop()
			timer.Reset(wait)
			select {
			case <-ctx.Done():
				return
			case <-timer.C:
			case <-s.kick:
			}
		}
	}()
}

// Kick wakes the loop to look for due jobs now.
func (s *Service) Kick() {
	select {
	case s.kick <- struct{}{}:
	default:
	}
}

// Running is how many jobs this router is drawing or posting, for a draining router to wait for.
func (s *Service) Running() int {
	return len(s.slots)
}

// Close stops claiming, cuts the jobs in flight short and waits for them. A job cut short stays
// queued under its lease, for another router once it runs out.
func (s *Service) Close() {
	s.cancel()
	s.working.Wait()
}

// drawDue claims as many due jobs as there are free slots and runs each on its own goroutine,
// and returns how long to wait before looking again.
func (s *Service) drawDue(ctx context.Context) time.Duration {
	s.prune(ctx)
	free := cap(s.slots) - len(s.slots)
	if free == 0 {
		return look
	}
	now := time.Now()
	claimed, err := s.store.ClaimImageJobs(ctx, now, free, now.Add(s.lease))
	if err != nil {
		if ctx.Err() == nil {
			s.logger.Error("could not take the image jobs due", "error", err)
		}
		return s.poll
	}
	for _, job := range claimed {
		s.slots <- struct{}{}
		s.working.Add(1)
		go func() {
			defer s.working.Done()
			defer func() {
				<-s.slots
				s.Kick()
			}()
			s.run(job)
		}()
	}
	next, queued, err := s.store.NextImageJobAt(ctx)
	switch {
	case err != nil:
		if ctx.Err() == nil {
			s.logger.Error("could not find when the next image job is due", "error", err)
		}
		return s.poll
	case !queued:
		return look
	}
	return min(max(time.Until(next), s.poll), look)
}

// prune deletes the rows past their keep, once every pruneEvery.
func (s *Service) prune(ctx context.Context) {
	if time.Since(s.pruned) < pruneEvery {
		return
	}
	s.pruned = time.Now()
	if _, err := s.store.PruneImageJobs(ctx, s.pruned); err != nil && ctx.Err() == nil {
		s.logger.Error("could not delete the image jobs past their keep", "error", err)
	}
}

// run takes one claimed job as far as it goes: drawn, kept, posted.
func (s *Service) run(job store.ImageJob) {
	log := s.logger.With("job", job.ID, "customer", job.CustomerID, "channel", job.CID, "attempts", job.Attempts)
	switch {
	case job.Picture != nil:
		var picture sandbox.Attachment
		if err := json.Unmarshal(job.Picture, &picture); err != nil {
			log.Error("an image job's kept picture is not readable", "error", err)
			s.post(log, job, store.ImageJobFailed, failedText, nil)
			return
		}
		s.post(log, job, store.ImageJobDone, pictureText, []sandbox.Attachment{picture})
	case job.Attempts > maxDraws:
		log.Warn("an image job was claimed more times than it may be drawn; posting its failure")
		s.post(log, job, store.ImageJobFailed, failedText, nil)
	default:
		s.draw(log, job)
	}
}

// draw draws the job's picture under the attribution of the session that asked, keeps it and
// posts it, or posts why it could not be drawn.
func (s *Service) draw(log *slog.Logger, job store.ImageJob) {
	var attribution routing.Attribution
	if err := json.Unmarshal(job.Attribution, &attribution); err != nil {
		log.Error("an image job's attribution is not readable", "error", err)
		s.post(log, job, store.ImageJobFailed, failedText, nil)
		return
	}
	started := time.Now()
	generation, err := s.images.Generate(routing.WithAttribution(s.ctx, attribution), imagerouter.Request{
		CustomerID: job.CustomerID, AgentID: job.AgentID, CallID: job.CallID, Region: job.Region,
		Tags: job.Tags, Target: job.Target,
		Image: imagegen.Request{Prompt: job.Prompt, AspectRatio: job.AspectRatio, N: 1},
	})
	if err != nil && s.ctx.Err() != nil {
		// Close cut it short: the lease runs out and another router draws it.
		log.Info("left an image job for another router", "error", err)
		return
	}
	if err == nil && len(generation.Images) == 0 {
		err = errors.New("imagejobs: the image provider answered with no picture")
	}
	if err != nil {
		text := failureText(err)
		log.Warn("an image job could not be drawn", "provider", generation.Provider, "error", err)
		s.post(log, job, store.ImageJobFailed, text, nil)
		return
	}
	log.Info("drew an image job", "provider", generation.Provider, "model", generation.Model,
		"took_ms", time.Since(started).Milliseconds(),
		"queued_ms", generation.Queued.Milliseconds(), "drawn_ms", generation.Drawn.Milliseconds())

	drawn := generation.Images[0]
	file := sandbox.File{Name: "picture." + strings.TrimPrefix(drawn.MediaType, "image/"), MIME: drawn.MediaType, Data: drawn.Data}
	picture, err := s.poster.UploadTo(s.ctx, job.CustomerID, job.StreamApp, job.CID, job.AgentUser, file)
	if err != nil {
		// The picture is billed and lost: the next claim draws it again, within maxDraws.
		log.Error("could not upload a drawn picture; the job is drawn again", "error", err)
		s.retry(log, job)
		return
	}
	kept, err := json.Marshal(picture)
	if err != nil {
		log.Error("could not keep a drawn picture", "error", err)
		s.post(log, job, store.ImageJobFailed, failedText, nil)
		return
	}
	writing, done := context.WithTimeout(context.WithoutCancel(s.ctx), recordTimeout)
	saved, err := s.store.SaveImageJobPicture(writing, job, kept)
	done()
	switch {
	case err != nil:
		// Posted all the same: the picture exists, and a claim after this one posts the
		// same message id, which is found written.
		log.Error("could not keep a drawn picture before posting it", "error", err)
	case !saved:
		log.Warn("an image job's lease ran out while it was drawn; another claim has it")
		return
	}
	s.post(log, job, store.ImageJobDone, pictureText, []sandbox.Attachment{picture})
}

// post writes the job's one message and records what came of it. A post that failed makes the
// job due again rather than finishing it, so the person is told, unless the claims ran out.
func (s *Service) post(log *slog.Logger, job store.ImageJob, state, text string, files []sandbox.Attachment) {
	err := s.poster.Post(s.ctx, job.CustomerID, job.StreamApp, job.CID, job.AgentUser, job.ID, job.CreatedAt, text, files)
	if err != nil && s.ctx.Err() != nil {
		log.Info("left an image job's message for another router", "error", err)
		return
	}
	if err != nil {
		if job.Attempts < maxClaims {
			log.Warn("could not post an image job's message; it is posted again", "error", err)
			s.retry(log, job)
			return
		}
		log.Error("gave up posting an image job's message", "error", err)
		state, text = store.ImageJobFailed, "not posted: "+err.Error()
	}
	writing, done := context.WithTimeout(context.WithoutCancel(s.ctx), recordTimeout)
	defer done()
	finished, err := s.store.FinishImageJob(writing, job, state, outcomeOf(state, text), time.Now())
	switch {
	case err != nil:
		log.Error("could not record an image job's message", "error", err)
	case !finished:
		log.Warn("an image job's lease ran out before its message was recorded; another claim posts the same message")
	default:
		log.Info("posted an image job's message", "state", state)
	}
}

// retry makes the job due again after the post retry wait.
func (s *Service) retry(log *slog.Logger, job store.ImageJob) {
	writing, done := context.WithTimeout(context.WithoutCancel(s.ctx), recordTimeout)
	defer done()
	if _, err := s.store.RetryImageJob(writing, job, time.Now().Add(s.postRetry)); err != nil {
		log.Error("could not make an image job due again; its lease runs out instead", "error", err)
	}
}

// outcomeOf is what a finished job records: nothing for a posted picture, else what was said.
func outcomeOf(state, text string) string {
	if state == store.ImageJobDone {
		return ""
	}
	return text
}

// failureText is what the person is told about a draw that failed.
func failureText(err error) string {
	for _, money := range []error{
		policy.ErrSuspended, policy.ErrNoCredits, policy.ErrTopUpRequired, policy.ErrSystemLimit,
		policy.ErrBudgetSpent, policy.ErrUserLimit,
	} {
		if errors.Is(err, money) {
			return spentText
		}
	}
	if routing.ReservationRefused(err) {
		return spentText
	}
	switch imagegen.CodeOf(err) {
	case imagegen.ContentFiltered:
		return filteredText
	case imagegen.UnsupportedOption:
		return shapeText
	case imagegen.Timeout:
		return slowText
	}
	return failedText
}
