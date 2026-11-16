// Package imagejobs draws the pictures agents are asked for after the turn that asked has
// ended, and posts each into its conversation when it is ready.
//
// A picture takes one to two minutes, longer than a turn should wait. generate_image queues a
// job (Queue) and answers at once; a worker on every router claims the jobs due from Postgres
// (image_jobs), draws each through the image router, under the attribution of the session that
// asked, and posts the picture, or a short failure, as an agent message in the conversation.
//
//	generate_image          Queue: image_jobs row, Kick
//	worker (every router)   claim: draw slots for jobs to draw, post slots for decided ones
//	  to draw               draws >= maxDraws: decide the failure
//	                        else count the draw, draw it
//	                          not taken (a provider never took it in time): give it back, uncounted
//	                          refused or failed: decide why
//	                          drawn: keep it and its message on the row, never drawn again
//	  decided               upload a kept picture (once), then post the decided message
//	                          a transient failure: due again, backing off; nothing is drawn again
//	                          a permanent one: an upload decides the failure, a post gives up
//	                        posted: done (a picture) or failed (why there is none)
//
// Every write after the claim names it (store attempts), so a slow claim and the one that took
// the job after its lease ran out never both write it. A message is decided before it is posted
// and posted under an id its kind names, so a post written again after a crash is found written
// and posts nothing (conversation.Service.Post), and a job never ends done while its person was
// shown a failure.
package imagejobs

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
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

// drawSlots is how many jobs one router draws at once: FAL's own limit per process
// (imagegen/fal slots). A job claimed beyond it would wait for a slot inside its own deadline,
// and time out undrawn while another router's slots sat idle.
const drawSlots = 2

// postSlots is how many decided jobs one router uploads and posts at once. A choice: each is a
// couple of Stream Chat writes.
const postSlots = 16

// maxDraws is how many draws a provider took may be started for a job. The second is for a
// router that stopped in the middle of the first; a third would say the draw itself ends
// routers, and the job posts its failure instead.
const maxDraws = 2

// untakenWait is how long a draw no provider took in time, such as one that waited out its
// deadline for a slot the endpoint's own requests held, waits before it is tried again, and
// untakenFor how long after it was asked such a job keeps being tried before it posts that it
// took too long.
const (
	untakenWait = 10 * time.Second
	untakenFor  = 30 * time.Minute
)

// defaultPostBackoff are the waits after each post that failed for a reason a later one may get
// past, over about two and a half hours, so an outage of Stream Chat that long loses no picture.
// After them the job is recorded failed and logged, there being nowhere to say so.
var defaultPostBackoff = []time.Duration{
	30 * time.Second, time.Minute, 2 * time.Minute, 5 * time.Minute, 10 * time.Minute,
	15 * time.Minute, 30 * time.Minute, 30 * time.Minute, 30 * time.Minute, 30 * time.Minute,
}

// maxClaims bounds the claims of one job, each a router that took it and stopped before writing
// what came of it: a job that keeps ending routers is given up and logged.
const maxClaims = 20

// look is the longest wait between looks for due jobs, so a job a stopped router held is
// claimed within about a lease and a look. A choice, chatinbound's.
const look = time.Minute

// defaultPoll is the shortest wait between looks while jobs are queued.
const defaultPoll = time.Second

// pruneEvery is how often the worker deletes the rows past their keep.
const pruneEvery = time.Hour

// recordTimeout bounds writing what came of a step; keepFor bounds trying to keep a drawn
// picture, which is worth waiting for: without it a later claim cannot know it was drawn.
const (
	recordTimeout = 10 * time.Second
	keepFor       = 30 * time.Second
)

// What a job posts. The model is gone by the time the picture is drawn, so these are written
// to the person.
const (
	pictureText    = "Here is the picture."
	failedText     = "I could not draw that picture: something went wrong. Ask me again to try once more."
	filteredText   = "I could not draw that picture: the safety filter refused it."
	spentText      = "I could not draw that picture: the spending limit for pictures has been reached."
	shapeText      = "I could not draw that picture in that shape."
	slowText       = "I could not draw that picture: it took too long. Ask me again to try once more."
	unuploadedText = "I drew the picture, but it could not be posted here."
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
	// SessionID, TurnID and ToolCallID are the call that asked. One request queues one job,
	// however often its call runs in its turn; the same call id in another turn, or asking for
	// another picture, is another job.
	SessionID  string
	TurnID     string
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

// Options configures a Service. Store and Images are required.
type Options struct {
	Store  *store.Store
	Images *imagerouter.Router
	// Lease is how far a claim pushes a job's next attempt. Zero is Lease. Tests shorten it.
	Lease time.Duration
	// Poll is the shortest wait between looks while jobs are queued. Zero is defaultPoll.
	Poll time.Duration
	// PostBackoff are the waits after each failed post. Nil is defaultPostBackoff. Tests
	// shorten them.
	PostBackoff []time.Duration
	// UntakenWait is how long a draw no provider took waits. Zero is untakenWait. Tests shorten it.
	UntakenWait time.Duration
	Logger      *slog.Logger
}

// Service queues the pictures agents are asked for and, once started, draws and posts them.
type Service struct {
	store       *store.Store
	images      *imagerouter.Router
	lease       time.Duration
	poll        time.Duration
	postBackoff []time.Duration
	untakenWait time.Duration
	logger      *slog.Logger

	// kick wakes the loop: a job was queued, or a slot freed.
	kick chan struct{}
	// drawing and posting hold one token for each job running in that kind of slot.
	drawing chan struct{}
	posting chan struct{}
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
		postBackoff: options.PostBackoff, untakenWait: options.UntakenWait, logger: options.Logger, kick: make(chan struct{}, 1),
		drawing: make(chan struct{}, drawSlots), posting: make(chan struct{}, postSlots),
	}
	if s.lease <= 0 {
		s.lease = Lease
	}
	if s.poll <= 0 {
		s.poll = defaultPoll
	}
	if s.untakenWait <= 0 {
		s.untakenWait = untakenWait
	}
	if s.postBackoff == nil {
		s.postBackoff = defaultPostBackoff
	}
	if s.logger == nil {
		s.logger = slog.Default()
	}
	s.ctx, s.cancel = context.WithCancel(context.Background())
	return s, nil
}

// Queue writes the job for a worker to draw, on this router or another, wakes this one's, and
// reports whether it queued one. A request queued already, the same call of the same turn
// asking for the same picture, is that call run again, and queues nothing more.
func (s *Service) Queue(ctx context.Context, request Request) (bool, error) {
	attribution, err := json.Marshal(request.Attribution)
	if err != nil {
		return false, stack.Wrap(err)
	}
	job := store.ImageJob{
		ID: uuid.NewString(), CustomerID: request.CustomerID, StreamApp: request.StreamApp, CID: request.CID,
		AgentUser: request.AgentUser, SessionID: request.SessionID, TurnID: request.TurnID, ToolCallID: request.ToolCallID,
		RequestKey: requestKey(request),
		AgentID:    request.AgentID, CallID: request.CallID, Region: request.Region, Target: request.Target,
		Tags: request.Tags, Attribution: attribution, Prompt: request.Prompt, AspectRatio: request.AspectRatio,
	}
	queued, err := s.store.QueueImageJob(ctx, &job)
	if err != nil {
		return false, err
	}
	s.Kick()
	return queued, nil
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
			wait := s.runDue(ctx)
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
	return len(s.drawing) + len(s.posting)
}

// Close stops claiming, cuts the jobs in flight short and waits for them. A job cut short is
// given back at once, its cut draw uncounted, for another router to take it from where it was.
func (s *Service) Close() {
	s.cancel()
	s.working.Wait()
}

// runDue claims as many jobs as there are free slots of each kind and runs each on its own
// goroutine, and returns how long to wait before looking again.
func (s *Service) runDue(ctx context.Context) time.Duration {
	s.prune(ctx)
	for _, kind := range []struct {
		slots   chan struct{}
		decided bool
	}{{s.drawing, false}, {s.posting, true}} {
		free := cap(kind.slots) - len(kind.slots)
		if free == 0 {
			continue
		}
		now := time.Now()
		claimed, err := s.store.ClaimImageJobs(ctx, now, free, now.Add(s.lease), kind.decided)
		if err != nil {
			if ctx.Err() == nil {
				s.logger.Error("could not take the image jobs due", "error", err)
			}
			return s.poll
		}
		for _, job := range claimed {
			kind.slots <- struct{}{}
			s.working.Add(1)
			go func() {
				defer s.working.Done()
				defer func() {
					<-kind.slots
					s.Kick()
				}()
				s.run(job)
			}()
		}
	}
	// Only a kind with a free slot is worth waking for: a job due behind full slots waits for
	// one to free, which kicks the loop, rather than having it ask again every poll.
	wait := look
	for _, kind := range []struct {
		slots   chan struct{}
		decided bool
	}{{s.drawing, false}, {s.posting, true}} {
		if len(kind.slots) == cap(kind.slots) {
			continue
		}
		next, queued, err := s.store.NextImageJobAt(ctx, kind.decided)
		switch {
		case err != nil:
			if ctx.Err() == nil {
				s.logger.Error("could not find when the next image job is due", "error", err)
			}
			return s.poll
		case queued:
			wait = min(wait, max(time.Until(next), s.poll))
		}
	}
	return wait
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

// run takes one claimed job as far as it goes.
func (s *Service) run(job store.ImageJob) {
	log := s.logger.With("job", job.ID, "customer", job.CustomerID, "channel", job.CID, "attempts", job.Attempts)
	switch {
	case job.Attempts-job.Released > maxClaims:
		log.Error("gave up an image job claimed more times than any job should be")
		s.finish(log, job, store.ImageJobFailed, "given up: claimed too often")
	case job.Message == "":
		s.draw(log, job)
	default:
		s.deliver(log, job)
	}
}

// draw draws the job's picture under the attribution of the session that asked and keeps it,
// or decides why there is none, then delivers what was decided.
func (s *Service) draw(log *slog.Logger, job store.ImageJob) {
	if job.Draws >= maxDraws {
		log.Warn("an image job was drawn as often as it may be without finishing; posting its failure")
		s.decideFailure(log, job, failedText)
		return
	}
	var attribution routing.Attribution
	if err := json.Unmarshal(job.Attribution, &attribution); err != nil {
		log.Error("an image job's attribution is not readable", "error", err)
		s.decideFailure(log, job, failedText)
		return
	}
	if !s.write(log, "start an image job's draw", func(ctx context.Context) (bool, error) {
		return s.store.StartImageJobDraw(ctx, job)
	}) {
		return
	}
	job.Draws++

	started := time.Now()
	generation, err := s.images.Generate(routing.WithAttribution(s.ctx, attribution), imagerouter.Request{
		CustomerID: job.CustomerID, AgentID: job.AgentID, CallID: job.CallID, Region: job.Region,
		Tags: job.Tags, Target: job.Target, UsageID: usageID(job),
		Image: imagegen.Request{Prompt: job.Prompt, AspectRatio: job.AspectRatio, N: 1},
	})
	switch {
	case err != nil && s.ctx.Err() != nil:
		// Close cut it short: a draw cut is not billed, so it is not counted either. One that
		// finished as Close came is kept below, like any other.
		log.Info("gave an image job back as the router stops", "error", err)
		s.release(log, job, true, time.Now())
		return
	case err != nil && untaken(err) && time.Since(job.CreatedAt) < untakenFor:
		log.Info("no provider took an image job in time; it is tried again", "error", err)
		s.release(log, job, true, time.Now().Add(s.untakenWait))
		return
	case err != nil:
		log.Warn("an image job could not be drawn", "provider", generation.Provider, "error", err)
		s.decideFailure(log, job, failureText(err))
		return
	case len(generation.Images) == 0:
		log.Error("the image provider answered with no picture")
		s.decideFailure(log, job, failedText)
		return
	}
	log.Info("drew an image job", "provider", generation.Provider, "model", generation.Model,
		"took_ms", time.Since(started).Milliseconds(),
		"queued_ms", generation.Queued.Milliseconds(), "drawn_ms", generation.Drawn.Milliseconds())

	drawn := generation.Images[0]
	job.Drawn, job.DrawnType, job.Message, job.Failed = drawn.Data, drawn.MediaType, pictureText, false
	if !s.keep(log, job) {
		// Not kept, the picture is posted all the same: a later claim draws it again, and posts
		// under the same id, which finds this post written.
		log.Error("could not keep a drawn picture; it is posted without being kept")
	}
	s.deliver(log, job)
}

// keep writes the drawn picture and its message on the row, trying for keepFor: once it is
// written the job is never drawn again.
func (s *Service) keep(log *slog.Logger, job store.ImageJob) bool {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(s.ctx), keepFor)
	defer cancel()
	for wait := 100 * time.Millisecond; ; wait = min(2*wait, 5*time.Second) {
		kept, err := s.store.KeepImageJobDrawing(ctx, job, job.Drawn, job.DrawnType, job.Message)
		switch {
		case err == nil && !kept:
			log.Warn("an image job's lease ran out while it was drawn; it is kept by no claim")
			return false
		case err == nil:
			return true
		}
		log.Warn("could not keep a drawn picture yet", "error", err)
		select {
		case <-ctx.Done():
			return false
		case <-time.After(wait):
		}
	}
}

// decideFailure decides the job posts message, why there is no picture, and delivers it.
func (s *Service) decideFailure(log *slog.Logger, job store.ImageJob, message string) {
	if !s.write(log, "decide an image job's failure", func(ctx context.Context) (bool, error) {
		return s.store.DecideImageJobFailure(ctx, job, message)
	}) {
		return
	}
	job.Message, job.Failed, job.Drawn, job.Picture = message, true, nil, nil
	s.deliver(log, job)
}

// deliver uploads the job's picture if it is not yet, then posts the decided message.
func (s *Service) deliver(log *slog.Logger, job store.ImageJob) {
	var files []sandbox.Attachment
	if !job.Failed {
		picture, ok := s.upload(log, job)
		if !ok {
			return
		}
		if picture == nil {
			s.decideFailure(log, job, unuploadedText)
			return
		}
		files = []sandbox.Attachment{*picture}
	}

	err := s.poster.Post(s.ctx, job.CustomerID, job.StreamApp, job.CID, job.AgentUser, messageID(job), job.CreatedAt, job.Message, files)
	switch {
	case err == nil:
		state, outcome := store.ImageJobDone, ""
		if job.Failed {
			state, outcome = store.ImageJobFailed, job.Message
		}
		s.finish(log, job, state, outcome)
	case s.ctx.Err() != nil:
		log.Info("gave an image job's post back as the router stops", "error", err)
		s.release(log, job, false, time.Now())
	case permanent(err):
		log.Error("Stream Chat refused an image job's message; it is not posted", "error", err)
		s.finish(log, job, store.ImageJobFailed, "not posted: "+err.Error())
	default:
		s.retryPost(log, job, err)
	}
}

// upload is the job's picture uploaded: kept from an earlier claim, or uploaded now and kept.
// ok is false when the job was let go of for later; a nil picture with ok is an upload Stream
// Chat refused for good, which is not tried again.
func (s *Service) upload(log *slog.Logger, job store.ImageJob) (_ *sandbox.Attachment, ok bool) {
	if job.Picture != nil {
		var picture sandbox.Attachment
		if err := json.Unmarshal(job.Picture, &picture); err != nil {
			log.Error("an image job's kept picture is not readable", "error", err)
			return nil, true
		}
		return &picture, true
	}
	if job.Drawn == nil {
		drawn, err := s.store.ImageJobDrawn(s.ctx, job.ID)
		switch {
		case err != nil && s.ctx.Err() != nil:
			s.release(log, job, false, time.Now())
			return nil, false
		case err != nil:
			s.retryPost(log, job, err)
			return nil, false
		case drawn == nil:
			log.Error("an image job decided on a picture keeps none")
			return nil, true
		}
		job.Drawn = drawn
	}
	file := sandbox.File{Name: "picture." + strings.TrimPrefix(job.DrawnType, "image/"), MIME: job.DrawnType, Data: job.Drawn}
	picture, err := s.poster.UploadTo(s.ctx, job.CustomerID, job.StreamApp, job.CID, job.AgentUser, file)
	switch {
	case err == nil:
	case s.ctx.Err() != nil:
		log.Info("gave an image job's upload back as the router stops", "error", err)
		s.release(log, job, false, time.Now())
		return nil, false
	case permanent(err):
		log.Error("Stream Chat refused a drawn picture for good", "error", err)
		return nil, true
	default:
		s.retryPost(log, job, err)
		return nil, false
	}
	kept, err := json.Marshal(picture)
	if err == nil {
		s.write(log, "keep an image job's uploaded picture", func(ctx context.Context) (bool, error) {
			return s.store.SaveImageJobPicture(ctx, job, kept)
		})
	}
	// Posted even when not kept: the drawn picture is still on the row, so a later claim uploads
	// it again rather than drawing it.
	return &picture, true
}

// retryPost makes the job due again after the next wait, or gives it up once they ran out.
func (s *Service) retryPost(log *slog.Logger, job store.ImageJob, failure error) {
	if job.Posts >= len(s.postBackoff) {
		log.Error("gave up posting an image job's message", "posts", job.Posts, "error", failure)
		s.finish(log, job, store.ImageJobFailed, "not posted: "+failure.Error())
		return
	}
	next := time.Now().Add(s.postBackoff[job.Posts])
	log.Warn("could not post an image job's message; it is tried again", "posts", job.Posts, "next", next, "error", failure)
	s.write(log, "make an image job's post due again", func(ctx context.Context) (bool, error) {
		return s.store.RetryImageJobPost(ctx, job, next)
	})
}

// release gives the job back for another claim at next, uncounting the draw it started when
// undraw.
func (s *Service) release(log *slog.Logger, job store.ImageJob, undraw bool, next time.Time) {
	s.write(log, "give an image job back", func(ctx context.Context) (bool, error) {
		return s.store.ReleaseImageJob(ctx, job, undraw, next)
	})
}

// finish records that the claim is done with the job.
func (s *Service) finish(log *slog.Logger, job store.ImageJob, state, outcome string) {
	if s.write(log, "record an image job's outcome", func(ctx context.Context) (bool, error) {
		return s.store.FinishImageJob(ctx, job, state, outcome, time.Now())
	}) {
		log.Info("finished an image job", "state", state)
	}
}

// write runs one write of the claim's, even while Close waits, and reports whether the claim
// still held the job. A write that failed, or a claim that lost the job, is logged.
func (s *Service) write(log *slog.Logger, doing string, write func(context.Context) (bool, error)) bool {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(s.ctx), recordTimeout)
	defer cancel()
	held, err := write(ctx)
	switch {
	case err != nil:
		log.Error("could not "+doing, "error", err)
	case !held:
		log.Warn("could not " + doing + ": the claim's lease ran out and another claim has the job")
	}
	return err == nil && held
}

// messageID is the id of the message the job posts: its kind's own, so a failure and a picture
// are never taken for one another when a post is found written.
func messageID(job store.ImageJob) string {
	if job.Failed {
		return job.ID + "-failed"
	}
	return job.ID
}

// untaken reports whether a draw never left the router: it waited out its deadline for a
// provider's slot (imagegen.ErrNotStarted), so nothing was sent and nothing billed. A draw that
// was sent and then timed out is not one: the vendor may have drawn, and billed, it.
func untaken(err error) bool {
	return errors.Is(err, imagegen.ErrNotStarted)
}

// usageID is the usage id of the job's draws: one id for the job, so a picture drawn again
// after a router stopped between drawing it and keeping it settles on the same row, billed once.
func usageID(job store.ImageJob) uuid.UUID {
	return routing.UsageID("image_job", job.ID)
}

// requestKey is the request a job is queued once for: its session, turn and call, and the
// picture asked for. Each part goes after its length, so no two requests share a key however
// their parts split.
func requestKey(request Request) string {
	digest := sha256.New()
	for _, part := range []string{request.SessionID, request.TurnID, request.ToolCallID, request.Prompt, request.AspectRatio} {
		_, _ = fmt.Fprintf(digest, "%d:%s", len(part), part)
	}
	return hex.EncodeToString(digest.Sum(nil))
}

// permanent reports whether Stream Chat refused a write a later attempt would be refused too:
// a 4xx other than 408 and 429, such as a channel deleted or frozen, or a file it will not keep.
// No answer, a 408 (its RequestTimeoutError), a 5xx or a 429 may pass later (RFC 9110 sections
// 15.5.9 and 15.6, RFC 6585 section 4). So may a 400 saying the message is already written,
// which the post gets when the message it is writing again was written and reading it back
// failed: the next try reads it back.
func permanent(err error) bool {
	var refused *getstream.StreamError
	if !errors.As(err, &refused) || refused.StatusCode < http.StatusBadRequest || refused.StatusCode >= http.StatusInternalServerError {
		return false
	}
	switch refused.StatusCode {
	case http.StatusRequestTimeout, http.StatusTooManyRequests:
		return false
	}
	return !alreadyWritten(refused)
}

// alreadyWritten is Stream Chat refusing a message because it already holds one with its id:
// HTTP 400, code 4 and «a message with ID <id> already exists», as channelbridge reads it.
func alreadyWritten(refused *getstream.StreamError) bool {
	return refused.StatusCode == http.StatusBadRequest && refused.Code == 4 &&
		strings.Contains(refused.Message, "a message with ID ") && strings.Contains(refused.Message, " already exists")
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
