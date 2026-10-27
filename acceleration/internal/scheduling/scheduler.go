package scheduling

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// scheduledLease is how long a worker holds a task it claimed, renewed while
// its run is in progress: longer than any delivery wait the worker sits out,
// short enough that a stopped worker's tasks are taken up again within about
// a minute. It is also the longest the worker sleeps between looks, so a task
// another router claimed and dropped is found again.
const scheduledLease = time.Minute

// renewLeaseEvery is how often a worker holding a run extends its claim.
const renewLeaseEvery = 10 * time.Second

// inFlight is how many occurrences one router runs at once.
const inFlight = 16

// occurrenceBudget is the longest one occurrence holds a worker slot, retries
// included. One attempt can wait out a call for the API's scheduledWindow
// (10 min) and then read the agent's answer for scheduledReplyWindow (3 min),
// so 20 min lets a slow first attempt finish, while an occurrence whose
// attempts fail at once still gets every DeliveryRetries wait (12.5 min in
// all). Without it four slow attempts held a slot ~50-90 min, and a router
// has inFlight slots.
const occurrenceBudget = 20 * time.Minute

// errClaimLost cuts a delivery short whose claim was taken from it: the task
// was paused, edited or deleted, or another worker holds it now.
var errClaimLost = errors.New("the task changed under its run")

// errDraining ends a delivery's retries when the worker's context ends: the
// run is left to its lease, as when Close cuts it short.
var errDraining = errors.New("the router is draining")

// errOutOfTime cuts a delivery short that used up occurrenceBudget.
var errOutOfTime = errors.New("the occurrence ran out of time")

// poll is the shortest wait before looking again while occurrences are queued
// or claimed elsewhere, so another router's claims are not chased in a loop.
const poll = time.Second

// deliveryWindow is how late each kind of occurrence may be: a task that runs
// once is still worth delivering a day late, told that it is late; a repeating
// one is only run at its latest due occurrence, the older ones written missed.
const (
	oneShotGrace = 24 * time.Hour
	repeatGrace  = time.Hour
)

// maxConsecutiveFailures is how many occurrences in a row may fail before the
// task gives up, pausing itself as failed.
const maxConsecutiveFailures = 5

// DeliveryRetries are the waits before an occurrence is delivered again after
// a failed attempt. A delivered task survives a router being busy or a call
// being in progress; a task whose retries run out has failed an occurrence.
var DeliveryRetries = []time.Duration{30 * time.Second, 2 * time.Minute, 10 * time.Minute}

// DeliverFunc delivers one claimed occurrence: it returns where the follow-up
// landed (conversation, call, ...) and the session it landed in. An error is
// a failed attempt; the occurrence counts failed when the retries run out.
type DeliverFunc func(ctx context.Context, claimed store.ClaimedScheduledTask) (deliveredTo string, sessionID string, err error)

// Postponed is a delivery held rather than failed: the occurrence is not
// attempted now and is tried again at Until, under the claim that holds it
// still. Quiet hours holding a task to the morning is the one there is. It
// counts against nothing: the run stays running, and when the task is
// claimed again the occurrence's age is counted from the hold's end.
type Postponed struct {
	Until time.Time
}

func (p *Postponed) Error() string {
	return "held until " + p.Until.Format("2006-01-02 15:04 MST")
}

// Missed is an occurrence the delivery found too late to make, such as a
// call that would ring long after the time the user asked for: the run is
// written missed, as one past its grace is, and not tried again.
type Missed struct {
	Err error
}

func (m *Missed) Error() string { return m.Err.Error() }

// Failure is a failed attempt as its run records it. Reason is one of the
// Reason codes, which a device is shown; Err is the whole story, which can
// name numbers, vendors and internals, and goes to the logs only. Final says
// another attempt would only repeat it, or ring the user a second time, so
// it is not retried.
type Failure struct {
	Reason string
	Final  bool
	Err    error
}

func (f *Failure) Error() string { return f.Err.Error() }

// The reasons a failed run gives: short codes a device may show or act on.
const (
	ReasonFailed      = "delivery_failed"
	ReasonOutOfTime   = "timed_out"
	ReasonUnreadable  = "schedule_unreadable"
	ReasonOnACall     = "user_on_call"
	ReasonUnreachable = "channel_unreachable"
	ReasonNoReply     = "no_reply"
	ReasonReplyFailed = "reply_failed"
	ReasonNoNumber    = "no_phone_number"
	ReasonNoLine      = "no_agent_number"
	ReasonNotAnswered = "not_answered"
	ReasonRungAlready = "already_called"
	// ReasonOutsideWindow is a thread whose person last wrote longer ago than the provider
	// takes free text (WhatsApp's 24 h), with no template to send after it.
	ReasonOutsideWindow = "outside_window"
	// ReasonThreadGone is a task's thread that no longer exists, or is no longer its user's.
	ReasonThreadGone = "thread_gone"
	// ReasonConnectionRemoved is a task's thread whose connection, its line, was removed.
	ReasonConnectionRemoved = "connection_removed"
	// ReasonPaused is not a failure: it is on the missed run of a one-shot
	// whose time came while the task was paused.
	ReasonPaused = "task_paused"
)

// SleptThrough is the missed run of the occurrence a paused one-shot slept
// through, for a resume that finds its time gone; nil when there is none. A
// repeating task has none: a pause skips what it sleeps through. Nor has an
// active task, whose due occurrence is the worker's, or a failed one, whose
// occurrence has its failed run; paused after failing, it has no next run.
// Inserted, the occurrence's unique key leaves a run already there (one
// settled by the pause, or one a second resume wrote) as it is.
func SleptThrough(task store.ScheduledTask) *store.ScheduledTaskRun {
	if task.RRule != "" || task.State != store.ScheduledTaskPaused || task.NextRunAt == nil {
		return nil
	}
	finished, reason := time.Now().UTC(), ReasonPaused
	return &store.ScheduledTaskRun{
		TaskID: task.ID, CustomerID: task.CustomerID, ScheduledFor: *task.NextRunAt,
		Trigger: store.ScheduledTaskTriggerSchedule, Prompt: task.Instructions,
		State: store.ScheduledTaskRunMissed, Error: &reason, FinishedAt: &finished,
	}
}

// ReasonOf is the code a failed attempt is recorded under.
func ReasonOf(err error) string {
	if errors.Is(err, errOutOfTime) {
		return ReasonOutOfTime
	}
	var failure *Failure
	if errors.As(err, &failure) {
		return failure.Reason
	}
	return ReasonFailed
}

// final reports whether a failed attempt is not to be tried again.
func final(err error) bool {
	var failure *Failure
	var missed *Missed
	return errors.As(err, &missed) || errors.As(err, &failure) && failure.Final
}

// Scheduler runs the scheduled tasks that come due, on every router: each
// occurrence is claimed by one worker, delivered once, and the task advanced
// under the claim that held it.
type Scheduler struct {
	store   *store.Store
	deliver DeliverFunc
	logger  *slog.Logger

	now        func() time.Time
	lease      time.Duration
	renewEvery time.Duration
	retries    []time.Duration
	budget     time.Duration

	kick    chan struct{}
	slots   chan struct{}
	ctx     context.Context
	cancel  context.CancelFunc
	working sync.WaitGroup
}

// NewScheduler is a Scheduler delivering with deliver. It runs nothing until
// Start.
func NewScheduler(db *store.Store, deliver DeliverFunc, logger *slog.Logger) *Scheduler {
	if logger == nil {
		logger = slog.Default()
	}
	ctx, cancel := context.WithCancel(context.Background())
	return &Scheduler{
		store:      db,
		deliver:    deliver,
		logger:     logger,
		now:        time.Now,
		lease:      scheduledLease,
		renewEvery: renewLeaseEvery,
		retries:    DeliveryRetries,
		budget:     occurrenceBudget,
		kick:       make(chan struct{}, 1),
		slots:      make(chan struct{}, inFlight),
		ctx:        ctx,
		cancel:     cancel,
	}
}

// Start runs the worker until ctx ends or Close: it looks at start, when Kick
// wakes it, at the next task due, and at least once a lease. Once ctx ends it
// claims nothing more, and the deliveries in flight run on until they finish
// or Close cuts them short: a draining router leaves new occurrences to the
// others and finishes its own attempt. A delivery waiting to retry stops
// waiting and is left to its lease: another router takes it up.
func (s *Scheduler) Start(ctx context.Context) {
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
			wait, again := s.runDue(ctx)
			timer.Stop()
			if again {
				timer.Reset(wait)
			}
			select {
			case <-ctx.Done():
				return
			case <-timer.C:
			case <-s.kick:
			}
		}
	}()
}

// Close stops the worker and cancels the deliveries in flight. One cut short
// is not recorded: its lease runs out and a router takes the occurrence up
// again under the same run, so the same command id.
func (s *Scheduler) Close() {
	s.cancel()
	s.working.Wait()
}

// Running is how many occurrences the worker is delivering now.
func (s *Scheduler) Running() int {
	return len(s.slots)
}

// Kick wakes the worker: a task was made or changed somewhere.
func (s *Scheduler) Kick() {
	select {
	case s.kick <- struct{}{}:
	default:
	}
}

// runDue claims as many due tasks as there are free slots and runs each on
// its own goroutine. It returns how long to wait before looking again, with
// again false when only a finished run needs to wake the worker.
func (s *Scheduler) runDue(ctx context.Context) (time.Duration, bool) {
	for {
		if ctx.Err() != nil {
			return 0, false
		}
		free := cap(s.slots) - len(s.slots)
		if free == 0 {
			return 0, false
		}
		claimed, err := s.store.ClaimDueScheduledTasks(ctx, free, s.lease)
		if err != nil {
			if ctx.Err() == nil {
				s.logger.Error("could not claim the scheduled tasks due", "error", err)
			}
			return poll, true
		}
		for _, claim := range claimed {
			s.slots <- struct{}{}
			s.working.Add(1)
			go func() {
				defer s.working.Done()
				defer func() {
					<-s.slots
					s.Kick()
				}()
				s.run(ctx, claim)
			}()
		}
		if len(claimed) < free {
			break
		}
	}
	next, due, err := s.store.NextDueScheduledTask(ctx)
	switch {
	case err != nil:
		if ctx.Err() == nil {
			s.logger.Error("could not find when the next scheduled task is due", "error", err)
		}
		return poll, true
	case !due:
		return s.lease, true
	}
	return min(max(time.Until(next), poll), s.lease), true
}

// run delivers (or settles) one claimed occurrence and records what came of
// it under the claim, so an update or pause that landed meanwhile wins.
func (s *Scheduler) run(draining context.Context, claim store.ClaimedScheduledTask) {
	task := claim.Task
	schedule, err := FromStored(task)
	if err != nil {
		// A stored schedule that no longer reads can never run; say so on this
		// occurrence and pause the task instead of retrying it forever.
		s.logger.Error("a scheduled task's stored schedule does not read",
			"task", task.ID, "run", claim.Run.ID, "customer", task.CustomerID, "error", err)
		s.finish(claim, store.ScheduledTaskOutcome{
			RunState: store.ScheduledTaskRunFailed,
			Error:    ReasonUnreadable,
			State:    store.ScheduledTaskPaused,
		})
		return
	}
	if claim.WorkersLost {
		s.logger.Error("a scheduled task's occurrence failed: it was started this many times and its worker was lost each time",
			"task", task.ID, "run", claim.Run.ID, "customer", task.CustomerID, "attempts", claim.Run.Attempts)
	}
	if claim.AlreadySettled {
		// The occurrence was settled and only the advance was lost: settle the
		// task from the run as it stands, without delivering again.
		s.finish(claim, s.settledOutcome(claim, schedule))
		return
	}
	if missed := s.missed(claim, schedule); missed != nil {
		s.finish(claim, *missed)
		return
	}

	// The delivery runs while the claim is this worker's, and for no longer
	// than the budget.
	owned, lose := context.WithCancelCause(s.ctx)
	defer lose(nil)
	ctx, cancel := context.WithTimeoutCause(owned, s.budget, errOutOfTime)
	defer cancel()
	stopRenewing := s.renewing(owned, lose, claim)
	deliveredTo, sessionID, failure := s.deliverWithRetries(ctx, draining, claim)
	stopRenewing()
	var held *Postponed
	var late *Missed
	switch {
	case failure != nil && errors.As(failure, &held):
		s.postpone(claim, held.Until)
	case failure != nil && errors.As(failure, &late):
		s.logger.Info("a scheduled task was too late to deliver",
			"task", task.ID, "run", claim.Run.ID, "customer", task.CustomerID, "error", failure)
		s.finish(claim, s.missedOutcome(claim, schedule))
	case failure == nil:
		next, state := s.after(schedule, claim.Run.ScheduledFor, false, 0)
		s.finish(claim, store.ScheduledTaskOutcome{
			RunState: store.ScheduledTaskRunDelivered, DeliveredTo: deliveredTo,
			SessionID: sessionID, NextRunAt: next, State: state, ConsecutiveFailures: 0,
		})
	case errors.Is(failure, errDraining):
		// The router is draining and the delivery was waiting to retry: leave
		// the run to its lease, for another router to take up.
		s.logger.Info("a scheduled task's retry was left to its lease: the router is draining",
			"task", task.ID, "run", claim.Run.ID, "customer", task.CustomerID, "error", failure)
		return
	case owned.Err() != nil:
		// Close cut the delivery short, or the claim was lost to a change
		// that decides instead: leave the run to its lease.
		if errors.Is(context.Cause(owned), errClaimLost) {
			s.logger.Info("a scheduled task changed under its run; its delivery was stopped",
				"task", task.ID, "run", claim.Run.ID, "customer", task.CustomerID)
		}
		return
	default:
		if ctx.Err() != nil {
			failure = fmt.Errorf("%w after %s: %w", errOutOfTime, s.budget, failure)
		}
		s.logger.Warn("a scheduled task failed to deliver",
			"task", task.ID, "run", claim.Run.ID, "customer", task.CustomerID, "error", failure)
		failures := task.ConsecutiveFailures + 1
		next, state := s.after(schedule, claim.Run.ScheduledFor, true, failures)
		s.finish(claim, store.ScheduledTaskOutcome{
			RunState: store.ScheduledTaskRunFailed, Error: ReasonOf(failure),
			NextRunAt: next, State: state, ConsecutiveFailures: failures,
		})
	}
}

// missed is the outcome for an occurrence too old to deliver, or nil for one
// still worth delivering. Each skipped occurrence of a repeating task is
// finished as missed in turn, so the latest due one is delivered and the
// history shows the gap. A held occurrence's age is counted from the hold's
// end: held overnight is delivered in the morning, not missed.
func (s *Scheduler) missed(claim store.ClaimedScheduledTask, schedule *Schedule) *store.ScheduledTaskOutcome {
	due := claim.Run.ScheduledFor
	if claim.Run.PostponedUntil != nil && claim.Run.PostponedUntil.After(due) {
		due = *claim.Run.PostponedUntil
	}
	age := s.now().Sub(due)
	task := claim.Task
	if task.RRule == "" && age > oneShotGrace || task.RRule != "" && age > repeatGrace {
		missed := s.missedOutcome(claim, schedule)
		return &missed
	}
	return nil
}

// missedOutcome is a missed occurrence's outcome: a one-shot is over, a
// repeating task moves on to its next occurrence, and the failure streak is
// left as it is.
func (s *Scheduler) missedOutcome(claim store.ClaimedScheduledTask, schedule *Schedule) store.ScheduledTaskOutcome {
	task := claim.Task
	if task.RRule == "" {
		return store.ScheduledTaskOutcome{
			RunState: store.ScheduledTaskRunMissed, State: store.ScheduledTaskCompleted,
			ConsecutiveFailures: task.ConsecutiveFailures,
		}
	}
	next, state := s.after(schedule, claim.Run.ScheduledFor, false, task.ConsecutiveFailures)
	return store.ScheduledTaskOutcome{
		RunState: store.ScheduledTaskRunMissed, NextRunAt: next, State: state,
		ConsecutiveFailures: task.ConsecutiveFailures,
	}
}

// deliverWithRetries delivers once, then again after each wait, until one
// attempt lands or the waits run out. A held delivery is not retried: the
// hold says when it is worth trying again, and that is longer than any wait
// here. Nor is a missed or final one: trying again would only ring the user
// again, or late. Once draining ends, no wait is begun or finished: the
// failure is errDraining and the run is left to its lease.
func (s *Scheduler) deliverWithRetries(ctx, draining context.Context, claim store.ClaimedScheduledTask) (string, string, error) {
	deliveredTo, sessionID, err := s.deliver(ctx, claim)
	for _, wait := range s.retries {
		var held *Postponed
		if err == nil || errors.As(err, &held) || final(err) {
			return deliveredTo, sessionID, err
		}
		timer := time.NewTimer(wait)
		select {
		case <-draining.Done():
			timer.Stop()
			return deliveredTo, sessionID, fmt.Errorf("%w: %w", errDraining, err)
		case <-ctx.Done():
			timer.Stop()
			return deliveredTo, sessionID, err
		case <-timer.C:
		}
		earlier := err
		deliveredTo, sessionID, err = s.deliver(ctx, claim)
		err = rungEarlier(earlier, err)
	}
	return deliveredTo, sessionID, err
}

// rungEarlier is a later attempt's failure, kept under the reason the earlier
// attempt failed under when the later one found the occurrence's call placed
// already: that call is the earlier attempt's, and how it went (nobody
// answered, say) is the run's reason, not that it was placed.
func rungEarlier(earlier, later error) error {
	var failure *Failure
	if !errors.As(later, &failure) || failure.Reason != ReasonRungAlready {
		return later
	}
	return &Failure{Reason: ReasonOf(earlier), Final: failure.Final, Err: later}
}

// postpone holds a claimed task until until, under the claim that delivered
// nothing yet: the run stays running and the lease becomes the hold, so the
// occurrence is taken up again when the hold ends. A claim fenced meanwhile
// is reported and left: the change that fenced it decides instead.
func (s *Scheduler) postpone(claim store.ClaimedScheduledTask, until time.Time) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(s.ctx), 15*time.Second)
	defer cancel()
	held, err := s.store.PostponeScheduledTask(ctx, claim.Task.CustomerID, claim.Task.ID, claim.Run.ID, claim.ClaimID, until)
	log := s.logger.With("task", claim.Task.ID, "run", claim.Run.ID, "customer", claim.Task.CustomerID)
	switch {
	case err != nil:
		log.Error("could not hold a scheduled task", "error", err)
	case !held:
		log.Info("a scheduled task changed under its run; the hold was left alone")
	default:
		log.Info("a scheduled task is held", "until", until)
	}
}

// renewing extends the claim while the run is in progress. A claim no longer
// this worker's to extend is lost: lose cuts the delivery short, so a paused,
// edited or deleted task is not delivered on, nor delivered twice.
func (s *Scheduler) renewing(ctx context.Context, lose context.CancelCauseFunc, claim store.ClaimedScheduledTask) func() {
	done := make(chan struct{})
	go func() {
		ticker := time.NewTicker(s.renewEvery)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ctx.Done():
				return
			case <-ticker.C:
			}
			renew, cancel := context.WithTimeout(ctx, s.renewEvery)
			held, err := s.store.RenewScheduledTaskLease(renew, claim.Task.CustomerID, claim.Task.ID, claim.ClaimID, s.lease)
			cancel()
			if err != nil {
				s.logger.Error("could not renew a scheduled task's lease",
					"task", claim.Task.ID, "error", err)
				continue
			}
			if !held {
				lose(errClaimLost)
				return
			}
		}
	}()
	return func() { close(done) }
}

// settledOutcome is the advance of a task whose run is finished already: the
// run's own outcome, carried on from its occurrence.
func (s *Scheduler) settledOutcome(claim store.ClaimedScheduledTask, schedule *Schedule) store.ScheduledTaskOutcome {
	run := claim.Run
	outcome := store.ScheduledTaskOutcome{
		RunState: run.State, DeliveredTo: run.DeliveredTo, SessionID: run.SessionID,
	}
	failed := run.State == store.ScheduledTaskRunFailed
	switch {
	case failed:
		outcome.ConsecutiveFailures = claim.Task.ConsecutiveFailures + 1
		if run.Error != nil {
			outcome.Error = *run.Error
		}
	case run.State == store.ScheduledTaskRunMissed:
		// A missed occurrence leaves the failure streak as it is, as it did
		// when it was first written missed.
		outcome.ConsecutiveFailures = claim.Task.ConsecutiveFailures
	}
	outcome.NextRunAt, outcome.State = s.after(schedule, run.ScheduledFor, failed, outcome.ConsecutiveFailures)
	return outcome
}

// after is a task's next run and state after the occurrence at scheduledFor:
// completed when the schedule has nothing left, failed when the failures ran
// out or a last occurrence failed, active otherwise.
func (s *Scheduler) after(schedule *Schedule, scheduledFor time.Time, failed bool, consecutiveFailures int) (*time.Time, string) {
	next, ok := schedule.Next(scheduledFor)
	switch {
	case consecutiveFailures >= maxConsecutiveFailures:
		if ok {
			return &next, store.ScheduledTaskFailed
		}
		return nil, store.ScheduledTaskFailed
	case !ok && failed:
		return nil, store.ScheduledTaskFailed
	case !ok:
		return nil, store.ScheduledTaskCompleted
	default:
		return &next, store.ScheduledTaskActive
	}
}

// finish records an occurrence's outcome under the claim that held it. A
// claim an update or pause fenced cannot advance the task, and that is
// reported rather than treated as a failure.
func (s *Scheduler) finish(claim store.ClaimedScheduledTask, outcome store.ScheduledTaskOutcome) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(s.ctx), 15*time.Second)
	defer cancel()
	applied, err := s.store.FinishScheduledTaskRun(ctx, claim.Task.CustomerID, claim.Run.ID, claim.ClaimID, outcome)
	log := s.logger.With("task", claim.Task.ID, "run", claim.Run.ID, "customer", claim.Task.CustomerID)
	switch {
	case err != nil:
		log.Error("could not record a scheduled task run", "error", err)
	case !applied:
		log.Info("a scheduled task changed under its run; the schedule was left alone")
	}
}
