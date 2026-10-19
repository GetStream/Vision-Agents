package scheduling

import (
	"context"
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
		kick:       make(chan struct{}, 1),
		slots:      make(chan struct{}, inFlight),
		ctx:        ctx,
		cancel:     cancel,
	}
}

// Start runs the worker until Close: it looks at start, when Kick wakes it,
// at the next task due, and at least once a lease.
func (s *Scheduler) Start() {
	s.working.Add(1)
	go func() {
		defer s.working.Done()
		timer := time.NewTimer(0)
		defer timer.Stop()
		for {
			wait, again := s.runDue()
			timer.Stop()
			if again {
				timer.Reset(wait)
			}
			select {
			case <-s.ctx.Done():
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
func (s *Scheduler) runDue() (time.Duration, bool) {
	for {
		if s.ctx.Err() != nil {
			return 0, false
		}
		free := cap(s.slots) - len(s.slots)
		if free == 0 {
			return 0, false
		}
		claimed, err := s.store.ClaimDueScheduledTasks(s.ctx, free, s.lease)
		if err != nil {
			if s.ctx.Err() == nil {
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
				s.run(claim)
			}()
		}
		if len(claimed) < free {
			break
		}
	}
	next, due, err := s.store.NextDueScheduledTask(s.ctx)
	switch {
	case err != nil:
		if s.ctx.Err() == nil {
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
func (s *Scheduler) run(claim store.ClaimedScheduledTask) {
	task := claim.Task
	schedule, err := storedSchedule(task)
	if err != nil {
		// A stored schedule that no longer reads can never run; say so on this
		// occurrence and pause the task instead of retrying it forever.
		s.finish(claim, store.ScheduledTaskOutcome{
			RunState: store.ScheduledTaskRunFailed,
			Error:    "the stored schedule does not read: " + err.Error(),
			State:    store.ScheduledTaskPaused,
		})
		return
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

	stopRenewing := s.renewing(claim)
	deliveredTo, sessionID, failure := s.deliverWithRetries(claim)
	stopRenewing()
	switch {
	case failure == nil:
		next, state := s.after(schedule, claim.Run.ScheduledFor, false, 0)
		s.finish(claim, store.ScheduledTaskOutcome{
			RunState: store.ScheduledTaskRunDelivered, DeliveredTo: deliveredTo,
			SessionID: sessionID, NextRunAt: next, State: state, ConsecutiveFailures: 0,
		})
	case s.ctx.Err() != nil:
		// Close cut the delivery short: leave the run to its lease.
		return
	default:
		failures := task.ConsecutiveFailures + 1
		next, state := s.after(schedule, claim.Run.ScheduledFor, true, failures)
		s.finish(claim, store.ScheduledTaskOutcome{
			RunState: store.ScheduledTaskRunFailed, Error: failure.Error(),
			NextRunAt: next, State: state, ConsecutiveFailures: failures,
		})
	}
}

// missed is the outcome for an occurrence too old to deliver, or nil for one
// still worth delivering. Each skipped occurrence of a repeating task is
// finished as missed in turn, so the latest due one is delivered and the
// history shows the gap.
func (s *Scheduler) missed(claim store.ClaimedScheduledTask, schedule *Schedule) *store.ScheduledTaskOutcome {
	age := s.now().Sub(claim.Run.ScheduledFor)
	task := claim.Task
	switch {
	case task.RRule == "" && age > oneShotGrace:
		return &store.ScheduledTaskOutcome{
			RunState: store.ScheduledTaskRunMissed, State: store.ScheduledTaskCompleted,
			ConsecutiveFailures: task.ConsecutiveFailures,
		}
	case task.RRule != "" && age > repeatGrace:
		next, state := s.after(schedule, claim.Run.ScheduledFor, false, task.ConsecutiveFailures)
		return &store.ScheduledTaskOutcome{
			RunState: store.ScheduledTaskRunMissed, NextRunAt: next, State: state,
			ConsecutiveFailures: task.ConsecutiveFailures,
		}
	}
	return nil
}

// deliverWithRetries delivers once, then again after each wait, until one
// attempt lands or the waits run out.
func (s *Scheduler) deliverWithRetries(claim store.ClaimedScheduledTask) (string, string, error) {
	deliveredTo, sessionID, err := s.deliver(s.ctx, claim)
	for _, wait := range s.retries {
		if err == nil {
			return deliveredTo, sessionID, nil
		}
		timer := time.NewTimer(wait)
		select {
		case <-s.ctx.Done():
			timer.Stop()
			return deliveredTo, sessionID, err
		case <-timer.C:
		}
		deliveredTo, sessionID, err = s.deliver(s.ctx, claim)
	}
	return deliveredTo, sessionID, err
}

// renewing extends the claim while the run is in progress, stopping when the
// claim is no longer this worker's to extend.
func (s *Scheduler) renewing(claim store.ClaimedScheduledTask) func() {
	done := make(chan struct{})
	go func() {
		ticker := time.NewTicker(s.renewEvery)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-s.ctx.Done():
				return
			case <-ticker.C:
			}
			ctx, cancel := context.WithTimeout(s.ctx, s.renewEvery)
			held, err := s.store.RenewScheduledTaskLease(ctx, claim.Task.CustomerID, claim.Task.ID, claim.ClaimID, s.lease)
			cancel()
			if err != nil {
				s.logger.Error("could not renew a scheduled task's lease",
					"task", claim.Task.ID, "error", err)
				continue
			}
			if !held {
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
	if failed {
		outcome.ConsecutiveFailures = claim.Task.ConsecutiveFailures + 1
		if run.Error != nil {
			outcome.Error = *run.Error
		}
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
	case outcome.RunState == store.ScheduledTaskRunFailed:
		log.Warn("a scheduled task failed to deliver", "error", outcome.Error)
	}
}

// storedSchedule rebuilds the schedule a stored task describes.
func storedSchedule(task store.ScheduledTask) (*Schedule, error) {
	var until time.Time
	if task.UntilDate != nil {
		until = *task.UntilDate
	}
	return FromStored(task.StartLocal, task.Timezone, task.RRule, until)
}
