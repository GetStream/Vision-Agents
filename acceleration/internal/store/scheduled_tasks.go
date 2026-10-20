package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ErrNoScheduledTask says a customer holds no task by that id.
var ErrNoScheduledTask = errors.New("store: no such scheduled task")

// ErrScheduledTaskIDTaken says a create named a task id the customer already holds.
var ErrScheduledTaskIDTaken = errors.New("store: a scheduled task with that id already exists")

// ErrTooManyScheduledTasks says a create or resume would put a user over their
// active-task limit on the task's agent config.
var ErrTooManyScheduledTasks = errors.New("store: too many scheduled tasks")

// The states a scheduled task is in.
const (
	ScheduledTaskActive    = "active"
	ScheduledTaskPaused    = "paused"
	ScheduledTaskCompleted = "completed"
	ScheduledTaskFailed    = "failed"
)

// The states a scheduled task run is in, and the channels a run is delivered to.
const (
	ScheduledTaskRunRunning   = "running"
	ScheduledTaskRunDelivered = "delivered"
	ScheduledTaskRunFailed    = "failed"
	ScheduledTaskRunMissed    = "missed"

	ScheduledTaskTriggerSchedule = "schedule"
	ScheduledTaskTriggerManual   = "manual"

	ScheduledTaskDeliveredConversation = "conversation"
	ScheduledTaskDeliveredCall         = "call"
	ScheduledTaskDeliveredPhone        = "phone"
)

// scheduledTaskStaleRun is how long a run may sit in running before another
// worker takes it over: the worker that claimed it renews the task's lease as
// it goes, so a run this old in running belongs to one that died.
const scheduledTaskStaleRun = 10 * time.Minute

// maxScheduledTaskRunAttempts is how many times one occurrence is claimed
// before it is failed rather than taken over again: the first claim, the one
// taking it up after a quiet-hours hold, and one after a lost worker. Each
// takeover follows a worker that died mid-delivery, so a third one says the
// delivery is what stops them, and taking it up again would stop another
// router every scheduledTaskStaleRun until the occurrence's grace ran out.
const maxScheduledTaskRunAttempts = 3

// ScheduledTask is something the agent agreed to do for one end user at a
// wall-clock time in that user's zone, once or on a repeat.
type ScheduledTask struct {
	bun.BaseModel `bun:"table:scheduled_tasks,alias:st"`

	ID         string `bun:"id,pk,notnull"`
	CustomerID string `bun:"customer_id,pk,notnull"`
	ConfigID   string `bun:"config_id,notnull"`
	UserID     string `bun:"user_id,notnull"`
	CallerKind string `bun:"caller_kind,notnull"`
	// OriginSessionID is the session the task was created in, SessionID the one
	// it follows up in, ConversationID that session's Chat conversation.
	OriginSessionID string `bun:"origin_session_id,notnull"`
	SessionID       string `bun:"session_id,notnull"`
	ConversationID  string `bun:"conversation_id,notnull"`
	Title           string `bun:"title,notnull"`
	Instructions    string `bun:"instructions,notnull"`
	// StartLocal is the first occurrence as the user's clock reads it, held in
	// UTC so its fields are the wall clock's; Timezone says which clock.
	StartLocal time.Time  `bun:"start_local,notnull"`
	Timezone   string     `bun:"timezone,notnull"`
	RRule      string     `bun:"rrule,nullzero"`
	UntilDate  *time.Time `bun:"until_date"`
	// NextRunAt is the next occurrence as an instant. Nil when there is none.
	NextRunAt *time.Time `bun:"next_run_at"`
	State     string     `bun:"state,notnull"`
	Channels  []string   `bun:"channels,type:jsonb,notnull"`
	// QuietHours is the span of the user's local day an occurrence is held
	// through rather than delivered into. Nil is no hold.
	QuietHours *ScheduledTaskQuietHours `bun:"quiet_hours,type:jsonb"`
	Custom     map[string]any           `bun:"custom,type:jsonb,nullzero"`
	LeaseUntil *time.Time               `bun:"lease_until"`
	ClaimID    string                   `bun:"claim_id,notnull"`
	// ConsecutiveFailures is how many occurrences in a row failed to deliver.
	ConsecutiveFailures int       `bun:"consecutive_failures,notnull"`
	CreatedBy           string    `bun:"created_by,notnull"`
	CreatedAt           time.Time `bun:"created_at,notnull"`
	UpdatedAt           time.Time `bun:"updated_at,notnull"`
}

// ScheduledTaskQuietHours is the span of the user's local day an occurrence
// is held through rather than delivered into, as the clock reads it in the
// task's zone: "22:00" to "07:00" holds the night.
type ScheduledTaskQuietHours struct {
	Start string `json:"start"`
	End   string `json:"end"`
}

// ScheduledTaskRun is one occurrence of a task that came due (or was run by
// hand), and what became of it.
type ScheduledTaskRun struct {
	bun.BaseModel `bun:"table:scheduled_task_runs,alias:str"`

	ID           string     `bun:"id,pk"`
	TaskID       string     `bun:"task_id,notnull"`
	CustomerID   string     `bun:"customer_id,notnull"`
	ScheduledFor time.Time  `bun:"scheduled_for,notnull"`
	Trigger      string     `bun:"trigger,notnull"`
	Prompt       string     `bun:"prompt,notnull"`
	State        string     `bun:"state,notnull"`
	DeliveredTo  string     `bun:"delivered_to,notnull"`
	SessionID    string     `bun:"session_id,notnull"`
	Attempts     int        `bun:"attempts,notnull"`
	Error        *string    `bun:"error"`
	StartedAt    *time.Time `bun:"started_at"`
	FinishedAt   *time.Time `bun:"finished_at"`
	// PostponedUntil is when a hold on this run ends (quiet hours), recorded
	// when the worker holds it. The next claim counts the occurrence's age
	// from here, so a held occurrence is delivered, not missed.
	PostponedUntil *time.Time `bun:"postponed_until"`
	CreatedAt      time.Time  `bun:"created_at,notnull"`
}

// ScheduledTaskPosition is where a page of tasks or runs ends.
type ScheduledTaskPosition struct {
	CreatedAt time.Time `json:"c"`
	ID        string    `json:"id"`
}

const (
	defaultScheduledTaskLimit = 25
	maxScheduledTaskLimit     = 200
)

// ScheduledTaskLimit is the page size a task or run list uses.
func ScheduledTaskLimit(asked int) int {
	return clampLimit(asked, defaultScheduledTaskLimit, maxScheduledTaskLimit)
}

// ScheduledTaskScope narrows a list to one user's tasks as one kind of caller
// may reach them: Kinds are the stored caller kinds in reach (a verified user
// reaches their own kind and the server's). An empty UserID is every user's.
type ScheduledTaskScope struct {
	UserID string
	Kinds  []string
}

// ScheduledTaskFilter narrows a task list.
type ScheduledTaskFilter struct {
	Scope     ScheduledTaskScope
	State     string
	ConfigID  string
	SessionID string
}

// CreateScheduledTask stores a new task, refusing one over the user's active
// limit on the task's config or under an id already taken. It fills in the id when none is named.
func (s *Store) CreateScheduledTask(ctx context.Context, task *ScheduledTask, maxActive int) error {
	if task.CustomerID == "" || task.UserID == "" || task.Title == "" {
		return stack.Wrap(errors.New("store: a scheduled task needs a customer, a user and a title"))
	}
	if task.ID == "" {
		task.ID = newID()
	}
	if task.State == "" {
		task.State = ScheduledTaskActive
	}
	if task.CreatedBy == "" {
		task.CreatedBy = "api"
	}
	if len(task.Channels) == 0 {
		task.Channels = []string{ScheduledTaskDeliveredConversation}
	}
	now := time.Now().UTC()
	task.CreatedAt = now
	task.UpdatedAt = now

	return stack.Wrap(s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if maxActive > 0 {
			count, err := tx.NewSelect().Model((*ScheduledTask)(nil)).
				Where("customer_id = ?", task.CustomerID).
				Where("user_id = ?", task.UserID).
				Where("config_id = ?", task.ConfigID).
				Where("state = ?", ScheduledTaskActive).
				Count(ctx)
			if err != nil {
				return fmt.Errorf("store: count scheduled tasks: %w", err)
			}
			if count >= maxActive {
				return ErrTooManyScheduledTasks
			}
		}
		if _, err := tx.NewInsert().Model(task).Exec(ctx); err != nil {
			if constraint(err) == "scheduled_tasks_pkey" {
				return ErrScheduledTaskIDTaken
			}
			return fmt.Errorf("store: create scheduled task: %w", err)
		}
		return nil
	}))
}

// ScheduledTask returns one of a customer's tasks by id.
func (s *Store) ScheduledTask(ctx context.Context, customerID, id string) (ScheduledTask, error) {
	if customerID == "" || id == "" {
		return ScheduledTask{}, stack.Wrap(errors.New("store: a customer and a task id are required"))
	}
	var task ScheduledTask
	err := s.db.NewSelect().Model(&task).
		Where("customer_id = ?", customerID).
		Where("id = ?", id).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ScheduledTask{}, stack.Wrap(fmt.Errorf("%w: %s", ErrNoScheduledTask, id))
	}
	if err != nil {
		return ScheduledTask{}, stack.Wrap(fmt.Errorf("store: scheduled task: %w", err))
	}
	return task, nil
}

// ScheduledTasks returns a page of a customer's tasks, newest first.
func (s *Store) ScheduledTasks(ctx context.Context, customerID string, filter ScheduledTaskFilter, limit int, after *ScheduledTaskPosition) ([]ScheduledTask, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}
	var tasks []ScheduledTask
	query := s.db.NewSelect().Model(&tasks).Where("customer_id = ?", customerID)
	if filter.Scope.UserID != "" {
		query = query.Where("user_id = ?", filter.Scope.UserID)
		if len(filter.Scope.Kinds) > 0 {
			query = query.Where("caller_kind IN (?)", bun.In(filter.Scope.Kinds))
		}
	}
	if filter.State != "" {
		query = query.Where("state = ?", filter.State)
	}
	if filter.ConfigID != "" {
		query = query.Where("config_id = ?", filter.ConfigID)
	}
	if filter.SessionID != "" {
		query = query.Where("session_id = ?", filter.SessionID)
	}
	if after != nil {
		query = query.Where("(created_at, id) < (?, ?)", after.CreatedAt, after.ID)
	}
	err := query.Order("created_at DESC", "id DESC").Limit(ScheduledTaskLimit(limit) + 1).Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: scheduled tasks: %w", err))
	}
	return tasks, nil
}

// SaveScheduledTask replaces a task's editable fields and clears any claim, so
// a worker finishing an older claim cannot advance what this save changed.
// An occurrence still running (held by quiet hours, or cut short when its
// worker lost the claim) is written missed: the change decides what happens
// next, and the occurrence is neither left running for good nor taken up
// later with the instructions it had before the change. An active one-shot
// whose occurrence that was is due again at the save, so it is delivered as
// it now reads (and held again if quiet hours still hold); a repeating task
// goes on to its next occurrence. The caller reads the task, changes it, and
// saves it back.
func (s *Store) SaveScheduledTask(ctx context.Context, task *ScheduledTask) error {
	if task.CustomerID == "" || task.ID == "" {
		return stack.Wrap(errors.New("store: a customer and a task id are required"))
	}
	task.UpdatedAt = time.Now().UTC()
	task.ClaimID = ""
	task.LeaseUntil = nil
	if len(task.Channels) == 0 {
		task.Channels = []string{ScheduledTaskDeliveredConversation}
	}
	return stack.Wrap(s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		// A run made by hand is left to the delivery that made it, which
		// always records its outcome.
		var settled []time.Time
		if err := tx.NewUpdate().Model((*ScheduledTaskRun)(nil)).
			Set("state = ?", ScheduledTaskRunMissed).
			Set("finished_at = now()").
			Where("customer_id = ?", task.CustomerID).
			Where("task_id = ?", task.ID).
			Where("trigger = ?", ScheduledTaskTriggerSchedule).
			Where("state = ?", ScheduledTaskRunRunning).
			Returning("scheduled_for").
			Scan(ctx, &settled); err != nil {
			return fmt.Errorf("store: settle scheduled task runs: %w", err)
		}
		// The occurrence keeps its run, so a one-shot left at it would only
		// be advanced past it: it comes due again at the save instead, on the
		// database's clock, which the claim compares against.
		if task.RRule == "" && task.State == ScheduledTaskActive && task.NextRunAt != nil {
			for _, scheduledFor := range settled {
				if !scheduledFor.Equal(*task.NextRunAt) {
					continue
				}
				var again time.Time
				if err := tx.QueryRowContext(ctx, "SELECT now()").Scan(&again); err != nil {
					return fmt.Errorf("store: save scheduled task: %w", err)
				}
				task.NextRunAt = &again
				break
			}
		}
		result, err := tx.NewUpdate().Model(task).
			ExcludeColumn("customer_id", "created_at").
			Where("customer_id = ?", task.CustomerID).
			Where("id = ?", task.ID).
			Exec(ctx)
		if err != nil {
			return fmt.Errorf("store: save scheduled task: %w", err)
		}
		affected, err := result.RowsAffected()
		if err != nil {
			return fmt.Errorf("store: save scheduled task: %w", err)
		}
		if affected == 0 {
			return fmt.Errorf("%w: %s", ErrNoScheduledTask, task.ID)
		}
		return nil
	}))
}

// DeleteScheduledTask removes a task and, by cascade, its runs.
func (s *Store) DeleteScheduledTask(ctx context.Context, customerID, id string) error {
	if customerID == "" || id == "" {
		return stack.Wrap(errors.New("store: a customer and a task id are required"))
	}
	result, err := s.db.NewDelete().Model((*ScheduledTask)(nil)).
		Where("customer_id = ?", customerID).
		Where("id = ?", id).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete scheduled task: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete scheduled task: %w", err))
	}
	if affected == 0 {
		return stack.Wrap(fmt.Errorf("%w: %s", ErrNoScheduledTask, id))
	}
	return nil
}

// DeleteScheduledTasksForSession removes every task that follows up in a
// session, as deleting that session must: a user who deleted a conversation
// is not contacted from it again.
func (s *Store) DeleteScheduledTasksForSession(ctx context.Context, customerID, sessionID string) error {
	if customerID == "" || sessionID == "" {
		return nil
	}
	_, err := s.db.NewDelete().Model((*ScheduledTask)(nil)).
		Where("customer_id = ?", customerID).
		Where("session_id = ? OR origin_session_id = ?", sessionID, sessionID).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete scheduled tasks for session: %w", err))
	}
	return nil
}

// SetScheduledTaskSession records the session a task follows up in, once its
// first occurrence has opened one. It touches nothing else: the task may be
// claimed when this runs, and clearing the claim would fence the worker out
// of finishing the run.
func (s *Store) SetScheduledTaskSession(ctx context.Context, customerID, taskID, sessionID string) error {
	_, err := s.db.NewUpdate().Model((*ScheduledTask)(nil)).
		Set("session_id = ?", sessionID).
		Set("updated_at = now()").
		Where("customer_id = ?", customerID).
		Where("id = ?", taskID).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: set scheduled task session: %w", err))
	}
	return nil
}

// ActiveScheduledTaskCount is how many active tasks a user holds on one agent
// config, which is what that config's limit counts.
func (s *Store) ActiveScheduledTaskCount(ctx context.Context, customerID, userID, configID string) (int, error) {
	count, err := s.db.NewSelect().Model((*ScheduledTask)(nil)).
		Where("customer_id = ?", customerID).
		Where("user_id = ?", userID).
		Where("config_id = ?", configID).
		Where("state = ?", ScheduledTaskActive).
		Count(ctx)
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: count scheduled tasks: %w", err))
	}
	return count, nil
}

// ClaimedScheduledTask is a task a worker has claimed, with the run it is to
// deliver. AlreadySettled says the occurrence's run is finished already and
// only the task's advance was lost; the worker settles the task without
// delivering again.
type ClaimedScheduledTask struct {
	Task           ScheduledTask
	Run            ScheduledTaskRun
	ClaimID        string
	AlreadySettled bool
}

// ClaimDueScheduledTasks claims up to limit due tasks under a lease of the
// given length, in one transaction, comparing against the database's own
// clock. Each claim inserts the occurrence's run, or takes over a stale one.
func (s *Store) ClaimDueScheduledTasks(ctx context.Context, limit int, lease time.Duration) ([]ClaimedScheduledTask, error) {
	if limit < 1 {
		limit = 1
	}
	claimed := []ClaimedScheduledTask{}
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		var now time.Time
		if err := tx.QueryRowContext(ctx, "SELECT now()").Scan(&now); err != nil {
			return fmt.Errorf("store: claim scheduled tasks: %w", err)
		}
		var tasks []ScheduledTask
		err := tx.NewSelect().Model(&tasks).
			Where("state = ?", ScheduledTaskActive).
			Where("next_run_at <= now()").
			Where("(lease_until IS NULL OR lease_until < now())").
			Order("next_run_at ASC").
			Limit(limit).
			For("UPDATE SKIP LOCKED").
			Scan(ctx)
		if err != nil {
			return fmt.Errorf("store: claim scheduled tasks: %w", err)
		}
		for _, task := range tasks {
			if task.NextRunAt == nil {
				continue
			}
			scheduledFor := *task.NextRunAt
			claimID := newID()
			leaseUntil := now.Add(lease)

			var run ScheduledTaskRun
			runErr := tx.NewSelect().Model(&run).
				Where("customer_id = ?", task.CustomerID).
				Where("task_id = ?", task.ID).
				Where("scheduled_for = ?", scheduledFor).
				Where("trigger = ?", ScheduledTaskTriggerSchedule).
				Limit(1).
				Scan(ctx)
			if runErr != nil && !errors.Is(runErr, sql.ErrNoRows) {
				return fmt.Errorf("store: claim scheduled tasks: %w", runErr)
			}

			if runErr == nil {
				switch {
				case run.State != ScheduledTaskRunRunning:
					// The occurrence was settled and the advance was lost.
				case run.StartedAt != nil && now.Sub(*run.StartedAt) < scheduledTaskStaleRun &&
					!heldRun(run, now):
					// A worker had it recently and its lease lapsed without a
					// renewal only just now; park the task until the run is
					// stale enough to take over.
					park := run.StartedAt.Add(scheduledTaskStaleRun)
					if _, err := tx.NewUpdate().Model((*ScheduledTask)(nil)).
						Set("lease_until = ?", park).
						Set("updated_at = ?", now).
						Where("customer_id = ?", task.CustomerID).
						Where("id = ?", task.ID).
						Exec(ctx); err != nil {
						return fmt.Errorf("store: claim scheduled tasks: %w", err)
					}
					continue
				case run.Attempts >= maxScheduledTaskRunAttempts:
					failure := fmt.Sprintf("the occurrence was started %d times and its worker was lost each time", run.Attempts)
					if _, err := tx.NewUpdate().Model((*ScheduledTaskRun)(nil)).
						Set("state = ?", ScheduledTaskRunFailed).
						Set("error = ?", failure).
						Set("finished_at = ?", now).
						Where("id = ?", run.ID).
						Exec(ctx); err != nil {
						return fmt.Errorf("store: fail scheduled task run: %w", err)
					}
					run.State, run.Error, run.FinishedAt = ScheduledTaskRunFailed, &failure, &now
				default:
					// Stale: take it over under the same run, so the same
					// command id, so a copy that did land is refused.
					if _, err := tx.NewUpdate().Model((*ScheduledTaskRun)(nil)).
						Set("attempts = attempts + 1").
						Set("started_at = ?", now).
						Set("error = NULL").
						Where("id = ?", run.ID).
						Exec(ctx); err != nil {
						return fmt.Errorf("store: reclaim scheduled task run: %w", err)
					}
					run.Attempts++
					run.StartedAt = &now
					run.Error = nil
				}
				if run.State != ScheduledTaskRunRunning {
					// The worker settles the task from the run as it stands,
					// without delivering.
					if _, err := tx.NewUpdate().Model((*ScheduledTask)(nil)).
						Set("lease_until = ?", leaseUntil).
						Set("claim_id = ?", claimID).
						Set("updated_at = ?", now).
						Where("customer_id = ?", task.CustomerID).
						Where("id = ?", task.ID).
						Exec(ctx); err != nil {
						return fmt.Errorf("store: claim scheduled tasks: %w", err)
					}
					task.LeaseUntil = &leaseUntil
					task.ClaimID = claimID
					claimed = append(claimed, ClaimedScheduledTask{Task: task, Run: run, ClaimID: claimID, AlreadySettled: true})
					continue
				}
			} else {
				run = ScheduledTaskRun{
					ID:           newID(),
					TaskID:       task.ID,
					CustomerID:   task.CustomerID,
					ScheduledFor: scheduledFor,
					Trigger:      ScheduledTaskTriggerSchedule,
					Prompt:       task.Instructions,
					State:        ScheduledTaskRunRunning,
					Attempts:     1,
					StartedAt:    &now,
					CreatedAt:    now,
				}
				if _, err := tx.NewInsert().Model(&run).Exec(ctx); err != nil {
					return fmt.Errorf("store: insert scheduled task run: %w", err)
				}
			}

			if _, err := tx.NewUpdate().Model((*ScheduledTask)(nil)).
				Set("lease_until = ?", leaseUntil).
				Set("claim_id = ?", claimID).
				Set("updated_at = ?", now).
				Where("customer_id = ?", task.CustomerID).
				Where("id = ?", task.ID).
				Exec(ctx); err != nil {
				return fmt.Errorf("store: claim scheduled tasks: %w", err)
			}
			task.LeaseUntil = &leaseUntil
			task.ClaimID = claimID
			claimed = append(claimed, ClaimedScheduledTask{Task: task, Run: run, ClaimID: claimID})
		}
		return nil
	})
	if err != nil {
		return nil, stack.Wrap(err)
	}
	return claimed, nil
}

// heldRun reports whether a running run is a held one whose hold has ended:
// nobody is delivering it, however recently it started, so it is taken over
// rather than parked. A run another worker took a moment ago has no hold
// recorded, or one still to run out, or one that ended before that worker
// took it up: its lease lapsing is a worker lost, parked like any other. The
// run has started.
func heldRun(run ScheduledTaskRun, now time.Time) bool {
	return run.PostponedUntil != nil && !run.PostponedUntil.After(now) &&
		run.PostponedUntil.After(*run.StartedAt)
}

// PostponeScheduledTask holds a claimed task until until, under its claim:
// the lease moves to the hold's end and the claim stays, so no other worker
// takes the occurrence meanwhile, and the run stays running with the hold
// recorded on it. False is a claim an update or pause fenced meanwhile: the
// change that fenced it decides what happens instead.
func (s *Store) PostponeScheduledTask(ctx context.Context, customerID, taskID, runID, claimID string, until time.Time) (bool, error) {
	held := false
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		result, err := tx.NewUpdate().Model((*ScheduledTask)(nil)).
			Set("lease_until = ?", until).
			Set("updated_at = now()").
			Where("customer_id = ?", customerID).
			Where("id = ?", taskID).
			Where("claim_id = ?", claimID).
			Exec(ctx)
		if err != nil {
			return fmt.Errorf("store: postpone scheduled task: %w", err)
		}
		affected, err := result.RowsAffected()
		if err != nil {
			return fmt.Errorf("store: postpone scheduled task: %w", err)
		}
		held = affected > 0
		if !held {
			return nil
		}
		if _, err := tx.NewUpdate().Model((*ScheduledTaskRun)(nil)).
			Set("postponed_until = ?", until).
			Where("id = ?", runID).
			Exec(ctx); err != nil {
			return fmt.Errorf("store: postpone scheduled task run: %w", err)
		}
		return nil
	})
	if err != nil {
		return false, stack.Wrap(err)
	}
	return held, nil
}

// RenewScheduledTaskLease extends a claim the worker still holds. False is a
// claim it no longer holds: an update, pause or delete took it.
func (s *Store) RenewScheduledTaskLease(ctx context.Context, customerID, taskID, claimID string, lease time.Duration) (bool, error) {
	result, err := s.db.NewUpdate().Model((*ScheduledTask)(nil)).
		Set("lease_until = now() + make_interval(secs => ?)", lease.Seconds()).
		Where("customer_id = ?", customerID).
		Where("id = ?", taskID).
		Where("claim_id = ?", claimID).
		Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: renew scheduled task lease: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: renew scheduled task lease: %w", err))
	}
	return affected > 0, nil
}

// ScheduledTaskOutcome is what a worker reports of an occurrence.
type ScheduledTaskOutcome struct {
	// RunState is delivered or failed.
	RunState    string
	DeliveredTo string
	SessionID   string
	Error       string
	// NextRunAt and State are the task's after this occurrence; the finish
	// applies them only under the claim that delivered it.
	NextRunAt           *time.Time
	State               string
	ConsecutiveFailures int
}

// FinishScheduledTaskRun records an occurrence's outcome and, under the
// claim, advances the task. applied is false when the claim was fenced by an
// update or pause: the run is still recorded, the schedule is left alone.
func (s *Store) FinishScheduledTaskRun(ctx context.Context, customerID, runID, claimID string, outcome ScheduledTaskOutcome) (bool, error) {
	applied := false
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		var run ScheduledTaskRun
		if err := tx.NewSelect().Model(&run).Where("id = ?", runID).Limit(1).Scan(ctx); err != nil {
			return fmt.Errorf("store: finish scheduled task run: %w", err)
		}
		now := time.Now().UTC()
		finished := now
		run.State = outcome.RunState
		run.DeliveredTo = outcome.DeliveredTo
		if outcome.SessionID != "" {
			run.SessionID = outcome.SessionID
		}
		run.FinishedAt = &finished
		if outcome.Error != "" {
			run.Error = &outcome.Error
		} else {
			run.Error = nil
		}
		if _, err := tx.NewUpdate().Model(&run).
			Column("state", "delivered_to", "session_id", "finished_at", "error").
			Where("id = ?", run.ID).
			Exec(ctx); err != nil {
			return fmt.Errorf("store: finish scheduled task run: %w", err)
		}

		result, err := tx.NewUpdate().Model((*ScheduledTask)(nil)).
			Set("next_run_at = ?", outcome.NextRunAt).
			Set("state = ?", outcome.State).
			Set("consecutive_failures = ?", outcome.ConsecutiveFailures).
			Set("lease_until = NULL").
			Set("claim_id = ''").
			Set("updated_at = ?", now).
			Where("customer_id = ?", customerID).
			Where("id = ?", run.TaskID).
			Where("claim_id = ?", claimID).
			Exec(ctx)
		if err != nil {
			return fmt.Errorf("store: finish scheduled task run: %w", err)
		}
		affected, err := result.RowsAffected()
		if err != nil {
			return fmt.Errorf("store: finish scheduled task run: %w", err)
		}
		applied = affected > 0
		return nil
	})
	if err != nil {
		return false, stack.Wrap(err)
	}
	return applied, nil
}

// InsertScheduledTaskRun stores a run a caller made by hand (trigger manual)
// or a missed occurrence. A scheduled occurrence that already has its run is
// left as it is.
func (s *Store) InsertScheduledTaskRun(ctx context.Context, run *ScheduledTaskRun) error {
	if run.ID == "" {
		run.ID = newID()
	}
	if run.CreatedAt.IsZero() {
		run.CreatedAt = time.Now().UTC()
	}
	if run.Trigger == "" {
		run.Trigger = ScheduledTaskTriggerSchedule
	}
	if run.State == "" {
		run.State = ScheduledTaskRunRunning
	}
	_, err := s.db.NewInsert().Model(run).
		On("CONFLICT DO NOTHING").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: insert scheduled task run: %w", err))
	}
	return nil
}

// FinishManualScheduledTaskRun records the outcome of a run made by hand.
func (s *Store) FinishManualScheduledTaskRun(ctx context.Context, runID, state, deliveredTo, sessionID, failure string) error {
	now := time.Now().UTC()
	update := s.db.NewUpdate().Model((*ScheduledTaskRun)(nil)).
		Set("state = ?", state).
		Set("delivered_to = ?", deliveredTo).
		Set("session_id = ?", sessionID).
		Set("finished_at = ?", now).
		Where("id = ?", runID)
	if failure != "" {
		update = update.Set("error = ?", failure)
	} else {
		update = update.Set("error = NULL")
	}
	if _, err := update.Exec(ctx); err != nil {
		return stack.Wrap(fmt.Errorf("store: finish scheduled task run: %w", err))
	}
	return nil
}

// ScheduledTaskRuns returns a page of a task's runs, newest first.
func (s *Store) ScheduledTaskRuns(ctx context.Context, customerID, taskID string, limit int, after *ScheduledTaskPosition) ([]ScheduledTaskRun, error) {
	if customerID == "" || taskID == "" {
		return nil, stack.Wrap(errors.New("store: a customer and a task id are required"))
	}
	var runs []ScheduledTaskRun
	query := s.db.NewSelect().Model(&runs).
		Where("customer_id = ?", customerID).
		Where("task_id = ?", taskID)
	if after != nil {
		query = query.Where("(created_at, id) < (?, ?)", after.CreatedAt, after.ID)
	}
	err := query.Order("created_at DESC", "id DESC").Limit(ScheduledTaskLimit(limit) + 1).Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: scheduled task runs: %w", err))
	}
	return runs, nil
}

// LastScheduledTaskRunsAt is when each of a customer's tasks last finished a
// run, by task id, in one query; a task that never finished one is left out.
func (s *Store) LastScheduledTaskRunsAt(ctx context.Context, customerID string, taskIDs []string) (map[string]time.Time, error) {
	var rows []struct {
		TaskID     string    `bun:"task_id"`
		FinishedAt time.Time `bun:"finished_at"`
	}
	err := s.db.NewSelect().Model((*ScheduledTaskRun)(nil)).
		Column("task_id").
		ColumnExpr("max(finished_at) AS finished_at").
		Where("customer_id = ?", customerID).
		Where("task_id IN (?)", bun.In(taskIDs)).
		Where("finished_at IS NOT NULL").
		Group("task_id").
		Scan(ctx, &rows)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: last scheduled task runs: %w", err))
	}
	last := make(map[string]time.Time, len(rows))
	for _, row := range rows {
		last[row.TaskID] = row.FinishedAt
	}
	return last, nil
}

// NextDueScheduledTask is the soonest an active task may be claimed, for a
// worker deciding how long to sleep: its next run, or the end of the lease or
// hold on it when that is later. GREATEST skips a null lease
// (https://www.postgresql.org/docs/current/functions-conditional.html#FUNCTIONS-GREATEST-LEAST),
// and is written as scheduled_tasks_wake_idx indexes it.
func (s *Store) NextDueScheduledTask(ctx context.Context) (time.Time, bool, error) {
	var wake time.Time
	err := s.db.NewSelect().Model((*ScheduledTask)(nil)).
		ColumnExpr("GREATEST(next_run_at, lease_until)").
		Where("state = ?", ScheduledTaskActive).
		Where("next_run_at IS NOT NULL").
		OrderExpr("GREATEST(next_run_at, lease_until) ASC").
		Limit(1).
		Scan(ctx, &wake)
	if errors.Is(err, sql.ErrNoRows) {
		return time.Time{}, false, nil
	}
	if err != nil {
		return time.Time{}, false, stack.Wrap(fmt.Errorf("store: next due scheduled task: %w", err))
	}
	return wake, true, nil
}

// SetUserTimezone records the zone a user was explicitly known to be in. The
// row is created when there is none; on an existing row only the zone moves,
// never the kind the claim path reads.
func (s *Store) SetUserTimezone(ctx context.Context, customerID, userID, kind, zone string) error {
	if customerID == "" || userID == "" || zone == "" {
		return nil
	}
	if kind == "" {
		kind = UserKindAuthenticated
	}
	user := &User{CustomerID: customerID, ID: userID, Kind: kind, Timezone: zone, CreatedAt: time.Now().UTC()}
	_, err := s.db.NewInsert().Model(user).
		On("CONFLICT (customer_id, id) DO UPDATE").
		Set("timezone = EXCLUDED.timezone").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: set user timezone: %w", err))
	}
	return nil
}

// UserTimezone returns the zone stored for a user, or empty.
func (s *Store) UserTimezone(ctx context.Context, customerID, userID string) (string, error) {
	if customerID == "" || userID == "" {
		return "", nil
	}
	var user User
	err := s.db.NewSelect().Model(&user).
		Column("timezone").
		Where("customer_id = ?", customerID).
		Where("id = ?", userID).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return "", nil
	}
	if err != nil {
		return "", stack.Wrap(fmt.Errorf("store: user timezone: %w", err))
	}
	return user.Timezone, nil
}
