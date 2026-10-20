package api

import (
	"context"
	"errors"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// errNoScheduledTasks is what the task paths say on a deployment without a database.
var errNoScheduledTasks = notConfigured("scheduled tasks are not available: no database configured")

// errUnknownScheduledTask is what a caller is told about a task that is not theirs, or not
// there: the same answer for both, so a guessed id reveals nothing.
var errUnknownScheduledTask = APIError{
	Type: ErrorTypeNotFound, Code: codeScheduledTaskNotFound,
	Message: "there is no such scheduled task",
}

// errTooManyScheduledTasks is a user at their active-task limit.
var errTooManyScheduledTasks = APIError{Type: ErrorTypeConflict, Code: codeTooManyTasks,
	Message: "the user already has the most active scheduled tasks this agent allows; " +
		"cancel or pause one first"}

// errUnkeptTaskSession is a create naming a session whose conversation is not kept: an
// occurrence follows up by reopening the conversation, which an incognito session leaves
// nothing of.
var errUnkeptTaskSession = invalidRequest("session_id names a session whose conversation is not kept, " +
	"such as an incognito one, so there is nothing to follow up in; leave session_id out and name the agent instead")

// scheduleFailure renders a scheduling refusal with its own code, as the model is told it.
func scheduleFailure(err error) error {
	var failure *scheduling.Error
	if errors.As(err, &failure) {
		return APIError{Type: ErrorTypeInvalidRequest, Code: failure.Code, Message: failure.Message}
	}
	return err
}

// scheduledTasksCustomer is the preamble of every task operation: who is calling, and
// whether this deployment holds tasks at all.
func (s *Server) scheduledTasksCustomer(ctx context.Context) (string, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return "", errMissingCustomer
	}
	if s.store == nil {
		return "", errNoScheduledTasks
	}
	return customerID, nil
}

// loadScheduledTask reads one of the customer's tasks and refuses it as not there when
// the caller may not reach it: a task belongs to its user, recorded as user id plus
// caller kind, and reachability is a session owner's (Owner.Reaches), so a task a
// backend made is still its user's own device's to read.
func (s *Server) loadScheduledTask(ctx context.Context, customerID, id string) (store.ScheduledTask, error) {
	task, err := s.store.ScheduledTask(ctx, customerID, id)
	if errors.Is(err, store.ErrNoScheduledTask) {
		return store.ScheduledTask{}, errUnknownScheduledTask
	}
	if err != nil {
		return store.ScheduledTask{}, err
	}
	owner := session.Owner{CustomerID: task.CustomerID, UserID: task.UserID, Kind: auth.Kind(task.CallerKind)}
	if !OwnerFrom(ctx).Reaches(owner) {
		return store.ScheduledTask{}, errUnknownScheduledTask
	}
	return task, nil
}

// scheduledTaskScope is who a list is of: a backend naming a user lists that user's
// tasks, a backend naming nobody lists the customer's, and a device lists its own,
// including the ones its backend made in its name. A device naming nobody has no tasks
// of its own, so ok is false and it is told about none, as session lists do
// (session.Manager.find): an empty user id narrows nothing in the store.
func (s *Server) scheduledTaskScope(ctx context.Context) (store.ScheduledTaskScope, bool) {
	caller := CallerFrom(ctx)
	if ServerSideFrom(ctx) {
		return store.ScheduledTaskScope{UserID: caller.UserID}, true
	}
	if caller.UserID == "" {
		return store.ScheduledTaskScope{}, false
	}
	kinds := []string{string(KindFrom(ctx))}
	if KindFrom(ctx).Verified() {
		kinds = append(kinds, string(auth.KindServer))
	}
	return store.ScheduledTaskScope{UserID: caller.UserID, Kinds: kinds}, true
}

func (s *Server) listScheduledTasks(ctx context.Context, request *listScheduledTasksRequest) (*listScheduledTasksResponse, error) {
	customerID, failure := s.scheduledTasksCustomer(ctx)
	if failure != nil {
		return nil, failure
	}
	scope, ok := s.scheduledTaskScope(ctx)
	if !ok {
		return &listScheduledTasksResponse{Body: ScheduledTaskPage{Items: []ScheduledTask{}}}, nil
	}
	after, err := decodeCursor[store.ScheduledTaskPosition](request.Cursor.ptr())
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	limit := store.ScheduledTaskLimit(value(request.Limit.ptr()))
	rows, err := s.store.ScheduledTasks(ctx, customerID, store.ScheduledTaskFilter{
		Scope:     scope,
		State:     value(request.State.ptr()),
		ConfigID:  value(request.ConfigId.ptr()),
		SessionID: value(request.SessionId.ptr()),
	}, limit, after)
	if err != nil {
		return nil, err
	}
	rows, more := page(rows, limit)
	listed := ScheduledTaskPage{Items: make([]ScheduledTask, 0, len(rows)), HasMore: more}
	for _, row := range rows {
		rendered, err := s.scheduledTaskOf(ctx, row)
		if err != nil {
			return nil, err
		}
		listed.Items = append(listed.Items, rendered)
	}
	if more {
		last := rows[len(rows)-1]
		listed.NextCursor = encodeCursor(store.ScheduledTaskPosition{CreatedAt: last.CreatedAt, ID: last.ID})
	}
	return &listScheduledTasksResponse{Body: listed}, nil
}

func (s *Server) createScheduledTask(ctx context.Context, request *createScheduledTaskRequest) (*createScheduledTaskResponse, error) {
	customerID, failure := s.scheduledTasksCustomer(ctx)
	if failure != nil {
		return nil, failure
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	body := request.Body
	// A task is made for one named user: without one there is nobody for it to
	// belong to, which is also what a router without authentication answers,
	// since no user is verified there for anybody to name.
	userID := CallerFrom(ctx).UserID
	if userID == "" {
		return nil, invalidRequest("name the user the task is for with X-Stream-User-Id")
	}

	// Huma's minLength counts spaces, which the stored task would not.
	if strings.TrimSpace(body.Title) == "" {
		return nil, invalidRequest("title cannot be empty")
	}
	if strings.TrimSpace(body.Instructions) == "" {
		return nil, invalidRequest("instructions cannot be empty")
	}
	task := store.ScheduledTask{
		CustomerID: customerID, UserID: userID, CallerKind: string(auth.KindServer),
		Title: strings.TrimSpace(body.Title), Instructions: body.Instructions,
		State: store.ScheduledTaskActive, CreatedBy: "api",
	}
	where, err := deliverToOf(body.DeliverTo)
	if err != nil {
		return nil, err
	}
	task.Channels = where
	quiet, err := quietHoursOf(body.QuietHours)
	if err != nil {
		return nil, err
	}
	task.QuietHours = quiet
	if body.Id != nil {
		task.ID = *body.Id
	}
	if body.Custom != nil {
		task.Custom = *body.Custom
	}

	var config *store.AgentConfig
	switch {
	case value(body.SessionId) != "":
		configID, conversationID, failure := s.scheduledTaskSession(ctx, customerID, userID, value(body.SessionId))
		if failure != nil {
			return nil, failure
		}
		task.SessionID, task.OriginSessionID, task.ConversationID = value(body.SessionId), value(body.SessionId), conversationID
		task.ConfigID = configID
		if configID != "" {
			found, failure := s.configFor(ctx, customerID, &configID, nil)
			if failure != nil {
				return nil, failure
			}
			config = found
		}
	default:
		found, failure := s.configFor(ctx, customerID, body.ConfigId, body.Agent)
		if failure != nil {
			return nil, failure
		}
		if found == nil {
			return nil, invalidRequest("name the agent by agent or config_id, or give session_id for the conversation to follow up in")
		}
		config = found
		task.ConfigID = config.ID
	}

	stored, err := s.store.UserTimezone(ctx, customerID, userID)
	if err != nil {
		return nil, err
	}
	configDefault := ""
	if config != nil {
		configDefault = config.Scheduling.DefaultTimezone
	}
	zone, err := scheduling.ResolveZone(value(body.Schedule.Timezone), "", stored, configDefault)
	if err != nil {
		return nil, scheduleFailure(err)
	}
	now := time.Now()
	parsed, err := scheduling.Parse(scheduling.Input{
		At: value(body.Schedule.At), In: value(body.Schedule.In),
		Repeat: value(body.Schedule.Repeat), Until: value(body.Schedule.Until),
		Timezone: zone,
	}, now)
	if err != nil {
		return nil, scheduleFailure(err)
	}
	// A repeat's first occurrence may have passed (09:00 daily, written at
	// 10:00); it starts at the next one, as the agent's tools do.
	next, ok := parsed.Next(now)
	if !ok {
		return nil, scheduleFailure(&scheduling.Error{Code: scheduling.CodeScheduleInPast,
			Message: "that schedule has no occurrence left; its last one has passed"})
	}
	task.StartLocal, task.Timezone, task.RRule = parsed.StartLocal, parsed.Timezone, parsed.Repeat
	if !parsed.Until.IsZero() {
		until := parsed.Until
		task.UntilDate = &until
	}
	task.NextRunAt = &next

	maxActive := scheduling.MaxActivePerUser
	if config != nil && config.Scheduling.MaxTasksPerUser > 0 {
		maxActive = config.Scheduling.MaxTasksPerUser
	}
	if err := s.store.CreateScheduledTask(ctx, &task, maxActive); err != nil {
		switch {
		case errors.Is(err, store.ErrScheduledTaskIDTaken):
			return nil, conflict("a scheduled task with this id already exists; read it back, or choose another id")
		case errors.Is(err, store.ErrTooManyScheduledTasks):
			return nil, errTooManyScheduledTasks
		}
		return nil, err
	}
	s.kickScheduled()
	rendered, err := s.scheduledTaskOf(ctx, task)
	if err != nil {
		return nil, err
	}
	return &createScheduledTaskResponse{Body: rendered}, nil
}

// scheduledTaskSession is the config and conversation of the session a create
// names, or errUnknownSession when it is not the named user's own. A live
// session answers first: its row is written as it goes, and may not be there
// yet for one just opened.
func (s *Server) scheduledTaskSession(ctx context.Context, customerID, userID, id string) (string, string, error) {
	if s.sessions != nil {
		if live, ok := s.sessions.Get(id, session.Owner{CustomerID: customerID, UserID: userID, Kind: auth.KindServer}); ok {
			spec := live.Spec()
			if spec.Caller.UserID != userID {
				return "", "", errUnknownSession
			}
			if !spec.PersistConversation {
				return "", "", errUnkeptTaskSession
			}
			return spec.ConfigID, spec.ConversationID, nil
		}
	}
	row, err := s.store.StoredSession(ctx, customerID, id)
	if err != nil || row.UserID != userID {
		return "", "", errUnknownSession
	}
	// What reopenedFromRow needs to carry the conversation on.
	if row.ConversationID == "" {
		return "", "", errUnkeptTaskSession
	}
	return row.ConfigID, row.ConversationID, nil
}

func (s *Server) getScheduledTask(ctx context.Context, request *getScheduledTaskRequest) (*getScheduledTaskResponse, error) {
	customerID, failure := s.scheduledTasksCustomer(ctx)
	if failure != nil {
		return nil, failure
	}
	task, err := s.loadScheduledTask(ctx, customerID, request.Id)
	if err != nil {
		return nil, err
	}
	rendered, err := s.scheduledTaskOf(ctx, task)
	if err != nil {
		return nil, err
	}
	return &getScheduledTaskResponse{Body: rendered}, nil
}

func (s *Server) updateScheduledTask(ctx context.Context, request *updateScheduledTaskRequest) (*updateScheduledTaskResponse, error) {
	customerID, failure := s.scheduledTasksCustomer(ctx)
	if failure != nil {
		return nil, failure
	}
	task, err := s.loadScheduledTask(ctx, customerID, request.Id)
	if err != nil {
		return nil, err
	}
	body := request.Body
	// What the agent is told at each occurrence is the backend's to write: a
	// device may retitle and reschedule its own reminders, not re-aim them.
	if body.Instructions != nil && !ServerSideFrom(ctx) {
		return nil, errServerSideOnly
	}
	if body.Title != nil {
		title := strings.TrimSpace(*body.Title)
		if title == "" {
			return nil, invalidRequest("title cannot be empty")
		}
		task.Title = title
	}
	if body.Instructions != nil {
		if strings.TrimSpace(*body.Instructions) == "" {
			return nil, invalidRequest("instructions cannot be empty")
		}
		task.Instructions = *body.Instructions
	}
	if body.DeliverTo != nil {
		where, err := deliverToOf(*body.DeliverTo)
		if err != nil {
			return nil, err
		}
		task.Channels = where
	}
	if body.QuietHours != nil {
		quiet, err := quietHoursOf(body.QuietHours)
		if err != nil {
			return nil, err
		}
		task.QuietHours = quiet
	}
	now := time.Now()
	current := task.NextRunAt
	if body.Schedule != nil {
		if err := s.rescheduleTask(&task, *body.Schedule, current, now); err != nil {
			return nil, err
		}
	}
	if body.Next != nil {
		if task.State == store.ScheduledTaskCompleted {
			return nil, invalidRequest("a completed task has no next occurrence to move")
		}
		moved, err := scheduling.Parse(scheduling.Input{
			At: value(body.Next.At), In: value(body.Next.In), Timezone: task.Timezone,
		}, now)
		if err != nil {
			return nil, scheduleFailure(err)
		}
		next, ok := moved.First()
		if !ok {
			return nil, scheduleFailure(&scheduling.Error{Code: scheduling.CodeScheduleInPast,
				Message: "that time has passed; give a time in the future"})
		}
		if err := scheduling.CheckMove(next, current, now); err != nil {
			return nil, scheduleFailure(err)
		}
		task.NextRunAt = &next
		if task.RRule == "" {
			// A one-shot has no schedule after its one occurrence, so moving
			// the next moves the schedule with it.
			task.StartLocal = moved.StartLocal
		}
	}
	if err := s.store.SaveScheduledTask(ctx, &task); err != nil {
		return nil, err
	}
	s.kickScheduled()
	rendered, err := s.scheduledTaskOf(ctx, task)
	if err != nil {
		return nil, err
	}
	return &updateScheduledTaskResponse{Body: rendered}, nil
}

// scheduledTaskChannels are the channels a task may be delivered on: the
// conversation the follow-up lands in, the channels the user already talks
// to the agent on, and a call.
var scheduledTaskChannels = []string{"conversation", "sms", "whatsapp", "imessage", "phone"}

// deliverToOf reads a written deliver_to, in the order written, each
// channel once. Left out or empty is the conversation alone, as every task
// before channels was. A name nothing carries is refused here rather than
// at three in the morning.
func deliverToOf(written []string) ([]string, error) {
	where := make([]string, 0, len(written))
	for _, channel := range written {
		if !slices.Contains(scheduledTaskChannels, channel) {
			return nil, invalidRequest("deliver_to names channels nothing carries: " +
				"use conversation, sms, whatsapp, imessage or phone")
		}
		if !slices.Contains(where, channel) {
			where = append(where, channel)
		}
	}
	if len(where) == 0 {
		where = []string{store.ScheduledTaskDeliveredConversation}
	}
	return where, nil
}

// quietHoursOf reads a written quiet_hours, or nil for no hold: both
// boundaries left empty is the hold cleared, which is also what an update
// that clears one sends.
func quietHoursOf(written *ScheduledTaskQuietHours) (*store.ScheduledTaskQuietHours, error) {
	if written == nil || (written.Start == "" && written.End == "") {
		return nil, nil
	}
	if _, _, ok := scheduling.QuietHours(written.Start, written.End); !ok {
		return nil, invalidRequest("quiet_hours is a start and an end the user's clock reads, " +
			"such as 22:00 and 07:00, and they cannot read the same")
	}
	return &store.ScheduledTaskQuietHours{Start: written.Start, End: written.End}, nil
}

// rescheduleTask replaces a task's schedule with the one written, keeping the
// task's zone when the new one names none.
func (s *Server) rescheduleTask(task *store.ScheduledTask, written ScheduledTaskSchedule, current *time.Time, now time.Time) error {
	zone := value(written.Timezone)
	if zone == "" {
		zone = task.Timezone
	}
	parsed, err := scheduling.Parse(scheduling.Input{
		At: value(written.At), In: value(written.In),
		Repeat: value(written.Repeat), Until: value(written.Until),
		Timezone: zone,
	}, now)
	if err != nil {
		return scheduleFailure(err)
	}
	task.StartLocal, task.Timezone, task.RRule = parsed.StartLocal, parsed.Timezone, parsed.Repeat
	task.UntilDate = nil
	if !parsed.Until.IsZero() {
		until := parsed.Until
		task.UntilDate = &until
	}
	if task.State == store.ScheduledTaskActive || task.State == store.ScheduledTaskPaused {
		next, ok := parsed.Next(now)
		if !ok {
			return scheduleFailure(&scheduling.Error{Code: scheduling.CodeScheduleInPast,
				Message: "that schedule has no occurrence left; its last one has passed"})
		}
		if err := scheduling.CheckMove(next, current, now); err != nil {
			return scheduleFailure(err)
		}
		task.NextRunAt = &next
	}
	return nil
}

func (s *Server) pauseScheduledTask(ctx context.Context, request *pauseScheduledTaskRequest) (*pauseScheduledTaskResponse, error) {
	customerID, failure := s.scheduledTasksCustomer(ctx)
	if failure != nil {
		return nil, failure
	}
	task, err := s.loadScheduledTask(ctx, customerID, request.Id)
	if err != nil {
		return nil, err
	}
	if task.State == store.ScheduledTaskActive {
		task.State = store.ScheduledTaskPaused
		if err := s.store.SaveScheduledTask(ctx, &task); err != nil {
			return nil, err
		}
	}
	rendered, err := s.scheduledTaskOf(ctx, task)
	if err != nil {
		return nil, err
	}
	return &pauseScheduledTaskResponse{Body: rendered}, nil
}

func (s *Server) resumeScheduledTask(ctx context.Context, request *resumeScheduledTaskRequest) (*resumeScheduledTaskResponse, error) {
	customerID, failure := s.scheduledTasksCustomer(ctx)
	if failure != nil {
		return nil, failure
	}
	task, err := s.loadScheduledTask(ctx, customerID, request.Id)
	if err != nil {
		return nil, err
	}
	if task.State == store.ScheduledTaskCompleted {
		return nil, conflict("a completed task cannot be resumed; schedule a new one")
	}
	if task.State == store.ScheduledTaskPaused || task.State == store.ScheduledTaskFailed {
		maxActive := scheduling.MaxActivePerUser
		if config, err := s.store.AgentConfig(ctx, customerID, task.ConfigID); err == nil && config.Scheduling.MaxTasksPerUser > 0 {
			maxActive = config.Scheduling.MaxTasksPerUser
		}
		active, err := s.store.ActiveScheduledTaskCount(ctx, customerID, task.UserID)
		if err != nil {
			return nil, err
		}
		if active >= maxActive {
			return nil, errTooManyScheduledTasks
		}
		schedule, err := scheduling.FromStored(task.StartLocal, task.Timezone, task.RRule, untilOf(task))
		if err != nil {
			return nil, scheduleFailure(err)
		}
		// Counted from now: what a pause slept through is never replayed.
		next, ok := schedule.Next(time.Now())
		if !ok {
			task.State, task.NextRunAt = store.ScheduledTaskCompleted, nil
		} else {
			task.State, task.NextRunAt = store.ScheduledTaskActive, &next
		}
		task.ConsecutiveFailures = 0
		if err := s.store.SaveScheduledTask(ctx, &task); err != nil {
			return nil, err
		}
		s.kickScheduled()
	}
	rendered, err := s.scheduledTaskOf(ctx, task)
	if err != nil {
		return nil, err
	}
	return &resumeScheduledTaskResponse{Body: rendered}, nil
}

// runScheduledTask sets one occurrence going by hand, now, and answers 202
// with its run. The schedule does not move: this is the task tested, or
// triggered from the customer's own system, not an occurrence brought forward.
func (s *Server) runScheduledTask(ctx context.Context, request *runScheduledTaskRequest) (*runScheduledTaskResponse, error) {
	customerID, failure := s.scheduledTasksCustomer(ctx)
	if failure != nil {
		return nil, failure
	}
	task, err := s.loadScheduledTask(ctx, customerID, request.Id)
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC()
	run := store.ScheduledTaskRun{
		TaskID: task.ID, CustomerID: customerID, ScheduledFor: now,
		Trigger: store.ScheduledTaskTriggerManual, Prompt: task.Instructions,
		State: store.ScheduledTaskRunRunning, Attempts: 1, StartedAt: &now,
	}
	if err := s.store.InsertScheduledTaskRun(ctx, &run); err != nil {
		return nil, err
	}
	go s.deliverManualRun(task, run)
	return &runScheduledTaskResponse{Body: scheduledTaskRunOf(task, run)}, nil
}

// deliverManualRun delivers a run made by hand and records how it landed.
func (s *Server) deliverManualRun(task store.ScheduledTask, run store.ScheduledTaskRun) {
	ctx := context.Background()
	deliveredTo, sessionID, err := s.DeliverScheduledTask(ctx, store.ClaimedScheduledTask{Task: task, Run: run})
	state := store.ScheduledTaskRunDelivered
	failure := ""
	if err != nil {
		state, failure = store.ScheduledTaskRunFailed, err.Error()
		s.logger.Error("a scheduled task run by hand did not land",
			"task", task.ID, "run", run.ID, "error", err)
	}
	if err := s.store.FinishManualScheduledTaskRun(ctx, run.ID, state, deliveredTo, sessionID, failure); err != nil {
		s.logger.Error("could not record a scheduled task run by hand",
			"task", task.ID, "run", run.ID, "error", err)
	}
}

func (s *Server) deleteScheduledTask(ctx context.Context, request *deleteScheduledTaskRequest) (*struct{}, error) {
	customerID, failure := s.scheduledTasksCustomer(ctx)
	if failure != nil {
		return nil, failure
	}
	if _, err := s.loadScheduledTask(ctx, customerID, request.Id); err != nil {
		return nil, err
	}
	if err := s.store.DeleteScheduledTask(ctx, customerID, request.Id); err != nil {
		if errors.Is(err, store.ErrNoScheduledTask) {
			return nil, errUnknownScheduledTask
		}
		return nil, err
	}
	return nil, nil
}

func (s *Server) listScheduledTaskRuns(ctx context.Context, request *listScheduledTaskRunsRequest) (*listScheduledTaskRunsResponse, error) {
	customerID, failure := s.scheduledTasksCustomer(ctx)
	if failure != nil {
		return nil, failure
	}
	task, err := s.loadScheduledTask(ctx, customerID, request.Id)
	if err != nil {
		return nil, err
	}
	after, err := decodeCursor[store.ScheduledTaskPosition](request.Cursor.ptr())
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	limit := store.ScheduledTaskLimit(value(request.Limit.ptr()))
	rows, err := s.store.ScheduledTaskRuns(ctx, customerID, task.ID, limit, after)
	if err != nil {
		return nil, err
	}
	rows, more := page(rows, limit)
	listed := ScheduledTaskRunPage{Items: make([]ScheduledTaskRun, 0, len(rows)), HasMore: more}
	for _, row := range rows {
		listed.Items = append(listed.Items, scheduledTaskRunOf(task, row))
	}
	if more {
		last := rows[len(rows)-1]
		listed.NextCursor = encodeCursor(store.ScheduledTaskPosition{CreatedAt: last.CreatedAt, ID: last.ID})
	}
	return &listScheduledTaskRunsResponse{Body: listed}, nil
}

// scheduledTaskOf renders a task, its times read in its own zone: one string
// gives both the instant and the user's wall clock.
func (s *Server) scheduledTaskOf(ctx context.Context, task store.ScheduledTask) (ScheduledTask, error) {
	loc, err := scheduling.LoadZone(task.Timezone)
	if err != nil {
		loc = time.UTC
	}
	schedule, _ := scheduling.FromStored(task.StartLocal, task.Timezone, task.RRule, untilOf(task))
	rendered := ScheduledTask{
		Id: task.ID, UserId: task.UserID, ConfigId: task.ConfigID, SessionId: task.SessionID,
		Title: task.Title, Instructions: task.Instructions, State: task.State,
		ConsecutiveFailures: task.ConsecutiveFailures, CreatedBy: task.CreatedBy,
		Custom: task.Custom, CreatedAt: task.CreatedAt, UpdatedAt: task.UpdatedAt,
		Upcoming:  []string{},
		DeliverTo: task.Channels,
	}
	if len(rendered.DeliverTo) == 0 {
		rendered.DeliverTo = []string{store.ScheduledTaskDeliveredConversation}
	}
	if task.QuietHours != nil {
		rendered.QuietHours = &ScheduledTaskQuietHours{Start: task.QuietHours.Start, End: task.QuietHours.End}
	}
	rendered.Schedule = ScheduledTaskSchedule{
		At:       optional(task.StartLocal.Format("2006-01-02T15:04:05")),
		Timezone: optional(task.Timezone),
	}
	if task.RRule != "" {
		rendered.Schedule.Repeat = optional(task.RRule)
	}
	if task.UntilDate != nil {
		rendered.Schedule.Until = optional(task.UntilDate.Format("2006-01-02"))
	}
	if schedule != nil {
		rendered.Schedule.Description = schedule.Describe()
	}
	if task.NextRunAt != nil {
		next := task.NextRunAt.In(loc).Format(time.RFC3339)
		rendered.NextRunAt = &next
		if schedule != nil {
			for _, occurrence := range schedule.Upcoming(task.NextRunAt.Add(-time.Second), 5) {
				rendered.Upcoming = append(rendered.Upcoming, occurrence.In(loc).Format(time.RFC3339))
			}
		}
	}
	if s.store != nil {
		if last, err := s.store.LastScheduledTaskRunAt(ctx, task.CustomerID, task.ID); err == nil && !last.IsZero() {
			rendered.LastRunAt = optional(last.UTC().Format(time.RFC3339))
		}
	}
	return rendered, nil
}

// scheduledTaskRunOf renders a run, its due time in its task's zone.
func scheduledTaskRunOf(task store.ScheduledTask, run store.ScheduledTaskRun) ScheduledTaskRun {
	loc, err := scheduling.LoadZone(task.Timezone)
	if err != nil {
		loc = time.UTC
	}
	return ScheduledTaskRun{
		Id: run.ID, TaskId: run.TaskID,
		ScheduledFor: run.ScheduledFor.In(loc).Format(time.RFC3339),
		Trigger:      run.Trigger, State: run.State, DeliveredTo: run.DeliveredTo,
		SessionId: run.SessionID, Attempts: run.Attempts,
		StartedAt: run.StartedAt, FinishedAt: run.FinishedAt, Error: run.Error,
	}
}

// registerScheduledTasks declares the operations served in scheduled_tasks.go.
func (s *Server) registerScheduledTasks(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listScheduledTasks",
		Method:      http.MethodGet,
		Path:        "/v1/agents/scheduled-tasks",
		Summary:     "The scheduled tasks of the user, or of the customer",
		Description: "Newest first, cursor-paged. A device lists its own user's tasks, including the ones " +
			"its backend made in its name. A backend naming a user with X-Stream-User-Id lists that user's; " +
			"a backend naming nobody lists every task of the customer, instructions included: a customer's " +
			"whole task list is support-visible, so put nothing in instructions a user must not have read " +
			"back to them. Devices see instructions too, for the same reason.",
		Responses:  map[string]*huma.Response{"200": {Description: "A page of scheduled tasks"}},
		Errors:     []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotImplemented},
		Extensions: map[string]any{clientAccessibleExtension: true},
	}, s.listScheduledTasks)
	huma.Register(api, huma.Operation{
		OperationID: "createScheduledTask",
		Method:      http.MethodPost,
		Path:        "/v1/agents/scheduled-tasks",
		Summary:     "Schedule a task for a user",
		Description: "Server-side only. The task belongs to the user named with X-Stream-User-Id, and without " +
			"one there is nobody for it to belong to, so the create is refused. The agent is named by agent " +
			"or config_id, or taken from session_id's own config, and the follow-up lands in that session's " +
			"conversation; without a session, the first occurrence opens one. An id the caller chooses may " +
			"be sent again only to be told it is taken (409), the same answer session create gives. " +
			"scheduling.enabled on the config is not required: that flag only offers the tools to the model. " +
			"deliver_to says where an occurrence goes, in order, and quiet_hours holds one that comes due " +
			"in them until they end.",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The task was stored"},
			"409": errorResponse("The id is taken, or the user is at their active-task limit"),
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound, http.StatusNotImplemented},
	}, s.createScheduledTask)
	huma.Register(api, huma.Operation{
		OperationID: "getScheduledTask",
		Method:      http.MethodGet,
		Path:        "/v1/agents/scheduled-tasks/{id}",
		Summary:     "One scheduled task",
		Description: "A task that is not the caller's to reach is answered as not there.",
		Responses:   map[string]*huma.Response{"200": {Description: "The task"}},
		Errors:      []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
		Extensions:  map[string]any{clientAccessibleExtension: true},
	}, s.getScheduledTask)
	huma.Register(api, huma.Operation{
		OperationID: "updateScheduledTask",
		Method:      http.MethodPatch,
		Path:        "/v1/agents/scheduled-tasks/{id}",
		Summary:     "Change a scheduled task",
		Description: "A field left out is kept. A schedule given replaces the whole schedule, except a " +
			"timezone left out of it keeps the task's. next moves the next occurrence only — a snooze, or " +
			"a skip when it lands past the following one — and on a one-shot it is the reschedule. An end " +
			"user's device may change title, schedule, next, deliver_to and quiet_hours; instructions are " +
			"server-side, and a device sending them is refused with a 403.",
		Responses:  map[string]*huma.Response{"200": {Description: "The task as it now is"}},
		Errors:     []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
		Extensions: map[string]any{clientAccessibleExtension: true},
	}, s.updateScheduledTask)
	huma.Register(api, huma.Operation{
		OperationID: "pauseScheduledTask",
		Method:      http.MethodPost,
		Path:        "/v1/agents/scheduled-tasks/{id}/pause",
		Summary:     "Pause a scheduled task",
		Description: "Nothing comes due while a task is paused; what it slept through is never replayed.",
		Responses:   map[string]*huma.Response{"200": {Description: "The task, paused"}},
		Errors:      []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
		Extensions:  map[string]any{clientAccessibleExtension: true},
	}, s.pauseScheduledTask)
	huma.Register(api, huma.Operation{
		OperationID: "resumeScheduledTask",
		Method:      http.MethodPost,
		Path:        "/v1/agents/scheduled-tasks/{id}/resume",
		Summary:     "Resume a scheduled task",
		Description: "The next occurrence is counted from now, so missed occurrences are never replayed, " +
			"and a run of failures is forgiven. A completed task cannot be resumed (409).",
		Responses: map[string]*huma.Response{
			"200": {Description: "The task, running again"},
			"409": errorResponse("The task is completed, or the user is at their active-task limit"),
		},
		Errors:     []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
		Extensions: map[string]any{clientAccessibleExtension: true},
	}, s.resumeScheduledTask)
	huma.Register(api, huma.Operation{
		OperationID: "runScheduledTask",
		Method:      http.MethodPost,
		Path:        "/v1/agents/scheduled-tasks/{id}/run",
		Summary:     "Run a scheduled task now",
		Description: "Server-side only. Sets one occurrence going by hand and answers with its run while it " +
			"is still running; the schedule does not move. Test a task with it, or trigger one from your " +
			"own system.",
		DefaultStatus: http.StatusAccepted,
		Responses:     map[string]*huma.Response{"202": {Description: "The run that was set going"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound, http.StatusNotImplemented},
	}, s.runScheduledTask)
	huma.Register(api, huma.Operation{
		OperationID:   "deleteScheduledTask",
		Method:        http.MethodDelete,
		Path:          "/v1/agents/scheduled-tasks/{id}",
		Summary:       "Delete a scheduled task",
		Description:   "This is cancel: the task and its runs are gone, and there is no cancelled state to clutter lists.",
		DefaultStatus: http.StatusNoContent,
		Responses:     map[string]*huma.Response{"204": {Description: "The task is gone"}},
		Errors:        []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
		Extensions:    map[string]any{clientAccessibleExtension: true},
	}, s.deleteScheduledTask)
	huma.Register(api, huma.Operation{
		OperationID: "listScheduledTaskRuns",
		Method:      http.MethodGet,
		Path:        "/v1/agents/scheduled-tasks/{id}/runs",
		Summary:     "A scheduled task's runs",
		Description: "Newest first, cursor-paged. Only occurrences that came due, or were run by hand, " +
			"are recorded; snoozes and pauses write no runs.",
		Responses:  map[string]*huma.Response{"200": {Description: "A page of the task's runs"}},
		Errors:     []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
		Extensions: map[string]any{clientAccessibleExtension: true},
	}, s.listScheduledTaskRuns)
}

// ScheduledTaskSchedule is the schedule object, the same on create, on update and in
// responses: responses only ever carry at, never in.
type ScheduledTaskSchedule struct {
	At          *string `json:"at,omitempty" doc:"When it first runs, as the user's clock reads it, 2026-10-20T09:00:00; with an offset or Z it is that instant, read in the schedule's zone. Exactly one of at and in is given on a write."`
	In          *string `json:"in,omitempty" doc:"How long from now instead, an ISO 8601 duration: PT20M, P1D. Days are calendar days: P1D is the same time tomorrow. Resolved on write; a response never carries it." example:"PT20M"`
	Timezone    *string `json:"timezone,omitempty" doc:"An IANA zone, such as Europe/Amsterdam. On create, left out resolves to the user's stored zone, then the agent's default_timezone; on update, to the task's own zone." example:"Europe/Amsterdam"`
	Repeat      *string `json:"repeat,omitempty" doc:"Omit for once. Otherwise an RRULE subset without DTSTART or UNTIL: FREQ=DAILY, FREQ=WEEKLY;BYDAY=MO,WE,FR, FREQ=MONTHLY;BYMONTHDAY=1." example:"FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"`
	Until       *string `json:"until,omitempty" doc:"The last local date it repeats on, inclusive, 2026-12-31." example:"2026-12-31"`
	Description string  `json:"description,omitempty" doc:"What the schedule says, in one sentence. Read-only: written by the router, and the same sentence the model reads back." example:"Every weekday at 09:00 (Europe/Amsterdam)"`
}

// ScheduledTaskNext moves a task's next occurrence only.
type ScheduledTaskNext struct {
	At *string `json:"at,omitempty" doc:"The next occurrence as the user's clock reads it, in the task's zone."`
	In *string `json:"in,omitempty" doc:"How long from now instead, an ISO 8601 duration: PT15M." example:"PT15M"`
}

// ScheduledTaskQuietHours is the part of the user's day an occurrence is
// held through rather than delivered into, as their clock reads it in the
// task's zone.
type ScheduledTaskQuietHours struct {
	Start string `json:"start" doc:"When the hold begins, as the user's clock reads it. Both boundaries empty is no hold, which is also how an update clears one." example:"22:00"`
	End   string `json:"end" doc:"When the hold ends. It may read before the start: 22:00 to 07:00 holds the night." example:"07:00"`
}

// CreateScheduledTaskRequest is the body of createScheduledTask.
type CreateScheduledTaskRequest struct {
	Id           *string                  `json:"id,omitempty" doc:"The task's id, chosen by the caller as a session id can be: up to 64 letters, digits, - and _. Left out, the router generates one. Sending one that is used answers 409." pattern:"^[A-Za-z0-9_-]{1,64}$" example:"booking-4821-reminder"`
	Agent        *string                  `json:"agent,omitempty" doc:"The agent config by name, as session create names it. config_id or this, or session_id."`
	ConfigId     *string                  `json:"config_id,omitempty" doc:"The agent config by id. agent or this, or session_id."`
	SessionId    *string                  `json:"session_id,omitempty" doc:"The named user's own session whose conversation the follow-up lands in; its config is the task's. Left out, the first occurrence opens a session."`
	Title        string                   `json:"title" doc:"A short name the user will recognise." minLength:"1" maxLength:"80" example:"Dentist tomorrow"`
	Instructions string                   `json:"instructions" doc:"What the agent is told when the task comes due. Read back by the owner and by the customer's backend, so put nothing in it a user must not read." minLength:"1" maxLength:"2000"`
	Schedule     ScheduledTaskSchedule    `json:"schedule" doc:"When it runs."`
	DeliverTo    []string                 `json:"deliver_to,omitempty" doc:"Where an occurrence is delivered, in order, the first that lands: conversation, sms, whatsapp, imessage or phone. The agent's reply is written in its conversation either way; a channel carries the same words out, and phone rings the user and speaks them. Left out or empty is the conversation alone."`
	QuietHours   *ScheduledTaskQuietHours `json:"quiet_hours,omitempty" doc:"The part of the user's day an occurrence that comes due in is held through, until it ends, on every channel."`
	Custom       *map[string]any          `json:"custom,omitempty" doc:"The caller's own labels, handed back untouched."`
}

// UpdateScheduledTaskRequest is the body of updateScheduledTask. A field left out is kept.
type UpdateScheduledTaskRequest struct {
	Title        *string                  `json:"title,omitempty" doc:"A short name the user will recognise." maxLength:"80"`
	Instructions *string                  `json:"instructions,omitempty" doc:"What the agent is told when the task comes due. Server-side only: a device sending it is refused." maxLength:"2000"`
	Schedule     *ScheduledTaskSchedule   `json:"schedule,omitempty" doc:"Replaces the whole schedule, except a timezone left out of it keeps the task's. Its first occurrence follows the rule of next: 15 minutes from now at the soonest, unless it lands no earlier than the current next occurrence."`
	Next         *ScheduledTaskNext       `json:"next,omitempty" doc:"Moves the next occurrence only: a snooze, or a skip when it lands past the following one. It lands 15 minutes from now at the soonest, unless it lands no earlier than the current next occurrence; sooner is refused with interval_too_short."`
	DeliverTo    *[]string                `json:"deliver_to,omitempty" doc:"Replaces where an occurrence is delivered, in order. An empty list is the conversation alone."`
	QuietHours   *ScheduledTaskQuietHours `json:"quiet_hours,omitempty" doc:"Replaces the part of the day an occurrence is held through. Both boundaries empty clears the hold."`
}

// ScheduledTask is a scheduled task as the API renders it.
type ScheduledTask struct {
	Id                  string                   `json:"id"`
	UserId              string                   `json:"user_id"`
	ConfigId            string                   `json:"config_id"`
	SessionId           string                   `json:"session_id" doc:"The conversation the task follows up in. Empty until its first occurrence opens one."`
	Title               string                   `json:"title"`
	Instructions        string                   `json:"instructions"`
	Schedule            ScheduledTaskSchedule    `json:"schedule"`
	DeliverTo           []string                 `json:"deliver_to" nullable:"false" doc:"Where an occurrence is delivered, in order, the first that lands."`
	QuietHours          *ScheduledTaskQuietHours `json:"quiet_hours,omitempty" doc:"The part of the user's day an occurrence that comes due in is held through. Absent is no hold."`
	State               string                   `json:"state" doc:"active, paused, completed (a one-shot that ran, or a recurrence past its end), or failed (auto-paused after five failed occurrences in a row; resume clears it)." enum:"active,paused,completed,failed"`
	NextRunAt           *string                  `json:"next_run_at" doc:"The next occurrence, RFC 3339 with the offset of the task's own zone. Null when there is none."`
	Upcoming            []string                 `json:"upcoming" nullable:"false" doc:"Up to the next five occurrences, RFC 3339 with the task zone's offset."`
	LastRunAt           *string                  `json:"last_run_at" doc:"When a run last finished, in UTC. Null when none has."`
	ConsecutiveFailures int                      `json:"consecutive_failures"`
	CreatedBy           string                   `json:"created_by" doc:"agent when the model made it in a conversation, api when a backend did." enum:"agent,api"`
	Custom              map[string]any           `json:"custom,omitempty"`
	CreatedAt           time.Time                `json:"created_at"`
	UpdatedAt           time.Time                `json:"updated_at"`
}

// TransformSchema names created_by's constants: AgentLogSource has agent too, and two enums
// sharing a value make oapi-codegen prefix both, renaming the Go SDK's Agent (as
// ConnectionOwnerType explains).
func (*ScheduledTask) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["created_by"].Extensions = map[string]any{
		"x-enum-varnames": []any{"ScheduledTaskCreatedByAgent", "ScheduledTaskCreatedByApi"},
	}
	return schema
}

// ScheduledTaskRun is one occurrence of a task, as the API renders it.
type ScheduledTaskRun struct {
	Id           string     `json:"id"`
	TaskId       string     `json:"task_id"`
	ScheduledFor string     `json:"scheduled_for" doc:"When the occurrence was due, RFC 3339 with the task zone's offset."`
	Trigger      string     `json:"trigger" doc:"schedule when it came due, manual when it was run by hand." enum:"schedule,manual"`
	State        string     `json:"state" doc:"running (retries included), delivered, failed, or missed (it fell outside the grace period)." enum:"running,delivered,failed,missed"`
	DeliveredTo  string     `json:"delivered_to" doc:"Where the follow-up landed: conversation, sms, whatsapp, imessage or phone. Empty until it lands."`
	SessionId    string     `json:"session_id"`
	Attempts     int        `json:"attempts"`
	StartedAt    *time.Time `json:"started_at"`
	FinishedAt   *time.Time `json:"finished_at"`
	Error        *string    `json:"error" doc:"Why it failed, when it did. Null otherwise."`
}

// ScheduledTaskPage is a page of scheduled tasks.
type ScheduledTaskPage struct {
	HasMore    bool            `json:"has_more"`
	Items      []ScheduledTask `json:"items" nullable:"false"`
	NextCursor *string         `json:"next_cursor,omitempty" doc:"Pass as cursor for the next page. Absent on the last one."`
}

// ScheduledTaskRunPage is a page of a task's runs.
type ScheduledTaskRunPage struct {
	HasMore    bool               `json:"has_more"`
	Items      []ScheduledTaskRun `json:"items" nullable:"false"`
	NextCursor *string            `json:"next_cursor,omitempty" doc:"Pass as cursor for the next page. Absent on the last one."`
}

type listScheduledTasksRequest struct {
	Limit     optionalParam[int]    `query:"limit" doc:"Up to 200. Omitted is 25." minimum:"1" maximum:"200"`
	Cursor    optionalParam[string] `query:"cursor" doc:"The next_cursor of the previous page. Omitted is the first page."`
	State     optionalParam[string] `query:"state" doc:"Only tasks in this state." enum:"active,paused,completed,failed"`
	ConfigId  optionalParam[string] `query:"config_id" doc:"Only tasks on this agent config."`
	SessionId optionalParam[string] `query:"session_id" doc:"Only tasks following up in this session."`
}

type listScheduledTasksResponse struct {
	Body ScheduledTaskPage
}

type createScheduledTaskRequest struct {
	UserID string                      `header:"X-Stream-User-Id" doc:"The user the task is for, named by the backend acting for them. Required: without a named user the create is refused." required:"false"`
	Body   *CreateScheduledTaskRequest `required:"true"`
}

type createScheduledTaskResponse struct {
	Body ScheduledTask
}

type getScheduledTaskRequest struct {
	Id string `path:"id" doc:"The task, as returned when it was created."`
}

type getScheduledTaskResponse struct {
	Body ScheduledTask
}

type updateScheduledTaskRequest struct {
	Id   string                      `path:"id" doc:"The task, as returned when it was created."`
	Body *UpdateScheduledTaskRequest `required:"true"`
}

type updateScheduledTaskResponse struct {
	Body ScheduledTask
}

type pauseScheduledTaskRequest struct {
	Id string `path:"id" doc:"The task, as returned when it was created."`
}

type pauseScheduledTaskResponse struct {
	Body ScheduledTask
}

type resumeScheduledTaskRequest struct {
	Id string `path:"id" doc:"The task, as returned when it was created."`
}

type resumeScheduledTaskResponse struct {
	Body ScheduledTask
}

type runScheduledTaskRequest struct {
	Id string `path:"id" doc:"The task, as returned when it was created."`
}

type runScheduledTaskResponse struct {
	Body ScheduledTaskRun
}

type deleteScheduledTaskRequest struct {
	Id string `path:"id" doc:"The task, as returned when it was created."`
}

type listScheduledTaskRunsRequest struct {
	Id     string                `path:"id" doc:"The task, as returned when it was created."`
	Limit  optionalParam[int]    `query:"limit" doc:"Up to 200. Omitted is 25." minimum:"1" maximum:"200"`
	Cursor optionalParam[string] `query:"cursor" doc:"The next_cursor of the previous page. Omitted is the first page."`
}

type listScheduledTaskRunsResponse struct {
	Body ScheduledTaskRunPage
}
