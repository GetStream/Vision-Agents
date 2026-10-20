package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The scheduling tools' names, as the model asks for them.
const (
	scheduleTaskTool      = "schedule_task"
	listScheduledTool     = "list_scheduled_tasks"
	updateScheduledTool   = "update_scheduled_task"
	cancelScheduledTool   = "cancel_scheduled_task"
	scheduledTaskNotFound = "scheduled_task_not_found"
	tooManyTasks          = "too_many_tasks"
	conflictCode          = "conflict"
	serverSideOnly        = "server_side_only"
)

// scheduleTools are the scheduling tools a session may offer. The four names
// are the router's own: a caller that declared one of them keeps it, and the
// router's is simply not offered (see Manager.schedulingRunner).
func scheduleTools() []harness.Tool {
	return []harness.Tool{
		{
			Name: scheduleTaskTool,
			Description: "Schedule something for you to do for the user later: a reminder, a check-in, " +
				"or a task to carry out once or on a repeating schedule. When it is due, you are woken in " +
				"this conversation with the instructions and act on them. Give the time with at, or the " +
				"span with in. Agree the day and time with the user first, in their own zone unless they " +
				"name another, then read back the schedule this returns.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"title":        map[string]any{"type": "string", "maxLength": 80, "description": "A short name the user will recognise, e.g. \"Stretch\"."},
					"instructions": map[string]any{"type": "string", "maxLength": 2000, "description": "What to do when it is due, written to yourself, e.g. \"Remind Sam to stretch and ask how his back is.\""},
					"at":           map[string]any{"type": "string", "description": "The user's local date and time, 2026-10-20T09:00, or a time of day, 09:00, meaning the next one. Give at or in."},
					"in":           map[string]any{"type": "string", "description": "How long from now, as an ISO 8601 duration: PT20M, PT2H, P1D. Days are calendar days: P1D is the same time tomorrow."},
					"repeat":       map[string]any{"type": "string", "description": "Omit for once. Otherwise an RRULE without DTSTART or UNTIL: FREQ=DAILY, FREQ=WEEKLY;BYDAY=MO,WE,FR, FREQ=MONTHLY;BYMONTHDAY=1. FREQ=MONTHLY started on the 29th to 31st skips the months without that day; FREQ=MONTHLY;BYMONTHDAY=-1 is the last day of every month."},
					"until":        map[string]any{"type": "string", "description": "The last local date it repeats on, 2026-12-31."},
					"timezone":     map[string]any{"type": "string", "description": "An IANA zone such as Europe/Amsterdam, only when the user names one. Otherwise theirs is used."},
				},
				"required": []string{"title", "instructions"},
			},
		},
		{
			Name: listScheduledTool,
			Description: "List the tasks you have scheduled for this user with this agent, each with when it runs next. " +
				"Running and paused tasks come first; more says when there are others not shown.",
			Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
		},
		{
			Name: updateScheduledTool,
			Description: "Change a task you scheduled, by the id list_scheduled_tasks gave it: reschedule it, " +
				"snooze its next run to a new at or in, skip its next run, pause it or resume it. With " +
				"reschedule, fields left out keep what the task has. Returns the task as it now stands.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"id":           map[string]any{"type": "string", "description": "The task's id."},
					"action":       map[string]any{"type": "string", "enum": []string{"reschedule", "snooze", "skip_next", "pause", "resume"}, "description": "What to do to the task."},
					"title":        map[string]any{"type": "string", "maxLength": 80},
					"instructions": map[string]any{"type": "string", "maxLength": 2000},
					"at":           map[string]any{"type": "string", "description": "The user's local date and time, 2026-10-20T09:00, or a time of day, 09:00."},
					"in":           map[string]any{"type": "string", "description": "How long from now, as an ISO 8601 duration: PT20M, PT2H, P1D. Days are calendar days: P1D is the same time tomorrow."},
					"repeat":       map[string]any{"type": "string", "description": "An RRULE without DTSTART or UNTIL. An empty string makes the task run once. FREQ=MONTHLY started on the 29th to 31st skips the months without that day; BYMONTHDAY=-1 is the last day."},
					"until":        map[string]any{"type": "string", "description": "The last local date it repeats on, 2026-12-31. An empty string lifts the end."},
					"timezone":     map[string]any{"type": "string", "description": "An IANA zone such as Europe/Amsterdam."},
				},
				"required": []string{"id", "action"},
			},
		},
		{
			Name:        cancelScheduledTool,
			Description: "Cancel a task you scheduled, by the id list_scheduled_tasks gave it. It will not run again.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"id": map[string]any{"type": "string", "description": "The task's id."},
				},
				"required": []string{"id"},
			},
		},
	}
}

// schedulingRunner builds the schedule runner for a session, and the tools it
// adds to what the session offers, or nil when this session may not schedule:
// the config has not enabled it, there is no store to hold tasks, the session
// is incognito, or the caller is not a verified end user or a backend acting
// for one — guests and anonymous callers have ids too and must not get the
// tools. A caller that declared a tool with one of these names keeps theirs.
func (m *Manager) schedulingRunner(spec Spec, sessionID string, offered []harness.Tool) (*scheduleRunner, []harness.Tool) {
	if !spec.Scheduling.Enabled || m.options.Store == nil || spec.Incognito ||
		spec.ConfigID == "" || !verifiedCaller(spec) {
		return nil, nil
	}
	taken := map[string]bool{}
	for _, tool := range offered {
		taken[tool.Name] = true
	}
	runner := &scheduleRunner{
		db: m.options.Store, spec: spec, sessionID: sessionID,
		now: time.Now, logger: m.logger, owned: map[string]bool{},
	}
	var added []harness.Tool
	for _, tool := range scheduleTools() {
		if taken[tool.Name] {
			continue
		}
		runner.owned[tool.Name] = true
		added = append(added, tool)
	}
	if len(added) == 0 {
		return nil, nil
	}
	return runner, added
}

// scheduleRunner runs the scheduling tools against the store. A task belongs
// to the session's user on this session's agent config, whatever conversation
// of theirs created it.
type scheduleRunner struct {
	db        *store.Store
	spec      Spec
	sessionID string
	now       func() time.Time
	logger    *slog.Logger
	owned     map[string]bool
	next      agent.ToolRunner
}

func (r *scheduleRunner) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	if !r.owned[call.Name] {
		return r.next.Run(ctx, call)
	}
	switch call.Name {
	case scheduleTaskTool:
		return r.schedule(ctx, call)
	case listScheduledTool:
		return r.list(ctx)
	case updateScheduledTool:
		return r.update(ctx, call)
	default:
		return r.cancel(ctx, call)
	}
}

type scheduleArgs struct {
	Title        string `json:"title"`
	Instructions string `json:"instructions"`
	At           string `json:"at"`
	In           string `json:"in"`
	Repeat       string `json:"repeat"`
	Until        string `json:"until"`
	Timezone     string `json:"timezone"`
}

func (r *scheduleRunner) schedule(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	var args scheduleArgs
	if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
		return refusal(scheduling.CodeInvalidSchedule, "the arguments are not JSON I can read"), nil
	}
	if refusal := checkText(args.Title, args.Instructions); refusal != nil {
		return refusal, nil
	}
	now := r.now()
	zone, err := r.zone(ctx, args.Timezone)
	if err != nil {
		if parts, refused := refusalOf(err); refused {
			return parts, nil
		}
		return nil, err
	}
	parsed, err := scheduling.Parse(scheduling.Input{
		At: args.At, In: args.In, Repeat: args.Repeat, Until: args.Until, Timezone: zone,
	}, now)
	if err != nil {
		parts, _ := refusalOf(err)
		return parts, nil
	}
	next, ok := parsed.Next(now)
	if !ok {
		return refusal(scheduling.CodeScheduleInPast, "that schedule has no occurrence left; its last one has passed"), nil
	}
	task := store.ScheduledTask{
		CustomerID:      r.spec.CustomerID,
		ConfigID:        r.spec.ConfigID,
		UserID:          r.spec.Caller.UserID,
		CallerKind:      string(r.spec.CallerKind),
		OriginSessionID: r.sessionID,
		SessionID:       r.sessionID,
		ConversationID:  r.spec.ConversationID,
		Title:           strings.TrimSpace(args.Title),
		Instructions:    args.Instructions,
		StartLocal:      parsed.StartLocal,
		Timezone:        parsed.Timezone,
		RRule:           parsed.Repeat,
		NextRunAt:       &next,
		State:           store.ScheduledTaskActive,
		Channels:        []string{store.ScheduledTaskDeliveredConversation},
		CreatedBy:       "agent",
	}
	if !parsed.Until.IsZero() {
		until := parsed.Until
		task.UntilDate = &until
	}
	maxActive := r.maxActive()
	if err := r.db.CreateScheduledTask(ctx, &task, maxActive); err != nil {
		if errors.Is(err, store.ErrTooManyScheduledTasks) {
			return refusal(tooManyTasks, fmt.Sprintf("the user already has %d tasks running, the most this agent allows; cancel one first, or pause one until it is wanted", maxActive)), nil
		}
		return nil, err
	}
	// A zone the user named for this task is theirs from now on.
	if args.Timezone != "" {
		kind := store.UserKindAuthenticated
		if r.spec.CallerKind == auth.KindGuest {
			kind = store.UserKindGuest
		}
		if err := r.db.SetUserTimezone(ctx, r.spec.CustomerID, r.spec.Caller.UserID, kind, parsed.Timezone); err != nil {
			r.logger.Error("could not store the user's timezone", "user", r.spec.Caller.UserID, "error", err)
		}
	}
	return result(r.taskResult(&task, parsed, now)), nil
}

func (r *scheduleRunner) list(ctx context.Context) ([]llm.ContentPart, error) {
	tasks, more, err := r.listed(ctx)
	if err != nil {
		return nil, err
	}
	now := r.now()
	items := make([]map[string]any, 0, len(tasks))
	var display *time.Location
	for i := range tasks {
		task := &tasks[i]
		item := map[string]any{"id": task.ID, "title": task.Title, "state": task.State}
		if schedule, err := storedSchedule(task); err == nil {
			item["schedule"] = schedule.Describe()
			if display == nil {
				display = schedule.Location()
			}
			if task.NextRunAt != nil {
				item["next"] = localNext(*task.NextRunAt, schedule.Location(), now)
			}
		}
		items = append(items, item)
	}
	if display == nil {
		display = r.displayZone()
	}
	listed := map[string]any{
		"now":   localNow(now, display),
		"tasks": items,
	}
	if more {
		listed["more"] = "the user has more tasks than these; the running and paused ones are listed first"
	}
	return result(listed), nil
}

// listedStates is the order list_scheduled_tasks gives tasks in: the ones that
// will still run, then the paused, which can be resumed, before the finished.
var listedStates = []string{store.ScheduledTaskActive, store.ScheduledTaskPaused,
	store.ScheduledTaskFailed, store.ScheduledTaskCompleted}

// listed is the user's tasks on this agent as list_scheduled_tasks shows them,
// state by state in listedStates, at most as many as the user may have running
// so every running one fits, and whether there are more than that.
func (r *scheduleRunner) listed(ctx context.Context) ([]store.ScheduledTask, bool, error) {
	limit := max(scheduling.MaxActivePerUser, r.maxActive())
	filter := store.ScheduledTaskFilter{
		Scope: store.ScheduledTaskScope{UserID: r.spec.Caller.UserID,
			Kinds: []string{string(r.spec.CallerKind), string(auth.KindServer)}},
		ConfigID: r.spec.ConfigID,
	}
	var tasks []store.ScheduledTask
	for _, state := range listedStates {
		filter.State = state
		// The store answers one more than the limit, so a state that overflows it says so.
		page, err := r.db.ScheduledTasks(ctx, r.spec.CustomerID, filter, limit, nil)
		if err != nil {
			return nil, false, err
		}
		tasks = append(tasks, page...)
		if len(tasks) > limit {
			return tasks[:limit], true, nil
		}
	}
	return tasks, false, nil
}

type updateArgs struct {
	ID           string  `json:"id"`
	Action       string  `json:"action"`
	Title        *string `json:"title"`
	Instructions *string `json:"instructions"`
	At           *string `json:"at"`
	In           *string `json:"in"`
	Repeat       *string `json:"repeat"`
	Until        *string `json:"until"`
	Timezone     *string `json:"timezone"`
}

func (r *scheduleRunner) update(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	var args updateArgs
	if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
		return refusal(scheduling.CodeInvalidSchedule, "the arguments are not JSON I can read"), nil
	}
	task, parts, err := r.ownTask(ctx, args.ID)
	if parts != nil || err != nil {
		return parts, err
	}
	schedule, err := storedSchedule(&task)
	if err != nil {
		return nil, err
	}
	now := r.now()
	if args.Title != nil {
		if refusal := checkTitle(*args.Title); refusal != nil {
			return refusal, nil
		}
		task.Title = strings.TrimSpace(*args.Title)
	}
	if args.Instructions != nil {
		// What a backend told the agent to do is the backend's to change, as over
		// the API, where a device may not rewrite it either.
		if task.CreatedBy != "agent" {
			return refusal(serverSideOnly, "the instructions of a task the app scheduled are the app's to change; you may retitle or reschedule it"), nil
		}
		if refusal := checkInstructions(*args.Instructions); refusal != nil {
			return refusal, nil
		}
		task.Instructions = *args.Instructions
	}
	switch args.Action {
	case "pause":
		if task.State == store.ScheduledTaskCompleted {
			return refusal(conflictCode, "that task has already finished; there is nothing to pause"), nil
		}
		task.State = store.ScheduledTaskPaused
	case "resume":
		if task.State == store.ScheduledTaskCompleted {
			return refusal(conflictCode, "that task has finished; schedule a new one instead"), nil
		}
		next, ok := schedule.Next(now)
		if !ok {
			return refusal(scheduling.CodeScheduleInPast, "that task has no occurrence left; reschedule it to a new time instead"), nil
		}
		if parts, err := r.room(ctx, &task); parts != nil || err != nil {
			return parts, err
		}
		task.State = store.ScheduledTaskActive
		task.NextRunAt = &next
		task.ConsecutiveFailures = 0
	case "reschedule":
		parsed, parts := r.rescheduled(&task, args, now)
		if parts != nil {
			return parts, nil
		}
		schedule = parsed
		if task.State != store.ScheduledTaskPaused {
			if parts, err := r.room(ctx, &task); parts != nil || err != nil {
				return parts, err
			}
			task.State = store.ScheduledTaskActive
		}
	case "snooze":
		target, parts := r.snoozeTarget(task, args, now)
		if parts != nil {
			return parts, nil
		}
		task.NextRunAt = &target
		if task.RRule == "" {
			// A task that runs once is moved whole: its one time is the new one,
			// held the way StartLocal always is, as the wall clock read in UTC.
			local := target.In(schedule.Location())
			task.StartLocal = time.Date(local.Year(), local.Month(), local.Day(),
				local.Hour(), local.Minute(), local.Second(), 0, time.UTC)
		}
	case "skip_next":
		if task.NextRunAt == nil {
			return refusal(conflictCode, "that task has no next run to skip"), nil
		}
		if next, ok := schedule.Next(*task.NextRunAt); ok {
			task.NextRunAt = &next
		} else {
			task.State = store.ScheduledTaskCompleted
			task.NextRunAt = nil
		}
	default:
		return refusal(scheduling.CodeInvalidSchedule, "action must be reschedule, snooze, skip_next, pause or resume"), nil
	}
	if err := r.db.SaveScheduledTask(ctx, &task); err != nil {
		return nil, err
	}
	// Read the schedule back off the task, so what is reported is what now stands.
	if fresh, err := storedSchedule(&task); err == nil {
		schedule = fresh
	}
	return result(r.taskResult(&task, schedule, now)), nil
}

func (r *scheduleRunner) cancel(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	var args struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
		return refusal(scheduling.CodeInvalidSchedule, "the arguments are not JSON I can read"), nil
	}
	task, parts, err := r.ownTask(ctx, args.ID)
	if parts != nil || err != nil {
		return parts, err
	}
	if err := r.db.DeleteScheduledTask(ctx, r.spec.CustomerID, task.ID); err != nil {
		return nil, err
	}
	return result(map[string]any{"cancelled": task.Title}), nil
}

// ownTask reads one of the session user's tasks on this agent config. Another
// user's, or another agent's, is reported as not there: the tools reach only
// what list_scheduled_tasks would have shown.
func (r *scheduleRunner) ownTask(ctx context.Context, id string) (store.ScheduledTask, []llm.ContentPart, error) {
	task, err := r.db.ScheduledTask(ctx, r.spec.CustomerID, id)
	if errors.Is(err, store.ErrNoScheduledTask) {
		return task, refusal(scheduledTaskNotFound, "there is no task with that id among this user's tasks on this agent; list_scheduled_tasks shows them"), nil
	}
	if err != nil {
		return task, nil, err
	}
	if task.UserID != r.spec.Caller.UserID || task.ConfigID != r.spec.ConfigID {
		return task, refusal(scheduledTaskNotFound, "there is no task with that id among this user's tasks on this agent; list_scheduled_tasks shows them"), nil
	}
	return task, nil, nil
}

// rescheduled works out the schedule an update asks for, keeping what it left
// out, and writes it onto the task.
func (r *scheduleRunner) rescheduled(task *store.ScheduledTask, args updateArgs, now time.Time) (*scheduling.Schedule, []llm.ContentPart) {
	input := scheduling.Input{Timezone: task.Timezone, Repeat: task.RRule}
	if task.UntilDate != nil {
		input.Until = task.UntilDate.Format("2006-01-02")
	}
	if args.Timezone != nil {
		input.Timezone = *args.Timezone
	}
	if args.Repeat != nil {
		input.Repeat = *args.Repeat
	}
	if args.Until != nil {
		input.Until = *args.Until
	}
	switch {
	case args.At != nil:
		input.At = *args.At
	case args.In != nil:
		input.In = *args.In
	default:
		input.At = task.StartLocal.Format("2006-01-02T15:04")
	}
	parsed, err := scheduling.Parse(input, now)
	if err != nil {
		parts, _ := refusalOf(err)
		return nil, parts
	}
	next, ok := parsed.Next(now)
	if !ok {
		return nil, refusal(scheduling.CodeScheduleInPast, "that schedule has no occurrence left; its last one has passed")
	}
	// A reschedule without at or in re-parses the stored start, a change of
	// title, instructions or repeat, not a move of the run.
	if args.At != nil || args.In != nil {
		if err := scheduling.CheckMove(next, task.NextRunAt, now); err != nil {
			parts, _ := refusalOf(err)
			return nil, parts
		}
	}
	task.StartLocal = parsed.StartLocal
	task.Timezone = parsed.Timezone
	task.RRule = parsed.Repeat
	task.NextRunAt = &next
	if parsed.Until.IsZero() {
		task.UntilDate = nil
	} else {
		until := parsed.Until
		task.UntilDate = &until
	}
	return parsed, nil
}

// snoozeTarget is the instant a snooze moves the next run to.
func (r *scheduleRunner) snoozeTarget(task store.ScheduledTask, args updateArgs, now time.Time) (time.Time, []llm.ContentPart) {
	input := scheduling.Input{Timezone: task.Timezone}
	switch {
	case args.At != nil:
		input.At = *args.At
	case args.In != nil:
		input.In = *args.In
	default:
		return time.Time{}, refusal(scheduling.CodeInvalidSchedule, "snooze needs at or in: when it should run instead")
	}
	parsed, err := scheduling.Parse(input, now)
	if err != nil {
		parts, _ := refusalOf(err)
		return time.Time{}, parts
	}
	target, ok := parsed.First()
	if !ok {
		return time.Time{}, refusal(scheduling.CodeScheduleInPast, "that time has passed; snooze to a time in the future")
	}
	if err := scheduling.CheckMove(target, task.NextRunAt, now); err != nil {
		parts, _ := refusalOf(err)
		return time.Time{}, parts
	}
	return target, nil
}

// room refuses the change when putting the task back among the active ones
// would take its user over the limit.
func (r *scheduleRunner) room(ctx context.Context, task *store.ScheduledTask) ([]llm.ContentPart, error) {
	if task.State == store.ScheduledTaskActive {
		return nil, nil
	}
	count, err := r.db.ActiveScheduledTaskCount(ctx, r.spec.CustomerID, r.spec.Caller.UserID)
	if err != nil {
		return nil, err
	}
	if count >= r.maxActive() {
		return refusal(tooManyTasks, fmt.Sprintf("the user already has %d tasks running, the most this agent allows; cancel one first, or pause one until it is wanted", r.maxActive())), nil
	}
	return nil, nil
}

// zone picks the zone for a new task: what was named for it, the session's,
// the user's stored one, then the config's default.
func (r *scheduleRunner) zone(ctx context.Context, argument string) (string, error) {
	stored, err := r.db.UserTimezone(ctx, r.spec.CustomerID, r.spec.Caller.UserID)
	if err != nil {
		return "", err
	}
	return scheduling.ResolveZone(argument, r.spec.Timezone, stored, r.spec.Scheduling.DefaultTimezone)
}

func (r *scheduleRunner) maxActive() int {
	if r.spec.Scheduling.MaxTasksPerUser > 0 {
		return r.spec.Scheduling.MaxTasksPerUser
	}
	return scheduling.MaxActivePerUser
}

// displayZone is where the current time is told from when no task says: the
// session's zone, the config's default, or UTC.
func (r *scheduleRunner) displayZone() *time.Location {
	for _, candidate := range []string{r.spec.Timezone, r.spec.Scheduling.DefaultTimezone} {
		if candidate == "" {
			continue
		}
		if loc, err := scheduling.LoadZone(candidate); err == nil {
			return loc
		}
	}
	return time.UTC
}

// taskResult is what a tool answers with for a task that now stands as stored.
func (r *scheduleRunner) taskResult(task *store.ScheduledTask, schedule *scheduling.Schedule, now time.Time) map[string]any {
	out := map[string]any{
		"id":       task.ID,
		"title":    task.Title,
		"schedule": schedule.Describe(),
		"now":      localNow(now, schedule.Location()),
	}
	if task.NextRunAt != nil {
		out["next"] = localNext(*task.NextRunAt, schedule.Location(), now)
	}
	if task.State != store.ScheduledTaskActive {
		out["state"] = task.State
	}
	return out
}

// storedSchedule rebuilds the schedule a stored task describes.
func storedSchedule(task *store.ScheduledTask) (*scheduling.Schedule, error) {
	var until time.Time
	if task.UntilDate != nil {
		until = *task.UntilDate
	}
	return scheduling.FromStored(task.StartLocal, task.Timezone, task.RRule, until)
}

// checkText refuses a task without a title or instructions, or with more of
// either than a task may hold.
func checkText(title, instructions string) []llm.ContentPart {
	if parts := checkTitle(title); parts != nil {
		return parts
	}
	return checkInstructions(instructions)
}

func checkTitle(title string) []llm.ContentPart {
	if strings.TrimSpace(title) == "" {
		return refusal(scheduling.CodeInvalidSchedule, "a task needs a title: a short name the user will recognise")
	}
	if len([]rune(title)) > 80 {
		return refusal(scheduling.CodeInvalidSchedule, "the title is over 80 characters; shorten it")
	}
	return nil
}

func checkInstructions(instructions string) []llm.ContentPart {
	if strings.TrimSpace(instructions) == "" {
		return refusal(scheduling.CodeInvalidSchedule, "a task needs instructions: what to do when it is due, written to yourself")
	}
	if len([]rune(instructions)) > 2000 {
		return refusal(scheduling.CodeInvalidSchedule, "the instructions are over 2000 characters; shorten them")
	}
	return nil
}

// result is a tool result carrying payload as JSON.
func result(payload map[string]any) []llm.ContentPart {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return llm.TextParts("{}")
	}
	return llm.TextParts(string(encoded))
}

// refusal is a tool result saying why not, in a code the model can act on. A
// refusal is a result, not a failed call: the model reads it and does better.
func refusal(code, message string) []llm.ContentPart {
	return result(map[string]any{"error": code, "message": message})
}

// refusalOf turns a scheduling refusal into a tool result. Anything else is
// not the caller's to fix, and is reported as not refused.
func refusalOf(err error) ([]llm.ContentPart, bool) {
	var failure *scheduling.Error
	if errors.As(err, &failure) {
		return refusal(failure.Code, failure.Message), true
	}
	return nil, false
}

// localNow is an instant as the user's clock reads it, zone named.
func localNow(instant time.Time, loc *time.Location) string {
	return instant.In(loc).Format("Mon 2 Jan 2006 15:04") + " " + loc.String()
}

// localNext is an occurrence as the user's clock reads it, with how far away.
func localNext(instant time.Time, loc *time.Location, now time.Time) string {
	return instant.In(loc).Format("Mon 2 Jan 2006 15:04") + ", " + fromNow(now, instant)
}

// fromNow says how long after now an instant is, in the units people count.
func fromNow(now, instant time.Time) string {
	away := instant.Sub(now)
	if away < time.Minute {
		return "in under a minute"
	}
	days := int(away / (24 * time.Hour))
	hours := int(away % (24 * time.Hour) / time.Hour)
	minutes := int(away % time.Hour / time.Minute)
	parts := make([]string, 0, 3)
	if days > 0 {
		parts = append(parts, counted(days, "day"))
	}
	if hours > 0 {
		parts = append(parts, counted(hours, "hour"))
	}
	if minutes > 0 && days == 0 {
		parts = append(parts, counted(minutes, "minute"))
	}
	if len(parts) == 0 {
		parts = append(parts, counted(minutes, "minute"))
	}
	return "in " + strings.Join(parts, " ")
}

func counted(n int, unit string) string {
	if n == 1 {
		return "1 " + unit
	}
	return fmt.Sprintf("%d %ss", n, unit)
}
