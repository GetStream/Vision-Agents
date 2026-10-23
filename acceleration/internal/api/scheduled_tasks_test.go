//go:build integration

package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/websocket"
	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/channels"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// schedulingScript is the tool calls the scheduling models run, queued by the
// test, one entry per person who speaks. A test and the models share it
// because which model a session opened is only known once it has spoken.
type schedulingScript struct {
	mu    sync.Mutex
	turns [][]llm.ToolCall
	// rounds is the tool calls of the next rounds a tool's answer starts, one entry for each.
	rounds [][]llm.ToolCall
	// failing is how many scheduled turns each model opened from now on fails.
	failing int
}

// failScheduled has every model opened from now on fail its next times a scheduled task comes
// due, for a session the test cannot reach before it opens.
func (s *schedulingScript) failScheduled(times int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.failing = times
}

func (s *schedulingScript) failingTurns() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.failing
}

// planRound queues the tool calls of the next round a tool's answer starts: a later round
// of the turn a person's words began.
func (s *schedulingScript) planRound(calls ...llm.ToolCall) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.rounds = append(s.rounds, calls)
}

func (s *schedulingScript) nextRound() []llm.ToolCall {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.rounds) == 0 {
		return nil
	}
	calls := s.rounds[0]
	s.rounds = s.rounds[1:]
	return calls
}

// plan queues the tool calls of the next turn somebody speaks.
func (s *schedulingScript) plan(calls ...llm.ToolCall) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.turns = append(s.turns, calls)
}

func (s *schedulingScript) next() []llm.ToolCall {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.turns) == 0 {
		return nil
	}
	calls := s.turns[0]
	s.turns = s.turns[1:]
	return calls
}

// reset forgets what was queued, so one test's turns cannot leak into another's.
func (s *schedulingScript) reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.turns, s.rounds, s.failing = nil, nil, 0
}

// schedulingModel is a model that asks for the tool calls its test queued and
// records what it was offered and told, so a test can say what the model did
// and what it was given to do it with.
type schedulingModel struct {
	script *schedulingScript

	mu      sync.Mutex
	offered [][]string
	asked   []llm.ResponseParams
	// failing is how many more scheduled turns the provider fails.
	failing int
	// held, when set, keeps a scheduled turn's words back until it is closed.
	held chan struct{}
}

// failScheduled has the provider fail the next times a scheduled task comes due.
func (m *schedulingModel) failScheduled(times int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.failing = times
}

// holdScheduled keeps a scheduled turn running, saying nothing, until release.
func (m *schedulingModel) holdScheduled() (release func()) {
	held := make(chan struct{})
	m.mu.Lock()
	m.held = held
	m.mu.Unlock()
	var once sync.Once
	return func() { once.Do(func() { close(held) }) }
}

func (m *schedulingModel) Start(context.Context) error { return nil }

func (m *schedulingModel) Create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	names := make([]string, 0, len(params.Tools))
	for _, tool := range params.Tools {
		names = append(names, tool.Name)
	}
	scheduled := strings.Contains(startedBy(params), "<scheduled_task")
	m.mu.Lock()
	m.offered = append(m.offered, names)
	m.asked = append(m.asked, params)
	failing := scheduled && m.failing > 0
	if failing {
		m.failing--
	}
	held := m.held
	if !scheduled {
		held = nil
	}
	m.mu.Unlock()

	// A turn somebody started runs what the test queued; the round a tool's
	// answer starts runs what the test queued for a later round, and is
	// otherwise simply agreed with, and so is a call offering no tools, such
	// as the summary of a thread's episode.
	var calls []llm.ToolCall
	if len(params.Tools) > 0 && len(params.Input) > 0 {
		switch params.Input[len(params.Input)-1].Role {
		case llm.User:
			calls = m.script.next()
		case llm.ToolResult:
			calls = m.script.nextRound()
		}
	}
	script := llmtest.New(llm.StreamOptions{ResponseID: params.ID, Provider: m.Provider(), Model: m.Model()})
	if failing {
		script.Fail(errors.New("the provider is down"), "scheduling")
		return script.Stream(), nil
	}
	if held != nil {
		go func() {
			select {
			case <-held:
			case <-ctx.Done():
			}
			script.OutputText("Done.")
			script.Done()
		}()
		return script.Stream(), nil
	}
	script.OutputText("Done.")
	if len(calls) > 0 {
		script.ToolCalls(calls...)
	}
	script.Done()
	return script.Stream(), nil
}

func (m *schedulingModel) Provider() string               { return "scheduling" }
func (m *schedulingModel) Model() string                  { return "scheduling-model" }
func (m *schedulingModel) Capabilities() llm.Capabilities { return llm.Capabilities{} }
func (m *schedulingModel) Close() error                   { return nil }

// toolsOffered is every tool name the model was offered, once each.
func (m *schedulingModel) toolsOffered() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	seen := map[string]bool{}
	var names []string
	for _, offered := range m.offered {
		for _, name := range offered {
			if !seen[name] {
				seen[name] = true
				names = append(names, name)
			}
		}
	}
	return names
}

// heard reports whether the model was ever sent text holding needle, in any
// turn: what a follow-up sent it is found this way.
func (m *schedulingModel) heard(needle string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, asked := range m.asked {
		for _, input := range asked.Input {
			if strings.Contains(input.Content, needle) {
				return true
			}
			for _, part := range input.Parts {
				if strings.Contains(part.Text, needle) {
					return true
				}
			}
		}
	}
	return false
}

// turnsOn counts the turns somebody started by saying text holding needle:
// how many times a follow-up was run, not how often history repeats it.
func (m *schedulingModel) turnsOn(needle string) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	turns := 0
	for _, asked := range m.asked {
		if strings.Contains(startedBy(asked), needle) {
			turns++
		}
	}
	return turns
}

// startedBy is what somebody said to start the turn params asks for, or
// nothing for a turn a tool's answer started.
func startedBy(params llm.ResponseParams) string {
	if len(params.Input) == 0 {
		return ""
	}
	last := params.Input[len(params.Input)-1]
	if last.Role != llm.User {
		return ""
	}
	said := last.Content
	for _, part := range last.Parts {
		said += part.Text
	}
	return said
}

// instructions is what the model was last told to be.
func (m *schedulingModel) instructions() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.asked) == 0 {
		return ""
	}
	return m.asked[len(m.asked)-1].Instructions
}

// ScheduledTasksSuite covers scheduled tasks end to end: the tools the agent
// schedules with, the endpoints a customer drives them by, and what arrives
// when one comes due.
type ScheduledTasksSuite struct {
	RouterSuite
	sink *channelSink
	// runReads counts what the router reads of scheduled_task_runs while on.
	runReads *runReads
	// worker is the test's running worker, if it started one.
	worker *scheduling.Scheduler
}

// runReads counts the queries a pool sends to scheduled_task_runs while on.
type runReads struct {
	on    atomic.Bool
	reads atomic.Int64
}

func (*runReads) BeforeQuery(ctx context.Context, _ *bun.QueryEvent) context.Context { return ctx }

func (r *runReads) AfterQuery(_ context.Context, event *bun.QueryEvent) {
	if r.on.Load() && strings.Contains(event.Query, "scheduled_task_runs") {
		r.reads.Add(1)
	}
}

func TestScheduledTasksSuite(t *testing.T) {
	runSuite(t, new(ScheduledTasksSuite))
}

func (s *ScheduledTasksSuite) SetupSuite() {
	s.sink = &channelSink{sent: map[string][]string{}}
	s.channelAPI = httptest.NewTLSServer(http.HandlerFunc(s.sink.serve))
	s.T().Cleanup(s.channelAPI.Close)
	// A worker claims every task due in its database, so the suite's workers and those of
	// the other suites and packages running beside it must not share one.
	s.database = "scheduled_tasks"
	s.RouterSuite.SetupSuite()
	s.runReads = &runReads{}
	if s.store != nil {
		s.store.DB().AddQueryHook(s.runReads)
	}
}

func (s *ScheduledTasksSuite) SetupTest() {
	s.useApp(s.data.createApp())
	// The test's tasks go with it, once its worker stopped, so none it leaves due is
	// claimed by the next test's worker.
	customer := s.customerID()
	s.T().Cleanup(func() {
		_, err := s.store.DB().NewDelete().Model((*store.ScheduledTask)(nil)).
			Where("customer_id = ?", customer).Exec(context.Background())
		s.Require().NoError(err)
	})
	s.scheduledMu.Lock()
	s.scheduled = nil
	s.scheduledMu.Unlock()
	s.schedulingScript.reset()
}

// agentWithScheduling is an agent config with scheduling on and the scheduling
// model answering, changed as a test says.
func (s *ScheduledTasksSuite) agentWithScheduling(change func(*AgentConfigRequest)) string {
	request := AgentConfigRequest{
		Name:       "agent-" + s.utils.uuid(),
		Llm:        pointerTo("scheduling/scheduling-model"),
		Scheduling: &AgentScheduling{Enabled: pointerTo(true)},
	}
	if change != nil {
		change(&request)
	}
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", request, &created))
	return created.Id
}

// agentWithoutScheduling is an agent config that never turned scheduling on.
func (s *ScheduledTasksSuite) agentWithoutScheduling() string {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name: "agent-" + s.utils.uuid(), Llm: pointerTo("scheduling/scheduling-model"),
	}, &created))
	return created.Id
}

// openScheduling opens a session on the config for the user, held in writing.
func (s *ScheduledTasksSuite) openScheduling(user *testClient, configID string, change func(*CreateSessionRequest)) Session {
	request := CreateSessionRequest{ConfigId: &configID}
	if change != nil {
		change(&request)
	}
	return user.createSession(request)
}

// watch opens the session's events as its user: a conversation that persists is
// only readable by the person whose it is.
func (s *ScheduledTasksSuite) watch(user *testClient, id string) *websocket.Conn {
	return user.opens("/v1/agents/sessions/" + id + "/events")
}

// say has the user say text in the session, through the backend acting in
// their name: putting words in a conversation is server-side whoever asks. A
// conversation that persists wants a command id, so the say is one command.
func (s *ScheduledTasksSuite) say(user *testClient, id, text string) {
	status, payload := s.serverClient.actingFor(user).call(http.MethodPost,
		"/v1/agents/sessions/"+id+"/respond",
		RespondRequest{Text: text, RequestId: pointerTo(s.utils.uuid())})
	s.Require().True(status == http.StatusOK || status == http.StatusNoContent, string(payload))
}

// model is the scheduling model the newest session opened.
func (s *ScheduledTasksSuite) model() *schedulingModel {
	s.scheduledMu.Lock()
	defer s.scheduledMu.Unlock()
	s.Require().NotEmpty(s.scheduled, "no session opened the scheduling model")
	return s.scheduled[len(s.scheduled)-1]
}

// tasksOf is every task the user holds, however it was made.
func (s *ScheduledTasksSuite) tasksOf(user *testClient) []store.ScheduledTask {
	tasks, err := s.store.ScheduledTasks(context.Background(), s.customerID(), store.ScheduledTaskFilter{
		Scope: store.ScheduledTaskScope{UserID: user.userID},
	}, 25, nil)
	s.Require().NoError(err)
	return tasks
}

// runWorker starts the scheduled-task worker over the suite's store,
// delivering through the suite's own router, as every router does.
func (s *ScheduledTasksSuite) runWorker() {
	worker := scheduling.NewScheduler(s.store, s.router.DeliverScheduledTask, slog.New(slog.DiscardHandler))
	worker.Start(context.Background())
	s.worker = worker
	s.router.SetSchedulerKick(worker.Kick)
	s.T().Cleanup(func() {
		s.router.SetSchedulerKick(nil)
		worker.Close()
		s.worker = nil
	})
}

// makeDue moves a task's next run into the past, so the worker claims it at
// its next look.
func (s *ScheduledTasksSuite) makeDue(task store.ScheduledTask) {
	past := time.Now().Add(-time.Minute)
	task.NextRunAt = &past
	s.Require().NoError(s.store.SaveScheduledTask(context.Background(), &task))
	// The endpoints kick the worker after a change; a write straight to the
	// store must too, or a running worker only looks again a lease later.
	if s.worker != nil {
		s.worker.Kick()
	}
}

// runsOf is the task's runs, newest first.
func (s *ScheduledTasksSuite) runsOf(task store.ScheduledTask) []store.ScheduledTaskRun {
	runs, err := s.store.ScheduledTaskRuns(context.Background(), s.customerID(), task.ID, 25, nil)
	s.Require().NoError(err)
	return runs
}

// taskReloaded is the task as the store holds it now.
func (s *ScheduledTasksSuite) taskReloaded(task store.ScheduledTask) store.ScheduledTask {
	stored, err := s.store.ScheduledTask(context.Background(), s.customerID(), task.ID)
	s.Require().NoError(err)
	return stored
}

// delivered waits for the task's latest run to be delivered, and returns it.
func (s *ScheduledTasksSuite) delivered(task store.ScheduledTask) store.ScheduledTaskRun {
	var latest store.ScheduledTaskRun
	s.Require().Eventually(func() bool {
		runs := s.runsOf(task)
		if len(runs) == 0 {
			return false
		}
		latest = runs[0]
		return latest.State == store.ScheduledTaskRunDelivered
	}, 30*time.Second, 250*time.Millisecond, "the occurrence was not delivered")
	return latest
}

// ranResult is the text a tool that ran in the watched session answered with.
func (s *ScheduledTasksSuite) ranResult(watching *websocket.Conn) string {
	ran := s.await(watching, "tool_ran")
	result, ok := ran["result"].(string)
	s.Require().True(ok, "the tool answered with %v", ran["result"])
	return result
}

func (s *ScheduledTasksSuite) TestTheAgentSchedulesATaskForItsUser() {
	config := s.agentWithScheduling(func(request *AgentConfigRequest) {
		request.Scheduling.DefaultTimezone = pointerTo("Europe/Amsterdam")
	})
	user := s.data.createUser()
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "schedule_task",
		Arguments: `{"title":"Stretch","instructions":"Remind them to stretch.","in":"PT2H"}`})
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)
	before := time.Now()
	s.say(user, opened.Id, "remind me to stretch in two hours")

	result := s.ranResult(watching)
	s.Contains(result, `"title":"Stretch"`)
	s.Contains(result, "Europe/Amsterdam")

	tasks := s.tasksOf(user)
	s.Require().Len(tasks, 1)
	task := tasks[0]
	s.Equal("Stretch", task.Title)
	s.Equal("Remind them to stretch.", task.Instructions)
	s.Equal(store.ScheduledTaskActive, task.State)
	s.Equal("agent", task.CreatedBy)
	s.Equal("authenticated", task.CallerKind)
	s.Equal(config, task.ConfigID)
	s.Equal(opened.Id, task.SessionID)
	s.Require().NotNil(task.NextRunAt)
	s.WithinDuration(before.Add(2*time.Hour), *task.NextRunAt, time.Minute)

	offered := s.model().toolsOffered()
	for _, name := range []string{"schedule_task", "list_scheduled_tasks", "update_scheduled_task", "cancel_scheduled_task"} {
		s.Contains(offered, name)
	}
}

func (s *ScheduledTasksSuite) TestTheAgentListsSnoozesAndCancelsItsTasks() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "schedule_task",
		Arguments: `{"title":"Stretch","instructions":"Remind them to stretch.","in":"PT2H","timezone":"Europe/Amsterdam"}`})
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)
	s.say(user, opened.Id, "remind me to stretch in two hours")
	s.ranResult(watching)
	s.await(watching, "responded")

	tasks := s.tasksOf(user)
	s.Require().Len(tasks, 1)
	id := tasks[0].ID

	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "list_scheduled_tasks", Arguments: `{}`})
	s.say(user, opened.Id, "what have I got coming up")
	result := s.ranResult(watching)
	s.Contains(result, id)
	s.Contains(result, "Stretch")
	s.await(watching, "responded")

	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "update_scheduled_task",
		Arguments: `{"id":"` + id + `","action":"snooze","in":"PT30M"}`})
	before := time.Now()
	s.say(user, opened.Id, "half an hour later, please")
	s.ranResult(watching)
	tasks = s.tasksOf(user)
	s.Require().Len(tasks, 1)
	s.Require().NotNil(tasks[0].NextRunAt)
	s.WithinDuration(before.Add(30*time.Minute), *tasks[0].NextRunAt, time.Minute)
	s.await(watching, "responded")

	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "cancel_scheduled_task",
		Arguments: `{"id":"` + id + `"}`})
	s.say(user, opened.Id, "never mind")
	result = s.ranResult(watching)
	s.Contains(result, `"cancelled":"Stretch"`)
	s.Empty(s.tasksOf(user))
}

func (s *ScheduledTasksSuite) TestTheAgentCannotSnoozeATaskSoonerThanAQuarterHour() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "schedule_task",
		Arguments: `{"title":"Stretch","instructions":"Remind them to stretch.","in":"PT2H","timezone":"Europe/Amsterdam"}`})
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)
	s.say(user, opened.Id, "remind me to stretch in two hours")
	s.ranResult(watching)
	s.await(watching, "responded")
	tasks := s.tasksOf(user)
	s.Require().Len(tasks, 1)

	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "update_scheduled_task",
		Arguments: `{"id":"` + tasks[0].ID + `","action":"snooze","in":"PT1M"}`})
	s.say(user, opened.Id, "do it again in a minute")
	s.Contains(s.ranResult(watching), `"error":"interval_too_short"`)
	s.Equal(*tasks[0].NextRunAt, *s.taskReloaded(tasks[0]).NextRunAt)
}

func (s *ScheduledTasksSuite) TestTheAgentCannotRescheduleATaskSoonerThanAQuarterHour() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, taskRequest(config))
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)

	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "update_scheduled_task",
		Arguments: `{"id":"` + created.Id + `","action":"reschedule","in":"PT1M"}`})
	s.say(user, opened.Id, "make it a minute from now")
	s.Contains(s.ranResult(watching), `"error":"interval_too_short"`)
	var read ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodGet, "/v1/agents/scheduled-tasks/"+created.Id, nil, &read))
	s.Equal(created.NextRunAt, read.NextRunAt)
}

func (s *ScheduledTasksSuite) TestTheAgentMovesATaskPastItsNextRunUnderAQuarterHourButNotBeforeIt() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	body := taskRequest(config)
	body.Schedule.In = pointerTo("PT5M")
	created := s.createTask(user, body)
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)
	update := func(arguments string) string {
		s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "update_scheduled_task",
			Arguments: `{"id":"` + created.Id + `",` + arguments + `}`})
		s.say(user, opened.Id, "change it")
		result := s.ranResult(watching)
		s.await(watching, "responded")
		return result
	}

	s.Contains(update(`"action":"snooze","in":"PT2M"`), `"error":"interval_too_short"`)
	s.Contains(update(`"action":"reschedule","in":"PT2M"`), `"error":"interval_too_short"`)

	s.NotContains(update(`"action":"snooze","in":"PT10M"`), `"error"`)
	s.WithinDuration(time.Now().Add(10*time.Minute), *s.taskReloaded(store.ScheduledTask{ID: created.Id}).NextRunAt, time.Minute)
	s.NotContains(update(`"action":"reschedule","in":"PT12M"`), `"error"`)
	s.WithinDuration(time.Now().Add(12*time.Minute), *s.taskReloaded(store.ScheduledTask{ID: created.Id}).NextRunAt, time.Minute)

	// A reschedule that names no time re-parses the task's own, and moves no run.
	s.NotContains(update(`"action":"reschedule","title":"Walk"`), `"error"`)
	s.Equal("Walk", s.taskReloaded(store.ScheduledTask{ID: created.Id}).Title)
}

func (s *ScheduledTasksSuite) seedListedTask(config string, user *testClient, state string) {
	task := store.ScheduledTask{
		CustomerID: s.customerID(), ConfigID: config, UserID: user.userID,
		CallerKind: string(auth.KindServer), Title: "Stretch", Instructions: "Remind them to stretch.",
		StartLocal: time.Date(2026, 10, 20, 9, 0, 0, 0, time.UTC), Timezone: "Europe/Amsterdam", State: state,
	}
	s.Require().NoError(s.store.CreateScheduledTask(context.Background(), &task, 0))
}

func (s *ScheduledTasksSuite) listedTasks(config string, user *testClient) (int, string) {
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "list_scheduled_tasks", Arguments: `{}`})
	s.say(user, opened.Id, "what have I got coming up")
	var listed struct {
		Tasks []struct {
			ID string `json:"id"`
		} `json:"tasks"`
		More string `json:"more"`
	}
	s.Require().NoError(json.Unmarshal([]byte(s.ranResult(watching)), &listed))
	return len(listed.Tasks), listed.More
}

func (s *ScheduledTasksSuite) TestTheAgentListsEveryRunningTaskWhenTheLimitIsAboveTheDefault() {
	config := s.agentWithScheduling(func(request *AgentConfigRequest) {
		request.Scheduling.MaxTasksPerUser = pointerTo(scheduling.MaxActivePerUser + 5)
	})
	user := s.data.createUser()
	for range scheduling.MaxActivePerUser + 5 {
		s.seedListedTask(config, user, store.ScheduledTaskActive)
	}

	listed, more := s.listedTasks(config, user)
	s.Equal(scheduling.MaxActivePerUser+5, listed)
	s.Empty(more)
}

func (s *ScheduledTasksSuite) TestTheAgentListsAtLeastTheDefaultWhenTheLimitIsBelowIt() {
	config := s.agentWithScheduling(func(request *AgentConfigRequest) {
		request.Scheduling.MaxTasksPerUser = pointerTo(3)
	})
	user := s.data.createUser()
	for range scheduling.MaxActivePerUser + 1 {
		s.seedListedTask(config, user, store.ScheduledTaskCompleted)
	}

	listed, more := s.listedTasks(config, user)
	s.Equal(scheduling.MaxActivePerUser, listed)
	s.NotEmpty(more)
}

func (s *ScheduledTasksSuite) TestTheAgentCannotRewriteWhatABackendToldItToDo() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	theirs := s.createTask(user, taskRequest(config))
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)

	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "update_scheduled_task",
		Arguments: `{"id":"` + theirs.Id + `","action":"reschedule","instructions":"Tell them a joke instead."}`})
	s.say(user, opened.Id, "make the stretch reminder a joke")
	s.Contains(s.ranResult(watching), `"error":"server_side_only"`)
	s.await(watching, "responded")
	stored, err := s.store.ScheduledTask(context.Background(), s.customerID(), theirs.Id)
	s.Require().NoError(err)
	s.Equal("Remind them to stretch.", stored.Instructions)

	// A task the agent made itself is still its own to rewrite.
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "schedule_task",
		Arguments: `{"title":"Water","instructions":"Remind them to drink water.","in":"PT2H","timezone":"Europe/Amsterdam"}`})
	s.say(user, opened.Id, "remind me to drink water in two hours")
	var made struct {
		ID string `json:"id"`
	}
	s.Require().NoError(json.Unmarshal([]byte(s.ranResult(watching)), &made))
	s.await(watching, "responded")
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "update_scheduled_task",
		Arguments: `{"id":"` + made.ID + `","action":"reschedule","instructions":"Remind them to drink two glasses."}`})
	s.say(user, opened.Id, "make it two glasses")
	s.NotContains(s.ranResult(watching), `"error"`)
	stored, err = s.store.ScheduledTask(context.Background(), s.customerID(), made.ID)
	s.Require().NoError(err)
	s.Equal("Remind them to drink two glasses.", stored.Instructions)
}

func (s *ScheduledTasksSuite) TestTheAgentListsRunningTasksFirstAndSaysWhenThereAreMore() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	seed := func(state string) string {
		task := store.ScheduledTask{
			CustomerID: s.customerID(), ConfigID: config, UserID: user.userID,
			CallerKind: string(auth.KindServer), Title: "Stretch", Instructions: "Remind them to stretch.",
			StartLocal: time.Date(2026, 10, 20, 9, 0, 0, 0, time.UTC), Timezone: "Europe/Amsterdam", State: state,
		}
		s.Require().NoError(s.store.CreateScheduledTask(context.Background(), &task, 0))
		return task.ID
	}
	// The running ones are the oldest, so newest first alone would bury them.
	running := []string{seed(store.ScheduledTaskActive), seed(store.ScheduledTaskPaused)}
	for range scheduling.MaxActivePerUser {
		seed(store.ScheduledTaskCompleted)
	}
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)

	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "list_scheduled_tasks", Arguments: `{}`})
	s.say(user, opened.Id, "what have I got coming up")
	var listed struct {
		Tasks []struct {
			ID string `json:"id"`
		} `json:"tasks"`
		More string `json:"more"`
	}
	s.Require().NoError(json.Unmarshal([]byte(s.ranResult(watching)), &listed))
	s.Require().Len(listed.Tasks, scheduling.MaxActivePerUser)
	s.Equal(running, []string{listed.Tasks[0].ID, listed.Tasks[1].ID})
	s.NotEmpty(listed.More)
}

func (s *ScheduledTasksSuite) TestAnotherConversationWithTheAgentSeesTheTaskAndAnotherUserDoesNot() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "schedule_task",
		Arguments: `{"title":"Stretch","instructions":"Remind them to stretch.","in":"PT2H","timezone":"Europe/Amsterdam"}`})
	first := s.openScheduling(user, config, nil)
	watching := s.watch(user, first.Id)
	s.say(user, first.Id, "remind me to stretch in two hours")
	s.ranResult(watching)

	second := s.openScheduling(user, config, nil)
	secondWatch := s.watch(user, second.Id)
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "list_scheduled_tasks", Arguments: `{}`})
	s.say(user, second.Id, "what have I got coming up")
	s.Contains(s.ranResult(secondWatch), "Stretch")

	other := s.data.createUser()
	theirs := s.openScheduling(other, config, nil)
	theirWatch := s.watch(other, theirs.Id)
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "list_scheduled_tasks", Arguments: `{}`})
	s.say(other, theirs.Id, "what have I got coming up")
	s.NotContains(s.ranResult(theirWatch), "Stretch")
}

func (s *ScheduledTasksSuite) TestATaskIsRefusedInWordsWhenNoZoneIsKnown() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "schedule_task",
		Arguments: `{"title":"Stretch","instructions":"Remind them to stretch.","in":"PT2H"}`})
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)
	s.say(user, opened.Id, "remind me to stretch in two hours")

	result := s.ranResult(watching)
	s.Contains(result, `"error":"timezone_unknown"`)
	s.Contains(result, "ask which city")
	s.Empty(s.tasksOf(user))
}

func (s *ScheduledTasksSuite) TestTheAgentIsToldWhichRepeatsAreSupported() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "schedule_task",
		Arguments: `{"title":"Stretch","instructions":"Remind them to stretch.","at":"2027-01-04T09:00",` +
			`"repeat":"FREQ=YEARLY;BYDAY=1MO","timezone":"Europe/Amsterdam"}`})
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)
	s.say(user, opened.Id, "remind me to stretch on the first Monday of every year")

	result := s.ranResult(watching)
	s.Contains(result, `"error":"invalid_repeat"`)
	s.Contains(result, "FREQ=YEARLY alone, on the start's date")
	s.Empty(s.tasksOf(user))
}

func (s *ScheduledTasksSuite) TestTheAgentIsToldWhenTheUserHasTooManyTasks() {
	config := s.agentWithScheduling(func(request *AgentConfigRequest) {
		request.Scheduling.MaxTasksPerUser = pointerTo(1)
	})
	user := s.data.createUser()
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "schedule_task",
		Arguments: `{"title":"Stretch","instructions":"Remind them to stretch.","in":"PT2H","timezone":"Europe/Amsterdam"}`})
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)
	s.say(user, opened.Id, "remind me to stretch in two hours")
	s.ranResult(watching)
	s.await(watching, "responded")

	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "schedule_task",
		Arguments: `{"title":"Drink water","instructions":"Remind them to drink water.","in":"PT3H","timezone":"Europe/Amsterdam"}`})
	s.say(user, opened.Id, "and water in three")
	s.Contains(s.ranResult(watching), `"error":"too_many_tasks"`)
	s.Len(s.tasksOf(user), 1)
}

func (s *ScheduledTasksSuite) TestTheAgentIsToldWhatDayItIsForTheUser() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	opened := s.openScheduling(user, config, func(request *CreateSessionRequest) {
		request.Timezone = pointerTo("Europe/Amsterdam")
	})
	watching := s.watch(user, opened.Id)
	s.say(user, opened.Id, "hello")
	s.await(watching, "responded")

	s.Contains(s.model().instructions(), "Today is ")
	s.Contains(s.model().instructions(), "(Europe/Amsterdam)")
}

func (s *ScheduledTasksSuite) TestAPaddedSessionZoneIsTrimmedAndTheRoutersOwnIsRefused() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	opened := s.openScheduling(user, config, func(request *CreateSessionRequest) {
		request.Timezone = pointerTo(" Europe/Amsterdam ")
	})
	watching := s.watch(user, opened.Id)
	s.say(user, opened.Id, "hello")
	s.await(watching, "responded")
	s.Contains(s.model().instructions(), "(Europe/Amsterdam)")

	status, answered := s.taskFailure(user, http.MethodPost, "/v1/agents/sessions",
		CreateSessionRequest{ConfigId: &config, Timezone: pointerTo("Local")})
	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeInvalidTimezone, answered.Error.Code)
}

func (s *ScheduledTasksSuite) TestGuestsAndAnonymousCallersAreNotOfferedTheTools() {
	config := s.agentWithScheduling(nil)
	for _, user := range []*testClient{s.guestClient, s.anonymousClient} {
		s.Run(string(user.kind), func() {
			opened := s.openScheduling(user, config, nil)
			watching := s.watch(user, opened.Id)
			s.say(user, opened.Id, "remind me to stretch later")
			s.await(watching, "responded")
			offered := s.model().toolsOffered()
			for _, name := range []string{"schedule_task", "list_scheduled_tasks", "update_scheduled_task", "cancel_scheduled_task"} {
				s.NotContains(offered, name)
			}
		})
	}
}

func (s *ScheduledTasksSuite) TestAnIncognitoSessionIsNotOfferedTheTools() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	opened := s.openScheduling(user, config, func(request *CreateSessionRequest) {
		request.Incognito = pointerTo(true)
	})
	watching := s.watch(user, opened.Id)
	s.say(user, opened.Id, "hello")
	s.await(watching, "responded")

	s.NotContains(s.model().toolsOffered(), "schedule_task")
}

func (s *ScheduledTasksSuite) TestAnAgentWithoutSchedulingOnIsNotOfferedTheTools() {
	config := s.agentWithoutScheduling()
	user := s.data.createUser()
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)
	s.say(user, opened.Id, "hello")
	s.await(watching, "responded")

	offered := s.model().toolsOffered()
	for _, name := range []string{"schedule_task", "list_scheduled_tasks", "update_scheduled_task", "cancel_scheduled_task"} {
		s.NotContains(offered, name)
	}
}

func (s *ScheduledTasksSuite) TestADueTaskFollowsUpInTheConversationItWasScheduledIn() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "schedule_task",
		Arguments: `{"title":"Stretch","instructions":"Remind them to stretch.","in":"PT2H","timezone":"Europe/Amsterdam"}`})
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)
	s.say(user, opened.Id, "remind me to stretch in two hours")
	s.ranResult(watching)

	tasks := s.tasksOf(user)
	s.Require().Len(tasks, 1)
	s.makeDue(tasks[0])
	s.runWorker()

	run := s.delivered(tasks[0])
	s.Equal(store.ScheduledTaskDeliveredConversation, run.DeliveredTo)
	s.Equal(opened.Id, run.SessionID)
	// Only the pointer was moved, not the schedule, so the occurrence the
	// task was made for still stands.
	standing := s.taskReloaded(tasks[0])
	s.Equal(store.ScheduledTaskActive, standing.State)
	s.Require().NotNil(standing.NextRunAt)
	s.True(standing.NextRunAt.After(time.Now()), "the made occurrence still stands")

	model := s.model()
	s.True(model.heard("<scheduled_task"), "the model was never sent the task")
	s.True(model.heard("Remind them to stretch."), "the model was never sent the instructions")
	s.True(model.heard("not a message from the user"), "the model was not told what the follow-up is")
}

// The worker sleeps up to a lease (a minute) when nothing is due sooner; a
// task the agent makes or changes wakes it, so one due in seconds is
// delivered in seconds, well inside that minute.
func (s *ScheduledTasksSuite) TestATaskTheAgentMakesDueInSecondsIsDeliveredWithoutWaitingForTheWorker() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	s.runWorker()
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "schedule_task",
		Arguments: `{"title":"Stretch","instructions":"Remind them to stretch.","in":"PT3S","timezone":"UTC"}`})
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)
	s.say(user, opened.Id, "remind me to stretch in three seconds")
	s.ranResult(watching)

	tasks := s.tasksOf(user)
	s.Require().Len(tasks, 1)
	s.deliveredWithin(tasks[0], 20*time.Second)
}

func (s *ScheduledTasksSuite) TestATaskTheAgentResumesDueInSecondsIsDeliveredWithoutWaitingForTheWorker() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	s.runWorker()
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "schedule_task",
		Arguments: `{"title":"Stretch","instructions":"Remind them to stretch.","in":"PT2H","timezone":"UTC"}`})
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)
	s.say(user, opened.Id, "remind me to stretch in two hours")
	s.ranResult(watching)
	s.await(watching, "responded")
	tasks := s.tasksOf(user)
	s.Require().Len(tasks, 1)
	// Paused, its one time seconds away, written straight to the store so
	// nothing wakes the worker but the resume.
	soon := time.Now().UTC().Add(5 * time.Second).Truncate(time.Second)
	_, err := s.store.DB().NewUpdate().Model((*store.ScheduledTask)(nil)).
		Set("state = ?", store.ScheduledTaskPaused).Set("start_local = ?", soon).Set("next_run_at = ?", soon).
		Where("customer_id = ?", s.customerID()).Where("id = ?", tasks[0].ID).
		Exec(context.Background())
	s.Require().NoError(err)

	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "update_scheduled_task",
		Arguments: `{"id":"` + tasks[0].ID + `","action":"resume"}`})
	s.say(user, opened.Id, "go ahead after all")
	s.ranResult(watching)
	s.deliveredWithin(tasks[0], 20*time.Second)
}

// deliveredWithin waits up to within for the task's latest run to be delivered.
func (s *ScheduledTasksSuite) deliveredWithin(task store.ScheduledTask, within time.Duration) {
	s.Eventually(func() bool {
		runs := s.runsOf(task)
		return len(runs) > 0 && runs[0].State == store.ScheduledTaskRunDelivered
	}, within, 250*time.Millisecond, "the occurrence was not delivered within %s", within)
}

func (s *ScheduledTasksSuite) TestATaskWithNoSessionYetOpensOneAtItsFirstOccurrence() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	now := time.Now()
	next := now.Add(-time.Minute)
	task := store.ScheduledTask{
		CustomerID: s.customerID(), ConfigID: config, UserID: user.userID,
		CallerKind: "authenticated", Title: "Stretch",
		Instructions: "Remind them to stretch.",
		StartLocal:   now.UTC(), Timezone: "Europe/Amsterdam",
		NextRunAt: &next, State: store.ScheduledTaskActive, CreatedBy: "api",
	}
	s.Require().NoError(s.store.CreateScheduledTask(context.Background(), &task, 25))
	s.runWorker()

	run := s.delivered(task)
	opened := s.taskReloaded(task)
	s.NotEmpty(opened.SessionID, "the first occurrence opened no session")
	s.Equal(opened.SessionID, run.SessionID)
	s.True(s.model().heard("Remind them to stretch."), "the model was never sent the instructions")
}

func (s *ScheduledTasksSuite) TestATaskThatComesDueOnACallWaitsForTheCallToEnd() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	opened := s.openScheduling(user, config, func(request *CreateSessionRequest) {
		request.StartVoice = pointerTo(true)
	})
	now := time.Now()
	next := now.Add(-time.Minute)
	task := store.ScheduledTask{
		CustomerID: s.customerID(), ConfigID: config, UserID: user.userID,
		CallerKind: "authenticated", SessionID: opened.Id, Title: "Stretch",
		Instructions: "Remind them to stretch.",
		StartLocal:   now.UTC(), Timezone: "Europe/Amsterdam",
		NextRunAt: &next, State: store.ScheduledTaskActive, CreatedBy: "api",
	}
	s.Require().NoError(s.store.CreateScheduledTask(context.Background(), &task, 25))
	s.runWorker()

	// The worker claims the occurrence, but nothing reaches the model while
	// the call is live: a reminder is never spoken over the user.
	s.Never(func() bool { return s.model().heard("<scheduled_task") },
		5*time.Second, 250*time.Millisecond, "the task interrupted the call")

	s.Require().Equal(http.StatusOK, user.do(http.MethodDelete, "/v1/agents/sessions/"+opened.Id+"/voice", nil, nil))
	s.delivered(task)
	s.True(s.model().heard("Remind them to stretch."))
}

func (s *ScheduledTasksSuite) TestADueTaskIsHandedToTheNodeHoldingItsConversation() {
	other, _ := s.otherRouter(nil)
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	// The conversation lives on the other node; this node's worker claims
	// the task all the same, and hands the occurrence over.
	opened := user.on(other).createSession(CreateSessionRequest{ConfigId: &config})
	now := time.Now()
	next := now.Add(-time.Minute)
	task := store.ScheduledTask{
		CustomerID: s.customerID(), ConfigID: config, UserID: user.userID,
		CallerKind: "authenticated", SessionID: opened.Id, Title: "Stretch",
		Instructions: "Remind them to stretch.",
		StartLocal:   now.UTC(), Timezone: "Europe/Amsterdam",
		NextRunAt: &next, State: store.ScheduledTaskActive, CreatedBy: "api",
	}
	s.Require().NoError(s.store.CreateScheduledTask(context.Background(), &task, 25))
	s.runWorker()

	run := s.delivered(task)
	s.Equal(opened.Id, run.SessionID)
	s.True(s.model().heard("Remind them to stretch."), "the holding node never ran the follow-up")
}

func (s *ScheduledTasksSuite) TestDeletingASessionDeletesItsTasks() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "schedule_task",
		Arguments: `{"title":"Stretch","instructions":"Remind them to stretch.","in":"PT2H","timezone":"Europe/Amsterdam"}`})
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)
	s.say(user, opened.Id, "remind me to stretch in two hours")
	s.ranResult(watching)
	s.Require().Len(s.tasksOf(user), 1)

	s.Require().Equal(http.StatusNoContent, user.do(http.MethodDelete, "/v1/agents/sessions/"+opened.Id, nil, nil))
	s.Empty(s.tasksOf(user), "the deleted conversation's tasks went with it")
}

func (s *ScheduledTasksSuite) TestDeletingAConfigPausesItsTasks() {
	deleted, kept := s.agentWithScheduling(nil), s.agentWithScheduling(nil)
	user := s.data.createUser()
	orphaned := s.createTask(user, taskRequest(deleted))
	other := s.createTask(user, taskRequest(kept))

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/configs/"+deleted, nil, nil))

	states := map[string]string{}
	for _, task := range s.tasksOf(user) {
		states[task.ID] = task.State
	}
	s.Equal(map[string]string{orphaned.Id: store.ScheduledTaskPaused, other.Id: store.ScheduledTaskActive}, states,
		"a task of a deleted agent has nothing to run on, and stops counting against the user's limit")
}

func (s *ScheduledTasksSuite) TestATaskOfADeletedConfigCannotBeResumedOverTheAPI() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, taskRequest(config))
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/configs/"+config, nil, nil))

	status, failure := user.failure(http.MethodPost, "/v1/agents/scheduled-tasks/"+created.Id+"/resume", nil)
	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, errUnknownConfig.Message)
	s.Equal(store.ScheduledTaskPaused, s.taskReloaded(store.ScheduledTask{ID: created.Id}).State)
}

func (s *ScheduledTasksSuite) TestAConfigThatCannotBeReadIsNotReportedAsDeletedOnResume() {
	// A task with no config id makes the config lookup fail without the config
	// being gone: that is a failure of the lookup, not an unknown config.
	user := s.data.createUser()
	task := store.ScheduledTask{
		CustomerID: s.customerID(), UserID: user.userID,
		CallerKind: string(auth.KindServer), Title: "Stretch", Instructions: "Remind them to stretch.",
		StartLocal: time.Now().UTC().Add(48 * time.Hour).Truncate(time.Second), Timezone: "Europe/Amsterdam",
		State: store.ScheduledTaskPaused,
	}
	s.Require().NoError(s.store.CreateScheduledTask(context.Background(), &task, 0))

	var rendered ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, "/v1/agents/scheduled-tasks/"+task.ID+"/resume", nil, &rendered))
	s.Equal(store.ScheduledTaskActive, rendered.State)
}

func (s *ScheduledTasksSuite) TestTheAgentCannotResumeATaskOfAConfigDeletedUnderItsSession() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, taskRequest(config))
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/configs/"+config, nil, nil))

	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "update_scheduled_task",
		Arguments: `{"id":"` + created.Id + `","action":"resume"}`})
	s.say(user, opened.Id, "turn it back on")
	s.Contains(s.ranResult(watching), `"error":"unknown_config"`)
	s.Equal(store.ScheduledTaskPaused, s.taskReloaded(store.ScheduledTask{ID: created.Id}).State)
}

func (s *ScheduledTasksSuite) TestTheAgentCannotRescheduleAFailedTaskOfAConfigDeletedUnderItsSession() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, taskRequest(config))
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/configs/"+config, nil, nil))
	failed := s.taskReloaded(store.ScheduledTask{ID: created.Id})
	failed.State = store.ScheduledTaskFailed
	s.Require().NoError(s.store.SaveScheduledTask(context.Background(), &failed))

	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "update_scheduled_task",
		Arguments: `{"id":"` + created.Id + `","action":"reschedule","in":"PT2H"}`})
	s.say(user, opened.Id, "try it again in two hours")
	s.Contains(s.ranResult(watching), `"error":"unknown_config"`)
	s.Equal(store.ScheduledTaskFailed, s.taskReloaded(store.ScheduledTask{ID: created.Id}).State)
}

func (s *ScheduledTasksSuite) TestADueTaskIsNotCountedOnItsUsersDay() {
	// The user agreed to the task once, not to each time it runs: the turn it
	// fires in their own conversation is the customer's spend, and the turn
	// they type next is theirs again.
	s.capTheDay(2)
	config := s.agentWithScheduling(func(request *AgentConfigRequest) {
		request.Llm = pointerTo("counted/counted-model")
	})
	user := s.data.createUser()
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, nil)
	s.makeDue(s.tasksOf(user)[0])
	s.runWorker()

	s.delivered(s.tasksOf(user)[0])
	s.awaitAnswers(1)
	s.Equal("2", s.leftOfTheDay(user), "the scheduled turn spent none of the user's day")

	s.respondAs(user, opened.Id, "and what else")
	s.awaitAnswers(2)
	s.Equal("1", s.leftOfTheDay(user), "the user's own turn after it is counted on their day")
}

func (s *ScheduledTasksSuite) TestAUsersOwnTurnInTheirConversationIsCountedOnTheirDay() {
	// Unchanged by scheduling being on: what the user types is theirs.
	s.capTheDay(2)
	config := s.agentWithScheduling(func(request *AgentConfigRequest) {
		request.Llm = pointerTo("counted/counted-model")
	})
	user := s.data.createUser()
	opened := s.openScheduling(user, config, nil)

	s.respondAs(user, opened.Id, "how much does a call cost")
	s.awaitAnswers(1)
	s.Equal("1", s.leftOfTheDay(user))
}

// capTheDay sets, through the app's own policy, the messages one of its users
// may ask for in a day.
func (s *ScheduledTasksSuite) capTheDay(messages int64) {
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/policies/app",
		map[string]any{"quota": map[string]any{"messages_per_user_per_day": messages}}, nil))
}

// leftOfTheDay is what is left of the user's day, as a session they open is
// told.
func (s *ScheduledTasksSuite) leftOfTheDay(user *testClient) string {
	response := user.raw(http.MethodPost, "/v1/agents/sessions", textSession(nil))
	s.Require().Equal(http.StatusCreated, response.StatusCode)
	return response.Header.Get("X-Quota-Remaining")
}

// respondAs has the user type text into their conversation with their own
// token.
func (s *ScheduledTasksSuite) respondAs(user *testClient, id, text string) {
	status, failure := user.failure(http.MethodPost, "/v1/agents/sessions/"+id+"/responses",
		CreateResponseRequest{Text: text, RequestId: pointerTo(s.utils.uuid())})
	s.Require().Equal(http.StatusAccepted, status, failure)
}

// awaitAnswers waits until the app's answers by the counted model have all been
// recorded; the stub model's rows are the sessions' own. A day is
// debited just before an answer's request row is, so a row means its debit,
// if there was going to be one, has landed.
func (s *ScheduledTasksSuite) awaitAnswers(answers int) {
	s.Require().Eventually(func() bool {
		var rows []store.Request
		err := s.store.ShardDB().NewSelect().Model(&rows).Where("customer_id = ?", s.customerID()).Where("model = ?", "counted-model").Scan(s.T().Context())
		return err == nil && len(rows) >= answers
	}, settleFor, 50*time.Millisecond, "%d answers were never recorded", answers)
}

// createTask has the backend schedule a task for the user, as a customer's
// own system does.
func (s *ScheduledTasksSuite) createTask(user *testClient, body CreateScheduledTaskRequest) ScheduledTask {
	var created ScheduledTask
	s.Require().Equal(http.StatusCreated,
		s.serverClient.actingFor(user).do(http.MethodPost, "/v1/agents/scheduled-tasks", body, &created))
	return created
}

// taskFailure is the status and the error envelope a task call was answered with.
func (s *ScheduledTasksSuite) taskFailure(client *testClient, method, path string, body any) (int, ErrorResponse) {
	status, raw := client.call(method, path, body)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(raw, &answered), string(raw))
	return status, answered
}

// taskRequest is a create body for a stretch reminder two hours out.
func taskRequest(config string) CreateScheduledTaskRequest {
	return CreateScheduledTaskRequest{
		ConfigId:     &config,
		Title:        "Stretch",
		Instructions: "Remind them to stretch.",
		Schedule: ScheduledTaskSchedule{
			In:       pointerTo("PT2H"),
			Timezone: pointerTo("Europe/Amsterdam"),
		},
	}
}

func (s *ScheduledTasksSuite) TestABackendSchedulesATaskForANamedUser() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()

	created := s.createTask(user, taskRequest(config))

	s.Equal(user.userID, created.UserId)
	s.Equal(config, created.ConfigId)
	s.Equal("api", created.CreatedBy)
	s.Equal(store.ScheduledTaskActive, created.State)
	s.Equal("Stretch", created.Title)
	s.Contains(created.Schedule.Description, "Once")
	s.Require().NotNil(created.NextRunAt)
	s.Contains(*created.NextRunAt, "+0", "the next run reads in the task's own zone")
	s.Len(created.Upcoming, 1)
	s.Equal(created.NextRunAt, &created.Upcoming[0])
	s.Nil(created.LastRunAt)

	tasks := s.tasksOf(user)
	s.Require().Len(tasks, 1)
	s.Equal(created.Id, tasks[0].ID)
	s.Equal("server", tasks[0].CallerKind)
}

func (s *ScheduledTasksSuite) TestCreatingATaskWithoutANamedUserIsRefused() {
	config := s.agentWithScheduling(nil)

	status, answered := s.taskFailure(s.serverClient, http.MethodPost, "/v1/agents/scheduled-tasks", taskRequest(config))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(answered.Error.Message, "X-Stream-User-Id")
}

func (s *ScheduledTasksSuite) TestADeviceCannotCreateOrRunATask() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()

	status, answered := s.taskFailure(user, http.MethodPost, "/v1/agents/scheduled-tasks", taskRequest(config))
	s.Equal(http.StatusForbidden, status)
	s.Equal(codeServerSideOnly, answered.Error.Code)

	created := s.createTask(user, taskRequest(config))
	status, answered = s.taskFailure(user, http.MethodPost, "/v1/agents/scheduled-tasks/"+created.Id+"/run", nil)
	s.Equal(http.StatusForbidden, status)
	s.Equal(codeServerSideOnly, answered.Error.Code)
}

func (s *ScheduledTasksSuite) TestACreateWithAUsedIdAnswersConflict() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	body := taskRequest(config)
	body.Id = pointerTo("booking-4821-reminder")
	s.createTask(user, body)

	status, answered := s.taskFailure(s.serverClient.actingFor(user), http.MethodPost, "/v1/agents/scheduled-tasks", body)

	s.Equal(http.StatusConflict, status)
	s.Equal("conflict", answered.Error.Code)
	s.Len(s.tasksOf(user), 1)
}

func (s *ScheduledTasksSuite) TestBadSchedulesAreRefusedWithTheirCodes() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	backend := s.serverClient.actingFor(user)

	for name, change := range map[string]func(*CreateScheduledTaskRequest){
		"a time that has passed": func(body *CreateScheduledTaskRequest) {
			body.Schedule = ScheduledTaskSchedule{At: pointerTo("2020-01-01T09:00:00"), Timezone: pointerTo("Europe/Amsterdam")}
		},
		"a zone nobody knows": func(body *CreateScheduledTaskRequest) {
			body.Schedule.Timezone = pointerTo("Nowhere/Special")
		},
		"no zone anywhere": func(body *CreateScheduledTaskRequest) {
			body.Schedule.Timezone = nil
		},
		"a repeat outside the subset": func(body *CreateScheduledTaskRequest) {
			body.Schedule.Repeat = pointerTo("FREQ=HOURLY")
		},
	} {
		s.Run(name, func() {
			body := taskRequest(config)
			change(&body)
			_, answered := s.taskFailure(backend, http.MethodPost, "/v1/agents/scheduled-tasks", body)
			s.Contains([]string{"schedule_in_past", "invalid_timezone", "timezone_unknown", "invalid_repeat"},
				answered.Error.Code, answered.Error.Message)
		})
	}
	s.Empty(s.tasksOf(user))
}

func (s *ScheduledTasksSuite) TestARepeatItsDescriptionWouldMisstateIsRefused() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	body := taskRequest(config)
	body.Schedule = ScheduledTaskSchedule{At: pointerTo("2026-01-05T09:00:00"), Repeat: pointerTo("FREQ=DAILY;BYDAY=MO,WE"),
		Timezone: pointerTo("Europe/Amsterdam")}
	status, answered := s.taskFailure(s.serverClient.actingFor(user), http.MethodPost, "/v1/agents/scheduled-tasks", body)
	s.Equal(http.StatusBadRequest, status)
	s.Equal(scheduling.CodeInvalidRepeat, answered.Error.Code)
	s.Contains(answered.Error.Message, "supported: FREQ=DAILY; FREQ=WEEKLY, alone or with BYDAY=MO,WE,FR")
	s.Empty(s.tasksOf(user))

	body.Schedule.Repeat = pointerTo("FREQ=WEEKLY;BYDAY=MO,WE")
	created := s.createTask(user, body)
	s.Equal("Weekly on Mon, Wed at 09:00 (Europe/Amsterdam)", created.Schedule.Description)

	status, answered = s.taskFailure(user, http.MethodPatch, "/v1/agents/scheduled-tasks/"+created.Id,
		UpdateScheduledTaskRequest{Schedule: &ScheduledTaskSchedule{At: pointerTo("2026-01-05T09:00:00"),
			Repeat: pointerTo("FREQ=YEARLY;BYDAY=1MO")}})
	s.Equal(http.StatusBadRequest, status)
	s.Equal(scheduling.CodeInvalidRepeat, answered.Error.Code)
	s.Equal("FREQ=WEEKLY;BYDAY=MO,WE", s.tasksOf(user)[0].RRule, "a refused change leaves the task as it was")
}

func (s *ScheduledTasksSuite) TestATaskStoredWithARepeatNowRefusedStillReadsBack() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	body := taskRequest(config)
	body.Schedule = ScheduledTaskSchedule{At: pointerTo("2026-01-05T09:00:00"), Repeat: pointerTo("FREQ=WEEKLY;BYDAY=MO,WE"),
		Timezone: pointerTo("Europe/Amsterdam")}
	created := s.createTask(user, body)
	task, err := s.store.ScheduledTask(context.Background(), s.customerID(), created.Id)
	s.Require().NoError(err)
	task.RRule = "FREQ=DAILY;BYDAY=MO,WE"
	s.Require().NoError(s.store.SaveScheduledTask(context.Background(), &task))

	var read ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodGet, "/v1/agents/scheduled-tasks/"+created.Id, nil, &read))
	s.Equal("Daily at 09:00 (Europe/Amsterdam)", read.Schedule.Description, "read as the base read it")
}

func (s *ScheduledTasksSuite) TestTooManyTasksAnswersConflict() {
	config := s.agentWithScheduling(func(request *AgentConfigRequest) {
		request.Scheduling.MaxTasksPerUser = pointerTo(1)
	})
	user := s.data.createUser()
	s.createTask(user, taskRequest(config))

	status, answered := s.taskFailure(s.serverClient.actingFor(user), http.MethodPost, "/v1/agents/scheduled-tasks", taskRequest(config))

	s.Equal(http.StatusConflict, status)
	s.Equal(codeTooManyTasks, answered.Error.Code)
}

func (s *ScheduledTasksSuite) TestTheOwnerReachesATaskItsBackendMadeAndAnotherUserDoesNot() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, taskRequest(config))

	var read ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodGet, "/v1/agents/scheduled-tasks/"+created.Id, nil, &read))
	s.Equal("Remind them to stretch.", read.Instructions, "the owner reads what will be said in their name")

	var page ScheduledTaskPage
	s.Require().Equal(http.StatusOK, user.do(http.MethodGet, "/v1/agents/scheduled-tasks", nil, &page))
	s.Require().Len(page.Items, 1)
	s.Equal(created.Id, page.Items[0].Id)

	other := s.data.createUser()
	status, answered := s.taskFailure(other, http.MethodGet, "/v1/agents/scheduled-tasks/"+created.Id, nil)
	s.Equal(http.StatusNotFound, status)
	s.Equal(codeScheduledTaskNotFound, answered.Error.Code)

	var theirs ScheduledTaskPage
	s.Require().Equal(http.StatusOK, other.do(http.MethodGet, "/v1/agents/scheduled-tasks", nil, &theirs))
	s.Empty(theirs.Items)
}

func (s *ScheduledTasksSuite) TestADeviceMayChangeTitleScheduleAndNextButNotInstructions() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, taskRequest(config))
	path := "/v1/agents/scheduled-tasks/" + created.Id

	var updated ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodPatch, path,
		UpdateScheduledTaskRequest{Title: pointerTo("Stretch break")}, &updated))
	s.Equal("Stretch break", updated.Title)

	status, answered := s.taskFailure(user, http.MethodPatch, path,
		UpdateScheduledTaskRequest{Instructions: pointerTo("Say something else entirely.")})
	s.Equal(http.StatusForbidden, status)
	s.Equal(codeServerSideOnly, answered.Error.Code)

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, path,
		UpdateScheduledTaskRequest{Instructions: pointerTo("Remind them to stretch and ask how the back is.")}, &updated))
	s.Equal("Remind them to stretch and ask how the back is.", updated.Instructions)

	before := time.Now()
	s.Require().Equal(http.StatusOK, user.do(http.MethodPatch, path,
		UpdateScheduledTaskRequest{Next: &ScheduledTaskNext{In: pointerTo("PT30M")}}, &updated))
	s.Require().NotNil(updated.NextRunAt)
	next, err := time.Parse(time.RFC3339, *updated.NextRunAt)
	s.Require().NoError(err)
	s.WithinDuration(before.Add(30*time.Minute), next, 2*time.Minute)
}

func (s *ScheduledTasksSuite) TestANextSoonerThanAQuarterHourIsRefused() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, taskRequest(config))
	path := "/v1/agents/scheduled-tasks/" + created.Id

	status, answered := s.taskFailure(user, http.MethodPatch, path,
		UpdateScheduledTaskRequest{Next: &ScheduledTaskNext{In: pointerTo("PT1M")}})
	s.Equal(http.StatusBadRequest, status)
	s.Equal(scheduling.CodeIntervalTooShort, answered.Error.Code)
	var read ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodGet, path, nil, &read))
	s.Equal(created.NextRunAt, read.NextRunAt)

	var updated ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodPatch, path,
		UpdateScheduledTaskRequest{Next: &ScheduledTaskNext{In: pointerTo("PT15M")}}, &updated))
	s.NotEqual(created.NextRunAt, updated.NextRunAt)
}

func (s *ScheduledTasksSuite) TestANextOrScheduleMovedUnderAQuarterHourIsRefusedOnlyWhenEarlierThanTheCurrentRun() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	body := taskRequest(config)
	body.Schedule.In = pointerTo("PT3M")
	created := s.createTask(user, body)
	path := "/v1/agents/scheduled-tasks/" + created.Id
	refuse := func(update UpdateScheduledTaskRequest) {
		status, answered := s.taskFailure(user, http.MethodPatch, path, update)
		s.Equal(http.StatusBadRequest, status)
		s.Equal(scheduling.CodeIntervalTooShort, answered.Error.Code)
		var read ScheduledTask
		s.Require().Equal(http.StatusOK, user.do(http.MethodGet, path, nil, &read))
		s.Equal(created.NextRunAt, read.NextRunAt)
	}

	refuse(UpdateScheduledTaskRequest{Schedule: &ScheduledTaskSchedule{In: pointerTo("PT1M")}})
	refuse(UpdateScheduledTaskRequest{Next: &ScheduledTaskNext{In: pointerTo("PT2M")}})

	// Later than the run due in three minutes, though under fifteen from now.
	var updated ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodPatch, path,
		UpdateScheduledTaskRequest{Next: &ScheduledTaskNext{In: pointerTo("PT10M")}}, &updated))
	s.Require().Equal(http.StatusOK, user.do(http.MethodPatch, path,
		UpdateScheduledTaskRequest{Schedule: &ScheduledTaskSchedule{In: pointerTo("PT12M")}}, &updated))
	s.Require().NotNil(updated.NextRunAt)
	next, err := time.Parse(time.RFC3339, *updated.NextRunAt)
	s.Require().NoError(err)
	s.WithinDuration(time.Now().Add(12*time.Minute), next, 2*time.Minute)
}

func (s *ScheduledTasksSuite) TestANextMovedWithTheScheduleIsCheckedAgainstTheRunBeforeTheUpdate() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	body := taskRequest(config)
	body.Schedule.In = pointerTo("PT5M")
	created := s.createTask(user, body)

	// Ten minutes is later than the run due in five, though earlier than the
	// twenty the same update gives the schedule.
	var updated ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodPatch, "/v1/agents/scheduled-tasks/"+created.Id,
		UpdateScheduledTaskRequest{Schedule: &ScheduledTaskSchedule{In: pointerTo("PT20M")},
			Next: &ScheduledTaskNext{In: pointerTo("PT10M")}}, &updated))
	s.Require().NotNil(updated.NextRunAt)
	next, err := time.Parse(time.RFC3339, *updated.NextRunAt)
	s.Require().NoError(err)
	s.WithinDuration(time.Now().Add(10*time.Minute), next, 2*time.Minute)
}

func (s *ScheduledTasksSuite) TestARepeatWrittenFromAPastDateStartsAtItsNextOccurrence() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	nextOf := func(task ScheduledTask, clock string) {
		s.Require().NotNil(task.NextRunAt)
		next, err := time.Parse(time.RFC3339, *task.NextRunAt)
		s.Require().NoError(err)
		s.True(next.After(time.Now()), "the next run %s is not in the past", next)
		s.True(next.Before(time.Now().Add(25*time.Hour)), "a daily task is due within a day")
		amsterdam, err := time.LoadLocation("Europe/Amsterdam")
		s.Require().NoError(err)
		s.Equal(clock, next.In(amsterdam).Format("15:04"))
	}
	body := taskRequest(config)
	body.Schedule = ScheduledTaskSchedule{At: pointerTo("2026-01-01T09:00:00"), Repeat: pointerTo("FREQ=DAILY"),
		Timezone: pointerTo("Europe/Amsterdam")}
	created := s.createTask(user, body)
	nextOf(created, "09:00")

	var updated ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodPatch, "/v1/agents/scheduled-tasks/"+created.Id,
		UpdateScheduledTaskRequest{Schedule: &ScheduledTaskSchedule{At: pointerTo("2026-02-01T10:00:00"),
			Repeat: pointerTo("FREQ=DAILY")}}, &updated))
	nextOf(updated, "10:00")
}

func (s *ScheduledTasksSuite) TestPauseAndResumeCountsFromNow() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	body := taskRequest(config)
	body.Schedule.Repeat = pointerTo("FREQ=DAILY")
	created := s.createTask(user, body)
	path := "/v1/agents/scheduled-tasks/" + created.Id

	var rendered ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, path+"/pause", nil, &rendered))
	s.Equal(store.ScheduledTaskPaused, rendered.State)

	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, path+"/resume", nil, &rendered))
	s.Equal(store.ScheduledTaskActive, rendered.State)
	s.Require().NotNil(rendered.NextRunAt)
	next, err := time.Parse(time.RFC3339, *rendered.NextRunAt)
	s.Require().NoError(err)
	s.True(next.After(time.Now()), "the resumed task counts from now")
	s.True(next.Before(time.Now().Add(25*time.Hour)), "a daily task is due within a day")
}

func (s *ScheduledTasksSuite) TestAOneShotResumedAfterItsTimeRecordsTheOccurrenceAsMissed() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, utcTaskRequest(config))
	path := "/v1/agents/scheduled-tasks/" + created.Id
	due := s.pausedAt(created.Id, time.Now().Add(-time.Hour))

	var resumed ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, path+"/resume", nil, &resumed))
	s.Equal(store.ScheduledTaskCompleted, resumed.State)
	s.Nil(resumed.NextRunAt)

	var page ScheduledTaskRunPage
	s.Require().Equal(http.StatusOK, user.do(http.MethodGet, path+"/runs", nil, &page))
	s.Require().Len(page.Items, 1, "the occurrence the pause slept through is recorded")
	run := page.Items[0]
	s.Equal(store.ScheduledTaskRunMissed, run.State)
	s.Equal(store.ScheduledTaskTriggerSchedule, run.Trigger)
	s.Equal(due.Format(time.RFC3339), run.ScheduledFor)
	s.Equal(pointerTo(scheduling.ReasonPaused), run.Error)
	s.Require().NotNil(run.FinishedAt)
	s.Equal(pointerTo(run.FinishedAt.UTC().Format(time.RFC3339)), resumed.LastRunAt,
		"the missed run is the task's last run")
}

func (s *ScheduledTasksSuite) TestAOneShotTheAgentResumesAfterItsTimeRecordsTheOccurrenceAsMissed() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, utcTaskRequest(config))
	due := s.pausedAt(created.Id, time.Now().Add(-time.Hour))
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)

	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "update_scheduled_task",
		Arguments: `{"id":"` + created.Id + `","action":"resume"}`})
	s.say(user, opened.Id, "turn it back on")
	answer := s.ranResult(watching)
	s.Contains(answer, `"state":"completed"`)
	s.Contains(answer, `"missed":"`+due.Format(time.RFC3339)+` passed while the task was paused, so it did not run"`)

	task := s.taskReloaded(store.ScheduledTask{ID: created.Id})
	s.Equal(store.ScheduledTaskCompleted, task.State)
	s.Nil(task.NextRunAt)
	runs := s.runsOf(task)
	s.Require().Len(runs, 1, "the occurrence the pause slept through is recorded")
	s.Equal(store.ScheduledTaskRunMissed, runs[0].State)
	s.True(due.Equal(runs[0].ScheduledFor), "the run is the occurrence at %s, not %s", due, runs[0].ScheduledFor)
	s.Equal(pointerTo(scheduling.ReasonPaused), runs[0].Error)
	var read ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodGet, "/v1/agents/scheduled-tasks/"+created.Id, nil, &read))
	s.NotNil(read.LastRunAt, "the missed run is the task's last run")
}

func (s *ScheduledTasksSuite) TestAOneShotTheAgentResumesLateNamesTheMissedTimeInTheTasksZone() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	body := taskRequest(config)
	body.Schedule.Timezone = pointerTo("Asia/Tokyo")
	created := s.createTask(user, body)
	due := s.pausedAt(created.Id, time.Now().Add(-time.Hour))
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)

	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "update_scheduled_task",
		Arguments: `{"id":"` + created.Id + `","action":"resume"}`})
	s.say(user, opened.Id, "turn it back on")
	answer := s.ranResult(watching)
	tokyo, err := time.LoadLocation("Asia/Tokyo")
	s.Require().NoError(err)
	local := due.In(tokyo).Format(time.RFC3339)
	s.Contains(local, "+09:00")
	s.Contains(answer, `"missed":"`+local+` passed while the task was paused, so it did not run"`)
}

func (s *ScheduledTasksSuite) TestAOneShotResumedAfterItsTimeCompletesAtTheLimitAndOfADeletedConfig() {
	limited := s.agentWithScheduling(func(request *AgentConfigRequest) {
		request.Scheduling.MaxTasksPerUser = pointerTo(1)
	})
	user := s.data.createUser()
	late := s.createTask(user, utcTaskRequest(limited))
	s.pausedAt(late.Id, time.Now().Add(-time.Hour))
	running := s.createTask(user, taskRequest(limited))

	var resumed ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, "/v1/agents/scheduled-tasks/"+late.Id+"/resume", nil, &resumed),
		"a task that cannot run again takes no place, so the limit is not asked")
	s.Equal(store.ScheduledTaskCompleted, resumed.State)
	s.Len(s.runsOf(store.ScheduledTask{ID: late.Id}), 1)

	// An active one-shot still due later takes a place: the limit holds.
	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, "/v1/agents/scheduled-tasks/"+running.Id+"/pause", nil, nil))
	s.createTask(user, taskRequest(limited))
	status, answered := s.taskFailure(user, http.MethodPost, "/v1/agents/scheduled-tasks/"+running.Id+"/resume", nil)
	s.Equal(http.StatusConflict, status)
	s.Equal(codeTooManyTasks, answered.Error.Code)

	config := s.agentWithScheduling(nil)
	orphaned := s.createTask(user, utcTaskRequest(config))
	s.pausedAt(orphaned.Id, time.Now().Add(-time.Hour))
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/configs/"+config, nil, nil))
	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, "/v1/agents/scheduled-tasks/"+orphaned.Id+"/resume", nil, &resumed),
		"a late one-shot of a deleted config is closed with its missed run")
	s.Equal(store.ScheduledTaskCompleted, resumed.State)
	s.Len(s.runsOf(store.ScheduledTask{ID: orphaned.Id}), 1)
}

func (s *ScheduledTasksSuite) TestResumingALateOneShotTwiceRecordsItsOccurrenceOnce() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, utcTaskRequest(config))
	path := "/v1/agents/scheduled-tasks/" + created.Id
	s.pausedAt(created.Id, time.Now().Add(-time.Hour))

	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, path+"/resume", nil, nil))
	status, _ := s.taskFailure(user, http.MethodPost, path+"/resume", nil)
	s.Equal(http.StatusConflict, status, "a completed task cannot be resumed")
	s.Len(s.runsOf(store.ScheduledTask{ID: created.Id}), 1)
}

func (s *ScheduledTasksSuite) TestAOneShotResumedBeforeItsTimeRecordsNoRun() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, utcTaskRequest(config))
	path := "/v1/agents/scheduled-tasks/" + created.Id
	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, path+"/pause", nil, nil))

	var resumed ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, path+"/resume", nil, &resumed))
	s.Equal(store.ScheduledTaskActive, resumed.State)
	s.Equal(created.NextRunAt, resumed.NextRunAt)
	s.Nil(resumed.LastRunAt)
	s.Empty(s.runsOf(store.ScheduledTask{ID: created.Id}))
}

func (s *ScheduledTasksSuite) TestARepeatingTaskResumedAfterItsTimesRecordsNoRunAndCountsFromNow() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	body := utcTaskRequest(config)
	body.Schedule.Repeat = pointerTo("FREQ=DAILY")
	created := s.createTask(user, body)
	s.pausedAt(created.Id, time.Now().Add(-72*time.Hour))

	var resumed ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, "/v1/agents/scheduled-tasks/"+created.Id+"/resume", nil, &resumed))
	s.Equal(store.ScheduledTaskActive, resumed.State)
	s.Require().NotNil(resumed.NextRunAt)
	next, err := time.Parse(time.RFC3339, *resumed.NextRunAt)
	s.Require().NoError(err)
	s.True(next.After(time.Now()), "the resumed task counts from now")
	s.True(next.Before(time.Now().Add(25*time.Hour)), "a daily task is due within a day")
	s.Empty(s.runsOf(store.ScheduledTask{ID: created.Id}), "a pause skips what it sleeps through")
}

func (s *ScheduledTasksSuite) TestARepeatingTaskPausedPastItsLastTimeIsRefusedOnResume() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	body := utcTaskRequest(config)
	body.Schedule.Repeat = pointerTo("FREQ=DAILY")
	created := s.createTask(user, body)
	due := s.pausedAt(created.Id, time.Now().Add(-72*time.Hour))
	until := due.Add(24 * time.Hour)
	_, err := s.store.DB().NewUpdate().Model((*store.ScheduledTask)(nil)).Set("until_date = ?", until).
		Where("customer_id = ?", s.customerID()).Where("id = ?", created.Id).
		Exec(context.Background())
	s.Require().NoError(err)

	s.refusedAsInThePast(user, created.Id)
	task := s.taskReloaded(store.ScheduledTask{ID: created.Id})
	s.Equal(store.ScheduledTaskPaused, task.State)
	s.Require().NotNil(task.NextRunAt)
	s.True(due.Equal(*task.NextRunAt), "next_run_at stays %s, not %s", due, *task.NextRunAt)
	s.Empty(s.runsOf(task), "a pause skips what it sleeps through")
}

func (s *ScheduledTasksSuite) TestAFailedRepeatPastItsLastTimeIsRefusedOnResume() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	body := utcTaskRequest(config)
	body.Schedule.Repeat = pointerTo("FREQ=DAILY")
	created := s.createTask(user, body)
	due := s.ranOut(created.Id, store.ScheduledTaskFailed, time.Now().Add(-72*time.Hour), 5)
	until := due.Add(24 * time.Hour)
	_, err := s.store.DB().NewUpdate().Model((*store.ScheduledTask)(nil)).Set("until_date = ?", until).
		Where("customer_id = ?", s.customerID()).Where("id = ?", created.Id).
		Exec(context.Background())
	s.Require().NoError(err)

	s.refusedAsInThePast(user, created.Id)
	task := s.taskReloaded(store.ScheduledTask{ID: created.Id})
	s.Equal(store.ScheduledTaskFailed, task.State)
	s.Nil(task.NextRunAt)
	s.Equal(5, task.ConsecutiveFailures, "the failure streak is not forgiven")
}

func (s *ScheduledTasksSuite) TestAFailedOneShotIsRefusedOnResumeUntilItIsRescheduled() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, utcTaskRequest(config))
	path := "/v1/agents/scheduled-tasks/" + created.Id
	due := s.ranOut(created.Id, store.ScheduledTaskFailed, time.Now().Add(-time.Hour), 1)
	failure := scheduling.ReasonFailed
	s.Require().NoError(s.store.InsertScheduledTaskRun(context.Background(), &store.ScheduledTaskRun{
		TaskID: created.Id, CustomerID: s.customerID(), ScheduledFor: due,
		State: store.ScheduledTaskRunFailed, Error: &failure, FinishedAt: &due,
	}))

	s.refusedAsInThePast(user, created.Id)
	task := s.taskReloaded(store.ScheduledTask{ID: created.Id})
	s.Equal(store.ScheduledTaskFailed, task.State, "a reminder that never arrived is not completed")
	s.Nil(task.NextRunAt)
	s.Equal(1, task.ConsecutiveFailures)
	runs := s.runsOf(task)
	s.Require().Len(runs, 1)
	s.Equal(store.ScheduledTaskRunFailed, runs[0].State)

	// What the refusal tells the caller to do works.
	s.Require().Equal(http.StatusOK, user.do(http.MethodPatch, path,
		UpdateScheduledTaskRequest{Schedule: &ScheduledTaskSchedule{In: pointerTo("PT2H")}}, nil))
	var resumed ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, path+"/resume", nil, &resumed))
	s.Equal(store.ScheduledTaskActive, resumed.State)
	s.NotNil(resumed.NextRunAt)
}

func (s *ScheduledTasksSuite) TestATaskWithNoOccurrenceLeftIsRefusedBeforeItsLimitOrItsConfigIsAsked() {
	limited := s.agentWithScheduling(func(request *AgentConfigRequest) {
		request.Scheduling.MaxTasksPerUser = pointerTo(1)
	})
	user := s.data.createUser()
	failed := s.createTask(user, utcTaskRequest(limited))
	s.ranOut(failed.Id, store.ScheduledTaskFailed, time.Now().Add(-time.Hour), 1)
	s.createTask(user, taskRequest(limited))
	s.refusedAsInThePast(user, failed.Id)
	s.Equal(store.ScheduledTaskFailed, s.taskReloaded(store.ScheduledTask{ID: failed.Id}).State)

	config := s.agentWithScheduling(nil)
	orphaned := s.createTask(user, utcTaskRequest(config))
	s.ranOut(orphaned.Id, store.ScheduledTaskFailed, time.Now().Add(-time.Hour), 1)
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/configs/"+config, nil, nil))
	s.refusedAsInThePast(user, orphaned.Id)
	s.Equal(store.ScheduledTaskFailed, s.taskReloaded(store.ScheduledTask{ID: orphaned.Id}).State)
}

func (s *ScheduledTasksSuite) TestResumingALateOneShotLeavesTheRunItsOccurrenceAlreadyHas() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, utcTaskRequest(config))
	due := s.pausedAt(created.Id, time.Now().Add(-time.Hour))
	// What a pause leaves of a run held by quiet hours: missed, with no reason.
	finished := due.Add(time.Minute)
	s.Require().NoError(s.store.InsertScheduledTaskRun(context.Background(), &store.ScheduledTaskRun{
		TaskID: created.Id, CustomerID: s.customerID(), ScheduledFor: due,
		State: store.ScheduledTaskRunMissed, FinishedAt: &finished,
	}))

	var resumed ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, "/v1/agents/scheduled-tasks/"+created.Id+"/resume", nil, &resumed))
	s.Equal(store.ScheduledTaskCompleted, resumed.State)
	runs := s.runsOf(store.ScheduledTask{ID: created.Id})
	s.Require().Len(runs, 1)
	s.Nil(runs[0].Error, "the run already there is kept as it was")
}

func (s *ScheduledTasksSuite) TestAOneShotPausedAfterItsOccurrenceFailedIsRefusedOnResume() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, utcTaskRequest(config))
	// What the worker leaves of a one-shot whose one occurrence failed, paused
	// after by the agent, which pauses a failed task too.
	due := s.ranOut(created.Id, store.ScheduledTaskPaused, time.Now().Add(-time.Hour), 1)
	failure := scheduling.ReasonFailed
	s.Require().NoError(s.store.InsertScheduledTaskRun(context.Background(), &store.ScheduledTaskRun{
		TaskID: created.Id, CustomerID: s.customerID(), ScheduledFor: due,
		State: store.ScheduledTaskRunFailed, Error: &failure, FinishedAt: &due,
	}))

	s.refusedAsInThePast(user, created.Id)
	task := s.taskReloaded(store.ScheduledTask{ID: created.Id})
	s.Equal(store.ScheduledTaskPaused, task.State)
	s.Equal(1, task.ConsecutiveFailures)
	runs := s.runsOf(task)
	s.Require().Len(runs, 1, "no missed run beside the failed one")
	s.Equal(store.ScheduledTaskRunFailed, runs[0].State)
}

func (s *ScheduledTasksSuite) TestTheAgentCannotResumeAnActiveOneShotPastItsTime() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, utcTaskRequest(config))
	// Due an hour ago and still active: the occurrence is the worker's.
	past := time.Now().UTC().Add(-time.Hour).Truncate(time.Second)
	_, err := s.store.DB().NewUpdate().Model((*store.ScheduledTask)(nil)).
		Set("start_local = ?", past).Set("next_run_at = ?", past).
		Where("customer_id = ?", s.customerID()).Where("id = ?", created.Id).
		Exec(context.Background())
	s.Require().NoError(err)
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)

	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "update_scheduled_task",
		Arguments: `{"id":"` + created.Id + `","action":"resume"}`})
	s.say(user, opened.Id, "turn it back on")
	s.Contains(s.ranResult(watching), `"error":"schedule_in_past"`)
	s.Equal(store.ScheduledTaskActive, s.taskReloaded(store.ScheduledTask{ID: created.Id}).State)
	s.Empty(s.runsOf(store.ScheduledTask{ID: created.Id}))
}

func (s *ScheduledTasksSuite) TestTheAgentCannotResumeAFailedOneShotPastItsTime() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, utcTaskRequest(config))
	s.ranOut(created.Id, store.ScheduledTaskFailed, time.Now().Add(-time.Hour), 1)
	opened := s.openScheduling(user, config, nil)
	watching := s.watch(user, opened.Id)

	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "update_scheduled_task",
		Arguments: `{"id":"` + created.Id + `","action":"resume"}`})
	s.say(user, opened.Id, "turn it back on")
	s.Contains(s.ranResult(watching), `"error":"schedule_in_past"`)
	task := s.taskReloaded(store.ScheduledTask{ID: created.Id})
	s.Equal(store.ScheduledTaskFailed, task.State)
	s.Equal(1, task.ConsecutiveFailures)
	s.Empty(s.runsOf(task))
}

// refusedAsInThePast resumes a task over the API and checks it was refused
// for having no occurrence left.
func (s *ScheduledTasksSuite) refusedAsInThePast(user *testClient, id string) {
	status, answered := s.taskFailure(user, http.MethodPost, "/v1/agents/scheduled-tasks/"+id+"/resume", nil)
	s.Equal(http.StatusBadRequest, status)
	s.Equal(scheduling.CodeScheduleInPast, answered.Error.Code)
}

// ranOut leaves a task as the worker does once its last occurrence, at due,
// is behind it: in state, with no next run and the failure streak given. It
// answers due as stored.
func (s *ScheduledTasksSuite) ranOut(id, state string, due time.Time, failures int) time.Time {
	due = due.UTC().Truncate(time.Second)
	_, err := s.store.DB().NewUpdate().Model((*store.ScheduledTask)(nil)).
		Set("state = ?", state).Set("start_local = ?", due).Set("next_run_at = NULL").
		Set("consecutive_failures = ?", failures).
		Where("customer_id = ?", s.customerID()).Where("id = ?", id).
		Exec(context.Background())
	s.Require().NoError(err)
	return due
}

// pausedAt pauses a task with its one time, or its first, at due, written
// straight to the store so the pause sleeps through it without waiting. It
// answers due as stored.
func (s *ScheduledTasksSuite) pausedAt(id string, due time.Time) time.Time {
	due = due.UTC().Truncate(time.Second)
	_, err := s.store.DB().NewUpdate().Model((*store.ScheduledTask)(nil)).
		Set("state = ?", store.ScheduledTaskPaused).Set("start_local = ?", due).Set("next_run_at = ?", due).
		Where("customer_id = ?", s.customerID()).Where("id = ?", id).
		Exec(context.Background())
	s.Require().NoError(err)
	return due
}

// utcTaskRequest is taskRequest in UTC, so a task's local start is its instant.
func utcTaskRequest(config string) CreateScheduledTaskRequest {
	body := taskRequest(config)
	body.Schedule.Timezone = pointerTo("UTC")
	return body
}

func (s *ScheduledTasksSuite) TestRunningATaskByHandDeliversWithoutMovingTheSchedule() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, taskRequest(config))
	path := "/v1/agents/scheduled-tasks/" + created.Id

	var run ScheduledTaskRun
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, path+"/run", nil, &run))
	s.Equal(store.ScheduledTaskTriggerManual, run.Trigger)
	s.Equal(created.Id, run.TaskId)

	s.Eventually(func() bool {
		runs, err := s.store.ScheduledTaskRuns(context.Background(), s.customerID(), created.Id, 25, nil)
		return err == nil && len(runs) == 1 && runs[0].State == store.ScheduledTaskRunDelivered
	}, 30*time.Second, 250*time.Millisecond, "the manual run was not delivered")

	var page ScheduledTaskRunPage
	s.Require().Equal(http.StatusOK, user.do(http.MethodGet, path+"/runs", nil, &page))
	s.Require().Len(page.Items, 1)
	s.Equal(store.ScheduledTaskDeliveredConversation, page.Items[0].DeliveredTo)
	s.Nil(page.Items[0].Error, "a run that landed lists no reason")
	s.True(s.model().heard("Remind them to stretch."), "the model was never sent the instructions")

	var read ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodGet, path, nil, &read))
	s.Equal(created.NextRunAt, read.NextRunAt, "running by hand does not move the schedule")
	s.NotNil(read.LastRunAt)
}

func (s *ScheduledTasksSuite) TestDeletingATaskDeletesItsRuns() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, taskRequest(config))
	path := "/v1/agents/scheduled-tasks/" + created.Id

	s.Require().Equal(http.StatusNoContent, user.do(http.MethodDelete, path, nil, nil))

	status, answered := s.taskFailure(user, http.MethodGet, path, nil)
	s.Equal(http.StatusNotFound, status)
	s.Equal(codeScheduledTaskNotFound, answered.Error.Code)
	status, _ = s.taskFailure(user, http.MethodGet, path+"/runs", nil)
	s.Equal(http.StatusNotFound, status)
	s.Empty(s.tasksOf(user))
}

func (s *ScheduledTasksSuite) TestListingPagesAndFilters() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	backend := s.serverClient.actingFor(user)
	made := make([]string, 0, 3)
	for range 3 {
		var created ScheduledTask
		s.Require().Equal(http.StatusCreated, backend.do(http.MethodPost, "/v1/agents/scheduled-tasks", taskRequest(config), &created))
		made = append(made, created.Id)
	}
	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, "/v1/agents/scheduled-tasks/"+made[0]+"/pause", nil, nil))

	var first ScheduledTaskPage
	s.Require().Equal(http.StatusOK, user.do(http.MethodGet, "/v1/agents/scheduled-tasks?limit=2", nil, &first))
	s.Require().Len(first.Items, 2)
	s.True(first.HasMore)
	s.Require().NotNil(first.NextCursor)

	var second ScheduledTaskPage
	s.Require().Equal(http.StatusOK, user.do(http.MethodGet,
		"/v1/agents/scheduled-tasks?limit=2&cursor="+*first.NextCursor, nil, &second))
	s.Len(second.Items, 1)
	s.False(second.HasMore)

	var paused ScheduledTaskPage
	s.Require().Equal(http.StatusOK, user.do(http.MethodGet, "/v1/agents/scheduled-tasks?state=paused", nil, &paused))
	s.Require().Len(paused.Items, 1)
	s.Equal(made[0], paused.Items[0].Id)

	var onConfig ScheduledTaskPage
	s.Require().Equal(http.StatusOK, user.do(http.MethodGet, "/v1/agents/scheduled-tasks?config_id="+config, nil, &onConfig))
	s.Len(onConfig.Items, 3)
}

func (s *ScheduledTasksSuite) TestACreateNamingASessionFollowsUpInIt() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	opened := s.openScheduling(user, config, nil)

	body := taskRequest(config)
	body.ConfigId, body.SessionId = nil, &opened.Id
	created := s.createTask(user, body)
	s.Equal(opened.Id, created.SessionId)
	s.Equal(config, created.ConfigId)

	other := s.data.createUser()
	body.SessionId = &opened.Id
	status, _ := s.taskFailure(s.serverClient.actingFor(other), http.MethodPost, "/v1/agents/scheduled-tasks", body)
	s.Equal(http.StatusNotFound, status, "a session is only its own user's to schedule into")
}

// An occurrence follows up by carrying the conversation on, so a create naming a session
// that keeps none is refused rather than failing at every occurrence until it pauses.
func (s *ScheduledTasksSuite) TestACreateNamingASessionThatKeepsNoConversationIsRefused() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	incognito := s.openScheduling(user, config, func(request *CreateSessionRequest) {
		request.Incognito = pointerTo(true)
	})
	unkept := store.AgentSession{ID: s.utils.uuid(), CustomerID: s.customerID(), UserID: user.userID, ConfigID: config}
	s.Require().NoError(s.store.SaveSession(context.Background(), &unkept))

	for name, id := range map[string]string{"incognito": incognito.Id, "a row with no conversation": unkept.ID} {
		s.Run(name, func() {
			body := taskRequest(config)
			body.ConfigId, body.SessionId = nil, &id
			status, answered := s.taskFailure(s.serverClient.actingFor(user), http.MethodPost, "/v1/agents/scheduled-tasks", body)
			s.Equal(http.StatusBadRequest, status)
			s.Contains(answered.Error.Message, "not kept")
		})
	}
	s.Empty(s.tasksOf(user))
}

// Huma's minLength counts spaces; a title or instructions of nothing else are refused as
// the caller's mistake, on create and on update.
func (s *ScheduledTasksSuite) TestATitleOrInstructionsOfOnlySpacesAreRefused() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	backend := s.serverClient.actingFor(user)

	for name, change := range map[string]func(*CreateScheduledTaskRequest){
		"title":        func(body *CreateScheduledTaskRequest) { body.Title = "   " },
		"instructions": func(body *CreateScheduledTaskRequest) { body.Instructions = " \n " },
	} {
		s.Run(name, func() {
			body := taskRequest(config)
			change(&body)
			status, answered := s.taskFailure(backend, http.MethodPost, "/v1/agents/scheduled-tasks", body)
			s.Equal(http.StatusBadRequest, status)
			s.Contains(answered.Error.Message, name+" cannot be empty")
		})
	}
	s.Empty(s.tasksOf(user))

	created := s.createTask(user, taskRequest(config))
	for _, instructions := range []string{"", "  "} {
		status, answered := s.taskFailure(backend, http.MethodPatch, "/v1/agents/scheduled-tasks/"+created.Id,
			UpdateScheduledTaskRequest{Instructions: &instructions})
		s.Equal(http.StatusBadRequest, status)
		s.Contains(answered.Error.Message, "instructions cannot be empty")
	}
	s.Equal("Remind them to stretch.", s.tasksOf(user)[0].Instructions)
}

// A device that names nobody has no tasks of its own, so it lists none: an empty user id
// must not read as every user. A verified user still lists their own, and a backend
// naming nobody still lists the customer's.
func (s *ScheduledTasksSuite) TestADeviceNamingNobodyListsNoTasks() {
	config := s.agentWithScheduling(nil)
	user, other := s.data.createUser(), s.data.createUser()
	mine := s.createTask(user, taskRequest(config))
	theirs := s.createTask(other, taskRequest(config))

	var nameless ScheduledTaskPage
	s.Require().Equal(http.StatusOK, s.signedIn(jwt.MapClaims{}, auth.AuthTypeAnonymous, anonymous, "").
		do(http.MethodGet, "/v1/agents/scheduled-tasks", nil, &nameless))
	s.Empty(nameless.Items)

	var own ScheduledTaskPage
	s.Require().Equal(http.StatusOK, user.do(http.MethodGet, "/v1/agents/scheduled-tasks", nil, &own))
	s.Equal([]string{mine.Id}, taskIDs(own))

	var all ScheduledTaskPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/scheduled-tasks", nil, &all))
	s.ElementsMatch([]string{mine.Id, theirs.Id}, taskIDs(all))
}

func (s *ScheduledTasksSuite) TestListingByAStateThatIsNoneIsRefused() {
	user := s.data.createUser()

	status, answered := s.taskFailure(user, http.MethodGet, "/v1/agents/scheduled-tasks?state=cancelled", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeValidationFailed, answered.Error.Code)
}

func taskIDs(page ScheduledTaskPage) []string {
	ids := make([]string, 0, len(page.Items))
	for _, task := range page.Items {
		ids = append(ids, task.Id)
	}
	return ids
}

func (s *ScheduledTasksSuite) TestACallersOwnScheduleTaskKeepsTheName() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: "schedule_task",
		Arguments: `{"title":"Stretch","instructions":"Remind them to stretch.","in":"PT2H","timezone":"Europe/Amsterdam"}`})
	opened := s.openScheduling(user, config, func(request *CreateSessionRequest) {
		request.Tools = &[]SessionTool{{
			Name:        "schedule_task",
			Description: "the caller's own scheduling",
			Parameters:  pointerTo(map[string]any{"type": "object", "properties": map[string]any{}}),
		}}
	})
	watching := s.watch(user, opened.Id)
	s.say(user, opened.Id, "remind me to stretch in two hours")

	asked := s.await(watching, "tool_call")
	s.Equal("schedule_task", asked["name"])
	s.Empty(s.tasksOf(user), "the caller's own tool was asked, not the router's")
}

// channelSink stands in for the channel providers' APIs, Meta's and
// Telnyx's both, keeping the text of every message sent to it, by who it
// was sent to.
type channelSink struct {
	mu   sync.Mutex
	sent map[string][]string
	// failing is how many more sends to a person the provider refuses.
	failing map[string]int
}

// fail has the provider refuse the next times sends to number.
func (c *channelSink) fail(number string, times int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.failing == nil {
		c.failing = map[string]int{}
	}
	c.failing[number] = times
}

func (c *channelSink) serve(w http.ResponseWriter, r *http.Request) {
	var body struct {
		To   string          `json:"to"`
		Text json.RawMessage `json:"text"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)
	text := ""
	var meta struct {
		Body string `json:"body"`
	}
	if err := json.Unmarshal(body.Text, &meta); err == nil {
		text = meta.Body
	} else {
		_ = json.Unmarshal(body.Text, &text)
	}
	c.mu.Lock()
	if c.failing[body.To] > 0 {
		c.failing[body.To]--
		c.mu.Unlock()
		http.Error(w, `{"error":{"message":"the provider is down"}}`, http.StatusBadGateway)
		return
	}
	c.sent[body.To] = append(c.sent[body.To], text)
	c.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprint(w, `{"messages":[{"id":"wamid.sent"}]}`)
}

// to is everything the sink was sent for one person, in order.
func (c *channelSink) to(number string) []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.sent[number]...)
}

// connectWhatsApp hands the router a WhatsApp line's credentials.
func (s *ScheduledTasksSuite) connectWhatsApp(number string) ChannelAccount {
	var line ChannelAccount
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/channels",
		ConnectChannelRequest{
			Kind:      "whatsapp",
			Number:    number,
			AccountId: pointerTo(strings.TrimPrefix(number, "+")),
			Token:     pointerTo("meta-token"),
			Signing:   pointerTo(appSecret),
			Challenge: pointerTo("verify-" + s.app.key),
		}, &line))
	return line
}

// lineNumber is a line nobody else in the run is using, and writerNumber a
// person's phone, as the channel providers write them.
func (s *ScheduledTasksSuite) lineNumber() string { return "+1555" + digits(s.utils.uuid()) }
func (s *ScheduledTasksSuite) writerNumber() string {
	return "+1347" + digits(s.utils.uuid())
}

// agentOnWhatsApp is a scheduling agent answering on a connected WhatsApp
// number.
func (s *ScheduledTasksSuite) agentOnWhatsApp(number string) string {
	return s.agentWithScheduling(func(request *AgentConfigRequest) {
		request.Channels = &AgentChannels{Whatsapp: &ChannelLineRequest{Number: number}}
	})
}

// seedWriter records the user as somebody who writes to the agent on
// WhatsApp from address, just now, so WhatsApp's window stands open.
func (s *ScheduledTasksSuite) seedWriter(user *testClient, configID, number, address string) {
	ctx := context.Background()
	s.Require().NoError(s.store.SaveChannelIdentity(ctx, &store.ChannelIdentity{
		CustomerID: s.customerID(), ConfigID: configID, Kind: "whatsapp",
		Address: address, UserID: user.userID,
	}))
	account, err := s.store.ChannelAccount(ctx, s.customerID(), "whatsapp", number)
	s.Require().NoError(err)
	fresh, err := s.store.ClaimChannelMessage(ctx, account.ID, "wamid."+s.utils.uuid(), address)
	s.Require().NoError(err)
	s.Require().True(fresh)
}

// taskOn is a task the backend schedules for the user on the config,
// following up in the session opened, delivered as deliverTo says.
func (s *ScheduledTasksSuite) taskOn(user *testClient, configID, sessionID string, deliverTo []string) ScheduledTask {
	body := taskRequest(configID)
	body.SessionId = &sessionID
	body.DeliverTo = deliverTo
	return s.createTask(user, body)
}

func (s *ScheduledTasksSuite) TestATaskTextedOnWhatsAppArrivesThere() {
	number, writer := s.lineNumber(), s.writerNumber()
	s.connectWhatsApp(number)
	config := s.agentOnWhatsApp(number)
	user := s.data.createUser()
	s.seedWriter(user, config, number, writer)
	opened := s.openScheduling(user, config, nil)
	created := s.taskOn(user, config, opened.Id, []string{"whatsapp"})
	s.runWorker()
	s.makeDue(s.tasksOf(user)[0])

	run := s.delivered(s.tasksOf(user)[0])
	s.Equal("whatsapp", run.DeliveredTo)
	s.Equal(created.Id, run.TaskID)
	s.Require().Eventually(func() bool { return len(s.sink.to(writer)) > 0 },
		10*time.Second, 50*time.Millisecond, "nothing was texted to %s", writer)
	s.Equal("Done.", s.sink.to(writer)[0], "the words the agent answered with are the words carried out")
	s.True(s.model().heard("<scheduled_task"), "the follow-up ran in the conversation")
}

// seedSMSLine gives the app a text line the way buying the number and
// connecting it would, without Telnyx on the line to be pointed anywhere.
func (s *ScheduledTasksSuite) seedSMSLine(number string) {
	ctx := context.Background()
	s.Require().NoError(s.store.RecordNumber(ctx, &store.PhoneNumber{
		CustomerID: s.customerID(), E164: number, Vendor: "telnyx",
		Capabilities: []string{"sms"},
	}))
	sealed, err := s.router.sealChannelSecrets(s.customerID(), channels.Secrets{
		Token: "telnyx-key", Signing: "a-public-key",
	})
	s.Require().NoError(err)
	s.Require().NoError(s.store.SaveChannelAccount(ctx, &store.ChannelAccount{
		CustomerID: s.customerID(), Kind: "sms", E164: number,
		Token: "line-" + s.utils.uuid(), SecretsSealed: sealed,
		SecretsKEKVersion: s.sealer.CurrentVersion(),
	}))
}

func (s *ScheduledTasksSuite) TestATaskTextedBySMSArrivesThere() {
	number, writer := s.lineNumber(), s.writerNumber()
	s.seedSMSLine(number)
	config := s.agentWithScheduling(func(request *AgentConfigRequest) {
		request.Channels = &AgentChannels{Sms: &ChannelLineRequest{Number: number}}
	})
	user := s.data.createUser()
	s.Require().NoError(s.store.SaveChannelIdentity(context.Background(), &store.ChannelIdentity{
		CustomerID: s.customerID(), ConfigID: config, Kind: "sms",
		Address: writer, UserID: user.userID,
	}))
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, []string{"sms"})
	s.runWorker()
	s.makeDue(s.tasksOf(user)[0])

	run := s.delivered(s.tasksOf(user)[0])
	s.Equal("sms", run.DeliveredTo)
	s.Require().Eventually(func() bool { return len(s.sink.to(writer)) > 0 },
		10*time.Second, 50*time.Millisecond, "nothing was texted to %s", writer)
	s.Equal("Done.", s.sink.to(writer)[0])
}

func (s *ScheduledTasksSuite) TestAWhatsAppTaskOutsideItsWindowLandsInTheConversation() {
	number, writer := s.lineNumber(), s.writerNumber()
	s.connectWhatsApp(number)
	config := s.agentOnWhatsApp(number)
	user := s.data.createUser()
	// The identity is there, but the person never wrote: no window.
	s.Require().NoError(s.store.SaveChannelIdentity(context.Background(), &store.ChannelIdentity{
		CustomerID: s.customerID(), ConfigID: config, Kind: "whatsapp",
		Address: writer, UserID: user.userID,
	}))
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, []string{"whatsapp", "conversation"})
	s.runWorker()
	s.makeDue(s.tasksOf(user)[0])

	run := s.delivered(s.tasksOf(user)[0])
	s.Equal("conversation", run.DeliveredTo)
	s.Empty(s.sink.to(writer), "nothing was texted outside the window")
}

func (s *ScheduledTasksSuite) TestAnUnreachablePhoneFallsBackToWhatsApp() {
	number, writer := s.lineNumber(), s.writerNumber()
	s.connectWhatsApp(number)
	config := s.agentOnWhatsApp(number)
	user := s.data.createUser()
	s.seedWriter(user, config, number, writer)
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, []string{"phone", "whatsapp"})
	s.runWorker()
	s.makeDue(s.tasksOf(user)[0])

	run := s.delivered(s.tasksOf(user)[0])
	s.Equal("whatsapp", run.DeliveredTo, "no call could be placed here, so the text carried it")
	s.Eventually(func() bool { return len(s.sink.to(writer)) > 0 },
		10*time.Second, 50*time.Millisecond)
}

func (s *ScheduledTasksSuite) TestAPhoneOnlyTaskFailsWhenNoCallCanBePlaced() {
	retries := scheduling.DeliveryRetries
	scheduling.DeliveryRetries = []time.Duration{time.Millisecond, time.Millisecond}
	s.T().Cleanup(func() { scheduling.DeliveryRetries = retries })

	number, writer := s.lineNumber(), s.writerNumber()
	s.connectWhatsApp(number)
	config := s.agentOnWhatsApp(number)
	user := s.data.createUser()
	s.seedWriter(user, config, number, writer)
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, []string{"phone"})
	s.runWorker()
	task := s.tasksOf(user)[0]
	s.makeDue(task)

	var runs []store.ScheduledTaskRun
	s.Require().Eventually(func() bool {
		runs = s.runsOf(task)
		return len(runs) > 0 && runs[0].State == store.ScheduledTaskRunFailed
	}, 30*time.Second, 250*time.Millisecond, "the occurrence never finished failing")
	s.Require().NotNil(runs[0].Error)
	s.Equal(scheduling.ReasonFailed, *runs[0].Error, "a device is told a short reason, not the vendor's words")
	s.Empty(runs[0].SessionID, "the call that could not be placed was given back")
	s.Equal(1, s.taskReloaded(task).ConsecutiveFailures)
}

func (s *ScheduledTasksSuite) TestARunByHandThatFailsListsItsReasonAndNotItsWords() {
	number, writer := s.lineNumber(), s.writerNumber()
	s.connectWhatsApp(number)
	config := s.agentOnWhatsApp(number)
	user := s.data.createUser()
	s.seedWriter(user, config, number, writer)
	opened := s.openScheduling(user, config, nil)
	created := s.taskOn(user, config, opened.Id, []string{"phone"})
	path := "/v1/agents/scheduled-tasks/" + created.Id
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, path+"/run", nil, nil))

	var page ScheduledTaskRunPage
	s.Require().Eventually(func() bool {
		s.Require().Equal(http.StatusOK, user.do(http.MethodGet, path+"/runs", nil, &page))
		return len(page.Items) == 1 && page.Items[0].State == store.ScheduledTaskRunFailed
	}, 30*time.Second, 250*time.Millisecond, "the run by hand never finished failing")
	s.Require().NotNil(page.Items[0].Error)
	s.Equal(scheduling.ReasonFailed, *page.Items[0].Error)
}

func (s *ScheduledTasksSuite) TestATaskForSomebodyWhoNeverTextedHasNoNumberToRing() {
	s.retryAtOnce()
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, []string{"phone"})
	// Due before the worker starts, so its first look claims it.
	s.makeDue(s.tasksOf(user)[0])
	s.runWorker()

	s.Equal(scheduling.ReasonNoNumber, *s.failedRun(s.tasksOf(user)[0]).Error)
}

func (s *ScheduledTasksSuite) TestARunByHandThatFailsListsTheReasonItFailedUnder() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	opened := s.openScheduling(user, config, nil)
	created := s.taskOn(user, config, opened.Id, []string{"phone"})
	path := "/v1/agents/scheduled-tasks/" + created.Id
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, path+"/run", nil, nil))

	var page ScheduledTaskRunPage
	s.Require().Eventually(func() bool {
		s.Require().Equal(http.StatusOK, user.do(http.MethodGet, path+"/runs", nil, &page))
		return len(page.Items) == 1 && page.Items[0].State == store.ScheduledTaskRunFailed
	}, 30*time.Second, 250*time.Millisecond, "the run by hand never finished failing")
	s.Require().NotNil(page.Items[0].Error)
	s.Equal(scheduling.ReasonNoNumber, *page.Items[0].Error)
}

func (s *ScheduledTasksSuite) TestATaskForAnAgentWithoutANumberHasNoLineToRingFrom() {
	s.retryAtOnce()
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	s.Require().NoError(s.store.SaveChannelIdentity(context.Background(), &store.ChannelIdentity{
		CustomerID: s.customerID(), ConfigID: config, Kind: "sms", Address: s.writerNumber(), UserID: user.userID,
	}))
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, []string{"phone"})
	// Due before the worker starts, so its first look claims it.
	s.makeDue(s.tasksOf(user)[0])
	s.runWorker()

	s.Equal(scheduling.ReasonNoLine, *s.failedRun(s.tasksOf(user)[0]).Error)
}

func (s *ScheduledTasksSuite) TestACallTaskForSomebodyOnACallWithTheAgentIsUserOnCall() {
	s.retryAtOnce()
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	opened := s.openScheduling(user, config, func(request *CreateSessionRequest) {
		request.StartVoice = pointerTo(true)
	})
	s.taskOn(user, config, opened.Id, []string{"phone"})
	// Due before the worker starts, so its first look claims it.
	s.makeDue(s.tasksOf(user)[0])
	s.runWorker()

	s.Equal(scheduling.ReasonOnACall, *s.failedRun(s.tasksOf(user)[0]).Error)
}

func (s *ScheduledTasksSuite) TestACallThatOutlastsTheWaitIsUserOnCall() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	opened := s.openScheduling(user, config, func(request *CreateSessionRequest) {
		request.StartVoice = pointerTo(true)
	})
	found, live := s.router.sessions.Get(opened.Id, s.ownerOf(user))
	s.Require().True(live)

	err := waitOutCall(context.Background(), found, 0)
	s.Equal(scheduling.ReasonOnACall, scheduling.ReasonOf(err), "%v", err)
}

func (s *ScheduledTasksSuite) TestACallNobodyPicksUpIsNotAnswered() {
	window := scheduledPhoneWindow
	scheduledPhoneWindow = 50 * time.Millisecond
	s.T().Cleanup(func() { scheduledPhoneWindow = window })
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	opened := s.openScheduling(user, config, nil)
	found, live := s.router.sessions.Get(opened.Id, s.ownerOf(user))
	s.Require().True(live)

	err := s.router.holdCall(context.Background(), found, scheduledDelivery{})
	s.Equal(scheduling.ReasonNotAnswered, scheduling.ReasonOf(err), "the ring ran out: %v", err)
	s.True(s.failure(err).Final, "the user is not rung again")

	scheduledPhoneWindow = time.Minute
	held := make(chan error, 1)
	go func() { held <- s.router.holdCall(context.Background(), found, scheduledDelivery{}) }()
	s.Require().Equal(http.StatusNoContent, user.do(http.MethodDelete, "/v1/agents/sessions/"+opened.Id, nil, nil))
	select {
	case err := <-held:
		s.Equal(scheduling.ReasonNotAnswered, scheduling.ReasonOf(err), "the call ended unanswered: %v", err)
		s.True(s.failure(err).Final, "the user is not rung again")
	case <-time.After(30 * time.Second):
		s.Fail("the call that ended was still held")
	}
}

// ownerOf is the owner the user's sessions are held under.
func (s *ScheduledTasksSuite) ownerOf(user *testClient) session.Owner {
	return scheduledDelivery{CustomerID: s.customerID(), UserID: user.userID, CallerKind: "authenticated"}.owner()
}

func (s *ScheduledTasksSuite) TestAReplyNeverWrittenIsNoReply() {
	s.retryAtOnce()
	window := scheduledReplyWindow
	scheduledReplyWindow = time.Second
	s.T().Cleanup(func() { scheduledReplyWindow = window })
	number, writer := s.lineNumber(), s.writerNumber()
	s.connectWhatsApp(number)
	config := s.agentOnWhatsApp(number)
	user := s.data.createUser()
	s.seedWriter(user, config, number, writer)
	opened := s.openScheduling(user, config, nil)
	s.T().Cleanup(s.model().holdScheduled())
	s.taskOn(user, config, opened.Id, []string{"whatsapp"})
	// Due before the worker starts, so its first look claims it.
	s.makeDue(s.tasksOf(user)[0])
	s.runWorker()

	s.Equal(scheduling.ReasonNoReply, *s.failedRun(s.tasksOf(user)[0]).Error)
	s.Empty(s.sink.to(writer))
}

func (s *ScheduledTasksSuite) TestAFailureOnEveryChannelIsListedUnderTheFirstChannelsReason() {
	s.retryAtOnce()
	number, writer := s.lineNumber(), s.writerNumber()
	s.connectWhatsApp(number)
	config := s.agentOnWhatsApp(number)
	user := s.data.createUser()
	// The person never wrote, so WhatsApp is out of its window; the call
	// rings the same number and cannot be placed here.
	s.Require().NoError(s.store.SaveChannelIdentity(context.Background(), &store.ChannelIdentity{
		CustomerID: s.customerID(), ConfigID: config, Kind: "whatsapp", Address: writer, UserID: user.userID,
	}))
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, []string{"whatsapp", "phone"})
	// Due before the worker starts, so its first look claims it.
	s.makeDue(s.tasksOf(user)[0])
	s.runWorker()

	s.Equal(scheduling.ReasonUnreachable, *s.failedRun(s.tasksOf(user)[0]).Error)
}

// retryAtOnce has a failed occurrence tried again straight away, twice.
func (s *ScheduledTasksSuite) retryAtOnce() {
	retries := scheduling.DeliveryRetries
	scheduling.DeliveryRetries = []time.Duration{time.Millisecond, time.Millisecond}
	s.T().Cleanup(func() { scheduling.DeliveryRetries = retries })
}

// failedRun waits for the task's latest run to fail, and returns it.
func (s *ScheduledTasksSuite) failedRun(task store.ScheduledTask) store.ScheduledTaskRun {
	var latest store.ScheduledTaskRun
	s.Eventually(func() bool {
		runs := s.runsOf(task)
		if len(runs) == 0 {
			return false
		}
		latest = runs[0]
		return latest.State == store.ScheduledTaskRunFailed
	}, 30*time.Second, 250*time.Millisecond, "the occurrence never finished failing")
	s.Require().NotNil(latest.Error)
	return latest
}

func (s *ScheduledTasksSuite) TestATaskOutsideWhatsAppsWindowRunsNoTurnForIt() {
	s.retryAtOnce()
	number, writer := s.lineNumber(), s.writerNumber()
	s.connectWhatsApp(number)
	config := s.agentOnWhatsApp(number)
	user := s.data.createUser()
	// The identity is there, but the person never wrote: no window.
	s.Require().NoError(s.store.SaveChannelIdentity(context.Background(), &store.ChannelIdentity{
		CustomerID: s.customerID(), ConfigID: config, Kind: "whatsapp",
		Address: writer, UserID: user.userID,
	}))
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, []string{"whatsapp"})
	s.runWorker()
	task := s.tasksOf(user)[0]
	s.makeDue(task)

	run := s.failedRun(task)
	s.Equal(scheduling.ReasonUnreachable, *run.Error)
	s.Zero(s.model().turnsOn("<scheduled_task"), "no turn was paid for words WhatsApp would refuse")
}

func (s *ScheduledTasksSuite) TestATaskForSomebodyWhoOptedOutRunsNoTurnForThem() {
	s.retryAtOnce()
	number, writer := s.lineNumber(), s.writerNumber()
	s.connectWhatsApp(number)
	config := s.agentOnWhatsApp(number)
	user := s.data.createUser()
	s.seedWriter(user, config, number, writer)
	s.Require().NoError(s.store.OptOut(context.Background(), &store.OptOut{
		CustomerID: s.customerID(), Recipient: writer, Channel: "whatsapp", Source: "keyword",
	}))
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, []string{"whatsapp"})
	s.runWorker()
	task := s.tasksOf(user)[0]
	s.makeDue(task)

	run := s.failedRun(task)
	s.Equal(scheduling.ReasonUnreachable, *run.Error)
	s.Zero(s.model().turnsOn("<scheduled_task"), "no turn was paid for words the gate would refuse")
	s.Empty(s.sink.to(writer))
}

func (s *ScheduledTasksSuite) TestAReplyOneTextChannelCouldNotSendIsTextedOnTheNext() {
	whatsAppLine, smsLine := s.lineNumber(), s.lineNumber()
	onWhatsApp, bySMS := s.writerNumber(), s.writerNumber()
	s.connectWhatsApp(whatsAppLine)
	s.seedSMSLine(smsLine)
	config := s.agentWithScheduling(func(request *AgentConfigRequest) {
		request.Channels = &AgentChannels{
			Whatsapp: &ChannelLineRequest{Number: whatsAppLine},
			Sms:      &ChannelLineRequest{Number: smsLine},
		}
	})
	user := s.data.createUser()
	s.seedWriter(user, config, whatsAppLine, onWhatsApp)
	s.Require().NoError(s.store.SaveChannelIdentity(context.Background(), &store.ChannelIdentity{
		CustomerID: s.customerID(), ConfigID: config, Kind: "sms",
		Address: bySMS, UserID: user.userID,
	}))
	s.sink.fail(onWhatsApp, 1)
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, []string{"whatsapp", "sms"})
	s.runWorker()
	s.makeDue(s.tasksOf(user)[0])

	run := s.delivered(s.tasksOf(user)[0])
	s.Equal("sms", run.DeliveredTo)
	s.Equal([]string{"Done."}, s.sink.to(bySMS), "the words WhatsApp could not take are the ones texted")
	s.Empty(s.sink.to(onWhatsApp))
	s.Equal(1, s.model().turnsOn("<scheduled_task"), "one turn wrote the words for both channels")
}

func (s *ScheduledTasksSuite) TestARetriedOccurrenceTextsTheReplyItAlreadyHas() {
	s.retryAtOnce()
	number, writer := s.lineNumber(), s.writerNumber()
	s.connectWhatsApp(number)
	config := s.agentOnWhatsApp(number)
	user := s.data.createUser()
	s.seedWriter(user, config, number, writer)
	s.sink.fail(writer, 1)
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, []string{"whatsapp"})
	s.runWorker()
	s.makeDue(s.tasksOf(user)[0])

	run := s.delivered(s.tasksOf(user)[0])
	s.Equal("whatsapp", run.DeliveredTo)
	s.Equal([]string{"Done."}, s.sink.to(writer), "the retry carried the reply the first attempt could not")
	s.Equal(1, s.model().turnsOn("<scheduled_task"), "the retry ran no second turn")
}

func (s *ScheduledTasksSuite) TestAReplyThatFailedIsNotWaitedOnWhenTheOccurrenceIsRetried() {
	s.retryAtOnce()
	number, writer := s.lineNumber(), s.writerNumber()
	s.connectWhatsApp(number)
	config := s.agentOnWhatsApp(number)
	user := s.data.createUser()
	s.seedWriter(user, config, number, writer)
	opened := s.openScheduling(user, config, nil)
	s.model().failScheduled(1)
	s.taskOn(user, config, opened.Id, []string{"whatsapp"})
	task := s.tasksOf(user)[0]
	// Due before the worker starts, so its first look claims it.
	s.makeDue(task)
	s.runWorker()

	run := s.failedRun(task)
	s.Equal(scheduling.ReasonReplyFailed, *run.Error, "the retry read the failed reply back at once")
	s.Equal(1, s.model().turnsOn("<scheduled_task"), "the retry ran no second turn")
	s.Empty(s.sink.to(writer))
}

func (s *ScheduledTasksSuite) TestARetryWaitsForAReplyStillBeingWritten() {
	s.retryAtOnce()
	window := scheduledReplyWindow
	scheduledReplyWindow = 6 * time.Second
	s.T().Cleanup(func() { scheduledReplyWindow = window })
	number, writer := s.lineNumber(), s.writerNumber()
	s.connectWhatsApp(number)
	config := s.agentOnWhatsApp(number)
	user := s.data.createUser()
	s.seedWriter(user, config, number, writer)
	opened := s.openScheduling(user, config, nil)
	release := s.model().holdScheduled()
	s.T().Cleanup(release)
	s.taskOn(user, config, opened.Id, []string{"whatsapp"})
	task := s.tasksOf(user)[0]
	// Due before the worker starts, so its first look claims it.
	s.makeDue(task)
	s.runWorker()

	// The first attempt gives up on the reply once the window is spent, and
	// the retry straight after it finds the turn still running; the reply is
	// written a second into the retry's own window.
	s.Eventually(func() bool { return s.model().turnsOn("<scheduled_task") == 1 },
		30*time.Second, 50*time.Millisecond, "the occurrence never ran its turn")
	time.Sleep(scheduledReplyWindow + time.Second)
	release()

	run := s.delivered(task)
	s.Equal("whatsapp", run.DeliveredTo)
	s.Equal([]string{"Done."}, s.sink.to(writer), "the retry carried the reply once it was written")
	s.Equal(1, s.model().turnsOn("<scheduled_task"), "the retry ran no second turn")
}

func (s *ScheduledTasksSuite) TestAQuietTaskIsHeldUntilItsQuietHoursEnd() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	opened := s.openScheduling(user, config, nil)
	now := time.Now().UTC()
	body := taskRequest(config)
	body.SessionId = &opened.Id
	body.Schedule.Timezone = pointerTo("UTC")
	body.QuietHours = &ScheduledTaskQuietHours{
		Start: now.Add(-time.Hour).Format("15:04"),
		End:   now.Add(2 * time.Hour).Format("15:04"),
	}
	created := s.createTask(user, body)
	s.Require().NotNil(created.QuietHours)
	s.runWorker()
	task := s.tasksOf(user)[0]
	s.makeDue(task)

	s.Eventually(func() bool {
		held := s.taskReloaded(task)
		return held.LeaseUntil != nil && held.LeaseUntil.After(time.Now().Add(time.Hour))
	}, 15*time.Second, 250*time.Millisecond, "the task was never held")
	runs := s.runsOf(task)
	s.Require().Len(runs, 1)
	s.Equal(store.ScheduledTaskRunRunning, runs[0].State, "a held run is still running")
	s.Require().NotNil(runs[0].PostponedUntil)
	s.Equal(store.ScheduledTaskActive, s.taskReloaded(task).State)
	s.False(s.model().heard("<scheduled_task"), "nothing was delivered into the quiet hours")
}

func (s *ScheduledTasksSuite) TestAQuietOneShotEditedWhileHeldIsHeldAgainAsItNowReads() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	opened := s.openScheduling(user, config, nil)
	now := time.Now().UTC()
	body := taskRequest(config)
	body.SessionId = &opened.Id
	body.Schedule.Timezone = pointerTo("UTC")
	body.QuietHours = &ScheduledTaskQuietHours{
		Start: now.Add(-time.Hour).Format("15:04"),
		End:   now.Add(2 * time.Hour).Format("15:04"),
	}
	created := s.createTask(user, body)
	s.runWorker()
	task := s.tasksOf(user)[0]
	s.makeDue(task)
	held := func() bool {
		stored := s.taskReloaded(task)
		return stored.LeaseUntil != nil && stored.LeaseUntil.After(time.Now().Add(time.Hour))
	}
	s.Require().Eventually(held, 15*time.Second, 250*time.Millisecond, "the task was never held")
	first := s.runsOf(task)[0]

	var updated ScheduledTask
	s.Require().Equal(http.StatusOK, s.serverClient.actingFor(user).do(http.MethodPatch, "/v1/agents/scheduled-tasks/"+created.Id,
		UpdateScheduledTaskRequest{Instructions: pointerTo("Remind them to drink water.")}, &updated))
	s.Equal(store.ScheduledTaskActive, updated.State)

	// The held occurrence is settled, and the one-shot comes due again as it
	// now reads: quiet hours still hold, so it is held again, not delivered.
	s.Require().Eventually(func() bool { return len(s.runsOf(task)) == 2 && held() },
		15*time.Second, 250*time.Millisecond, "the edited one-shot was never taken up again")
	runs := s.runsOf(task)
	s.Equal(first.ID, runs[1].ID)
	s.Equal(store.ScheduledTaskRunMissed, runs[1].State)
	s.Equal(store.ScheduledTaskRunRunning, runs[0].State)
	s.Equal("Remind them to drink water.", runs[0].Prompt)
	s.Require().NotNil(runs[0].PostponedUntil, "the new occurrence is held through the quiet hours")
	s.Equal(store.ScheduledTaskActive, s.taskReloaded(task).State)
	s.False(s.model().heard("<scheduled_task"), "nothing was delivered into the quiet hours")
}

func (s *ScheduledTasksSuite) TestDeliverToAndQuietHoursReadBackAndCanBeChanged() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	body := taskRequest(config)
	body.DeliverTo = []string{"phone", "sms", "phone"}
	body.QuietHours = &ScheduledTaskQuietHours{Start: "22:00", End: "07:00"}
	created := s.createTask(user, body)

	s.Equal([]string{"phone", "sms"}, created.DeliverTo, "each channel once, in the order written")
	s.Require().NotNil(created.QuietHours)
	s.Equal("22:00", created.QuietHours.Start)
	s.Equal("07:00", created.QuietHours.End)

	var updated ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodPatch, "/v1/agents/scheduled-tasks/"+created.Id,
		UpdateScheduledTaskRequest{DeliverTo: pointerTo([]string{})}, &updated))
	s.Equal([]string{"conversation"}, updated.DeliverTo, "an empty list is the conversation alone")

	var cleared ScheduledTask
	s.Require().Equal(http.StatusOK, user.do(http.MethodPatch, "/v1/agents/scheduled-tasks/"+created.Id,
		UpdateScheduledTaskRequest{QuietHours: &ScheduledTaskQuietHours{}}, &cleared))
	s.Nil(cleared.QuietHours, "both boundaries empty is the hold cleared")
}

func (s *ScheduledTasksSuite) TestDeliverToAndQuietHoursThatDoNotReadAreRefused() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()

	body := taskRequest(config)
	body.DeliverTo = []string{"carrier-pigeon"}
	status, answered := s.taskFailure(s.serverClient.actingFor(user),
		http.MethodPost, "/v1/agents/scheduled-tasks", body)
	s.Equal(http.StatusBadRequest, status)
	s.Contains(answered.Error.Message, "deliver_to")

	body = taskRequest(config)
	body.QuietHours = &ScheduledTaskQuietHours{Start: "22:00", End: "22:00"}
	status, answered = s.taskFailure(s.serverClient.actingFor(user),
		http.MethodPost, "/v1/agents/scheduled-tasks", body)
	s.Equal(http.StatusBadRequest, status)
	s.Contains(answered.Error.Message, "quiet_hours")

	body = taskRequest(config)
	body.QuietHours = &ScheduledTaskQuietHours{Start: "ten", End: "07:00"}
	status, _ = s.taskFailure(s.serverClient.actingFor(user),
		http.MethodPost, "/v1/agents/scheduled-tasks", body)
	s.Equal(http.StatusBadRequest, status)
}

func (s *ScheduledTasksSuite) TestPausingATaskWritesTheOccurrenceItHeldMissed() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, taskRequest(config))
	task := s.tasksOf(user)[0]
	// An occurrence quiet hours hold is a run left running until the hold ends.
	held := store.ScheduledTaskRun{TaskID: task.ID, CustomerID: s.customerID(), ScheduledFor: *task.NextRunAt, Attempts: 1}
	s.Require().NoError(s.store.InsertScheduledTaskRun(context.Background(), &held))
	path := "/v1/agents/scheduled-tasks/" + created.Id

	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, path+"/pause", nil, nil))

	var page ScheduledTaskRunPage
	s.Require().Equal(http.StatusOK, user.do(http.MethodGet, path+"/runs", nil, &page))
	s.Require().Len(page.Items, 1)
	s.Equal(store.ScheduledTaskRunMissed, page.Items[0].State, "the held occurrence is settled, not left running for good")
	s.NotNil(page.Items[0].FinishedAt)
}

func (s *ScheduledTasksSuite) TestAnOccurrenceWhoseWorkersKeepDyingListsACodeAsItsReason() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	created := s.createTask(user, taskRequest(config))
	task := s.tasksOf(user)[0]
	due := time.Now().UTC().Add(-time.Minute).Truncate(time.Second)
	task.NextRunAt = &due
	s.Require().NoError(s.store.SaveScheduledTask(context.Background(), &task))
	// Three workers took it up and died: the store's cap.
	started := time.Now().Add(-time.Hour)
	stale := store.ScheduledTaskRun{TaskID: task.ID, CustomerID: s.customerID(), ScheduledFor: due,
		Trigger: store.ScheduledTaskTriggerSchedule, Prompt: task.Instructions,
		State: store.ScheduledTaskRunRunning, Attempts: 3, StartedAt: &started}
	s.Require().NoError(s.store.InsertScheduledTaskRun(context.Background(), &stale))
	s.runWorker()

	path := "/v1/agents/scheduled-tasks/" + created.Id
	var page ScheduledTaskRunPage
	s.Require().Eventually(func() bool {
		page = ScheduledTaskRunPage{}
		return user.do(http.MethodGet, path+"/runs", nil, &page) == http.StatusOK &&
			len(page.Items) == 1 && page.Items[0].State == store.ScheduledTaskRunFailed
	}, 30*time.Second, 250*time.Millisecond, "the occurrence was never failed")
	s.Require().NotNil(page.Items[0].Error)
	s.Equal(scheduling.ReasonFailed, *page.Items[0].Error, "a code from the closed set, the story in the logs")
}

func (s *ScheduledTasksSuite) TestListingReadsWhenEveryTaskLastRanInOneQuery() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	finished := map[string]string{}
	for hours := range 3 {
		created := s.createTask(user, taskRequest(config))
		at := time.Now().UTC().Add(-time.Duration(hours+1) * time.Hour).Truncate(time.Second)
		run := store.ScheduledTaskRun{TaskID: created.Id, CustomerID: s.customerID(), ScheduledFor: at,
			Trigger: store.ScheduledTaskTriggerManual, State: store.ScheduledTaskRunDelivered, FinishedAt: &at}
		s.Require().NoError(s.store.InsertScheduledTaskRun(context.Background(), &run))
		finished[created.Id] = at.Format(time.RFC3339)
	}

	s.runReads.reads.Store(0)
	s.runReads.on.Store(true)
	var page ScheduledTaskPage
	status := user.do(http.MethodGet, "/v1/agents/scheduled-tasks", nil, &page)
	s.runReads.on.Store(false)

	s.Require().Equal(http.StatusOK, status)
	s.Equal(int64(1), s.runReads.reads.Load(), "one read for the whole page, not one per task")
	s.Require().Len(page.Items, 3)
	for _, item := range page.Items {
		s.Require().NotNil(item.LastRunAt)
		s.Equal(finished[item.Id], *item.LastRunAt)
	}
}

func (s *ScheduledTasksSuite) TestTheActiveLimitCountsTheTasksOnTheirOwnAgentOnly() {
	limited := s.agentWithScheduling(func(request *AgentConfigRequest) {
		request.Scheduling.MaxTasksPerUser = pointerTo(1)
	})
	other := s.agentWithScheduling(nil)
	user := s.data.createUser()
	s.createTask(user, taskRequest(other))

	first := s.createTask(user, taskRequest(limited))
	path := "/v1/agents/scheduled-tasks/" + first.Id
	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, path+"/pause", nil, nil))
	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, path+"/resume", nil, nil),
		"the task on the other agent leaves this one room")

	// Another task running on this agent takes its one place.
	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, path+"/pause", nil, nil))
	s.createTask(user, taskRequest(limited))
	status, answered := s.taskFailure(user, http.MethodPost, path+"/resume", nil)
	s.Equal(http.StatusConflict, status)
	s.Equal(codeTooManyTasks, answered.Error.Code)
}

func (s *ScheduledTasksSuite) TestTheAgentResumesATaskWhileOnlyAnotherAgentsTasksRun() {
	limited := s.agentWithScheduling(func(request *AgentConfigRequest) {
		request.Scheduling.MaxTasksPerUser = pointerTo(1)
	})
	other := s.agentWithScheduling(nil)
	user := s.data.createUser()
	s.createTask(user, taskRequest(other))
	paused := s.createTask(user, taskRequest(limited))
	path := "/v1/agents/scheduled-tasks/" + paused.Id
	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, path+"/pause", nil, nil))
	opened := s.openScheduling(user, limited, nil)
	watching := s.watch(user, opened.Id)
	resume := func(title string) llm.ToolCall {
		return llm.ToolCall{ID: store.NewID(), Name: "update_scheduled_task",
			Arguments: `{"id":"` + paused.Id + `","action":"resume","title":"` + title + `"}`}
	}

	s.schedulingScript.plan(resume("Stretch"))
	s.say(user, opened.Id, "turn the stretch reminder back on")
	s.NotContains(s.ranResult(watching), `"error"`)
	s.await(watching, "responded")
	stored, err := s.store.ScheduledTask(context.Background(), s.customerID(), paused.Id)
	s.Require().NoError(err)
	s.Equal(store.ScheduledTaskActive, stored.State)

	// Another task running on this agent takes its one place.
	s.Require().Equal(http.StatusOK, user.do(http.MethodPost, path+"/pause", nil, nil))
	s.createTask(user, taskRequest(limited))
	s.schedulingScript.plan(resume("Stretch again"))
	s.say(user, opened.Id, "and back on again")
	s.Contains(s.ranResult(watching), `"error":"too_many_tasks"`)
}
