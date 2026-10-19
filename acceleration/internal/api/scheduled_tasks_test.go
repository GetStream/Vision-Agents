//go:build integration

package api

import (
	"context"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// schedulingScript is the tool calls the scheduling models run, queued by the
// test, one entry per person who speaks. A test and the models share it
// because which model a session opened is only known once it has spoken.
type schedulingScript struct {
	mu    sync.Mutex
	turns [][]llm.ToolCall
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
	s.turns = nil
}

// schedulingModel is a model that asks for the tool calls its test queued and
// records what it was offered and told, so a test can say what the model did
// and what it was given to do it with.
type schedulingModel struct {
	script *schedulingScript

	mu      sync.Mutex
	offered [][]string
	asked   []llm.ResponseParams
}

func (m *schedulingModel) Start(context.Context) error { return nil }

func (m *schedulingModel) Create(_ context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	names := make([]string, 0, len(params.Tools))
	for _, tool := range params.Tools {
		names = append(names, tool.Name)
	}
	m.mu.Lock()
	m.offered = append(m.offered, names)
	m.asked = append(m.asked, params)
	m.mu.Unlock()

	// A turn somebody started runs what the test queued; the turn a tool's
	// answer starts is simply agreed with.
	var calls []llm.ToolCall
	if len(params.Input) > 0 && params.Input[len(params.Input)-1].Role == llm.User {
		calls = m.script.next()
	}
	script := llmtest.New(llm.StreamOptions{ResponseID: params.ID, Provider: m.Provider(), Model: m.Model()})
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
}

func TestScheduledTasksSuite(t *testing.T) {
	runSuite(t, new(ScheduledTasksSuite))
}

func (s *ScheduledTasksSuite) SetupTest() {
	s.useApp(s.data.createApp())
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
	worker.Start()
	s.T().Cleanup(worker.Close)
}

// makeDue moves a task's next run into the past, so the worker claims it at
// its next look.
func (s *ScheduledTasksSuite) makeDue(task store.ScheduledTask) {
	past := time.Now().Add(-time.Minute)
	task.NextRunAt = &past
	s.Require().NoError(s.store.SaveScheduledTask(context.Background(), &task))
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
	s.Eventually(func() bool {
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
