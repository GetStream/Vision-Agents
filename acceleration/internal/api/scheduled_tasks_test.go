//go:build integration

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/channels"
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
	sink *channelSink
}

func TestScheduledTasksSuite(t *testing.T) {
	runSuite(t, new(ScheduledTasksSuite))
}

func (s *ScheduledTasksSuite) SetupSuite() {
	s.sink = &channelSink{sent: map[string][]string{}}
	s.channelAPI = httptest.NewTLSServer(http.HandlerFunc(s.sink.serve))
	s.T().Cleanup(s.channelAPI.Close)
	s.RouterSuite.SetupSuite()
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
	worker.Start(context.Background())
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
	s.Eventually(func() bool { return len(s.sink.to(writer)) > 0 },
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
	s.Eventually(func() bool { return len(s.sink.to(writer)) > 0 },
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
	s.Eventually(func() bool {
		runs = s.runsOf(task)
		return len(runs) > 0 && runs[0].State == store.ScheduledTaskRunFailed
	}, 30*time.Second, 250*time.Millisecond, "the occurrence never finished failing")
	s.Require().NotNil(runs[0].Error)
	s.Contains(*runs[0].Error, "phone", "the run says the call is what could not be made")
	s.Equal(1, s.taskReloaded(task).ConsecutiveFailures)
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
