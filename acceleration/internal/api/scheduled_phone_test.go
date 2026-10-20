//go:build integration

package api

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ringingEdge is the call a task placed: the person rung joins it as a phone
// caller once they pick up, the way the vendor bridges the answered leg in.
type ringingEdge struct {
	silentEdge
	attendance chan agent.Attendance
}

func newRingingEdge() *ringingEdge {
	return &ringingEdge{
		silentEdge: silentEdge{inbound: make(chan agent.InboundAudio, 4)},
		attendance: make(chan agent.Attendance, 4),
	}
}

func (e *ringingEdge) Attendance() <-chan agent.Attendance { return e.attendance }

func (e *ringingEdge) Leave() error {
	close(e.attendance)
	return e.silentEdge.Leave()
}

// pickUp has the person rung at number answer.
func (e *ringingEdge) pickUp(number string) {
	e.attendance <- agent.Attendance{
		Participant: stt.Participant{ID: "sip-" + number, UserID: "sip-" + number}, Joined: true,
	}
}

// placedCall is a phone task's occurrence as deliverByPhone holds it once the
// vendor has placed the call: the delivery, and the session it opens for the
// call, on an edge the test answers.
type placedCall struct {
	delivery scheduledDelivery
	spec     session.Spec
	edge     *ringingEdge
	to       string
	opened   Session
}

func (s *ScheduledTasksSuite) placeCall(user *testClient) placedCall {
	configID := s.agentWithScheduling(nil)
	opened := s.openScheduling(user, configID, nil)
	s.taskOn(user, configID, opened.Id, []string{"phone"})
	task := s.tasksOf(user)[0]
	now := time.Now()
	run := store.ScheduledTaskRun{ID: store.NewID(), TaskID: task.ID, ScheduledFor: now, Prompt: task.Instructions}
	delivery := scheduledDelivery{
		CustomerID: task.CustomerID, TaskID: task.ID, RunID: run.ID, SessionID: task.SessionID,
		ConversationID: task.ConversationID, ConfigID: task.ConfigID, UserID: task.UserID,
		CallerKind: task.CallerKind, Timezone: task.Timezone, CommandID: "st_" + run.ID,
		Text: scheduledPrompt(store.ClaimedScheduledTask{Task: task, Run: run}, now), Due: now,
	}
	config, err := s.store.AgentConfig(context.Background(), s.customerID(), configID)
	s.Require().NoError(err)
	to := s.writerNumber()
	found, live := s.router.sessions.Get(opened.Id, delivery.owner())
	s.Require().True(live)
	spec := callSpec(config, delivery, found.Spec().AgentID, phone.Placed{SessionID: uuid.NewString()}, s.lineNumber(), to)
	edge := newRingingEdge()
	spec.Edge = edge
	// The suite's sessions have no telephony to press digits or transfer
	// with, which is all the number is for once the call is placed.
	spec.Phone = nil
	return placedCall{delivery: delivery, spec: spec, edge: edge, to: to, opened: opened}
}

// callOutcome is what carrying out a task on its call came to.
type callOutcome struct {
	deliveredTo, sessionID string
	err                    error
}

// carryOut has the router carry out the task on its call, as a scheduled
// delivery does, and answers once it is done.
func (s *ScheduledTasksSuite) carryOut(call placedCall) <-chan callOutcome {
	done := make(chan callOutcome, 1)
	go func() {
		deliveredTo, sessionID, err := s.router.carryOutOnCall(
			routing.WithScheduledTurn(context.Background()), call.delivery, call.spec)
		done <- callOutcome{deliveredTo: deliveredTo, sessionID: sessionID, err: err}
	}()
	return done
}

// callOpened waits for the call's session to be open.
func (s *ScheduledTasksSuite) callOpened(call placedCall) {
	s.Require().Eventually(func() bool {
		_, live := s.router.sessions.Get(call.spec.ID, session.OwnerOf(call.spec))
		return live
	}, 30*time.Second, 20*time.Millisecond, "the call's session never opened")
}

func (s *ScheduledTasksSuite) outcomeOf(done <-chan callOutcome) callOutcome {
	select {
	case outcome := <-done:
		return outcome
	case <-time.After(60 * time.Second):
		s.FailNow("the call was never let go")
		return callOutcome{}
	}
}

// shortCalls has calls ring for ring and last for limit.
func (s *ScheduledTasksSuite) shortCalls(ring, limit time.Duration) {
	window, longest := scheduledPhoneWindow, scheduledCallLimit
	scheduledPhoneWindow, scheduledCallLimit = ring, limit
	s.T().Cleanup(func() { scheduledPhoneWindow, scheduledCallLimit = window, longest })
}

func (s *ScheduledTasksSuite) TestACallATaskPlacedIsGivenTheTaskAsItsFirstTurnAndNotAsInstructions() {
	s.shortCalls(30*time.Second, 2*time.Second)
	call := s.placeCall(s.data.createUser())

	started := time.Now()
	done := s.carryOut(call)
	s.callOpened(call)
	model := s.model()
	s.Zero(model.turnsOn("<scheduled_task"), "the agent kept the task until somebody picked up")
	call.edge.pickUp(call.to)
	outcome := s.outcomeOf(done)

	s.Require().NoError(outcome.err)
	s.Equal(store.ScheduledTaskDeliveredPhone, outcome.deliveredTo)
	s.Equal(call.spec.ID, outcome.sessionID)
	s.Equal(1, model.turnsOn("<scheduled_task"), "the task was the call's first turn")
	s.True(model.heard("Remind them to stretch."))
	s.NotContains(model.instructions(), "<scheduled_task", "the task is no part of what the agent is told it is")
	s.NotContains(model.instructions(), "Remind them to stretch.")
	s.Contains(model.instructions(), "You have called this number", "the agent knows it placed the call")
	s.GreaterOrEqual(time.Since(started), scheduledCallLimit, "the call was held until its limit")
	_, live := s.router.sessions.Get(call.spec.ID, session.OwnerOf(call.spec))
	s.False(live, "the agent hung up at the call's limit")
}

func (s *ScheduledTasksSuite) TestACallNobodyPicksUpIsNotAnsweredAndNotRetried() {
	s.shortCalls(time.Second, time.Minute)
	call := s.placeCall(s.data.createUser())

	outcome := s.outcomeOf(s.carryOut(call))

	var failure *scheduling.Failure
	s.Require().True(errors.As(outcome.err, &failure), "%v", outcome.err)
	s.Equal(scheduling.ReasonNotAnswered, failure.Reason, "%v", outcome.err)
	s.True(failure.Final, "the user is not rung again")
	s.Zero(s.model().turnsOn("<scheduled_task"), "nothing was said into a call nobody took")
	_, live := s.router.sessions.Get(call.spec.ID, session.OwnerOf(call.spec))
	s.False(live)
}

func (s *ScheduledTasksSuite) TestSomebodyWhoPickedUpBeforeTheCallWasWatchedIsStillGivenTheTask() {
	s.shortCalls(2*time.Second, time.Second)
	call := s.placeCall(s.data.createUser())
	// Answered before the session opened: the edge reports them on joining,
	// before anything can watch the session.
	call.edge.pickUp(call.to)

	outcome := s.outcomeOf(s.carryOut(call))

	s.Require().NoError(outcome.err)
	s.Equal(1, s.model().turnsOn("<scheduled_task"))
}

func (s *ScheduledTasksSuite) TestACallIsGivenTheTaskWhenAnEarlierChannelAlreadyRanIt() {
	s.shortCalls(30*time.Second, time.Second)
	user := s.data.createUser()
	call := s.placeCall(user)
	// A text channel tried first ran the occurrence's turn in the
	// conversation, under the run's own id.
	found, live := s.router.sessions.Get(call.opened.Id, call.delivery.owner())
	s.Require().True(live)
	s.Require().NoError(found.FollowUp(context.Background(), call.delivery.CommandID, call.delivery.Text))
	s.Require().Eventually(func() bool {
		state, _, _ := found.FollowUpReply(call.delivery.CommandID)
		return state == "completed"
	}, 30*time.Second, 50*time.Millisecond, "the earlier channel's turn never finished")

	done := s.carryOut(call)
	s.callOpened(call)
	model := s.model()
	call.edge.pickUp(call.to)
	outcome := s.outcomeOf(done)

	s.Require().NoError(outcome.err)
	s.Equal(1, model.turnsOn("<scheduled_task"), "the agent spoke the task on the call too")
}

// phoneTask is a task the backend schedules for the user, delivered by phone
// only, on an agent with a WhatsApp line the user has written on, due at due.
func (s *ScheduledTasksSuite) phoneTask(user *testClient, due time.Time) store.ScheduledTask {
	number, writer := s.lineNumber(), s.writerNumber()
	s.connectWhatsApp(number)
	config := s.agentOnWhatsApp(number)
	s.seedWriter(user, config, number, writer)
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, []string{"phone"})
	task := s.tasksOf(user)[0]
	task.NextRunAt = &due
	s.Require().NoError(s.store.SaveScheduledTask(context.Background(), &task))
	return task
}

// settledRun waits for the task's latest run to finish, and returns it.
func (s *ScheduledTasksSuite) settledRun(task store.ScheduledTask) store.ScheduledTaskRun {
	var latest store.ScheduledTaskRun
	s.Require().Eventually(func() bool {
		runs := s.runsOf(task)
		if len(runs) == 0 {
			return false
		}
		latest = runs[0]
		return latest.State != store.ScheduledTaskRunRunning
	}, 30*time.Second, 250*time.Millisecond, "the occurrence never finished")
	return latest
}

func (s *ScheduledTasksSuite) TestAPhoneTaskMoreThanAQuarterHourLateIsMissedWithoutACall() {
	s.retryAtOnce()
	user := s.data.createUser()
	task := s.phoneTask(user, time.Now().Add(-20*time.Minute))
	s.runWorker()

	run := s.settledRun(task)
	s.Equal(store.ScheduledTaskRunMissed, run.State)
	s.Empty(run.SessionID, "no call was placed")
	s.Nil(run.Error)
	stored := s.taskReloaded(task)
	s.Equal(store.ScheduledTaskCompleted, stored.State)
	s.Zero(stored.ConsecutiveFailures, "a missed call is no failure")
}

func (s *ScheduledTasksSuite) TestAPhoneTaskHeldOvernightIsCountedLateFromTheHoldsEnd() {
	s.retryAtOnce()
	user := s.data.createUser()
	due := time.Now().Add(-8 * time.Hour).UTC().Truncate(time.Second)
	task := s.phoneTask(user, due)
	// The occurrence was held through the night and the hold ended a minute
	// ago: the call is a minute late, not eight hours.
	started, ended := due, time.Now().Add(-time.Minute)
	s.Require().NoError(s.store.InsertScheduledTaskRun(context.Background(), &store.ScheduledTaskRun{
		TaskID: task.ID, CustomerID: task.CustomerID, ScheduledFor: due, Trigger: store.ScheduledTaskTriggerSchedule,
		Prompt: task.Instructions, State: store.ScheduledTaskRunRunning, Attempts: 1,
		StartedAt: &started, PostponedUntil: &ended,
	}))
	s.runWorker()

	run := s.settledRun(task)
	s.Equal(store.ScheduledTaskRunFailed, run.State, "the call was tried, not missed")
	s.Equal(scheduling.ReasonFailed, *run.Error, "no call can be placed here")
}

func (s *ScheduledTasksSuite) TestAnOccurrenceThatRangAlreadyIsNotRungAgain() {
	s.retryAtOnce()
	user := s.data.createUser()
	due := time.Now().Add(-11 * time.Minute).UTC().Truncate(time.Second)
	task := s.phoneTask(user, due)
	// A worker rang for this occurrence and stopped before it was settled;
	// its run has gone stale and is taken over.
	s.Require().NoError(s.store.InsertScheduledTaskRun(context.Background(), &store.ScheduledTaskRun{
		TaskID: task.ID, CustomerID: task.CustomerID, ScheduledFor: due, Trigger: store.ScheduledTaskTriggerSchedule,
		Prompt: task.Instructions, State: store.ScheduledTaskRunRunning, Attempts: 1,
		StartedAt: &due, SessionID: "the-earlier-call",
	}))
	s.runWorker()

	run := s.settledRun(task)
	s.Equal(store.ScheduledTaskRunFailed, run.State)
	s.Equal(scheduling.ReasonAlreadyCalled, *run.Error)
	s.Equal("the-earlier-call", run.SessionID)
}

func (s *ScheduledTasksSuite) TestAPhoneTaskForSomebodyWhoNeverTextedFailsForWantOfANumber() {
	s.retryAtOnce()
	number := s.lineNumber()
	s.connectWhatsApp(number)
	config := s.agentOnWhatsApp(number)
	user := s.data.createUser()
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, []string{"phone"})
	task := s.tasksOf(user)[0]
	s.makeDue(task)
	s.runWorker()

	s.Equal(scheduling.ReasonNoNumber, *s.failedRun(task).Error)
}

func (s *ScheduledTasksSuite) TestAPhoneTaskOnAnAgentWithNoLineFailsForWantOfOne() {
	s.retryAtOnce()
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	s.Require().NoError(s.store.SaveChannelIdentity(context.Background(), &store.ChannelIdentity{
		CustomerID: s.customerID(), ConfigID: config, Kind: "sms", Address: s.writerNumber(), UserID: user.userID,
	}))
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, []string{"phone"})
	task := s.tasksOf(user)[0]
	s.makeDue(task)
	s.runWorker()

	s.Equal(scheduling.ReasonNoLine, *s.failedRun(task).Error)
}

func (s *ScheduledTasksSuite) TestAPhoneTaskWhileTheUserIsOnACallFailsAsSuch() {
	s.retryAtOnce()
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	opened := s.openScheduling(user, config, func(request *CreateSessionRequest) {
		request.StartVoice = pointerTo(true)
	})
	s.taskOn(user, config, opened.Id, []string{"phone"})
	task := s.tasksOf(user)[0]
	s.makeDue(task)
	s.runWorker()

	s.Equal(scheduling.ReasonOnACall, *s.failedRun(task).Error)
	s.Require().Equal(http.StatusOK, user.do(http.MethodDelete, "/v1/agents/sessions/"+opened.Id+"/voice", nil, nil))
}

func (s *ScheduledTasksSuite) TestACallTheUserHangsUpIsDelivered() {
	s.shortCalls(30*time.Second, time.Minute)
	call := s.placeCall(s.data.createUser())

	done := s.carryOut(call)
	s.callOpened(call)
	model := s.model()
	call.edge.pickUp(call.to)
	s.Require().Eventually(func() bool { return model.turnsOn("<scheduled_task") == 1 },
		30*time.Second, 20*time.Millisecond, "the task was never said")
	_, err := s.router.sessions.Close(call.spec.ID, session.OwnerOf(call.spec))
	s.Require().NoError(err)
	outcome := s.outcomeOf(done)

	s.Require().NoError(outcome.err)
	s.Equal(store.ScheduledTaskDeliveredPhone, outcome.deliveredTo)
}

func (s *ScheduledTasksSuite) TestACallThatEndsBeforeAnybodyPicksUpIsNotAnswered() {
	s.shortCalls(time.Minute, time.Minute)
	call := s.placeCall(s.data.createUser())

	done := s.carryOut(call)
	s.callOpened(call)
	_, err := s.router.sessions.Close(call.spec.ID, session.OwnerOf(call.spec))
	s.Require().NoError(err)
	outcome := s.outcomeOf(done)

	var failure *scheduling.Failure
	s.Require().True(errors.As(outcome.err, &failure), "%v", outcome.err)
	s.Equal(scheduling.ReasonNotAnswered, failure.Reason, "%v", outcome.err)
	s.True(failure.Final)
}

func (s *ScheduledTasksSuite) TestACallWhoseSessionCannotOpenIsNotRungAgain() {
	call := s.placeCall(s.data.createUser())
	call.spec.AgentID = "another-agent"

	outcome := s.outcomeOf(s.carryOut(call))

	var failure *scheduling.Failure
	s.Require().True(errors.As(outcome.err, &failure), "%v", outcome.err)
	s.Equal(scheduling.ReasonFailed, failure.Reason)
	s.True(failure.Final, "the user was rung already")
}

// deliverNow has the router deliver the task's occurrence due at due once,
// on a run of its own, the way the worker hands it a claim.
func (s *ScheduledTasksSuite) deliverNow(task store.ScheduledTask, due time.Time, call string) error {
	run := store.ScheduledTaskRun{
		TaskID: task.ID, CustomerID: task.CustomerID, ScheduledFor: due, Trigger: store.ScheduledTaskTriggerManual,
		Prompt: task.Instructions, State: store.ScheduledTaskRunRunning, Attempts: 1, StartedAt: &due, SessionID: call,
	}
	s.Require().NoError(s.store.InsertScheduledTaskRun(context.Background(), &run))
	_, _, err := s.router.DeliverScheduledTask(context.Background(), store.ClaimedScheduledTask{Task: task, Run: run})
	return err
}

// textAndPhoneTask is a task delivered on channels, in order, for a user who
// has a WhatsApp identity but never wrote, so WhatsApp has no window open.
func (s *ScheduledTasksSuite) textAndPhoneTask(channels []string) store.ScheduledTask {
	number := s.lineNumber()
	s.connectWhatsApp(number)
	config := s.agentOnWhatsApp(number)
	user := s.data.createUser()
	s.Require().NoError(s.store.SaveChannelIdentity(context.Background(), &store.ChannelIdentity{
		CustomerID: s.customerID(), ConfigID: config, Kind: "whatsapp", Address: s.writerNumber(), UserID: user.userID,
	}))
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, channels)
	return s.tasksOf(user)[0]
}

func (s *ScheduledTasksSuite) TestAnOccurrenceThatRangAlreadyFailsFinally() {
	task := s.textAndPhoneTask([]string{"phone"})

	err := s.deliverNow(task, time.Now(), "the-earlier-call")

	var failure *scheduling.Failure
	s.Require().True(errors.As(err, &failure), "%v", err)
	s.Equal(scheduling.ReasonAlreadyCalled, failure.Reason)
	s.True(failure.Final)
}

func (s *ScheduledTasksSuite) TestACallThatCouldNotBePlacedIsTriedAgain() {
	task := s.textAndPhoneTask([]string{"phone"})

	err := s.deliverNow(task, time.Now(), "")

	var failure *scheduling.Failure
	s.Require().True(errors.As(err, &failure), "%v", err)
	s.Equal(scheduling.ReasonFailed, failure.Reason)
	s.False(failure.Final)
}

func (s *ScheduledTasksSuite) TestACallTooLateLeavesTheNextChannelToSayWhyItFailed() {
	task := s.textAndPhoneTask([]string{"phone", "whatsapp"})

	err := s.deliverNow(task, time.Now().Add(-20*time.Minute), "")

	var failure *scheduling.Failure
	s.Require().True(errors.As(err, &failure), "a channel that failed makes the occurrence failed, not missed: %v", err)
	s.Equal(scheduling.ReasonUnreachable, failure.Reason)
	s.False(failure.Final)
}

func (s *ScheduledTasksSuite) TestAnOccurrenceFailedOnEveryChannelGivesTheFirstChannelsReason() {
	task := s.textAndPhoneTask([]string{"whatsapp", "phone"})

	err := s.deliverNow(task, time.Now(), "")

	var failure *scheduling.Failure
	s.Require().True(errors.As(err, &failure), "%v", err)
	s.Equal(scheduling.ReasonUnreachable, failure.Reason)
	s.False(failure.Final, "neither channel's failure is final")
}

func (s *ScheduledTasksSuite) TestATaskRunByHandThatFailsGivesItsReason() {
	number := s.lineNumber()
	s.connectWhatsApp(number)
	config := s.agentOnWhatsApp(number)
	user := s.data.createUser()
	opened := s.openScheduling(user, config, nil)
	created := s.taskOn(user, config, opened.Id, []string{"phone"})
	path := "/v1/agents/scheduled-tasks/" + created.Id

	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, path+"/run", nil, nil))

	var page ScheduledTaskRunPage
	s.Require().Eventually(func() bool {
		return user.do(http.MethodGet, path+"/runs", nil, &page) == http.StatusOK &&
			len(page.Items) == 1 && page.Items[0].State == store.ScheduledTaskRunFailed
	}, 30*time.Second, 250*time.Millisecond, "the run by hand never failed")
	s.Require().NotNil(page.Items[0].Error)
	s.Equal(scheduling.ReasonNoNumber, *page.Items[0].Error)
}

func (s *ScheduledTasksSuite) TestACallThatOutlastsTheWaitIsTheUserOnACall() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	opened := s.openScheduling(user, config, func(request *CreateSessionRequest) {
		request.StartVoice = pointerTo(true)
	})
	found, live := s.router.sessions.Get(opened.Id, session.Owner{CustomerID: s.customerID(), UserID: user.userID, Kind: "authenticated"})
	s.Require().True(live)

	err := waitOutCall(context.Background(), found, time.Millisecond)

	var failure *scheduling.Failure
	s.Require().True(errors.As(err, &failure), "%v", err)
	s.Equal(scheduling.ReasonOnACall, failure.Reason)
	s.Require().Equal(http.StatusOK, user.do(http.MethodDelete, "/v1/agents/sessions/"+opened.Id+"/voice", nil, nil))
}
