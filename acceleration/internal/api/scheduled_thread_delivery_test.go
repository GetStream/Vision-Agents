//go:build integration

package api

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A task made in an SMS or WhatsApp thread is carried out in that thread when it comes due:
// the Router opens the thread's conversation, the agent answers the task there, and the
// bridge sends the reply to the person on the line.

// manualRun is a run of task made by hand, as the worker would hand it over claimed.
func (s *RouterSuite) manualRun(task store.ScheduledTask) store.ScheduledTaskRun {
	now := time.Now()
	run := store.ScheduledTaskRun{
		TaskID: task.ID, CustomerID: task.CustomerID, ScheduledFor: now, Trigger: store.ScheduledTaskTriggerManual,
		Prompt: task.Instructions, State: store.ScheduledTaskRunRunning, Attempts: 1, StartedAt: &now,
	}
	s.Require().NoError(s.store.InsertScheduledTaskRun(context.Background(), &run))
	return run
}

// deliverTask has the router deliver an occurrence of task once, on a run made by hand, and
// returns the run so a retry can be handed it again.
func (s *RouterSuite) deliverTask(task store.ScheduledTask) (store.ScheduledTaskRun, string, error) {
	run := s.manualRun(task)
	deliveredTo, err := s.deliverRun(task, run)
	return run, deliveredTo, err
}

// deliverRun has the router deliver a run already made, as a retry does.
func (s *RouterSuite) deliverRun(task store.ScheduledTask, run store.ScheduledTaskRun) (string, error) {
	deliveredTo, _, err := s.router.DeliverScheduledTask(context.Background(), store.ClaimedScheduledTask{Task: task, Run: run})
	return deliveredTo, err
}

// scheduledTurnTools is every tool a due task's turn was offered.
func (s *RouterSuite) scheduledTurnTools() []string {
	s.scheduledMu.Lock()
	defer s.scheduledMu.Unlock()
	var names []string
	for _, model := range s.scheduled {
		model.mu.Lock()
		for _, asked := range model.asked {
			if strings.HasPrefix(startedBy(asked), "<scheduled_task") {
				for _, tool := range asked.Tools {
					names = append(names, tool.Name)
				}
			}
		}
		model.mu.Unlock()
	}
	return names
}

// deliveredByTheWorker makes task due, has a worker deliver it, and returns its settled run.
func (s *RouterSuite) deliveredByTheWorker(task store.ScheduledTask) store.ScheduledTaskRun {
	worker := scheduling.NewScheduler(s.store, s.router.DeliverScheduledTask, slog.New(slog.DiscardHandler))
	worker.Start(context.Background())
	defer worker.Close()
	past := time.Now().Add(-time.Minute)
	task.NextRunAt = &past
	s.Require().NoError(s.store.SaveScheduledTask(context.Background(), &task))
	worker.Kick()
	var settled store.ScheduledTaskRun
	s.Require().Eventually(func() bool {
		runs, err := s.store.ScheduledTaskRuns(context.Background(), task.CustomerID, task.ID, 25, nil)
		s.Require().NoError(err)
		if len(runs) == 0 || runs[0].State == store.ScheduledTaskRunRunning {
			return false
		}
		settled = runs[0]
		return true
	}, settleFor, 20*time.Millisecond, "the task's run never settled")
	return settled
}

// failedAs checks err is a failure recorded under reason, final or not as final says.
func (s *RouterSuite) failedAs(err error, reason string, final bool) {
	var failure *scheduling.Failure
	s.Require().True(errors.As(err, &failure), "not a recorded failure: %v", err)
	s.Equal(reason, failure.Reason)
	s.Equal(final, failure.Final, "final")
}

// scheduledTurns counts the turns a due task started across the test's scheduling models: the
// ones it was sent as is, not an episode summary quoting it.
func (s *RouterSuite) scheduledTurns() int {
	s.scheduledMu.Lock()
	defer s.scheduledMu.Unlock()
	turns := 0
	for _, model := range s.scheduled {
		model.mu.Lock()
		for _, asked := range model.asked {
			if strings.HasPrefix(startedBy(asked), "<scheduled_task") {
				turns++
			}
		}
		model.mu.Unlock()
	}
	return turns
}

// threadOpen is whether a session is running on a thread channel.
func (s *RouterSuite) threadOpen(channel string) bool {
	_, running := s.router.sessions.ByConversationWhere("agent:"+channel, nil)
	return running
}

// scheduledInThread has person text the line and schedule a task, and returns the task and
// its thread channel.
func (s *telnyxLine) scheduledInThread(person string) (store.ScheduledTask, string) {
	s.schedulingOn(&s.config)
	s.planTool("schedule_task", stretchInTwoHours)
	channel := s.smsTurn(person, "remind me to stretch in two hours")
	return s.customersTasks(1)[0], channel
}

// The production path: the worker claims the task, the thread's conversation is opened, the
// agent answers the task there, the bridge texts the reply to the person from the line, and
// the run says it went by SMS. The session opened for it is closed.
func (s *TelnyxChannelSuite) TestATaskMadeInAnSMSThreadIsTextedBackOnTheThread() {
	person := "+13125550001"
	task, channel := s.scheduledInThread(person)
	sent, replies := len(s.telnyx.sent()), s.repliesIn(channel)

	run := s.deliveredByTheWorker(task)

	s.Equal(store.ScheduledTaskRunDelivered, run.State, "error %v", run.Error)
	s.Equal("sms", run.DeliveredTo)
	s.NotEmpty(run.SessionID)
	texted := s.took(sent + 1)[sent]
	s.Equal(telnyxMessage{from: s.line, to: person, text: "Done."}, texted.withoutAuthorization())
	s.Equal(1, s.scheduledTurns())
	s.Contains(s.scheduledTurnTools(), "schedule_task", "the turn answers the person, who may schedule again")
	s.Equal(replies+1, s.repliesIn(channel), "the reply is kept in the thread")
	s.Eventually(func() bool { return !s.threadOpen(channel) }, settleFor, 10*time.Millisecond, "the session was left open")
}

// A turn another router is taking in the thread is waited for, and the thread's turn is let go
// of after the task's.
func (s *TelnyxChannelSuite) TestATaskWaitsForTheThreadsTurnAndLetsGoOfIt() {
	task, channel := s.scheduledInThread("+13125550001")
	sent := len(s.telnyx.sent())
	ctx := context.Background()
	taken, err := s.store.TakeChannelThreadTurn(ctx, channel, "another-router", time.Now().Add(time.Minute))
	s.Require().NoError(err)
	s.Require().True(taken)
	run := s.manualRun(task)
	done := make(chan error, 1)
	go func() {
		_, err := s.deliverRun(task, run)
		done <- err
	}()

	s.Never(func() bool { return s.scheduledTurns() > 0 }, dropped, 20*time.Millisecond, "the task ran in another router's turn")
	s.Require().NoError(s.store.ReleaseChannelThreadTurn(ctx, channel, "another-router"))

	select {
	case err := <-done:
		s.Require().NoError(err)
	case <-time.After(settleFor):
		s.FailNow("the task was not delivered once the turn was free")
	}
	s.took(sent + 1)
	taken, err = s.store.TakeChannelThreadTurn(ctx, channel, "the-next-router", time.Now().Add(time.Minute))
	s.Require().NoError(err)
	s.True(taken, "the delivery let go of the thread's turn")
}

// A session the customer's backend opened on the thread carries the task, and is left running.
func (s *TelnyxChannelSuite) TestATaskIsCarriedInTheSessionTheBackendKeepsOnTheThread() {
	task, channel := s.scheduledInThread("+13125550001")
	held := s.serverClient.createSession(CreateSessionRequest{Agent: &s.config.Name, AgentId: &channel})
	sent := len(s.telnyx.sent())

	deliveredTo, sessionID, err := s.router.DeliverScheduledTask(context.Background(),
		store.ClaimedScheduledTask{Task: task, Run: s.manualRun(task)})

	s.Require().NoError(err)
	s.Equal("sms", deliveredTo)
	s.Equal(held.Id, sessionID)
	s.took(sent + 1)
	s.Nil(s.serverClient.getSession(held.Id).ClosedAt, "the backend's session is still running")
}

// A retried occurrence finds its turn in the thread and does not run or text it again.
func (s *TelnyxChannelSuite) TestARetriedOccurrenceIsNotTextedTwice() {
	task, _ := s.scheduledInThread("+13125550001")
	sent := len(s.telnyx.sent())
	run, deliveredTo, err := s.deliverTask(task)
	s.Require().NoError(err)
	s.Equal("sms", deliveredTo)
	s.took(sent + 1)

	deliveredTo, err = s.deliverRun(task, run)

	s.Require().NoError(err)
	s.Equal("sms", deliveredTo)
	s.Equal(1, s.scheduledTurns())
	s.Never(func() bool { return len(s.telnyx.sent()) > sent+1 }, dropped, 20*time.Millisecond)
}

// A turn that fails texts nothing, and the run may be tried again.
func (s *TelnyxChannelSuite) TestATaskWhoseTurnFailsTextsNothing() {
	task, _ := s.scheduledInThread("+13125550001")
	s.schedulingScript.failScheduled(1)
	sent := len(s.telnyx.sent())

	_, _, err := s.deliverTask(task)

	s.failedAs(err, scheduling.ReasonReplyFailed, false)
	s.Never(func() bool { return len(s.telnyx.sent()) > sent }, dropped, 20*time.Millisecond)
}

// A person who texted STOP is texted nothing, no turn is paid for, and the run is final.
func (s *TelnyxChannelSuite) TestAPersonWhoOptedOutIsNotTextedAndTheTaskIsNotRetried() {
	person := "+13125550001"
	task, _ := s.scheduledInThread(person)
	s.Require().NoError(s.store.OptOut(context.Background(), &store.OptOut{
		CustomerID: s.customerID(), Recipient: person, Channel: "sms", Source: "keyword",
	}))
	sent := len(s.telnyx.sent())

	_, _, err := s.deliverTask(task)

	s.failedAs(err, scheduling.ReasonUnreachable, true)
	s.Zero(s.scheduledTurns())
	s.Never(func() bool { return len(s.telnyx.sent()) > sent }, dropped, 20*time.Millisecond)
}

// A line whose connection was removed fails the task for good.
func (s *TelnyxChannelSuite) TestATaskOnARemovedLineFailsFinally() {
	task, _ := s.scheduledInThread("+13125550001")
	s.Require().NoError(s.store.DeleteConnectorConnection(context.Background(), s.customerID(), s.connection.ID))

	_, _, err := s.deliverTask(task)

	s.failedAs(err, scheduling.ReasonConnectionRemoved, true)
	s.Zero(s.scheduledTurns())
}

// A thread that no longer exists fails the task for good.
func (s *TelnyxChannelSuite) TestATaskWhoseThreadIsGoneFailsFinally() {
	task, _ := s.scheduledInThread("+13125550001")
	task.ChannelThread = "thread-" + s.utils.uuid()

	_, _, err := s.deliverTask(task)

	s.failedAs(err, scheduling.ReasonThreadGone, true)
	s.Zero(s.scheduledTurns())
}

// One person's task never reaches another person's thread on the same line.
func (s *TelnyxChannelSuite) TestATaskNeverReachesAnotherPersonsThread() {
	task, _ := s.scheduledInThread("+13125550001")
	s.smsTurn("+13125550002", "hello")
	task.ChannelThread = s.threadChannel("+13125550002")
	sent := len(s.telnyx.sent())

	_, _, err := s.deliverTask(task)

	s.failedAs(err, scheduling.ReasonThreadGone, true)
	s.Zero(s.scheduledTurns())
	s.Never(func() bool { return len(s.telnyx.sent()) > sent }, dropped, 20*time.Millisecond)
}

// The same person texting two customers' lines has a thread with each; a task reaches only its
// own customer's, even when it names the other customer's thread and person.
func (s *TelnyxChannelSuite) TestATaskReachesOnlyItsOwnCustomersThread() {
	person := "+13125550001"
	first, firstChannel := s.scheduledInThread(person)
	firstReplies := s.repliesIn(firstChannel)

	s.SetupTest()
	s.Require().NotEqual(first.CustomerID, s.customerID())
	second, secondChannel := s.scheduledInThread(person)
	forged := second
	forged.UserID, forged.ChannelThread = first.UserID, firstChannel
	_, _, err := s.deliverTask(forged)
	s.failedAs(err, scheduling.ReasonThreadGone, true)
	sent := len(s.telnyx.sent())

	_, deliveredTo, err := s.deliverTask(second)

	s.Require().NoError(err)
	s.Equal("sms", deliveredTo)
	s.Equal(person, s.took(sent + 1)[sent].to)
	s.Equal(1, s.scheduledTurns())
	s.Equal(firstReplies, s.repliesIn(firstChannel), "the first customer's thread was written nothing")
	s.NotEqual(firstChannel, secondChannel)
}

// A thread task is delivered on its thread, so its deliver_to cannot be changed; its other
// fields can.
func (s *TelnyxChannelSuite) TestAThreadTasksDeliverToCannotChange() {
	task, _ := s.scheduledInThread("+13125550001")
	path := "/v1/agents/scheduled-tasks/" + task.ID

	status, raw := s.serverClient.call(http.MethodPatch, path, UpdateScheduledTaskRequest{DeliverTo: pointerTo([]string{"phone"})})

	s.Equal(http.StatusBadRequest, status, string(raw))
	s.Contains(string(raw), "a task made in a thread is delivered on that thread")
	stored, err := s.store.ScheduledTask(context.Background(), task.CustomerID, task.ID)
	s.Require().NoError(err)
	s.Equal(task.Channels, stored.Channels)
	var renamed ScheduledTask
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, path,
		UpdateScheduledTaskRequest{Title: pointerTo("Stand up")}, &renamed))
	s.Equal("Stand up", renamed.Title)
}

// The control: a task an app user's backend made, with no thread, on the same config, is
// delivered as on base: into a conversation of its own, with nothing texted on the line. The
// expected values are a probe of this test on the base (753c55936).
func (s *TelnyxChannelSuite) TestATaskWithoutAThreadIsDeliveredAsBefore() {
	s.schedulingOn(&s.config)
	next := time.Now().Add(2 * time.Hour).UTC()
	task := store.ScheduledTask{CustomerID: s.customerID(), ConfigID: s.config.ID, UserID: "alice", CallerKind: "server",
		Title: "Stretch", Instructions: "Remind them to stretch.", StartLocal: next, Timezone: "Europe/Amsterdam",
		NextRunAt: &next, Channels: []string{"conversation"}}
	s.Require().NoError(s.store.CreateScheduledTask(context.Background(), &task, 0))

	run := s.deliveredByTheWorker(task)

	s.Equal(store.ScheduledTaskRunDelivered, run.State)
	s.Equal("conversation", run.DeliveredTo)
	s.Nil(run.Error)
	s.Never(func() bool { return len(s.telnyx.sent()) > 0 }, dropped, 20*time.Millisecond)
}

// A sandboxed customer past its day's one message is texted nothing; the run may be tried
// again, as the limit lifts the next day.
func (s *TelnyxSandboxSuite) TestATaskPastTheSandboxsDailyLimitIsHeldBack() {
	s.Require().Equal(200, s.deliver(s.received(s.line, sandboxRecipient, "first"), time.Now()))
	channel := s.threadChannel(sandboxRecipient)
	stored := s.written(channel, 1)
	s.Require().Equal(200, s.streamDelivers(channel, 0))
	s.took(1)
	next := time.Now().Add(2 * time.Hour).UTC()
	task := store.ScheduledTask{CustomerID: s.customerID(), ConfigID: s.config.ID, UserID: stored[0]["user_id"].(string),
		CallerKind: "server", ChannelThread: channel, Title: "Stretch", Instructions: "Remind them to stretch.",
		StartLocal: next, Timezone: "Europe/Amsterdam", NextRunAt: &next, Channels: []string{"conversation"}}
	s.Require().NoError(s.store.CreateScheduledTask(context.Background(), &task, 0))

	_, _, err := s.deliverTask(task)

	s.failedAs(err, scheduling.ReasonUnreachable, false)
	s.Never(func() bool { return len(s.telnyx.sent()) > 1 }, dropped, 20*time.Millisecond)
}

// scheduledOnWhatsApp has person write on WhatsApp and schedule a task, and returns the task
// and its thread channel.
func (s *WhatsAppChannelSuite) scheduledOnWhatsApp(person string) (store.ScheduledTask, string) {
	s.schedulingOn(&s.config)
	s.planTool("schedule_task", stretchInTwoHours)
	s.Require().Equal(200, s.deliver(s.received(s.unit, person, "remind me to stretch in two hours")))
	channel := s.threadChannel(person)
	s.written(channel, 1)
	s.Require().Equal(200, s.streamDelivers(channel, 0))
	task := s.customersTasks(1)[0]
	s.took(1)
	return task, channel
}

// Within the day after the person last wrote, the task's reply is sent on WhatsApp.
func (s *WhatsAppChannelSuite) TestAWhatsAppThreadTaskWithinTheWindowIsSentOnTheThread() {
	person := "16505551234"
	task, _ := s.scheduledOnWhatsApp(person)

	_, deliveredTo, err := s.deliverTask(task)

	s.Require().NoError(err)
	s.Equal("whatsapp", deliveredTo)
	sent := s.took(2)[1]
	s.Equal("+"+person, sent.to)
	s.Equal("Done.", sent.text)
}

// A day after the person last wrote, WhatsApp takes no free text, and the manifest has no
// template: nothing is sent, no turn is paid for, and the run says why. It is final: only the
// person writing opens the window again.
func (s *WhatsAppChannelSuite) TestAWhatsAppThreadTaskPastTheWindowIsNotSent() {
	task, channel := s.scheduledOnWhatsApp("16505551234")
	_, err := s.store.DB().ExecContext(context.Background(),
		"UPDATE channel_threads SET last_inbound_at = now() - interval '25 hours' WHERE channel_id = ?", channel)
	s.Require().NoError(err)

	_, _, err = s.deliverTask(task)

	s.failedAs(err, scheduling.ReasonOutsideWindow, true)
	s.Zero(s.scheduledTurns())
	s.Never(func() bool { return len(s.meta.sent()) > 1 }, dropped, 20*time.Millisecond)
}
