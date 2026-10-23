//go:build integration

package api

import (
	"context"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// AI-1842: a thread on a line of one person (SMS on Telnyx, WhatsApp, iMessage on Linq), on a
// config with scheduling on, is offered the scheduling tools. A task made there belongs to the
// person who wrote, the Stream user the bridge writes them as, and keeps its thread channel. A
// Slack thread is shared, so it is offered none.

// stretchInTwoHours is a schedule_task call the scheduling model makes.
const stretchInTwoHours = `{"title":"Stretch","instructions":"Remind them to stretch.","in":"PT2H"}`

// schedulingTools are the four tools a session that may schedule is offered.
var schedulingTools = []string{"schedule_task", "list_scheduled_tasks", "update_scheduled_task", "cancel_scheduled_task"}

// schedulingModelOn has config answer with the scheduling model, scheduling as it is, and
// forgets the models earlier tests of the suite opened and what they queued.
func (s *RouterSuite) schedulingModelOn(config *store.AgentConfig) {
	s.scheduledMu.Lock()
	s.scheduled = nil
	s.scheduledMu.Unlock()
	s.schedulingScript.reset()
	config.LLM = "scheduling/scheduling-model"
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), config))
}

// schedulingOn has config answer with the scheduling model, scheduling on.
func (s *RouterSuite) schedulingOn(config *store.AgentConfig) {
	config.Scheduling.Enabled, config.Scheduling.DefaultTimezone = true, "Europe/Amsterdam"
	s.schedulingModelOn(config)
}

// planTool queues one tool call for the next turn somebody starts.
func (s *RouterSuite) planTool(name, arguments string) {
	s.schedulingScript.plan(llm.ToolCall{ID: store.NewID(), Name: name, Arguments: arguments})
}

// customersTasks waits until the test's customer holds count tasks, and returns them.
func (s *RouterSuite) customersTasks(count int) []store.ScheduledTask {
	var tasks []store.ScheduledTask
	s.Require().Eventually(func() bool {
		var err error
		tasks, err = s.store.ScheduledTasks(context.Background(), s.customerID(), store.ScheduledTaskFilter{}, 25, nil)
		s.Require().NoError(err)
		return len(tasks) >= count
	}, settleFor, 10*time.Millisecond, "the customer holds no %d tasks", count)
	return tasks
}

// everyToolOffered is every tool name any scheduling model was offered: a thread's turn and
// its episode summary each open one.
func (s *RouterSuite) everyToolOffered() []string {
	s.scheduledMu.Lock()
	models := slices.Clone(s.scheduled)
	s.scheduledMu.Unlock()
	s.Require().NotEmpty(models, "no session opened the scheduling model")
	offered := []string{}
	for _, model := range models {
		for _, name := range model.toolsOffered() {
			if !slices.Contains(offered, name) {
				offered = append(offered, name)
			}
		}
	}
	return offered
}

// repliesIn counts the agent's replies in a thread channel: what it writes as the channel's
// own user.
func (s *RouterSuite) repliesIn(channel string) int {
	replies := 0
	for _, stored := range s.chat.Stored(channel) {
		if stored["user_id"] == channel {
			replies++
		}
	}
	return replies
}

// answeredAfter waits until a thread channel holds more than before of the agent's replies.
// The reply the bridge sends is not waited for: one of the suite's previous test may still
// reach this test's provider.
func (s *RouterSuite) answeredAfter(channel string, before int) {
	s.Require().Eventually(func() bool { return s.repliesIn(channel) > before }, settleFor, 10*time.Millisecond,
		"the agent wrote no reply in %s", channel)
}

// toldTheModel waits until a scheduling model of the test was sent text holding needle,
// such as a tool's answer.
func (s *RouterSuite) toldTheModel(needle string) {
	s.Require().Eventually(func() bool {
		s.scheduledMu.Lock()
		defer s.scheduledMu.Unlock()
		return slices.ContainsFunc(s.scheduled, func(model *schedulingModel) bool { return model.heard(needle) })
	}, settleFor, 10*time.Millisecond, "the model was never told %s", needle)
}

// smsTurn has person text the line and the Router answer it: the message is written into
// the person's thread channel, Stream Chat's message.new for it is delivered, and the
// agent's reply is sent, so none reaches the suite's next test's Telnyx.
func (s *telnyxLine) smsTurn(person, text string) string {
	channel := ""
	before, replies, sent := 0, 0, len(s.telnyx.sent())
	if s.hasThread(person) {
		channel = s.threadChannel(person)
		before, replies = len(s.chat.Stored(channel)), s.repliesIn(channel)
	}
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, person, text), time.Now()))
	channel = s.threadChannel(person)
	stored := s.written(channel, before+1)
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, len(stored)-1))
	s.answeredAfter(channel, replies)
	s.took(sent + 1)
	return channel
}

// hasThread is whether the test's customer has a thread channel for person yet.
func (s *telnyxLine) hasThread(person string) bool {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM channel_threads WHERE customer_id = ? AND connector_id = 'telnyx' AND thread_key = ?",
		s.customerID(), person).Scan(&count))
	return count > 0
}

// A person texting the line is offered the tools, and the task they make is theirs, made by
// the agent, in their thread's conversation, keeping the thread channel.
func (s *TelnyxChannelSuite) TestAPersonTextingTheLineSchedulesATaskThatKeepsTheirThread() {
	s.schedulingOn(&s.config)
	s.planTool("schedule_task", stretchInTwoHours)

	channel := s.smsTurn("+13125550001", "remind me to stretch in two hours")

	task := s.customersTasks(1)[0]
	s.Equal(s.chat.Stored(channel)[0]["user_id"], task.UserID, "the person the bridge writes the texts as")
	s.Equal("server", task.CallerKind)
	s.Equal(channel, task.ChannelThread)
	s.Equal("agent:"+channel, task.ConversationID)
	s.Equal(s.config.ID, task.ConfigID)
	s.Equal("agent", task.CreatedBy)
	s.NotEmpty(task.SessionID)
	offered := s.everyToolOffered()
	for _, name := range schedulingTools {
		s.Contains(offered, name)
	}
}

// Two people texting one line hold tasks of their own: one lists none of the other's and
// cannot cancel it, and the owner lists and cancels it from a later turn.
func (s *TelnyxChannelSuite) TestTwoPeopleOnOneLineNeverReachEachOthersTasks() {
	s.schedulingOn(&s.config)
	alice, bob := "+13125550001", "+13125550002"
	s.planTool("schedule_task", stretchInTwoHours)
	s.smsTurn(alice, "remind me to stretch in two hours")
	task := s.customersTasks(1)[0]

	s.planTool("list_scheduled_tasks", `{}`)
	s.smsTurn(bob, "what have I got coming up")
	s.toldTheModel(`"tasks":[]`)
	s.planTool("cancel_scheduled_task", `{"id":"`+task.ID+`"}`)
	s.smsTurn(bob, "cancel "+task.ID)
	s.toldTheModel(`"error":"scheduled_task_not_found"`)
	_, err := s.store.ScheduledTask(context.Background(), s.customerID(), task.ID)
	s.Require().NoError(err, "the other person's cancel left the task")

	s.planTool("list_scheduled_tasks", `{}`)
	s.smsTurn(alice, "what have I got coming up")
	s.toldTheModel(`"tasks":[{"id":"` + task.ID + `"`)
	s.planTool("cancel_scheduled_task", `{"id":"`+task.ID+`"}`)
	s.smsTurn(alice, "never mind")
	s.toldTheModel(`"cancelled":"Stretch"`)
	_, err = s.store.ScheduledTask(context.Background(), s.customerID(), task.ID)
	s.ErrorIs(err, store.ErrNoScheduledTask)
}

// The same person texting the same number of another customer reaches none of the first
// customer's tasks.
func (s *TelnyxChannelSuite) TestTheSamePersonOnAnotherCustomersLineReachesNoneOfTheTasks() {
	person := "+13125550001"
	s.schedulingOn(&s.config)
	s.planTool("schedule_task", stretchInTwoHours)
	s.smsTurn(person, "remind me to stretch in two hours")
	task := s.customersTasks(1)[0]
	first := s.customerID()

	s.SetupTest()
	s.Require().NotEqual(first, s.customerID())
	s.schedulingOn(&s.config)
	s.planTool("list_scheduled_tasks", `{}`)
	s.smsTurn(person, "what have I got coming up")
	s.toldTheModel(`"tasks":[]`)
	s.planTool("cancel_scheduled_task", `{"id":"`+task.ID+`"}`)
	s.smsTurn(person, "cancel "+task.ID)
	s.toldTheModel(`"error":"scheduled_task_not_found"`)

	_, err := s.store.ScheduledTask(context.Background(), first, task.ID)
	s.NoError(err, "the other customer's cancel left the task")
}

// A zone the person named for one task is theirs for the next, made in a later turn.
func (s *TelnyxChannelSuite) TestAZoneThePersonNamedIsTheirsForTheirNextTask() {
	s.schedulingOn(&s.config)
	person := "+13125550001"
	s.planTool("schedule_task", `{"title":"Stretch","instructions":"Remind them to stretch.","in":"PT2H","timezone":"America/New_York"}`)
	s.smsTurn(person, "remind me to stretch in two hours, I am in New York")
	s.customersTasks(1)

	s.planTool("schedule_task", `{"title":"Water","instructions":"Remind them to drink water.","in":"PT3H"}`)
	s.smsTurn(person, "and to drink water in three hours")

	for _, task := range s.customersTasks(2) {
		s.Equal("America/New_York", task.Timezone, task.Title)
	}
	s.scheduledMu.Lock()
	defer s.scheduledMu.Unlock()
	s.True(slices.ContainsFunc(s.scheduled, func(model *schedulingModel) bool {
		return strings.Contains(model.instructions(), "for the user (America/New_York).")
	}), "the later turn's clock line is in the person's zone")
}

// The person's limit counts their own tasks: with one allowed, a paused task is not resumed
// beside the one they made since.
func (s *TelnyxChannelSuite) TestAPersonsLimitCountsTheirOwnTasksOnResume() {
	s.config.Scheduling.MaxTasksPerUser = 1
	s.schedulingOn(&s.config)
	person := "+13125550001"
	s.planTool("schedule_task", stretchInTwoHours)
	s.smsTurn(person, "remind me to stretch in two hours")
	paused := s.customersTasks(1)[0]
	s.planTool("update_scheduled_task", `{"id":"`+paused.ID+`","action":"pause"}`)
	s.smsTurn(person, "pause it")
	s.planTool("schedule_task", `{"title":"Water","instructions":"Remind them to drink water.","in":"PT3H"}`)
	s.smsTurn(person, "remind me to drink water in three hours")
	s.customersTasks(2)

	s.planTool("update_scheduled_task", `{"id":"`+paused.ID+`","action":"resume"}`)
	s.smsTurn(person, "resume the stretch one")

	s.toldTheModel(`"error":"too_many_tasks"`)
	stored, err := s.store.ScheduledTask(context.Background(), s.customerID(), paused.ID)
	s.Require().NoError(err)
	s.Equal(store.ScheduledTaskPaused, stored.State)
}

// The control: a thread on a config without scheduling is offered what it was before
// AI-1842. The expected list is a probe of this test on the base (bd4b4fe93).
func (s *TelnyxChannelSuite) TestAThreadOnAConfigWithoutSchedulingIsOfferedWhatItWasBefore() {
	s.schedulingModelOn(&s.config)

	s.smsTurn("+13125550001", "remind me to stretch in two hours")

	s.Equal([]string{}, s.everyToolOffered())
	tasks, err := s.store.ScheduledTasks(context.Background(), s.customerID(), store.ScheduledTaskFilter{}, 25, nil)
	s.Require().NoError(err)
	s.Empty(tasks)
}

// A WhatsApp thread schedules for the person who wrote, keeping the thread channel.
func (s *WhatsAppChannelSuite) TestAWhatsAppThreadSchedulesForThePersonWhoWrote() {
	s.schedulingOn(&s.config)
	s.planTool("schedule_task", stretchInTwoHours)
	person := "16505551234"

	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.unit, person, "remind me to stretch in two hours")))
	channel := s.threadChannel(person)
	s.written(channel, 1)
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))

	task := s.customersTasks(1)[0]
	s.Equal(s.chat.Stored(channel)[0]["user_id"], task.UserID)
	s.Equal(channel, task.ChannelThread)
	// The reply is sent before the test ends, so it reaches no later test's Meta.
	s.took(1)
}

// Two senders in one Linq chat hold tasks of their own: the second lists none of the first's.
func (s *LinqChannelSuite) TestTwoSendersInOneChatHoldTasksOfTheirOwn() {
	s.schedulingOn(&s.config)
	chat := s.utils.uuid()
	s.planTool("schedule_task", stretchInTwoHours)
	s.Require().Equal(http.StatusOK, s.deliver(s.received(chat, s.line, "+13125550001", "remind me to stretch in two hours"), time.Now()))
	channel := s.threadChannel(chat)
	s.written(channel, 1)
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))
	s.answeredAfter(channel, 0)
	task := s.customersTasks(1)[0]
	s.Equal(s.chat.Stored(channel)[0]["user_id"], task.UserID)
	s.Equal(channel, task.ChannelThread)
	before := len(s.chat.Stored(channel))

	s.planTool("list_scheduled_tasks", `{}`)
	s.Require().Equal(http.StatusOK, s.deliver(s.received(chat, s.line, "+13125550002", "what have I got coming up"), time.Now()))
	stored := s.written(channel, before+1)
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, len(stored)-1))

	s.toldTheModel(`"tasks":[]`)
	// Both replies are sent before the test ends, so they reach no later test's Linq.
	s.took(2)
}

// A Slack thread is shared by the channel's people, so it is offered no scheduling tools.
func (s *SlackChannelSuite) TestASlackThreadIsNotOfferedTheTools() {
	s.schedulingOn(&s.config)

	channel := s.messaged("U0000ALICE", "remind me to stretch in two hours", "1759740000.000100", "")
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))
	s.answeredAfter(channel, 0)
	s.posted(1)

	offered := s.everyToolOffered()
	for _, name := range schedulingTools {
		s.False(slices.Contains(offered, name), name)
	}
}
