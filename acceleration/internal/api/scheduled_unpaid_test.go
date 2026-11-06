//go:build integration

package api

import (
	"context"
	"errors"
	"log/slog"
	"sync/atomic"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/channelbridge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/channels"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// TelnyxUnpaidSuite is a Telnyx line whose customer's organization has its money read from a
// real usage service, so the gate refuses its texts as a router refuses an organization that may
// not pay: suspended, or with no credit.
type TelnyxUnpaidSuite struct {
	telnyxLine
}

func TestTelnyxUnpaidSuite(t *testing.T) {
	runSuite(t, new(TelnyxUnpaidSuite))
}

// SetupSuite gives the suite a database of its own, as its tests start a scheduled-task worker.
func (s *TelnyxUnpaidSuite) SetupSuite() {
	s.database = "telnyx_unpaid"
	s.limited = true
	s.telnyxLine.SetupSuite()
}

// refusedForMoney waits until the gate refuses the customer a text to person for money: an
// organization's money is read after its first admission, which it admits.
func (s *TelnyxUnpaidSuite) refusedForMoney(person string) {
	s.Require().Eventually(func() bool {
		return errors.Is(s.gate.Allow(context.Background(), s.customerID(), dlc.SMS, person), dlc.ErrUnpaid)
	}, 10*time.Second, 50*time.Millisecond, "the organization's money is read")
}

// suspendedThread has person text the line while the organization has credit, so the thread is
// linked and answered, then suspends the organization, and returns the thread channel.
func (s *TelnyxUnpaidSuite) suspendedThread(person string) string {
	s.usage.Credit(s.T(), s.RouterSuite.app.organization.ID, 10_000_000)
	s.Require().Eventually(func() bool {
		return s.gate.Allow(context.Background(), s.customerID(), dlc.SMS, person) == nil
	}, 10*time.Second, 50*time.Millisecond, "the credit is read")
	channel := s.smsTurn(person, "hi")
	s.usage.Suspend(s.T(), s.RouterSuite.app.organization.ID, "fraud review")
	s.refusedForMoney(person)
	return channel
}

// settledByTheWorker makes task due, has a worker deliver it with its retries at once, and
// returns its settled run and how many attempts the worker made.
func (s *TelnyxUnpaidSuite) settledByTheWorker(task store.ScheduledTask) (store.ScheduledTaskRun, int32) {
	s.retryAtOnce()
	var attempts atomic.Int32
	deliver := func(ctx context.Context, claimed store.ClaimedScheduledTask) (string, string, error) {
		attempts.Add(1)
		return s.router.DeliverScheduledTask(ctx, claimed)
	}
	logger := slog.New(slog.NewTextHandler(s.logged, nil))
	worker := scheduling.NewScheduler(s.store, deliver, logger)
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
	}, deliveryWait, 20*time.Millisecond, "the task's run never settled")
	return settled, attempts.Load()
}

// A task made in a thread of a suspended organization texts nothing and is not tried again: the
// run says the account is suspended, and the router warns, naming the customer.
func (s *TelnyxUnpaidSuite) TestAThreadTaskOfASuspendedOrganizationIsNotRetried() {
	person := "+13125550001"
	channel := s.suspendedThread(person)
	next := time.Now().Add(2 * time.Hour).UTC()
	user := s.chat.Stored(channel)[0]["user_id"].(string)
	task := store.ScheduledTask{CustomerID: s.customerID(), ConfigID: s.config.ID, UserID: user,
		CallerKind: "server", APIKey: s.RouterSuite.app.key, ChannelThread: channel, Title: "Stretch",
		Instructions: "Remind them to stretch.", StartLocal: next, Timezone: "Europe/Amsterdam", NextRunAt: &next,
		Channels: []string{"conversation"}}
	s.Require().NoError(s.store.CreateScheduledTask(context.Background(), &task, 0))

	run, attempts := s.settledByTheWorker(task)

	s.Equal(store.ScheduledTaskRunFailed, run.State)
	s.Require().NotNil(run.Error)
	s.Equal(codeAccountSuspended, *run.Error)
	s.Equal(int32(1), attempts, "a suspension outlasts every retry of the run")
	s.Len(s.telnyx().sent(), 1, "only the answer to hi")
	s.Contains(s.logged.String(), `level=WARN msg="held a channel message back: the organization may not pay for it" connector=telnyx customer=`+s.customerID())
}

// A reply the agent finished in a thread of a suspended organization is not sent, and is no
// failure to log: the gate's warning says why.
func (s *TelnyxUnpaidSuite) TestAReplyOfASuspendedOrganizationIsNotSentAndLogsNoError() {
	person := "+13125550002"
	channel := s.suspendedThread(person)

	bridge, built := s.bridge.(*channelbridge.Bridge)
	s.Require().True(built)

	bridge.Reply(conversation.FinishedReply{Customer: s.customerID(), CID: "agent:" + channel,
		MessageID: s.utils.uuid(), Text: "Time to stretch."})

	s.Eventually(func() bool { return bridge.Replying() == 0 }, settleFor, 10*time.Millisecond)
	s.Len(s.telnyx().sent(), 1, "only the answer to hi")
	s.Contains(s.logged.String(), `level=WARN msg="held a channel message back: the organization may not pay for it"`)
	s.NotContains(s.logged.String(), "could not send a reply to its external thread")
}

// A task delivered by SMS for an organization whose credit ran out is not tried again: the run
// says the credit is spent, and the router warns, naming the task.
func (s *TelnyxUnpaidSuite) TestAnSMSTaskOfAnOrganizationWithNoCreditIsNotRetried() {
	number, writer := "+1555"+digits(s.utils.uuid()), "+1347"+digits(s.utils.uuid())
	s.seedSMSLine(number)
	s.config.Channels = store.AgentChannels{SMS: &store.ChannelLine{Number: number}}
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), &s.config))
	user := s.data.createUser()
	s.Require().NoError(s.store.SaveChannelIdentity(context.Background(), &store.ChannelIdentity{
		CustomerID: s.customerID(), ConfigID: s.config.ID, Kind: string(channels.SMS), Address: writer, UserID: user.userID,
	}))
	// The task follows up in a session opened while the organization had credit, so what its
	// spent credit refuses is the text.
	organization := s.RouterSuite.app.organization.ID
	s.usage.Credit(s.T(), organization, 10_000_000)
	s.Require().Eventually(func() bool {
		return s.gate.Allow(context.Background(), s.customerID(), dlc.SMS, writer) == nil
	}, 10*time.Second, 50*time.Millisecond, "the credit is read")
	opened := user.createSession(CreateSessionRequest{ConfigId: &s.config.ID})
	s.usage.Credit(s.T(), organization, -10_000_000)
	s.refusedForMoney(writer)
	next := time.Now().Add(2 * time.Hour).UTC()
	task := store.ScheduledTask{CustomerID: s.customerID(), ConfigID: s.config.ID, SessionID: opened.Id, UserID: user.userID,
		CallerKind: "authenticated", APIKey: s.RouterSuite.app.key, Title: "Stretch", Instructions: "Remind them to stretch.",
		StartLocal: next, Timezone: "Europe/Amsterdam", NextRunAt: &next, Channels: []string{"sms"}}
	s.Require().NoError(s.store.CreateScheduledTask(context.Background(), &task, 0))

	run, attempts := s.settledByTheWorker(task)

	s.Equal(store.ScheduledTaskRunFailed, run.State)
	s.Require().NotNil(run.Error)
	s.Equal(codeInsufficientCredits, *run.Error)
	s.Equal(int32(1), attempts, "spent credit outlasts every retry of the run")
	s.Contains(s.logged.String(), `level=WARN msg="a scheduled task failed to deliver" task=`+task.ID)
}
