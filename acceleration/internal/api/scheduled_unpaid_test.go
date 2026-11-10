//go:build integration

package api

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/channelbridge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/channels"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// TelnyxUnpaidSuite is a Telnyx line whose customer's organization has its money read from a
// real usage service, so the gate refuses its texts as a router refuses an organization that may
// not pay: suspended, or with no credit.
type TelnyxUnpaidSuite struct {
	telnyxLine
	// sharing is the Redis the router shares its holds through, behind a proxy a test can take
	// down, so its money cannot be checked.
	sharing *redisProxy
}

func TestTelnyxUnpaidSuite(t *testing.T) {
	runSuite(t, new(TelnyxUnpaidSuite))
}

// SetupSuite gives the suite a database of its own, as its tests start a scheduled-task worker.
func (s *TelnyxUnpaidSuite) SetupSuite() {
	s.database = "telnyx_unpaid"
	s.limited = true
	s.sharing, s.holds = newRedisProxy(s.T())
	s.telnyxLine.SetupSuite()
}

func (s *TelnyxUnpaidSuite) SetupTest() {
	s.sharing.down.Store(false)
	s.telnyxLine.SetupTest()
}

// refusedForMoney waits until the gate refuses the customer a text to person for money: an
// organization's money is read after its first admission, which it admits.
func (s *TelnyxUnpaidSuite) refusedForMoney(person string) {
	s.Require().Eventually(func() bool {
		return errors.Is(s.gate.Allow(context.Background(), s.customerID(), dlc.SMS, person), dlc.ErrUnpaid)
	}, 10*time.Second, 50*time.Millisecond, "the organization's money is read")
}

// creditedThread has person text the line while the organization has credit, so the thread is
// linked and answered, and returns the thread channel.
func (s *TelnyxUnpaidSuite) creditedThread(person string) string {
	s.usage.Credit(s.T(), s.RouterSuite.app.organization.ID, 10_000_000)
	s.Require().Eventually(func() bool {
		return s.gate.Allow(context.Background(), s.customerID(), dlc.SMS, person) == nil
	}, 10*time.Second, 50*time.Millisecond, "the credit is read")
	return s.smsTurn(person, "hi")
}

// suspendedThread is creditedThread, with the organization suspended afterwards.
func (s *TelnyxUnpaidSuite) suspendedThread(person string) string {
	channel := s.creditedThread(person)
	s.usage.Suspend(s.T(), s.RouterSuite.app.organization.ID, "fraud review")
	s.refusedForMoney(person)
	return channel
}

// threadTask is a task made in the thread channel, by the person who wrote in it.
func (s *TelnyxUnpaidSuite) threadTask(channel string) store.ScheduledTask {
	user := s.chat.Stored(channel)[0]["user_id"].(string)
	next := time.Now().Add(2 * time.Hour).UTC()
	task := store.ScheduledTask{CustomerID: s.customerID(), ConfigID: s.config.ID, UserID: user,
		CallerKind: "server", APIKey: s.RouterSuite.app.key, ChannelThread: channel, Title: "Stretch",
		Instructions: "Remind them to stretch.", StartLocal: next, Timezone: "Europe/Amsterdam", NextRunAt: &next,
		Channels: []string{"conversation"}}
	s.Require().NoError(s.store.CreateScheduledTask(context.Background(), &task, 0))
	return task
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
	task := s.threadTask(s.suspendedThread(person))

	run, attempts := s.settledByTheWorker(task)

	s.Equal(store.ScheduledTaskRunFailed, run.State)
	s.Require().NotNil(run.Error)
	s.Equal(codeAccountSuspended, *run.Error)
	s.Equal(int32(1), attempts, "a suspension outlasts every retry of the run")
	s.Len(s.telnyx().sent(), 1, "only the answer to hi")
	s.Contains(s.logged.String(), `level=WARN msg="held a channel message back: the organization may not pay for it" connector=telnyx customer=`+s.customerID())
}

// A thread task whose organization's money cannot be checked, as when the router cannot share
// what it holds with the others, is tried again: the check passes again on its own.
func (s *TelnyxUnpaidSuite) TestAThreadTaskWhoseMoneyCannotBeCheckedIsRetried() {
	person := "+13125550003"
	task := s.threadTask(s.creditedThread(person))
	s.sharing.stop()
	s.Require().Eventually(func() bool {
		// Each search the router admits holds its estimate, which cannot be shared.
		s.serverClient.call(http.MethodPost, "/v1/search", SearchRequest{Query: "again"})
		err := s.gate.Allow(context.Background(), s.customerID(), dlc.SMS, person)
		return errors.Is(err, dlc.ErrUnpaid) && policy.Transient(err)
	}, 10*time.Second, 20*time.Millisecond, "refused once a hold waited unshared")

	run, attempts := s.settledByTheWorker(task)

	s.Equal(store.ScheduledTaskRunFailed, run.State)
	s.Require().NotNil(run.Error)
	s.Equal(codeSpendCheckUnavailable, *run.Error)
	s.Equal(int32(1+len(scheduling.DeliveryRetries)), attempts, "tried again after each wait")
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

// spentAfterOpening is a user who texts the agent's SMS line, and a session they opened on the
// agent while the organization had credit; the credit is spent afterwards, so what costs money
// is refused.
func (s *TelnyxUnpaidSuite) spentAfterOpening() (*testClient, Session) {
	number, writer := "+1555"+digits(s.utils.uuid()), "+1347"+digits(s.utils.uuid())
	s.seedSMSLine(number)
	s.config.Channels = store.AgentChannels{SMS: &store.ChannelLine{Number: number}}
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), &s.config))
	user := s.data.createUser()
	s.Require().NoError(s.store.SaveChannelIdentity(context.Background(), &store.ChannelIdentity{
		CustomerID: s.customerID(), ConfigID: s.config.ID, Kind: string(channels.SMS), Address: writer, UserID: user.userID,
	}))
	organization := s.RouterSuite.app.organization.ID
	s.usage.Credit(s.T(), organization, 10_000_000)
	s.Require().Eventually(func() bool {
		return s.gate.Allow(context.Background(), s.customerID(), dlc.SMS, writer) == nil
	}, 10*time.Second, 50*time.Millisecond, "the credit is read")
	opened := user.createSession(CreateSessionRequest{ConfigId: &s.config.ID})
	s.usage.Credit(s.T(), organization, -10_000_000)
	s.refusedForMoney(writer)
	return user, opened
}

// userTask is a task of user's on channels, following up in the session sessionID, or opening
// one on its first occurrence when sessionID is empty.
func (s *TelnyxUnpaidSuite) userTask(user *testClient, sessionID string, channels ...string) store.ScheduledTask {
	next := time.Now().Add(2 * time.Hour).UTC()
	task := store.ScheduledTask{CustomerID: s.customerID(), ConfigID: s.config.ID, SessionID: sessionID, UserID: user.userID,
		CallerKind: "authenticated", APIKey: s.RouterSuite.app.key, Title: "Stretch", Instructions: "Remind them to stretch.",
		StartLocal: next, Timezone: "Europe/Amsterdam", NextRunAt: &next, Channels: channels}
	s.Require().NoError(s.store.CreateScheduledTask(context.Background(), &task, 0))
	return task
}

// A task delivered by SMS for an organization whose credit ran out is not tried again: the run
// says the credit is spent, and the router warns, naming the task.
func (s *TelnyxUnpaidSuite) TestAnSMSTaskOfAnOrganizationWithNoCreditIsNotRetried() {
	// The task follows up in a session opened while the organization had credit, so what its
	// spent credit refuses is the text.
	user, opened := s.spentAfterOpening()
	task := s.userTask(user, opened.Id, "sms")

	run, attempts := s.settledByTheWorker(task)

	s.Equal(store.ScheduledTaskRunFailed, run.State)
	s.Require().NotNil(run.Error)
	s.Equal(codeInsufficientCredits, *run.Error)
	s.Equal(int32(1), attempts, "spent credit outlasts every retry of the run")
	s.Contains(s.logged.String(), `level=WARN msg="a scheduled task failed to deliver" task=`+task.ID)
}

// A task whose organization is suspended while its turn runs passes Reachable, and the text
// Notify then sends is refused with dlc.ErrUnpaid: the run is not tried again.
func (s *TelnyxUnpaidSuite) TestAnSMSTaskWhoseOrganizationStopsPayingDuringItsTurnIsNotRetried() {
	number, writer := "+1555"+digits(s.utils.uuid()), "+1347"+digits(s.utils.uuid())
	s.seedSMSLine(number)
	s.config.Channels = store.AgentChannels{SMS: &store.ChannelLine{Number: number}}
	s.config.LLM = "holding/holding-model"
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), &s.config))
	user := s.data.createUser()
	s.Require().NoError(s.store.SaveChannelIdentity(context.Background(), &store.ChannelIdentity{
		CustomerID: s.customerID(), ConfigID: s.config.ID, Kind: string(channels.SMS), Address: writer, UserID: user.userID,
	}))
	organization := s.RouterSuite.app.organization.ID
	s.usage.Credit(s.T(), organization, 10_000_000)
	s.Require().Eventually(func() bool {
		return s.gate.Allow(context.Background(), s.customerID(), dlc.SMS, writer) == nil
	}, 10*time.Second, 50*time.Millisecond, "the credit is read")
	opened := user.createSession(CreateSessionRequest{ConfigId: &s.config.ID})
	next := time.Now().Add(2 * time.Hour).UTC()
	task := store.ScheduledTask{CustomerID: s.customerID(), ConfigID: s.config.ID, SessionID: opened.Id, UserID: user.userID,
		CallerKind: "authenticated", APIKey: s.RouterSuite.app.key, Title: "Stretch", Instructions: "Remind them to stretch.",
		StartLocal: next, Timezone: "Europe/Amsterdam", NextRunAt: &next, Channels: []string{"sms"}}
	s.Require().NoError(s.store.CreateScheduledTask(context.Background(), &task, 0))

	model := s.holding
	model.mu.Lock()
	before := model.turns
	model.mu.Unlock()
	model.holds()
	defer model.answers()
	type result struct {
		run      store.ScheduledTaskRun
		attempts int32
	}
	done := make(chan result, 1)
	go func() {
		run, attempts := s.settledByTheWorker(task)
		done <- result{run, attempts}
	}()
	s.Require().Eventually(func() bool {
		model.mu.Lock()
		defer model.mu.Unlock()
		return model.turns > before
	}, deliveryWait, 10*time.Millisecond, "the task's turn never began")
	s.usage.Suspend(s.T(), organization, "fraud review")
	s.refusedForMoney(writer)
	model.answers()

	got := <-done
	s.Equal(store.ScheduledTaskRunFailed, got.run.State)
	s.Require().NotNil(got.run.Error)
	s.Equal(codeAccountSuspended, *got.run.Error)
	s.Equal(int32(1), got.attempts, "a suspension outlasts every retry of the run")
}

// A task whose first occurrence cannot open its session, the credit being spent, is not tried
// again: the run says the credit is spent.
func (s *TelnyxUnpaidSuite) TestATaskWhoseSessionCannotOpenForMoneyIsNotRetried() {
	user, _ := s.spentAfterOpening()
	task := s.userTask(user, "", "conversation")

	run, attempts := s.settledByTheWorker(task)

	s.Equal(store.ScheduledTaskRunFailed, run.State)
	s.Require().NotNil(run.Error)
	s.Equal(codeInsufficientCredits, *run.Error)
	s.Equal(int32(1), attempts, "spent credit outlasts every retry of the run")
}

// A task whose session ended and cannot be carried on, the credit being spent, is not tried
// again: the run says the credit is spent.
func (s *TelnyxUnpaidSuite) TestATaskWhoseEndedSessionCannotReopenForMoneyIsNotRetried() {
	user, opened := s.spentAfterOpening()
	status, body := user.call(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/stop", nil)
	s.Require().Equal(http.StatusNoContent, status, string(body))
	task := s.userTask(user, opened.Id, "conversation")

	run, attempts := s.settledByTheWorker(task)

	s.Equal(store.ScheduledTaskRunFailed, run.State)
	s.Require().NotNil(run.Error)
	s.Equal(codeInsufficientCredits, *run.Error)
	s.Equal(int32(1), attempts, "spent credit outlasts every retry of the run")
}

// A message to an ended session that cannot be carried on, the credit being spent, is refused for
// money as any request is: the scheduled delivery's own answer to the refusal is its alone.
func (s *TelnyxUnpaidSuite) TestAMessageToAnEndedSessionOfAnOrganizationWithNoCreditIsRefusedForMoney() {
	user, opened := s.spentAfterOpening()
	status, body := user.call(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/stop", nil)
	s.Require().Equal(http.StatusNoContent, status, string(body))
	request := s.utils.uuid()

	status, body = s.serverClient.actingFor(user).call(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/respond",
		RespondRequest{Text: "still there?", RequestId: &request})

	s.Equal(http.StatusPaymentRequired, status, string(body))
	s.Contains(string(body), `"code":"insufficient_credits"`)
}

// A phone task whose call the gate refuses, the credit being spent, rings nobody and is not tried
// again: the run says the credit is spent.
func (s *TelnyxUnpaidSuite) TestAPhoneTaskOfAnOrganizationWithNoCreditIsNotRetried() {
	user, opened := s.spentAfterOpening()
	task := s.userTask(user, opened.Id, "phone")

	run, attempts := s.settledByTheWorker(task)

	s.Equal(store.ScheduledTaskRunFailed, run.State)
	s.Require().NotNil(run.Error)
	s.Equal(codeInsufficientCredits, *run.Error)
	s.Equal(int32(1), attempts, "spent credit outlasts every retry of the run")
}

// A task whose session cannot open because the organization's money cannot be checked is tried
// again: the check passes again on its own.
func (s *TelnyxUnpaidSuite) TestATaskWhoseSessionCannotOpenWhileMoneyCannotBeCheckedIsRetried() {
	person := "+1347" + digits(s.utils.uuid())
	s.usage.Credit(s.T(), s.RouterSuite.app.organization.ID, 10_000_000)
	s.Require().Eventually(func() bool {
		return s.gate.Allow(context.Background(), s.customerID(), dlc.SMS, person) == nil
	}, 10*time.Second, 50*time.Millisecond, "the credit is read")
	s.sharing.stop()
	s.Require().Eventually(func() bool {
		s.serverClient.call(http.MethodPost, "/v1/search", SearchRequest{Query: "again"})
		err := s.gate.Allow(context.Background(), s.customerID(), dlc.SMS, person)
		return errors.Is(err, dlc.ErrUnpaid) && policy.Transient(err)
	}, 10*time.Second, 20*time.Millisecond, "refused once a hold waited unshared")
	task := s.userTask(s.data.createUser(), "", "conversation")

	run, attempts := s.settledByTheWorker(task)

	s.Equal(store.ScheduledTaskRunFailed, run.State)
	s.Require().NotNil(run.Error)
	s.Equal(codeSpendCheckUnavailable, *run.Error)
	s.Equal(int32(1+len(scheduling.DeliveryRetries)), attempts, "tried again after each wait")
}

// reservationRefused is work the router refuses at its reservation, before any vendor is asked, as
// Notify's send is refused when the money changes after the gate let it through, recorded through a
// writer the policies gate, as cmd/router's telephony is. The work is a 10DLC brand fee, whose price
// needs none of the attributes a reservation leaves out (routing.Meter.Estimate).
func (s *TelnyxUnpaidSuite) reservationRefused() error {
	logger := slog.New(slog.DiscardHandler)
	writer := routing.NewWriter(routing.WriterOptions{Store: s.store, Logger: logger, Gate: s.enforcer})
	defer writer.Close(context.Background())
	list, err := pricing.NewList("vendor", pricing.AllAtCost(prices.Vendor().Entries), logger)
	s.Require().NoError(err)
	meter := routing.Meter{Recorder: routing.NewRecorderOn(routing.Phone, writer, list, logger),
		Owner:  routing.Owner{CustomerID: s.customerID(), Attribution: routing.Attribution{APIKey: s.RouterSuite.app.key}},
		Config: routing.ProviderConfig{Provider: "telnyx", Model: "dlc_brand"}, Estimate: pricing.Quantities{pricing.Requests: 1}}
	_, err = routing.Call(context.Background(), meter, func(context.Context) (channels.Sent, error) {
		s.Fail("a refused reservation asks no vendor")
		return channels.Sent{}, nil
	})
	s.Require().True(routing.ReservationRefused(err), "refused at the reservation: %v", err)
	s.Require().NotErrorIs(err, dlc.ErrUnpaid, "the gate did not refuse it")
	return err
}

// A text of a suspended organization refused at its reservation is recorded under the
// suspension, and not tried again.
func (s *TelnyxUnpaidSuite) TestATextRefusedAtItsReservationIsFinal() {
	s.suspendedThread("+1347" + digits(s.utils.uuid()))

	var failure *scheduling.Failure
	s.Require().ErrorAs(unpaid(s.reservationRefused()), &failure)
	s.Equal(codeAccountSuspended, failure.Reason)
	s.True(failure.Final, "a suspension outlasts every retry of the run")
}

// A text refused at its reservation because the organization's money cannot be checked is tried
// again.
func (s *TelnyxUnpaidSuite) TestATextRefusedAtItsReservationWhileMoneyCannotBeCheckedIsRetried() {
	person := "+1347" + digits(s.utils.uuid())
	s.creditedThread(person)
	s.sharing.stop()
	s.Require().Eventually(func() bool {
		s.serverClient.call(http.MethodPost, "/v1/search", SearchRequest{Query: "again"})
		return policy.Transient(s.gate.Allow(context.Background(), s.customerID(), dlc.SMS, person))
	}, 10*time.Second, 20*time.Millisecond, "refused once a hold waited unshared")

	var failure *scheduling.Failure
	s.Require().ErrorAs(unpaid(s.reservationRefused()), &failure)
	s.Equal(codeSpendCheckUnavailable, failure.Reason)
	s.False(failure.Final, "the check passes again on its own")
}

// refusedTurn is the follow-up of a task due in the session opened before the credit ran out,
// captured as a text channel captures it once the gate let the send through: the money changed
// after it did, so the turn is what the organization's money refuses.
func (s *TelnyxUnpaidSuite) refusedTurn(user *testClient, opened Session) error {
	task := s.userTask(user, opened.Id, "sms")
	found, running := s.router.sessions.Get(opened.Id, session.Owner{CustomerID: s.customerID(), UserID: user.userID, Kind: auth.KindAuthenticated})
	s.Require().True(running)
	delivery := scheduledDelivery{CustomerID: s.customerID(), ConfigID: s.config.ID, UserID: user.userID,
		CallerKind: "authenticated", TaskID: task.ID, RunID: "run", CommandID: "st_run", Text: "stretch"}
	_, _, err := s.router.captureReply(scheduledOwnerContext(context.Background(), delivery), found, delivery)
	return err
}

// A turn refused for money inside a session that is open is not tried again: the run records the
// money, not an agent that said nothing.
func (s *TelnyxUnpaidSuite) TestATurnRefusedForMoneyInAnOpenSessionIsFinal() {
	var failure *scheduling.Failure
	user, opened := s.spentAfterOpening()
	s.Require().ErrorAs(s.refusedTurn(user, opened), &failure)
	s.Equal(codeInsufficientCredits, failure.Reason)
	s.True(failure.Final, "spent credit outlasts every retry of the run")
}

// A turn refused because the organization's money cannot be checked is tried again.
func (s *TelnyxUnpaidSuite) TestATurnRefusedWhileMoneyCannotBeCheckedIsRetried() {
	number, writer := "+1555"+digits(s.utils.uuid()), "+1347"+digits(s.utils.uuid())
	s.seedSMSLine(number)
	s.config.Channels = store.AgentChannels{SMS: &store.ChannelLine{Number: number}}
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), &s.config))
	user := s.data.createUser()
	s.usage.Credit(s.T(), s.RouterSuite.app.organization.ID, 10_000_000)
	s.Require().Eventually(func() bool {
		return s.gate.Allow(context.Background(), s.customerID(), dlc.SMS, writer) == nil
	}, 10*time.Second, 50*time.Millisecond, "the credit is read")
	opened := user.createSession(CreateSessionRequest{ConfigId: &s.config.ID})
	s.sharing.stop()
	s.Require().Eventually(func() bool {
		s.serverClient.call(http.MethodPost, "/v1/search", SearchRequest{Query: "again"})
		return policy.Transient(s.gate.Allow(context.Background(), s.customerID(), dlc.SMS, writer))
	}, 10*time.Second, 20*time.Millisecond, "refused once a hold waited unshared")

	var failure *scheduling.Failure
	s.Require().ErrorAs(s.refusedTurn(user, opened), &failure)
	s.Equal(codeSpendCheckUnavailable, failure.Reason)
	s.False(failure.Final, "the check passes again on its own")
}

// teamSpent tags the config with a team and spends the team's weekly limit. The text gate admits
// a delivery with the attribution it carries, which has no team, so it lets the task through and
// only the session the config opens, or the turn it runs, is refused (spend_limit_reached).
func (s *TelnyxUnpaidSuite) teamSpent(person string) {
	if s.config.Tags == nil {
		s.config.Tags = map[string]string{}
	}
	s.config.Tags[session.TeamTag] = "finance"
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), &s.config))
	status, payload := s.serverClient.call(http.MethodPost, "/v1/spend/limits",
		map[string]any{"subject": "team", "team": "finance", "interval": "weekly", "limit_micros": 1_000_000})
	s.Require().Equal(http.StatusCreated, status, string(payload))
	s.usage.Spend(s.T(), s.RouterSuite.app.organization.ID, 1_000_000, func(event *usage.Event) {
		event.AppID = s.customerID()
		event.UsageCategories = []string{"team:finance"}
	})
	teamed := routing.WithAttribution(context.Background(), routing.Attribution{APIKey: s.RouterSuite.app.key, UsageCategories: []string{"team:finance"}})
	s.Require().Eventually(func() bool {
		s.serverClient.call(http.MethodPost, "/v1/search", SearchRequest{Query: "again"})
		return s.gate.Allow(teamed, s.customerID(), dlc.SMS, person) != nil
	}, 10*time.Second, 50*time.Millisecond, "the team's limit is read")
	s.Require().NoError(s.gate.Allow(context.Background(), s.customerID(), dlc.SMS, person), "no team, no refusal")
}

// A task made in a thread whose conversation has no session running opens one on the task's
// config, which a spent team limit refuses though the text gate let the task through: the run
// records the limit and is not tried again.
func (s *TelnyxUnpaidSuite) TestAThreadTaskWhoseSessionCannotOpenForATeamLimitIsNotRetried() {
	person := "+1347" + digits(s.utils.uuid())
	channel := s.creditedThread(person)
	cid := chatlog.ChannelType + ":" + channel
	if found, running := s.router.sessions.ByConversationWhere(cid, func(string, int64) bool { return true }); running {
		_, err := s.router.sessions.Close(found.ID(), session.OwnerOf(found.Spec()))
		s.Require().NoError(err)
	}
	s.teamSpent(person)
	task := s.threadTask(channel)

	run, attempts := s.settledByTheWorker(task)

	s.Equal(store.ScheduledTaskRunFailed, run.State)
	s.Require().NotNil(run.Error)
	s.Equal(codeSpendLimitReached, *run.Error)
	s.Equal(int32(1), attempts, "a spent limit outlasts every retry of the run")
}

// A turn on which the session reports a failure that is not about money, such as a tool's error,
// still carries the reply the agent then gives: only a refusal for money ends the capture.
func (s *TelnyxUnpaidSuite) TestATurnThatReportsAFailureNotAboutMoneyStillCarriesItsReply() {
	s.config.LLM = "holding/holding-model"
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), &s.config))
	user := s.data.createUser()
	s.usage.Credit(s.T(), s.RouterSuite.app.organization.ID, 10_000_000)
	s.Require().Eventually(func() bool {
		return s.gate.Allow(context.Background(), s.customerID(), dlc.SMS, "+13125550009") == nil
	}, 10*time.Second, 50*time.Millisecond, "the credit is read")
	opened := user.createSession(CreateSessionRequest{ConfigId: &s.config.ID})
	found, running := s.router.sessions.Get(opened.Id, session.Owner{CustomerID: s.customerID(), UserID: user.userID, Kind: auth.KindAuthenticated})
	s.Require().True(running)
	delivery := scheduledDelivery{CustomerID: s.customerID(), ConfigID: s.config.ID, UserID: user.userID,
		CallerKind: "authenticated", TaskID: "task", RunID: "run", CommandID: "st_run", Text: "stretch"}

	model := s.holding
	model.mu.Lock()
	before := model.turns
	model.mu.Unlock()
	model.holds()
	defer model.answers()
	type result struct {
		text string
		err  error
	}
	done := make(chan result, 1)
	go func() {
		text, _, err := s.router.captureReply(scheduledOwnerContext(context.Background(), delivery), found, delivery)
		done <- result{text, err}
	}()
	s.Require().Eventually(func() bool {
		model.mu.Lock()
		defer model.mu.Unlock()
		return model.turns > before
	}, deliveryWait, 10*time.Millisecond, "the turn never began")
	found.Report(errors.New("the weather tool timed out"), "tool")
	model.answers()

	got := <-done
	s.Require().NoError(got.err)
	s.Equal("Held.", got.text)
}
