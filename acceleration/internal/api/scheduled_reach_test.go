//go:build integration

package api

import (
	"context"
	"os"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/channels"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// reachOf is why a text on channel would not reach the user from the config now, if anything.
func (s *ScheduledTasksSuite) reachOf(config string, user *testClient, channel string) error {
	return s.router.reachableText(context.Background(), scheduledDelivery{
		CustomerID: s.customerID(), ConfigID: config, UserID: user.userID,
	}, channel)
}

// textsFrom records the user as writing to the config on kind from a number of their own.
func (s *ScheduledTasksSuite) textsFrom(user *testClient, config, kind string) {
	s.Require().NoError(s.store.SaveChannelIdentity(context.Background(), &store.ChannelIdentity{
		CustomerID: s.customerID(), ConfigID: config, Kind: kind, Address: s.writerNumber(), UserID: user.userID,
	}))
}

// gateOnClosedStore has the gate every text passes read from a store closed for the test, so
// its opt-out check fails as an unreachable database does, which is no refusal.
func (s *ScheduledTasksSuite) gateOnClosedStore() {
	closed, err := store.Open(testDatabase(s.T(), os.Getenv("ROUTER_POSTGRES_DSN")))
	s.Require().NoError(err)
	s.Require().NoError(closed.Close())
	original := *s.gate
	*s.gate = *dlc.NewGate(closed, s.live.Redis(), s.sandbox, nil)
	s.T().Cleanup(func() { *s.gate = original })
}

// agentOnSMS is a scheduling agent answering on the text line number.
func (s *ScheduledTasksSuite) agentOnSMS(number string) string {
	return s.agentWithScheduling(func(request *AgentConfigRequest) {
		request.Channels = &AgentChannels{Sms: &ChannelLineRequest{Number: number}}
	})
}

func (s *ScheduledTasksSuite) TestATextToSomebodyWhoNeverTextedIsUnreachable() {
	number := s.lineNumber()
	s.seedSMSLine(number)
	config := s.agentOnSMS(number)

	s.ErrorIs(s.reachOf(config, s.data.createUser(), "sms"), channels.ErrUnreachable)
}

func (s *ScheduledTasksSuite) TestATextFromAnAgentWithNoLineThereIsUnreachable() {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	s.textsFrom(user, config, "sms")

	s.ErrorIs(s.reachOf(config, user, "sms"), channels.ErrUnreachable)
}

func (s *ScheduledTasksSuite) TestATextOnALineDisconnectedSinceIsUnreachable() {
	number := s.lineNumber()
	s.seedSMSLine(number)
	config := s.agentOnSMS(number)
	user := s.data.createUser()
	s.textsFrom(user, config, "sms")
	s.Require().NoError(s.store.DeleteChannelAccount(context.Background(), s.customerID(), "sms", number))

	s.ErrorIs(s.reachOf(config, user, "sms"), channels.ErrUnreachable)
}

func (s *ScheduledTasksSuite) TestAWhatsAppMessageMoreThanADayAfterThePersonWroteIsUnreachable() {
	number, writer := s.lineNumber(), s.writerNumber()
	line := s.connectWhatsApp(number)
	config := s.agentOnWhatsApp(number)
	user := s.data.createUser()
	s.seedWriter(user, config, number, writer)
	_, err := s.store.DB().NewRaw("UPDATE channel_deliveries SET received_at = ? WHERE account_id = ?",
		time.Now().Add(-25*time.Hour), line.Id).Exec(context.Background())
	s.Require().NoError(err)

	s.ErrorIs(s.reachOf(config, user, "whatsapp"), channels.ErrUnreachable)
}

func (s *ScheduledTasksSuite) TestAnIMessageWithNoChatToWriteToIsUnreachable() {
	number := s.lineNumber()
	sealed, err := s.router.sealChannelSecrets(s.customerID(), channels.Secrets{Token: "linq-key", Signing: "linq-secret"})
	s.Require().NoError(err)
	s.Require().NoError(s.store.SaveChannelAccount(context.Background(), &store.ChannelAccount{
		CustomerID: s.customerID(), Kind: "imessage", E164: number, Token: "line-" + s.utils.uuid(),
		SecretsSealed: sealed, SecretsKEKVersion: s.sealer.CurrentVersion(),
	}))
	config := s.agentWithScheduling(func(request *AgentConfigRequest) {
		request.Channels = &AgentChannels{Imessage: &ChannelLineRequest{Number: number}}
	})
	user := s.data.createUser()
	// Known by number, before any chat with them was recorded.
	s.textsFrom(user, config, "imessage")

	s.ErrorIs(s.reachOf(config, user, "imessage"), channels.ErrUnreachable)
}

func (s *ScheduledTasksSuite) TestATextOnALineWhoseCredentialsDoNotOpenIsAFailureNotUnreachable() {
	number := s.lineNumber()
	s.Require().NoError(s.store.SaveChannelAccount(context.Background(), &store.ChannelAccount{
		CustomerID: s.customerID(), Kind: "sms", E164: number, Token: "line-" + s.utils.uuid(),
	}))
	config := s.agentOnSMS(number)
	user := s.data.createUser()
	s.textsFrom(user, config, "sms")
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, []string{"sms"})

	_, err := s.deliverNow(s.tasksOf(user)[0], time.Now(), "")
	failure := s.failure(err)
	s.Equal(scheduling.ReasonFailed, failure.Reason)
	s.False(failure.Final, "a line put right by the next attempt delivers")
}

// Only the gate's refusal makes a text unreachable (internal/channels/service.go reach). A gate
// that cannot read is the router failing, so the run is tried again rather than given up on.
func (s *ScheduledTasksSuite) TestATextWhoseGateCannotReadIsAFailureNotUnreachable() {
	number := s.lineNumber()
	s.seedSMSLine(number)
	config := s.agentOnSMS(number)
	user := s.data.createUser()
	s.textsFrom(user, config, "sms")
	opened := s.openScheduling(user, config, nil)
	s.taskOn(user, config, opened.Id, []string{"sms"})
	s.Require().NoError(s.reachOf(config, user, "sms"), "with a gate that reads, the text reaches them")
	s.gateOnClosedStore()

	_, err := s.deliverNow(s.tasksOf(user)[0], time.Now(), "")

	s.ErrorContains(err, "database is closed")
	failure := s.failure(err)
	s.Equal(scheduling.ReasonFailed, failure.Reason)
	s.False(failure.Final, "a gate that reads again delivers on the next attempt")
}
