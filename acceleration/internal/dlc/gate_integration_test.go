//go:build integration

package dlc_test

import (
	"bytes"
	"context"
	"log/slog"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy/policytest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// GateSuite is the gate keeping an organization to the channel limits its tier and overrides
// set, as a router does: its admission from a real enforcer reading a real usage service, its
// counters in Redis and the calls up in Postgres.
type GateSuite struct {
	suite.Suite
	ctx   context.Context
	money *policytest.Money
	redis rueidis.Client
	logs  *lockedBuffer
	// owner is the test's app, in an organization of its own with credit.
	owner routing.Owner
	gate  *dlc.Gate
}

func TestGateSuite(t *testing.T) {
	suite.Run(t, new(GateSuite))
}

func (s *GateSuite) SetupSuite() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR not set")
	}
	s.ctx = context.Background()
	s.money = policytest.Start(s.T(), "dlc_gate")
	var err error
	s.redis, err = rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(s.redis.Close)
}

func (s *GateSuite) SetupTest() {
	s.owner = s.money.Owner(s.T(), 1_000_000_000)
	s.logs = &lockedBuffer{}
	s.gate = s.newGate(dlc.Sandbox{}, config.LimitsEnforce)
	organization := s.owner.OrganizationID
	s.T().Cleanup(func() { s.forget(organization) })
}

// newGate is a gate over the suite's money and Redis, holding apps to sandbox, keeping channel
// limits in mode.
func (s *GateSuite) newGate(sandbox dlc.Sandbox, mode config.LimitsMode) *dlc.Gate {
	logger := slog.New(slog.NewTextHandler(s.logs, nil))
	return dlc.NewGate(s.money.Store, s.redis, sandbox, logger).Charging(s.money.Enforcer).Limiting(mode)
}

func (s *GateSuite) app() string { return s.owner.CustomerID }

// limit gives the test's organization channels as its own limits, and waits for the enforcer
// to have read them.
func (s *GateSuite) limit(channels map[string]usage.ChannelLimits) {
	s.money.Usage.SetLimits(s.T(), s.owner.OrganizationID, usage.Limits{Channels: channels})
	s.waitForLimits(func(limits usage.Limits) bool { return len(limits.Channels) == len(channels) })
}

func (s *GateSuite) waitForLimits(read func(usage.Limits) bool) {
	s.Require().Eventually(func() bool {
		admission, err := s.money.Enforcer.Admit(s.ctx, s.owner)
		s.Require().NoError(err)
		return admission.Limits != nil && read(*admission.Limits)
	}, 5*time.Second, 10*time.Millisecond, "the organization's limits are read")
}

// forget deletes the organization's counters of today.
func (s *GateSuite) forget(organization string) {
	counts := "channel:" + time.Now().UTC().Format("20060102") + ":{" + organization + "}"
	keys := []string{counts}
	for _, channel := range []string{dlc.SMS, dlc.Voice, dlc.WhatsApp, dlc.IMessage, "rcs"} {
		keys = append(keys, counts+":"+channel+":to")
	}
	s.Require().NoError(s.redis.Do(s.ctx, s.redis.B().Del().Key(keys...).Build()).Error())
}

// sent is how many the organization sent on channel today, as the counters say.
func (s *GateSuite) sent(channel string) int64 {
	counts := "channel:" + time.Now().UTC().Format("20060102") + ":{" + s.owner.OrganizationID + "}"
	count, err := s.redis.Do(s.ctx, s.redis.B().Hget().Key(counts).Field(channel).Build()).AsInt64()
	if rueidis.IsRedisNil(err) {
		return 0
	}
	s.Require().NoError(err)
	return count
}

func (s *GateSuite) TestATextToACountryTheTierDoesNotAllowIsRefused() {
	s.money.Usage.SetTier(s.T(), usage.Tier{Name: usage.DefaultTier, FailMode: usage.FailClosed, Limits: usage.Limits{
		Channels: map[string]usage.ChannelLimits{dlc.SMS: {Countries: []string{"US", "CA"}}},
	}})
	s.T().Cleanup(func() { s.money.Usage.SetTier(s.T(), usage.BuiltInDefaultTier()) })
	s.waitForLimits(func(limits usage.Limits) bool { return len(limits.Channels) == 1 })

	err := s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+442079460000")

	s.ErrorIs(err, dlc.ErrRefused)
	s.ErrorContains(err, "the default tier is not allowed to reach GB by sms")
	s.NotContains(err.Error(), "+44", "the refusal does not name the number")
	s.NoError(s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550123"))
	s.NoError(s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+14165550123"), "Toronto")
	s.NoError(s.gate.Allow(s.ctx, s.app(), dlc.WhatsApp, "442079460000"), "a channel the tier does not limit")
}

func (s *GateSuite) TestAnOverrideOfTheCountriesLetsTheOrganizationReachThem() {
	s.money.Usage.SetTier(s.T(), usage.Tier{Name: usage.DefaultTier, FailMode: usage.FailClosed, Limits: usage.Limits{
		Channels: map[string]usage.ChannelLimits{dlc.SMS: {Countries: []string{"US", "CA"}}},
	}})
	s.T().Cleanup(func() { s.money.Usage.SetTier(s.T(), usage.BuiltInDefaultTier()) })

	s.limit(map[string]usage.ChannelLimits{dlc.SMS: {Countries: []string{"US", "CA", "GB"}}})

	s.NoError(s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+442079460000"))
}

func (s *GateSuite) TestAnUnparseableNumberIsRefusedWhenCountriesAreLimited() {
	s.limit(map[string]usage.ChannelLimits{dlc.IMessage: {Countries: []string{"US"}}})

	for _, to := range []string{"someone@example.com", "+88200000000", "not a number"} {
		err := s.gate.Allow(s.ctx, s.app(), dlc.IMessage, to)
		s.ErrorIs(err, dlc.ErrRefused, to)
		s.ErrorContains(err, "in no country", to)
	}
	s.NoError(s.gate.Allow(s.ctx, s.app(), dlc.IMessage, "+15125550123"))
}

func (s *GateSuite) TestTheDailyMessageLimitRefusesTheNextText() {
	s.limit(map[string]usage.ChannelLimits{dlc.SMS: {PerDay: new(2)}})

	for range 2 {
		s.Require().NoError(s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550123"))
		s.gate.Sent(s.ctx, s.app(), dlc.SMS, "+15125550123")
	}

	err := s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550123")
	s.ErrorIs(err, dlc.ErrRefused)
	s.ErrorContains(err, "allows 2 messages of sms a day")
	s.NoError(s.gate.Allow(s.ctx, s.app(), dlc.WhatsApp, "15125550123"), "another channel counts apart")
}

func (s *GateSuite) TestTheRecipientLimitRefusesANewNumberAndAllowsAKnownOne() {
	s.limit(map[string]usage.ChannelLimits{dlc.SMS: {RecipientsPerDay: new(2)}})
	for _, to := range []string{"+15125550001", "+15125550002"} {
		s.Require().NoError(s.gate.Allow(s.ctx, s.app(), dlc.SMS, to))
		s.gate.Sent(s.ctx, s.app(), dlc.SMS, to)
	}

	err := s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550003")

	s.ErrorIs(err, dlc.ErrRefused)
	s.ErrorContains(err, "allows sms to 2 recipients a day")
	s.NoError(s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550001"), "somebody already reached today")
	s.gate.Sent(s.ctx, s.app(), dlc.SMS, "+15125550001")
	s.NoError(s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550002"))
}

func (s *GateSuite) TestTheConcurrentCallLimitCountsOpenLegs() {
	s.limit(map[string]usage.ChannelLimits{dlc.Voice: {Concurrent: new(1)}})
	s.NoError(s.gate.Allow(s.ctx, s.app(), dlc.Voice, "+15125550123"))

	s.leg(false)
	s.NoError(s.gate.Allow(s.ctx, s.app(), dlc.Voice, "+15125550123"), "a leg still ringing is no call up")

	up := s.leg(true)
	err := s.gate.Allow(s.ctx, s.app(), dlc.Voice, "+15125550123")
	s.ErrorIs(err, dlc.ErrRefused)
	s.ErrorContains(err, "allows 1 calls at once")
	s.NoError(s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550123"), "a text is no call")

	ended, err := s.money.Store.EndCallLegs(s.ctx, store.AppScope{Unpinned: true}, "default", up, time.Now())
	s.Require().NoError(err)
	s.Require().Len(ended, 1)
	s.NoError(s.gate.Allow(s.ctx, s.app(), dlc.Voice, "+15125550123"), "once it ended")
}

// leg opens an outbound leg of the organization's, its phone joined when up, and returns its
// call.
func (s *GateSuite) leg(up bool) string {
	call := "call-" + uuid.NewString()
	leg := store.CallLeg{
		CustomerID: s.app(), OrganizationID: s.owner.OrganizationID, CallType: "default", CallID: call,
		Vendor: "telnyx", Direction: store.Outbound, LocalNumber: "+15125550100", RemoteNumber: "+15125550123",
	}
	if up {
		now := time.Now()
		leg.StartedAt = &now
	}
	opened, err := s.money.Store.OpenCallLeg(s.ctx, &leg)
	s.Require().NoError(err)
	s.Require().True(opened)
	return call
}

func (s *GateSuite) TestCallMinutesCountRoundedUpAgainstTheDay() {
	s.limit(map[string]usage.ChannelLimits{dlc.Voice: {PerDay: new(3)}})

	s.gate.Talked(s.ctx, s.app(), s.owner.OrganizationID, 61*time.Second)
	s.Equal(int64(2), s.sent(dlc.Voice), "two started minutes")
	s.NoError(s.gate.Allow(s.ctx, s.app(), dlc.Voice, "+15125550123"))

	s.gate.Talked(s.ctx, s.app(), "", time.Second)
	s.Equal(int64(3), s.sent(dlc.Voice), "the organization found by the app")
	err := s.gate.Allow(s.ctx, s.app(), dlc.Voice, "+15125550123")
	s.ErrorIs(err, dlc.ErrRefused)
	s.ErrorContains(err, "allows 3 minutes of voice a day")
}

func (s *GateSuite) TestACallsRecipientIsCountedWhenDialed() {
	s.limit(map[string]usage.ChannelLimits{dlc.Voice: {RecipientsPerDay: new(1)}})

	s.gate.Dialed(s.ctx, s.app(), s.owner.OrganizationID, "+15125550001")

	s.Zero(s.sent(dlc.Voice), "dialing is no minute")
	s.NoError(s.gate.Allow(s.ctx, s.app(), dlc.Voice, "+15125550001"))
	s.ErrorIs(s.gate.Allow(s.ctx, s.app(), dlc.Voice, "+15125550002"), dlc.ErrRefused)
}

func (s *GateSuite) TestAnIMessageTextIsCheckedAgainstItsOwnKey() {
	s.limit(map[string]usage.ChannelLimits{dlc.IMessage: {PerDay: new(1)}, "rcs": {PerDay: new(100)}})
	s.Require().NoError(s.gate.Allow(s.ctx, s.app(), dlc.IMessage, "+15125550123"))
	s.gate.Sent(s.ctx, s.app(), dlc.IMessage, "+15125550123")

	s.ErrorIs(s.gate.Allow(s.ctx, s.app(), dlc.IMessage, "+15125550123"), dlc.ErrRefused)
	s.Equal(int64(1), s.sent(dlc.IMessage))
	s.Zero(s.sent("rcs"), "the rcs count is unchanged")
	s.NoError(s.gate.Allow(s.ctx, s.app(), "rcs", "+15125550123"))
}

func (s *GateSuite) TestAnOrganizationWithNoCreditMayNotText() {
	s.owner = s.money.Owner(s.T(), 0)

	err := s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550123")

	s.ErrorIs(err, dlc.ErrUnpaid)
	s.ErrorIs(err, policy.ErrNoCredits)
}

func (s *GateSuite) TestOffModeKeepsOptOutAndTheSandbox() {
	s.limit(map[string]usage.ChannelLimits{dlc.SMS: {Countries: []string{"GB"}, PerDay: new(0)}})
	gate := s.newGate(dlc.Sandbox{Enabled: true, Recipients: 2}, config.LimitsOff)
	s.Require().NoError(s.money.Store.SetSandboxRecipients(s.ctx, s.app(), []string{"+15125550001", "+15125550002"}))
	s.Require().NoError(s.money.Store.OptOut(s.ctx, &store.OptOut{
		CustomerID: s.app(), Recipient: "+15125550002", Channel: dlc.SMS, Source: "test",
	}))

	s.NoError(gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550001"), "off checks no channel limit")
	s.ErrorIs(gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550002"), dlc.ErrRefused, "an opt-out still refuses")
	err := gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550003")
	s.ErrorIs(err, dlc.ErrRefused)
	s.ErrorContains(err, "sandbox", "the sandbox still holds the app")

	gate.Sent(s.ctx, s.app(), dlc.SMS, "+15125550001")
	s.Zero(s.sent(dlc.SMS), "and off counts nothing")
}

func (s *GateSuite) TestShadowLogsWhatItWouldRefuseAndAllows() {
	s.limit(map[string]usage.ChannelLimits{dlc.SMS: {Countries: []string{"US"}}})
	gate := s.newGate(dlc.Sandbox{}, config.LimitsShadow)

	s.NoError(gate.Allow(s.ctx, s.app(), dlc.SMS, "+442079460000"))
	s.NoError(gate.Allow(s.ctx, s.app(), dlc.SMS, "+442079460001"))

	logged := s.logs.String()
	s.Equal(1, strings.Count(logged, "would refuse"), "once a minute per organization and reason: %s", logged)
	s.Contains(logged, "organization="+s.owner.OrganizationID)
	s.Contains(logged, "reason=country")
	s.NotContains(logged, "+44", "the logs do not name the number")
}

func (s *GateSuite) TestAGateWithNoLimitsKeepsTodaysBehaviour() {
	s.limit(map[string]usage.ChannelLimits{dlc.SMS: {Countries: []string{"US"}, PerDay: new(0)}})
	gate := dlc.NewGate(s.money.Store, s.redis, dlc.Sandbox{}, nil).Charging(s.money.Enforcer)

	s.NoError(gate.Allow(s.ctx, s.app(), dlc.SMS, "+442079460000"))
	gate.Sent(s.ctx, s.app(), dlc.SMS, "+442079460000")
	s.Zero(s.sent(dlc.SMS))
}

func (s *GateSuite) TestCountersCountAnAppWithAnApprovedUseCase() {
	gate := s.newGate(dlc.Sandbox{Enabled: true, Recipients: 2, MessagesPerDay: 30}, config.LimitsEnforce)
	s.Require().NoError(s.money.Store.CreateUseCase(s.ctx, &store.UseCase{
		CustomerID: s.app(), Name: "approved", Status: dlc.Approved, UseCaseType: "CUSTOMER_CARE",
	}))
	sandboxed, err := gate.Sandboxed(s.ctx, s.app())
	s.Require().NoError(err)
	s.Require().False(sandboxed)

	gate.Sent(s.ctx, s.app(), dlc.SMS, "+15125550123")
	gate.Sent(s.ctx, s.app(), dlc.SMS, "+15125550124")

	s.Equal(int64(2), s.sent(dlc.SMS))
	held, err := gate.Usage(s.ctx, s.app())
	s.Require().NoError(err)
	s.Zero(held.Messages, "the sandbox's counter counts only a sandboxed app")
}

// lockedBuffer is a log the gate writes while a test reads it.
type lockedBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (l *lockedBuffer) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.b.Write(p)
}

func (l *lockedBuffer) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.b.String()
}