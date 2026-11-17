//go:build integration

package dlc_test

import (
	"errors"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

func (s *GateSuite) TestARefusalOverTheConcurrentLimitSaysSo() {
	s.limit(map[string]usage.ChannelLimits{dlc.Voice: {Concurrent: new(1)}})
	s.leg(store.Outbound, time.Now(), true)

	err := s.gate.Allow(s.ctx, s.app(), dlc.Voice, "+15125550123")

	s.Equal("concurrent", dlc.RefusalCode(err))
	s.True(dlc.RefusalResets(dlc.RefusalCode(err)))
}

func (s *GateSuite) TestARefusalOfARecipientOutsideTheSandboxDoesNotReset() {
	gate := s.newGate(dlc.Sandbox{Enabled: true, Recipients: 2}, config.LimitsEnforce)
	s.Require().NoError(s.money.Store.SetSandboxRecipients(s.ctx, s.app(), []string{"+15125550001"}))

	err := gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550003")

	s.Equal("sandbox_recipient", dlc.RefusalCode(err))
	s.False(dlc.RefusalResets(dlc.RefusalCode(err)))
}

func (s *GateSuite) TestARefusalOfACountryDoesNotReset() {
	s.limit(map[string]usage.ChannelLimits{dlc.SMS: {Countries: []string{"US"}}})

	err := s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+442079460000")

	s.Equal("country", dlc.RefusalCode(err))
	s.False(dlc.RefusalResets(dlc.RefusalCode(err)))
}

func (s *GateSuite) TestARefusalOfAnOptedOutRecipientDoesNotReset() {
	s.Require().NoError(s.money.Store.OptOut(s.ctx, &store.OptOut{
		CustomerID: s.app(), Recipient: "+15125550002", Channel: dlc.SMS, Source: "test",
	}))

	err := s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550002")

	s.Equal("opted_out", dlc.RefusalCode(err))
	s.False(dlc.RefusalResets(dlc.RefusalCode(err)))
}

func (s *GateSuite) TestARefusalOverTheDailyLimitResets() {
	s.limit(map[string]usage.ChannelLimits{dlc.SMS: {PerDay: new(0)}})

	err := s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550123")

	s.Equal("per_day", dlc.RefusalCode(err))
	s.True(dlc.RefusalResets("per_day"))
}

func (s *GateSuite) TestARefusalStillIsErrRefused() {
	s.limit(map[string]usage.ChannelLimits{dlc.SMS: {Countries: []string{"US"}}})

	err := s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+442079460000")

	s.ErrorIs(err, dlc.ErrRefused)
	s.ErrorContains(err, "the default tier is not allowed to reach GB by sms")
}

func (s *GateSuite) TestAnErrorThatIsNoRefusalHasNoCode() {
	s.Empty(dlc.RefusalCode(errors.New("boom")))
	s.Empty(dlc.RefusalCode(dlc.ErrRefused))
	s.Empty(dlc.RefusalCode(nil))
}

func (s *GateSuite) TestARefusalOverACountryIsATierLimit() {
	s.limit(map[string]usage.ChannelLimits{dlc.SMS: {Countries: []string{"US"}}})

	s.Equal("country", dlc.TierLimit(s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+442079460000")))
}

func (s *GateSuite) TestARefusalOverTheDailyLimitIsATierLimit() {
	s.limit(map[string]usage.ChannelLimits{dlc.SMS: {PerDay: new(0)}})

	s.Equal("per_day", dlc.TierLimit(s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550123")))
}

func (s *GateSuite) TestARefusalOverTheRecipientsADayIsATierLimit() {
	s.limit(map[string]usage.ChannelLimits{dlc.SMS: {RecipientsPerDay: new(0)}})

	s.Equal("recipients_per_day", dlc.TierLimit(s.gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550123")))
}

func (s *GateSuite) TestARefusalOverTheCallsAtOnceIsATierLimit() {
	s.limit(map[string]usage.ChannelLimits{dlc.Voice: {Concurrent: new(1)}})
	s.leg(store.Outbound, time.Now(), true)

	s.Equal("concurrent", dlc.TierLimit(s.gate.Allow(s.ctx, s.app(), dlc.Voice, "+15125550123")))
}

func (s *GateSuite) TestARefusalOfTheSandboxOrAnOptOutIsNoTierLimit() {
	gate := s.newGate(dlc.Sandbox{Enabled: true, Recipients: 2}, config.LimitsEnforce)
	s.Require().NoError(s.money.Store.SetSandboxRecipients(s.ctx, s.app(), []string{"+15125550001"}))
	s.Require().NoError(s.money.Store.OptOut(s.ctx, &store.OptOut{
		CustomerID: s.app(), Recipient: "+15125550001", Channel: dlc.SMS, Source: "test",
	}))

	outside := gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550003")
	optedOut := gate.Allow(s.ctx, s.app(), dlc.SMS, "+15125550001")

	s.Require().ErrorIs(outside, dlc.ErrRefused)
	s.Require().ErrorIs(optedOut, dlc.ErrRefused)
	s.Empty(dlc.TierLimit(outside))
	s.Empty(dlc.TierLimit(optedOut))
	s.Empty(dlc.TierLimit(errors.New("boom")))
}
