//go:build integration

package policy

import (
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// modes are every limits mode, each of which books.
var modes = []config.LimitsMode{config.LimitsOff, config.LimitsShadow, config.LimitsEnforce}

func mode(m config.LimitsMode) func(*Settings) { return func(s *Settings) { s.Mode = m } }

func (s *PolicySuite) book(enforcer *Enforcer, micros int64) (routing.Booked, error) {
	return enforcer.Book(s.ctx, ownerOf(s.app), micros)
}

func (s *PolicySuite) TestABookingNeedsTheWholeAmountInEveryLimitsMode() {
	for _, m := range modes {
		s.Run(string(m), func() {
			s.SetupTest()
			enforcer := s.billed(mode(m))
			s.usage().Credit(s.T(), s.org, 1_000_000)

			_, err := s.book(enforcer.Enforcer, 1_000_001)
			s.Require().ErrorIs(err, ErrNoCredits, "a micro short")
			s.False(Transient(err))
			s.Contains(err.Error(), "$1.00 of credit to spend")

			booked, err := s.book(enforcer.Enforcer, 1_000_000)
			s.Require().NoError(err, "exactly the credit")
			_, err = s.book(enforcer.Enforcer, 1)
			s.Require().ErrorIs(err, ErrNoCredits, "the first booking holds all of it")

			booked.Release()
			second, err := s.book(enforcer.Enforcer, 1)
			s.Require().NoError(err, "a released booking frees its credit at once")
			second.Release()
		})
	}
}

func (s *PolicySuite) TestAPurchaseWithNoKnownPriceStillNeedsSomeCredit() {
	enforcer := s.billed(mode(config.LimitsOff))
	_, err := s.book(enforcer.Enforcer, 0)
	s.Require().ErrorIs(err, ErrNoCredits)

	s.usage().Credit(s.T(), s.org, 1)
	booked, err := s.book(enforcer.Enforcer, 0)
	s.Require().NoError(err)
	booked.Release()
}

func (s *PolicySuite) TestABookingCountsTheOrganizationsHolds() {
	enforcer := s.billed()
	enforcer.Join(s.app, s.org, openTier)
	s.usage().Credit(s.T(), s.org, 1_000_000)
	s.read(enforcer.Enforcer, s.org)
	s.hold(enforcer, 1)
	s.Require().EqualValues(3, balanceHeld(enforcer.Enforcer, s.org))

	_, err := s.book(enforcer.Enforcer, 1_000_000)
	s.Require().ErrorIs(err, ErrNoCredits, "work recorded and not yet counted holds part of the credit")
	booked, err := s.book(enforcer.Enforcer, 999_997)
	s.Require().NoError(err)
	booked.Release()
}

func (s *PolicySuite) TestASuspendedOrganizationBooksNothing() {
	enforcer := s.billed(mode(config.LimitsOff))
	s.usage().Credit(s.T(), s.org, 1_000_000)
	s.usage().Suspend(s.T(), s.org, "test")
	defer s.usage().Restore(s.T(), s.org)

	_, err := s.book(enforcer.Enforcer, 1)
	s.Require().ErrorIs(err, ErrSuspended)
}

func (s *PolicySuite) TestAnOrganizationWithOnlyTrialCreditBooksNothingInEveryLimitsMode() {
	for _, m := range modes {
		s.Run(string(m), func() {
			s.SetupTest()
			enforcer := s.billed(mode(m))
			s.usage().Trial(s.T(), s.org, 5_000_000)

			_, err := s.book(enforcer.Enforcer, 1)
			s.Require().ErrorIs(err, ErrTopUpRequired)
			s.Equal("top_up_required", Code(err))
			s.False(Transient(err))
		})
	}
}

func (s *PolicySuite) TestAnOrganizationThatToppedUpBooksOnItsTrialCreditToo() {
	enforcer := s.billed(mode(config.LimitsOff))
	s.usage().Trial(s.T(), s.org, 5_000_000)
	s.usage().Credit(s.T(), s.org, 1_000_000)

	booked, err := s.book(enforcer.Enforcer, 6_000_000)
	s.Require().NoError(err, "once funded, all of its credit pays for a purchase")
	booked.Release()
}

func (s *PolicySuite) TestABookingWhoseCreditCannotBeReadIsRefusedUntilItCan() {
	enforcer := s.billed(mode(config.LimitsOff))
	s.usage().Credit(s.T(), s.org, 1_000_000)
	enforcer.front.down.Store(true)

	_, err := s.book(enforcer.Enforcer, 1)
	s.Require().ErrorIs(err, ErrSpendCheckUnavailable)
	s.True(Transient(err))

	enforcer.front.down.Store(false)
	booked, err := s.book(enforcer.Enforcer, 1)
	s.Require().NoError(err)
	booked.Release()
}

func (s *PolicySuite) TestAnAppInNoOrganizationBooksNothing() {
	enforcer := s.billed(mode(config.LimitsOff))
	booked, err := enforcer.Book(s.ctx, ownerOf("app-on-its-own-"+s.app), 1_000_000)
	s.Require().NoError(err)
	s.Equal(routing.NoBooking, booked)
}

func (s *PolicySuite) TestTwoRoutersBookingAtOnceNeverBothSpendTheSameCredit() {
	_, client := s.redis()
	shared := func(settings *Settings) { settings.Holds = client }
	routers := []*reading{s.billed(mode(config.LimitsOff), shared), s.billed(mode(config.LimitsOff), shared)}
	s.usage().Credit(s.T(), s.org, 1_000_000)

	for round := range 20 {
		var wg sync.WaitGroup
		answers := make([]routing.Booked, len(routers))
		errs := make([]error, len(routers))
		for i, router := range routers {
			wg.Go(func() { answers[i], errs[i] = s.book(router.Enforcer, 600_000) })
		}
		wg.Wait()
		booked := 0
		for i := range routers {
			if errs[i] == nil {
				booked++
				answers[i].Release()
				continue
			}
			s.Require().ErrorIs(errs[i], ErrNoCredits, "round %d", round)
		}
		s.Equal(1, booked, "round %d: one of two bookings of 60%% of the credit", round)
	}
}

func (s *PolicySuite) TestABookingLeftByARouterThatStoppedExpiresInAnHour() {
	_, client := s.redis()
	direct := s.direct()
	enforcer := s.billed(mode(config.LimitsOff), func(settings *Settings) { settings.Holds = client })
	s.usage().Credit(s.T(), s.org, 1_000_000)
	key := bookingPrefix + ":{" + s.org + "}"
	s.T().Cleanup(func() { direct.Do(s.ctx, direct.B().Del().Key(key).Build()) })

	_, err := s.book(enforcer.Enforcer, 1_000_000)
	s.Require().NoError(err, "booked, then never charged nor released")
	ttl, err := direct.Do(s.ctx, direct.B().Pttl().Key(key).Build()).AsInt64()
	s.Require().NoError(err)
	s.InDelta(BookingTTL.Milliseconds(), ttl, float64(time.Minute.Milliseconds()), "the key lives as long as the booking")
	_, err = s.book(enforcer.Enforcer, 1)
	s.Require().ErrorIs(err, ErrNoCredits)

	// The hour passes for the booking alone.
	fields, err := direct.Do(s.ctx, direct.B().Hgetall().Key(key).Build()).AsStrMap()
	s.Require().NoError(err)
	s.Require().Len(fields, 1)
	past := strconv.FormatInt(time.Now().Add(-time.Second).UnixMilli(), 10)
	for id, value := range fields {
		parts := strings.Split(value, ":")
		s.Require().Len(parts, 3)
		s.Equal("1000000", parts[0])
		s.Equal("0", parts[2], "not charged")
		s.Require().NoError(direct.Do(s.ctx, direct.B().Hset().Key(key).FieldValue().
			FieldValue(id, parts[0]+":"+past+":0").Build()).Error())
	}

	expired := testutil.ToFloat64(bookingsExpired)
	booked, err := s.book(enforcer.Enforcer, 1_000_000)
	s.Require().NoError(err, "the expired booking holds nothing")
	booked.Release()
	s.Equal(expired+1, testutil.ToFloat64(bookingsExpired))
	s.Contains(enforcer.logs.String(), "bookings expired neither charged nor released")
}

func (s *PolicySuite) TestAChargedBookingCountsUntilTheChargeIsCounted() {
	_, client := s.redis()
	direct := s.direct()
	enforcer := s.billed(mode(config.LimitsOff), func(settings *Settings) { settings.Holds = client })
	s.usage().Credit(s.T(), s.org, 1_000_000)
	key := bookingPrefix + ":{" + s.org + "}"
	s.T().Cleanup(func() { direct.Do(s.ctx, direct.B().Del().Key(key).Build()) })

	booked, err := s.book(enforcer.Enforcer, 1_000_000)
	s.Require().NoError(err)
	booked.Charged()
	booked.Release()

	fields, err := direct.Do(s.ctx, direct.B().Hgetall().Key(key).Build()).AsStrMap()
	s.Require().NoError(err)
	s.Require().Len(fields, 1, "a release after the charge changes nothing")
	for _, value := range fields {
		parts := strings.Split(value, ":")
		s.Require().Len(parts, 3)
		s.Equal("1", parts[2], "charged")
		expires, err := strconv.ParseInt(parts[1], 10, 64)
		s.Require().NoError(err)
		s.LessOrEqual(expires, time.Now().Add(ChargedTTL).UnixMilli()+1_000, "kept for ChargedTTL, not the hour")
	}
	_, err = s.book(enforcer.Enforcer, 1)
	s.Require().ErrorIs(err, ErrNoCredits, "the charge is not in the standing yet")
}
