//go:build integration

package policy

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

func pointer[T any](value T) *T { return &value }

// streamDestination gives the app a destination of the router's own events, past the create
// check, and returns its id. Nothing is sent to it here: what is queued for it is read back.
func (s *PolicySuite) streamDestination(appID string) string {
	destination := store.EventDestination{
		ID: uuid.NewString(), CustomerID: appID, ConnectorID: store.StreamSource, URL: "https://example.com/" + appID,
		Forward: store.ForwardAll, SecretSealed: []byte("sealed"), KEKVersion: 1,
	}
	s.Require().NoError(s.store.CreateEventDestination(s.ctx, &destination))
	return destination.ID
}

// notices are the warnings written for the organization, oldest first.
func (s *PolicySuite) notices(organization string) []store.BalanceNotice {
	var written []store.BalanceNotice
	s.Require().NoError(s.store.DB().NewSelect().Model(&written).Where("organization_id = ?", organization).
		Order("created_at", "granted_micros").Scan(s.ctx))
	return written
}

// queued is the deliveries queued for the destination, by webhook-id, with their bodies.
func (s *PolicySuite) queued(destinationID string) map[string]balanceEvent {
	var deliveries []store.EventDelivery
	s.Require().NoError(s.store.DB().NewSelect().Model(&deliveries).Where("destination_id = ?", destinationID).Scan(s.ctx))
	bodies := make(map[string]balanceEvent, len(deliveries))
	for _, delivery := range deliveries {
		var event balanceEvent
		s.Require().NoError(json.Unmarshal(delivery.Body, &event))
		bodies[delivery.ID] = event
	}
	return bodies
}

// warnable is an organization of its own, warned at 5 dollars, with credit micros, read by
// the enforcers.
func (s *PolicySuite) warnable(micros int64, enforcers ...*reading) {
	s.usage().SetLimits(s.T(), s.org, usage.Limits{LowBalanceMicros: pointer(int64(5_000_000))})
	s.usage().Credit(s.T(), s.org, micros)
	for _, enforcer := range enforcers {
		enforcer.Join(s.app, s.org, usage.DefaultTier)
		s.read(enforcer.Enforcer, s.org)
	}
}

func (s *PolicySuite) TestCrossingLowSendsOneEventAcrossTwoRouters() {
	first, second := s.billed(), s.billed()
	other := fmt.Sprintf("app-other-%d", time.Now().UnixNano())
	s.Require().NoError(s.store.JoinOrganization(s.ctx, other, s.org))
	mine, theirs := s.streamDestination(s.app), s.streamDestination(other)
	s.warnable(10_000_000, first, second)
	s.Empty(s.notices(s.org), "above the threshold, no warning")

	s.usage().Spend(s.T(), s.org, 6_000_000)

	s.eventually(func() bool { return len(s.notices(s.org)) == 1 }, 5*time.Second, 10*time.Millisecond, "one warning is written")
	// Both routers have read the low balance a few times over.
	time.Sleep(10 * quick.Fresh)
	written := s.notices(s.org)
	s.Require().Len(written, 1, "one warning however many routers saw it")
	s.Equal(store.NoticeBalanceLow, written[0].Kind)
	s.EqualValues(10_000_000, written[0].GrantedMicros)
	s.EqualValues(4_000_000, written[0].BalanceMicros)
	s.EqualValues(5_000_000, written[0].ThresholdMicros)
	id := "balance:" + s.org + ":balance_low:10000000"
	for app, destination := range map[string]string{s.app: mine, other: theirs} {
		queued := s.queued(destination)
		s.Require().Len(queued, 1, "one forward for each stream destination of every app of the organization")
		event := queued[id]
		s.Equal(EventBalanceLow, event.Type)
		s.Equal(s.org, event.Data.OrganizationID)
		s.Equal(app, event.Data.AppID, "each app is told it is its own")
		s.EqualValues(4_000_000, event.Data.BalanceMicros)
		s.EqualValues(5_000_000, *event.Data.ThresholdMicros)
		s.Equal(usage.DefaultTier, event.Data.Tier)
	}
}

func (s *PolicySuite) TestAGrantRearmsTheWarning() {
	enforcer := s.billed()
	destination := s.streamDestination(s.app)
	s.warnable(10_000_000, enforcer)
	s.usage().Spend(s.T(), s.org, 6_000_000)
	s.eventually(func() bool { return len(s.notices(s.org)) == 1 }, 5*time.Second, 10*time.Millisecond, "the first warning")

	s.usage().Credit(s.T(), s.org, 3_000_000)
	s.read(enforcer.Enforcer, s.org)
	s.usage().Spend(s.T(), s.org, 3_000_000)

	s.eventually(func() bool { return len(s.notices(s.org)) == 2 }, 5*time.Second, 10*time.Millisecond, "the grant re-armed it")
	written := s.notices(s.org)
	s.EqualValues(13_000_000, written[1].GrantedMicros)
	queued := s.queued(destination)
	s.Len(queued, 2)
	s.Contains(queued, "balance:"+s.org+":balance_low:13000000")
}

func (s *PolicySuite) TestRunningOutSendsExhaustedEvenInShadow() {
	enforcer := s.billed(func(settings *Settings) { settings.Mode = config.LimitsShadow })
	destination := s.streamDestination(s.app)
	s.warnable(10_000_000, enforcer)

	s.usage().Spend(s.T(), s.org, 10_000_000)

	s.eventually(func() bool { return len(s.notices(s.org)) == 1 }, 5*time.Second, 10*time.Millisecond, "a warning is information, not a refusal")
	s.Equal(store.NoticeBalanceExhausted, s.notices(s.org)[0].Kind, "straight past low to nothing")
	s.Equal(EventBalanceExhausted, s.queued(destination)["balance:"+s.org+":balance_exhausted:10000000"].Type)
}

func (s *PolicySuite) TestTheCustomersThresholdWarns() {
	enforcer := s.billed()
	s.save(store.ScopeOrganization, s.org, store.PolicyDocument{BalanceWarningMicros: pointer(int64(20_000_000))})
	s.warnable(15_000_000, enforcer)

	s.eventually(func() bool { return len(s.notices(s.org)) == 1 }, 5*time.Second, 10*time.Millisecond, "under the customer's own threshold")
	s.EqualValues(20_000_000, s.notices(s.org)[0].ThresholdMicros)
}

func (s *PolicySuite) TestABigReservationThatSettlesSmallSendsNoWarning() {
	enforcer := s.billed()
	s.streamDestination(s.app)
	s.warnable(10_000_000, enforcer)

	reserved := newID()
	s.Require().NoError(enforcer.Reserve(s.owner(), reserved, 12_000_000, time.Minute))
	s.Equal(BalanceExhausted, s.balance(enforcer.Enforcer).State, "the answers count the reservation")
	s.never(func() bool { return len(s.notices(s.org)) > 0 }, 10*quick.Fresh, 10*time.Millisecond,
		"a reservation is not spend")
	enforcer.Settle(s.owner(), reserved, 10_000)
	s.never(func() bool { return len(s.notices(s.org)) > 0 }, 10*quick.Fresh, 10*time.Millisecond,
		"it settled far above the threshold")

	enforcer.Settle(s.owner(), newID(), 6_000_000)

	s.eventually(func() bool { return len(s.notices(s.org)) == 1 }, 5*time.Second, 10*time.Millisecond,
		"spending past the threshold still warns")
	written := s.notices(s.org)[0]
	s.Equal(store.NoticeBalanceLow, written.Kind)
	// The small unit is let go once nothing acks it, so it may or may not count.
	s.GreaterOrEqual(written.BalanceMicros, int64(3_990_000), "what was spent, the reservation aside")
	s.LessOrEqual(written.BalanceMicros, int64(4_000_000))
}

func (s *PolicySuite) TestABlockedNoticeWriteDoesNotHoldUpTheBalances() {
	enforcer := s.billed()
	s.streamDestination(s.app)
	s.warnable(10_000_000, enforcer)
	// A transaction holding the table makes every warning's write wait, as a Postgres that
	// does not answer would.
	blocking, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer func() { _ = blocking.Rollback() }()
	_, err = blocking.ExecContext(s.ctx, "LOCK TABLE balance_notices IN ACCESS EXCLUSIVE MODE")
	s.Require().NoError(err)

	s.usage().Spend(s.T(), s.org, 6_000_000)
	s.read(enforcer.Enforcer, s.org)
	s.Equal(BalanceLow, s.balance(enforcer.Enforcer).State)
	started := time.Now()
	for range 5 {
		s.read(enforcer.Enforcer, s.org)
	}
	s.Less(time.Since(started), noticesTimeout, "the balances are read on while the warning waits")

	s.Require().NoError(blocking.Rollback())
	s.eventually(func() bool { return len(s.notices(s.org)) == 1 }, 2*noticesTimeout, 10*time.Millisecond,
		"written once Postgres answers")
}

// balance is the suite's organization's balance as the enforcer answers it.
func (s *PolicySuite) balance(enforcer *Enforcer) Balance {
	balance, ok := enforcer.Balance(s.app, s.org)
	s.Require().True(ok, "the account is read")
	return balance
}

func (s *PolicySuite) TestAnOrganizationNeverGrantedCreditIsNotWarned() {
	enforcer := s.billed()
	s.streamDestination(s.app)
	enforcer.Join(s.app, s.org, usage.DefaultTier)
	s.read(enforcer.Enforcer, s.org)

	s.never(func() bool { return len(s.notices(s.org)) > 0 }, 10*quick.Fresh, 10*time.Millisecond, "nothing ran out")
}

func (s *PolicySuite) TestOffSendsNothing() {
	enforcer := s.billed(func(settings *Settings) { settings.Mode = config.LimitsOff })
	s.streamDestination(s.app)
	s.usage().Credit(s.T(), s.org, 1_000_000)
	enforcer.Join(s.app, s.org, usage.DefaultTier)

	s.never(func() bool { return len(s.notices(s.org)) > 0 }, 10*quick.Fresh, 10*time.Millisecond, "off reads no money")
}
