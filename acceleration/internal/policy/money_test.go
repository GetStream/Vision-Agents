//go:build integration

package policy

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

// The tiers the money tests read: one whose row admits billable work it cannot bill, one whose
// row refuses it, and one with no row, which the default tier answers for.
const (
	openTier    = "tier-a"
	closedTier  = "tier-b"
	rowlessTier = "tier-unknown"
)

// startUsage runs the money tests' usage service for the suite: the real one, on a database of
// the suite's own, with its management API. Every test has organizations of its own.
func (s *PolicySuite) startUsage() {
	s.service = usagetest.Start(s.T(), usagetest.Options{
		DSN: usagetest.Database(s.T(), "usage_policy"), ManageKey: "manage-" + uuid.NewString(),
		Logger: slog.New(slog.DiscardHandler),
	})
	s.service.SetTier(s.T(), usage.Tier{Name: openTier, FailMode: usage.FailOpen})
	s.service.SetTier(s.T(), usage.Tier{Name: closedTier, FailMode: usage.FailClosed})
}

func (s *PolicySuite) usage() *usagetest.Service { return s.service }

// front is the usage service as an enforcer reaches it: the real one, through a proxy the test
// can take down, which counts the standings calls made through it.
type front struct {
	server    *httptest.Server
	down      atomic.Bool
	standings atomic.Int64
}

func (s *PolicySuite) front(service *usagetest.Service) *front {
	target, err := url.Parse(service.URL)
	s.Require().NoError(err)
	proxy := httputil.NewSingleHostReverseProxy(target)
	f := &front{}
	f.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if f.down.Load() {
			// The connection drops, as it does when the service is gone.
			panic(http.ErrAbortHandler)
		}
		if r.URL.Path == usage.PathStandings {
			f.standings.Add(1)
		}
		proxy.ServeHTTP(w, r)
	}))
	s.T().Cleanup(f.server.Close)
	return f
}

// logs is a log buffer safe to read while the enforcer writes to it.
type logs struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (l *logs) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.Write(p)
}

func (l *logs) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.String()
}

// reading is an enforcer reading money from the suite's usage service through a front of its
// own, and what it logged.
type reading struct {
	*Enforcer
	front *front
	logs  *logs
}

func (s *PolicySuite) billed(change ...func(*Settings)) *reading {
	service := s.usage()
	f := s.front(service)
	client, err := usage.NewClient(usage.ClientOptions{URL: f.server.URL, Token: service.Token})
	s.Require().NoError(err)
	configs, err := appconfig.New(appconfig.Options{Store: s.store})
	s.Require().NoError(err)
	written := &logs{}
	logger := slog.New(slog.NewTextHandler(written, &slog.HandlerOptions{Level: slog.LevelInfo}))
	enforcer := s.newEnforcer(configs, logger, append([]func(*Settings){func(settings *Settings) { settings.Usage = client }}, change...)...)
	return &reading{Enforcer: enforcer, front: f, logs: written}
}

// admit is Admit, which must refuse nothing outright.
func (s *PolicySuite) admit(enforcer *Enforcer, appID string) routing.Admission {
	admitted, err := enforcer.Admit(s.ctx, appID)
	s.Require().NoError(err)
	return admitted
}

// read waits for the organization's account to have landed a read since now.
func (s *PolicySuite) read(enforcer *Enforcer, organization string) {
	since := time.Now()
	s.eventually(func() bool {
		_, _ = enforcer.Admit(s.ctx, s.app)
		enforcer.money.mu.Lock()
		defer enforcer.money.mu.Unlock()
		acct := enforcer.money.accounts[organization]
		return acct != nil && acct.read.After(since)
	}, 5*time.Second, 5*time.Millisecond, "the account is read")
}

// pending is what the organization has held against it now.
func pending(enforcer *Enforcer, organization string) int64 {
	enforcer.money.mu.Lock()
	defer enforcer.money.mu.Unlock()
	acct := enforcer.money.accounts[organization]
	if acct == nil {
		return 0
	}
	return acct.pending(time.Now(), scope{})
}

// available is the organization's balance as an admission counts it: the standing's, less what
// is held.
func available(enforcer *Enforcer, organization string) int64 {
	enforcer.money.mu.Lock()
	defer enforcer.money.mu.Unlock()
	acct := enforcer.money.accounts[organization]
	if acct == nil {
		return 0
	}
	return acct.standing.BalanceMicros - acct.pending(time.Now(), scope{})
}

// owner is who the app's work is recorded under.
func (s *PolicySuite) owner() routing.Owner {
	return routing.Owner{CustomerID: s.app, Attribution: routing.Attribution{OrganizationID: s.org, APIKey: "key-a"}}
}

func (s *PolicySuite) TestAnOrganizationWithNoCreditMayNotSpend() {
	for _, tier := range []string{usage.DefaultTier, openTier} {
		s.Run(tier, func() {
			s.SetupTest()
			enforcer := s.billed()
			enforcer.Join(s.app, s.org, tier)

			spend := s.spendRefused(enforcer.Enforcer, s.app, ErrNoCredits)

			s.ErrorContains(spend, "add credit", "every tier is prepaid, so no tier skips the check")
		})
	}
}

func (s *PolicySuite) TestAGrantLetsItSpend() {
	enforcer := s.billed()
	enforcer.Join(s.app, s.org, openTier)
	s.spendRefused(enforcer.Enforcer, s.app, ErrNoCredits)

	s.usage().Credit(s.T(), s.org, 1_000_000)

	s.eventually(func() bool { return s.admit(enforcer.Enforcer, s.app).Spend == nil }, 5*time.Second, 10*time.Millisecond)
}

func (s *PolicySuite) TestASuspendedOrganizationIsRefusedOutright() {
	enforcer := s.billed()
	s.usage().Credit(s.T(), s.org, 1_000_000)
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)

	s.usage().Suspend(s.T(), s.org, "chargeback on the card")

	var refused error
	s.eventually(func() bool {
		_, refused = enforcer.Admit(s.ctx, s.app)
		return errors.Is(refused, ErrSuspended)
	}, 5*time.Second, 10*time.Millisecond)
	s.NotContains(refused.Error(), "chargeback", "why stays with the operator")
	s.Contains(enforcer.logs.String(), "chargeback", "and is in the log")

	s.usage().Restore(s.T(), s.org)

	s.eventually(func() bool {
		_, err := enforcer.Admit(s.ctx, s.app)
		return err == nil
	}, 5*time.Second, 10*time.Millisecond)
}

func (s *PolicySuite) TestAnAppBudgetIsMeasuredInTheBackground() {
	reads := &requestReads{}
	s.store.DB().AddQueryHook(reads)
	s.save(store.ScopeApp, s.app, store.PolicyDocument{Budget: &store.Budget{LimitMicros: 1_000_000, Interval: store.BudgetDaily}})
	s.spend(s.app, 1_500_000)

	spend := s.spendRefused(s.enforcer, s.app, ErrBudgetSpent)
	s.ErrorContains(spend, "app's daily budget of $1.00")
	s.enforcer.Close()
	before := reads.reads.Load()
	for range 100 {
		s.ErrorIs(s.admit(s.enforcer, s.app).Spend, ErrBudgetSpent)
	}

	s.Equal(before, reads.reads.Load(), "no request row is summed on the request path")
}

// requestReads counts the reads of requests a store makes.
type requestReads struct{ reads atomic.Int64 }

func (*requestReads) BeforeQuery(ctx context.Context, _ *bun.QueryEvent) context.Context { return ctx }

func (r *requestReads) AfterQuery(_ context.Context, event *bun.QueryEvent) {
	if strings.HasPrefix(strings.TrimSpace(event.Query), "SELECT") && strings.Contains(event.Query, "requests") {
		r.reads.Add(1)
	}
}

// spendHardLimit is a spend hard limit of daily and monthly micros, nil for none.
func spendHardLimit(daily, monthly *int64) usage.Limits {
	return usage.Limits{SpendHardLimit: &usage.SpendHardLimit{DailyMicros: daily, MonthlyMicros: monthly}}
}

func (s *PolicySuite) TestTheDailySpendHardLimitRefusesBillableWork() {
	enforcer := s.billed()
	s.usage().Credit(s.T(), s.org, 100_000_000)
	s.usage().SetLimits(s.T(), s.org, spendHardLimit(new(int64(1_000_000)), nil))
	enforcer.Join(s.app, s.org, openTier)

	s.usage().Spend(s.T(), s.org, 1_000_000)

	spend := s.spendRefused(enforcer.Enforcer, s.app, ErrSystemLimit)
	s.NotErrorIs(spend, ErrNoCredits, "$99 of credit is left")
	_, midnight := usage.Window(usage.Daily, time.Now())
	s.Equal(fmt.Sprintf("the organization's daily spend hard limit of $1.00 is reached; it resets at %s",
		midnight.Format(time.RFC3339)), spend.Error())
}

func (s *PolicySuite) TestTheMonthlySpendHardLimitNamesTheMonth() {
	enforcer := s.billed()
	s.usage().Credit(s.T(), s.org, 10_000_000_000)
	s.usage().SetLimits(s.T(), s.org, spendHardLimit(nil, new(int64(2_500_000_000))))
	enforcer.Join(s.app, s.org, openTier)

	s.usage().Spend(s.T(), s.org, 2_500_000_000)

	spend := s.spendRefused(enforcer.Enforcer, s.app, ErrSystemLimit)
	_, nextMonth := usage.Window(usage.Monthly, time.Now())
	s.Equal(fmt.Sprintf("the organization's monthly spend hard limit of $2,500.00 is reached; it resets at %s",
		nextMonth.Format(time.RFC3339)), spend.Error())
	s.Equal(1, nextMonth.Day())
}

func (s *PolicySuite) TestAHoldCountsAgainstTheSpendHardLimit() {
	enforcer := s.billed()
	s.usage().Credit(s.T(), s.org, 100_000_000)
	s.usage().SetLimits(s.T(), s.org, spendHardLimit(new(int64(1_000_000)), nil))
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)
	s.Require().NoError(s.admit(enforcer.Enforcer, s.app).Spend)

	enforcer.Spent(s.owner(), uuid.Must(uuid.NewV7()), 1_000_000)

	s.ErrorIs(s.admit(enforcer.Enforcer, s.app).Spend, ErrSystemLimit, "with no write and no read")
}

func (s *PolicySuite) TestTheSpendHardLimitStartsAgainAtMidnightUTC() {
	var clock atomic.Pointer[time.Time]
	enforcer := s.billed(func(settings *Settings) {
		settings.Now = func() time.Time {
			if moved := clock.Load(); moved != nil {
				return *moved
			}
			return time.Now()
		}
	})
	s.usage().Credit(s.T(), s.org, 100_000_000)
	s.usage().SetLimits(s.T(), s.org, spendHardLimit(new(int64(1_000_000)), nil))
	enforcer.Join(s.app, s.org, openTier)
	s.usage().Spend(s.T(), s.org, 1_000_000)
	s.spendRefused(enforcer.Enforcer, s.app, ErrSystemLimit)

	_, midnight := usage.Window(usage.Daily, time.Now())
	tomorrow := midnight.Add(time.Second)
	clock.Store(&tomorrow)

	s.NoError(s.admit(enforcer.Enforcer, s.app).Spend, "the standing still counts yesterday's window, which counts as nothing today")
}

func (s *PolicySuite) TestAnOverrideRaisesTheSpendHardLimit() {
	enforcer := s.billed()
	s.usage().Credit(s.T(), s.org, 100_000_000)
	s.usage().SetLimits(s.T(), s.org, spendHardLimit(new(int64(1_000_000)), nil))
	enforcer.Join(s.app, s.org, openTier)
	s.usage().Spend(s.T(), s.org, 1_000_000)
	s.spendRefused(enforcer.Enforcer, s.app, ErrSystemLimit)

	s.usage().SetLimits(s.T(), s.org, spendHardLimit(new(int64(2_000_000)), nil))
	s.eventually(func() bool { return s.admit(enforcer.Enforcer, s.app).Spend == nil }, 5*time.Second, 10*time.Millisecond, "raised")

	s.usage().SetLimits(s.T(), s.org, spendHardLimit(new(int64(1_000_000)), nil))
	s.spendRefused(enforcer.Enforcer, s.app, ErrSystemLimit)
	s.usage().SetLimits(s.T(), s.org, usage.Limits{})
	s.eventually(func() bool { return s.admit(enforcer.Enforcer, s.app).Spend == nil }, 5*time.Second, 10*time.Millisecond,
		"back to the tier's, which has none")
}

func (s *PolicySuite) TestShadowModeAdmitsAndLogs() {
	for name, arrange := range map[string]func(organization string){
		"no credit": func(string) {},
		"the spend hard limit": func(organization string) {
			s.usage().Credit(s.T(), organization, 100_000_000)
			s.usage().SetLimits(s.T(), organization, spendHardLimit(new(int64(1_000_000)), nil))
			s.usage().Spend(s.T(), organization, 1_000_000)
		},
		"a suspension": func(organization string) {
			s.usage().Credit(s.T(), organization, 100_000_000)
			s.usage().Suspend(s.T(), organization, "a test")
		},
	} {
		s.Run(name, func() {
			s.SetupTest()
			enforcer := s.billed(func(settings *Settings) { settings.Mode = config.LimitsShadow })
			arrange(s.org)
			enforcer.Join(s.app, s.org, openTier)
			s.read(enforcer.Enforcer, s.org)

			s.eventually(func() bool {
				return strings.Contains(enforcer.logs.String(), "would refuse")
			}, 5*time.Second, 10*time.Millisecond)
			admitted, err := enforcer.Admit(s.ctx, s.app)
			s.NoError(err)
			s.NoError(admitted.Spend)
			s.Contains(enforcer.logs.String(), "organization="+s.org)
		})
	}
}

func (s *PolicySuite) TestOffModeChecksNothing() {
	enforcer := s.billed(func(settings *Settings) { settings.Mode = config.LimitsOff })
	s.usage().Suspend(s.T(), s.org, "a test")
	enforcer.Join(s.app, s.org, closedTier)

	s.never(func() bool {
		admitted, err := enforcer.Admit(s.ctx, s.app)
		return err != nil || admitted.Spend != nil
	}, 300*time.Millisecond, 10*time.Millisecond, "a suspended organization with no credit is admitted")
	s.Zero(enforcer.front.standings.Load(), "nothing is read")
}

func (s *PolicySuite) TestOneCallReadsEveryOrganization() {
	enforcer := s.billed(func(settings *Settings) { settings.Fresh, settings.Idle = time.Hour, time.Hour })
	organizations := make([]string, 150)
	for i := range organizations {
		organizations[i] = fmt.Sprintf("%s-%03d", s.org, i)
		enforcer.Join(fmt.Sprintf("%s-%03d", s.app, i), organizations[i], openTier)
	}
	s.eventually(func() bool {
		for _, organization := range organizations {
			enforcer.money.mu.Lock()
			acct := enforcer.money.accounts[organization]
			read := acct != nil && !acct.read.IsZero()
			enforcer.money.mu.Unlock()
			if !read {
				enforcer.money.wake()
				return false
			}
		}
		return true
	}, 10*time.Second, 20*time.Millisecond)
	// The test makes the next pass itself, so the loop's own cannot be counted with it.
	enforcer.money.stop()
	before := enforcer.front.standings.Load()

	enforcer.money.pass(s.ctx)

	s.EqualValues(2, enforcer.front.standings.Load()-before, "150 organizations are two calls of 100")
}

func (s *PolicySuite) TestAnOldAccountAnswersAtOnce() {
	enforcer := s.billed()
	s.usage().Credit(s.T(), s.org, 1_000_000)
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)
	enforcer.front.down.Store(true)
	s.usage().Credit(s.T(), s.org, -1_000_000)

	started := time.Now()
	for range 1_000 {
		s.Require().NoError(s.admit(enforcer.Enforcer, s.app).Spend, "the old standing")
	}
	s.Less(time.Since(started), 10*time.Millisecond, "nothing waits on the usage service")

	enforcer.front.down.Store(false)
	s.spendRefused(enforcer.Enforcer, s.app, ErrNoCredits)
}

func (s *PolicySuite) TestAMissReadsThePoliciesNowAndTheMoneyLater() {
	no := false
	s.save(store.ScopeApp, s.app, store.PolicyDocument{
		DataPolicy:    options.DataPolicy{AllowTraining: &no},
		AllowedModels: &[]string{"openai/gpt-5.6-luna"},
	})
	enforcer := s.billed()
	enforcer.front.down.Store(true)

	admitted, err := enforcer.Admit(s.ctx, s.app)

	s.Require().NoError(err)
	s.Require().NotNil(admitted.DataPolicy.AllowTraining)
	s.False(*admitted.DataPolicy.AllowTraining)
	s.Equal([]string{"openai/gpt-5.6-luna"}, admitted.Models)
	s.NoError(admitted.Spend, "no money read yet admits")
}

func (s *PolicySuite) TestJoinWarmsTheDecisionAndTheAccount() {
	no := false
	s.save(store.ScopeApp, s.app, store.PolicyDocument{DataPolicy: options.DataPolicy{AllowTraining: &no}})
	enforcer := s.billed()

	enforcer.Join(s.app, s.org, openTier)
	s.eventually(func() bool {
		enforcer.money.mu.Lock()
		defer enforcer.money.mu.Unlock()
		acct := enforcer.money.accounts[s.org]
		_, decided := enforcer.decided(s.app)
		return decided && acct != nil && !acct.read.IsZero()
	}, 5*time.Second, 5*time.Millisecond, "both are read without a request asking")

	locked, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer locked.Rollback() //nolint:errcheck // nothing was written
	_, err = locked.ExecContext(s.ctx, "LOCK TABLE policies IN ACCESS EXCLUSIVE MODE")
	s.Require().NoError(err)
	started := time.Now()
	admitted, err := enforcer.Admit(s.ctx, s.app)
	elapsed := time.Since(started)

	s.Require().NoError(err)
	s.Less(elapsed, 10*time.Millisecond, "answered from memory with the policies locked")
	s.Require().NotNil(admitted.DataPolicy.AllowTraining)
	s.False(*admitted.DataPolicy.AllowTraining, "the app's floor")
	s.ErrorIs(admitted.Spend, ErrNoCredits, "the account's balance, which is nothing")
}

func (s *PolicySuite) TestSpendRecordedHereCountsAtOnce() {
	enforcer := s.billed()
	s.usage().Credit(s.T(), s.org, 1_000_000)
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)
	s.Require().NoError(s.admit(enforcer.Enforcer, s.app).Spend)

	enforcer.Spent(s.owner(), uuid.Must(uuid.NewV7()), 1_000_000)

	s.ErrorIs(s.admit(enforcer.Enforcer, s.app).Spend, ErrNoCredits, "with no write and no read")
}

func (s *PolicySuite) TestAHoldIsKeptUntilAReadAfterItsAck() {
	enforcer := s.billed()
	s.usage().Credit(s.T(), s.org, 3_000_000)
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)
	var over atomic.Int64
	sampling, stop := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(sampling)
		for {
			select {
			case <-stop:
				return
			case <-time.After(5 * time.Millisecond):
				_, _ = enforcer.Admit(s.ctx, s.app)
				over.Store(max(over.Load(), available(enforcer.Enforcer, s.org)))
			}
		}
	}()
	usageID := uuid.Must(uuid.NewV7())

	enforcer.Spent(s.owner(), usageID, 1_000_000)
	event := s.usage().Spend(s.T(), s.org, 1_000_000, func(event *usage.Event) { event.UsageID = usageID })
	s.read(enforcer.Enforcer, s.org)
	s.EqualValues(1_000_000, pending(enforcer.Enforcer, s.org), "the ledger has the row, and nobody said so")
	enforcer.Acked([]routing.Acked{{UsageID: event.UsageID, OrganizationID: s.org, CostMicros: 1_000_000}})
	s.eventually(func() bool { return pending(enforcer.Enforcer, s.org) == 0 }, 5*time.Second, 5*time.Millisecond,
		"let go once a read asked for after the ack has landed")
	close(stop)
	<-sampling

	s.EqualValues(2_000_000, available(enforcer.Enforcer, s.org))
	s.LessOrEqual(over.Load(), int64(2_000_000), "never more than is truly left: counted twice for a while, never zero times")
}

func (s *PolicySuite) TestSpentTwiceUnderOneUsageIDHoldsOnce() {
	enforcer := s.billed()
	usageID := uuid.Must(uuid.NewV7())

	enforcer.Spent(s.owner(), usageID, 1_000_000)
	enforcer.Spent(s.owner(), usageID, 1_000_000)

	s.EqualValues(1_000_000, pending(enforcer.Enforcer, s.org))
}

func (s *PolicySuite) TestAnUnackedHoldExpires() {
	enforcer := s.billed()
	enforcer.Join(s.app, s.org, openTier)

	enforcer.Spent(s.owner(), uuid.Must(uuid.NewV7()), 1_250_000)

	s.eventually(func() bool { return pending(enforcer.Enforcer, s.org) == 0 }, 5*time.Second, 10*time.Millisecond)
	s.eventually(func() bool {
		return strings.Contains(enforcer.logs.String(), "expired before the ledger counted it") &&
			strings.Contains(enforcer.logs.String(), "micros=1250000")
	}, 5*time.Second, 10*time.Millisecond, "and how much is logged")
}

func (s *PolicySuite) TestAnAckForAnotherReplicasRowIsIgnored() {
	enforcer := s.billed()
	usageID := uuid.Must(uuid.NewV7())
	enforcer.Spent(s.owner(), usageID, 1_000_000)

	enforcer.Acked([]routing.Acked{{UsageID: uuid.Must(uuid.NewV7()), OrganizationID: s.org, CostMicros: 7}})
	enforcer.Acked([]routing.Acked{{UsageID: uuid.Must(uuid.NewV7()), OrganizationID: "org-elsewhere", CostMicros: 7}})

	s.EqualValues(1_000_000, pending(enforcer.Enforcer, s.org))
	enforcer.money.mu.Lock()
	defer enforcer.money.mu.Unlock()
	s.True(enforcer.money.accounts[s.org].holds[usageID].ackedAt.IsZero(), "the hold is not the row acked")
}

func (s *PolicySuite) TestPastMaxStaleAClosedTierIsRefusedBillableWork() {
	for tier, refused := range map[string]bool{closedTier: true, rowlessTier: true, openTier: false} {
		s.Run(tier, func() {
			s.SetupTest()
			enforcer := s.billed()
			s.usage().Credit(s.T(), s.org, 1_000_000)
			enforcer.Join(s.app, s.org, tier)
			s.read(enforcer.Enforcer, s.org)
			s.Require().NoError(s.admit(enforcer.Enforcer, s.app).Spend)

			enforcer.front.down.Store(true)

			if !refused {
				s.never(func() bool { return s.admit(enforcer.Enforcer, s.app).Spend != nil }, 500*time.Millisecond, 10*time.Millisecond,
					"its row admits work while the balance cannot be read")
				return
			}
			spend := s.spendRefused(enforcer.Enforcer, s.app, ErrNoCredits)
			s.ErrorContains(spend, "could not be read")
		})
	}
}

func (s *PolicySuite) TestAnAccountNeverReadFollowsItsTiersFailMode() {
	learned := s.billed()
	learned.Join(s.app, s.org, openTier)
	s.read(learned.Enforcer, s.org)
	other := s.app + "-other"
	s.Require().NoError(s.store.JoinOrganization(s.ctx, other, s.org+"-other"))
	learned.front.down.Store(true)

	learned.Join(other, s.org+"-other", openTier)
	s.never(func() bool { return s.admit(learned.Enforcer, other).Spend != nil }, 500*time.Millisecond, 10*time.Millisecond,
		"a tier whose row was read open admits an organization not read yet")

	unknown := s.billed()
	unknown.front.down.Store(true)
	unknown.Join(s.app, s.org, openTier)
	s.NoError(s.admit(unknown.Enforcer, s.app).Spend, "admitted until the first read is due")
	spend := s.spendRefused(unknown.Enforcer, s.app, ErrNoCredits)
	s.ErrorContains(spend, "could not be read", "past MaxStale with no row ever read, the default fails closed")
}

func (s *PolicySuite) TestWithNoUsageServiceBudgetsAreReadFromPostgres() {
	s.save(store.ScopeOrganization, s.org, store.PolicyDocument{
		Budget: &store.Budget{LimitMicros: 1_000_000, Interval: store.BudgetDaily},
	})
	s.never(func() bool { return s.admit(s.enforcer, s.app).Spend != nil }, 200*time.Millisecond, 10*time.Millisecond,
		"no credit is checked: the organization has none and is admitted")

	s.spend(s.app, 1_000_000)

	s.spendRefused(s.enforcer, s.app, ErrBudgetSpent)
}

func (s *PolicySuite) TestWithNoUsageServiceAWrittenRowLetsItsHoldGo() {
	s.save(store.ScopeApp, s.app, store.PolicyDocument{Budget: &store.Budget{LimitMicros: 2_000_000, Interval: store.BudgetDaily}})
	s.admit(s.enforcer, s.app)
	usageID := uuid.Must(uuid.NewV7())
	s.enforcer.Spent(s.owner(), usageID, 1_500_000)
	s.Require().NoError(s.admit(s.enforcer, s.app).Spend)
	s.Require().NoError(s.store.RecordRequest(s.ctx, &store.Request{
		UsageID: usageID, Modality: "llm", CustomerID: s.app, Provider: "openai", Model: "gpt-5.6-luna",
		StartedAt: time.Now().UTC(), CostMicros: 1_500_000, Success: true,
	}))

	s.enforcer.Written([]routing.Acked{{UsageID: usageID, OrganizationID: s.org, CostMicros: 1_500_000}})

	s.eventually(func() bool { return pending(s.enforcer, s.org) == 0 }, 5*time.Second, 10*time.Millisecond)
	s.NoError(s.admit(s.enforcer, s.app).Spend, "counted once, in the shard")
}

func (s *PolicySuite) TestAnAppInNoOrganizationKeepsToItsOwnBudget() {
	alone := s.app + "-alone"
	s.save(store.ScopeApp, alone, store.PolicyDocument{Budget: &store.Budget{LimitMicros: 1_000_000, Interval: store.BudgetDaily}})
	s.spend(alone, 1_000_000)

	s.spendRefused(s.enforcer, alone, ErrBudgetSpent)
	for _, enforcer := range []*Enforcer{s.enforcer, s.billed().Enforcer} {
		s.ErrorIs(s.spendRefused(enforcer, alone, ErrBudgetSpent), ErrBudgetSpent,
			"with or without a usage service, which is never asked about an app that bills nobody")
	}
}

func (s *PolicySuite) TestAnAppInNoOrganizationWithNoBudgetIsNeverRefused() {
	enforcer := s.billed()
	alone := s.app + "-alone"

	s.never(func() bool { return s.admit(enforcer.Enforcer, alone).Spend != nil }, 400*time.Millisecond, 10*time.Millisecond,
		"no organization is no credit to read, and its tier's fail mode never applies")
	s.Zero(enforcer.front.standings.Load())
}

func (s *PolicySuite) TestAnIdleDecisionIsDropped() {
	s.admit(s.enforcer, s.app)
	_, kept := s.enforcer.decided(s.app)
	s.Require().True(kept)

	s.eventually(func() bool {
		_, kept := s.enforcer.decided(s.app)
		return !kept
	}, 5*time.Second, 10*time.Millisecond, "dropped once unused for Idle")
}
