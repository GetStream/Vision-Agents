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
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus/testutil"
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
	configs, err := appconfig.New(appconfig.Options{Store: s.store})
	s.Require().NoError(err)
	return s.billedOn(configs, change...)
}

// billedOn is billed, reading its policies and the shard through configs.
func (s *PolicySuite) billedOn(configs *appconfig.Store, change ...func(*Settings)) *reading {
	service := s.usage()
	f := s.front(service)
	client, err := usage.NewClient(usage.ClientOptions{URL: f.server.URL, Token: service.Token})
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

// pending is what the account still holds in either ledger.
func pending(enforcer *Enforcer, key string) int64 {
	return heldWhere(enforcer, key, func(*hold) bool { return true })
}

// balanceHeld is what the account holds against its balance and spend hard limit.
func balanceHeld(enforcer *Enforcer, key string) int64 {
	return heldWhere(enforcer, key, func(h *hold) bool { return h.balance })
}

// budgetHeld is what the account holds against its budgets.
func budgetHeld(enforcer *Enforcer, key string) int64 {
	return heldWhere(enforcer, key, func(h *hold) bool { return h.budget })
}

func heldWhere(enforcer *Enforcer, key string, counts func(*hold) bool) int64 {
	enforcer.money.mu.Lock()
	defer enforcer.money.mu.Unlock()
	acct := enforcer.money.accounts[key]
	if acct == nil {
		return 0
	}
	var sum int64
	for _, held := range acct.holds {
		if counts(held) {
			sum += held.micros
		}
	}
	return sum
}

// consistent checks that the running totals admission reads are what the holds add up to.
func (s *PolicySuite) consistent(enforcer *Enforcer, key string) {
	enforcer.money.mu.Lock()
	defer enforcer.money.mu.Unlock()
	acct := enforcer.money.accounts[key]
	if acct == nil {
		return
	}
	again := &account{}
	for _, held := range acct.holds {
		again.count(key, held, 1)
	}
	s.Equal(again.held, acct.held, "the balance's running total")
	if len(again.windows)+len(acct.windows) > 0 {
		s.Equal(again.windows, acct.windows, "the windows' running totals")
	}
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
	return acct.standing.BalanceMicros - acct.held
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
	// A store of the test's own, so the hook is on before anything reads through it.
	db, err := store.Open(os.Getenv("ROUTER_POSTGRES_DSN"))
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = db.Close() })
	reads := &requestReads{}
	db.DB().AddQueryHook(reads)
	configs, err := appconfig.New(appconfig.Options{Store: db})
	s.Require().NoError(err)
	enforcer := s.newEnforcer(configs, nil)
	s.Require().NoError(enforcer.Save(s.ctx, store.ScopeApp, s.app,
		store.PolicyDocument{Budget: &store.Budget{LimitMicros: 1_000_000, Interval: store.BudgetDaily}}))
	s.spend(s.app, 1_500_000)

	spend := s.spendRefused(enforcer, s.app, ErrBudgetSpent)
	s.ErrorContains(spend, "app's daily budget of $1.00")
	enforcer.Close()
	before := reads.reads.Load()
	for range 100 {
		s.ErrorIs(s.admit(enforcer, s.app).Spend, ErrBudgetSpent)
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
		return strings.Contains(enforcer.logs.String(), "expired before the usage service was known to count it") &&
			strings.Contains(enforcer.logs.String(), "micros=1250000")
	}, 5*time.Second, 10*time.Millisecond, "and how much is logged")
}

func (s *PolicySuite) TestAHoldOutlivesAnOutageOfTheUsageService() {
	for _, acked := range []bool{false, true} {
		s.Run(fmt.Sprintf("acked %t", acked), func() {
			s.SetupTest()
			enforcer := s.billed()
			s.usage().Credit(s.T(), s.org, 1_000_000)
			enforcer.Join(s.app, s.org, openTier)
			s.read(enforcer.Enforcer, s.org)
			usageID := uuid.Must(uuid.NewV7())
			enforcer.Spent(s.owner(), usageID, 1_000_000)
			s.Require().ErrorIs(s.admit(enforcer.Enforcer, s.app).Spend, ErrNoCredits)
			enforcer.front.down.Store(true)
			if acked {
				enforcer.Acked([]routing.Acked{{UsageID: usageID, OrganizationID: s.org, CostMicros: 1_000_000}})
			}

			app := s.app
			s.never(func() bool { return s.admit(enforcer.Enforcer, app).Spend == nil }, 4*quick.PendingTTL, 10*time.Millisecond,
				"an open tier answers on its last balance, less what was spent since, however long the outage")
			s.EqualValues(1_000_000, balanceHeld(enforcer.Enforcer, s.org))
			s.Contains(enforcer.logs.String(), "with no read of its balance to count it, so it is kept")

			enforcer.front.down.Store(false)
			recovered := time.Now()
			s.read(enforcer.Enforcer, s.org)
			if acked {
				s.eventually(func() bool { return balanceHeld(enforcer.Enforcer, s.org) == 0 }, 5*time.Second, 10*time.Millisecond,
					"an acked row is in the first standing asked for after the ack")
				s.NotContains(enforcer.logs.String(), "expired before")
				return
			}
			s.EqualValues(1_000_000, balanceHeld(enforcer.Enforcer, s.org),
				"the first read after an outage may come before the sweeper ships the backlog")
			s.never(func() bool { return balanceHeld(enforcer.Enforcer, s.org) == 0 },
				quick.PendingTTL/3, 10*time.Millisecond, "its timer starts again from the recovery")
			s.eventually(func() bool { return balanceHeld(enforcer.Enforcer, s.org) == 0 }, 5*time.Second, 10*time.Millisecond,
				"let go PendingTTL after the recovery, once a read lands")
			s.GreaterOrEqual(time.Since(recovered), quick.PendingTTL-quick.Fresh, "from the last failed read, at most Fresh before")
			s.Contains(enforcer.logs.String(), "expired before the usage service was known to count it")
			s.consistent(enforcer.Enforcer, s.org)
		})
	}
}

func (s *PolicySuite) TestAnUnackedHoldGoesOnceTheShardSaysItShipped() {
	enforcer := s.billed(func(settings *Settings) { settings.PendingTTL = time.Hour })
	s.usage().Credit(s.T(), s.org, 1_000_000)
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)
	usageID := uuid.Must(uuid.NewV7())
	enforcer.Spent(s.owner(), usageID, 1_000_000)
	s.Require().ErrorIs(s.admit(enforcer.Enforcer, s.app).Spend, ErrNoCredits)

	// Another router's sweeper ships the row, so no ack of it reaches this one.
	s.Require().NoError(s.store.RecordRequest(s.ctx, &store.Request{
		UsageID: usageID, Modality: "llm", CustomerID: s.app, Provider: "openai", Model: "gpt-5.6-luna",
		StartedAt: time.Now().UTC(), CostMicros: 1_000_000, Success: true, ToShip: true,
	}))
	s.Require().NoError(s.store.RecordShipment(s.ctx, []uuid.UUID{usageID}, nil))

	s.eventually(func() bool { return balanceHeld(enforcer.Enforcer, s.org) == 0 }, 5*time.Second, 10*time.Millisecond,
		"let go once a standing asked for after the shard said so lands, long before its timer")
	s.NotContains(enforcer.logs.String(), "expired before")
}

func (s *PolicySuite) TestAShardThatDoesNotAnswerKeepsUnackedHolds() {
	db, err := store.Open(os.Getenv("ROUTER_POSTGRES_DSN"))
	s.Require().NoError(err)
	configs, err := appconfig.New(appconfig.Options{Store: db})
	s.Require().NoError(err)
	enforcer := s.billedOn(configs)
	s.usage().Credit(s.T(), s.org, 1_000_000)
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)
	enforcer.Spent(s.owner(), uuid.Must(uuid.NewV7()), 400_000)

	// Postgres is gone while the usage service answers: the writer could not have written
	// the row, nor a sweeper shipped it.
	s.Require().NoError(db.Close())

	s.never(func() bool { return balanceHeld(enforcer.Enforcer, s.org) == 0 }, 4*quick.PendingTTL, 10*time.Millisecond,
		"its timer starts again at every pass the shard does not answer")
	s.Contains(enforcer.logs.String(), "past its pending_ttl with no read of its balance to count it, so it is kept")
}

func (s *PolicySuite) TestBalanceHoldsGoOnTheStandingsWhileBudgetsCannotBeRead() {
	enforcer := s.billed()
	s.usage().Credit(s.T(), s.org, 1_000_000)
	s.Require().NoError(enforcer.Save(s.ctx, store.ScopeApp, s.app,
		store.PolicyDocument{Budget: &store.Budget{LimitMicros: 10_000_000, Interval: store.BudgetDaily}}))
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)
	locked, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer locked.Rollback() //nolint:errcheck // nothing was written
	_, err = locked.ExecContext(s.ctx, "LOCK TABLE requests IN ACCESS EXCLUSIVE MODE")
	s.Require().NoError(err)

	usageID := uuid.Must(uuid.NewV7())
	enforcer.Spent(s.owner(), usageID, 400_000)
	enforcer.Acked([]routing.Acked{{UsageID: usageID, OrganizationID: s.org, CostMicros: 400_000}})
	enforcer.Written([]routing.Acked{{UsageID: usageID, OrganizationID: s.org, CostMicros: 400_000}})

	s.eventually(func() bool { return balanceHeld(enforcer.Enforcer, s.org) == 0 }, 5*time.Second, 10*time.Millisecond,
		"the balance counts it once the usage service does")
	s.NoError(s.admit(enforcer.Enforcer, s.app).Spend, "no insufficient_credits for spend the balance already counts")
	s.EqualValues(400_000, budgetHeld(enforcer.Enforcer, s.org), "the budgets wait for a sum that counts it")

	s.Require().NoError(locked.Rollback())
	s.eventually(func() bool { return budgetHeld(enforcer.Enforcer, s.org) == 0 }, 10*time.Second, 10*time.Millisecond,
		"and let it go once one lands")
	s.consistent(enforcer.Enforcer, s.org)
}

func (s *PolicySuite) TestBudgetHoldsGoOnTheShardWhileTheUsageServiceIsDown() {
	enforcer := s.billed()
	s.usage().Credit(s.T(), s.org, 10_000_000)
	s.Require().NoError(enforcer.Save(s.ctx, store.ScopeApp, s.app,
		store.PolicyDocument{Budget: &store.Budget{LimitMicros: 1_000_000, Interval: store.BudgetDaily}}))
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)
	enforcer.front.down.Store(true)

	usageID := uuid.Must(uuid.NewV7())
	enforcer.Spent(s.owner(), usageID, 400_000)
	s.spend(s.app, 400_000)
	enforcer.Written([]routing.Acked{{UsageID: usageID, OrganizationID: s.org, CostMicros: 400_000}})

	s.eventually(func() bool { return budgetHeld(enforcer.Enforcer, s.org) == 0 }, 5*time.Second, 10*time.Millisecond,
		"the budget's sum counts the written row, so it is not counted twice")
	s.EqualValues(400_000, balanceHeld(enforcer.Enforcer, s.org), "the balance waits for the usage service")
	s.consistent(enforcer.Enforcer, s.org)
}

func (s *PolicySuite) TestAHoldMadeBeforeMidnightCountsAgainstTheBalanceButNotTheNewDay() {
	var clock atomic.Pointer[time.Time]
	enforcer := s.billed(func(settings *Settings) {
		settings.PendingTTL = time.Hour
		settings.Now = func() time.Time {
			if moved := clock.Load(); moved != nil {
				return *moved
			}
			return time.Now()
		}
	})
	s.usage().Credit(s.T(), s.org, 10_000_000)
	s.usage().SetLimits(s.T(), s.org, spendHardLimit(new(int64(1_000_000)), nil))
	s.Require().NoError(enforcer.Save(s.ctx, store.ScopeApp, s.app,
		store.PolicyDocument{Budget: &store.Budget{LimitMicros: 5_000_000, Interval: store.BudgetDaily}}))
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)
	enforcer.front.down.Store(true)
	enforcer.Spent(s.owner(), uuid.Must(uuid.NewV7()), 1_000_000)
	s.ErrorIs(s.admit(enforcer.Enforcer, s.app).Spend, ErrSystemLimit, "today's limit counts today's hold")

	_, midnight := usage.Window(usage.Daily, time.Now())
	tomorrow := midnight.Add(time.Second)
	clock.Store(&tomorrow)

	s.NoError(s.admit(enforcer.Enforcer, s.app).Spend, "yesterday's hold is not spent against today's limit")
	s.EqualValues(9_000_000, available(enforcer.Enforcer, s.org), "but the balance still counts it")
	enforcer.money.mu.Lock()
	acct := enforcer.money.accounts[s.org]
	today := acct.heldAgainstBudget(scoped{scope: store.ScopeApp, id: s.app,
		budget: store.Budget{LimitMicros: 5_000_000, Interval: store.BudgetDaily}}, tomorrow)
	yesterday := acct.heldAgainstBudget(scoped{scope: store.ScopeApp, id: s.app,
		budget: store.Budget{LimitMicros: 5_000_000, Interval: store.BudgetDaily}}, midnight.Add(-time.Second))
	enforcer.money.mu.Unlock()
	s.Zero(today, "nor against today's budget")
	s.EqualValues(1_000_000, yesterday, "it stays in the window it was made in")
}

func (s *PolicySuite) TestAnAccountKeepsAtMostSoManyHoldsApart() {
	enforcer := s.billed()
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)
	enforcer.front.down.Store(true)
	ids := s.fold(enforcer)

	s.EqualValues(3*len(ids), pending(enforcer.Enforcer, s.org), "folding them counts the same money")
	s.consistent(enforcer.Enforcer, s.org)
	s.Contains(enforcer.logs.String(), "so the oldest is folded")

	enforcer.Spent(s.owner(), ids[0], 3)
	s.EqualValues(3*len(ids), pending(enforcer.Enforcer, s.org), "a folded row recorded again is held once")
	s.consistent(enforcer.Enforcer, s.org)
}

func (s *PolicySuite) TestAFoldedHoldKeepsItsAck() {
	for _, when := range []string{"before folding", "after folding"} {
		s.Run(when, func() {
			s.SetupTest()
			enforcer := s.billed(func(settings *Settings) { settings.PendingTTL = time.Hour })
			s.usage().Credit(s.T(), s.org, 1_000_000_000)
			enforcer.Join(s.app, s.org, openTier)
			s.read(enforcer.Enforcer, s.org)
			enforcer.front.down.Store(true)
			var ids []uuid.UUID
			if when == "before folding" {
				ids = s.hold(enforcer, maxHolds+10)
				s.ack(enforcer, ids)
				s.folded(enforcer)
			} else {
				ids = s.fold(enforcer)
				s.ack(enforcer, ids)
			}
			s.consistent(enforcer.Enforcer, s.org)

			enforcer.front.down.Store(false)
			s.eventually(func() bool { return balanceHeld(enforcer.Enforcer, s.org) == 0 }, 5*time.Second, 10*time.Millisecond,
				"every acked row goes with the first standing after its ack, folded or not")
			s.NotContains(enforcer.logs.String(), "expired before", "an acked fold is not taken for one the ledger never had")
			s.consistent(enforcer.Enforcer, s.org)
		})
	}
}

func (s *PolicySuite) TestAPolicyReadThatFailsAdmitsOnceAndIsNotKept() {
	configs, err := appconfig.New(appconfig.Options{Store: s.store})
	s.Require().NoError(err)
	enforcer := s.newEnforcer(configs, nil)
	locked, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer locked.Rollback() //nolint:errcheck // nothing was written
	_, err = locked.ExecContext(s.ctx, "LOCK TABLE policies IN ACCESS EXCLUSIVE MODE")
	s.Require().NoError(err)

	started := time.Now()
	admitted, err := enforcer.Admit(s.ctx, s.app)

	s.Require().NoError(err, "admitted on a read that did not land")
	s.NoError(admitted.Spend)
	s.Less(time.Since(started), 2*joinTimeout, "bounded by its own timeout")
	_, kept := enforcer.decided(s.app)
	s.False(kept, "nothing unread is kept to answer the next request")

	s.Require().NoError(locked.Rollback())
	_, err = enforcer.Admit(s.ctx, s.app)
	s.Require().NoError(err)
	_, kept = enforcer.decided(s.app)
	s.True(kept, "the next request reads them, and keeps what it read")
}

// hold holds n rows of 3 micros each against the suite's organization.
func (s *PolicySuite) hold(enforcer *reading, n int) []uuid.UUID {
	ids := make([]uuid.UUID, n)
	for i := range ids {
		ids[i] = uuid.Must(uuid.NewV7())
		enforcer.Spent(s.owner(), ids[i], 3)
	}
	return ids
}

// fold holds more rows than an account keeps apart and waits for the oldest to be folded.
func (s *PolicySuite) fold(enforcer *reading) []uuid.UUID {
	ids := s.hold(enforcer, maxHolds+10)
	s.folded(enforcer)
	return ids
}

func (s *PolicySuite) folded(enforcer *reading) {
	s.eventually(func() bool {
		enforcer.money.mu.Lock()
		defer enforcer.money.mu.Unlock()
		return len(enforcer.money.accounts[s.org].folded) > 0
	}, 5*time.Second, 10*time.Millisecond, "the oldest are folded")
}

func (s *PolicySuite) ack(enforcer *reading, ids []uuid.UUID) {
	acked := make([]routing.Acked, len(ids))
	for i, id := range ids {
		acked[i] = routing.Acked{UsageID: id, OrganizationID: s.org, CostMicros: 3}
	}
	enforcer.Acked(acked)
}

func (s *PolicySuite) TestStandingsStayFreshWhenBudgetsCannotBeRead() {
	enforcer := s.billed()
	s.usage().Credit(s.T(), s.org, 1_000_000)
	s.Require().NoError(enforcer.Save(s.ctx, store.ScopeApp, s.app,
		store.PolicyDocument{Budget: &store.Budget{LimitMicros: 1_000_000, Interval: store.BudgetDaily}}))
	enforcer.Join(s.app, s.org, closedTier)
	s.read(enforcer.Enforcer, s.org)
	locked, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer locked.Rollback() //nolint:errcheck // nothing was written
	_, err = locked.ExecContext(s.ctx, "LOCK TABLE requests IN ACCESS EXCLUSIVE MODE")
	s.Require().NoError(err)

	app, organization := s.app, s.org
	s.never(func() bool { return s.admit(enforcer.Enforcer, app).Spend != nil }, 3*quick.MaxStale, 10*time.Millisecond,
		"a closed tier is not refused for a balance that is read on time while the budgets wait on Postgres")
	s.read(enforcer.Enforcer, organization)
}

func (s *PolicySuite) TestAChangedBudgetIsHeldToTheLastMeasureAtOnce() {
	configs, err := appconfig.New(appconfig.Options{Store: s.store})
	s.Require().NoError(err)
	enforcer := s.newEnforcer(configs, nil, func(settings *Settings) { settings.BudgetEvery = time.Hour })
	s.Require().NoError(enforcer.Save(s.ctx, store.ScopeApp, s.app,
		store.PolicyDocument{Budget: &store.Budget{LimitMicros: 2_000_000, Interval: store.BudgetDaily}}))
	s.spend(s.app, 1_500_000)
	s.Require().NoError(s.admit(enforcer, s.app).Spend)
	s.eventually(func() bool {
		enforcer.money.mu.Lock()
		defer enforcer.money.mu.Unlock()
		acct := enforcer.money.accounts[s.org]
		return acct != nil && len(acct.measured) == 1
	}, 5*time.Second, 10*time.Millisecond, "measured once, when the organization was first wanted")

	s.Require().NoError(enforcer.Save(s.ctx, store.ScopeApp, s.app,
		store.PolicyDocument{Budget: &store.Budget{LimitMicros: 1_000_000, Interval: store.BudgetDaily}}))

	s.ErrorIs(s.admit(enforcer, s.app).Spend, ErrBudgetSpent, "the lower limit against what was measured, with no new sum")
}

func (s *PolicySuite) TestAnAppInNoOrganizationLetsGoOfItsHoldOnceWritten() {
	enforcer := s.billed()
	alone := routing.Owner{CustomerID: s.app + "-alone", Attribution: routing.Attribution{APIKey: "key-a"}}
	usageID := uuid.Must(uuid.NewV7())
	enforcer.Spent(alone, usageID, 1_000_000)
	s.Require().EqualValues(1_000_000, pending(enforcer.Enforcer, appAccount(alone.CustomerID)))

	enforcer.Written([]routing.Acked{{UsageID: usageID, CostMicros: 1_000_000}})

	s.eventually(func() bool { return pending(enforcer.Enforcer, appAccount(alone.CustomerID)) == 0 }, 5*time.Second, 10*time.Millisecond,
		"the usage service never ships a row with no organization, so the shard is its ledger")
}

func (s *PolicySuite) TestAnOrganizationsHoldWaitsForTheUsageServiceNotTheShard() {
	enforcer := s.billed()
	enforcer.Join(s.app, s.org, openTier)
	usageID := uuid.Must(uuid.NewV7())
	enforcer.Spent(s.owner(), usageID, 1_000_000)

	enforcer.Written([]routing.Acked{{UsageID: usageID, OrganizationID: s.org, CostMicros: 1_000_000}})

	enforcer.money.mu.Lock()
	defer enforcer.money.mu.Unlock()
	s.True(enforcer.money.accounts[s.org].holds[usageID].ackedAt.IsZero(), "the balance is the usage service's")
}

func (s *PolicySuite) TestShadowCountsEveryRefusalAndSaysWhatItRead() {
	enforcer := s.billed(func(settings *Settings) { settings.Mode = config.LimitsShadow })
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)
	before := testutil.ToFloat64(wouldRefuse.WithLabelValues(reasonNoCredit))

	for range 5 {
		s.NoError(s.admit(enforcer.Enforcer, s.app).Spend)
	}

	s.GreaterOrEqual(testutil.ToFloat64(wouldRefuse.WithLabelValues(reasonNoCredit))-before, 5.0, "every one, though the log is throttled")
	logged := enforcer.logs.String()
	s.Equal(1, strings.Count(logged, `msg="would refuse"`), "once a minute per organization and reason")
	for _, field := range []string{"balance_micros=0", "held_micros=0", "read_age=", "organization=" + s.org, "app=" + s.app} {
		s.Contains(logged, field)
	}
}

func (s *PolicySuite) TestShadowNeverSaysASuspendedOrganizationWasRefused() {
	enforcer := s.billed(func(settings *Settings) { settings.Mode = config.LimitsShadow })
	s.usage().Credit(s.T(), s.org, 1_000_000)
	s.usage().Suspend(s.T(), s.org, "a test")
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)

	s.NoError(s.admit(enforcer.Enforcer, s.app).Spend)

	s.Contains(enforcer.logs.String(), "reason=suspended")
	s.Contains(enforcer.logs.String(), `status_reason="a test"`)
	s.NotContains(enforcer.logs.String(), "refused a suspended")
}

func (s *PolicySuite) TestAnAppsOwnAccountIsNamedAsTheApp() {
	s.Equal([]any{"app", "app-1"}, who(appAccount("app-1")))
	s.Equal([]any{"organization", "org-1"}, who("org-1"))
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
