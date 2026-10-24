//go:build integration

package policy

import (
	"context"
	"net"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

// sharing is how fast the reservation tests share holds: a flush every 10 ms, the other
// routers' holds read every 50 ms, and billable work refused once a hold waits 200 ms unshared.
func sharing(redis rueidis.Client) func(*Settings) {
	return func(settings *Settings) {
		settings.Holds = redis
		settings.FlushEvery, settings.HoldRefresh, settings.FlushMaxLag = 10*time.Millisecond, 50*time.Millisecond, 200*time.Millisecond
	}
}

// redisProxy is the suites' Redis behind a TCP proxy the test controls: held, every command
// waits at the proxy until it is let go; down, every connection is closed and new ones refused.
type redisProxy struct {
	listener net.Listener
	held     atomic.Bool
	down     atomic.Bool
	mu       sync.Mutex
	conns    []net.Conn
}

// redis is a client of the suites' Redis through a proxy of the test's own.
func (s *PolicySuite) redis() (*redisProxy, rueidis.Client) {
	target := os.Getenv("ROUTER_REDIS_ADDR")
	if target == "" {
		s.T().Skip("ROUTER_REDIS_ADDR not set")
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	p := &redisProxy{listener: listener}
	s.T().Cleanup(func() {
		_ = listener.Close()
		p.stop()
	})
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			if p.down.Load() {
				_ = conn.Close()
				continue
			}
			upstream, err := net.Dial("tcp", target)
			if err != nil {
				_ = conn.Close()
				continue
			}
			p.mu.Lock()
			p.conns = append(p.conns, conn, upstream)
			p.mu.Unlock()
			go p.pump(upstream, conn)
			go p.pump(conn, upstream)
		}
	}()
	options, err := rueidis.ParseURL("redis://" + listener.Addr().String() + "/" + strconv.Itoa(usagetest.RedisDB))
	s.Require().NoError(err)
	options.DisableCache = true
	client, err := rueidis.NewClient(options)
	s.Require().NoError(err)
	s.T().Cleanup(client.Close)
	return p, client
}

// pump copies one direction of a connection, waiting while the proxy holds it.
func (p *redisProxy) pump(to, from net.Conn) {
	defer func() { _ = to.Close() }()
	buf := make([]byte, 32<<10)
	for {
		n, err := from.Read(buf)
		for p.held.Load() && !p.down.Load() {
			time.Sleep(time.Millisecond)
		}
		if n > 0 && !p.down.Load() {
			if _, werr := to.Write(buf[:n]); werr != nil {
				return
			}
		}
		if err != nil || p.down.Load() {
			return
		}
	}
}

// stop takes Redis away: every connection through the proxy is closed, and every one after.
func (p *redisProxy) stop() {
	p.down.Store(true)
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, conn := range p.conns {
		_ = conn.Close()
	}
	p.conns = nil
}

// direct is a client of the suites' Redis, for reading and deleting the test's own keys.
func (s *PolicySuite) direct() rueidis.Client {
	client, err := rueidis.NewClient(rueidis.ClientOption{
		InitAddress: []string{os.Getenv("ROUTER_REDIS_ADDR")}, SelectDB: usagetest.RedisDB, DisableCache: true,
	})
	s.Require().NoError(err)
	s.T().Cleanup(client.Close)
	return client
}

// dropHolds deletes every shared hold of the organization, as Redis losing its data does to
// it, and nothing of anyone else's.
func (s *PolicySuite) dropHolds(client rueidis.Client, organization string) int {
	var cursor uint64
	dropped := 0
	for {
		entry, err := client.Do(s.ctx, client.B().Scan().Cursor(cursor).Match(holdPrefix+":{"+organization+"}:*").Count(1000).Build()).AsScanEntry()
		s.Require().NoError(err)
		for _, key := range entry.Elements {
			s.Require().NoError(client.Do(s.ctx, client.B().Del().Key(key).Build()).Error())
			dropped++
		}
		if entry.Cursor == 0 {
			return dropped
		}
		cursor = entry.Cursor
	}
}

// sharer is a billed enforcer sharing holds through redis, whose organization has been read.
func (s *PolicySuite) sharer(redis rueidis.Client, change ...func(*Settings)) *reading {
	enforcer := s.billed(append([]func(*Settings){sharing(redis)}, change...)...)
	s.T().Cleanup(func() { s.dropHolds(s.direct(), s.org) })
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)
	return enforcer
}

// availableShared is the organization's balance as a reservation counts it: the standing's,
// less every hold, the other routers' included.
func availableShared(enforcer *Enforcer, organization string) int64 {
	enforcer.money.mu.Lock()
	defer enforcer.money.mu.Unlock()
	acct := enforcer.money.accounts[organization]
	if acct == nil {
		return 0
	}
	return acct.standing.BalanceMicros - acct.held - acct.remoteOrganization(organization)
}

// remoteHeld is what an enforcer counts the other routers as holding against the organization,
// asked as an admission of the app's, which keeps the organization in use there.
func remoteHeld(enforcer *Enforcer, organization string) int64 {
	_, _ = enforcer.Admit(context.Background(), ownerOf(appOf(enforcer, organization)))
	enforcer.money.mu.Lock()
	defer enforcer.money.mu.Unlock()
	acct := enforcer.money.accounts[organization]
	if acct == nil {
		return 0
	}
	return acct.remoteOrganization(organization)
}

// appOf is an app the enforcer was told is in the organization (Join), which it never forgets.
func appOf(enforcer *Enforcer, organization string) string {
	found := ""
	enforcer.members.Range(func(app, joined any) bool {
		if joined == organization {
			found = app.(string)
			return false
		}
		return true
	})
	return found
}

func newID() uuid.UUID { return uuid.Must(uuid.NewV7()) }

func (s *PolicySuite) TestAReservationCountsAtOnce() {
	proxy, redis := s.redis()
	s.usage().Credit(s.T(), s.org, 1_000_000)
	enforcer := s.sharer(redis)
	proxy.held.Store(true)
	defer proxy.held.Store(false)

	started := time.Now()
	first := enforcer.Reserve(s.owner(), newID(), 600_000, time.Minute)
	second := enforcer.Reserve(s.owner(), newID(), 600_000, time.Minute)
	third := enforcer.Reserve(s.owner(), newID(), 600_000, time.Minute)
	answered := time.Since(started)

	s.NoError(first)
	s.NoError(second, "there was room left, if not for all of it")
	s.ErrorIs(third, ErrNoCredits)
	s.Less(answered, 100*time.Millisecond, "answered from memory while every Redis command is held")
	s.EqualValues(-200_000, available(enforcer.Enforcer, s.org), "the last admitted unit may take the balance below zero by its estimate")
}

func (s *PolicySuite) TestTwoRoutersSeeEachOthersHolds() {
	_, redisA := s.redis()
	_, redisB := s.redis()
	s.usage().Credit(s.T(), s.org, 1_000_000)
	a, b := s.sharer(redisA), s.sharer(redisB)

	s.Require().NoError(a.Reserve(s.owner(), newID(), 900_000, time.Minute))
	reserved := time.Now()

	s.eventually(func() bool { return remoteHeld(b.Enforcer, s.org) == 900_000 }, time.Second, 5*time.Millisecond,
		"the other router reads it")
	s.Less(time.Since(reserved), 500*time.Millisecond, "within a flush and a refresh")
	s.NoError(b.Reserve(s.owner(), newID(), 200_000, time.Minute), "$0.10 is left, so the next unit may start")
	s.ErrorIs(b.Reserve(s.owner(), newID(), 200_000, time.Minute), ErrNoCredits)
	s.EqualValues(200_000, pending(b.Enforcer, s.org), "a refusal holds nothing")
	s.Zero(remoteHeld(a.Enforcer, s.org), "a router never counts its own holds twice")
	s.ErrorIs(s.admit(b.Enforcer, s.app).Spend, ErrNoCredits, "an admission counts them too")
}

func (s *PolicySuite) TestReservingTwiceUnderOneIDHoldsOnce() {
	_, redisA := s.redis()
	_, redisB := s.redis()
	s.usage().Credit(s.T(), s.org, 1_000_000)
	a, b := s.sharer(redisA), s.sharer(redisB)
	id := newID()

	s.Require().NoError(a.Reserve(s.owner(), id, 400_000, time.Minute))
	s.Require().NoError(a.Reserve(s.owner(), id, 400_000, time.Minute))
	s.Require().NoError(a.Reserve(s.owner(), id, 900_000, time.Minute), "its own first reservation is no reason to refuse it")

	s.EqualValues(100_000, available(a.Enforcer, s.org))
	s.eventually(func() bool { return remoteHeld(b.Enforcer, s.org) == 900_000 }, time.Second, 5*time.Millisecond,
		"the other router sees it once")
	s.EqualValues(100_000, availableShared(b.Enforcer, s.org))
}

func (s *PolicySuite) TestASettledHoldIsTheCost() {
	_, redisA := s.redis()
	_, redisB := s.redis()
	s.usage().Credit(s.T(), s.org, 1_000_000)
	a, b := s.sharer(redisA), s.sharer(redisB)
	id := newID()

	s.Require().NoError(a.Reserve(s.owner(), id, 500_000, time.Minute))
	a.Settle(s.owner(), id, 200_000)

	s.EqualValues(800_000, available(a.Enforcer, s.org))
	s.EqualValues(200_000, pending(a.Enforcer, s.org))
	s.eventually(func() bool { return availableShared(b.Enforcer, s.org) == 800_000 }, time.Second, 5*time.Millisecond,
		"the other router counts the cost, not the estimate")
}

func (s *PolicySuite) TestAHoldIsKeptUntilTheStandingHasTheRow() {
	_, redisA := s.redis()
	_, redisB := s.redis()
	s.usage().Credit(s.T(), s.org, 3_000_000)
	a, b := s.sharer(redisA), s.sharer(redisB)
	var over atomic.Int64
	sampling, stop := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(sampling)
		for {
			select {
			case <-stop:
				return
			case <-time.After(5 * time.Millisecond):
				_, _ = b.Admit(s.ctx, ownerOf(s.app))
				over.Store(max(over.Load(), availableShared(b.Enforcer, s.org)))
			}
		}
	}()
	id := newID()

	s.Require().NoError(a.Reserve(s.owner(), id, 1_500_000, time.Minute))
	s.eventually(func() bool { return remoteHeld(b.Enforcer, s.org) == 1_500_000 }, time.Second, 5*time.Millisecond)
	a.Settle(s.owner(), id, 1_000_000)
	s.eventually(func() bool { return remoteHeld(b.Enforcer, s.org) == 1_000_000 }, time.Second, 5*time.Millisecond)
	event := s.usage().Spend(s.T(), s.org, 1_000_000, func(event *usage.Event) { event.UsageID = id })
	a.Acked([]routing.Acked{{UsageID: event.UsageID, OrganizationID: s.org, CostMicros: 1_000_000}})
	acked := time.Now()
	s.eventually(func() bool { return remoteHeld(b.Enforcer, s.org) == 0 }, time.Second, 5*time.Millisecond,
		"let go on every router once their reads have the row")
	gone := time.Since(acked)
	close(stop)
	<-sampling

	s.GreaterOrEqual(gone, 2*quick.Fresh, "kept 2 × Fresh after the ack")
	s.EqualValues(2_000_000, availableShared(b.Enforcer, s.org))
	s.LessOrEqual(over.Load(), int64(3_000_000), "never more than is truly left")
	s.eventually(func() bool { return pending(a.Enforcer, s.org) == 0 }, time.Second, 5*time.Millisecond,
		"and on the router that made it")
}

func (s *PolicySuite) TestAnUnsettledHoldExpires() {
	_, redisA := s.redis()
	_, redisB := s.redis()
	s.usage().Credit(s.T(), s.org, 1_000_000)
	a, b := s.sharer(redisA), s.sharer(redisB)

	s.Require().NoError(a.Reserve(s.owner(), newID(), 700_000, 150*time.Millisecond))
	s.eventually(func() bool { return remoteHeld(b.Enforcer, s.org) == 700_000 }, time.Second, 5*time.Millisecond)

	s.eventually(func() bool { return pending(a.Enforcer, s.org) == 0 }, 2*time.Second, 5*time.Millisecond,
		"the work was never recorded, so its reservation goes")
	s.eventually(func() bool { return remoteHeld(b.Enforcer, s.org) == 0 }, time.Second, 5*time.Millisecond,
		"and Redis drops it at the same moment")
	s.consistent(a.Enforcer, s.org)
}

func (s *PolicySuite) TestASettledHoldWithNoAckExpires() {
	_, redisA := s.redis()
	_, redisB := s.redis()
	s.usage().Credit(s.T(), s.org, 1_000_000)
	a, b := s.sharer(redisA), s.sharer(redisB)

	settled := time.Now()
	a.Settle(s.owner(), newID(), 700_000)
	s.eventually(func() bool { return remoteHeld(b.Enforcer, s.org) == 700_000 }, time.Second, 5*time.Millisecond)

	s.eventually(func() bool { return remoteHeld(b.Enforcer, s.org) == 0 }, 2*time.Second, 5*time.Millisecond,
		"a row another router's sweeper shipped is never acked here")
	s.GreaterOrEqual(time.Since(settled), quick.PendingTTL, "after PendingTTL")
	s.eventually(func() bool { return pending(a.Enforcer, s.org) == 0 }, 2*time.Second, 5*time.Millisecond)
}

func (s *PolicySuite) TestRedisLosingItsDataChangesNoBalance() {
	_, redisA := s.redis()
	_, redisB := s.redis()
	s.usage().Credit(s.T(), s.org, 1_000_000)
	a, b := s.sharer(redisA), s.sharer(redisB)
	s.Require().NoError(a.Reserve(s.owner(), newID(), 300_000, time.Minute))
	s.eventually(func() bool { return availableShared(b.Enforcer, s.org) == 700_000 }, time.Second, 5*time.Millisecond)

	s.Positive(s.dropHolds(s.direct(), s.org))

	s.eventually(func() bool { return remoteHeld(b.Enforcer, s.org) == 0 }, time.Second, 5*time.Millisecond)
	s.EqualValues(1_000_000, availableShared(b.Enforcer, s.org), "it rises by what was held, and no more")
	s.EqualValues(700_000, available(a.Enforcer, s.org), "the router doing the work still holds it")
	s.read(b.Enforcer, s.org)
	b.money.mu.Lock()
	balance := b.money.accounts[s.org].standing.BalanceMicros
	b.money.mu.Unlock()
	s.EqualValues(1_000_000, balance, "the ledger is untouched")
}

func (s *PolicySuite) TestRedisDownAdmitsFromMemoryThenRefuses() {
	proxy, redis := s.redis()
	s.usage().Credit(s.T(), s.org, 100_000_000)
	enforcer := s.sharer(redis)
	proxy.stop()
	stopped := time.Now()

	s.NoError(enforcer.Reserve(s.owner(), newID(), 1_000, time.Minute), "admitted from memory")
	var refused error
	s.eventually(func() bool {
		refused = enforcer.Reserve(s.owner(), newID(), 1_000, time.Minute)
		return refused != nil
	}, 2*time.Second, 5*time.Millisecond)
	s.GreaterOrEqual(time.Since(stopped), 200*time.Millisecond, "only once a hold waited FlushMaxLag unshared")
	s.ErrorIs(refused, ErrSpendCheckUnavailable)
	s.Equal("spend_check_unavailable", Code(refused))
	admitted, err := enforcer.Admit(s.ctx, ownerOf(s.app))
	s.NoError(err, "free work is admitted")
	s.ErrorIs(admitted.Spend, ErrSpendCheckUnavailable, "billable work is not")
	s.eventually(func() bool {
		return strings.Contains(enforcer.logs.String(), "could not share holds with the other routers")
	},
		2*time.Second, 5*time.Millisecond, "logged once the flush gives up")

	proxy.down.Store(false)
	restored := time.Now()
	s.eventually(func() bool { return enforcer.Reserve(s.owner(), newID(), 1_000, time.Minute) == nil }, 5*time.Second, 5*time.Millisecond)
	s.Less(time.Since(restored), 2*time.Second, "admitted again once a flush lands")
}

func (s *PolicySuite) TestWithoutRedisHoldsStayLocalAndNeverRefuse() {
	s.usage().Credit(s.T(), s.org, 1_000_000)
	enforcer := s.billed(func(settings *Settings) { settings.FlushMaxLag = 50 * time.Millisecond })
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)

	s.Require().NoError(enforcer.Reserve(s.owner(), newID(), 400_000, time.Minute))
	time.Sleep(150 * time.Millisecond)

	s.NoError(enforcer.Reserve(s.owner(), newID(), 400_000, time.Minute), "nothing to share, so nothing waits to be")
	s.EqualValues(200_000, available(enforcer.Enforcer, s.org), "held in this process")
	s.NoError(enforcer.Reserve(s.owner(), newID(), 400_000, time.Minute), "some room is left")
	s.ErrorIs(enforcer.Reserve(s.owner(), newID(), 400_000, time.Minute), ErrNoCredits)
}

func (s *PolicySuite) TestALeaseIsRenewedAndReleased() {
	_, redisA := s.redis()
	_, redisB := s.redis()
	s.usage().Credit(s.T(), s.org, 1_000_000)
	a, b := s.sharer(redisA), s.sharer(redisB)
	lease := newID()

	var seen int64
	for range 10 {
		s.Require().NoError(a.Reserve(s.owner(), lease, 800_000, 150*time.Millisecond), "renewed under its own amount")
		time.Sleep(40 * time.Millisecond)
		seen = max(seen, remoteHeld(b.Enforcer, s.org))
	}
	s.EqualValues(800_000, pending(a.Enforcer, s.org), "renewed past its first ttl")
	s.EqualValues(800_000, seen, "and seen by the other router")
	s.eventually(func() bool { return remoteHeld(b.Enforcer, s.org) == 800_000 }, 100*time.Millisecond, 5*time.Millisecond)

	a.Release(lease)

	s.Zero(pending(a.Enforcer, s.org))
	s.eventually(func() bool { return remoteHeld(b.Enforcer, s.org) == 0 }, time.Second, 5*time.Millisecond,
		"released on every router")
}

func (s *PolicySuite) TestShadowHoldsWhatItWouldRefuse() {
	_, redis := s.redis()
	s.usage().Credit(s.T(), s.org, 1_000_000)
	enforcer := s.sharer(redis, func(settings *Settings) { settings.Mode = "shadow" })

	s.NoError(enforcer.Reserve(s.owner(), newID(), 1_200_000, time.Minute))
	s.NoError(enforcer.Reserve(s.owner(), newID(), 300_000, time.Minute), "logged, not refused")

	s.EqualValues(-500_000, available(enforcer.Enforcer, s.org), "and held, so a shadow day shows the numbers")
	s.Contains(enforcer.logs.String(), "would refuse")
	s.Contains(enforcer.logs.String(), "reason=no_credit")
}

// grantRemaining is what a router counts a grant as having left: its amount, less what it had
// paid for at the last read, less every router's holds against it.
func grantRemaining(enforcer *Enforcer, organization string, id int64) (int64, bool) {
	enforcer.money.mu.Lock()
	defer enforcer.money.mu.Unlock()
	acct := enforcer.money.accounts[organization]
	if acct == nil {
		return 0, false
	}
	for _, grant := range acct.grants {
		if grant.grant.ID == id {
			return grant.remaining(acct.grantsHeld()), true
		}
	}
	return 0, false
}

func (s *PolicySuite) TestAGrantsRemainingNeverRisesPastTheTruthAcrossItsCount() {
	_, redisA := s.redis()
	_, redisB := s.redis()
	configs := func() *appconfig.Store {
		c, err := appconfig.New(appconfig.Options{Store: s.store})
		s.Require().NoError(err)
		return c
	}
	// Both routers read the grants once a second, so the other reads a count up to a second
	// after this one has let its hold go.
	slow := func(settings *Settings) { settings.Fresh, settings.MaxStale = time.Second, time.Minute }
	a := s.newEnforcer(configs(), nil, sharing(redisA), slow)
	// Out of step with the first, as two routers started apart are.
	time.Sleep(500 * time.Millisecond)
	b := s.newEnforcer(configs(), nil, sharing(redisB), slow)
	s.T().Cleanup(func() { s.dropHolds(s.direct(), s.org) })
	s.limit(store.Limit{Subject: usage.SubjectApp, Level: store.LevelApp, OrganizationID: s.org, AppID: s.app, LimitMicros: 1_000_000})
	s.spend(s.app, 1_000_000)
	grant := store.LimitGrant{OrganizationID: s.org, Subject: usage.SubjectApp, AppID: s.app, AmountMicros: 1_000_000, Reason: "launch", Actor: "test"}
	s.Require().NoError(s.store.AddGrant(s.ctx, &grant, store.LimitAudit{CustomerID: s.app, Source: store.AuditSourceAPI}))
	for _, enforcer := range []*Enforcer{a, b} {
		s.eventually(func() bool {
			admitted := s.admit(enforcer, s.app)
			return admitted.Spend == nil && slices.Contains(admitted.Grants, grant.ID)
		}, 10*time.Second, 10*time.Millisecond, "admitted on the grant once the limit and the grant are read")
	}
	owner := routing.Owner{CustomerID: s.app, Attribution: routing.Attribution{OrganizationID: s.org}}

	// truth is what the grant has left, its holds counted; the other router must never count more.
	var truth, worst atomic.Int64
	truth.Store(1_000_000)
	sampling, stop := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(sampling)
		for {
			select {
			case <-stop:
				return
			case <-time.After(2 * time.Millisecond):
				_, _ = b.Admit(s.ctx, ownerOf(s.app))
				if left, ok := grantRemaining(b, s.org, grant.ID); ok {
					worst.Store(max(worst.Load(), left-truth.Load()))
				}
			}
		}
	}()
	defer func() {
		close(stop)
		<-sampling
	}()
	const cost = 600_000
	id := newID()
	s.Require().NoError(a.Reserve(owner, id, cost, time.Minute))
	s.Require().Equal([]int64{grant.ID}, a.Settle(owner, id, cost), "settled on the grant it was reserved on")
	s.eventually(func() bool {
		left, ok := grantRemaining(b, s.org, grant.ID)
		return ok && left == 1_000_000-cost
	}, time.Second, 2*time.Millisecond, "the other router counts the hold")
	truth.Store(1_000_000 - cost)

	s.spendAs(store.Request{UsageID: id, CustomerID: s.app, CostMicros: cost, LimitGrants: []int64{grant.ID}})
	_, err := s.store.AddGrantUses(s.ctx, []store.GrantUse{{GrantID: grant.ID, UsageID: id, Micros: cost}})
	s.Require().NoError(err)
	a.Written([]routing.Acked{{UsageID: id, OrganizationID: s.org, CostMicros: cost}})
	a.Counted([]routing.Acked{{UsageID: id, OrganizationID: s.org, CostMicros: cost}})
	// This router reads the grants at once and lets go of its hold; the other's next read is
	// up to a second away.
	nudge(a.money.kickGrants)
	s.eventually(func() bool {
		a.money.mu.Lock()
		defer a.money.mu.Unlock()
		return a.money.accounts[s.org].grantHeld[grant.ID] == 0
	}, 3*time.Second, 5*time.Millisecond, "let go here once this router's read has the count")
	s.eventually(func() bool {
		b.money.mu.Lock()
		defer b.money.mu.Unlock()
		return b.money.accounts[s.org].remoteGrants[grant.ID] == 0
	}, 4*time.Second, 5*time.Millisecond, "and on the other once its shared copy is gone")
	s.eventually(func() bool {
		left, _ := grantRemaining(b, s.org, grant.ID)
		return left == 1_000_000-cost
	}, 3*time.Second, 5*time.Millisecond, "what the grant paid for is read")

	s.LessOrEqual(worst.Load(), int64(0), "the other router never counted the grant as having more than it has")
}

func (s *PolicySuite) TestARouterNeverWritesOverAnothersHoldUnderTheSameID() {
	_, redisA := s.redis()
	_, redisB := s.redis()
	s.usage().Credit(s.T(), s.org, 1_000_000)
	a, b := s.sharer(redisA), s.sharer(redisB)
	id := newID()

	s.Require().NoError(a.Reserve(s.owner(), id, 300_000, time.Minute))
	s.Require().NoError(b.Reserve(s.owner(), id, 200_000, time.Minute))
	s.eventually(func() bool { return remoteHeld(a.Enforcer, s.org) == 200_000 }, time.Second, 5*time.Millisecond)
	s.eventually(func() bool { return remoteHeld(b.Enforcer, s.org) == 300_000 }, time.Second, 5*time.Millisecond,
		"each router's hold is its own field")

	a.Release(id)

	s.eventually(func() bool { return remoteHeld(b.Enforcer, s.org) == 0 }, time.Second, 5*time.Millisecond)
	time.Sleep(200 * time.Millisecond)
	s.eventually(func() bool { return remoteHeld(a.Enforcer, s.org) == 200_000 }, time.Second, 5*time.Millisecond,
		"letting go of its own leaves the other's")
}

func (s *PolicySuite) TestTheScriptExpiresHoldsByRedisClock() {
	client := s.direct()
	key := holdPrefix + ":{clock-" + newID().String() + "}:organization"
	s.T().Cleanup(func() { _ = client.Do(s.ctx, client.B().Del().Key(key).Build()).Error() })
	exec := func(instance string, args ...string) []int64 {
		values, err := holdsScript.Exec(s.ctx, client, []string{key}, append([]string{instance}, args...)).AsIntSlice()
		s.Require().NoError(err)
		return values
	}

	exec("a", "u1", "5:150")
	stored, err := client.Do(s.ctx, client.B().Hget().Key(key).Field("u1:a").Build()).ToString()
	s.Require().NoError(err)
	clock, err := client.Do(s.ctx, client.B().Time().Build()).AsStrSlice()
	s.Require().NoError(err)
	seconds, _ := strconv.ParseInt(clock[0], 10, 64)
	micros, _ := strconv.ParseInt(clock[1], 10, 64)
	expires, _ := strconv.ParseInt(strings.SplitN(stored, ":", 2)[1], 10, 64)
	s.InDelta(seconds*1000+micros/1000+150, expires, 100, "the expiry is Redis's now plus the hold's ttl")

	s.Equal([]int64{5, 1}, exec("b"), "another router counts it")
	time.Sleep(200 * time.Millisecond)
	s.Equal([]int64{0, 0}, exec("b"), "gone at its expiry, by Redis's clock")
}

func (s *PolicySuite) TestAFlushOfMoreKeysThanAChunkLandsWhole() {
	_, redisA := s.redis()
	_, redisB := s.redis()
	s.usage().Credit(s.T(), s.org, 100_000_000)
	a, b := s.sharer(redisA), s.sharer(redisB)
	users := 2 * flushChunk

	for i := range users {
		owner := s.owner()
		owner.UserID = "user-" + strconv.Itoa(i)
		s.Require().NoError(a.Reserve(owner, newID(), 1_000, time.Minute))
	}

	s.eventually(func() bool { return remoteHeld(b.Enforcer, s.org) == int64(users)*1_000 }, 2*time.Second, 5*time.Millisecond,
		"every hold reaches the organization's key, whatever chunk carried it")
	a.money.mu.Lock()
	unflushed := len(a.money.dirty)
	a.money.mu.Unlock()
	s.Zero(unflushed, "and each of its users' keys took it")
}

func (s *PolicySuite) TestAGrantHeldOnOneRouterIsHeldOnEvery() {
	_, redisA := s.redis()
	_, redisB := s.redis()
	configs := func() *appconfig.Store {
		c, err := appconfig.New(appconfig.Options{Store: s.store})
		s.Require().NoError(err)
		return c
	}
	a := s.newEnforcer(configs(), nil, sharing(redisA))
	b := s.newEnforcer(configs(), nil, sharing(redisB))
	s.T().Cleanup(func() { s.dropHolds(s.direct(), s.org) })
	s.limit(store.Limit{Subject: usage.SubjectApp, Level: store.LevelApp, OrganizationID: s.org, AppID: s.app, LimitMicros: 1_000_000})
	s.spend(s.app, 1_000_000)
	grant := store.LimitGrant{OrganizationID: s.org, Subject: usage.SubjectApp, AppID: s.app, AmountMicros: 1_000_000, Reason: "launch", Actor: "test"}
	s.Require().NoError(s.store.AddGrant(s.ctx, &grant, store.LimitAudit{CustomerID: s.app, Source: store.AuditSourceAPI}))
	for _, enforcer := range []*Enforcer{a, b} {
		s.eventually(func() bool { return s.admit(enforcer, s.app).Spend == nil }, 5*time.Second, 10*time.Millisecond,
			"admitted on the grant once it is read")
	}

	s.Require().NoError(a.Reserve(ownerOf(s.app), newID(), 1_000_000, time.Minute))

	var refused error
	s.eventually(func() bool {
		refused = b.Reserve(ownerOf(s.app), newID(), 100_000, time.Minute)
		return refused != nil
	}, time.Second, 5*time.Millisecond, "the other router sees the grant used up")
	s.ErrorIs(refused, ErrBudgetSpent)
}
