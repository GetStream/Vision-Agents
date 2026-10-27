//go:build integration

package routing

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

// testSource is the source the suite's writers and sweepers ship under.
var testSource = Source("test", "pod")

// usageService is the real usage service, emptied for the test.
func (s *WriterSuite) usageService() *usagetest.Service {
	service := usagetest.Start(s.T(), usagetest.Options{DSN: s.usageDSN, Logger: s.logger})
	usagetest.Empty(s.T(), service.Ledger)
	return service
}

// shippingWriter is a writer over the suite's store and Redis that ships to service.
func (s *WriterSuite) shippingWriter(service *usagetest.Service, change func(*WriterOptions)) *Writer {
	return s.newWriter(func(options *WriterOptions) {
		options.Usage, options.Source = service.Client, testSource
		if change != nil {
			change(options)
		}
	})
}

// sweeper is a sweeper of the suite's store that ships to service whatever is waiting, however
// new, with its gauges read as Run has them read. It gives its lock back when the test ends.
func (s *WriterSuite) sweeper(client *usage.Client) *Sweeper {
	return s.sweeperWith(SweeperOptions{Usage: client})
}

func (s *WriterSuite) sweeperWith(options SweeperOptions) *Sweeper {
	options.Store, options.Source, options.Logger, options.Settle = s.store, testSource, s.logger, -1
	sweeper := NewSweeper(options)
	sweepers.Store(sweeper, struct{}{})
	s.T().Cleanup(func() {
		sweepers.Delete(sweeper)
		sweeper.lock.Release()
	})
	return sweeper
}

// swept runs one sweep, which must hold the lock.
func (s *WriterSuite) swept(sweeper *Sweeper) {
	held, err := sweeper.sweep(s.ctx)
	s.Require().NoError(err)
	s.Require().True(held)
}

// owned is a row of the test's customer billed to the test's organization.
func (s *WriterSuite) owned(change func(*store.Request)) queued {
	return s.row(func(request *store.Request) {
		request.OrganizationID = s.organization
		if change != nil {
			change(request)
		}
	})
}

// shipped waits until the test's customer has n rows and none is left to ship, and returns them.
func (s *WriterSuite) shipped(n int) []store.Request {
	var rows []store.Request
	s.Require().Eventually(func() bool {
		rows = s.rows()
		if len(rows) != n {
			return false
		}
		for _, row := range rows {
			if row.ToShip {
				return false
			}
		}
		return true
	}, 10*time.Second, 20*time.Millisecond, "want %d rows shipped", n)
	return rows
}

// events is what the service recorded of the test's customer.
func (s *WriterSuite) events(service *usagetest.Service) []ledger.UsageEvent {
	var events []ledger.UsageEvent
	s.Require().NoError(service.Ledger.DB().NewSelect().Model(&events).Where("app_id = ?", s.customer).Order("occurred_at", "usage_id").Scan(s.ctx))
	return events
}

func (s *WriterSuite) TestAWrittenRowIsShipped() {
	before := testutil.ToFloat64(usageShipped)
	service := s.usageService()
	writer := s.shippingWriter(service, nil)
	// The last instant of yesterday, with nanoseconds no column keeps.
	midnight := time.Now().UTC().Truncate(24 * time.Hour)
	lastOfYesterday := midnight.Add(-500 * time.Nanosecond)
	row := s.owned(func(request *store.Request) { request.StartedAt = lastOfYesterday })

	s.Require().True(writer.enqueue(row))

	rows := s.shipped(1)
	s.NotNil(rows[0].ShippedAt)
	s.Empty(rows[0].ShipError)
	events := s.events(service)
	s.Require().Len(events, 1)
	s.Equal(row.request.UsageID, events[0].UsageID)
	s.EqualValues(25, events[0].CostMicros)
	s.Equal(s.organization, events[0].OrganizationID)
	s.Equal("llm", events[0].Meter)
	s.Equal(map[string]int64{"output_tokens": 10}, events[0].Quantities)
	s.Equal("router/test/pod", events[0].Source)
	s.Equal(midnight.Add(-time.Microsecond), rows[0].StartedAt.UTC(), "Postgres keeps the microsecond it was given")
	s.Equal(rows[0].StartedAt.UTC(), events[0].OccurredAt.UTC(), "the event happened when the row started, the same day")
	s.Equal(1.0, testutil.ToFloat64(usageShipped)-before)
}

func (s *WriterSuite) TestShippedRowsAreAckedOneByOne() {
	// Mirrored to the suites' own Redis database, so the acks say Redis has the totals.
	another := usagetest.Organization()
	service := usagetest.Start(s.T(), usagetest.Options{DSN: s.usageDSN, RedisURL: usagetest.RedisURL(s.T()), Logger: s.logger})
	usagetest.Empty(s.T(), service.Ledger)
	s.T().Cleanup(func() { usagetest.Forget(s.T(), usagetest.Redis(s.T()), s.organization, another) })
	var mu sync.Mutex
	var acked []Acked
	writer := s.shippingWriter(service, func(options *WriterOptions) {
		options.Shipped = func(rows []Acked) {
			mu.Lock()
			defer mu.Unlock()
			acked = append(acked, rows...)
		}
	})
	fresh := s.owned(nil)
	known := s.owned(func(request *store.Request) { request.OrganizationID, request.CostMicros = another, 40 })
	ack, err := service.Client.Ship(s.ctx, []usage.Event{usagetest.Event(another, func(event *usage.Event) {
		event.UsageID, event.AppID = known.request.UsageID, s.customer
	})})
	s.Require().NoError(err)
	s.Require().Equal(1, ack.Accepted, "the service has the second row already")

	writer.enqueue(fresh)
	writer.enqueue(known)

	s.shipped(2)
	s.Require().Eventually(func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(acked) == 2
	}, 5*time.Second, 10*time.Millisecond)
	s.ElementsMatch([]Acked{
		{UsageID: fresh.request.UsageID, OrganizationID: s.organization, CostMicros: 25, Mirrored: true},
		{UsageID: known.request.UsageID, OrganizationID: another, CostMicros: 40, Mirrored: true},
	}, acked, "a duplicate is acked as well: the service has it")
}

func (s *WriterSuite) TestWithoutAMirrorAnAckSaysRedisDoesNotHaveIt() {
	service := s.usageService()
	acked := make(chan Acked, 1)
	writer := s.shippingWriter(service, func(options *WriterOptions) {
		options.Shipped = func(rows []Acked) { acked <- rows[0] }
	})

	writer.enqueue(s.owned(nil))

	select {
	case row := <-acked:
		s.False(row.Mirrored)
	case <-time.After(5 * time.Second):
		s.Fail("no ack")
	}
}

// answering is a service that answers every batch with answer, counting the batches.
func (s *WriterSuite) answering(answer string) (*usage.Client, *atomic.Int64) {
	var batches atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		batches.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(answer))
	}))
	s.T().Cleanup(server.Close)
	client, err := usage.NewClient(usage.ClientOptions{URL: server.URL, Token: "token"})
	s.Require().NoError(err)
	return client, &batches
}

func (s *WriterSuite) TestAnAnswerThatDoesNotAccountForEveryRowMarksNone() {
	errorsBefore := testutil.ToFloat64(usageShipErrors)
	short, batches := s.answering(`{"accepted":1,"duplicates":0,"rejected":[],"mirrored":true}`)
	writer := s.newWriter(func(options *WriterOptions) { options.Usage, options.Source = short, testSource })
	release := s.hold(writer)
	writer.enqueue(s.owned(nil))
	writer.enqueue(s.owned(nil))
	release()

	s.risen(usageShipErrors, errorsBefore, 1)
	s.Equal(int64(1), batches.Load())
	for _, row := range s.written(3) {
		if row.Provider == heldProvider {
			continue
		}
		s.True(row.ToShip, "left for the sweeper to send again")
		s.Nil(row.ShippedAt)
		s.Empty(row.ShipError)
	}
	s.Contains(s.logs.String(), "does not match the usage rows sent")
}

func (s *WriterSuite) TestABatchThatWaitedForTheShipperPastTheSettleTimeIsLeftToTheSweeper() {
	deferredBefore := testutil.ToFloat64(usageShipDeferred)
	answer := make(chan struct{})
	var answered sync.Once
	var batches atomic.Int64
	hanging := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		batches.Add(1)
		select {
		case <-answer:
		case <-r.Context().Done():
		}
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	s.T().Cleanup(hanging.Close)
	slow, err := usage.NewClient(usage.ClientOptions{URL: hanging.URL, Token: "token"})
	s.Require().NoError(err)
	writer := s.newWriter(func(options *WriterOptions) {
		options.Usage, options.Source, options.Settle = slow, testSource, 100*time.Millisecond
	})
	s.T().Cleanup(func() { answered.Do(func() { close(answer) }) })
	writer.enqueue(s.owned(nil))
	s.Require().Eventually(func() bool { return batches.Load() == 1 }, 5*time.Second, time.Millisecond, "the first batch is being sent")
	waiting := s.owned(nil)
	writer.enqueue(waiting)
	s.Require().Eventually(func() bool { return writer.pending() == 0 }, 5*time.Second, time.Millisecond)

	time.Sleep(200 * time.Millisecond)
	answered.Do(func() { close(answer) })

	s.risen(usageShipDeferred, deferredBefore, 1)
	s.Never(func() bool { return batches.Load() > 1 }, 200*time.Millisecond, 10*time.Millisecond,
		"the batch that waited is the sweeper's, which may be sending it already")
	for _, row := range s.written(2) {
		s.True(row.ToShip)
	}
}

func (s *WriterSuite) TestAShippedEventCarriesTheRowsIdentityAndBillable() {
	service := s.usageService()
	writer := s.shippingWriter(service, nil)
	row := s.owned(func(request *store.Request) {
		request.APIKey, request.OrganizationTier, request.UserID = "key-1", "tier-a", "user-1"
		request.UsageCategories = []string{"team:core", "support"}
		request.SessionID, request.AgentID, request.CallID, request.Purpose = "session-1", "agent-1", "call-1", "reply"
		request.Billable, request.BaseCostMicros, request.PriceVersion = false, 20, "price-change-1"
		request.SKU = "llm-openai-gpt"
		request.Meters = pricing.Quantities{pricing.Requests: 2}
		request.Attributes = map[string]string{"service_tier": "flex"}
		request.Tags = map[string]string{"project": "p"}
		request.Success, request.ErrorCode = false, "timeout"
	})

	writer.enqueue(row)

	s.shipped(1)
	events := s.events(service)
	s.Require().Len(events, 1)
	event := events[0]
	s.Equal("key-1", event.APIKey)
	s.Equal("tier-a", event.OrganizationTier)
	s.Equal("user-1", event.UserID)
	s.Equal([]string{"support", "team:core"}, event.UsageCategories)
	s.Equal([]string{"session-1", "agent-1", "call-1", "reply"}, []string{event.SessionID, event.AgentID, event.CallID, event.Purpose})
	s.False(event.Billable)
	s.EqualValues(20, event.BaseCostMicros)
	s.Equal("price-change-1", event.PriceVersion)
	s.Equal("llm-openai-gpt", event.SKU)
	s.Equal(map[string]int64{"output_tokens": 10, "requests": 2}, event.Quantities)
	s.Equal(map[string]string{"service_tier": "flex"}, event.Attributes)
	s.Equal(map[string]string{"project": "p"}, event.Tags)
	s.False(event.Success)
	s.Equal("timeout", event.ErrorCode)
}

func (s *WriterSuite) TestWithTheServiceDownRowsWaitForTheSweeper() {
	errorsBefore := testutil.ToFloat64(usageShipErrors)
	down, err := usage.NewClient(usage.ClientOptions{URL: "http://127.0.0.1:1", Token: "token"})
	s.Require().NoError(err)
	writer := s.newWriter(func(options *WriterOptions) { options.Usage, options.Source = down, testSource })
	sent := []queued{s.owned(nil), s.owned(nil), s.owned(nil)}
	for _, row := range sent {
		s.Require().True(writer.enqueue(row))
	}

	s.risen(usageShipErrors, errorsBefore, 1)
	for _, row := range s.written(3) {
		s.True(row.ToShip, "in the shard, waiting for the sweeper")
		s.Nil(row.ShippedAt)
	}
	s.Contains(s.logs.String(), "could not ship usage rows, the sweeper sends them again")

	service := s.usageService()
	sweeper := s.sweeper(service.Client)
	s.swept(sweeper)

	s.shipped(3)
	events := s.events(service)
	s.Require().Len(events, 3, "every row shipped once")
	for i, row := range sent {
		s.Equal(row.request.UsageID, events[i].UsageID)
	}
	s.Zero(sweeper.unshipped.Load())
	s.Zero(sweeper.oldest.Load())
}

func (s *WriterSuite) TestAFullShipperLeavesTheRowsForTheSweeper() {
	deferredBefore := testutil.ToFloat64(usageShipDeferred)
	answer := make(chan struct{})
	hanging := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-answer:
		case <-r.Context().Done():
		}
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	s.T().Cleanup(hanging.Close)
	slow, err := usage.NewClient(usage.ClientOptions{URL: hanging.URL, Token: "token"})
	s.Require().NoError(err)
	writer := s.newWriter(func(options *WriterOptions) { options.Usage, options.Source = slow, testSource })
	s.T().Cleanup(func() { close(answer) })

	// One batch the shipper is sending, shipQueue waiting, and one it has no room for.
	for range shipQueue + 2 {
		s.Require().True(writer.enqueue(s.owned(nil)))
		s.Require().Eventually(func() bool { return writer.pending() == 0 }, 5*time.Second, time.Millisecond)
	}

	s.risen(usageShipDeferred, deferredBefore, 1)
	service := s.usageService()
	s.swept(s.sweeper(service.Client))
	s.shipped(shipQueue + 2)
	s.Len(s.events(service), shipQueue+2)
}

func (s *WriterSuite) TestOnlyOneSweeperWorksPerShard() {
	service := s.usageService()
	first, second := s.sweeper(service.Client), s.sweeper(service.Client)

	held, err := first.sweep(s.ctx)
	s.Require().NoError(err)
	s.True(held)
	inUse := s.store.ShardDB().Stats().InUse
	held, err = second.sweep(s.ctx)
	s.Require().NoError(err)
	s.False(held, "the shard has one sweeper at a time")
	s.Equal(inUse, s.store.ShardDB().Stats().InUse, "a router without the lock keeps no connection for it")
	held, err = first.sweep(s.ctx)
	s.Require().NoError(err)
	s.True(held, "the holder keeps the lock")

	first.lock.Release()
	held, err = second.sweep(s.ctx)
	s.Require().NoError(err)
	s.True(held, "the next tick takes the lock once the holder lets it go")
}

func (s *WriterSuite) TestAStuckHolderGivesUpAndEveryRouterStillCountsWhatWaits() {
	service := s.usageService()
	writer := s.newWriter(nil)
	stale := s.owned(func(request *store.Request) { request.StartedAt = time.Now().UTC().Add(-20 * time.Minute) })
	writer.enqueue(stale)
	s.written(1)
	// Only this test's rows: a row another test left would be shipped too, by the same sweep.
	_, err := s.store.ShardDB().ExecContext(s.ctx, "UPDATE requests SET to_ship = true WHERE customer_id = ?", s.customer)
	s.Require().NoError(err)
	// Another transaction holds the row, so marking it shipped waits, as on a database that
	// stopped answering.
	tx, err := s.store.ShardDB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = tx.Rollback() })
	_, err = tx.ExecContext(s.ctx, "SELECT 1 FROM requests WHERE usage_id = ? FOR UPDATE", stale.request.UsageID)
	s.Require().NoError(err)
	holder := s.sweeperWith(SweeperOptions{Usage: service.Client, Timeout: time.Second})
	other := s.sweeper(service.Client)

	started := time.Now()
	swept := make(chan error, 1)
	go func() {
		_, err := holder.sweep(s.ctx)
		swept <- err
	}()
	s.Require().Eventually(func() bool { return len(s.events(service)) == 1 }, 5*time.Second, 5*time.Millisecond,
		"the holder has sent the row and waits to mark it")
	held, err := other.sweep(s.ctx)
	s.Require().NoError(err)
	s.Require().False(held)
	s.EqualValues(1, other.unshipped.Load(), "a router without the lock counts what waits")
	s.GreaterOrEqual(time.Duration(other.oldest.Load()), 20*time.Minute)
	s.GreaterOrEqual(unshippedOldest(), (20 * time.Minute).Seconds(), "which is what ship_stale reads")

	s.ErrorIs(<-swept, context.DeadlineExceeded)
	s.Less(time.Since(started), 5*time.Second, "the sweep's deadline ends it")
	s.Require().NoError(tx.Rollback())
	s.swept(other)
	s.shipped(1)
	s.Len(s.events(service), 1, "sent twice, recorded once")
}

func (s *WriterSuite) TestAHolderDrainingABacklogBiggerThanOneSweepKeepsTheLock() {
	service := s.usageService()
	target, err := url.Parse(service.URL)
	s.Require().NoError(err)
	proxy := httputil.NewSingleHostReverseProxy(target)
	// Every second shipment hangs until the sweep's deadline cuts it, so each sweep marks one
	// page and runs out of time in the next.
	var shipments atomic.Int64
	slowed := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if shipments.Add(1)%2 == 0 {
			// The server sees the client hang up only once the body is read.
			_, _ = io.Copy(io.Discard, r.Body)
			<-r.Context().Done()
			return
		}
		proxy.ServeHTTP(w, r)
	}))
	s.T().Cleanup(slowed.Close)
	slow, err := usage.NewClient(usage.ClientOptions{URL: slowed.URL, Token: service.Token})
	s.Require().NoError(err)
	_, err = s.store.ShardDB().ExecContext(s.ctx, `
INSERT INTO requests (usage_id, modality, customer_id, organization_id, provider, model, started_at, success, output_tokens, cost_micros, to_ship)
SELECT gen_random_uuid(), 'llm', ?, ?, 'openai', 'gpt', now() - interval '1 hour', true, 10, 25, true
FROM generate_series(1, 3 * ?)`, s.customer, s.organization, sweepPage)
	s.Require().NoError(err)
	holder := s.sweeperWith(SweeperOptions{Usage: slow, Timeout: 2 * time.Second})
	other := s.sweeper(service.Client)

	for sweep := 1; sweep <= 3; sweep++ {
		held, err := holder.sweep(s.ctx)
		s.Require().NoError(err, "sweep %d", sweep)
		s.Require().True(held, "sweep %d", sweep)
		held, err = other.sweep(s.ctx)
		s.Require().NoError(err)
		s.Require().False(held, "the holder kept the lock after sweep %d", sweep)
	}

	for sweep := 4; holder.unshipped.Load() > 0 && sweep <= 6; sweep++ {
		held, err := holder.sweep(s.ctx)
		s.Require().NoError(err, "sweep %d", sweep)
		s.Require().True(held, "sweep %d", sweep)
	}

	s.Zero(holder.unshipped.Load(), "the backlog drains")
	s.Len(s.events(service), 3*sweepPage)
	s.Contains(s.logs.String(), "ran out of time with rows still waiting")
}

func (s *WriterSuite) TestARejectedRowIsKeptAndCounted() {
	before := testutil.ToFloat64(usageRejected)
	service := s.usageService()
	writer := s.shippingWriter(service, nil)
	ahead := s.owned(func(request *store.Request) { request.StartedAt = time.Now().UTC().Add(2 * time.Hour) })
	fine := s.owned(nil)

	writer.enqueue(fine)
	writer.enqueue(ahead)

	byID := map[uuid.UUID]store.Request{}
	for _, row := range s.shipped(2) {
		byID[row.UsageID] = row
	}
	s.Equal(usage.ReasonOccurredAt, byID[ahead.request.UsageID].ShipError)
	s.Nil(byID[ahead.request.UsageID].ShippedAt, "refused, not shipped")
	s.Empty(byID[fine.request.UsageID].ShipError)
	s.NotNil(byID[fine.request.UsageID].ShippedAt)
	s.Equal(1.0, testutil.ToFloat64(usageRejected)-before)
	s.Contains(s.logs.String(), "the usage service refused usage rows")
	s.Contains(s.logs.String(), ahead.request.UsageID.String()+" "+usage.ReasonOccurredAt)
	s.Len(s.events(service), 1)
}

func (s *WriterSuite) TestARowWithNoOrganizationWaitsForOne() {
	before := testutil.ToFloat64(usageUnowned)
	service := s.usageService()
	writer := s.shippingWriter(service, nil)
	writer.enqueue(s.row(nil))
	s.Require().True(s.written(1)[0].ToShip)
	sweeper := s.sweeper(service.Client)

	s.swept(sweeper)

	s.Equal(1.0, testutil.ToFloat64(usageUnowned)-before)
	s.True(s.rows()[0].ToShip, "waiting for its app to join an organization")
	s.Empty(s.events(service))
	s.EqualValues(1, sweeper.unshipped.Load())
	s.Contains(s.logs.String(), "app="+s.customer)

	s.Require().NoError(s.store.JoinOrganization(s.ctx, s.customer, s.organization))
	s.swept(sweeper)

	rows := s.shipped(1)
	s.Equal(s.organization, rows[0].OrganizationID, "written on the row first")
	events := s.events(service)
	s.Require().Len(events, 1)
	s.Equal(s.organization, events[0].OrganizationID)
	s.Zero(sweeper.unshipped.Load())
}

func (s *WriterSuite) TestTheSweeperPagesPastRowsThatWaitForAnOrganization() {
	service := s.usageService()
	writer := s.shippingWriter(service, func(options *WriterOptions) { options.Usage = nil })
	earlier := time.Now().UTC().Add(-time.Minute)
	for range sweepPage {
		writer.enqueue(s.row(func(request *store.Request) { request.StartedAt = earlier }))
	}
	owned := s.owned(func(request *store.Request) { request.StartedAt = earlier.Add(time.Second) })
	writer.enqueue(owned)
	s.written(sweepPage + 1)
	// Only this test's rows: a row another test left would be shipped too, by the same sweep.
	_, err := s.store.ShardDB().ExecContext(s.ctx, "UPDATE requests SET to_ship = true WHERE customer_id = ?", s.customer)
	s.Require().NoError(err)

	s.swept(s.sweeper(service.Client))

	events := s.events(service)
	s.Require().Len(events, 1, "a full page of rows without an organization does not stop the sweep")
	s.Equal(owned.request.UsageID, events[0].UsageID)
}
