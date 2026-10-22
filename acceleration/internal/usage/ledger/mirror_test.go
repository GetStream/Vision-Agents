//go:build integration

package ledger_test

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/url"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

// MirrorSuite records events with the mirror on, against a real Redis, reached through a
// relay a test can cut to see Redis go away and come back.
type MirrorSuite struct {
	suite.Suite
	ctx    context.Context
	cancel context.CancelFunc
	done   chan struct{}
	dsn    string
	relay  *relay
	redis  rueidis.Client
	mirror *ledger.Mirror
	ledger *ledger.Ledger
	// organization is this test's own, so its keys are too.
	organization string
}

func TestMirrorSuite(t *testing.T) {
	suite.Run(t, new(MirrorSuite))
}

func (s *MirrorSuite) SetupSuite() {
	s.dsn = usagetest.Database(s.T(), "usage_mirror")
	s.redis = usagetest.Redis(s.T())
	target, err := url.Parse(usagetest.RedisURL(s.T()))
	s.Require().NoError(err)
	s.relay = startRelay(s.T(), target.Host)
}

func (s *MirrorSuite) SetupTest() {
	s.ctx, s.cancel = context.WithCancel(context.Background())
	s.organization = usagetest.Organization()
	s.relay.open(s.T())
	mirror, err := ledger.NewMirror(ledger.MirrorOptions{
		URL:        "redis://" + s.relay.address + "/" + strconv.Itoa(usagetest.RedisDB),
		Timeout:    500 * time.Millisecond,
		RetryEvery: 50 * time.Millisecond,
	})
	s.Require().NoError(err)
	s.mirror = mirror
	s.done = make(chan struct{})
	go func() {
		defer close(s.done)
		mirror.Run(s.ctx)
	}()
	s.ledger, err = ledger.Open(ledger.Options{DSN: s.dsn, Mirror: mirror})
	s.Require().NoError(err)
}

func (s *MirrorSuite) TearDownTest() {
	s.cancel()
	<-s.done
	s.mirror.Close()
	s.Require().NoError(s.ledger.Close())
	s.relay.open(s.T())
	usagetest.Forget(s.T(), s.redis, s.organization)
}

// fields is a subject's hash, field by field.
func (s *MirrorSuite) fields(ref usage.SubjectRef) map[string]string {
	fields, err := s.redis.Do(context.Background(), s.redis.B().Hgetall().Key(ref.Key(ledger.KeyPrefix)).Build()).AsStrMap()
	s.Require().NoError(err)
	return fields
}

// current is the four fields of the windows holding at, as admission reads them.
func current(at time.Time) []string {
	fields := make([]string, 0, len(usage.Intervals))
	for _, interval := range usage.Intervals {
		start, _ := usage.Window(interval, at)
		fields = append(fields, usage.Field(interval, start))
	}
	return fields
}

func (s *MirrorSuite) hmget(ref usage.SubjectRef, fields []string) []string {
	values, err := s.redis.Do(context.Background(), s.redis.B().Hmget().Key(ref.Key(ledger.KeyPrefix)).Field(fields...).Build()).ToArray()
	s.Require().NoError(err)
	read := make([]string, 0, len(values))
	for _, value := range values {
		text, err := value.ToString()
		if rueidis.IsRedisNil(err) {
			text = "<missing>"
		} else {
			s.Require().NoError(err)
		}
		read = append(read, text)
	}
	return read
}

// window is a window total for the test's organization, as the mirror is given it.
func (s *MirrorSuite) window(interval usage.Interval, at time.Time, micros int64) ledger.UsageWindow {
	start, end := usage.Window(interval, at)
	return ledger.UsageWindow{
		OrganizationID: s.organization, Subject: usage.SubjectOrganization, Interval: interval,
		WindowStart: start, WindowEnd: end, SpentMicros: micros,
	}
}

func (s *MirrorSuite) TestAnAckedEventIsInRedis() {
	event := usagetest.Event(s.organization, func(e *usage.Event) {
		e.UserID, e.UsageCategories, e.CostMicros = "u1", []string{"team:finance"}, 2500
	})
	_, err := s.ledger.Ingest(s.ctx, []usage.Event{event})
	s.Require().NoError(err)

	_, err = s.ledger.Ingest(s.ctx, []usage.Event{usagetest.Event(s.organization, func(e *usage.Event) { e.CostMicros = 500 })})
	s.Require().NoError(err)

	fields := current(event.OccurredAt)
	for _, ref := range usage.Subjects(event) {
		want := []string{"2500", "2500", "2500", "2500"}
		if ref.Subject == usage.SubjectOrganization || ref.Subject == usage.SubjectApp || ref.Subject == usage.SubjectAPIKey {
			want = []string{"3000", "3000", "3000", "3000"}
		}
		s.Equal(want, s.hmget(ref, fields), "%+v", ref)
	}
}

func (s *MirrorSuite) TestAnOlderTotalNeverLowersTheMirror() {
	now := time.Now()
	s.mirror.Write(s.ctx, []ledger.UsageWindow{s.window(usage.Daily, now, 150)})
	s.mirror.Write(s.ctx, []ledger.UsageWindow{s.window(usage.Daily, now, 100)})

	organization := usage.SubjectRef{OrganizationID: s.organization, Subject: usage.SubjectOrganization}
	start, _ := usage.Window(usage.Daily, now)
	s.Equal(map[string]string{usage.Field(usage.Daily, start): "150"}, s.fields(organization))
}

func (s *MirrorSuite) TestATotalPastWhatADoubleHoldsIsComparedExactly() {
	now := time.Now()
	s.mirror.Write(s.ctx, []ledger.UsageWindow{s.window(usage.Daily, now, 9007199254740993)})
	s.mirror.Write(s.ctx, []ledger.UsageWindow{s.window(usage.Daily, now, 9007199254740992)})

	start, _ := usage.Window(usage.Daily, now)
	s.Equal("9007199254740993", s.fields(usage.SubjectRef{OrganizationID: s.organization, Subject: usage.SubjectOrganization})[usage.Field(usage.Daily, start)])
}

func (s *MirrorSuite) TestAMissingFieldIsWrittenAtZero() {
	now := time.Now()
	s.mirror.Write(s.ctx, []ledger.UsageWindow{s.window(usage.Weekly, now, 0)})

	start, _ := usage.Window(usage.Weekly, now)
	s.Equal(map[string]string{usage.Field(usage.Weekly, start): "0"}, s.fields(usage.SubjectRef{OrganizationID: s.organization, Subject: usage.SubjectOrganization}))
}

func (s *MirrorSuite) TestOldWindowFieldsAreDropped() {
	day := time.Date(2026, 10, 20, 12, 0, 0, 0, time.UTC)
	s.mirror.Write(s.ctx, []ledger.UsageWindow{s.window(usage.Daily, day, 1), s.window(usage.Monthly, day.AddDate(0, -2, 0), 7)})
	s.mirror.Write(s.ctx, []ledger.UsageWindow{s.window(usage.Daily, day.AddDate(0, 0, 1), 2)})
	s.mirror.Write(s.ctx, []ledger.UsageWindow{s.window(usage.Daily, day.AddDate(0, 0, 2), 3)})

	s.Equal(map[string]string{
		"daily:20261021":   "2",
		"daily:20261022":   "3",
		"monthly:20260801": "7",
	}, s.fields(usage.SubjectRef{OrganizationID: s.organization, Subject: usage.SubjectOrganization}),
		"the day before yesterday is gone, yesterday and other intervals stay")
}

func (s *MirrorSuite) TestTheKeyExpiresFortyDaysAfterItsLongestWindowEnds() {
	now := time.Now()
	s.mirror.Write(s.ctx, []ledger.UsageWindow{s.window(usage.Daily, now, 1), s.window(usage.Monthly, now, 1)})
	s.mirror.Write(s.ctx, []ledger.UsageWindow{s.window(usage.Daily, now, 2)})

	expires, err := s.redis.Do(context.Background(), s.redis.B().Expiretime().Key(
		usage.SubjectRef{OrganizationID: s.organization, Subject: usage.SubjectOrganization}.Key(ledger.KeyPrefix)).Build()).AsInt64()
	s.Require().NoError(err)
	_, monthEnd := usage.Window(usage.Monthly, now)
	s.Equal(monthEnd.Add(ledger.Retention).Unix(), expires)
}

func (s *MirrorSuite) TestAMirrorFailureDoesNotFailTheBatch() {
	first, err := s.ledger.Ingest(s.ctx, []usage.Event{usagetest.Event(s.organization)})
	s.Require().NoError(err)
	s.True(first.Ack().Mirrored)
	failures := testutil.ToFloat64(ledger.MirrorFailures)
	s.relay.cut()
	event := usagetest.Event(s.organization)

	result, err := s.ledger.Ingest(s.ctx, []usage.Event{event})

	s.Require().NoError(err)
	s.Equal(1, result.Ack().Accepted)
	s.False(result.Ack().Mirrored, "the producer is told Redis does not have it yet")
	s.Greater(testutil.ToFloat64(ledger.MirrorFailures), failures)
	s.Equal(1, s.mirror.Pending())
	organization := usage.SubjectRef{OrganizationID: s.organization, Subject: usage.SubjectOrganization}
	s.Equal([]string{"1000", "1000", "1000", "1000"}, s.hmget(organization, current(event.OccurredAt)), "Redis still has the first event alone")

	s.relay.open(s.T())
	s.Eventually(func() bool { return s.mirror.Pending() == 0 }, 10*time.Second, 50*time.Millisecond)
	s.Equal([]string{"2000", "2000", "2000", "2000"}, s.hmget(organization, current(event.OccurredAt)))
}

func (s *MirrorSuite) TestARedisThatIsDownAtStartDelaysOnlyTheMirror() {
	s.relay.cut()
	event := usagetest.Event(s.organization)

	result, err := s.ledger.Ingest(s.ctx, []usage.Event{event})

	s.Require().NoError(err)
	s.Equal(1, result.Ack().Accepted)
	s.relay.open(s.T())
	s.Eventually(func() bool { return s.mirror.Pending() == 0 }, 10*time.Second, 50*time.Millisecond)
	s.Equal([]string{"1000", "1000", "1000", "1000"}, s.hmget(usage.SubjectRef{OrganizationID: s.organization, Subject: usage.SubjectOrganization}, current(event.OccurredAt)))
}

func (s *MirrorSuite) TestAFailedWriteIsGivenUpAfterAMinuteAndTheNextWriteRepairsIt() {
	clock := &fakeClock{now: time.Now()}
	mirror, err := ledger.NewMirror(ledger.MirrorOptions{
		URL: "redis://" + s.relay.address + "/" + strconv.Itoa(usagetest.RedisDB), Timeout: 200 * time.Millisecond, Now: clock.read,
	})
	s.Require().NoError(err)
	defer mirror.Close()
	s.relay.cut()
	now := time.Now()
	s.False(mirror.Write(s.ctx, []ledger.UsageWindow{s.window(usage.Daily, now, 40)}))
	s.Equal(1, mirror.Pending())
	givenUp := testutil.ToFloat64(ledger.MirrorGivenUp)

	clock.add(time.Minute)
	ctx, cancel := context.WithTimeout(s.ctx, 3*time.Second)
	defer cancel()
	go mirror.Run(ctx)
	s.Eventually(func() bool { return mirror.Pending() == 0 }, 3*time.Second, 20*time.Millisecond)
	s.Equal(givenUp+1, testutil.ToFloat64(ledger.MirrorGivenUp))

	s.relay.open(s.T())
	mirror.Write(s.ctx, []ledger.UsageWindow{s.window(usage.Daily, now, 41)})
	start, _ := usage.Window(usage.Daily, now)
	s.Equal(map[string]string{usage.Field(usage.Daily, start): "41"}, s.fields(usage.SubjectRef{OrganizationID: s.organization, Subject: usage.SubjectOrganization}))
}

func (s *MirrorSuite) TestEachTotalIsGivenUpOnItsOwnAndTheRestAreLoggedAtShutdown() {
	clock := &fakeClock{now: time.Now()}
	logs := &lockedBuffer{}
	mirror, err := ledger.NewMirror(ledger.MirrorOptions{
		URL: "redis://" + s.relay.address + "/" + strconv.Itoa(usagetest.RedisDB), Timeout: 200 * time.Millisecond,
		RetryEvery: 20 * time.Millisecond, Now: clock.read, Logger: slog.New(slog.NewTextHandler(logs, nil)),
	})
	s.Require().NoError(err)
	defer mirror.Close()
	s.relay.cut()
	now := time.Now()
	mirror.Write(s.ctx, []ledger.UsageWindow{s.window(usage.Daily, now, 40)})
	clock.add(40 * time.Second)
	mirror.Write(s.ctx, []ledger.UsageWindow{s.window(usage.Weekly, now, 40)})
	givenUp := testutil.ToFloat64(ledger.MirrorGivenUp)

	clock.add(25 * time.Second)
	ctx, cancel := context.WithCancel(s.ctx)
	done := make(chan struct{})
	go func() {
		defer close(done)
		mirror.Run(ctx)
	}()
	s.Eventually(func() bool { return testutil.ToFloat64(ledger.MirrorGivenUp) == givenUp+1 }, 3*time.Second, 10*time.Millisecond,
		"the daily total, waiting 65 s")
	s.Equal(1, mirror.Pending(), "the weekly one, waiting 25 s, stays")
	cancel()
	<-done

	s.Zero(mirror.Pending())
	s.Equal(givenUp+2, testutil.ToFloat64(ledger.MirrorGivenUp))
	s.Contains(logs.String(), "stopping with window totals not mirrored to redis")
	s.Contains(logs.String(), "organization="+s.organization+" counters=1")
}

func (s *MirrorSuite) TestAWriteWhileAnotherConnectsFailsAtOnce() {
	silent, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	defer silent.Close() //nolint:errcheck // the test is done with it
	accepted := make(chan net.Conn, 4)
	go func() {
		for {
			conn, err := silent.Accept()
			if err != nil {
				return
			}
			accepted <- conn
		}
	}()
	mirror, err := ledger.NewMirror(ledger.MirrorOptions{URL: "redis://" + silent.Addr().String(), Timeout: 2 * time.Second})
	s.Require().NoError(err)
	defer mirror.Close()
	now := time.Now()
	first := make(chan bool, 1)
	go func() { first <- mirror.Write(s.ctx, []ledger.UsageWindow{s.window(usage.Daily, now, 1)}) }()
	conn := <-accepted
	defer conn.Close() //nolint:errcheck // the test is done with it

	started := time.Now()
	s.False(mirror.Write(s.ctx, []ledger.UsageWindow{s.window(usage.Weekly, now, 1)}))

	s.Less(time.Since(started), 500*time.Millisecond, "it did not wait on the other's connect")
	select {
	case <-first:
		s.Fail("the first write was still connecting")
	default:
	}
	s.False(<-first)
	s.Equal(1, mirror.Pending())
}

func (s *MirrorSuite) TestAnOrganizationWithMoreTotalsThanOneScriptCallIsWrittenWhole() {
	now := time.Now()
	start, end := usage.Window(usage.Daily, now)
	windows := make([]ledger.UsageWindow, 0, 2500)
	for i := range 2500 {
		windows = append(windows, ledger.UsageWindow{
			OrganizationID: s.organization, Subject: usage.SubjectApp, AppID: strconv.Itoa(i),
			Interval: usage.Daily, WindowStart: start, WindowEnd: end, SpentMicros: int64(i),
		})
	}

	s.True(s.mirror.Write(s.ctx, windows))

	commands := make(rueidis.Commands, 0, len(windows))
	for _, window := range windows {
		commands = append(commands, s.redis.B().Hget().Key(window.Ref().Key(ledger.KeyPrefix)).Field(usage.Field(usage.Daily, start)).Build())
	}
	for i, result := range s.redis.DoMulti(context.Background(), commands...) {
		total, err := result.ToString()
		s.Require().NoError(err, i)
		s.Equal(strconv.Itoa(i), total)
	}
}

func (s *MirrorSuite) TestABatchSentAgainAfterRedisLostItsTotalsRepairsThem() {
	event := usagetest.Event(s.organization, func(e *usage.Event) { e.UserID = "u1" })
	_, err := s.ledger.Ingest(s.ctx, []usage.Event{event})
	s.Require().NoError(err)
	// What a crash between the commit and the mirror write leaves.
	usagetest.Forget(s.T(), s.redis, s.organization)

	again, err := s.ledger.Ingest(s.ctx, []usage.Event{event})

	s.Require().NoError(err)
	s.Equal(usage.Ack{Accepted: 0, Duplicates: 1, Rejected: []usage.Rejection{}, Mirrored: true}, again.Ack())
	for _, ref := range usage.Subjects(event) {
		s.Equal([]string{"1000", "1000", "1000", "1000"}, s.hmget(ref, current(event.OccurredAt)), "%+v", ref)
	}
}

func (s *MirrorSuite) TestABatchSentAgainReadsTheWindowsOfItsDuplicates() {
	event := usagetest.Event(s.organization, func(e *usage.Event) { e.UserID = "u1" })
	_, err := s.ledger.Ingest(s.ctx, []usage.Event{event})
	s.Require().NoError(err)
	other := usagetest.Organization()
	defer usagetest.Forget(s.T(), s.redis, other)

	again, err := s.ledger.Ingest(s.ctx, []usage.Event{event, usagetest.Event(other)})

	s.Require().NoError(err)
	s.Equal(1, again.Duplicates)
	s.True(again.Mirrored)
	s.Len(again.Repaired, 16, "the four subjects of the event recorded before, in four intervals each")
	for _, window := range again.Repaired {
		s.Equal(s.organization, window.OrganizationID)
		s.Equal(int64(1000), window.SpentMicros)
	}
	s.Len(again.Windows, 12, "the new event's alone")
}

func (s *MirrorSuite) TestABatchOfFiveHundredDuplicatesIsSentAgainQuickly() {
	// Each event has a key, a user and six categories of its own: 7,002 subjects, 28,008
	// windows. Reading them by an 8-column IN list took over 5 s; joined, well under 1 s.
	events := make([]usage.Event, 500)
	for i := range events {
		events[i] = usagetest.Event(s.organization, func(e *usage.Event) {
			e.APIKey, e.UserID = fmt.Sprintf("k%d", i), fmt.Sprintf("u%d", i)
			for c := range 6 {
				e.UsageCategories = append(e.UsageCategories, fmt.Sprintf("c%d-%d", i, c))
			}
		})
	}
	first, err := s.ledger.Ingest(s.ctx, events)
	s.Require().NoError(err)
	s.Require().Equal(28008, len(first.Windows))
	usagetest.Forget(s.T(), s.redis, s.organization)

	started := time.Now()
	again, err := s.ledger.Ingest(s.ctx, events)
	took := time.Since(started)

	s.Require().NoError(err)
	s.Equal(500, again.Duplicates)
	s.Equal(28008, len(again.Repaired))
	s.True(again.Mirrored)
	s.Less(took, 2*time.Second)
	s.T().Logf("all-duplicate resend of 500 events, 28,008 windows: %s", took)
	last := events[499]
	user := usage.SubjectRef{OrganizationID: s.organization, Subject: usage.SubjectUser, AppID: last.AppID, UserID: last.UserID}
	s.Equal([]string{"1000", "1000", "1000", "1000"}, s.hmget(user, current(last.OccurredAt)), "Redis has them again")
}

func (s *MirrorSuite) TestCurrentRewritesAKeyRedisLost() {
	event := usagetest.Event(s.organization)
	_, err := s.ledger.Ingest(s.ctx, []usage.Event{event})
	s.Require().NoError(err)
	organization := usage.SubjectRef{OrganizationID: s.organization, Subject: usage.SubjectOrganization}
	usagetest.Forget(s.T(), s.redis, s.organization)

	_, err = s.ledger.Current(s.ctx, []usage.SubjectRef{organization})

	s.Require().NoError(err)
	s.Equal([]string{"1000", "1000", "1000", "1000"}, s.hmget(organization, current(time.Now())))
}

type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *fakeClock) read() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) add(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// relay forwards a local port to Redis. Cutting it closes the port and every connection
// through it, which is what a Redis that went away looks like to the mirror.
type relay struct {
	target  string
	address string

	mu       sync.Mutex
	listener net.Listener
	conns    []net.Conn
}

func startRelay(t *testing.T, target string) *relay {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	r := &relay{target: target, address: listener.Addr().String()}
	r.serve(listener)
	t.Cleanup(r.cut)
	return r
}

func (r *relay) open(t *testing.T) {
	r.mu.Lock()
	open := r.listener != nil
	r.mu.Unlock()
	if open {
		return
	}
	listener, err := net.Listen("tcp", r.address)
	if err != nil {
		t.Fatal(err)
	}
	r.serve(listener)
}

func (r *relay) serve(listener net.Listener) {
	r.mu.Lock()
	r.listener = listener
	r.mu.Unlock()
	go func() {
		for {
			client, err := listener.Accept()
			if err != nil {
				return
			}
			upstream, err := net.Dial("tcp", r.target)
			if err != nil {
				_ = client.Close()
				continue
			}
			r.mu.Lock()
			r.conns = append(r.conns, client, upstream)
			r.mu.Unlock()
			go pipe(client, upstream)
			go pipe(upstream, client)
		}
	}()
}

func pipe(to, from net.Conn) {
	_, _ = io.Copy(to, from)
	_ = to.Close()
	_ = from.Close()
}

func (r *relay) cut() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.listener != nil {
		_ = r.listener.Close()
		r.listener = nil
	}
	for _, conn := range r.conns {
		_ = conn.Close()
	}
	r.conns = nil
}
