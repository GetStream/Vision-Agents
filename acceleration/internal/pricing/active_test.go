package pricing

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"time"
)

// answer is one read of a Source.
type answer struct {
	entries []Entry
	next    time.Time
	err     error
}

// answers is a Source written for the test: each read takes the next answer, and the last
// is answered again once the rest are taken.
type answers struct {
	mu     sync.Mutex
	queued []answer
	reads  int
}

func (a *answers) Active(context.Context, time.Time) ([]Entry, time.Time, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	got := a.queued[min(a.reads, len(a.queued)-1)]
	a.reads++
	return got.entries, got.next, got.err
}

func (a *answers) Upcoming(context.Context, time.Time) ([]UpcomingEntry, error) { return nil, nil }

// scheduled is a Source written for the test with one change scheduled at at: before it the
// list is before, from it after.
type scheduled struct {
	at            time.Time
	before, after []Entry
}

func (s scheduled) Active(_ context.Context, at time.Time) ([]Entry, time.Time, error) {
	if at.Before(s.at) {
		return s.before, s.at, nil
	}
	return s.after, time.Time{}, nil
}

func (s scheduled) Upcoming(_ context.Context, after time.Time) ([]UpcomingEntry, error) {
	if !after.Before(s.at) {
		return nil, nil
	}
	upcoming := make([]UpcomingEntry, 0, len(s.after))
	for _, entry := range s.after {
		upcoming = append(upcoming, UpcomingEntry{Entry: entry, EffectiveAt: s.at, Reason: "scheduled"})
	}
	return upcoming, nil
}

// versioned is the entry for model m at $1 per million input tokens, written by change.
func versioned(change string) Entry {
	entry := atCost(llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1)}))[0]
	entry.Version = change
	return entry
}

// active is an Active over source that reads again every interval, until the test ends.
func (s *PricingSuite) active(source Source, interval time.Duration, requireEntries bool) *Active {
	ctx, cancel := context.WithCancel(context.Background())
	s.T().Cleanup(cancel)
	fallback, err := NewList("embedded:test", atCost(llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1)})), nil)
	s.Require().NoError(err)
	return NewActive(ctx, ActiveOptions{
		Source: source, Fallback: fallback, Interval: interval, RequireEntries: requireEntries,
		Logger: slog.New(slog.NewTextHandler(s.logs, nil)),
	})
}

func (s *PricingSuite) TestActiveFollowsTheSourceAndKeepsTheLastListWhenItFails() {
	source := &answers{queued: []answer{
		{entries: []Entry{versioned("a")}},
		{err: errors.New("the database is down")},
		{entries: []Entry{versioned("b")}},
	}}
	prices := s.active(source, time.Hour, false)
	version := func() string { return prices.Quote(llmItem("m", nil), Quantities{InputTokens: 1}).Version }

	s.Equal("a", version())
	s.Require().Error(prices.Reload(context.Background()))
	s.Equal("a", version(), "a failed read keeps the list read before")
	s.Require().NoError(prices.Reload(context.Background()))
	s.Equal("b", version())
	s.Equal(1, strings.Count(s.logs.String(), "could not read the price list"))
}

func (s *PricingSuite) TestAnUnpricedItemIsLoggedOnceAcrossReads() {
	active := s.active(&answers{queued: []answer{{entries: []Entry{versioned("c1")}}}}, time.Hour, false)
	item := Item{Modality: "runtime", Provider: "stream", Model: "voice"}

	active.Quote(item, Quantities{Seconds: 60})
	s.Require().NoError(active.Reload(context.Background()))
	s.Require().NoError(active.Reload(context.Background()))
	active.Quote(item, Quantities{Seconds: 60})

	s.Equal(1, strings.Count(s.logs.String(), "billable work has no price"),
		"a list read again within the interval does not log the item again")
}

func (s *PricingSuite) TestAScheduledChangeStartsOnTime() {
	at := time.Now().Add(200 * time.Millisecond)
	prices := s.active(scheduled{at: at, before: []Entry{versioned("a")}, after: []Entry{versioned("b")}}, time.Minute, false)
	version := func() string { return prices.Quote(llmItem("m", nil), Quantities{InputTokens: 1}).Version }
	s.Equal("a", version())
	s.Require().Len(prices.Upcoming(), 1)
	s.Equal(at, prices.Upcoming()[0].EffectiveAt)

	for version() != "b" && time.Since(at) < time.Second {
		time.Sleep(time.Millisecond)
	}

	s.Equal("b", version())
	s.Less(time.Since(at), 50*time.Millisecond, "an interval of a minute did not hold the change up")
	s.Empty(prices.Upcoming())
}

func (s *PricingSuite) TestAnEmptyDatabaseKeepsTheFallback() {
	prices := s.active(&answers{queued: []answer{{}}}, time.Hour, false)

	s.Equal("embedded:test", prices.Quote(llmItem("m", nil), Quantities{InputTokens: 1}).Version)
	s.Require().NoError(prices.Reload(context.Background()))
	s.Equal(1, strings.Count(s.logs.String(), "level=ERROR"), "logged once a minute, not on every read")
	s.Contains(s.logs.String(), "the price list is empty")
	s.NoError(prices.Ready(), "a read that answered nothing still succeeded")
}

func (s *PricingSuite) TestAListThatMustHaveEntriesIsNotReadyUntilItHasThem() {
	source := &answers{queued: []answer{{}, {entries: []Entry{versioned("seed")}}}}
	prices := s.active(source, time.Hour, true)

	s.Error(prices.Ready())
	s.Require().NoError(prices.Reload(context.Background()))
	s.NoError(prices.Ready())
}

func (s *PricingSuite) TestAListNeverReadIsNotReady() {
	prices := s.active(&answers{queued: []answer{{err: errors.New("the database is down")}}}, time.Hour, false)

	s.ErrorContains(prices.Ready(), "not been read")
	s.Equal("embedded:test", prices.Quote(llmItem("m", nil), Quantities{InputTokens: 1}).Version)
}

func (s *PricingSuite) TestTheListReadChargesItsPrices() {
	entry := versioned("a")
	entry.Prices = map[Meter]Rate{InputTokens: perMillion(1.25)}
	prices := s.active(&answers{queued: []answer{{entries: []Entry{entry}}}}, time.Hour, false)

	quote := prices.Quote(llmItem("m", nil), Quantities{InputTokens: 1_000_000})

	s.Equal(int64(1_000_000), quote.BaseCostMicros)
	s.Equal(int64(1_250_000), quote.CostMicros)
	read, ok := prices.Entry(llmItem("m", nil))
	s.Require().True(ok)
	s.Equal(1.25, read.Prices[InputTokens].USD)
	s.Equal(1.0, read.Rates[InputTokens].USD)
}

// TestThePriceIsSummedOnceBeforeRounding: each meter's fraction of a micro adds up before the
// row is rounded, for the price and the vendor cost alike.
func (s *PricingSuite) TestThePriceIsSummedOnceBeforeRounding() {
	entry := llmEntry("m", map[Meter]Rate{InputTokens: perMillion(0.25), OutputTokens: perMillion(0.25)})
	entry.Prices = map[Meter]Rate{InputTokens: perMillion(0.3), OutputTokens: perMillion(0.3)}
	list, err := NewList("test", []Entry{entry}, nil)
	s.Require().NoError(err)

	quote := list.Quote(llmItem("m", nil), Quantities{InputTokens: 5, OutputTokens: 5})

	s.Equal(int64(3), quote.BaseCostMicros, "2.5 µ$ rounds to 3")
	s.Equal(int64(3), quote.CostMicros, "1.5 + 1.5 µ$, not 2 + 2")
	s.Equal(int64(2), quote.OutputCostMicros)
}

func (s *PricingSuite) TestFreeAndLocalStayFree() {
	free := Entry{Modality: "phone", Provider: "telnyx", Model: "call", Free: true}
	local := Entry{Modality: "llm", Provider: "gemma", Model: "gemma", Local: true}
	list, err := NewList("test", []Entry{free, local}, nil)
	s.Require().NoError(err)

	for _, item := range []Item{{Modality: "phone", Provider: "telnyx", Model: "call"}, {Modality: "llm", Provider: "gemma", Model: "gemma"}} {
		quote := list.Quote(item, Quantities{CallSeconds: 60, InputTokens: 1_000})
		s.Zero(quote.CostMicros)
		s.Zero(quote.BaseCostMicros)
		s.False(quote.Billable)
	}
}

func (s *PricingSuite) TestAPriceWorkedOutFromARateReadsAsWritten() {
	vendorRate := 1.10
	s.NotEqual(1.32, vendorRate*12_000/10_000, "multiplied as floats, the price has noise to drop")

	priced := Priced(llmEntry("m", map[Meter]Rate{CachedInputTokens: perMillion(1.10)}))
	encoded, err := json.Marshal(priced.Prices[CachedInputTokens])

	s.Require().NoError(err)
	s.JSONEq(`{"usd": 1.32, "per": 1000000}`, string(encoded))
	s.Contains(string(encoded), `"usd":1.32,`)
	s.Equal(1.10, priced.Rates[CachedInputTokens].USD, "the vendor's rate is kept as it is")
}

// change is one scheduled change of a timeline: the version m is priced at from at on.
type change struct {
	at      time.Time
	version string
}

// timeline is a Source written for the test: m priced at base, then at each change from its
// time on. Its changes can be replaced while the list runs, as a cancel or a new change does.
type timeline struct {
	mu      sync.Mutex
	base    string
	changes []change
	reads   int
}

func (t *timeline) set(changes ...change) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.changes = changes
}

func (t *timeline) Active(_ context.Context, at time.Time) ([]Entry, time.Time, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.reads++
	version, next := t.base, time.Time{}
	for _, scheduled := range t.changes {
		if !scheduled.at.After(at) {
			version = scheduled.version
		} else if next.IsZero() || scheduled.at.Before(next) {
			next = scheduled.at
		}
	}
	return []Entry{versioned(version)}, next, nil
}

func (t *timeline) Upcoming(context.Context, time.Time) ([]UpcomingEntry, error) { return nil, nil }

func (t *timeline) readCount() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.reads
}

// versionOf is the version m is priced at now.
func versionOf(prices *Active) string {
	return prices.Quote(llmItem("m", nil), Quantities{InputTokens: 1}).Version
}

// awaitVersion waits for m to be priced at version, and says how long after at it was.
func (s *PricingSuite) awaitVersion(prices *Active, version string, at time.Time) time.Duration {
	for versionOf(prices) != version && time.Since(at) < time.Second {
		time.Sleep(time.Millisecond)
	}
	s.Require().Equal(version, versionOf(prices))
	return time.Since(at)
}

func (s *PricingSuite) TestChainedScheduledChangesEachStartOnTime() {
	first, second := time.Now().Add(150*time.Millisecond), time.Now().Add(300*time.Millisecond)
	prices := s.active(&timeline{base: "a", changes: []change{{first, "b"}, {second, "c"}}}, time.Minute, false)
	s.Equal("a", versionOf(prices))

	s.Less(s.awaitVersion(prices, "b", first), 50*time.Millisecond)
	s.Less(s.awaitVersion(prices, "c", second), 50*time.Millisecond, "the read at the first change armed the second")
}

func (s *PricingSuite) TestACancelledChangeIsNotReadAndTheNextOneIs() {
	source := &timeline{base: "a"}
	cancelled := time.Now().Add(100 * time.Millisecond)
	source.set(change{cancelled, "b"})
	prices := s.active(source, time.Minute, false)

	source.set()
	s.Require().NoError(prices.Reload(context.Background()))
	reads := source.readCount()
	time.Sleep(time.Until(cancelled) + 50*time.Millisecond)
	s.Equal("a", versionOf(prices))
	s.Equal(reads, source.readCount(), "the cancelled change's timer was disarmed")

	scheduled := time.Now().Add(100 * time.Millisecond)
	source.set(change{scheduled, "c"})
	s.Require().NoError(prices.Reload(context.Background()))
	s.Less(s.awaitVersion(prices, "c", scheduled), 50*time.Millisecond, "a change scheduled after the cancel is armed")
}

func (s *PricingSuite) TestATimerThatFiresWhileTheClockIsBehindWaitsOutTheRest() {
	source := &timeline{base: "a"}
	due := time.Now().Add(time.Hour).Round(0)
	source.set(change{due, "b"})
	prices := s.active(source, time.Hour, false)
	reads := source.readCount()

	prices.mu.Lock()
	prices.clock = func() time.Time { return due.Add(-time.Minute) }
	prices.mu.Unlock()
	prices.fire(due)

	s.Equal(reads, source.readCount(), "the list is not read before the change is due")
	s.Equal("a", versionOf(prices))
	prices.mu.Lock()
	defer prices.mu.Unlock()
	s.True(prices.timerAt.Equal(due))
	s.Require().NotNil(prices.timer, "armed again for the rest of the wait")
	s.True(prices.timer.Stop(), "the wait is still running")
}
