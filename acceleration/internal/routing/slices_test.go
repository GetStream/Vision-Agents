//go:build integration

package routing

import (
	"context"
	"io"
	"log/slog"
	"os"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// clock is a time a test moves by hand, for slices that end exactly where the test says.
type clock struct {
	mu    sync.Mutex
	start time.Time
	at    time.Time
}

func newClock() *clock {
	start := time.Date(2026, 10, 23, 12, 0, 0, 0, time.UTC)
	return &clock{start: start, at: start}
}

func (c *clock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.at
}

// to moves the clock to d after it started.
func (c *clock) to(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.at = c.start.Add(d)
}

// SlicesSuite runs things a slice at a time through the writer into Postgres, and reads back
// the rows their slices left.
type SlicesSuite struct {
	suite.Suite
	ctx    context.Context
	store  *store.Store
	prices *pricing.List
}

func TestSlicesSuite(t *testing.T) {
	suite.Run(t, new(SlicesSuite))
}

func (s *SlicesSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "routing"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
	s.prices, err = pricing.NewList("slices", pricing.AllAtCost([]pricing.Entry{
		{Modality: string(Runtime), Provider: "stream", Model: "voice", Rates: map[pricing.Meter]pricing.Rate{
			pricing.Seconds: {USD: 0.06, Per: 60},
		}},
		{Modality: string(Runtime), Provider: "stream", Model: "text", Free: true},
	}), slog.New(slog.NewTextHandler(io.Discard, nil)))
	s.Require().NoError(err)
}

// options are slices of a run of its own, billed as model, recording through a recorder the
// test closes with rows.
func (s *SlicesSuite) options(model string) SliceOptions {
	return SliceOptions{
		Recorder: NewRecorder(Runtime, s.store, nil, s.prices, slog.New(slog.NewTextHandler(io.Discard, nil))),
		Config:   ProviderConfig{Provider: "stream", Model: model},
		Owner:    Owner{CustomerID: uuid.NewString(), Attribution: Attribution{APIKey: "key-a"}},
		Kind:     "runtime",
		ID:       "session-" + uuid.NewString(),
	}
}

// faked is options whose slices end only where the test moves the clock and calls next.
func (s *SlicesSuite) faked(model string) (SliceOptions, *clock) {
	options, at := s.options(model), newClock()
	options.Every = time.Hour
	options.now = at.now
	return options, at
}

// rows drains the recorder and returns the run's rows, oldest first.
func (s *SlicesSuite) rows(options SliceOptions) []store.Request {
	options.Recorder.Close()
	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).
		Where("customer_id = ?", options.Owner.CustomerID).Order("started_at").Scan(s.ctx))
	return rows
}

// seconds is what each row billed, in order.
func seconds(rows []store.Request) []int64 {
	billed := make([]int64, 0, len(rows))
	for _, row := range rows {
		billed = append(billed, row.Meters[pricing.Seconds])
	}
	return billed
}

func (s *SlicesSuite) TestASliceIsRecordedEachInterval() {
	options := s.options("voice")
	options.Every = 50 * time.Millisecond

	run := StartSlices(options)
	time.Sleep(175 * time.Millisecond)
	run.Stop()

	rows := s.rows(options)
	s.Require().Len(rows, 4, "three full slices and the partial one Stop ended")
	s.Equal([]int64{0, 0, 0, 1}, seconds(rows), "each slice is the whole seconds its boundaries cross; the last rounds the run up")
	for i, row := range rows {
		s.Equal(string(Runtime), row.Modality)
		s.Equal("stream", row.Provider)
		s.Equal("voice", row.Model)
		s.Equal(options.ID, row.OperationID)
		s.Equal(UsageID("runtime", options.ID, run.run, []string{"0", "1", "2", "3"}[i]), row.UsageID)
		s.True(row.Success)
		s.Equal(MeasuredByRouter, row.Attributes[AttributeMeasuredBy], "the router's own clock is the count, even at zero")
	}
}

func (s *SlicesSuite) TestSlicesSumToTheRunRoundedUp() {
	options, at := s.faked("voice")
	run := StartSlices(options)

	at.to(60*time.Second + time.Millisecond)
	run.next()
	at.to(120*time.Second + 2*time.Millisecond)
	run.next()
	at.to(121*time.Second + 500*time.Millisecond)
	run.Stop()

	rows := s.rows(options)
	s.Equal([]int64{60, 60, 2}, seconds(rows), "late ticks lose nothing, and the run is rounded up once, at its end")
	s.Equal(at.start, rows[0].StartedAt.UTC(), "a slice starts where the one before it ended")
	s.Equal(at.start.Add(60*time.Second+time.Millisecond), rows[1].StartedAt.UTC())
	s.EqualValues(60_001, *rows[0].DurationMs)
	s.Equal([]string{"0", "1", "2"}, []string{rows[0].Attributes[AttributeSlice], rows[1].Attributes[AttributeSlice], rows[2].Attributes[AttributeSlice]})
	s.EqualValues(122*1_000, rows[0].CostMicros+rows[1].CostMicros+rows[2].CostMicros, "122 seconds at six cents a minute")
	for _, row := range rows {
		s.True(row.Billable)
	}
}

func (s *SlicesSuite) TestASliceAcrossMidnightIsSplitAtIt() {
	options, at := s.faked("voice")
	before := time.Date(2026, 10, 23, 23, 59, 30, 0, time.UTC)
	at.start, at.at = before, before
	run := StartSlices(options)

	at.to(60*time.Second + 400*time.Millisecond)
	run.next()
	at.to(24*time.Hour + 90*time.Second)
	run.Stop()

	rows := s.rows(options)
	midnight := time.Date(2026, 10, 24, 0, 0, 0, 0, time.UTC)
	s.Require().Len(rows, 4, "each slice that crosses a midnight is one row per day")
	s.Equal([]int64{30, 30, 86_370, 60}, seconds(rows), "the run adds up all the same")
	s.Equal([]time.Time{before, midnight, before.Add(60*time.Second + 400*time.Millisecond), midnight.Add(24 * time.Hour)},
		[]time.Time{rows[0].StartedAt.UTC(), rows[1].StartedAt.UTC(), rows[2].StartedAt.UTC(), rows[3].StartedAt.UTC()},
		"each dated within the day it covers")
	s.EqualValues(30_000, *rows[0].DurationMs)
	ids := map[string]bool{}
	for i, row := range rows {
		s.Equal(strconv.Itoa(i), row.Attributes[AttributeSlice])
		s.Equal(UsageID("runtime", options.ID, run.run, strconv.Itoa(i)), row.UsageID, "numbered in turn, so writing it twice bills it once")
		ids[row.UsageID.String()] = true
	}
	s.Len(ids, 4)
}

func (s *SlicesSuite) TestSwitchEndsTheSliceAndChangesTheModel() {
	options, at := s.faked("text")
	run := StartSlices(options)

	at.to(30*time.Second + 700*time.Millisecond)
	run.Switch("voice")
	at.to(45 * time.Second)
	run.Switch("voice")
	at.to(61*time.Second + 200*time.Millisecond)
	run.Stop()

	rows := s.rows(options)
	s.Require().Len(rows, 2, "switching to the model already billed ends nothing")
	s.Equal("text", rows[0].Model)
	s.Equal("voice", rows[1].Model)
	s.Equal([]int64{30, 32}, seconds(rows))
}

func (s *SlicesSuite) TestAFreeModeIsRecordedAtNoCostAndNotBilled() {
	options, at := s.faked("text")
	run := StartSlices(options)

	at.to(90 * time.Second)
	run.Switch("voice")
	at.to(150 * time.Second)
	run.Stop()

	rows := s.rows(options)
	s.Require().Len(rows, 2)
	s.EqualValues(90, rows[0].Meters[pricing.Seconds], "free time is still recorded, so it can be seen")
	s.Zero(rows[0].CostMicros)
	s.False(rows[0].Billable)
	s.EqualValues(60_000, rows[1].CostMicros)
	s.True(rows[1].Billable)
}

func (s *SlicesSuite) TestStopTwiceRecordsOnce() {
	options, at := s.faked("voice")
	run := StartSlices(options)

	at.to(10 * time.Second)
	run.Stop()
	at.to(20 * time.Second)
	run.Stop()
	run.Switch("text")
	run.next()

	rows := s.rows(options)
	s.Require().Len(rows, 1, "a run that stopped records nothing more")
	s.EqualValues(10, rows[0].Meters[pricing.Seconds])
}

func (s *SlicesSuite) TestASliceWrittenTwiceIsBilledOnce() {
	options, at := s.faked("voice")
	first := StartSlices(options)
	again := StartSlices(options)

	at.to(60 * time.Second)
	first.next()
	again.next()
	at.to(70 * time.Second)
	first.Stop()
	again.Stop()

	rows := s.rows(options)
	s.Equal([]int64{60, 10}, seconds(rows), "the same slice of the same run has the same usage id")
}

func (s *SlicesSuite) TestARunThatStartsAgainIsBilledAgain() {
	options, at := s.faked("voice")
	first := StartSlices(options)
	at.to(10 * time.Second)
	first.Stop()
	at.to(20 * time.Second)
	reopened := StartSlices(options)
	at.to(25 * time.Second)
	reopened.Stop()

	s.Equal([]int64{10, 5}, seconds(s.rows(options)), "a session reopened under its id is a run of its own")
}

func (s *SlicesSuite) TestCancelRecordsNothingMore() {
	options, at := s.faked("voice")
	run := StartSlices(options)

	at.to(61 * time.Second)
	run.next()
	at.to(65 * time.Second)
	run.Cancel()
	run.Stop()

	s.Equal([]int64{61}, seconds(s.rows(options)), "the slice already written stays; the one in progress is dropped")
}

func (s *SlicesSuite) TestANilRunRecordsNothing() {
	var run *Slices
	s.NotPanics(func() {
		run.Switch("voice")
		run.Attribute("orphan", "true")
		run.Stop()
		run.Cancel()
	})
}

func (s *SlicesSuite) TestWhatASliceUsedFollowsItsModelAndAttributesFollowTheRun() {
	options, at := s.faked("running")
	options.Quantities = func(model string, seconds int64) pricing.Quantities {
		if model == "stopped" {
			return pricing.Quantities{pricing.DiskGBSeconds: 3 * seconds}
		}
		return pricing.Quantities{pricing.VCPUSeconds: seconds, pricing.DiskGBSeconds: 3 * seconds}
	}
	options.Attributes = map[string]string{"sandbox_id": "sandbox-1"}
	run := StartSlices(options)

	at.to(30 * time.Second)
	run.Switch("stopped")
	at.to(45 * time.Second)
	run.Attribute("orphan", "true")
	run.Attribute("orphan", "true")
	at.to(61 * time.Second)
	run.Stop()

	rows := s.rows(options)
	s.Require().Len(rows, 3, "marking a run that is already marked ends nothing")
	s.Equal(pricing.Quantities{pricing.VCPUSeconds: 30, pricing.DiskGBSeconds: 90}, rows[0].Meters)
	s.Equal(pricing.Quantities{pricing.DiskGBSeconds: 45}, rows[1].Meters, "stopped holds disk alone")
	s.Equal(pricing.Quantities{pricing.DiskGBSeconds: 48}, rows[2].Meters)
	for i, row := range rows {
		s.Equal("sandbox-1", row.Attributes["sandbox_id"], "every slice carries the run's attributes")
		s.Equal(i == 2, row.Attributes["orphan"] == "true", "and from the mark on, the mark")
	}
}
