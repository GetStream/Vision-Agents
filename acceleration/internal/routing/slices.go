package routing

import (
	"maps"
	"strconv"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

// SliceEvery is how long a slice runs when SliceOptions does not say: a process that dies
// loses at most a minute of what it was running.
const SliceEvery = time.Minute

// AttributeSlice is a slice row's number within its run, counted from 0, so a gap in a run's
// rows shows which slice a dying process lost.
const AttributeSlice = "slice"

// Slices records something that runs for a while as one row per slice of it, written as each
// slice ends, so a process that dies loses at most the slice it was in. Each slice's usage id
// is derived from what runs, when it started and the slice's number, so writing a slice twice
// bills it once.
//
// A slice bills the whole seconds between its boundaries, both measured from the run's start,
// so the slices add up to the run however late a tick is. The last slice rounds the total up
// instead: a run of 61 s is 60 + 1, and a slice that ends at 60.001 s still bills 60. Rounding
// every slice up would add about a second a minute.
//
// The rows are measured by the router and keep out of provider health: nothing a provider did
// is timed by them.
//
// One goroutine per Slices ticks; Stop or Cancel ends it. A nil Slices records nothing. A run
// belongs to the process that started it: a restarted process does not resume another's.
type Slices struct {
	options SliceOptions
	now     func() time.Time
	start   time.Time
	// run is the run's start, as it goes into every slice's usage id.
	run string

	mu sync.Mutex
	// model is what the current slice is billed as, from is when it started, n is its number,
	// and billed is the whole seconds the slices before it billed. attributes are what every
	// slice from here on carries.
	model      string
	from       time.Time
	n          int
	billed     int64
	stopped    bool
	attributes map[string]string

	ticker   *time.Ticker
	stop     chan struct{}
	done     chan struct{}
	stopOnce sync.Once
}

// SliceOptions is what runs, whose it is, and how it is billed.
type SliceOptions struct {
	// Recorder is where the rows go, its modality theirs. Nil records nothing.
	Recorder *Recorder
	// Config is the provider the rows are billed under, and its model what the first slice is
	// billed as, such as a session's mode. Switch changes the model.
	Config ProviderConfig
	Owner  Owner
	// Kind and ID name what runs, such as "runtime" and a session id. ID is the rows'
	// operation id.
	Kind string
	ID   string
	// Quantities turns a slice's whole seconds, billed as model, into what it used, such as a
	// sandbox's vCPU, memory and disk seconds while it runs and its disk alone once stopped.
	// Nil is the seconds as pricing.Seconds.
	Quantities func(model string, seconds int64) pricing.Quantities
	// Attributes are stored with every slice's row, such as what a sandbox reserves.
	Attributes map[string]string
	// Every is how long a slice runs. Zero is SliceEvery.
	Every time.Duration

	// now is the clock. Nil is time.Now.
	now func() time.Time
}

// StartSlices starts the run now, its first slice billed as options.Config.Model.
func StartSlices(options SliceOptions) *Slices {
	if options.Every <= 0 {
		options.Every = SliceEvery
	}
	if options.Quantities == nil {
		options.Quantities = func(_ string, seconds int64) pricing.Quantities { return pricing.Quantities{pricing.Seconds: seconds} }
	}
	now := options.now
	if now == nil {
		now = time.Now
	}
	start := now()
	s := &Slices{
		options:    options,
		now:        now,
		start:      start,
		run:        start.UTC().Format(time.RFC3339Nano),
		model:      options.Config.Model,
		from:       start,
		attributes: maps.Clone(options.Attributes),
		ticker:     time.NewTicker(options.Every),
		stop:       make(chan struct{}),
		done:       make(chan struct{}),
	}
	go s.tick()
	return s
}

func (s *Slices) tick() {
	defer close(s.done)
	for {
		select {
		case <-s.ticker.C:
			s.next()
		case <-s.stop:
			return
		}
	}
}

// next ends the current slice now and starts the next under the same model.
func (s *Slices) next() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.stopped {
		s.endLocked(s.now(), false)
	}
}

// Switch ends the current slice now and starts the next under another model, such as a
// session moving from writing to a call. The same model, or a run that has stopped, changes
// nothing.
func (s *Slices) Switch(model string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped || model == s.model {
		return
	}
	s.endLocked(s.now(), false)
	s.model = model
	s.ticker.Reset(s.options.Every)
}

// Attribute ends the current slice now, and has every slice after it carry name=value, such as
// a sandbox's time after its session closed. A run already carrying it, or one that has
// stopped, changes nothing.
func (s *Slices) Attribute(name, value string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped || s.attributes[name] == value {
		return
	}
	s.endLocked(s.now(), false)
	if s.attributes == nil {
		s.attributes = map[string]string{}
	}
	s.attributes[name] = value
	s.ticker.Reset(s.options.Every)
}

// Stop records the last, partial slice, rounding the run up to a whole second, and ends the
// run. Calling it twice records it once.
func (s *Slices) Stop() {
	s.end(true)
}

// Cancel ends the run without recording the slice it was in, for something that never got
// going, such as a session that failed to open. Slices already written stay written.
func (s *Slices) Cancel() {
	s.end(false)
}

func (s *Slices) end(record bool) {
	if s == nil {
		return
	}
	s.stopOnce.Do(func() {
		s.mu.Lock()
		if record {
			s.endLocked(s.now(), true)
		}
		s.stopped = true
		s.mu.Unlock()
		s.ticker.Stop()
		close(s.stop)
		<-s.done
	})
}

// endLocked records the current slice as ending at, and starts the next there. A slice that
// crosses a UTC midnight is recorded as one slice per day it covers, each numbered in turn:
// a row is dated by its start, and each day's rows are what is checked against that day of a
// provider's records. The caller holds the lock.
func (s *Slices) endLocked(at time.Time, last bool) {
	for midnight := nextMidnight(s.from); midnight.Before(at); midnight = nextMidnight(midnight) {
		s.sliceLocked(midnight, false)
	}
	s.sliceLocked(at, last)
}

// nextMidnight is the first UTC midnight after t.
func nextMidnight(t time.Time) time.Time {
	return t.UTC().Truncate(24 * time.Hour).Add(24 * time.Hour)
}

// sliceLocked records the slice from s.from to at, and starts the next there. The caller
// holds the lock.
func (s *Slices) sliceLocked(at time.Time, last bool) {
	elapsed := at.Sub(s.start)
	total := int64(elapsed / time.Second)
	if last && elapsed%time.Second > 0 {
		total++
	}
	seconds := max(total-s.billed, 0)
	s.record(at, seconds)
	s.billed += seconds
	s.n++
	s.from = at
}

func (s *Slices) record(at time.Time, seconds int64) {
	if s.options.Recorder == nil {
		return
	}
	n := strconv.Itoa(s.n)
	stat := Stat{
		Owner:       s.options.Owner,
		UsageID:     UsageID(s.options.Kind, s.options.ID, s.run, n),
		StartedAt:   s.from.UTC(),
		OperationID: s.options.ID,
		DurationMs:  float64(at.Sub(s.from).Microseconds()) / 1000,
		Success:     true,
		Unmeasured:  true,
	}
	stat.Use(s.options.Quantities(s.model, seconds))
	stat.MeasuredByRouter()
	for name, value := range s.attributes {
		stat.Attribute(name, value)
	}
	stat.Attribute(AttributeSlice, n)
	config := s.options.Config
	config.Model = s.model
	s.options.Recorder.Record(config, stat)
}
