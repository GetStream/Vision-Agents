package audiofloor

import (
	"errors"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// EndOfTurn is an agent's choice of how eager turn-taking is, which the server's policy turns
// into a way of ending a turn. It names at most one of these, and naming none takes the
// server's default preset:
//
//   - MaxWaitMs, the most a caller who has finished may wait for the answer: the point of the
//     server's measured frontier with the fewest cut-ins whose measured p90 wait is within it;
//   - Preset, one of the server's presets by name, each naming a point;
//   - Bars of the agent's own, a turn ending at the first check that reaches its bar.
//
// A point brings its bars and how many checks in a row must reach them. When a pause is
// checked, how much more a turn's first pause asks and the pause that ends a turn whatever the
// model says stay the server's.
type EndOfTurn struct {
	MaxWaitMs int
	Preset    string
	Bars      []PolicyBar
}

// FrontierPoint is bars fitted to have a caller wait at most MaxWaitMs for the answer, the
// checks in a row that must reach them, and what they did on held-out calls.
type FrontierPoint struct {
	MaxWaitMs    int         `json:"max_wait_ms"`
	Bars         []PolicyBar `json:"bars"`
	ChecksInARow int         `json:"checks_in_a_row"`
	Measured     Measured    `json:"measured"`
}

// Measured is what a frontier point did on held-out calls: the share of mid-turn pauses it
// ended a turn in, how long a caller who had finished waited for the answer, the share of
// real pauses inside a dictated number, ID or spelled name it ended a turn in, how many of
// SlowCounts counts read one number a second it ended before their last number, and whether
// it passes the dictation guard.
type Measured struct {
	CutIns             float64 `json:"cut_ins"`
	AnswerMsMedian     int     `json:"answer_ms_median"`
	AnswerMsP90        int     `json:"answer_ms_p90"`
	DictationPausesCut float64 `json:"dictation_pauses_cut"`
	SlowCountsCut      int     `json:"slow_counts_cut"`
	SlowCounts         int     `json:"slow_counts"`
	DictationGuard     bool    `json:"dictation_guard"`
}

// Ending is how a session's turns end: its bars, the checks in a row that must reach them,
// and the measured point it is, nil for an agent's own bars. Name is what its log and metrics
// call it, one of a bounded few: preset:NAME, max_wait_ms:N for the point fitted to N (never
// the wait an agent asked for), or custom for an agent's own bars.
type Ending struct {
	Name         string
	Bars         []PolicyBar
	ChecksInARow int
	Point        *FrontierPoint
}

// Custom names an agent's own bars in an Ending.
const Custom = "custom"

// Guard is what the log and metrics say of an Ending's dictation guard: pass or fail for a
// measured point, and unmeasured for an agent's own bars.
func (e Ending) Guard() string {
	switch {
	case e.Point == nil:
		return "unmeasured"
	case e.Point.Measured.DictationGuard:
		return "pass"
	}
	return "fail"
}

// Check says what is wrong with the choice itself, before any server is asked: more than one
// way of choosing named, a wait that is not positive, or bars that do not begin at 0 ms and
// then each after the one before, each a probability strictly between 0 and 1.
func (e EndOfTurn) Check() error {
	var named []string
	if e.MaxWaitMs != 0 {
		named = append(named, "max_wait_ms")
	}
	if e.Preset != "" {
		named = append(named, "preset")
	}
	if e.Bars != nil {
		named = append(named, "bars")
	}
	if len(named) > 1 {
		return stack.Wrap(fmt.Errorf("end_of_turn names %s: it takes one of max_wait_ms, preset and bars", strings.Join(named, " and ")))
	}
	if e.MaxWaitMs < 0 {
		return stack.Wrap(fmt.Errorf("end_of_turn.max_wait_ms is %d: a wait is a number of milliseconds above 0", e.MaxWaitMs))
	}
	if e.Bars != nil {
		if err := checkBars(e.Bars); err != nil {
			return stack.Wrap(fmt.Errorf("end_of_turn.%w", err))
		}
	}
	return nil
}

// checkBars says what is wrong with bars, if anything, naming the bar.
func checkBars(bars []PolicyBar) error {
	if len(bars) == 0 {
		return errors.New("bars is empty: name at least the bar from 0 ms")
	}
	if bars[0].FromMs != 0 {
		return fmt.Errorf("bars[0].from_ms is %d: the first bar begins at 0 ms into the pause", bars[0].FromMs)
	}
	for i, b := range bars {
		if !(b.Bar > 0 && b.Bar < 1) {
			return fmt.Errorf("bars[%d].bar is %s: a bar is a probability strictly between 0 and 1",
				i, strconv.FormatFloat(b.Bar, 'g', -1, 64))
		}
		if i > 0 && b.FromMs <= bars[i-1].FromMs {
			return fmt.Errorf("bars[%d].from_ms is %d, not after bars[%d]'s %d: each bar begins later into the pause than the one before",
				i, b.FromMs, i-1, bars[i-1].FromMs)
		}
	}
	return nil
}

// Resolve is how the choice ends turns under p. An agent's own bars are its own, ending a
// turn at the first check that reaches its bar; a preset the server doesn't publish, or a
// wait shorter than every point of its frontier makes a caller wait, is an error that says
// what the server does publish.
func (p Policy) Resolve(e EndOfTurn) (Ending, error) {
	if err := e.Check(); err != nil {
		return Ending{}, err
	}
	switch {
	case e.Bars != nil:
		return Ending{Name: Custom, Bars: e.Bars, ChecksInARow: 1}, nil
	case e.Preset != "":
		ending, ok := p.preset(e.Preset)
		if !ok {
			return Ending{}, stack.Wrap(fmt.Errorf("end_of_turn.preset: there is no preset %q; the model's are %s", e.Preset, p.presetNames()))
		}
		return ending, nil
	case e.MaxWaitMs > 0:
		return p.within(e.MaxWaitMs)
	}
	return p.DefaultEnding(), nil
}

// DefaultEnding is the server's default preset, which an agent that chooses nothing ends
// its turns by.
func (p Policy) DefaultEnding() Ending {
	ending, _ := p.preset(p.Default)
	return ending
}

// preset is the point a preset names.
func (p Policy) preset(name string) (Ending, bool) {
	wait, ok := p.Presets[name]
	if !ok {
		return Ending{}, false
	}
	point, ok := p.point(wait)
	if !ok {
		return Ending{}, false
	}
	return point.ending("preset:" + name), true
}

// point is the frontier's point fitted to wait.
func (p Policy) point(wait int) (*FrontierPoint, bool) {
	for i := range p.Frontier {
		if p.Frontier[i].MaxWaitMs == wait {
			return &p.Frontier[i], true
		}
	}
	return nil, false
}

// within is the frontier point with the fewest cut-ins whose measured p90 wait is at most
// maxWait. A tie goes to the shorter wait, then to fewer dictation pauses cut.
func (p Policy) within(maxWait int) (Ending, error) {
	if len(p.Frontier) == 0 {
		return Ending{}, stack.Wrap(fmt.Errorf("end_of_turn.max_wait_ms: the model was measured at no waits to choose from; name a preset (%s) or bars", p.presetNames()))
	}
	var best *FrontierPoint
	quickest := p.Frontier[0].Measured.AnswerMsP90
	for i := range p.Frontier {
		point := &p.Frontier[i]
		quickest = min(quickest, point.Measured.AnswerMsP90)
		if point.Measured.AnswerMsP90 > maxWait {
			continue
		}
		if best == nil || point.before(*best) {
			best = point
		}
	}
	if best == nil {
		return Ending{}, stack.Wrap(fmt.Errorf("end_of_turn.max_wait_ms is %d, under every wait the model was measured at: the quickest has a caller wait %d ms at p90", maxWait, quickest))
	}
	return best.ending("max_wait_ms:" + strconv.Itoa(best.MaxWaitMs)), nil
}

// ending is the point as an Ending called name.
func (f *FrontierPoint) ending(name string) Ending {
	return Ending{Name: name, Bars: f.Bars, ChecksInARow: f.ChecksInARow, Point: f}
}

// before says the point is a better choice than other within the same wait.
func (f FrontierPoint) before(other FrontierPoint) bool {
	a, b := f.Measured, other.Measured
	switch {
	case a.CutIns != b.CutIns:
		return a.CutIns < b.CutIns
	case a.AnswerMsP90 != b.AnswerMsP90:
		return a.AnswerMsP90 < b.AnswerMsP90
	case a.DictationPausesCut != b.DictationPausesCut:
		return a.DictationPausesCut < b.DictationPausesCut
	}
	return f.MaxWaitMs < other.MaxWaitMs
}

// valid says the point has its wait, valid bars, at least one check in a row, and measures.
func (f FrontierPoint) valid() bool {
	m := f.Measured
	return f.MaxWaitMs > 0 && checkBars(f.Bars) == nil && f.ChecksInARow >= 1 && m.AnswerMsP90 > 0 && m.AnswerMsMedian >= 0 &&
		m.CutIns >= 0 && m.CutIns <= 1 && m.DictationPausesCut >= 0 && m.DictationPausesCut <= 1 &&
		m.SlowCounts >= 0 && m.SlowCountsCut >= 0 && m.SlowCountsCut <= m.SlowCounts
}

// presetNames are the server's presets, in order, as a sentence says them.
func (p Policy) presetNames() string {
	names := slices.Sorted(maps.Keys(p.Presets))
	if len(names) < 2 {
		return strings.Join(names, "")
	}
	return strings.Join(names[:len(names)-1], ", ") + " and " + names[len(names)-1]
}

// Bar is what p_eot must reach, by bars, at a check paused into the pause: the last of the
// bars to have begun by then, raised on the turn's first pause.
func (p Policy) Bar(bars []PolicyBar, paused time.Duration, firstPause bool) float64 {
	var bar float64
	for _, b := range bars {
		if paused >= time.Duration(b.FromMs)*time.Millisecond {
			bar = b.Bar
		}
	}
	if firstPause {
		bar = raised(bar, p.FirstPauseLogit)
	}
	return bar
}

// runs counts, for each participant, the checks of the pause they are in that reached their
// bar in a row, which a point asking for more than one ends a turn on.
type runs struct {
	mu sync.Mutex
	of map[string]run
}

// run is a participant's checks in a row that reached their bar, and the time into the pause
// of the last of them.
type run struct {
	reached int
	paused  time.Duration
}

// count records a check paused into a participant's pause, and whether it reached its bar,
// and is how many checks in a row have, this one included. The first check of a pause starts
// again, which is what the caller speaking does, and so does a check that is not the one after
// the last counted, being more than half an interval late: a check not asked about, or not
// answered, is not one that reached its bar.
func (r *runs) count(participant string, first bool, paused, interval time.Duration, reached bool) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.of == nil {
		r.of = map[string]run{}
	}
	last := r.of[participant]
	if first || paused <= last.paused || paused-last.paused > interval+interval/2 {
		last.reached = 0
	}
	last.paused = paused
	if reached {
		last.reached++
	} else {
		last.reached = 0
	}
	r.of[participant] = last
	return last.reached
}
