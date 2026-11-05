package audiofloor

import (
	"errors"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// EndOfTurn is an agent's choice of how eager turn-taking is, which the server's policy turns
// into bars. It names at most one of these, and naming none takes the server's default preset:
//
//   - MaxWaitMs, the most a caller who has finished may wait for the answer: the point of the
//     server's measured frontier with the fewest cut-ins whose measured p90 wait is within it;
//   - Preset, one of the server's presets by name;
//   - Bars of the agent's own.
//
// Whichever it names replaces only the bars. When a pause is checked, how much more a turn's
// first pause asks and the pause that ends a turn whatever the model says stay the server's.
type EndOfTurn struct {
	MaxWaitMs int
	Preset    string
	Bars      []PolicyBar
}

// FrontierPoint is bars fitted to have a caller wait at most MaxWaitMs for the answer, and
// what they did on held-out calls.
type FrontierPoint struct {
	MaxWaitMs int         `json:"max_wait_ms"`
	Bars      []PolicyBar `json:"bars"`
	Measured  Measured    `json:"measured"`
}

// Measured is what a frontier point's bars did on held-out calls: the share of turns they
// ended while the caller was still speaking, how long a caller who had finished waited for
// the answer, and the share of real pauses inside dictated values they took for an end.
type Measured struct {
	CutIns             float64 `json:"cut_ins"`
	AnswerMsMedian     int     `json:"answer_ms_median"`
	AnswerMsP90        int     `json:"answer_ms_p90"`
	DictationPausesCut float64 `json:"dictation_pauses_cut"`
}

// Ending is the bars a session's turns end by. Name is what its log and metrics call them,
// one of a bounded few: preset:NAME, max_wait_ms:N for the frontier point fitted to N (never
// the wait an agent asked for), or custom for an agent's own bars.
type Ending struct {
	Name string
	Bars []PolicyBar
}

// Custom names an agent's own bars in an Ending.
const Custom = "custom"

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

// Resolve is the bars the choice names under p. An agent's own bars are its own; a preset the
// server doesn't publish, or a wait shorter than every point of its frontier makes a caller
// wait, is an error that says what the server does publish.
func (p Policy) Resolve(e EndOfTurn) (Ending, error) {
	if err := e.Check(); err != nil {
		return Ending{}, err
	}
	switch {
	case e.Bars != nil:
		return Ending{Name: Custom, Bars: e.Bars}, nil
	case e.Preset != "":
		preset, ok := p.Presets[e.Preset]
		if !ok {
			return Ending{}, stack.Wrap(fmt.Errorf("end_of_turn.preset: there is no preset %q; the model's are %s", e.Preset, p.presetNames()))
		}
		return Ending{Name: "preset:" + e.Preset, Bars: preset.Bars}, nil
	case e.MaxWaitMs > 0:
		return p.within(e.MaxWaitMs)
	}
	return p.DefaultEnding(), nil
}

// DefaultEnding is the server's default preset, which an agent that chooses nothing ends
// its turns by.
func (p Policy) DefaultEnding() Ending {
	return Ending{Name: "preset:" + p.Default, Bars: p.Presets[p.Default].Bars}
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
	return Ending{Name: "max_wait_ms:" + strconv.Itoa(best.MaxWaitMs), Bars: best.Bars}, nil
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

// valid says the point has its wait, valid bars and measures.
func (f FrontierPoint) valid() bool {
	m := f.Measured
	return f.MaxWaitMs > 0 && checkBars(f.Bars) == nil && m.AnswerMsP90 > 0 && m.AnswerMsMedian >= 0 &&
		m.CutIns >= 0 && m.CutIns <= 1 && m.DictationPausesCut >= 0 && m.DictationPausesCut <= 1
}

// presetNames are the server's presets, in order, as a sentence says them.
func (p Policy) presetNames() string {
	names := slices.Sorted(maps.Keys(p.Presets))
	if len(names) < 2 {
		return strings.Join(names, "")
	}
	return strings.Join(names[:len(names)-1], ", ") + " and " + names[len(names)-1]
}

// Bar is what p_eot must reach, by bars, at a check paused into the pause for the turn to
// have ended: the last of the bars to have begun by then, raised on the turn's first pause.
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
