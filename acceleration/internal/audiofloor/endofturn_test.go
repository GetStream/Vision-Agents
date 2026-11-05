package audiofloor

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
)

func policyOf(t *testing.T, info string) Policy {
	t.Helper()
	var read struct {
		EndOfTurn Policy `json:"end_of_turn"`
	}
	require.NoError(t, json.Unmarshal([]byte(info), &read))
	require.True(t, read.EndOfTurn.valid(), info)
	return read.EndOfTurn
}

// A choice names one way of choosing at most, and bars that begin at 0 ms and each after the
// one before, each a probability strictly between 0 and 1.
func TestAChoiceIsCheckedForWhatItSays(t *testing.T) {
	for _, valid := range []EndOfTurn{
		{},
		{MaxWaitMs: 800},
		{Preset: "fast"},
		{Bars: []PolicyBar{{FromMs: 0, Bar: 0.9}}},
		{Bars: []PolicyBar{{FromMs: 0, Bar: 0.9}, {FromMs: 700, Bar: 0.95}, {FromMs: 1500, Bar: 0.1}}},
	} {
		require.NoError(t, valid.Check(), "%+v", valid)
	}
	for _, c := range []struct {
		choice EndOfTurn
		says   string
	}{
		{EndOfTurn{MaxWaitMs: 800, Preset: "fast"}, "end_of_turn names max_wait_ms and preset: it takes one of max_wait_ms, preset and bars"},
		{EndOfTurn{Preset: "fast", Bars: []PolicyBar{{FromMs: 0, Bar: 0.9}}}, "end_of_turn names preset and bars"},
		{EndOfTurn{MaxWaitMs: -1}, "end_of_turn.max_wait_ms is -1: a wait is a number of milliseconds above 0"},
		{EndOfTurn{Bars: []PolicyBar{}}, "end_of_turn.bars is empty"},
		{EndOfTurn{Bars: []PolicyBar{{FromMs: 200, Bar: 0.9}}}, "end_of_turn.bars[0].from_ms is 200: the first bar begins at 0 ms into the pause"},
		{EndOfTurn{Bars: []PolicyBar{{FromMs: 0, Bar: 0.9}, {FromMs: 800, Bar: 0.3}, {FromMs: 800, Bar: 0.2}}},
			"end_of_turn.bars[2].from_ms is 800, not after bars[1]'s 800: each bar begins later into the pause than the one before"},
		{EndOfTurn{Bars: []PolicyBar{{FromMs: 0, Bar: 0.9}, {FromMs: 900, Bar: 0.3}, {FromMs: 500, Bar: 0.2}}}, "end_of_turn.bars[2].from_ms is 500, not after bars[1]'s 900"},
		{EndOfTurn{Bars: []PolicyBar{{FromMs: 0, Bar: 1}}}, "end_of_turn.bars[0].bar is 1: a bar is a probability strictly between 0 and 1"},
		{EndOfTurn{Bars: []PolicyBar{{FromMs: 0, Bar: 0.9}, {FromMs: 500, Bar: 0}}}, "end_of_turn.bars[1].bar is 0: a bar is a probability strictly between 0 and 1"},
		{EndOfTurn{Bars: []PolicyBar{{FromMs: 0, Bar: 1.5}}}, "end_of_turn.bars[0].bar is 1.5"},
		{EndOfTurn{Bars: []PolicyBar{{FromMs: 0, Bar: -0.2}}}, "end_of_turn.bars[0].bar is -0.2"},
	} {
		err := c.choice.Check()
		require.Error(t, err, "%+v", c.choice)
		require.Contains(t, err.Error(), c.says)
	}
}

// Nothing chosen is the server's default preset, and a preset is the point it names: bars,
// checks in a row and measures. An agent's own bars end a turn at the first check that reaches
// its bar, and were measured by nobody.
func TestAChoiceResolvesToAPoint(t *testing.T) {
	p := policyOf(t, testInfo)
	got, err := p.Resolve(EndOfTurn{})
	require.NoError(t, err)
	require.Equal(t, "preset:safe", got.Name)
	require.Equal(t, []PolicyBar{{FromMs: 0, Bar: 0.995}, {FromMs: 1100, Bar: 0.2}}, got.Bars)
	require.Equal(t, 1, got.ChecksInARow)
	require.Equal(t, 1100, got.Point.MaxWaitMs)
	require.Equal(t, "pass", got.Guard())

	got, err = p.Resolve(EndOfTurn{Preset: "fast"})
	require.NoError(t, err)
	require.Equal(t, "preset:fast", got.Name)
	require.Equal(t, []PolicyBar{{FromMs: 0, Bar: 0.8}, {FromMs: 500, Bar: 0.3}}, got.Bars)
	require.Equal(t, "fail", got.Guard())
	require.Equal(t, Measured{CutIns: 0.1451, AnswerMsMedian: 500, AnswerMsP90: 800, DictationPausesCut: 0.16, SlowCountsCut: 9, SlowCounts: 9}, got.Point.Measured)

	own := []PolicyBar{{FromMs: 0, Bar: 0.97}, {FromMs: 900, Bar: 0.25}}
	got, err = p.Resolve(EndOfTurn{Bars: own})
	require.NoError(t, err)
	require.Equal(t, Ending{Name: Custom, Bars: own, ChecksInARow: 1}, got)
	require.Equal(t, "unmeasured", got.Guard())

	_, err = p.Resolve(EndOfTurn{Preset: "eager"})
	require.EqualError(t, err, `end_of_turn.preset: there is no preset "eager"; the model's are fast and safe`)
	_, err = p.Resolve(EndOfTurn{Preset: "fast", MaxWaitMs: 800})
	require.ErrorContains(t, err, "it takes one of")
}

// A wait chooses the point with the fewest cut-ins whose measured p90 wait is within it, so an
// app that waits 800 ms gets the 500 ms point, which fails the dictation guard, and one that
// waits 1.4 s gets the point that ends a turn on two checks in a row.
func TestAWaitChoosesTheFewestCutInsWithinIt(t *testing.T) {
	p := policyOf(t, testInfo)
	for _, c := range []struct {
		wait         int
		want, guard  string
		checksInARow int
	}{
		{800, "max_wait_ms:500", "fail", 1},
		{1099, "max_wait_ms:500", "fail", 1},
		{1100, "max_wait_ms:1100", "pass", 1},
		{1399, "max_wait_ms:1100", "pass", 1},
		{1400, "max_wait_ms:1400", "pass", 2},
		{60000, "max_wait_ms:1400", "pass", 2},
	} {
		got, err := p.Resolve(EndOfTurn{MaxWaitMs: c.wait})
		require.NoError(t, err, "%d ms", c.wait)
		require.Equal(t, c.want, got.Name, "%d ms", c.wait)
		require.Equal(t, c.guard, got.Guard(), "%d ms", c.wait)
		require.Equal(t, c.checksInARow, got.ChecksInARow, "%d ms", c.wait)
	}
	_, err := p.Resolve(EndOfTurn{MaxWaitMs: 799})
	require.EqualError(t, err, "end_of_turn.max_wait_ms is 799, under every wait the model was measured at: the quickest has a caller wait 800 ms at p90")
}

// Between points with as few cut-ins, the shorter wait wins, then fewer dictation pauses cut.
func TestATieGoesToTheShorterWait(t *testing.T) {
	bars := []PolicyBar{{FromMs: 0, Bar: 0.9}}
	p := Policy{Frontier: []FrontierPoint{
		{MaxWaitMs: 1000, Bars: bars, ChecksInARow: 1, Measured: Measured{CutIns: 0.05, AnswerMsP90: 1000, DictationPausesCut: 0.1}},
		{MaxWaitMs: 800, Bars: bars, ChecksInARow: 1, Measured: Measured{CutIns: 0.05, AnswerMsP90: 800, DictationPausesCut: 0.3}},
		{MaxWaitMs: 900, Bars: bars, ChecksInARow: 1, Measured: Measured{CutIns: 0.05, AnswerMsP90: 800, DictationPausesCut: 0.2}},
	}}
	got, err := p.Resolve(EndOfTurn{MaxWaitMs: 1000})
	require.NoError(t, err)
	require.Equal(t, "max_wait_ms:900", got.Name)
}

// A point without its wait, its checks in a row, valid bars or measures makes the policy
// incomplete.
func TestAnIncompletePointIsAnIncompletePolicy(t *testing.T) {
	measured := `"measured":{"cut_ins":0.1,"answer_ms_median":500,"answer_ms_p90":1100,"slow_counts_cut":0,"slow_counts":9}`
	for _, point := range []string{
		`{"bars":[{"from_ms":0,"bar":0.9}],"checks_in_a_row":1,` + measured + `}`,
		`{"max_wait_ms":1100,"bars":[{"from_ms":0,"bar":0.9}],` + measured + `}`,
		`{"max_wait_ms":1100,"bars":[{"from_ms":0,"bar":0.9}],"checks_in_a_row":1,"measured":{"cut_ins":0.1,"answer_ms_median":500}}`,
		`{"max_wait_ms":1100,"bars":[{"from_ms":0,"bar":0.9}],"checks_in_a_row":1,"measured":{"cut_ins":1.5,"answer_ms_median":500,"answer_ms_p90":1100}}`,
		`{"max_wait_ms":1100,"bars":[{"from_ms":0,"bar":0.9}],"checks_in_a_row":1,"measured":{"cut_ins":0.1,"answer_ms_median":500,"answer_ms_p90":1100,"slow_counts_cut":10,"slow_counts":9}}`,
		`{"max_wait_ms":1100,"bars":[{"from_ms":100,"bar":0.9}],"checks_in_a_row":1,` + measured + `}`,
		`{"max_wait_ms":1100,"checks_in_a_row":1,` + measured + `}`,
	} {
		var read struct {
			EndOfTurn Policy `json:"end_of_turn"`
		}
		info := `{"end_of_turn":{"first_check_ms":200,"interval_ms":300,"cap_ms":2500,"default":"safe","presets":{"safe":1100},"frontier":[` + point + `]}}`
		require.NoError(t, json.Unmarshal([]byte(info), &read))
		require.False(t, read.EndOfTurn.valid(), point)
	}
}

// EndOfTurnSuite has Hearers of agents that chose differently ask one server.
type EndOfTurnSuite struct {
	serverSuite
	log bytes.Buffer
}

func TestEndOfTurnSuite(t *testing.T) {
	suite.Run(t, new(EndOfTurnSuite))
}

func (s *EndOfTurnSuite) SetupTest() {
	s.serverSuite.SetupTest()
	s.answering(0.6)
	s.log.Reset()
}

// answering has the server answer every turn with p_eot.
func (s *EndOfTurnSuite) answering(pEOT float64) {
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	s.server.decision = `{"answer":"wait","answer_probability":0.4,"p_eot":` + strconv.FormatFloat(pEOT, 'g', -1, 64) +
		`,"options":[],"decide_ms":1,"ruling":{"disposition":"respond","floor":"continue"}}`
}

func (s *EndOfTurnSuite) hearer(choice EndOfTurn) *Hearer {
	return NewHearer(s.client, choice, slog.New(slog.NewTextHandler(&s.log, nil)))
}

// checkAt is what the hearer makes of the server's answer at a check paused into a pause of
// participant p, the first check of its pause unless waited.
func (s *EndOfTurnSuite) checkAt(h *Hearer, paused time.Duration, waited bool) Check {
	s.T().Helper()
	check, err := h.Check(context.Background(), harness.FlowTurn{Participant: "p", Audio: some, Waited: waited, Paused: paused})
	s.Require().NoError(err)
	return check
}

// One server, one moment, one p_eot: whether the turn has ended is each agent's own.
func (s *EndOfTurnSuite) TestEachAgentEndsTurnsByItsOwnChoice() {
	for _, c := range []struct {
		choice EndOfTurn
		name   string
		bar    float64
		ended  bool
	}{
		{EndOfTurn{}, "preset:safe", 0.995, false},
		{EndOfTurn{Preset: "safe"}, "preset:safe", 0.995, false},
		{EndOfTurn{Preset: "fast"}, "preset:fast", 0.3, true},
		{EndOfTurn{MaxWaitMs: 1000}, "max_wait_ms:500", 0.3, true},
		{EndOfTurn{MaxWaitMs: 1300}, "max_wait_ms:1100", 0.995, false},
		{EndOfTurn{Bars: []PolicyBar{{FromMs: 0, Bar: 0.99}, {FromMs: 700, Bar: 0.55}}}, Custom, 0.55, true},
	} {
		got := s.checkAt(s.hearer(c.choice), 800*time.Millisecond, true)
		s.Equal(c.name, got.Ending.Name, "%+v", c.choice)
		s.InDelta(c.bar, got.Bar, 1e-9, "%+v", c.choice)
		s.Equal(c.ended, got.Ended, "%+v", c.choice)
		if !c.ended {
			s.Equal(harness.Decided{Disposition: harness.Wait, Floor: harness.Continue, Again: 300 * time.Millisecond}, got.Ruling)
		}
	}
	s.Equal(1, s.server.infos, "the policy is the client's, asked once for every agent")
	s.NotContains(s.log.String(), "level=WARN")
}

// The 1400 ms point ends a turn on two checks in a row that reach the bar: one holds, and the
// next ends it.
func (s *EndOfTurnSuite) TestTwoChecksInARowEndATurn() {
	h := s.hearer(EndOfTurn{MaxWaitMs: 1400})
	for _, paused := range []time.Duration{200, 500, 800} {
		got := s.checkAt(h, paused*time.Millisecond, paused != 200)
		s.False(got.Ended, "%v ms: p_eot 0.6 is under 0.99", paused)
		s.Zero(got.InARow)
	}
	once := s.checkAt(h, 1100*time.Millisecond, true)
	s.Equal(1, once.InARow)
	s.False(once.Ended, "one check over the bar holds")
	s.Equal(harness.Decided{Disposition: harness.Wait, Floor: harness.Continue, Again: 300 * time.Millisecond}, once.Ruling)
	twice := s.checkAt(h, 1400*time.Millisecond, true)
	s.Equal(2, twice.InARow)
	s.True(twice.Ended, "the second in a row ends the turn")
	s.Equal(harness.Respond, twice.Ruling.Disposition)
}

// The caller speaking starts the count again, and so does a check that isn't the one after the
// last, or one under the bar.
func (s *EndOfTurnSuite) TestARunIsOfOnePausesChecksOneAfterAnother() {
	s.answering(0.999)
	h := s.hearer(EndOfTurn{MaxWaitMs: 1400})
	s.False(s.checkAt(h, 200*time.Millisecond, false).Ended)
	// The caller speaks again: the next check is a new pause's first.
	again := s.checkAt(h, 200*time.Millisecond, false)
	s.Equal(1, again.InARow)
	s.False(again.Ended)
	s.True(s.checkAt(h, 500*time.Millisecond, true).Ended)

	// 500 ms, then 1100 ms: the 800 ms check between them was not asked about, or not answered.
	skipped := s.hearer(EndOfTurn{MaxWaitMs: 1400})
	s.Equal(1, s.checkAt(skipped, 500*time.Millisecond, false).InARow)
	late := s.checkAt(skipped, 1100*time.Millisecond, true)
	s.Equal(1, late.InARow)
	s.False(late.Ended)

	s.answering(0.6)
	under := s.hearer(EndOfTurn{MaxWaitMs: 1400})
	s.Equal(1, s.checkAt(under, 1100*time.Millisecond, false).InARow)
	s.answering(0.1)
	s.Zero(s.checkAt(under, 1400*time.Millisecond, true).InARow)
	s.answering(0.6)
	s.Equal(1, s.checkAt(under, 1700*time.Millisecond, true).InARow)
}

// Another participant's checks are a run of their own.
func (s *EndOfTurnSuite) TestEachParticipantHasARunOfTheirOwn() {
	s.answering(0.999)
	h := s.hearer(EndOfTurn{MaxWaitMs: 1400})
	s.Equal(1, s.checkAt(h, 200*time.Millisecond, false).InARow)
	other, err := h.Check(context.Background(), harness.FlowTurn{Participant: "q", Audio: some, Paused: 500 * time.Millisecond, Waited: true})
	s.Require().NoError(err)
	s.Equal(1, other.InARow)
	s.False(other.Ended)
	s.True(s.checkAt(h, 500*time.Millisecond, true).Ended)
}

// The server's checks, first-pause raise and cap are kept under an agent's own bars.
func (s *EndOfTurnSuite) TestOwnBarsKeepTheServersCapAndFirstPause() {
	h := s.hearer(EndOfTurn{Bars: []PolicyBar{{FromMs: 0, Bar: 0.999}}})
	s.False(s.checkAt(h, 2400*time.Millisecond, true).Ended)
	s.True(s.checkAt(h, 2500*time.Millisecond, true).Ended)
	first, err := h.Check(context.Background(), harness.FlowTurn{Audio: some, Paused: 200 * time.Millisecond, FirstPause: true})
	s.Require().NoError(err)
	s.InDelta(raised(0.999, 0.5), first.Bar, 1e-12)
}

// Which point a session's turns end by, and what it was measured at, is said once.
func (s *EndOfTurnSuite) TestThePointASessionEndsTurnsByIsSaidOnce() {
	h := s.hearer(EndOfTurn{MaxWaitMs: 800})
	for _, paused := range []time.Duration{200, 500, 800} {
		s.checkAt(h, paused*time.Millisecond, paused != 200)
	}
	said := s.log.String()
	s.Equal(1, strings.Count(said, "chorus-asr ends this agent's turns by"), said)
	for _, attribute := range []string{"end_of_turn=max_wait_ms:500", "checks_in_a_row=1", "dictation_guard=fail",
		"max_wait_ms=500", "cut_ins=0.1451", "answer_ms_p90=800", "dictation_pauses_cut=0.16", "slow_counts_cut=9", "slow_counts=9"} {
		s.Contains(said, attribute)
	}
}

// A choice the server's policy no longer resolves, as after it moved to a model with other
// presets or waits, ends turns by its default, and is said once.
func (s *EndOfTurnSuite) TestAChoiceTheServerNoLongerHasFallsBackToItsDefault() {
	for _, choice := range []EndOfTurn{{Preset: "eager"}, {MaxWaitMs: 100}} {
		s.log.Reset()
		h := s.hearer(choice)
		for range 3 {
			got := s.checkAt(h, 800*time.Millisecond, true)
			s.Equal("preset:safe", got.Ending.Name)
			s.InDelta(0.995, got.Bar, 1e-9)
		}
		s.Equal(1, strings.Count(s.log.String(), "names nothing the server publishes"), s.log.String())
	}
}
