package audiofloor

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
)

// testFrontier is a measured frontier as a server publishes one. Its fewest cut-ins are not
// at its longest wait, so a choice by wait is the fewest within it, not the longest.
const testFrontier = `[
	{"max_wait_ms":600,"bars":[{"from_ms":0,"bar":0.7},{"from_ms":300,"bar":0.35}],"measured":{"cut_ins":0.15,"answer_ms_median":400,"answer_ms_p90":600,"dictation_pauses_cut":0.4}},
	{"max_wait_ms":900,"bars":[{"from_ms":0,"bar":0.9},{"from_ms":600,"bar":0.3}],"measured":{"cut_ins":0.08,"answer_ms_median":600,"answer_ms_p90":900,"dictation_pauses_cut":0.2}},
	{"max_wait_ms":1200,"bars":[{"from_ms":0,"bar":0.995},{"from_ms":1100,"bar":0.2}],"measured":{"cut_ins":0.045,"answer_ms_median":1100,"answer_ms_p90":1100,"dictation_pauses_cut":0.126}},
	{"max_wait_ms":1500,"bars":[{"from_ms":0,"bar":0.995},{"from_ms":1400,"bar":0.2}],"measured":{"cut_ins":0.05,"answer_ms_median":1400,"answer_ms_p90":1400,"dictation_pauses_cut":0.1}}
]`

// testInfoWithFrontier is testInfo with a frontier.
var testInfoWithFrontier = strings.Replace(testInfo, `"default":"safe",`, `"default":"safe","frontier":`+testFrontier+`,`, 1)

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

// Nothing chosen is the server's default; a preset is its bars; an agent's own bars are its
// own.
func TestAChoiceResolvesToBars(t *testing.T) {
	p := policyOf(t, testInfo)
	got, err := p.Resolve(EndOfTurn{})
	require.NoError(t, err)
	require.Equal(t, Ending{Name: "preset:safe", Bars: []PolicyBar{{FromMs: 0, Bar: 0.995}, {FromMs: 1100, Bar: 0.2}}}, got)

	got, err = p.Resolve(EndOfTurn{Preset: "fast"})
	require.NoError(t, err)
	require.Equal(t, Ending{Name: "preset:fast", Bars: []PolicyBar{{FromMs: 0, Bar: 0.8}, {FromMs: 500, Bar: 0.3}}}, got)

	own := []PolicyBar{{FromMs: 0, Bar: 0.97}, {FromMs: 900, Bar: 0.25}}
	got, err = p.Resolve(EndOfTurn{Bars: own})
	require.NoError(t, err)
	require.Equal(t, Ending{Name: Custom, Bars: own}, got)

	_, err = p.Resolve(EndOfTurn{Preset: "eager"})
	require.EqualError(t, err, `end_of_turn.preset: there is no preset "eager"; the model's are fast and safe`)
	_, err = p.Resolve(EndOfTurn{Preset: "fast", MaxWaitMs: 800})
	require.ErrorContains(t, err, "it takes one of")
}

// A wait chooses the frontier point with the fewest cut-ins whose measured p90 wait is within
// it, which need not be the point fitted to the longest wait.
func TestAWaitChoosesTheFewestCutInsWithinIt(t *testing.T) {
	p := policyOf(t, testInfoWithFrontier)
	for _, c := range []struct {
		wait int
		want string
	}{
		{600, "max_wait_ms:600"},
		{899, "max_wait_ms:600"},
		{900, "max_wait_ms:900"},
		{1099, "max_wait_ms:900"},
		{1100, "max_wait_ms:1200"},
		{1400, "max_wait_ms:1200"},
		{60000, "max_wait_ms:1200"},
	} {
		got, err := p.Resolve(EndOfTurn{MaxWaitMs: c.wait})
		require.NoError(t, err, "%d ms", c.wait)
		require.Equal(t, c.want, got.Name, "%d ms", c.wait)
	}
	got, err := p.Resolve(EndOfTurn{MaxWaitMs: 1000})
	require.NoError(t, err)
	require.Equal(t, []PolicyBar{{FromMs: 0, Bar: 0.9}, {FromMs: 600, Bar: 0.3}}, got.Bars)

	_, err = p.Resolve(EndOfTurn{MaxWaitMs: 599})
	require.EqualError(t, err, "end_of_turn.max_wait_ms is 599, under every wait the model was measured at: the quickest has a caller wait 600 ms at p90")

	_, err = policyOf(t, testInfo).Resolve(EndOfTurn{MaxWaitMs: 1000})
	require.EqualError(t, err, "end_of_turn.max_wait_ms: the model was measured at no waits to choose from; name a preset (fast and safe) or bars")
}

// Between points with as few cut-ins, the shorter wait wins, then fewer dictation pauses cut.
func TestATieGoesToTheShorterWait(t *testing.T) {
	bars := []PolicyBar{{FromMs: 0, Bar: 0.9}}
	p := Policy{Frontier: []FrontierPoint{
		{MaxWaitMs: 1000, Bars: bars, Measured: Measured{CutIns: 0.05, AnswerMsP90: 1000, DictationPausesCut: 0.1}},
		{MaxWaitMs: 800, Bars: bars, Measured: Measured{CutIns: 0.05, AnswerMsP90: 800, DictationPausesCut: 0.3}},
		{MaxWaitMs: 900, Bars: bars, Measured: Measured{CutIns: 0.05, AnswerMsP90: 800, DictationPausesCut: 0.2}},
	}}
	got, err := p.Resolve(EndOfTurn{MaxWaitMs: 1000})
	require.NoError(t, err)
	require.Equal(t, "max_wait_ms:900", got.Name)
}

// A frontier point without its wait, its measures or valid bars makes the policy incomplete.
func TestAnIncompleteFrontierIsAnIncompletePolicy(t *testing.T) {
	require.True(t, policyOf(t, testInfoWithFrontier).valid())
	for _, point := range []string{
		`{"bars":[{"from_ms":0,"bar":0.9}],"measured":{"cut_ins":0.1,"answer_ms_median":500,"answer_ms_p90":800}}`,
		`{"max_wait_ms":800,"bars":[{"from_ms":0,"bar":0.9}],"measured":{"cut_ins":0.1,"answer_ms_median":500}}`,
		`{"max_wait_ms":800,"bars":[{"from_ms":0,"bar":0.9}],"measured":{"cut_ins":1.5,"answer_ms_median":500,"answer_ms_p90":800}}`,
		`{"max_wait_ms":800,"bars":[{"from_ms":100,"bar":0.9}],"measured":{"cut_ins":0.1,"answer_ms_median":500,"answer_ms_p90":800}}`,
		`{"max_wait_ms":800,"measured":{"cut_ins":0.1,"answer_ms_median":500,"answer_ms_p90":800}}`,
	} {
		var read struct {
			EndOfTurn Policy `json:"end_of_turn"`
		}
		info := strings.Replace(testInfo, `"default":"safe",`, `"default":"safe","frontier":[`+point+`],`, 1)
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
	s.server.info = testInfoWithFrontier
	s.server.decision = `{"answer":"wait","answer_probability":0.4,"p_eot":0.6,"options":[],"decide_ms":1,"ruling":{"disposition":"respond","floor":"continue"}}`
	s.log.Reset()
}

func (s *EndOfTurnSuite) hearer(choice EndOfTurn) *Hearer {
	return NewHearer(s.client, choice, slog.New(slog.NewTextHandler(&s.log, nil)))
}

// checkAt is what the hearer makes of p_eot 0.6 at a check paused into the pause.
func (s *EndOfTurnSuite) checkAt(h *Hearer, paused time.Duration) Check {
	s.T().Helper()
	check, err := h.Check(context.Background(), harness.FlowTurn{Audio: some, Waited: true, Paused: paused})
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
		{EndOfTurn{MaxWaitMs: 1000}, "max_wait_ms:900", 0.3, true},
		{EndOfTurn{MaxWaitMs: 1300}, "max_wait_ms:1200", 0.995, false},
		{EndOfTurn{Bars: []PolicyBar{{FromMs: 0, Bar: 0.99}, {FromMs: 700, Bar: 0.55}}}, Custom, 0.55, true},
	} {
		got := s.checkAt(s.hearer(c.choice), 800*time.Millisecond)
		s.Equal(c.name, got.EndOfTurn, "%+v", c.choice)
		s.InDelta(c.bar, got.Bar, 1e-9, "%+v", c.choice)
		s.Equal(c.ended, got.Ended, "%+v", c.choice)
		if !c.ended {
			s.Equal(harness.Decided{Disposition: harness.Wait, Floor: harness.Continue, Again: 300 * time.Millisecond}, got.Ruling)
		}
	}
	s.Equal(1, s.server.infos, "the policy is the client's, asked once for every agent")
	s.Empty(s.log.String())
}

// The server's checks, first-pause raise and cap are kept under an agent's own bars.
func (s *EndOfTurnSuite) TestOwnBarsKeepTheServersCapAndFirstPause() {
	h := s.hearer(EndOfTurn{Bars: []PolicyBar{{FromMs: 0, Bar: 0.999}}})
	s.False(s.checkAt(h, 2400*time.Millisecond).Ended)
	s.True(s.checkAt(h, 2500*time.Millisecond).Ended)
	first, err := h.Check(context.Background(), harness.FlowTurn{Audio: some, Paused: 200 * time.Millisecond, FirstPause: true})
	s.Require().NoError(err)
	s.InDelta(raised(0.999, 0.5), first.Bar, 1e-12)
}

// A choice the server's policy no longer resolves, as after it moved to a model with other
// presets, ends turns by its default, and is said once.
func (s *EndOfTurnSuite) TestAChoiceTheServerNoLongerHasFallsBackToItsDefault() {
	s.server.info = testInfo
	for _, choice := range []EndOfTurn{{Preset: "eager"}, {MaxWaitMs: 1000}} {
		s.log.Reset()
		h := s.hearer(choice)
		for range 3 {
			got := s.checkAt(h, 800*time.Millisecond)
			s.Equal("preset:safe", got.EndOfTurn)
			s.InDelta(0.995, got.Bar, 1e-9)
		}
		s.Equal(1, strings.Count(s.log.String(), "names nothing the server publishes"), s.log.String())
	}
}
