package report

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// GroupResult is how a group of trials came out. Invalid trials produced no verdict, so they
// are counted beside the group rather than in it.
type GroupResult struct {
	Name    string
	Passed  int
	Valid   int
	Invalid int
}

// Score is the pass rate on 0-100, or -1 when no trial in the group produced a verdict.
func (g GroupResult) Score() int {
	if g.Valid == 0 {
		return -1
	}
	return int(math.Round(100 * float64(g.Passed) / float64(g.Valid)))
}

// Text is the group as it reads in a message: "2/8 passed, 1 invalid".
func (g GroupResult) Text() string {
	out := fmt.Sprintf("%d/%d passed", g.Passed, g.Valid)
	if g.Invalid > 0 {
		out += fmt.Sprintf(", %d invalid", g.Invalid)
	}
	return out
}

// ScoreText is the score as it reads in a message, or a dash when there is none.
func (g GroupResult) ScoreText() string {
	if score := g.Score(); score >= 0 {
		return fmt.Sprint(score)
	}
	return "—"
}

// Results is one run's trials grouped by pack and by scenario type. The overall score is the
// mean of the pack scores, so a pack with more scenarios does not outweigh the others; packs
// with no verdict are left out of it.
type Results struct {
	Overall GroupResult
	Score   int
	ByPack  []GroupResult
	ByKind  []GroupResult
	// Causes counts the failed calls by why they failed, when their artifacts were read.
	Causes []CauseCount
}

// ScoreText is the overall score as it reads in a message, or a dash when there is none.
func (r Results) ScoreText() string {
	if r.Score >= 0 {
		return fmt.Sprint(r.Score)
	}
	return "—"
}

// SummarizeResults groups a run's calls by pack and by scenario type.
func SummarizeResults(calls []CallResult) Results {
	out := Results{Overall: GroupResult{Name: "all"}, Score: -1}
	packs := map[string]*GroupResult{}
	kinds := map[string]*GroupResult{}
	add := func(groups map[string]*GroupResult, name, outcome string) {
		group, ok := groups[name]
		if !ok {
			group = &GroupResult{Name: name}
			groups[name] = group
		}
		count(group, outcome)
	}
	for _, call := range calls {
		outcome := callOutcome(call)
		count(&out.Overall, outcome)
		add(packs, call.Pack, outcome)
		add(kinds, scenarioKind(call.ScenarioID), outcome)
	}
	out.ByPack = sortedGroups(packs)
	out.ByKind = sortedGroups(kinds)
	var sum, scored int
	for _, pack := range out.ByPack {
		if score := pack.Score(); score >= 0 {
			sum += score
			scored++
		}
	}
	if scored > 0 {
		out.Score = int(math.Round(float64(sum) / float64(scored)))
	}
	return out
}

func count(group *GroupResult, outcome string) {
	switch outcome {
	case OutcomeInvalid:
		group.Invalid++
	case OutcomePass:
		group.Passed++
		group.Valid++
	default:
		group.Valid++
	}
}

// scenarioKind is what a scenario tests, its id without the pack: restaurant.interrupt is an
// interrupt.
func scenarioKind(id string) string {
	if _, kind, ok := strings.Cut(id, "."); ok {
		return kind
	}
	return id
}

func sortedGroups(groups map[string]*GroupResult) []GroupResult {
	out := make([]GroupResult, 0, len(groups))
	for _, group := range groups {
		out = append(out, *group)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}
