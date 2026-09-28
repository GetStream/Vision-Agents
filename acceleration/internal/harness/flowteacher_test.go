package harness

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
)

// teacherDirEnvVar is a directory to put the controller's question in, one file per labelled
// case and sample, for a model the router does not reach, such as one behind a local CLI, to
// answer beside it. A labeller of training data is measured this way before its labels are
// trusted.
const teacherDirEnvVar = "FLOW_TEACHER_DIR"

// teacherSamplesEnvVar is how often each case is asked, three when unset. Several samples
// show whether the labeller agrees with itself, which is what a label filter keys on.
const teacherSamplesEnvVar = "FLOW_TEACHER_SAMPLES"

// TestFlowTeacher writes <set>/<case>-<sample>.prompt under FLOW_TEACHER_DIR, and scores every
// <set>/<case>-<sample>.answer found there with the controller's own parser, against the
// label. Run it once to write the prompts, again once they are answered.
func TestFlowTeacher(t *testing.T) {
	dir := os.Getenv(teacherDirEnvVar)
	if dir == "" {
		t.Skip(teacherDirEnvVar + " not set")
	}
	samples := 3
	if raw := os.Getenv(teacherSamplesEnvVar); raw != "" {
		var err error
		samples, err = strconv.Atoi(raw)
		require.NoError(t, err)
	}
	for _, name := range []string{writtenSet, amiSet} {
		set, err := loadFlowSet(name)
		require.NoError(t, err)
		require.NoError(t, os.MkdirAll(filepath.Join(dir, name), 0o755))
		var asked, right, unreadable, cases, majorityRight, unanimous, unanimousRight int
		missed := map[flowState][2]int{}
		for _, one := range set.Cases {
			turn := one.turn(one.ID, set.Contracts)
			prompt := flowInstructions + "\n\nThe agent has been told:\n" + turn.Instructions +
				"\n\n" + flowQuestion(turn)
			votes := map[flowOutcome]int{}
			answered := 0
			for sample := 1; sample <= samples; sample++ {
				base := filepath.Join(dir, name, fmt.Sprintf("%s-%d", one.ID, sample))
				require.NoError(t, os.WriteFile(base+".prompt", []byte(prompt), 0o644))
				raw, err := os.ReadFile(base + ".answer")
				if err != nil {
					continue
				}
				asked++
				answer, err := parseFlow(lastJSONObject(raw))
				if err != nil {
					unreadable++
					continue
				}
				got := one.outcome(answer.Disposition, answer.Floor)
				votes[got]++
				answered++
				tally := missed[one.State]
				tally[1]++
				if got == one.Expect {
					right++
				} else {
					tally[0]++
				}
				missed[one.State] = tally
			}
			if answered == 0 {
				continue
			}
			cases++
			var top flowOutcome
			for outcome, n := range votes {
				if n > votes[top] {
					top = outcome
				}
			}
			if top == one.Expect {
				majorityRight++
			}
			if votes[top] == answered && answered == samples {
				unanimous++
				if top == one.Expect {
					unanimousRight++
				}
			}
		}
		if asked == 0 {
			t.Logf("%s: %d prompts written, no answers yet", name, len(set.Cases)*samples)
			continue
		}
		t.Logf("%s: %d answers over %d cases: %.1f%% right per answer, %d unreadable; "+
			"majority %.1f%%; unanimous on %d cases, %.1f%% of them right",
			name, asked, cases, percent(right, asked-unreadable), unreadable,
			percent(majorityRight, cases), unanimous, percent(unanimousRight, unanimous))
		for _, state := range flowStates {
			if tally, ok := missed[state]; ok {
				t.Logf("  %-20s %5.1f%% right of %d", state, percent(tally[1]-tally[0], tally[1]), tally[1])
			}
		}
	}
}

// lastJSONObject is the last {...} in a CLI's output, which ends with the model's answer.
func lastJSONObject(out []byte) string {
	end := bytes.LastIndexByte(out, '}')
	if end < 0 {
		return string(out)
	}
	depth := 0
	for i := end; i >= 0; i-- {
		switch out[i] {
		case '}':
			depth++
		case '{':
			depth--
			if depth == 0 {
				return string(out[i : end+1])
			}
		}
	}
	return string(out)
}

func percent(n, of int) float64 {
	if of == 0 {
		return 0
	}
	return 100 * float64(n) / float64(of)
}
