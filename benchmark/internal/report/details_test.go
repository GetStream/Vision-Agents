package report

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
)

// callDir writes a call's artifacts the way a run leaves them.
func callDir(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

var bookingScenario = scenario.Scenario{
	ID: "restaurant.selectivity",
	Turns: []scenario.Turn{
		{ID: "intro", Text: "Table for two at 7:30, no allergies, name Chen, callback 512-555-0142."},
		{ID: "cough"},
	},
	Entities: []scenario.Entity{{Name: "name", Value: "Chen"}, {Name: "phone", Value: "512-555-0142"}},
}

func TestAValueTheAgentNeverHeardIsPutDownToHearing(t *testing.T) {
	dir := callDir(t, map[string]string{
		"heard.json": `[{"kind":"answer","said":"Table for two at seven thirty, no allergies, name Chin, callback five one two five five five zero one four two."}]`,
		"tools.json": `[{"name":"create_reservation","args":{"name":"Chin","phone":"512-555-0142"},"result":{"ok":true},"duration_ms":4}]`,
		"judge.json": `{"notes":"Booked the table."}`,
	})
	call := CallResult{ScenarioID: "restaurant.selectivity", Outcome: OutcomeFail, Dir: dir}
	call.Metrics.EndStateFail = []string{"reservation.name want Chen got Chin", "reservation.allergen want none got <nil>"}
	call.Metrics.ExpectedToolFail = []string{"create_reservation.phone want 512-555-0142 got 51255"}
	call.Metrics.GateNotes = []string{"end_state", "expected_tools", "selectivity"}

	detail := LoadCallDetail(call, &bookingScenario)
	causes := map[string]string{}
	for _, failure := range detail.Failures {
		causes[failure.Message[:strings.Index(failure.Message+" ", " ")]] = failure.Cause
	}
	if causes["reservation.name"] != CauseHeard {
		t.Errorf("Chen was heard as Chin, so the name is a hearing failure: %+v", detail.Failures)
	}
	if causes["reservation.allergen"] != CauseDid {
		t.Errorf("the caller never says the word none, so it cannot be misheard: %+v", detail.Failures)
	}
	if causes["create_reservation.phone"] != CauseDid {
		t.Errorf("the phone was heard digit by digit, so a wrong argument is the model's: %+v", detail.Failures)
	}
	if causes["agent"] != CauseTurns {
		t.Errorf("selectivity is turn-taking: %+v", detail.Failures)
	}
	if detail.Cause() != CauseHeard {
		t.Errorf("main cause = %q, the first failure's", detail.Cause())
	}
	if len(detail.Caller) != 1 || strings.Join(detail.Caller[0].Missed, ",") != "Chen" {
		t.Errorf("the caller line should name the value it lost: %+v", detail.Caller)
	}
	if len(detail.Tools) != 1 || !strings.Contains(detail.Tools[0].Args, `"name":"Chin"`) || detail.JudgeNotes != "Booked the table." {
		t.Errorf("tools and judge = %+v %q", detail.Tools, detail.JudgeNotes)
	}
}

func TestAnInvalidCallIsInfra(t *testing.T) {
	detail := LoadCallDetail(CallResult{Outcome: OutcomeInvalid, InvalidReason: []string{"agent stt: deepgram HTTP 401"}}, nil)
	if detail.Cause() != CauseInfra || len(detail.Failures) != 1 {
		t.Fatalf("detail = %+v", detail)
	}
}

func TestADigestSaysWhyItsCallsFailed(t *testing.T) {
	heard := `[{"kind":"answer","said":"Table for two at seven thirty, no allergies, name Chin."}]`
	misheard := CallResult{ScenarioID: "restaurant.selectivity", Pack: "restaurant", Outcome: OutcomeFail,
		Dir: callDir(t, map[string]string{"heard.json": heard})}
	misheard.Metrics.EndStateFail = []string{"reservation.name want Chen got Chin"}
	notBooked := CallResult{ScenarioID: "restaurant.selectivity", Pack: "restaurant", Outcome: OutcomeFail,
		Dir: callDir(t, map[string]string{"heard.json": heard})}
	notBooked.Metrics.ExpectedToolFail = []string{"create_reservation not called"}
	notBooked.Metrics.FalseCutoff = 2
	notBooked.Metrics.V2V = []score.Timing{{TurnID: "intro", V2VMS: 900}}
	runs := []LabeledRun{{Label: "accelerated", Summary: BuildSummary("accelerated", "r", 1, []CallResult{misheard, notBooked})}}

	d := BuildDigest(runs)
	d.Scenarios = map[string]scenario.Scenario{bookingScenario.ID: bookingScenario}
	d.ReadCauses(runs)
	if text := d.SlackText("Voicebench"); !strings.Contains(text, "• Why it failed: accelerated 1 heard wrong, 1 did wrong") {
		t.Fatalf("slack text:\n%s", text)
	}
	page, err := d.HTML("Voicebench", []byte("png"), runs)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"What happened", "missed Chen", "talked over the caller 2 time(s)", "heard wrong", "did wrong"} {
		if !strings.Contains(page, want) {
			t.Errorf("report is missing %q", want)
		}
	}
}

func TestToolUseIsDescribedWithoutFailingTheCall(t *testing.T) {
	call := CallResult{Outcome: OutcomePass, Passed: true}
	call.Metrics.ArgsRight, call.Metrics.ArgsExpected = 7, 9
	call.Metrics.RepeatedTools = []string{"check_availability ×2"}
	call.Metrics.UnknownTools = []string{"book_table"}
	detail := LoadCallDetail(call, nil)
	want := "7 of 9 expected arguments right|called more than once: check_availability ×2|called tools that do not exist: book_table"
	if got := strings.Join(detail.ToolNotes, "|"); got != want || len(detail.Failures) != 0 {
		t.Fatalf("tool notes = %q, failures = %v", got, detail.Failures)
	}
}
