package report

import (
	"bytes"
	"image/png"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
)

// steadyRun is a run whose replies all take about the same time, so its interval is narrow.
func steadyRun(system, pack string, replyMs, passed, calls int) LabeledRun {
	var results []CallResult
	for i := range calls {
		call := CallResult{ScenarioID: pack + ".golden", Pack: pack, Category: "golden", Trial: i + 1, Passed: i < passed, Outcome: OutcomePass}
		if !call.Passed {
			call.Outcome = OutcomeFail
			call.Metrics.EndStateFail = []string{"reservation.allergen want peanut got <nil>"}
		}
		for j := range 4 {
			call.Metrics.V2V = append(call.Metrics.V2V, score.Timing{V2VMS: replyMs + 10*j})
		}
		call.Metrics.FirstResponse = &score.Timing{V2VMS: replyMs}
		results = append(results, call)
	}
	sum := BuildSummary(system, system, 1, results)
	sum.Manifest.GitCommit = "0a045b76e8fa1151"
	sum.Manifest.NetworkProfile = "local-test"
	return LabeledRun{Label: system, Summary: sum}
}

func TestDigestNamesAWinnerTheIntervalsClear(t *testing.T) {
	d := BuildDigest([]LabeledRun{steadyRun("accelerated", "restaurant", 1800, 6, 8), steadyRun("livekit", "restaurant", 3200, 4, 8)})

	reply := d.Rows[1]
	if reply.Name != "Reply time P50" || reply.Best != 0 || !reply.Decided() {
		t.Fatalf("reply row: %+v", reply)
	}
	text := d.SlackText("Voicebench nightly")
	for _, want := range []string{
		"*Voicebench nightly*",
		"restaurant · k=1 · local-test · 0a045b7",
		"• Reply time P50: *accelerated 1.81 s* · livekit 3.21 s → accelerated",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("missing %q in:\n%s", want, text)
		}
	}
}

func TestDigestDoesNotCallAGapInsideTheNoise(t *testing.T) {
	d := BuildDigest([]LabeledRun{steadyRun("accelerated", "restaurant", 1800, 2, 8), steadyRun("livekit", "restaurant", 1800, 3, 8)})

	if pass := d.Rows[0]; pass.Name != "Pass rate" || pass.Decided() {
		t.Fatalf("two passes in eight against three is not a difference: %+v", pass)
	}
	if text := d.SlackText("Voicebench"); !strings.Contains(text, "• Pass rate: accelerated 25% · livekit 38% → within noise") {
		t.Fatalf("message:\n%s", text)
	}
}

func TestDigestReadsOneSystemAcrossPacksAsOne(t *testing.T) {
	runs := MergeRuns([]LabeledRun{
		steadyRun("accelerated", "restaurant", 1800, 1, 2),
		steadyRun("livekit", "restaurant", 3200, 1, 2),
		steadyRun("accelerated", "telecom", 1800, 1, 2),
	})

	if len(runs) != 2 || len(runs[0].Summary.Calls) != 4 {
		t.Fatalf("merged %d runs, the first with %d calls", len(runs), len(runs[0].Summary.Calls))
	}
	d := BuildDigest(runs)
	if strings.Join(d.Packs, ",") != "restaurant,telecom" || d.Rows[0].Cells[0].Samples != 4 {
		t.Fatalf("packs %v, pass-rate samples %d", d.Packs, d.Rows[0].Cells[0].Samples)
	}
}

func TestDigestCardIsAPicture(t *testing.T) {
	d := BuildDigest([]LabeledRun{steadyRun("accelerated", "restaurant", 1800, 6, 8), steadyRun("livekit", "restaurant", 3200, 4, 8)})

	card, err := d.PNG("Voicebench nightly")
	if err != nil {
		t.Fatal(err)
	}
	img, err := png.Decode(bytes.NewReader(card))
	if err != nil {
		t.Fatal(err)
	}
	if img.Bounds().Dx() != cardWidth {
		t.Fatalf("card is %d wide", img.Bounds().Dx())
	}
}

func TestDigestReportCarriesTheCardAndWhatFailedEachCall(t *testing.T) {
	runs := []LabeledRun{steadyRun("accelerated", "restaurant", 1800, 1, 2), steadyRun("livekit", "restaurant", 3200, 2, 2)}

	page, err := BuildDigest(runs).HTML("Voicebench nightly", []byte("png"), runs)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`src="data:image/png;base64,`,
		"reservation.allergen want peanut got &lt;nil&gt;",
		"<h2>accelerated</h2>",
		"<h2>livekit</h2>",
	} {
		if !strings.Contains(page, want) {
			t.Fatalf("missing %q", want)
		}
	}
}

func TestDigestReportShowsTheSpeechToTextWERBesideTheCalls(t *testing.T) {
	runs := []LabeledRun{steadyRun("accelerated", "restaurant", 1800, 1, 2)}
	d := BuildDigest(runs)

	page, err := d.HTML("Voicebench", []byte("png"), runs)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(page, "<h2>Speech-to-text</h2>") {
		t.Fatal("a run without the speech-to-text bench has no section for it")
	}

	d.STT = []STTSummary{{
		Target: "deepgram/flux-general-en", Clips: 40, Failed: 1, PooledWER: 0.042, PooledWERRaw: 0.118,
		Substitutions: 9, Insertions: 2, Deletions: 3, PerfectRate: 0.65, TimedClips: 39, ToSettleP50Ms: 420,
	}}
	page, err = d.HTML("Voicebench", []byte("png"), runs)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"<h2>Speech-to-text</h2>", "deepgram/flux-general-en", "<strong>4.2%</strong>", "11.8%", "9 / 2 / 3", "65%", "1 failed"} {
		if !strings.Contains(page, want) {
			t.Errorf("report is missing %q", want)
		}
	}
	if strings.Contains(d.SlackText("Voicebench"), "WER") {
		t.Error("the Slack message keeps its approved shape; WER is in the full report")
	}
}
