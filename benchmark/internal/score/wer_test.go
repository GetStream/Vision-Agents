package score

import "testing"

func TestNormalizeFoldsCurrencyAndContractions(t *testing.T) {
	got := Normalize("It's $50 at 3pm — uh, really.")
	if got != "it is 50 dollars at 3pm really" {
		t.Fatalf("got %q", got)
	}
}

func TestScoreWERRawPenalizesFormatting(t *testing.T) {
	raw := ScoreWER("It's $50", "it is 50 dollars", false)
	if raw.WER == 0 {
		t.Fatal("raw WER should see a formatting difference")
	}
	norm := ScoreWER("It's $50", "it is 50 dollars", true)
	if norm.WER != 0 {
		t.Fatalf("normalized WER %v %+v", norm.WER, norm)
	}
}

func TestScoreWERCountsASubstitution(t *testing.T) {
	got := ScoreWER("one two three", "one two four", false)
	if got.Substitutions != 1 || got.Insertions != 0 || got.Deletions != 0 {
		t.Fatalf("%+v", got)
	}
	if got.WER < 0.33 || got.WER > 0.34 {
		t.Fatalf("wer %v", got.WER)
	}
}

func TestClockTimesAreReadAsSpoken(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{"Booked for 7:30.", "booked for seven thirty"},
		{"at 07:05 tonight", "at seven oh five tonight"},
		{"19:45 works", "seven forty five works"},
		{"see you at 8:00", "see you at eight"},
	} {
		if got := Normalize(c.in); got != c.want {
			t.Errorf("Normalize(%q) = %q, want %q", c.in, got, c.want)
		}
	}
	if wer := ScoreWER("table at 7:30", "table at seven thirty", true).WER; wer != 0 {
		t.Fatalf("normalized WER = %v, want 0", wer)
	}
}
