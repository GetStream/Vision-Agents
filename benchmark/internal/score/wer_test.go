package score

import "testing"

func TestNormalizeFoldsCurrencyAndContractions(t *testing.T) {
	got := Normalize("It's $50 at 3pm — uh, really.")
	if got != "it is 5 0 dollars at 3 pm really" {
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
		{"Booked for 7:30.", "booked for 7 3 0"},
		{"at 07:05 tonight", "at 7 0 5 tonight"},
		{"19:45 works", "7 4 5 works"},
		{"see you at 8:00", "see you at 8"},
	} {
		if got := Normalize(c.in); got != c.want {
			t.Errorf("Normalize(%q) = %q, want %q", c.in, got, c.want)
		}
	}
	if wer := ScoreWER("table at 7:30", "table at seven thirty", true).WER; wer != 0 {
		t.Fatalf("normalized WER = %v, want 0", wer)
	}
}

func TestNumbersAndSpellingsMatchHoweverTheyWereWritten(t *testing.T) {
	for _, c := range []struct{ script, heard string }{
		{"Callback is 512-555-0142.", "Callback is five one two five five five zero one four two."},
		{"Date of birth March 4 1987.", "Date of birth, March fourth nineteen eighty seven."},
		{"Thursday 2pm with Dr Chen.", "Thursday two PM with doctor Chen."},
		{"Member ID ABC123456.", "Member ID a b c one two three four five six."},
		{"Member ABC123456.", "Member ABC one two three four five six."},
		{"Last name Alvarez, A L V A R E Z.", "Last name Alvarez, a l v a r e z."},
		{"A high chair for a toddler, and I want to be sure.", "A highchair for a toddler, and I wanna be sure."},
		{"Two people at 7:05.", "Two people at seven oh five."},
		{"Garage on Second Street.", "Garage on 2nd Street."},
	} {
		if got := ScoreWER(c.script, c.heard, true); got.Errors() != 0 {
			t.Errorf("%q against %q: %+v (%q vs %q)", c.script, c.heard, got, Normalize(c.script), Normalize(c.heard))
		}
	}
}

func TestADroppedDigitIsStillAnError(t *testing.T) {
	got := ScoreWER("Name is Alvarez, 512-555-0142.", "Name is Alvarez five two five five zero one four two.", true)
	if got.Deletions != 2 || got.Substitutions != 0 || got.Insertions != 0 {
		t.Fatalf("one two dropped from the phone number: %+v", got)
	}
	if got := ScoreWER("Yes, Chen, two people.", "Yes, Jen, two people.", true); got.Substitutions != 1 {
		t.Fatalf("a misheard name is still a substitution: %+v", got)
	}
}
