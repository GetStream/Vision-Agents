package testaudio

import (
	"strings"
	"unicode"
)

// Alignment is the word-level edit between a reference and a transcript.
type Alignment struct {
	Reference     int
	Hypothesis    int
	Substitutions int
	Insertions    int
	Deletions     int
}

// Errors is substitutions plus insertions plus deletions.
func (a Alignment) Errors() int {
	return a.Substitutions + a.Insertions + a.Deletions
}

// WER is errors over reference words. An empty reference with extra words scores 1.
func (a Alignment) WER() float64 {
	if a.Reference == 0 {
		if a.Hypothesis == 0 {
			return 0
		}
		return 1
	}
	rate := float64(a.Errors()) / float64(a.Reference)
	if rate > 1 {
		return 1
	}
	return rate
}

// Score is 1 − WER.
func (a Alignment) Score() float64 {
	return 1 - a.WER()
}

// Accuracy scores a transcript against what was actually said, as one minus the word
// error rate: 1 is every word in the right place, 0.9 leaves one word in ten wrong.
//
// Words are compared rather than characters, because that is the unit a caller loses:
// a missing "not" changes the meaning while a missing comma changes nothing. Case and
// punctuation are ignored for the same reason, which also keeps a provider that
// punctuates from scoring worse than one that does not.
func Accuracy(reference, heard string) float64 {
	return Align(reference, heard).Score()
}

// Align returns the substitution, insertion, and deletion counts behind Accuracy.
func Align(reference, heard string) Alignment {
	want := words(reference)
	got := words(heard)
	out := Alignment{Reference: len(want), Hypothesis: len(got)}
	if len(want) == 0 {
		out.Insertions = len(got)
		return out
	}
	out.Substitutions, out.Deletions, out.Insertions = editCounts(want, got)
	return out
}

// words reduces a sentence to the tokens worth comparing: lowercase, punctuation
// dropped, apostrophes kept so "don't" stays one word.
func words(text string) []string {
	cleaned := strings.Map(func(r rune) rune {
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r), r == '\'':
			return unicode.ToLower(r)
		default:
			return ' '
		}
	}, text)
	return strings.Fields(cleaned)
}

type counts struct {
	sub, del, ins int
}

func (c counts) total() int { return c.sub + c.del + c.ins }

func editCounts(want, got []string) (sub, del, ins int) {
	previous := make([]counts, len(got)+1)
	current := make([]counts, len(got)+1)
	for j := range previous {
		previous[j] = counts{ins: j}
	}
	for i := 1; i <= len(want); i++ {
		current[0] = counts{del: i}
		for j := 1; j <= len(got); j++ {
			diag := previous[j-1]
			if want[i-1] != got[j-1] {
				diag.sub++
			}
			up := previous[j]
			up.del++
			left := current[j-1]
			left.ins++
			current[j] = minCounts(diag, up, left)
		}
		previous, current = current, previous
	}
	best := previous[len(got)]
	return best.sub, best.del, best.ins
}

func minCounts(options ...counts) counts {
	best := options[0]
	for _, option := range options[1:] {
		if option.total() < best.total() {
			best = option
		}
	}
	return best
}
