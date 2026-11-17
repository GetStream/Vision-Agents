package testaudio

import (
	"strings"
	"unicode"
)

// Accuracy scores a transcript against what was actually said, as one minus the word
// error rate: 1 is every word in the right place, 0.9 leaves one word in ten wrong.
//
// Words are compared rather than characters, because that is the unit a caller loses:
// a missing "not" changes the meaning while a missing comma changes nothing. Case and
// punctuation are ignored for the same reason, which also keeps a provider that
// punctuates from scoring worse than one that does not.
func Accuracy(reference, heard string) float64 {
	want := words(reference)
	got := words(heard)
	if len(want) == 0 {
		if len(got) == 0 {
			return 1
		}
		return 0
	}

	wrong := distance(want, got)
	// More insertions than reference words would take the score below zero, which says
	// nothing more than "wrong".
	if wrong >= len(want) {
		return 0
	}
	return 1 - float64(wrong)/float64(len(want))
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

// distance is the number of substitutions, deletions and insertions that turn one word
// sequence into the other.
func distance(want, got []string) int {
	// Only the previous row of the edit matrix is ever read, so that is all it keeps.
	previous := make([]int, len(got)+1)
	current := make([]int, len(got)+1)
	for j := range previous {
		previous[j] = j
	}

	for i := 1; i <= len(want); i++ {
		current[0] = i
		for j := 1; j <= len(got); j++ {
			substitution := previous[j-1]
			if want[i-1] != got[j-1] {
				substitution++
			}
			current[j] = min(substitution, min(previous[j]+1, current[j-1]+1))
		}
		previous, current = current, previous
	}
	return previous[len(got)]
}
