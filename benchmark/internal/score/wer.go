package score

import (
	"regexp"
	"strings"
	"unicode"
)

const NormalizerVersion = "english-basic-v1"

var (
	currency = regexp.MustCompile(`\$([0-9]+(?:\.[0-9]+)?)`)
	fillers  = regexp.MustCompile(`\b(?:um+|uh+|er+|ah+)\b`)
)

var contractions = map[string]string{
	"i'm":       "i am",
	"it's":      "it is",
	"that's":    "that is",
	"there's":   "there is",
	"he's":      "he is",
	"she's":     "she is",
	"we're":     "we are",
	"they're":   "they are",
	"you're":    "you are",
	"i've":      "i have",
	"we've":     "we have",
	"they've":   "they have",
	"you've":    "you have",
	"i'll":      "i will",
	"we'll":     "we will",
	"you'll":    "you will",
	"they'll":   "they will",
	"he'll":     "he will",
	"she'll":    "she will",
	"don't":     "do not",
	"doesn't":   "does not",
	"didn't":    "did not",
	"can't":     "cannot",
	"won't":     "will not",
	"isn't":     "is not",
	"aren't":    "are not",
	"wasn't":    "was not",
	"weren't":   "were not",
	"haven't":   "have not",
	"hasn't":    "has not",
	"hadn't":    "had not",
	"wouldn't":  "would not",
	"couldn't":  "could not",
	"shouldn't": "should not",
	"let's":     "let us",
}

// Alignment is the word-level edit between a reference and a hypothesis.
type Alignment struct {
	Reference     int     `json:"reference_words"`
	Hypothesis    int     `json:"hypothesis_words"`
	Substitutions int     `json:"substitutions"`
	Insertions    int     `json:"insertions"`
	Deletions     int     `json:"deletions"`
	WER           float64 `json:"wer"`
}

func (a Alignment) Errors() int {
	return a.Substitutions + a.Insertions + a.Deletions
}

func (a Alignment) Accuracy() float64 {
	return 1 - a.WER
}

func (a *Alignment) finish() {
	if a.Reference == 0 {
		if a.Hypothesis == 0 {
			a.WER = 0
			return
		}
		a.WER = 1
		return
	}
	rate := float64(a.Errors()) / float64(a.Reference)
	if rate > 1 {
		rate = 1
	}
	a.WER = rate
}

// ScoreWER aligns reference and hypothesis after optional normalization.
func ScoreWER(reference, heard string, normalize bool) Alignment {
	if normalize {
		reference = Normalize(reference)
		heard = Normalize(heard)
	}
	want := werWords(reference, !normalize)
	got := werWords(heard, !normalize)
	out := Alignment{Reference: len(want), Hypothesis: len(got)}
	if len(want) == 0 {
		out.Insertions = len(got)
		out.finish()
		return out
	}
	out.Substitutions, out.Deletions, out.Insertions = werEditCounts(want, got)
	out.finish()
	return out
}

// Normalize is the english-basic-v1 preset: casefold, contractions, currency,
// fillers, then punctuation dropped. A new preset is a new version string.
func Normalize(text string) string {
	text = strings.ToLower(text)
	text = currency.ReplaceAllString(text, "$1 dollars")
	fields := strings.Fields(text)
	var expanded []string
	for _, field := range fields {
		trimmed := strings.Trim(field, ".,!?;:\"()[]")
		if replacement, ok := contractions[trimmed]; ok {
			expanded = append(expanded, strings.Fields(replacement)...)
			continue
		}
		expanded = append(expanded, trimmed)
	}
	text = strings.Join(expanded, " ")
	text = fillers.ReplaceAllString(text, " ")
	return strings.Join(werWords(text, false), " ")
}

func werWords(text string, fold bool) []string {
	cleaned := strings.Map(func(r rune) rune {
		if fold {
			r = unicode.ToLower(r)
		}
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r), r == '\'':
			return r
		default:
			return ' '
		}
	}, text)
	return strings.Fields(cleaned)
}

type werCounts struct {
	sub, del, ins int
}

func (c werCounts) total() int { return c.sub + c.del + c.ins }

func werEditCounts(want, got []string) (sub, del, ins int) {
	previous := make([]werCounts, len(got)+1)
	current := make([]werCounts, len(got)+1)
	for j := range previous {
		previous[j] = werCounts{ins: j}
	}
	for i := 1; i <= len(want); i++ {
		current[0] = werCounts{del: i}
		for j := 1; j <= len(got); j++ {
			diag := previous[j-1]
			if want[i-1] != got[j-1] {
				diag.sub++
			}
			up := previous[j]
			up.del++
			left := current[j-1]
			left.ins++
			current[j] = minWER(diag, up, left)
		}
		previous, current = current, previous
	}
	best := previous[len(got)]
	return best.sub, best.del, best.ins
}

func minWER(options ...werCounts) werCounts {
	best := options[0]
	for _, option := range options[1:] {
		if option.total() < best.total() {
			best = option
		}
	}
	return best
}
