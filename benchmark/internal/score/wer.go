package score

import (
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

// NormalizerVersion pins the WER fold and the structured matcher used for tool
// arguments. v2 adds date-form equivalence (1987-03-04 = 03/04/1987). v3 reads clock
// times as spoken (7:30 = seven thirty) and matches plain plurals (peanuts = peanut). v4
// reads numbers digit by digit whichever way they were written (512-555-0142 = five one two
// five five five zero one four two, 1987 = nineteen eighty seven, March 4 = March fourth,
// 2pm = two PM), joins spelled letters (A L V A R E Z = Alvarez, ABC123 = a b c one two
// three), and folds a few spellings (highchair, Dr, wanna).
const NormalizerVersion = "english-basic-v4"

var (
	clock    = regexp.MustCompile(`\b(\d{1,2}):(\d{2})\b`)
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
	"wanna":     "want to",
	"gonna":     "going to",
	"gotta":     "got to",
	"highchair": "high chair",
	"upfront":   "up front",
	"dr":        "doctor",
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

// Normalize is the english-basic-v4 preset: casefold, clock times as spoken, contractions, currency,
// fillers, punctuation dropped, then numbers as digits one by one and spelled letters joined. Date-form equivalence for tool arguments
// lives in scenario.MatchStructuredValue and is pinned by the same version.
func Normalize(text string) string {
	text = strings.ToLower(text)
	text = clock.ReplaceAllStringFunc(text, spokenClock)
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
	return strings.Join(joinSpelled(splitDigits(numberWords(werWords(text, false)))), " ")
}

var (
	numberValues = map[string]int{
		"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
		"eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
		"fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
		"nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
		"seventy": 70, "eighty": 80, "ninety": 90,
	}
	ordinalValues = map[string]int{
		"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
		"eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
		"fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17,
		"eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
	}
)

// numberWords writes spoken numbers as digits: a tens word takes the unit after it (eighty
// seven = 87), so a year read in pairs (nineteen eighty seven) comes out as 19 87, which the
// digit split makes the same as 1987. "oh" is zero only after another number, as in a clock
// time or a phone number.
func numberWords(words []string) []string {
	out := make([]string, 0, len(words))
	numeric := false
	for i := 0; i < len(words); i++ {
		word := words[i]
		value, ok := numberValues[word]
		if !ok {
			value, ok = ordinalValues[word]
		}
		if !ok && word == "oh" && numeric {
			value, ok = 0, true
		}
		if !ok {
			out = append(out, word)
			numeric = false
			continue
		}
		if value >= 20 && value%10 == 0 && i+1 < len(words) {
			unit, isUnit := numberValues[words[i+1]]
			if !isUnit {
				unit, isUnit = ordinalValues[words[i+1]]
			}
			if isUnit && unit > 0 && unit < 10 {
				value += unit
				i++
			}
		}
		out = append(out, strconv.Itoa(value))
		numeric = true
	}
	return out
}

var ordinalSuffixes = map[string]bool{"st": true, "nd": true, "rd": true, "th": true}

// splitDigits cuts a word where letters and digits meet (2pm = 2 pm) and writes every digit
// as a word of its own, so 512 and five one two align digit by digit. An ordinal's suffix is
// dropped (2nd = second = 2).
func splitDigits(words []string) []string {
	out := make([]string, 0, len(words))
	for _, word := range words {
		var letters strings.Builder
		afterDigit := false
		flush := func() {
			if letters.Len() > 0 && !(afterDigit && ordinalSuffixes[letters.String()]) {
				out = append(out, letters.String())
			}
			letters.Reset()
		}
		for _, r := range word {
			if unicode.IsDigit(r) {
				flush()
				out = append(out, string(r))
				afterDigit = true
				continue
			}
			letters.WriteRune(r)
		}
		flush()
	}
	return out
}

// joinSpelled joins three or more single letters in a row into one word, so a name or an id
// spelled out (a l v a r e z) reads the same as written (Alvarez). Two letters stay apart: "a i"
// is more likely two words than a spelling.
func joinSpelled(words []string) []string {
	out := make([]string, 0, len(words))
	for i := 0; i < len(words); {
		j := i
		for j < len(words) && isLetter(words[j]) {
			j++
		}
		if j-i >= 3 {
			out = append(out, strings.Join(words[i:j], ""))
			i = j
			continue
		}
		out = append(out, words[i])
		i++
	}
	return out
}

func isLetter(word string) bool {
	runes := []rune(word)
	return len(runes) == 1 && unicode.IsLetter(runes[0])
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

var (
	onesWords = []string{"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
		"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"}
	tensWords = []string{"", "", "twenty", "thirty", "forty", "fifty"}
)

// spokenClock says a clock time the way a caller does: 7:30 is "seven thirty", 7:05 "seven oh
// five", 7:00 "seven", and 19:30, as speech-to-text sometimes writes it, "seven thirty".
func spokenClock(match string) string {
	parts := clock.FindStringSubmatch(match)
	hour, _ := strconv.Atoi(parts[1])
	minute, _ := strconv.Atoi(parts[2])
	if hour > 23 || minute > 59 {
		return match
	}
	if hour > 12 {
		hour -= 12
	}
	if hour == 0 {
		hour = 12
	}
	words := []string{onesWords[hour]}
	switch {
	case minute == 0:
	case minute < 10:
		words = append(words, "oh", onesWords[minute])
	case minute < 20:
		words = append(words, onesWords[minute])
	default:
		words = append(words, tensWords[minute/10])
		if minute%10 > 0 {
			words = append(words, onesWords[minute%10])
		}
	}
	return strings.Join(words, " ")
}
