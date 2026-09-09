package scenario

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode"
)

var smallWords = [...]string{
	"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
	"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
	"seventeen", "eighteen", "nineteen", "twenty",
}

var wordToNum = func() map[string]int {
	m := make(map[string]int, len(smallWords))
	for i, w := range smallWords {
		m[w] = i
	}
	return m
}()

var (
	timeRe        = regexp.MustCompile(`^(\d{1,2}):(\d{2})$`)
	clockInTextRe = regexp.MustCompile(`(?i)(\d{1,2}):(\d{2})\s*(?:a\.?m\.?|p\.?m\.?)?`)
)

// MatchValue reports whether speech contains a value, allowing common spoken and
// punctuation variants without accepting partial identifiers or numeric substrings.
func MatchValue(text, value string) bool {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" {
		return true
	}
	textTokens := tokens(text)
	for _, variant := range valueVariants(value) {
		want := tokens(variant)
		if containsTokens(textTokens, want) {
			return true
		}
		compact := alnum(variant)
		if compact == "" {
			continue
		}
		for _, token := range textTokens {
			if token == compact {
				return true
			}
		}
	}
	return false
}

// MatchStructuredValue compares structured strings exactly after harmless case,
// whitespace, punctuation, and spoken-time normalization.
func MatchStructuredValue(got, want string) bool {
	got = strings.TrimSpace(got)
	want = strings.TrimSpace(want)
	if got == "" || want == "" {
		return got == want
	}
	if normalizedTime, ok := clockValue(want); ok {
		gotTime, gotOK := clockValue(got)
		return gotOK && gotTime == normalizedTime
	}
	if normalizedDate, ok := dateValue(want); ok {
		gotDate, gotOK := dateValue(got)
		return gotOK && gotDate == normalizedDate
	}
	if wantMinutes, ok := durationMinutes(want, true); ok {
		gotMinutes, gotOK := durationMinutes(got, false)
		return gotOK && gotMinutes == wantMinutes
	}
	return sameMeaning(got, want)
}

func sameMeaning(got, want string) bool {
	g := canonicalToken(got)
	w := canonicalToken(want)
	return g == w && g != ""
}

func canonicalToken(s string) string {
	compact := alnum(s)
	switch compact {
	case "none", "noallergies", "noallergy":
		return "none"
	default:
		return compact
	}
}

func containsTokens(text, want []string) bool {
	if len(want) == 0 || len(want) > len(text) {
		return false
	}
	for i := 0; i+len(want) <= len(text); i++ {
		matched := true
		for j := range want {
			if text[i+j] != want[j] {
				matched = false
				break
			}
		}
		if matched {
			return true
		}
	}
	return false
}

func tokens(s string) []string {
	var out []string
	var b strings.Builder
	flush := func() {
		if b.Len() == 0 {
			return
		}
		out = append(out, b.String())
		b.Reset()
	}
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return out
}

func dateValue(s string) (string, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", false
	}
	candidates := []string{s, titleWords(s)}
	layouts := []string{
		"2006-01-02",
		"01/02/2006",
		"1/2/2006",
		"01-02-2006",
		"1-2-2006",
		"January 2 2006",
		"January 2, 2006",
		"Jan 2 2006",
		"Jan 2, 2006",
		"2 January 2006",
		"2 January, 2006",
	}
	for _, candidate := range candidates {
		for _, layout := range layouts {
			if parsed, err := time.Parse(layout, candidate); err == nil {
				return parsed.Format("2006-01-02"), true
			}
		}
	}
	return "", false
}

func titleWords(s string) string {
	parts := strings.Fields(strings.ToLower(s))
	for i, part := range parts {
		runes := []rune(part)
		if len(runes) == 0 {
			continue
		}
		runes[0] = unicode.ToUpper(runes[0])
		parts[i] = string(runes)
	}
	return strings.Join(parts, " ")
}

// durationHedges are words a model wraps a wait in without changing it.
var durationHedges = map[string]bool{
	"in": true, "about": true, "around": true, "approximately": true,
	"roughly": true, "from": true, "now": true, "or": true, "so": true,
}

var durationUnits = map[string]int{
	"m": 1, "min": 1, "mins": 1, "minute": 1, "minutes": 1,
	"h": 60, "hr": 60, "hrs": 60, "hour": 60, "hours": 60,
}

// durationMinutes normalizes a spoken wait such as "in 20 minutes" to whole minutes.
//
// requireUnit is what keeps a party size out of this: only an expectation that names a
// unit opens the duration comparison, and only the value being judged against it may
// leave the unit off, where minutes is the reading a pickup window has anyway.
func durationMinutes(s string, requireUnit bool) (int, bool) {
	minutes := 0
	pending := -1
	unit := false
	counted := false
	for _, token := range tokens(s) {
		if durationHedges[token] {
			continue
		}
		if scale, ok := durationUnits[token]; ok {
			if pending < 0 {
				return 0, false
			}
			minutes += pending * scale
			pending = -1
			unit = true
			counted = true
			continue
		}
		if pending >= 0 {
			return 0, false
		}
		value, err := strconv.Atoi(token)
		if err != nil {
			word, known := wordToNum[token]
			if !known {
				return 0, false
			}
			value = word
		}
		pending = value
	}
	if pending >= 0 {
		minutes += pending
		counted = true
	}
	if !counted || (requireUnit && !unit) {
		return 0, false
	}
	return minutes, true
}

func clockValue(s string) (string, bool) {
	s = strings.ToLower(strings.TrimSpace(s))
	m := clockInTextRe.FindStringSubmatch(s)
	if m == nil {
		return "", false
	}
	hour, _ := strconv.Atoi(m[1])
	minute, _ := strconv.Atoi(m[2])
	if hour > 23 || minute > 59 {
		return "", false
	}
	if hour >= 13 {
		hour -= 12
	}
	if hour == 0 {
		hour = 12
	}
	return fmt.Sprintf("%d:%02d", hour, minute), true
}

func valueVariants(value string) []string {
	out := []string{value}
	seen := map[string]bool{value: true}
	add := func(s string) {
		if s == "" || seen[s] {
			return
		}
		seen[s] = true
		out = append(out, s)
	}
	if n, err := strconv.Atoi(value); err == nil && n >= 0 && n <= 20 {
		add(smallWords[n])
	}
	if n, ok := wordToNum[value]; ok {
		add(strconv.Itoa(n))
	}
	if m := timeRe.FindStringSubmatch(value); m != nil {
		h, _ := strconv.Atoi(m[1])
		min, _ := strconv.Atoi(m[2])
		add(fmt.Sprintf("%d:%02d", h, min))
		add(fmt.Sprintf("%02d:%02d", h, min))
		add(fmt.Sprintf("%d %02d", h, min))
		hw := hourWord(h)
		mw := minuteWord(min)
		if hw != "" && mw != "" {
			add(hw + " " + mw)
			if min != 0 {
				add(hw + "-" + mw)
			}
		}
	}
	return out
}

func hourWord(h int) string {
	if h == 0 {
		return "twelve"
	}
	if h > 12 {
		h -= 12
	}
	if h >= 0 && h <= 20 {
		return smallWords[h]
	}
	return ""
}

func minuteWord(m int) string {
	switch {
	case m == 0:
		return "o'clock"
	case m == 30:
		return "thirty"
	case m == 45:
		return "forty five"
	case m >= 0 && m <= 20:
		return smallWords[m]
	default:
		return ""
	}
}

func alnum(s string) string {
	var b strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(unicode.ToLower(r))
		}
	}
	return b.String()
}
