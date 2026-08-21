package scenario

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
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
	return alnum(got) == alnum(want)
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
