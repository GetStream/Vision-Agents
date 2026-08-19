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

var timeRe = regexp.MustCompile(`^(\d{1,2}):(\d{2})$`)

// MatchValue reports whether text contains value, allowing common spoken and
// punctuation variants (six/6, 7:30/seven thirty, phone dashes vs digits).
func MatchValue(text, value string) bool {
	text = strings.ToLower(text)
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" {
		return true
	}
	for _, v := range valueVariants(value) {
		if strings.Contains(text, v) {
			return true
		}
	}
	want := alnum(value)
	if want != "" && want != value && strings.Contains(alnum(text), want) {
		return true
	}
	return false
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
