package agent

import (
	"regexp"
	"strconv"
	"strings"
)

var (
	spokenDigits = map[string]int{
		"zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
		"six": 6, "seven": 7, "eight": 8, "nine": 9,
	}
	spokenTeens = map[string]int{
		"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
		"sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
	}
	spokenTens = map[string]int{
		"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
		"eighty": 80, "ninety": 90,
	}
)

// digitsFor rewrites each run of number words in text as digits. A run of one word is only
// rewritten when it is the whole value or opens a name, as a house number opens an address:
// elsewhere a lone "one" is as likely a word.
func digitsFor(text string) string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return text
	}
	var out []string
	changed := false
	for start := 0; start < len(words); {
		end := start
		for end < len(words) && isNumberWord(words[end]) {
			end++
			if end < len(words) && strings.HasSuffix(words[end-1], ",") && end-start == 1 {
				break
			}
		}
		if end == start {
			out = append(out, words[start])
			start++
			continue
		}
		run := words[start:end]
		opens := start == 0 || strings.HasSuffix(words[start-1], ",")
		opensName := opens && (end == len(words) || startsUpper(words[end]))
		if len(run) > 1 || opensName {
			if digits, ok := runDigits(run); ok {
				out = append(out, digits)
				changed = true
				start = end
				continue
			}
		}
		out = append(out, run...)
		start = end
	}
	if !changed {
		return text
	}
	return strings.Join(out, " ")
}

func startsUpper(word string) bool {
	return word != "" && word[0] >= 'A' && word[0] <= 'Z'
}

func isNumberWord(word string) bool {
	word = strings.ToLower(strings.Trim(word, ",.-"))
	if word == "" {
		return false
	}
	_, digit := spokenDigits[word]
	_, teen := spokenTeens[word]
	_, ten := spokenTens[word]
	return digit || teen || ten
}

// runDigits writes a run of number words as the digits said: each digit word is one digit,
// a teen is two, and a tens word takes the digit after it ("seventy one" is 71). A lone "o"
// or "oh" is not a number.
func runDigits(run []string) (string, bool) {
	var digits strings.Builder
	for index := 0; index < len(run); index++ {
		word := strings.ToLower(strings.Trim(run[index], ",.-"))
		if value, ok := spokenTens[word]; ok {
			if index+1 < len(run) {
				next := strings.ToLower(strings.Trim(run[index+1], ",.-"))
				if digit, ok := spokenDigits[next]; ok && digit > 0 && next != "o" && next != "oh" {
					digits.WriteString(strconv.Itoa(value + digit))
					index++
					continue
				}
			}
			digits.WriteString(strconv.Itoa(value))
			continue
		}
		if value, ok := spokenTeens[word]; ok {
			digits.WriteString(strconv.Itoa(value))
			continue
		}
		if (word == "o" || word == "oh") && len(run) == 1 {
			return "", false
		}
		digits.WriteString(strconv.Itoa(spokenDigits[word]))
	}
	return digits.String(), true
}

// spokenSeparator is the name of a character a caller reads out between the parts of an ID
// or a number: "kit underscore moran", "five five five dash zero one nine".
var spokenSeparator = regexp.MustCompile(`(?i)\s+\b(underscore|dash|hyphen)\b\s+`)

// withSpokenSeparators writes the separators a caller named as the characters they named,
// joining the parts either side: after digitsFor, "kit underscore moran underscore 4096" is
// kit_moran_4096 and "555 dash 019 dash 4416" is 555-019-4416, one value each, as written.
func withSpokenSeparators(text string) string {
	return spokenSeparator.ReplaceAllStringFunc(text, func(named string) string {
		if strings.EqualFold(strings.TrimSpace(named), "underscore") {
			return "_"
		}
		return "-"
	})
}
