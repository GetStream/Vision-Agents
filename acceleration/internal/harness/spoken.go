package harness

// spokenKind is what a word of a transcript says about a number, for the words that say anything.
type spokenKind uint8

const (
	spokenOther spokenKind = iota
	// spokenDigit is one of zero to nine, as a word or as a single numeral.
	spokenDigit
	// spokenOh is how a zero is said inside a number: "oh five", "five oh one".
	spokenOh
	// spokenTeen is ten to nineteen. spokenTens is twenty to fifty. spokenTwoDigits is a numeral
	// of two digits.
	spokenTeen
	spokenTens
	spokenTwoDigits
	// spokenMultiple is "double" or "triple", which say the digit after them more than once.
	spokenMultiple
	spokenHalf
	spokenQuarter
	// spokenPast is "past", "after" or "to", which join a fraction of an hour to the hour.
	spokenPast
	spokenOClock
	spokenClockWord
	spokenMeridiem
	// spokenAorP and spokenM are the letters of "a.m." and "p.m." when a transcriber spells them.
	spokenAorP
	spokenM
)

// spokenWord is a word of a transcript read as a number, if it is one.
type spokenWord struct {
	kind  spokenKind
	value int
}

// longestSpokenWord is the length of the longest word that is read as a number, "seventeen".
const longestSpokenWord = 9

// spokenWordOf reads one word. It lower-cases into a buffer on the stack and so allocates
// nothing, which matters because it runs on every caller turn.
func spokenWordOf(word string) spokenWord {
	if len(word) == 0 || len(word) > longestSpokenWord {
		return spokenWord{}
	}
	if word[0] >= '0' && word[0] <= '9' {
		return spokenNumeral(word)
	}
	var lowered [longestSpokenWord]byte
	for i := range len(word) {
		letter := word[i]
		if letter >= 'A' && letter <= 'Z' {
			letter += 'a' - 'A'
		}
		lowered[i] = letter
	}
	switch string(lowered[:len(word)]) {
	case "zero":
		return spokenWord{spokenDigit, 0}
	case "one":
		return spokenWord{spokenDigit, 1}
	case "two":
		return spokenWord{spokenDigit, 2}
	case "three":
		return spokenWord{spokenDigit, 3}
	case "four":
		return spokenWord{spokenDigit, 4}
	case "five":
		return spokenWord{spokenDigit, 5}
	case "six":
		return spokenWord{spokenDigit, 6}
	case "seven":
		return spokenWord{spokenDigit, 7}
	case "eight":
		return spokenWord{spokenDigit, 8}
	case "nine":
		return spokenWord{spokenDigit, 9}
	case "oh", "o":
		return spokenWord{spokenOh, 0}
	case "ten":
		return spokenWord{spokenTeen, 10}
	case "eleven":
		return spokenWord{spokenTeen, 11}
	case "twelve":
		return spokenWord{spokenTeen, 12}
	case "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen":
		return spokenWord{spokenTeen, 13}
	case "twenty", "thirty", "forty", "fifty":
		return spokenWord{spokenTens, 20}
	case "double", "triple":
		return spokenWord{spokenMultiple, 0}
	case "half":
		return spokenWord{spokenHalf, 0}
	case "quarter":
		return spokenWord{spokenQuarter, 0}
	case "past", "after", "to":
		return spokenWord{spokenPast, 0}
	case "o'clock", "oclock":
		return spokenWord{spokenOClock, 0}
	case "clock":
		return spokenWord{spokenClockWord, 0}
	case "am", "pm":
		return spokenWord{spokenMeridiem, 0}
	case "a", "p":
		return spokenWord{spokenAorP, 0}
	case "m":
		return spokenWord{spokenM, 0}
	}
	return spokenWord{}
}

// spokenNumeral reads a word that starts with a digit: a single numeral is a digit and one of
// two is a number of two digits, which a clock time may be made of. Anything longer is not read
// here.
func spokenNumeral(word string) spokenWord {
	for i := range len(word) {
		if word[i] < '0' || word[i] > '9' {
			return spokenWord{}
		}
	}
	switch len(word) {
	case 1:
		return spokenWord{spokenDigit, int(word[0] - '0')}
	case 2:
		return spokenWord{spokenTwoDigits, int(word[0]-'0')*10 + int(word[1]-'0')}
	}
	return spokenWord{}
}

// isHour reports whether a word can be the hour of a clock time: one to twelve.
func (w spokenWord) isHour() bool {
	switch w.kind {
	case spokenDigit:
		return w.value >= 1
	case spokenTeen, spokenTwoDigits:
		return w.value >= 10 && w.value <= 12
	}
	return false
}

// nextWord returns the next word of the text from the given offset, and where the one after
// begins. Words are made of letters, digits and an apostrophe, so anything else, punctuation
// and hyphens included, ends one: "forty-five" is two, and "o'clock" is one.
func nextWord(text string, from int) (word string, next int) {
	start := from
	for start < len(text) && !wordByte(text[start]) {
		start++
	}
	end := start
	for end < len(text) && wordByte(text[end]) {
		end++
	}
	return text[start:end], end
}

func wordByte(b byte) bool {
	return b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' || b >= '0' && b <= '9' || b == '\''
}

// minimumSpokenDigits is how many digits said one after another are taken for an identifier
// rather than a quantity, and minimumDistinctDigits how many of them must be said as digits
// rather than as "oh".
const (
	minimumSpokenDigits   = 4
	minimumDistinctDigits = 3
)

// spokenNumbers reports whether text spells out a clock time or a run of digits: "seven
// thirty", "half past seven", "quarter to eight", "seven o'clock", "seven pm", "oh five", or
// "five one two five five five zero one four two". A transcript writes the digits the caller
// said as words as often as it writes numerals, so what the numerals would have shown a
// pattern is read here from the words. It allocates nothing.
func spokenNumbers(text string) bool {
	var (
		// hour says the last word could be the hour of a clock time, and ohAfterHour that the one
		// before it was and this one was said as "oh".
		hour, ohAfterHour bool
		// fraction is 1 after a word that says part of an hour, "half", "quarter", "ten" or
		// "twenty five", and 2 once "past" or "to" followed it. tens says the last word was
		// twenty, thirty, forty or fifty, which a digit may follow.
		fraction int
		tens     bool
		// meridiem says the last word was the "a" or "p" of "a.m." or "p.m." after an hour.
		meridiem bool
		// digits and distinct count the run of digits being said, and multiple is how many times
		// the next digit is said when "double" or "triple" came before it.
		digits, distinct, multiple int
	)
	for offset := 0; offset < len(text); {
		var word string
		word, offset = nextWord(text, offset)
		if word == "" {
			break
		}
		current := spokenWordOf(word)

		switch {
		case current.isHour() && fraction == 2:
			return true
		case hour && (current.kind == spokenOClock || current.kind == spokenMeridiem):
			return true
		case current.kind == spokenM && meridiem:
			return true
		case ohAfterHour && (current.kind == spokenClockWord || current.kind == spokenDigit && current.value >= 1):
			return true
		case hour && (current.kind == spokenTeen || current.kind == spokenTens ||
			current.kind == spokenTwoDigits && current.value < 60):
			return true
		}

		ohAfterHour = hour && current.kind == spokenOh
		meridiem = hour && current.kind == spokenAorP
		switch {
		case current.kind == spokenHalf, current.kind == spokenQuarter, current.kind == spokenTeen,
			current.kind == spokenTens, current.kind == spokenDigit && current.value == 5:
			fraction = 1
		case current.kind == spokenDigit && tens && fraction == 1:
		case current.kind == spokenPast && fraction == 1:
			fraction = 2
		default:
			fraction = 0
		}
		tens = current.kind == spokenTens
		hour = current.isHour()

		switch current.kind {
		case spokenDigit:
			said := max(multiple, 1)
			digits, distinct, multiple = digits+said, distinct+said, 0
		case spokenOh:
			digits, multiple = digits+max(multiple, 1), 0
		case spokenMultiple:
			multiple = 2
			if word[0] == 't' || word[0] == 'T' {
				multiple = 3
			}
		default:
			digits, distinct, multiple = 0, 0, 0
		}
		if digits >= minimumSpokenDigits && distinct >= minimumDistinctDigits {
			return true
		}
	}
	return false
}
