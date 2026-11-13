// Package redact shortens what a log line may say about a person.
package redact

// Number is a phone number as a log carries it: its last four characters, enough to tell
// one line from another without naming whose it is. A number of four characters or fewer
// would be shown whole, so none of it is.
func Number(number string) string {
	if number == "" {
		return ""
	}
	// By rune, since a SIP trunk may hand over a URI rather than digits.
	runes := []rune(number)
	if len(runes) <= 4 {
		return "***"
	}
	return "***" + string(runes[len(runes)-4:])
}
