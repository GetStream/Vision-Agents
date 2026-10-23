package phone

import (
	"strings"

	"github.com/nyaruka/phonenumbers"
)

// Country is the ISO 3166-1 alpha-2 region a number is in, which is what the price of a text
// or a minute to it depends on: "US" and "CA" for +1 by area code, "GB" for +44. It is empty
// for anything that is not a number of a country, such as an iMessage handle or a number on an
// international network (+882, +883), which vendors price apart from any country. A number
// written without its "+", as WhatsApp names its users, is read as though it had one.
func Country(e164 string) string {
	number := strings.TrimSpace(e164)
	if number == "" {
		return ""
	}
	if !strings.HasPrefix(number, "+") {
		if strings.ContainsFunc(number, func(r rune) bool { return r < '0' || r > '9' }) {
			return ""
		}
		number = "+" + number
	}
	parsed, err := phonenumbers.Parse(number, "")
	if err != nil {
		return ""
	}
	region := phonenumbers.GetRegionCodeForNumber(parsed)
	if len(region) != 2 || region == "ZZ" {
		return ""
	}
	return region
}
