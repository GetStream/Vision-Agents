package phone

import (
	"strings"

	"github.com/nyaruka/phonenumbers"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/region"
)

// National is a number's digits after its country code, as a vendor's search takes them, or
// empty when it cannot be parsed.
func National(e164 string) string {
	parsed, err := phonenumbers.Parse(strings.TrimSpace(e164), "")
	if err != nil {
		return ""
	}
	return phonenumbers.GetNationalSignificantNumber(parsed)
}

// Country is the ISO 3166-1 alpha-2 region a number is in, which is what the price of a text
// or a minute to it depends on (region.Of).
func Country(e164 string) string {
	return region.Of(e164)
}
