package phone

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/nyaruka/phonenumbers"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/region"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
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

// soldIn are the countries numbers are sold in. A number elsewhere waits on documents and a
// carrier's review before it works, while a bought number is taken to work at once.
var soldIn = []string{"US", "CA"}

// sold refuses a country numbers are not sold in, before any vendor is asked.
func sold(country string) error {
	if slices.Contains(soldIn, strings.ToUpper(country)) {
		return nil
	}
	refused := "not in " + strconv.Quote(country)
	if country == "" {
		refused = "and this number is in neither"
	}
	return stack.Wrap(Invalid(fmt.Errorf("phone: numbers are sold only in the US and Canada (%s), %s",
		strings.Join(soldIn, ", "), refused)))
}
