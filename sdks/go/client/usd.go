package client

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// USD is an amount of dollars, held exactly as the millionths of a dollar the router counts
// in. Written the way a time.Duration is:
//
//	limit := 20_000 * client.Dollar
//	price := 3 * client.Cent
//	tiny := 300 * client.Micro // $0.0003, what one short LLM call can cost
//
// There is no constructor from a float on purpose: 0.1 + 0.2 is not a price. ParseUSD reads
// one written out, such as "12.50". What the router answers is micros too, so
// client.USD(limit.LimitMicros) is the limit in dollars.
type USD int64

const (
	// Micro is a millionth of a dollar, the router's unit.
	Micro USD = 1
	// Cent is a hundredth of a dollar.
	Cent USD = 10_000
	// Dollar is one dollar.
	Dollar USD = 1_000_000
)

// Micros is the amount in millionths of a dollar, as the router's fields want it.
func (u USD) Micros() int64 { return int64(u) }

// String is the amount as dollars, with cents always and finer digits only when it has them:
// $20000.00, $12.50, $0.0003. ParseUSD reads it back.
func (u USD) String() string {
	sign := ""
	magnitude := uint64(u)
	if u < 0 {
		sign = "-"
		magnitude = uint64(-(u + 1)) + 1
	}
	whole, fraction := magnitude/uint64(Dollar), magnitude%uint64(Dollar)
	digits := strings.TrimRight(fmt.Sprintf("%06d", fraction), "0")
	if len(digits) < 2 {
		digits += strings.Repeat("0", 2-len(digits))
	}
	return fmt.Sprintf("%s$%d.%s", sign, whole, digits)
}

// ParseUSD reads an amount of dollars written out: "12", "12.5", "$12.50", "-3.25" or
// "0.000300". It refuses anything finer than a millionth of a dollar, which the router cannot
// count, and anything that is not plain digits: no thousands separators, no exponent.
func ParseUSD(written string) (USD, error) {
	text := strings.TrimSpace(written)
	negative := strings.HasPrefix(text, "-")
	text = strings.TrimPrefix(text, "-")
	text = strings.TrimPrefix(text, "$")
	whole, fraction, dotted := strings.Cut(text, ".")
	if whole == "" && fraction == "" || !digitsOnly(whole) || !digitsOnly(fraction) || dotted && fraction == "" {
		return 0, fmt.Errorf("client: %q is not an amount of dollars", written)
	}
	if len(fraction) > 6 {
		if strings.Trim(fraction[6:], "0") != "" {
			return 0, fmt.Errorf("client: %q is finer than a millionth of a dollar", written)
		}
		fraction = fraction[:6]
	}
	var dollars, micros uint64
	var err error
	if whole != "" {
		if dollars, err = strconv.ParseUint(whole, 10, 64); err != nil {
			return 0, fmt.Errorf("client: %q is too large an amount", written)
		}
	}
	if fraction != "" {
		micros, _ = strconv.ParseUint(fraction+strings.Repeat("0", 6-len(fraction)), 10, 64)
	}
	if dollars > (math.MaxInt64-micros)/uint64(Dollar) {
		return 0, fmt.Errorf("client: %q is too large an amount", written)
	}
	amount := USD(dollars*uint64(Dollar) + micros)
	if negative {
		amount = -amount
	}
	return amount, nil
}

func digitsOnly(text string) bool {
	for _, r := range text {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}
