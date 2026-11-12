// Package vendorevent counts the account events a telephony vendor sends the router: a
// port-out, a spend limit, a failed number order. A monitor pages on the counter, so it
// depends on nothing in the router, and every receiver of such events can count with it.
package vendorevent

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Level is how urgently somebody should look at an event.
type Level string

const (
	// LevelPage is an event to wake somebody for: numbers or sending are being lost.
	LevelPage Level = "page"
	// LevelWarn is an event to look at in working hours.
	LevelWarn Level = "warn"
	// LevelInfo is an event that needs nobody.
	LevelInfo Level = "info"
)

// Other is the event label of every event type a vendor's map does not name, so a
// vendor's new or misspelled type cannot add a series.
const Other = "other"

// The vendor's name, the event and the level are the only labels: no organization, app,
// number or account id, so the series stay few and say nothing about a customer.
var events = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "vendor_account_events_total",
	Help: "Account events a telephony vendor sent with a valid signature, by vendor, event type " +
		"(other for a type the router does not name) and level (page, warn or info).",
}, []string{"vendor", "event", "level"})

// Count adds one event. The event is a name from the vendor's map or Other, never a value
// the vendor sent unchecked.
func Count(vendor, event string, level Level) {
	events.WithLabelValues(vendor, event, string(level)).Inc()
}
