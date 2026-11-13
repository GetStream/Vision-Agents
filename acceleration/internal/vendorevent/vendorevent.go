// Package vendorevent counts the account events a telephony vendor sends the router: a
// port-out, a spend limit, a failed number order. A monitor pages on the counter, so it
// depends on nothing in the router, and every receiver of such events can count with it.
//
// A type a vendor's map does not name is counted under one label that ends in "other", so a
// new or misspelled type adds no series. The labels are "other" (Telnyx), "error.other" (a
// Twilio Monitor alert) and "usage_trigger.other" (a Twilio usage trigger); the kind in front
// of the dot tells a monitor what kind of event it was. A monitor that wants every unnamed
// event of every vendor matches event=~".*other".
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

// DLCCampaignFailed is the event of a campaign a registrar suspended, rejected or failed
// (dlc.Failed): its numbers cannot send A2P texts. The router raises it from the registrar's
// report, not from a vendor's account event, so no vendor map holds it.
const DLCCampaignFailed = "10dlc.campaign_failed"

// Count adds one event. The event is a name from the vendor's map, Other or a constant of
// this package (DLCCampaignFailed), never a value the vendor sent unchecked.
func Count(vendor, event string, level Level) {
	events.WithLabelValues(vendor, event, string(level)).Inc()
}
