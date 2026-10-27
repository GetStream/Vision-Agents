package api

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// How an inbound connector event ended (receiveEvent), the outcome of connectorEvents.
const (
	// eventHandled: verified, and a signal in it was acted on or an agent answers a message.
	eventHandled = "handled"
	// eventUnhandled: verified and answered 200, with nothing in it the router acted on: no
	// message, every message skipped, or none an agent answers.
	eventUnhandled = "unhandled"
	// eventRefused: answered 401 or 404, the connector known: unverified, or nothing here
	// verifies its events.
	eventRefused = "refused"
	// eventFailed: answered 500, so the provider delivers it again.
	eventFailed = "failed"
	// eventChallenge: a verified handshake, answered with its challenge.
	eventChallenge = "challenge"
)

// connectorEvents counts only events whose connector the router has a definition of: the id a
// request's path names before that is the sender's choice, without a bound.
var connectorEvents = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "connector_events_total",
	Help:      "Inbound connector events, such as a Slack bot's message, by connector and how they ended: handled, unhandled, refused, failed or challenge (api receiveEvent)",
}, []string{"connector", "outcome"})

// countEvent counts one inbound event of connector id.
func countEvent(id, outcome string) {
	connectorEvents.WithLabelValues(store.ConnectorLabel(id), outcome).Inc()
}
