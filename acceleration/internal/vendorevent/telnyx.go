package vendorevent

// VendorTelnyx is the vendor label of Telnyx's events.
const VendorTelnyx = "telnyx"

// telnyxLevels is every Telnyx event type the router names, with its level. A type Telnyx
// documents with no string for its event_type (a deleted number, a requirement group's
// status change) is not here: it is counted as Other until a delivery shows its name.
var telnyxLevels = map[string]Level{
	// A port-out takes a number off the account, and the agent on it stops answering.
	// https://developers.telnyx.com/docs/numbers/porting/port-out-notifications.md
	"portout.status_changed":   LevelPage,
	"portout.new_comment":      LevelPage,
	"portout.foc_date_changed": LevelPage,
	// Port-ins bring numbers in, which nobody needs to act on at once.
	// https://developers.telnyx.com/docs/numbers/porting/port-in-notifications.md
	"porting_order.status_changed":    LevelInfo,
	"porting_order.new_comment":       LevelInfo,
	"porting_order.split":             LevelInfo,
	"porting_order.deleted":           LevelInfo,
	"porting_order.messaging_changed": LevelInfo,
	// Telnyx refuses every message of the profile once its spend limit is reached.
	// https://developers.telnyx.com/docs/messaging/messages/configurable-spend-limits.md
	"messaging-profile.spend-limit-reached": LevelPage,
	// Its level depends on the order's status: see Telnyx.
	// https://developers.telnyx.com/data/webhook-events.json (numberOrderStatusUpdate)
	telnyxNumberOrder: LevelInfo,
}

// telnyxNumberOrder is the event Telnyx sends when a number order ends.
const telnyxNumberOrder = "number_order.complete"

// telnyxOrderSucceeded is the status of a number order whose numbers were all bought. The
// others are pending and failure: NumberOrderWebhookPayload.status in
// https://raw.githubusercontent.com/team-telnyx/openapi/master/openapi/spec3.json
const telnyxOrderSucceeded = "success"

// Telnyx names a Telnyx event for the counter and says its level. orderStatus is
// data.payload.status, read only for a number order, whose numbers are missing unless it
// succeeded.
func Telnyx(eventType, orderStatus string) (string, Level) {
	level, named := telnyxLevels[eventType]
	if !named {
		return Other, LevelInfo
	}
	if eventType == telnyxNumberOrder && orderStatus != telnyxOrderSucceeded {
		return eventType, LevelWarn
	}
	return eventType, level
}
