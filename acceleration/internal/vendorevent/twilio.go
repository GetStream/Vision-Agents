package vendorevent

// VendorTwilio is the vendor label of Twilio's events. It is also the vendor name a 10DLC registrar
// stores (dlc.Registrar.Name()), so changing it orphans stored rows.
const VendorTwilio = "twilio"

// twilioAlertLevels is every Monitor alert error code the router names, with its level. The
// codes are Twilio's error codes: https://www.twilio.com/docs/api/errors.
var twilioAlertLevels = map[string]Level{
	// The account or subaccount is not active, so nothing it sends goes out.
	// https://www.twilio.com/docs/api/errors/10001
	"10001": LevelPage,
	// The account was suspended before a queued message went out.
	// https://www.twilio.com/docs/api/errors/30002
	"30002": LevelPage,
	// The credentials on a request are wrong, expired or cannot reach the resource.
	// https://www.twilio.com/docs/api/errors/20003
	"20003": LevelWarn,
	// A carrier blocked a call for the poor reputation of its caller ID.
	// https://www.twilio.com/docs/api/errors/32017
	"32017": LevelWarn,
}

// twilioUsageLevels is every usage category the router names a usage trigger for, with its
// level. The names are Twilio's usage categories:
// https://www.twilio.com/docs/usage/api/usage-record#usage-categories
var twilioUsageLevels = map[string]Level{
	// The price of all usage: a spend that crossed its trigger.
	"totalprice": LevelPage,
	// Inbound and outbound voice calls.
	"calls": LevelWarn,
}

// TwilioAlert names a Monitor alert for the counter and says its level. errorCode is
// error_code from the alert's Payload. A code the map does not name is one label value,
// error.other, so a new code adds no series.
func TwilioAlert(errorCode string) (string, Level) {
	level, named := twilioAlertLevels[errorCode]
	if !named {
		return "error." + Other, LevelInfo
	}
	return "error." + errorCode, level
}

// TwilioUsageTrigger names a usage-trigger callback for the counter and says its level.
// category is the trigger's UsageCategory. Every trigger is one somebody set to be told
// about, so a category the map does not name, usage_trigger.other, still warns.
func TwilioUsageTrigger(category string) (string, Level) {
	level, named := twilioUsageLevels[category]
	if !named {
		return "usage_trigger." + Other, LevelWarn
	}
	return "usage_trigger." + category, level
}
