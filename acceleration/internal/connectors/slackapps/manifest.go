package slackapps

import (
	"fmt"
	"net/netip"
	"net/url"
	"slices"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Limits from the app manifest reference (https://docs.slack.dev/reference/app-manifest,
// opened October 6, 2026).
const (
	// maxNameLength: display_information.name, «Maximum length is 35 characters».
	maxNameLength = 35
	// maxDescriptionLength: display_information.description, «Maximum length is 140
	// characters».
	maxDescriptionLength = 140
	// maxAllowedIPAddressRanges: settings.allowed_ip_address_ranges, «Maximum 10 items».
	maxAllowedIPAddressRanges = 10
)

// slackHost is where Slack's OAuth endpoints are: https://slack.com/oauth/v2/authorize for a
// bot token and https://slack.com/oauth/v2_user/authorize for a user token
// (https://docs.slack.dev/authentication/installing-with-oauth,
// https://docs.slack.dev/ai/slack-mcp-server/).
const slackHost = "slack.com"

// userAuthorizePath is the authorize endpoint of the user-token flow the Slack MCP server
// takes («Authorization: https://slack.com/oauth/v2_user/authorize»,
// https://docs.slack.dev/ai/slack-mcp-server/). A connector that authorizes there asks for
// user scopes; any other Slack authorize endpoint is the bot flow.
const userAuthorizePath = "/oauth/v2_user/authorize"

// eventTypePath is where a Slack Events API delivery names its event's type: the event
// object's type, inside the envelope (https://docs.slack.dev/apis/events-api/). A channel
// rule matching on it names an event the app subscribes to.
const eventTypePath = "$.event.type"

// automaticEvents are delivered without a subscription: app_uninstalled «is an
// automatically-delivered event» (https://docs.slack.dev/reference/events/app_uninstalled),
// so naming it in the manifest is not needed.
var automaticEvents = []string{"app_uninstalled"}

// Manifest is the part of a Slack app manifest the router writes, in the JSON shape
// apps.manifest.create and .update take («A JSON app manifest encoded as a string»). Field
// names and meanings are the app manifest reference's.
type Manifest struct {
	DisplayInformation DisplayInformation `json:"display_information"`
	Features           *Features          `json:"features,omitempty"`
	OAuthConfig        OAuthConfig        `json:"oauth_config"`
	Settings           Settings           `json:"settings"`
}

// DisplayInformation is what the workspace shows for the app.
type DisplayInformation struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

// Features holds the bot user of an app that asks for bot scopes.
type Features struct {
	BotUser *BotUser `json:"bot_user,omitempty"`
}

// BotUser is the app's bot user.
type BotUser struct {
	DisplayName string `json:"display_name"`
}

// OAuthConfig is where the install sends the browser back to and the scopes it asks for.
type OAuthConfig struct {
	RedirectURLs []string `json:"redirect_urls,omitempty"`
	Scopes       Scopes   `json:"scopes"`
}

// Scopes are the bot and user scopes requested on install.
type Scopes struct {
	Bot  []string `json:"bot,omitempty"`
	User []string `json:"user,omitempty"`
}

// Settings are the app's event subscriptions and token settings.
type Settings struct {
	EventSubscriptions     *EventSubscriptions `json:"event_subscriptions,omitempty"`
	AllowedIPAddressRanges []string            `json:"allowed_ip_address_ranges,omitempty"`
	OrgDeployEnabled       bool                `json:"org_deploy_enabled"`
	SocketModeEnabled      bool                `json:"socket_mode_enabled"`
	TokenRotationEnabled   bool                `json:"token_rotation_enabled"`
}

// EventSubscriptions is the Events API request URL and the events sent to it.
type EventSubscriptions struct {
	RequestURL string   `json:"request_url"`
	BotEvents  []string `json:"bot_events,omitempty"`
	UserEvents []string `json:"user_events,omitempty"`
}

// Template is what the router decides about one customer's app. The scopes and the events
// come from the connector's manifest.
type Template struct {
	// Name is the app's name in the customer's workspace, and its bot user's.
	Name string
	// RedirectURL is the router's OAuth callback, where the install sends the browser back.
	RedirectURL string
	// RequestURL is the router's events URL for this app. Empty leaves out event
	// subscriptions: the URL names the app id, which only apps.manifest.create returns.
	RequestURL string
	// AllowedIPAddressRanges restricts the app's tokens to these addresses or CIDR ranges,
	// at most maxAllowedIPAddressRanges. Empty restricts nothing.
	AllowedIPAddressRanges []string
}

// Serves reports whether the connector authorizes at Slack, so an app this package creates
// is one its consent can use.
func Serves(connector core.Manifest) bool {
	authorize, err := url.Parse(connector.Endpoints["authorize"])
	return err == nil && authorize.Scheme == "https" && authorize.Host == slackHost
}

// ManifestFor is the Slack app manifest of the customer's app for connector: the connector's
// scopes as user scopes when it authorizes at the user-token endpoint and as bot scopes (with
// a bot user) otherwise, the events its signal rules match on, token rotation on, and the
// template's name, URLs and IP ranges.
func ManifestFor(connector core.Manifest, template Template) (Manifest, error) {
	if !Serves(connector) {
		return Manifest{}, stack.Wrap(fmt.Errorf("slackapps: %s does not authorize at %s", connector.ID, slackHost))
	}
	if err := checkTemplate(template); err != nil {
		return Manifest{}, err
	}
	manifest := Manifest{
		DisplayInformation: DisplayInformation{Name: template.Name},
		OAuthConfig:        OAuthConfig{RedirectURLs: []string{template.RedirectURL}},
		Settings: Settings{
			AllowedIPAddressRanges: template.AllowedIPAddressRanges,
			// «Without token rotation, a Slack access token never expires», and once on, «may
			// not be turned off» (https://docs.slack.dev/authentication/using-token-rotation):
			// every app the router creates has short-lived tokens it refreshes
			// (connectors/planning, channels.md, «Rules for token export»).
			TokenRotationEnabled: true,
		},
	}
	if utf8.RuneCountInString(connector.Description) <= maxDescriptionLength {
		manifest.DisplayInformation.Description = connector.Description
	}
	authorize, _ := url.Parse(connector.Endpoints["authorize"])
	user := authorize.Path == userAuthorizePath
	scopes := slices.Clone(connector.Scopes.List)
	if user {
		manifest.OAuthConfig.Scopes.User = scopes
	} else {
		manifest.OAuthConfig.Scopes.Bot = scopes
		manifest.Features = &Features{BotUser: &BotUser{DisplayName: template.Name}}
	}
	if template.RequestURL != "" {
		subscriptions := &EventSubscriptions{RequestURL: template.RequestURL}
		if user {
			subscriptions.UserEvents = eventTypes(connector)
		} else {
			subscriptions.BotEvents = eventTypes(connector)
		}
		manifest.Settings.EventSubscriptions = subscriptions
	}
	return manifest, nil
}

// checkTemplate holds the template to the manifest reference's limits, so a request Slack
// would refuse is refused before any call.
func checkTemplate(template Template) error {
	if template.Name == "" || utf8.RuneCountInString(template.Name) > maxNameLength {
		return stack.Wrap(fmt.Errorf("slackapps: the app's name is 1 to %d characters", maxNameLength))
	}
	// settings.event_subscriptions.request_url is «the full https URL», and a redirect URL
	// is where an https consent returns to (RFC 6749 section 3.1.2.1).
	addresses := []string{template.RedirectURL}
	if template.RequestURL != "" {
		addresses = append(addresses, template.RequestURL)
	}
	for _, address := range addresses {
		parsed, err := url.Parse(address)
		if err != nil || parsed.Scheme != "https" || parsed.Host == "" {
			return stack.Wrap(fmt.Errorf("slackapps: %q is not an https URL", address))
		}
	}
	if len(template.AllowedIPAddressRanges) > maxAllowedIPAddressRanges {
		return stack.Wrap(fmt.Errorf("slackapps: at most %d allowed IP address ranges", maxAllowedIPAddressRanges))
	}
	for _, allowed := range template.AllowedIPAddressRanges {
		if _, err := netip.ParsePrefix(allowed); err == nil {
			continue
		}
		if _, err := netip.ParseAddr(allowed); err != nil {
			return stack.Wrap(fmt.Errorf("slackapps: %q is not an IP address or a CIDR range", allowed))
		}
	}
	return nil
}

// eventTypes are the event types the connector's signal rules match on, in their order, less
// the ones Slack delivers without a subscription. A signal's event type is also its
// subscription name (tokens_revoked, https://docs.slack.dev/reference/events/tokens_revoked).
// A message rule's is not: a message arrives as event.type message, but an app subscribes to
// message.channels, message.im and the rest, one per kind of conversation
// (https://docs.slack.dev/reference/events/message.im), which the channel block does not say.
// So messages are not subscribed to here.
func eventTypes(connector core.Manifest) []string {
	if connector.Channel == nil {
		return nil
	}
	var types []string
	for _, signal := range connector.Channel.Signals {
		event, ok := signal.Match[eventTypePath]
		if ok && !slices.Contains(types, event) && !slices.Contains(automaticEvents, event) {
			types = append(types, event)
		}
	}
	return types
}
