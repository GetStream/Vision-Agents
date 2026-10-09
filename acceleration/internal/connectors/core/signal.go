package core

import (
	"net/http"
	"time"
)

// Verifier checks one inbound provider request and reads what it says. A request that fails
// verification changes nothing, so the inbound endpoint needs no API auth.
type Verifier interface {
	Name() string
	// Verify checks r and body against m's channel.verifier with secret and returns what the
	// request says. An error means the request is not proven to come from the provider, and
	// the endpoint acts on nothing. A verified request the verifier has no mapping for, or
	// cannot parse, is a zero VerifiedEvent, not an error. body is the raw request body, read
	// once by the endpoint before anything parses it, because a signature is over those exact
	// bytes.
	//
	// m is the connector's manifest, not a ResolvedManifest: a request is verified before
	// anything picks a connection, since one provider app can serve many (ChannelRule.Read).
	// secret is the one channel.verifier.secret names, found by the route the request came
	// in on: the operator's on a connector's route, the customer's provider app's on a
	// provider app's route (T38). The verifier never looks it up, so one verifier serves
	// both routes.
	Verify(r *http.Request, body []byte, m Manifest, secret []byte) (VerifiedEvent, error)
}

// VerifiedEvent is what one verified inbound request says. Each part has one reader: Signals
// go to Resolver.Revoke, Messages go to the channel bridge, Challenge goes back to the
// provider. Any part may be empty. Signals and Messages may each hold more than one: one
// revocation event can name several tokens, and one delivery can batch several messages.
type VerifiedEvent struct {
	Signals  []Signal
	Messages []InboundMessage
	// Challenge is set when the request is a handshake that proves the endpoint is the
	// provider's to call, such as a URL verification. The endpoint answers 200 with it as a
	// text/plain body. A handshake carries no signals and no messages.
	Challenge string
	// Skipped is why each message the request held was not read, as ChannelEvent.Skipped
	// says it. The endpoint logs it at debug, so a skipped message is told apart from a
	// delivery that held none (AI-990 F21, F30).
	Skipped []string
}

// SignalKind is what an inbound event says happened to a grant. Each one ends it: the stored
// credentials no longer work, and only a reconnect gets new ones.
type SignalKind string

const (
	// SignalRevoked is a grant the account or its admin revoked, such as Slack tokens_revoked.
	SignalRevoked SignalKind = "revoked"
	// SignalUninstalled is the provider app removed from the account, which ends every grant
	// it gave, such as Slack app_uninstalled.
	SignalUninstalled SignalKind = "uninstalled"
	// SignalRotated is the account's credentials changed at the provider, such as a password
	// reset that ends its sessions (Google RISC, architecture doc «Incidents and provider
	// quirks»), so the stored ones are no longer accepted.
	SignalRotated SignalKind = "rotated"
)

// signalKinds are the kinds a manifest's channel.signals may name.
var signalKinds = []SignalKind{SignalRevoked, SignalUninstalled, SignalRotated}

// Signal is what a verified request says about the grants of an account. It names the
// account, not a connection, because the provider knows only its own ids.
type Signal struct {
	ConnectorID string
	// Identity is the identity parts the event names, by their names in the manifest's
	// identity, with the values the provider sent. Every part names one account. Fewer parts
	// name every account that has them, such as a workspace uninstall that names the team and
	// no user. They are parts, not a joined account id, so a reader matches each one and never
	// splits an id.
	Identity map[string]string
	Kind     SignalKind
	// At is when the provider says the event happened (SignalRule.At), to the second; zero
	// when it does not say. A provider that retries a delivery sends the same event, so a
	// retry that arrives after the account reconnected still says when the old grant ended,
	// and Resolver.Revoke leaves the new grant alone.
	At time.Time
}

// InboundMessage is one message a person sent on an external thread, named by the provider's
// ids and by a thread key the verifier builds. Like Signal it names no connection or Stream Chat channel: the channel bridge
// maps ProviderUnitID to a connection, ThreadKey to a thread channel and AuthorID to a
// Stream Chat user.
type InboundMessage struct {
	ConnectorID string
	// ProviderUnitID is the customer's own unit at the provider that received the message:
	// a workspace, a team, a bot or a business phone number.
	ProviderUnitID string
	// ThreadKey is the same for every message on one external thread and differs between
	// threads of one provider unit. The verifier builds it from the event; nothing parses it.
	ThreadKey string
	// AuthorID is the provider's id for the person who wrote the message: a user id or a
	// phone number.
	AuthorID string
	// Text is the message text. It may be empty, for a message that is only a file.
	Text string
	// ProviderMessageID is the provider's id for this message: WhatsApp messages[].id, Slack
	// event.ts, Twilio MessageSid. It finds this message's entry in Raw when one delivery
	// batches several, and it is the key for dropping a retried delivery. Slack's ts is
	// unique only within a channel ("the unique (per-channel) timestamp",
	// https://docs.slack.dev/reference/events/message), so a reader that needs one key for
	// a provider unit pairs it with ThreadKey.
	ProviderMessageID string
	// Raw is the verified request body, unchanged. Every message of a batched delivery
	// shares it. It is bytes, not json.RawMessage, because some providers post a form, not
	// JSON. A reader that needs a field the ones above do not hold reads it here.
	Raw []byte
}
