package core

import "net/http"

// Verifier checks one inbound provider request and reads what it says. A request that fails
// verification changes nothing, so the inbound endpoint needs no API auth.
type Verifier interface {
	Name() string
	// Verify checks r and body against m and returns what the request says. An error means
	// the request is not proven to come from the provider, and the endpoint acts on nothing.
	// A verified request the verifier has no mapping for is a zero VerifiedEvent, not an
	// error. body is the raw request body, read once by the endpoint before anything parses
	// it, because a signature is over those exact bytes.
	Verify(r *http.Request, body []byte, m ResolvedManifest) (VerifiedEvent, error)
}

// VerifiedEvent is what one verified inbound request says. Each part has one reader: Signals
// go to Resolver.Invalidate, Messages go to the channel bridge, Challenge goes back to the
// provider. Any part may be empty, and one request may carry more than one of each: one
// revocation event can name several tokens, and one delivery can batch several messages.
type VerifiedEvent struct {
	Signals  []Signal
	Messages []InboundMessage
	// Challenge is set when the request is a handshake that proves the endpoint is the
	// provider's to call, such as a URL verification. The endpoint answers 200 with it as a
	// text/plain body. A handshake carries no signals and no messages.
	Challenge string
}

// SignalKind is what an inbound event says happened to a grant.
type SignalKind string

const (
	SignalRevoked     SignalKind = "revoked"
	SignalUninstalled SignalKind = "uninstalled"
	SignalRotated     SignalKind = "rotated"
)

// Signal is a verified event about the grants of one account. It names the account, not a
// connection, because the provider knows only its own ids.
type Signal struct {
	ConnectorID string
	AccountID   string
	Kind        SignalKind
}

// InboundMessage is one message a person sent on an external thread, as the provider's ids
// name it. Like Signal it names no connection or Stream Chat channel: the channel bridge
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
	// Raw is the verified request body, unchanged. Every message of a batched delivery
	// shares it. It is bytes, not json.RawMessage, because some providers post a form, not
	// JSON. A reader that needs a field the ones above do not hold reads it here.
	Raw []byte
}
