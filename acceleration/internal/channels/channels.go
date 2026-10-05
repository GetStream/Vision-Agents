// Package channels carries a conversation over WhatsApp, text messages and iMessage.
//
// An agent config names the numbers it answers on, and the credentials for the providers
// carrying them are connected once for the app. A message that arrives is verified against
// the account it was delivered to, the sender is looked up as an end user, and the agent
// answers in that person's own conversation: the same transcript the dashboard shows, so a
// conversation that started in a browser carries on by text.
//
// What a provider has to do is small and the same three things every time: say whether a
// delivery really came from it, read the messages out of it, and send a reply back.
package channels

import (
	"context"
	"errors"
	"net/http"
	"time"
)

// HookPath is where providers deliver. The account's token follows it, which is what tells
// the provider apart from anybody who guessed the route.
const HookPath = "/v1/agents/channels/hooks/"

// SignatureTolerance is how old a signed delivery may be. A recording of one replayed later
// is refused, which is the point of the providers signing a timestamp alongside the body.
const SignatureTolerance = 5 * time.Minute

// MaxDeliveryBytes caps a delivery. The biggest thing any of these providers sends is a
// message with media described by URL, which is nowhere near this.
const MaxDeliveryBytes = 256 << 10

// ErrUnsigned is a delivery that did not come from the provider: the signature is missing,
// does not match, or is older than SignatureTolerance.
var ErrUnsigned = errors.New("channels: that delivery is not signed by the provider")

// Kind is a channel a conversation can be carried over.
//
// It is both what a config names and which provider carries it, because the two go together:
// a provider is chosen by what it can reach, not the other way round.
type Kind string

const (
	// WhatsApp is Meta's Cloud API.
	WhatsApp Kind = "whatsapp"
	// SMS is text messages, through Telnyx.
	SMS Kind = "sms"
	// IMessage is Apple's, through Linq, which falls back to RCS or SMS when Apple cannot
	// deliver.
	IMessage Kind = "imessage"
)

// Kinds are the channels there are, in the order they are shown.
var Kinds = []Kind{WhatsApp, SMS, IMessage}

// Valid reports whether this is a channel something carries.
func (k Kind) Valid() bool {
	for _, known := range Kinds {
		if k == known {
			return true
		}
	}
	return false
}

// Account is one app's line on one channel.
//
// The fields are named for what they do rather than for what each provider calls them,
// because all three need the same four things under different names: Token is WhatsApp's
// access token, Telnyx's API key and Linq's API key, and Signing is Meta's app secret,
// Telnyx's public key and Linq's webhook secret.
type Account struct {
	Kind Kind
	// E164 is the number people write to.
	E164 string
	// AccountID is the provider's own id for the line, such as WhatsApp's phone number id.
	// Not a secret: every delivery carries it too.
	AccountID string
	// Token authenticates this router when it sends.
	Token string
	// Signing is what the provider signs deliveries with.
	Signing string
	// Challenge is what a provider's webhook setup echoes back to prove the URL is ours.
	// WhatsApp alone has one.
	Challenge string
}

// Secrets are the credentials of an account, which are what is sealed in the database.
// Everything else about an account is either the app's own choice or public.
type Secrets struct {
	Token     string `json:"token,omitempty"`
	Signing   string `json:"signing,omitempty"`
	Challenge string `json:"challenge,omitempty"`
}

// Message is one message somebody sent an agent.
type Message struct {
	Kind Kind
	// ID is the provider's own, which a retried delivery repeats. It is what keeps one
	// message from being answered twice.
	ID string
	// From is who wrote it: an E.164 number, or the provider's own handle for them.
	From string
	// To is the line it was written to, which is the account's number.
	To string
	// Thread is where a reply is sent: the sender for WhatsApp and SMS, and the chat for
	// iMessage, where a reply is posted to the conversation rather than to a person.
	Thread string
	// Name is what the sender calls themselves, when the provider says.
	Name string
	Text string
}

// Reply is what the agent said, as a provider has to send it.
type Reply struct {
	Text string
	// Files are what the agent made: a render, a document. They are sent by URL, so the
	// provider fetches them rather than this router uploading them.
	Files []File
	// Link is a login the person has to finish before the agent can go on. A channel with
	// a button of its own shows one; the rest put the URL in the text.
	Link *Link
}

// File is something the agent made, where the provider can fetch it.
type File struct {
	Name     string
	MimeType string
	URL      string
}

// Link is somewhere the person has to go, such as a plugin login.
type Link struct {
	Text string
	URL  string
}

// Empty reports whether there is nothing to send.
func (r Reply) Empty() bool {
	return r.Text == "" && len(r.Files) == 0 && r.Link == nil
}

// Provider carries one kind of channel.
//
// Verify comes first and the rest only run on what it passed: a delivery is a request from
// the open internet, and the body is not read as a message until it is known to be one.
type Provider interface {
	// Kind is the channel it carries.
	Kind() Kind
	// Verify reports whether a delivery was signed by the provider, at a time close enough
	// to now that it is not a replay.
	Verify(account Account, header http.Header, body []byte, now time.Time) error
	// Parse reads the messages out of a delivery. A delivery carrying none -- a delivery
	// report, a reaction -- yields none rather than failing.
	Parse(body []byte) ([]Message, error)
	// Send writes a reply back to the thread a message came from.
	Send(ctx context.Context, account Account, thread string, reply Reply) error
}

// For is the provider carrying a channel.
func For(kind Kind, client *http.Client) (Provider, bool) {
	if client == nil {
		client = http.DefaultClient
	}
	switch kind {
	case WhatsApp:
		return &whatsAppProvider{client: client}, true
	case SMS:
		return &telnyxProvider{client: client}, true
	case IMessage:
		return &linqProvider{client: client}, true
	}
	return nil, false
}

// Telnyx is the SMS provider with the one call no other provider has: pointing a number
// bought here at this router, so nobody has to open a dashboard to finish connecting it.
func Telnyx(client *http.Client) (*telnyxProvider, bool) {
	provider, ok := For(SMS, client)
	if !ok {
		return nil, false
	}
	telnyx, ok := provider.(*telnyxProvider)
	return telnyx, ok
}

// fresh reports whether a signed timestamp is close enough to now to be acted on.
func fresh(signed, now time.Time) bool {
	gap := now.Sub(signed)
	if gap < 0 {
		gap = -gap
	}
	return gap <= SignatureTolerance
}
