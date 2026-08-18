// Package phone gives an agent a telephone number.
//
// Eleven vendors sell numbers and they all sell the same thing, so the contract here is
// the small part they agree on: find a number, buy it, point it at the bridge, dial out,
// give it back. Everything a vendor does beyond that is reached through its own client,
// the same escape hatch the model providers offer.
//
// Stream's SIP support is inbound only today. A call to a number reaches an agent by the
// vendor bridging it into a Stream inbound trunk; a call from an agent is originated at
// the vendor and bridged into the same call, because there is nothing to ask Stream to
// dial out with.
package phone

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
)

// ErrNotImplemented is what a vendor returns when this service knows it exists but cannot
// yet work with it. It is a real error rather than a silent success, so a caller cannot
// think a number was bought when it was not.
var ErrNotImplemented = errors.New("phone: not implemented for this vendor")

// Capability is what a number can carry.
type Capability string

const (
	Voice Capability = "voice"
	SMS   Capability = "sms"
	MMS   Capability = "mms"
	Fax   Capability = "fax"
)

// Search narrows the numbers a vendor offers.
type Search struct {
	// Country is an ISO 3166-1 alpha-2 code, e.g. "US".
	Country string
	// AreaCode restricts the search to one dialling area, where the vendor supports it.
	AreaCode string
	// Contains is a digit pattern the number must contain, e.g. "555".
	Contains string
	// Capabilities every number must have. Voice is the one an agent needs.
	Capabilities []Capability
	// Limit caps how many are returned. Zero leaves the vendor's default.
	Limit int
}

// Available is a number a vendor is offering, not one that has been bought.
type Available struct {
	// E164 is the number in +15551234567 form, which is the only format used here.
	E164         string
	Country      string
	Region       string
	Locality     string
	Capabilities []Capability
	// MonthlyCostMicros is millionths of a dollar per month, zero when the vendor does
	// not quote a price on the search.
	MonthlyCostMicros int64
}

// Number is a number this service owns.
type Number struct {
	E164    string
	Vendor  string
	Country string
	// VendorID is the vendor's own identifier for the purchase, which is what releasing
	// or reconfiguring it needs.
	VendorID          string
	Capabilities      []Capability
	MonthlyCostMicros int64
}

// Bridge is the Stream SIP trunk a vendor connects a call to. Inbound calls are sent to
// it and outbound calls are bridged into it, so both directions share one description.
type Bridge struct {
	// URI is the SIP address of the trunk, e.g. "sip:trunk-id@sip.stream-io-api.com".
	URI string
	// Username and Password authenticate the vendor to the trunk.
	Username string
	Password string
}

// Validate reports whether the bridge is usable.
func (b Bridge) Validate() error {
	if b.URI == "" {
		return errors.New("phone: a bridge uri is required")
	}
	if !strings.HasPrefix(b.URI, "sip:") && !strings.HasPrefix(b.URI, "sips:") {
		return fmt.Errorf("phone: %q is not a sip uri", b.URI)
	}
	return nil
}

// Inbound points a number at the bridge, so calling it reaches an agent.
type Inbound struct {
	// E164 is the number to configure.
	E164 string
	// Bridge is where the vendor should send the call.
	Bridge Bridge
}

// Outbound is a call the vendor places and bridges into the agent's call.
type Outbound struct {
	// From is one of this service's own numbers, which is what the person sees.
	From string
	// To is who to call.
	To string
	// Bridge is the trunk the answered call is joined to.
	Bridge Bridge
}

// Dialed is a call in progress at the vendor.
type Dialed struct {
	// VendorCallID identifies the call at the vendor, for hanging up or inspecting it.
	VendorCallID string
	// Status is the vendor's own word for where the call is, e.g. "queued" or "ringing".
	Status string
}

// Provider is one telephony vendor.
type Provider interface {
	// SearchNumbers returns numbers the vendor is offering.
	SearchNumbers(ctx context.Context, search Search) ([]Available, error)
	// BuyNumber buys one, which is what starts the monthly charge.
	BuyNumber(ctx context.Context, e164 string) (Number, error)
	// ReleaseNumber gives it back, which is what stops the charge.
	ReleaseNumber(ctx context.Context, e164 string) error
	// ConfigureInbound points the number at the bridge.
	ConfigureInbound(ctx context.Context, inbound Inbound) error
	// Dial places a call and bridges it into the agent's call. Stream cannot dial out
	// itself, so the vendor originates and the agent is already waiting on the trunk.
	Dial(ctx context.Context, outbound Outbound) (Dialed, error)
	// SendDigits presses digits on a leg the vendor is holding, which is how an agent
	// gets past a menu on a call it placed. The leg is named by the id Dial returned:
	// there is no such id for an inbound call, so this only reaches calls made from here.
	SendDigits(ctx context.Context, vendorCallID, digits string) error

	// Vendor is the stable vendor name used in stats, e.g. "twilio".
	Vendor() string
	// Client exposes the underlying HTTP client so a caller can reach what this does not
	// wrap without building a second one.
	Client() *http.Client
}

// notImplemented stands in for a vendor this service can name but cannot yet work with.
// It exists so those vendors list and resolve like the rest rather than being absent,
// while every call says plainly that it did nothing.
type notImplemented struct {
	vendor string
	client *http.Client
}

// NotImplemented returns a provider that refuses every operation, naming the vendor.
func NotImplemented(vendor string) Provider {
	return &notImplemented{vendor: vendor, client: http.DefaultClient}
}

func (n *notImplemented) SearchNumbers(context.Context, Search) ([]Available, error) {
	return nil, n.err()
}

func (n *notImplemented) BuyNumber(context.Context, string) (Number, error) {
	return Number{}, n.err()
}

func (n *notImplemented) ReleaseNumber(context.Context, string) error { return n.err() }

func (n *notImplemented) ConfigureInbound(context.Context, Inbound) error { return n.err() }

func (n *notImplemented) Dial(context.Context, Outbound) (Dialed, error) {
	return Dialed{}, n.err()
}

func (n *notImplemented) SendDigits(context.Context, string, string) error { return n.err() }

func (n *notImplemented) Vendor() string { return n.vendor }

func (n *notImplemented) Client() *http.Client { return n.client }

func (n *notImplemented) err() error {
	return fmt.Errorf("%w: %s", ErrNotImplemented, n.vendor)
}

// ValidateDigits reports whether a string can be pressed on a keypad.
//
// A keypad has twelve keys, and w is the pause every vendor spells the same way: a menu
// that asks for an extension after a beep needs one. Anything else is a model that has
// written words where it meant digits, and sending it would be a silent no-op on the call.
func ValidateDigits(digits string) error {
	if digits == "" {
		return errors.New("phone: pressing needs digits to press")
	}
	for _, key := range digits {
		switch {
		case key >= '0' && key <= '9', key == '*', key == '#', key == 'w':
		default:
			return fmt.Errorf("phone: %q is not something a keypad can press", string(key))
		}
	}
	return nil
}

// covers reports whether a set of capabilities includes every one asked for.
func covers(has, wanted []Capability) bool {
	for _, capability := range wanted {
		if !slices.Contains(has, capability) {
			return false
		}
	}
	return true
}
