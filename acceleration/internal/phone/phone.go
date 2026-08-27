// Package phone gives an agent a telephone number.
//
// Eleven vendors sell numbers and they all sell the same thing, so the contract here is
// the small part they agree on: find a number, buy it, point it at the bridge, dial out,
// give it back. Everything a vendor does beyond that is reached through its own client,
// the same escape hatch the model providers offer.
//
// What they do not agree on is how a search can be narrowed, so a provider also declares
// which filters its API can express. A search using one it cannot is skipped rather than
// asked without it: answering a search for Colorado with numbers from Ohio reads as a
// result rather than as a gap. Placing a call works the same way: a vendor declares which
// of a call's terms it can express, and one it cannot is refused rather than dropped.
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
	"time"
)

// maxInitialDigits is how many keys can be pressed when a call is answered. Twilio takes
// 32 and Telnyx 64, so 32 is what can be promised everywhere.
const maxInitialDigits = 32

// ErrNotImplemented is what a vendor returns when this service knows it exists but cannot
// yet work with it. It is a real error rather than a silent success, so a caller cannot
// think a number was bought when it was not.
var ErrNotImplemented = errors.New("phone: not implemented for this vendor")

// Capability is what a number can carry.
//
// The names are Telnyx's feature names, because they are the widest vocabulary any of these
// vendors offers and every other vendor's smaller set maps into them without loss.
type Capability string

const (
	Voice            Capability = "voice"
	SMS              Capability = "sms"
	MMS              Capability = "mms"
	Fax              Capability = "fax"
	Emergency        Capability = "emergency"
	HDVoice          Capability = "hd_voice"
	InternationalSMS Capability = "international_sms"
	LocalCalling     Capability = "local_calling"
)

// NumberType is what kind of number it is, which decides who pays for the call.
type NumberType string

const (
	// Local is a geographic number, the kind an agent usually wants.
	Local NumberType = "local"
	// TollFree is free to the caller and charged to whoever holds it.
	TollFree NumberType = "toll_free"
	// Mobile is a number on a mobile range, which some countries treat differently.
	Mobile NumberType = "mobile"
)

// Filter names one way of narrowing a search.
//
// Vendors do not agree on which of these they can express: Telnyx filters by US state,
// Sinch does not. A provider says which it speaks so a search asking for one it does not is
// skipped rather than answered with numbers from the wrong place.
type Filter string

const (
	FilterCountry            Filter = "country"
	FilterAreaCode           Filter = "area_code"
	FilterContains           Filter = "contains"
	FilterPrefix             Filter = "prefix"
	FilterLocality           Filter = "locality"
	FilterAdministrativeArea Filter = "administrative_area"
	FilterNumberType         Filter = "number_type"
)

// Search narrows the numbers a vendor offers.
type Search struct {
	// Country is an ISO 3166-1 alpha-2 code, e.g. "US".
	Country string
	// AreaCode restricts the search to one dialling area, where the vendor supports it.
	AreaCode string
	// Contains is a digit pattern the number must appear anywhere in, e.g. "555".
	Contains string
	// Prefix is digits the national part of the number must start with, e.g. "719" for
	// Colorado Springs. It differs from Contains in where the digits have to fall.
	Prefix string
	// Locality is a city, region or rate centre, e.g. "Denver".
	Locality string
	// AdministrativeArea is a US state or Canadian province, e.g. "CO".
	AdministrativeArea string
	// Type is the kind of number wanted. Empty leaves the vendor's default, which is
	// local everywhere that has one.
	Type NumberType
	// Capabilities every number must have. Voice is the one an agent needs.
	Capabilities []Capability
	// Limit caps how many are returned. Zero leaves the vendor's default.
	Limit int
}

// Filters are the ways this search is narrowed, which is what a vendor has to be able to
// express to answer it. Capabilities are not among them: every vendor reports what a number
// carries, so those are checked on the results instead.
func (s Search) Filters() []Filter {
	var used []Filter
	if s.Country != "" {
		used = append(used, FilterCountry)
	}
	if s.AreaCode != "" {
		used = append(used, FilterAreaCode)
	}
	if s.Contains != "" {
		used = append(used, FilterContains)
	}
	if s.Prefix != "" {
		used = append(used, FilterPrefix)
	}
	if s.Locality != "" {
		used = append(used, FilterLocality)
	}
	if s.AdministrativeArea != "" {
		used = append(used, FilterAdministrativeArea)
	}
	if s.Type != "" {
		used = append(used, FilterNumberType)
	}
	return used
}

// Unsupported are the filters this search uses that a provider cannot express.
func (s Search) Unsupported(provider Provider) []Filter {
	var missing []Filter
	for _, filter := range s.Filters() {
		if !provider.Supports(filter) {
			missing = append(missing, filter)
		}
	}
	return missing
}

// Matches reports whether an offer actually satisfies the capabilities asked for. A vendor
// that cannot filter by capability still says what each number carries, so this is what
// keeps its answer honest.
func (s Search) Matches(offer Available) bool {
	return covers(offer.Capabilities, s.Capabilities)
}

// Available is a number a vendor is offering, not one that has been bought.
type Available struct {
	// E164 is the number in +15551234567 form, which is the only format used here.
	E164 string
	// Vendor is who is offering it, which matters once several vendors are asked at once.
	Vendor       string
	Country      string
	Region       string
	Locality     string
	Type         NumberType
	Capabilities []Capability
	// MonthlyCostMicros is millionths of a dollar per month, zero when the vendor does
	// not quote a price on the search.
	MonthlyCostMicros int64
}

// Order is a number to buy, as the search offered it.
//
// It carries the country because some vendors buy out of a country's inventory rather than
// by number alone, and the offer already knows which country it came from. Guessing it back
// out of the dial code would be a table of every country to maintain for one vendor.
type Order struct {
	E164 string
	// Country is an ISO 3166-1 alpha-2 code. Empty is fine at the vendors that do not
	// need it, which is most of them.
	Country string
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

// CallFeature names one of the terms a call can be placed on.
//
// Vendors do not agree on which of these their call APIs can express: Telnyx takes custom
// SIP headers on a dial, Twilio does not. A provider says which it speaks, so a call asking
// for one it does not is refused rather than placed without it. Silently dropping a ring
// timeout means a call that should have given up in fifteen seconds sits in somebody's
// voicemail for a minute instead.
type CallFeature string

const (
	FeatureRingTimeout   CallFeature = "ring_timeout"
	FeatureInitialDigits CallFeature = "initial_digits"
	FeatureCustomHeaders CallFeature = "custom_headers"
)

// Outbound is a call the vendor places and bridges into the agent's call.
type Outbound struct {
	// From is one of this service's own numbers, which is what the person sees.
	From string
	// To is who to call.
	To string
	// Bridge is the trunk the answered call is joined to.
	Bridge Bridge
	// RingTimeout is how long to ring before giving up. Zero leaves the vendor's default,
	// which is around a minute and long enough to reach voicemail.
	RingTimeout time.Duration
	// InitialDigits are pressed once the person answers, which is how a call reaches an
	// extension behind a menu, e.g. "ww1234#" for a conference bridge.
	InitialDigits string
	// Headers are carried to the person's leg as custom SIP headers, for whatever is on
	// the other end to read.
	Headers map[string]string
	// AnswerURL is where a vendor that will not take a call plan on the request fetches
	// one from when the person answers. It is set only for those vendors, and what it
	// serves is that vendor's own Answer.
	AnswerURL string
}

// Features are the terms this call is placed on, which is what a vendor has to be able to
// express to place it.
func (o Outbound) Features() []CallFeature {
	var used []CallFeature
	if o.RingTimeout > 0 {
		used = append(used, FeatureRingTimeout)
	}
	if o.InitialDigits != "" {
		used = append(used, FeatureInitialDigits)
	}
	if len(o.Headers) > 0 {
		used = append(used, FeatureCustomHeaders)
	}
	return used
}

// Unsupported are the terms this call is placed on that a provider cannot express.
func (o Outbound) Unsupported(provider Provider) []CallFeature {
	var missing []CallFeature
	for _, feature := range o.Features() {
		if !provider.Dials(feature) {
			missing = append(missing, feature)
		}
	}
	return missing
}

// Validate reports whether the call can be placed as described.
func (o Outbound) Validate() error {
	if o.From == "" || o.To == "" {
		return errors.New("phone: a call needs a from and a to")
	}
	if err := o.Bridge.Validate(); err != nil {
		return err
	}
	if o.RingTimeout < 0 {
		return errors.New("phone: a call cannot ring for less than no time")
	}
	if o.InitialDigits != "" {
		if err := ValidateDigits(o.InitialDigits); err != nil {
			return err
		}
		if len(o.InitialDigits) > maxInitialDigits {
			return fmt.Errorf("phone: %d digits is more than the %d every vendor takes on answer",
				len(o.InitialDigits), maxInitialDigits)
		}
	}
	return nil
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
	BuyNumber(ctx context.Context, order Order) (Number, error)
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

	// Supports reports whether this vendor's API can express a way of narrowing a search.
	// A search using one it cannot is skipped, because answering it with numbers from
	// somewhere else would look like a result.
	Supports(filter Filter) bool
	// Dials reports whether this vendor's call API can express one of a call's terms. A
	// call asking for one it cannot is refused, because a ring timeout that was ignored
	// is not a ring timeout.
	Dials(feature CallFeature) bool

	// Vendor is the stable vendor name used in stats, e.g. "twilio".
	Vendor() string
	// Client exposes the underlying HTTP client so a caller can reach what this does not
	// wrap without building a second one.
	Client() *http.Client
}

// Plan is what a vendor is told to do when the person it called picks up.
type Plan struct {
	// ContentType is what the vendor expects to be served, which is XML at most of them.
	ContentType string
	// Body is the plan itself, in that vendor's dialect.
	Body []byte
}

// AnswerRenderer is a vendor that will not take a call plan on the request that places a
// call, and fetches one when the person answers instead.
//
// It is separate from Provider because it is true of three vendors out of eight, and a
// method every other vendor had to refuse would say less than not having it. Implementing it
// is also what tells the service this vendor needs an answer url in the first place.
type AnswerRenderer interface {
	// Answer renders this vendor's way of saying "press these, then bridge to that".
	Answer(bridge Bridge, initialDigits string) (Plan, error)
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

func (n *notImplemented) BuyNumber(context.Context, Order) (Number, error) {
	return Number{}, n.err()
}

func (n *notImplemented) ReleaseNumber(context.Context, string) error { return n.err() }

func (n *notImplemented) ConfigureInbound(context.Context, Inbound) error { return n.err() }

func (n *notImplemented) Dial(context.Context, Outbound) (Dialed, error) {
	return Dialed{}, n.err()
}

func (n *notImplemented) SendDigits(context.Context, string, string) error { return n.err() }

// Supports claims every filter, so a search reaches the operation and is refused there by
// name rather than being quietly skipped as if the vendor had nothing to sell.
func (n *notImplemented) Supports(Filter) bool { return true }

// Dials claims every feature, for the same reason Supports claims every filter: "didww
// cannot dial" is the useful answer, and "didww cannot express a ring timeout" would hide
// it behind a detail.
func (n *notImplemented) Dials(CallFeature) bool { return true }

func (n *notImplemented) Vendor() string { return n.vendor }

func (n *notImplemented) Client() *http.Client { return n.client }

func (n *notImplemented) err() error {
	return fmt.Errorf("%w: %s", ErrNotImplemented, n.vendor)
}

// ValidateDigits reports whether a string can be pressed on a keypad.
//
// A keypad has twelve keys, and w is the pause every vendor spells the same way: a menu
// that asks for an extension after a beep needs one. W is the longer pause and A to D are
// the fourth column no consumer handset has but every vendor's API accepts. Anything else
// is a model that has written words where it meant digits, and sending it would be a
// silent no-op on the call.
func ValidateDigits(digits string) error {
	if digits == "" {
		return errors.New("phone: pressing needs digits to press")
	}
	for _, key := range digits {
		switch {
		case key >= '0' && key <= '9', key >= 'A' && key <= 'D',
			key == '*', key == '#', key == 'w', key == 'W':
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
