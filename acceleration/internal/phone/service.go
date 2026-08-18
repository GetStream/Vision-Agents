package phone

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Service is what the CLI and the HTTP API both talk to.
//
// A number is three things at once: a purchase at a vendor, a trunk at Stream and a row
// here. Doing them one at a time from two callers would eventually leave a number that is
// bought but unreachable, so the order lives in one place.
type Service struct {
	registry *Registry
	store    *store.Store
	stream   *Stream
	recorder *routing.Recorder
	logger   *slog.Logger
}

// ServiceOptions configures a Service. Only the registry is required: without a store
// nothing is remembered, and without Stream a number can be bought but not attached.
type ServiceOptions struct {
	Registry *Registry
	Store    *store.Store
	Stream   *Stream
	// Recorder files purchases as request rows, so a number's monthly charge shows up in
	// cost reporting next to what the models cost.
	Recorder *routing.Recorder
	Logger   *slog.Logger
}

// NewService returns a Service.
func NewService(options ServiceOptions) (*Service, error) {
	if options.Registry == nil {
		return nil, errors.New("phone: a vendor registry is required")
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	return &Service{
		registry: options.Registry,
		store:    options.Store,
		stream:   options.Stream,
		recorder: options.Recorder,
		logger:   options.Logger,
	}, nil
}

// Registry is the vendors this service knows about.
func (s *Service) Registry() *Registry { return s.registry }

// Search returns the numbers one vendor is offering.
func (s *Service) Search(ctx context.Context, vendor string, search Search) ([]Available, error) {
	provider, err := s.registry.Open(vendor)
	if err != nil {
		return nil, err
	}
	return provider.SearchNumbers(ctx, search)
}

// Purchase is a number to buy on a customer's behalf.
type Purchase struct {
	Vendor string
	E164   string
	Owner  routing.Owner
}

// Buy buys a number and records that the customer now holds it.
//
// The vendor is the side that costs money, so it goes first: a row without a number is a
// lie, while a number without a row is a reconcilable mistake that is also logged.
func (s *Service) Buy(ctx context.Context, purchase Purchase) (store.PhoneNumber, error) {
	if purchase.Owner.CustomerID == "" {
		return store.PhoneNumber{}, errors.New("phone: a number must belong to a customer")
	}
	provider, err := s.registry.Open(purchase.Vendor)
	if err != nil {
		return store.PhoneNumber{}, err
	}

	started := time.Now()
	bought, err := provider.BuyNumber(ctx, purchase.E164)
	s.record(purchase.Vendor, "number", purchase.Owner, started, bought.MonthlyCostMicros, err)
	if err != nil {
		return store.PhoneNumber{}, err
	}

	held := store.PhoneNumber{
		E164:              bought.E164,
		Vendor:            bought.Vendor,
		Country:           bought.Country,
		Capabilities:      names(bought.Capabilities),
		MonthlyCostMicros: bought.MonthlyCostMicros,
		CustomerID:        purchase.Owner.CustomerID,
		Tags:              purchase.Owner.Tags,
		VendorID:          bought.VendorID,
		PurchasedAt:       time.Now().UTC(),
	}
	if s.store == nil {
		return held, nil
	}
	if err := s.store.RecordNumber(ctx, &held); err != nil {
		// The number is bought either way, so saying so is more useful than failing and
		// leaving the caller thinking it was not.
		return held, fmt.Errorf("phone: %s was bought but not recorded: %w", held.E164, err)
	}
	return held, nil
}

// Release gives a number back and stops the monthly charge.
func (s *Service) Release(ctx context.Context, customerID, e164 string) error {
	if s.store == nil {
		return errors.New("phone: releasing a number needs a database to know who holds it")
	}
	held, err := s.store.Number(ctx, customerID, e164)
	if err != nil {
		return err
	}
	provider, err := s.registry.Open(held.Vendor)
	if err != nil {
		return err
	}
	if err := provider.ReleaseNumber(ctx, e164); err != nil {
		return err
	}
	return s.store.ReleaseNumber(ctx, customerID, e164, time.Now().UTC())
}

// Attachment points a number at a Stream call.
type Attachment struct {
	CustomerID string
	E164       string
	// CallID is the call every caller joins. Empty gives each number its own call, named
	// after the number that was rung.
	CallID string
	// CallType is the Stream call type. Empty means "default".
	CallType string
	// AllowedIPs are the vendor's signalling addresses. Empty accepts calls from anywhere
	// that has the trunk password.
	AllowedIPs []string
}

// Attached is what a number was connected to.
type Attached struct {
	TrunkID string
	RouteID string
	Bridge  Bridge
}

// Attach creates the Stream trunk and routing rule for a number and tells the vendor to
// send calls there. This is what turns a bought number into one that reaches an agent.
func (s *Service) Attach(ctx context.Context, attachment Attachment) (Attached, error) {
	if s.stream == nil {
		return Attached{}, errors.New("phone: attaching a number needs stream credentials")
	}
	if s.store == nil {
		return Attached{}, errors.New("phone: attaching a number needs a database to know who holds it")
	}

	held, err := s.store.Number(ctx, attachment.CustomerID, attachment.E164)
	if err != nil {
		return Attached{}, err
	}
	provider, err := s.registry.Open(held.Vendor)
	if err != nil {
		return Attached{}, err
	}

	trunkID, bridge, err := s.stream.CreateTrunk(ctx, Trunk{
		Name:       "phone-" + attachment.E164,
		Numbers:    []string{attachment.E164},
		AllowedIPs: attachment.AllowedIPs,
	})
	if err != nil {
		return Attached{}, err
	}

	routeID, err := s.stream.CreateRoute(ctx, Route{
		Name:          "phone-" + attachment.E164,
		TrunkIDs:      []string{trunkID},
		CalledNumbers: []string{attachment.E164},
		CallID:        attachment.CallID,
		CallType:      attachment.CallType,
	})
	if err != nil {
		return Attached{}, err
	}

	err = provider.ConfigureInbound(ctx, Inbound{E164: attachment.E164, Bridge: bridge})
	if err != nil {
		return Attached{}, err
	}
	if err := s.store.AttachNumber(ctx, attachment.CustomerID, attachment.E164, trunkID); err != nil {
		return Attached{}, err
	}

	return Attached{TrunkID: trunkID, RouteID: routeID, Bridge: bridge}, nil
}

// CallRequest is a call to place from one of the customer's numbers.
type CallRequest struct {
	Owner routing.Owner
	// From is one of the customer's own numbers, which is what the person sees.
	From string
	// To is who to call.
	To string
	// Bridge is the trunk the answered call joins. Empty asks Stream for a fresh one, so
	// a one-off call does not need a number that was attached first.
	Bridge Bridge
}

// Call places an outbound call and bridges it into a Stream call.
//
// Stream's SIP is inbound only, so the vendor originates the call and connects it to a
// trunk the agent is already on, rather than Stream dialling out.
func (s *Service) Call(ctx context.Context, request CallRequest) (Dialed, error) {
	if s.store == nil {
		return Dialed{}, errors.New("phone: placing a call needs a database to know who holds the number")
	}
	held, err := s.store.Number(ctx, request.Owner.CustomerID, request.From)
	if err != nil {
		return Dialed{}, err
	}
	provider, err := s.registry.Open(held.Vendor)
	if err != nil {
		return Dialed{}, err
	}

	bridge := request.Bridge
	if bridge.URI == "" {
		if s.stream == nil {
			return Dialed{}, errors.New("phone: a call needs a bridge, or stream credentials to make one")
		}
		_, bridge, err = s.stream.CreateTrunk(ctx, Trunk{
			Name:    "call-" + request.To,
			Numbers: []string{request.From},
		})
		if err != nil {
			return Dialed{}, err
		}
	}

	started := time.Now()
	placed, err := provider.Dial(ctx, Outbound{From: request.From, To: request.To, Bridge: bridge})
	s.record(held.Vendor, "call", request.Owner, started, 0, err)
	return placed, err
}

// TransferRequest hands a live call to a human.
type TransferRequest struct {
	Owner routing.Owner
	// From is the customer's number the human is called from, which is what they see.
	From string
	// To is the human being brought onto the call.
	To string
	// CallID is the Stream call to bring them into, which is the one the agent and the
	// caller are already on.
	CallID string
	// CallType is the Stream call type. Empty means "default".
	CallType string
}

// Transfer brings a human onto a call that is already happening.
//
// Stream's SIP is inbound only, so a transfer is not a handover of the caller's leg: it is
// a second leg, originated at the vendor and routed into the same Stream call, after which
// three parties are on it and the agent leaves. The caller is never moved, which is why they
// hear nothing of it happening and why nothing is lost if the human does not answer.
//
// The routing rule is pinned to the live call rather than to the number's own template, so
// the human joins this conversation rather than whichever one their number would have
// landed in. It is created per transfer because a trunk's password is only readable when
// the trunk is made, and keeping SIP credentials to reuse later is a worse trade than
// making a second trunk.
func (s *Service) Transfer(ctx context.Context, request TransferRequest) (Dialed, error) {
	if s.stream == nil {
		return Dialed{}, errors.New("phone: transferring a call needs stream credentials")
	}
	if s.store == nil {
		return Dialed{}, errors.New("phone: transferring a call needs a database to know who holds the number")
	}
	if request.CallID == "" {
		return Dialed{}, errors.New("phone: a transfer needs the call to transfer into")
	}
	if request.To == "" {
		return Dialed{}, errors.New("phone: a transfer needs someone to transfer to")
	}

	held, err := s.store.Number(ctx, request.Owner.CustomerID, request.From)
	if err != nil {
		return Dialed{}, err
	}
	provider, err := s.registry.Open(held.Vendor)
	if err != nil {
		return Dialed{}, err
	}

	trunkID, bridge, err := s.stream.CreateTrunk(ctx, Trunk{
		Name:    "transfer-" + request.CallID,
		Numbers: []string{request.From},
	})
	if err != nil {
		return Dialed{}, err
	}
	if _, err := s.stream.CreateRoute(ctx, Route{
		Name:          "transfer-" + request.CallID,
		TrunkIDs:      []string{trunkID},
		CalledNumbers: []string{request.From},
		CallID:        request.CallID,
		CallType:      request.CallType,
	}); err != nil {
		return Dialed{}, err
	}

	started := time.Now()
	placed, err := provider.Dial(ctx, Outbound{From: request.From, To: request.To, Bridge: bridge})
	s.record(held.Vendor, "transfer", request.Owner, started, 0, err)
	return placed, err
}

// SendDigits presses digits on a call this service placed, which is how an agent answers a
// menu it has reached.
//
// The call is named by the id its vendor gave when it was dialled. An inbound call has no
// such id here, because it arrived at Stream rather than being made from this service, so
// only calls placed from here can be pressed at.
func (s *Service) SendDigits(ctx context.Context, vendor, vendorCallID, digits string) error {
	if vendorCallID == "" {
		return errors.New("phone: pressing digits needs the call to press them on")
	}
	if err := ValidateDigits(digits); err != nil {
		return err
	}
	provider, err := s.registry.Open(vendor)
	if err != nil {
		return err
	}
	return provider.SendDigits(ctx, vendorCallID, digits)
}

// Numbers returns what a customer holds.
func (s *Service) Numbers(ctx context.Context, customerID string, includeReleased bool) ([]store.PhoneNumber, error) {
	if s.store == nil {
		return nil, errors.New("phone: listing numbers needs a database")
	}
	return s.store.CustomerNumbers(ctx, customerID, includeReleased)
}

// record files a vendor operation as a request row, so what telephony costs is reported
// beside what the models cost and under the same labels.
func (s *Service) record(
	vendor, model string,
	owner routing.Owner,
	started time.Time,
	costMicros int64,
	err error,
) {
	if s.recorder == nil {
		return
	}

	stat := routing.Stat{
		Owner:      owner,
		StartedAt:  started.UTC(),
		LatencyMs:  routing.MsSince(started),
		CostMicros: costMicros,
		Success:    err == nil,
	}
	if err != nil {
		stat.ErrorCode = "vendor_failed"
	}
	s.recorder.Record(routing.ProviderConfig{Provider: vendor, Model: model}, stat)
}

// names renders capabilities for storage, since Postgres holds them as plain text.
func names(capabilities []Capability) []string {
	rendered := make([]string, 0, len(capabilities))
	for _, capability := range capabilities {
		rendered = append(rendered, string(capability))
	}
	return rendered
}
