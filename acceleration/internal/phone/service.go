package phone

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Service is what the CLI and the HTTP API both talk to.
//
// A number is three things at once: a purchase at a vendor, a trunk at Stream and a row
// here. Doing them one at a time from two callers would eventually leave a number that is
// bought but unreachable, so the order lives in one place.
type Service struct {
	registry  *Registry
	store     *store.Store
	stream    *Stream
	recorder  *routing.Recorder
	publicURL string
	logger    *slog.Logger
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
	// PublicURL is where this service is reachable from the internet, which the three
	// vendors that fetch a call plan on answer need in order to fetch it. Without it those
	// vendors say so rather than placing a call nothing will answer.
	PublicURL string
	Logger    *slog.Logger
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
		registry:  options.Registry,
		store:     options.Store,
		stream:    options.Stream,
		recorder:  options.Recorder,
		publicURL: strings.TrimSuffix(options.PublicURL, "/"),
		logger:    options.Logger,
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

// Skip is a vendor that was asked nothing, and why.
type Skip struct {
	Vendor string
	// Reason is in the same words an error would use, because to the caller it is one:
	// this vendor's inventory is missing from the answer.
	Reason string
}

// Offers is what the vendors between them have for sale, and which of them did not answer.
//
// The skipped list is part of the result rather than a log line: a search for a Colorado
// number that reached two of eight vendors found what two vendors had, and a caller deciding
// whether to buy needs to know which.
type Offers struct {
	Numbers []Available
	Skipped []Skip
}

// SearchAll asks every usable vendor at once and merges what they offer.
//
// A vendor whose API cannot express one of the filters is skipped rather than asked, because
// dropping the filter would answer a search for Colorado with numbers from Ohio. Capabilities
// are different: every vendor reports what a number carries even when it cannot filter on it,
// so those are checked here on the results.
func (s *Service) SearchAll(ctx context.Context, search Search) (Offers, error) {
	usable := s.registry.Available()
	if len(usable) == 0 {
		return Offers{}, errors.New("phone: no vendor has the credentials to be searched")
	}

	type answer struct {
		offered []Available
		skip    *Skip
	}
	answers := make([]answer, len(usable))

	var group sync.WaitGroup
	for index, name := range usable {
		provider, err := s.registry.Open(name)
		if err != nil {
			answers[index] = answer{skip: &Skip{Vendor: name, Reason: err.Error()}}
			continue
		}
		if missing := search.Unsupported(provider); len(missing) > 0 {
			answers[index] = answer{skip: &Skip{
				Vendor: name,
				Reason: "cannot search by " + join(missing),
			}}
			continue
		}

		group.Add(1)
		go func() {
			defer group.Done()
			offered, err := provider.SearchNumbers(ctx, search)
			if err != nil {
				answers[index] = answer{skip: &Skip{Vendor: name, Reason: err.Error()}}
				return
			}
			answers[index] = answer{offered: offered}
		}()
	}
	group.Wait()

	var offers Offers
	for _, each := range answers {
		if each.skip != nil {
			offers.Skipped = append(offers.Skipped, *each.skip)
			continue
		}
		for _, offer := range each.offered {
			if search.Matches(offer) {
				offers.Numbers = append(offers.Numbers, offer)
			}
		}
	}

	// Cheapest first is the order a buyer wants; the number breaks the tie so the same
	// search twice gives the same answer, and a vendor that quotes no price is not
	// thereby made to look free.
	slices.SortFunc(offers.Numbers, func(a, b Available) int {
		if a.MonthlyCostMicros != b.MonthlyCostMicros {
			return cmp.Compare(quoted(a.MonthlyCostMicros), quoted(b.MonthlyCostMicros))
		}
		return cmp.Compare(a.E164, b.E164)
	})
	return offers, nil
}

// quoted sorts an unquoted price last rather than first, since zero here means the vendor
// did not say what it costs.
func quoted(micros int64) int64 {
	if micros == 0 {
		return math.MaxInt64
	}
	return micros
}

// join renders filters for an error message.
func join(filters []Filter) string {
	rendered := make([]string, 0, len(filters))
	for _, filter := range filters {
		rendered = append(rendered, string(filter))
	}
	return strings.Join(rendered, " or ")
}

// Purchase is a number to buy on a customer's behalf.
type Purchase struct {
	Vendor string
	E164   string
	// Country is the inventory the number was offered from, an ISO 3166-1 alpha-2 code.
	// Only some vendors need it, and the search that found the number reported it.
	Country string
	Owner   routing.Owner
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
	bought, err := provider.BuyNumber(ctx, Order{E164: purchase.E164, Country: purchase.Country})
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
	// CallID and CallType are the call callers land in, resolved rather than templated,
	// which is what an agent waiting for one has to be in.
	CallID   string
	CallType string
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

	// The rule serves one number, so the call is named outright rather than through the
	// handlebars template CreateRoute would otherwise fall back to. The name has to be
	// recorded, and a template is not a name until Stream renders it.
	callType := attachment.CallType
	if callType == "" {
		callType = defaultCallType
	}
	callID := attachment.CallID
	if callID == "" {
		callID = "phone-" + attachment.E164
	}

	routeID, err := s.stream.CreateRoute(ctx, Route{
		Name:          "phone-" + attachment.E164,
		TrunkIDs:      []string{trunkID},
		CalledNumbers: []string{attachment.E164},
		CallID:        callID,
		CallType:      callType,
	})
	if err != nil {
		return Attached{}, err
	}

	err = provider.ConfigureInbound(ctx, Inbound{E164: attachment.E164, Bridge: bridge})
	if err != nil {
		return Attached{}, err
	}
	if err := s.store.AttachNumber(ctx, attachment.CustomerID, attachment.E164, trunkID, callType, callID); err != nil {
		return Attached{}, err
	}

	return Attached{TrunkID: trunkID, RouteID: routeID, Bridge: bridge, CallID: callID, CallType: callType}, nil
}

// CallRequest is a call to place from one of the customer's numbers.
type CallRequest struct {
	Owner routing.Owner
	// From is one of the customer's own numbers, which is what the person sees.
	From string
	// To is who to call.
	To string
	// CallID is the Stream call the answered leg joins, and so the one the agent has to be
	// in. Empty names a fresh call after this one, because two calls from the same number
	// are two conversations and must not land in the same place.
	CallID string
	// CallType is the Stream call type. Empty means "default".
	CallType string
	// RingTimeout is how long to ring before giving up. Zero leaves the vendor's default.
	RingTimeout time.Duration
	// InitialDigits are pressed once the person answers, for reaching an extension behind
	// a menu.
	InitialDigits string
	// Headers are carried to the person's leg as custom SIP headers.
	Headers map[string]string
	// Custom is put on the Stream call, where the agent in it can read it. It is set at
	// Stream rather than at the vendor, so every vendor can carry it.
	Custom map[string]string
}

// Placed is a call that is on its way, and where to meet it.
type Placed struct {
	// VendorCallID identifies the ringing leg at the vendor, for hanging up or pressing
	// digits on it.
	VendorCallID string
	// Status is the vendor's own word for where the call is, e.g. "queued" or "ringing".
	Status string
	// Vendor is who is placing it.
	Vendor string
	// CallID and CallType are the Stream call the answered leg is routed into. An agent
	// that is not in it hears nothing when the person picks up.
	CallID   string
	CallType string
}

// Call places an outbound call and bridges it into a Stream call.
//
// Stream's SIP is inbound only, so the vendor originates the call and connects it to a
// trunk, rather than Stream dialling out. The trunk alone is not enough: without a routing
// rule the answered leg arrives with nothing pointing it at a call, so this creates both and
// pins the rule to the call the agent is waiting in, the way Transfer does.
func (s *Service) Call(ctx context.Context, request CallRequest) (Placed, error) {
	if s.stream == nil {
		return Placed{}, errors.New("phone: placing a call needs stream credentials")
	}
	if s.store == nil {
		return Placed{}, errors.New("phone: placing a call needs a database to know who holds the number")
	}
	if request.To == "" {
		return Placed{}, errors.New("phone: a call needs someone to call")
	}

	held, err := s.store.Number(ctx, request.Owner.CustomerID, request.From)
	if err != nil {
		return Placed{}, err
	}
	provider, err := s.registry.Open(held.Vendor)
	if err != nil {
		return Placed{}, err
	}
	declared, _ := s.registry.Lookup(held.Vendor)

	outbound := Outbound{
		From:          request.From,
		To:            request.To,
		RingTimeout:   request.RingTimeout,
		InitialDigits: request.InitialDigits,
		Headers:       request.Headers,
	}
	// A term the vendor cannot express is refused rather than dropped: a call placed
	// without the ring timeout that was asked for is not the call that was asked for.
	if missing := outbound.Unsupported(provider); len(missing) > 0 {
		return Placed{}, fmt.Errorf("phone: %s cannot place a call with %s",
			held.Vendor, joinFeatures(missing))
	}

	allowedIPs, err := trunkAllowlist(declared)
	if err != nil {
		return Placed{}, err
	}

	callID := request.CallID
	if callID == "" {
		callID = "call-" + uuid.NewString()
	}
	callType := request.CallType
	if callType == "" {
		callType = defaultCallType
	}

	trunkID, bridge, err := s.stream.CreateTrunk(ctx, Trunk{
		Name:       "call-" + callID,
		Numbers:    []string{request.From},
		AllowedIPs: allowedIPs,
	})
	if err != nil {
		return Placed{}, err
	}
	if _, err := s.stream.CreateRoute(ctx, Route{
		Name:          "call-" + callID,
		TrunkIDs:      []string{trunkID},
		CalledNumbers: []string{request.From},
		CallID:        callID,
		CallType:      callType,
		Custom:        request.Custom,
	}); err != nil {
		return Placed{}, err
	}

	outbound.Bridge = bridge
	if err := outbound.Validate(); err != nil {
		return Placed{}, err
	}
	// Three vendors will not take the plan on this request and fetch one on answer, so
	// they get somewhere to fetch it from instead.
	if _, hosted := provider.(AnswerRenderer); hosted {
		outbound.AnswerURL, err = s.park(ctx, held.Vendor, request.Owner.CustomerID, callID, outbound)
		if err != nil {
			return Placed{}, err
		}
	}

	started := time.Now()
	dialed, err := provider.Dial(ctx, outbound)
	s.record(held.Vendor, "call", request.Owner, started, 0, err)
	if err != nil {
		return Placed{}, err
	}
	return Placed{
		VendorCallID: dialed.VendorCallID,
		Status:       dialed.Status,
		Vendor:       held.Vendor,
		CallID:       callID,
		CallType:     callType,
	}, nil
}

// bridgeLifetime is how long a parked bridge is claimable for. It has to outlast the longest
// a vendor will ring, and nothing beyond that: a bridge nobody claimed is a call nobody
// answered.
const bridgeLifetime = 10 * time.Minute

// park saves what the vendor should be told on answer, and returns where it fetches it from.
func (s *Service) park(
	ctx context.Context,
	vendor, customerID, callID string,
	outbound Outbound,
) (string, error) {
	if s.publicURL == "" {
		return "", fmt.Errorf(
			"phone: %s fetches its call plan when the person answers, so it needs a "+
				"public url to fetch it from, and none is configured", vendor)
	}

	token := uuid.NewString()
	err := s.store.ParkBridge(ctx, &store.CallBridge{
		Token:         token,
		CustomerID:    customerID,
		Vendor:        vendor,
		TrunkURI:      outbound.Bridge.URI,
		TrunkUsername: outbound.Bridge.Username,
		TrunkPassword: outbound.Bridge.Password,
		InitialDigits: outbound.InitialDigits,
		CallID:        callID,
		ExpiresAt:     time.Now().UTC().Add(bridgeLifetime),
	})
	if err != nil {
		return "", err
	}
	return s.publicURL + "/v1/phone/answer/" + token, nil
}

// Answer renders what a vendor should do now that the person it called has picked up.
//
// The token is the whole of the request's authentication: the vendor fetching this has no
// customer to name. Claiming spends the token, so a plan is served once and a token that
// leaks afterwards is a token for nothing.
func (s *Service) Answer(ctx context.Context, token string) (Plan, error) {
	if s.store == nil {
		return Plan{}, errors.New("phone: answering a call needs a database to know what to answer")
	}

	bridge, err := s.store.ClaimBridge(ctx, token)
	if err != nil {
		return Plan{}, err
	}
	provider, err := s.registry.Open(bridge.Vendor)
	if err != nil {
		return Plan{}, err
	}
	renderer, ok := provider.(AnswerRenderer)
	if !ok {
		return Plan{}, fmt.Errorf("phone: %s does not fetch a call plan, so it has none to serve",
			bridge.Vendor)
	}

	return renderer.Answer(Bridge{
		URI:      bridge.TrunkURI,
		Username: bridge.TrunkUsername,
		Password: bridge.TrunkPassword,
	}, bridge.InitialDigits)
}

// SweepBridges removes the bridges left behind by calls nobody answered.
func (s *Service) SweepBridges(ctx context.Context) (int64, error) {
	if s.store == nil {
		return 0, nil
	}
	return s.store.SweepBridges(ctx)
}

// trunkAllowlist is the addresses a vendor's trunk should accept calls from.
//
// A vendor that can present the trunk's password needs no allowlist. One that cannot needs
// one, and refusing to place the call without it is the point: Stream reads an empty
// allowlist as "accept everything" rather than "password only", so a trunk with neither is
// a way into a customer's calls for anyone who learns its uri.
func trunkAllowlist(vendor Vendor) ([]string, error) {
	if vendor.Authenticates() == TrunkPassword {
		return nil, nil
	}
	if len(vendor.Signalling) == 0 {
		return nil, fmt.Errorf(
			"phone: %s cannot send a password to a trunk, so the trunk has to know its "+
				"signalling addresses, and none are declared for it",
			vendor.Vendor)
	}
	return vendor.Signalling, nil
}

// joinFeatures renders call features for an error message.
func joinFeatures(features []CallFeature) string {
	rendered := make([]string, 0, len(features))
	for _, feature := range features {
		rendered = append(rendered, string(feature))
	}
	return strings.Join(rendered, " or ")
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
