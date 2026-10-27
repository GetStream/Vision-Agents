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

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/outcome"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/sipbridge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// Service is what the CLI and the HTTP API both talk to.
//
// A number is three things at once: a purchase at a vendor, a trunk at Stream and a row
// here. Doing them one at a time from two callers would eventually leave a number that is
// bought but unreachable, so the order lives in one place.
type Service struct {
	registry *Registry
	store    *store.Store
	groups   groupLocks
	stream   *Stream
	apps     Apps
	// Calls through a customer's own trunk are off unless both sealer and sipTrunks are
	// set: they need the key its password is sealed under and the provider that dials it.
	sealer    *auth.Sealer
	sipTrunks Provider
	recorder  *routing.Recorder
	gate      *dlc.Gate
	publicURL string
	// outcomes keeps how each placed call ended. Nil without a store.
	outcomes *outcome.Keeper
	clock    func() time.Time
	logger   *slog.Logger
	// heard is the calls whose participant events this process recorded lately (Heard), and
	// unhooked the apps it lately logged as not delivering them.
	heard, unhooked recently
	// connecting, when set, runs in Connect between reading the phone's kept leave and
	// connecting its leg, where a leave delivered at the same moment lands.
	connecting func()
}

// ServiceOptions configures a Service. Only the registry is required: without a store
// nothing is remembered, and without Stream a number can be bought but not attached.
type ServiceOptions struct {
	Registry *Registry
	Store    *store.Store
	Stream   *Stream
	// Apps answers which Stream app each customer's lines live in. When it is set, Stream
	// is not used.
	Apps Apps
	// Sealer seals the passwords of customers' own SIP trunks. Without it those trunks
	// are off, rather than their passwords stored in the clear.
	Sealer *auth.Sealer
	// SIPTrunks dials numbers on customers' own SIP trunks. It is kept out of the
	// registry, which is what every customer is offered to buy from.
	SIPTrunks Provider
	// Recorder files what telephony costs as request rows, beside what the models cost: each
	// month of a number, each call placed, and the minutes of every leg of a call.
	Recorder *routing.Recorder
	// Gate refuses a call to somebody who opted out, or past what the sandbox allows, and,
	// when it is charging, a call or a number the organization may not pay for.
	Gate *dlc.Gate
	// PublicURL is where this service is reachable from the internet, which the three
	// vendors that fetch a call plan on answer need in order to fetch it. Without it those
	// vendors say so rather than placing a call nothing will answer.
	PublicURL string
	// Now is the clock call legs are timed by. Nil is the wall clock.
	Now    func() time.Time
	Logger *slog.Logger
}

// NewService returns a Service.
func NewService(options ServiceOptions) (*Service, error) {
	if options.Registry == nil {
		return nil, errors.New("phone: a vendor registry is required")
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	service := &Service{
		registry:  options.Registry,
		store:     options.Store,
		stream:    options.Stream,
		apps:      options.Apps,
		sealer:    options.Sealer,
		sipTrunks: options.SIPTrunks,
		recorder:  options.Recorder,
		gate:      options.Gate,
		publicURL: strings.TrimSuffix(options.PublicURL, "/"),
		clock:     options.Now,
		logger:    options.Logger,
	}
	if options.Store != nil {
		service.outcomes = outcome.NewKeeper(options.Store, service.now)
	}
	return service, nil
}

// Outcomes keeps how each placed call ended. Nil without a store.
func (s *Service) Outcomes() *outcome.Keeper { return s.outcomes }

// outcomeOf is the outcome a vendor's result is. It is the one place a vendor's word for a
// call meets the router's: an empty result is a report that is not final, and so no outcome.
func outcomeOf(result CallResult) (outcome.Outcome, error) {
	switch result {
	case "":
		return "", nil
	case CallCompleted:
		return outcome.Completed, nil
	case CallBusy:
		return outcome.Busy, nil
	case CallNoAnswer:
		return outcome.NoAnswer, nil
	case CallFailed:
		return outcome.Failed, nil
	case CallCanceled:
		return outcome.Canceled, nil
	}
	return "", stack.Wrap(fmt.Errorf("phone: no outcome for the call result %q", result))
}

// Apps says which Stream app a customer's phone lines live in. A trunk and its routing rule
// have to be in the same app as the call the agent joins, which is the customer's.
type Apps interface {
	// For is the app new lines for a customer are made in, with its pin.
	For(ctx context.Context, customer string) (*Stream, int64, error)
	// ForApp is the app a line already made is in, to finish it there.
	ForApp(ctx context.Context, customer string, app int64) (*Stream, error)
	// ForAppRemoving is the app a line already made is in, to remove it there. It reaches
	// wherever reading that app does: a line in the deployment's app that the customer may
	// no longer add to is still the router's own to take down, and one left there is billed.
	ForAppRemoving(ctx context.Context, customer string, app int64) (*Stream, error)
}

// errNoStream is a service with no Stream app to make lines in.
var errNoStream = errors.New("phone: no stream credentials")

// Suspended returns ErrSuspended for an app Stream staff stopped from using the phone. A
// suspension that cannot be read refuses too: it is there to stop fraud that costs money.
func (s *Service) Suspended(ctx context.Context, customerID string) error {
	if s.store == nil {
		return nil
	}
	_, err := s.store.TelephonySuspension(ctx, customerID)
	if errors.Is(err, store.ErrNotSuspended) {
		return nil
	}
	if err != nil {
		return err
	}
	return stack.Wrap(ErrSuspended)
}

// Admit refuses a customer whose organization may not pay for telephony now, as the gate every
// call passes would, so work that would place calls can be refused before it starts.
func (s *Service) Admit(ctx context.Context, customerID string) error {
	return s.gate.Admit(ctx, customerID)
}

// EndCall ends a Stream call one of the customer's numbers brought somebody into, in the app
// the number's lines are in. Stream's SIP bridge has answered the caller by the time the call's
// session starts, so a call nobody may take is ended rather than left ringing.
func (s *Service) EndCall(ctx context.Context, customer string, app int64, callType, callID string) error {
	stream, err := s.streamForApp(ctx, customer, app)
	if err != nil {
		return err
	}
	return stream.EndCall(ctx, callType, callID)
}

// streamFor is the app new lines for a customer are made in, and its pin.
func (s *Service) streamFor(ctx context.Context, customer string) (*Stream, int64, error) {
	if s.apps != nil {
		return s.apps.For(ctx, customer)
	}
	if s.stream == nil {
		return nil, 0, errNoStream
	}
	return s.stream, 0, nil
}

// streamForApp is the app a line already made is in.
func (s *Service) streamForApp(ctx context.Context, customer string, app int64) (*Stream, error) {
	if s.apps != nil {
		return s.apps.ForApp(ctx, customer, app)
	}
	if s.stream == nil {
		return nil, errNoStream
	}
	return s.stream, nil
}

// streamToRemove is the app a line already made is in, to remove it there.
func (s *Service) streamToRemove(ctx context.Context, customer string, app int64) (*Stream, error) {
	if s.apps != nil {
		return s.apps.ForAppRemoving(ctx, customer, app)
	}
	if s.stream == nil {
		return nil, errNoStream
	}
	return s.stream, nil
}

// unwire removes a trunk and routing rule from the app they were made in. It is
// best-effort: a cleanup failure is logged rather than returned, so it cannot mask what
// the caller was doing.
func (s *Service) unwire(ctx context.Context, customer string, app int64, routeID, trunkID, why string) {
	if routeID == "" && trunkID == "" {
		return
	}
	stream, err := s.streamToRemove(ctx, customer, app)
	if err != nil {
		s.logger.Error("could not reach the Stream app a line was made in", "why", why,
			"trunk", trunkID, "route", routeID, "error", err)
		return
	}
	s.deleteLines(ctx, stream, routeID, trunkID, why)
}

// deleteLines deletes a routing rule and the trunk beside it from a Stream app. It is
// best-effort: a failure is logged, never returned, so it cannot mask what is being undone.
func (s *Service) deleteLines(ctx context.Context, stream *Stream, routeID, trunkID, why string) {
	if err := stream.DeleteRoute(ctx, routeID); err != nil {
		s.logger.Error("could not delete a routing rule", "why", why, "route", routeID, "error", err)
	}
	if err := stream.DeleteTrunk(ctx, trunkID); err != nil {
		s.logger.Error("could not delete a trunk", "why", why, "trunk", trunkID, "error", err)
	}
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
	// Err is the error behind a vendor that failed, for a caller that must not repeat its
	// words. It is nil for a vendor that was not asked.
	Err error
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
		return Offers{}, stack.Wrap(errors.New("phone: no vendor has the credentials to be searched"))
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
				answers[index] = answer{skip: &Skip{Vendor: name, Reason: err.Error(), Err: err}}
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
		return store.PhoneNumber{}, stack.Wrap(errors.New("phone: a number must belong to a customer"))
	}
	if err := s.Suspended(ctx, purchase.Owner.CustomerID); err != nil {
		return store.PhoneNumber{}, err
	}
	// A number is charged every month from the day it is bought, so an organization that may
	// not spend money does not buy one.
	if err := s.gate.Admit(ctx, purchase.Owner.CustomerID); err != nil {
		return store.PhoneNumber{}, err
	}
	// An app Stream blocked could not use a number, so it does not get to pay for one.
	if s.apps != nil {
		if _, _, err := s.apps.For(ctx, purchase.Owner.CustomerID); errors.Is(err, streamapp.ErrStreamAppDisconnected) {
			return store.PhoneNumber{}, stack.Wrap(err)
		}
	}
	// Refused before the vendor is paid, since a number in another form would not be recorded.
	if !store.IsE164(purchase.E164) {
		return store.PhoneNumber{}, stack.Wrap(fmt.Errorf("phone: %w: %q", store.ErrNotE164, purchase.E164))
	}
	provider, err := s.registry.Open(purchase.Vendor)
	if err != nil {
		return store.PhoneNumber{}, err
	}

	// Some vendors say what a number costs when it is offered and not when it is bought, so
	// the offer is read first, while the number is still for sale.
	quoted := s.quote(ctx, provider, purchase)
	// A failed purchase is a row of its own; a bought number is billed by its months, the
	// first of them below.
	buying := routing.Meter{
		Recorder: s.recorder,
		Config:   routing.ProviderConfig{Provider: purchase.Vendor, Model: ModelNumber},
		Owner:    purchase.Owner,
		Code:     vendorFailed,
		Omit:     func(pricing.Quantities) bool { return true },
	}
	bought, err := routing.Call(ctx, buying, func(ctx context.Context) (Number, error) {
		return provider.BuyNumber(ctx, Order{E164: purchase.E164, Country: purchase.Country})
	})
	if err != nil {
		return store.PhoneNumber{}, err
	}
	if bought.MonthlyCostMicros == 0 {
		bought.MonthlyCostMicros = quoted
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
		// To the microsecond Postgres keeps, so a month billed now and one billed from the
		// stored row name the same purchase.
		PurchasedAt: s.now().UTC().Truncate(time.Microsecond),
	}
	// The month it was bought in is charged at once, whether or not the row below is kept.
	s.billMonth(held, purchase.Owner, held.PurchasedAt)
	if s.store == nil {
		return held, nil
	}
	if err := s.store.RecordNumber(ctx, &held); err != nil {
		// The number is bought either way, so saying so is more useful than failing and
		// leaving the caller thinking it was not.
		return held, stack.Wrap(fmt.Errorf("phone: %s was bought but not recorded: %w", held.E164, err))
	}
	return held, nil
}

// Release gives a number back and stops the monthly charge.
func (s *Service) Release(ctx context.Context, customerID, e164 string) error {
	held, err := s.number(ctx, customerID, e164, "releasing")
	if err != nil {
		return err
	}
	// Callers stop first: a vendor that refuses fails the release before anything changes,
	// so it can be tried again.
	if err := s.detach(ctx, held); err != nil {
		return err
	}
	// Off its trunk first: a trunk Stream would not delete fails the release before the
	// number is given back, so the release can be tried again.
	if err := s.unline(ctx, held, "released a number", false); err != nil {
		return err
	}
	seller, err := s.seller(held.Vendor)
	if err != nil {
		return err
	}
	// A number on the customer's own trunk was never bought here, so there is nobody to
	// give it back to.
	if seller != nil {
		if err := seller.ReleaseNumber(ctx, e164); err != nil {
			return err
		}
	}
	// The trunk and rule the number was attached with would otherwise outlive it, billed
	// and pointing a number nobody holds at a call.
	return s.unline(ctx, held, "released a number", true)
}

// Detach stops a number's callers reaching the agent. The number keeps its trunk and rule,
// so its calls still go out on them and attaching it again picks them up.
func (s *Service) Detach(ctx context.Context, customerID, e164 string) error {
	held, err := s.number(ctx, customerID, e164, "detaching")
	if err != nil {
		return err
	}
	return s.detach(ctx, held)
}

// number is a number the customer holds, looked up for doing something to it.
func (s *Service) number(ctx context.Context, customerID, e164, doing string) (store.PhoneNumber, error) {
	if s.store == nil {
		return store.PhoneNumber{}, stack.Wrap(fmt.Errorf("phone: %s a number needs a database to know who holds it", doing))
	}
	return s.store.Number(ctx, customerID, e164)
}

// detach stops the vendor sending a held number's callers, then forgets it was attached.
func (s *Service) detach(ctx context.Context, held store.PhoneNumber) error {
	seller, err := s.seller(held.Vendor)
	if err != nil {
		return err
	}
	// The vendor first: one that refuses still sends callers, so the number stays attached
	// for them and the detach can be tried again. A number on the customer's own trunk was
	// never pointed anywhere by a vendor here.
	if stopper, ok := seller.(InboundStopper); ok {
		if err := stopper.StopInbound(ctx, held.E164); err != nil {
			return err
		}
	}
	group := store.TrunkGroup{CustomerID: held.CustomerID, StreamAppPK: held.StreamAppPK, Vendor: held.Vendor}
	return s.inTrunkGroup(ctx, group, func(ctx context.Context, tx *store.TrunkGroupTx) error {
		number, err := tx.Number(held.E164)
		if err != nil {
			return err
		}
		if number.AttachedAt == nil {
			return nil
		}
		return tx.Detach(held.E164)
	})
}

// seller is the vendor that sold a number, or none for one on the customer's own trunk,
// which was never bought here.
func (s *Service) seller(vendor string) (Provider, error) {
	if vendor == SIPTrunkVendor {
		return nil, nil
	}
	return s.registry.Open(vendor)
}

// unline takes a number off its trunk and deletes its rule, in the app they are in, and
// forgets them on the number. With release it also gives the number back under the same
// lock, so a line made for it meanwhile cannot put it back on a trunk. A Stream failure is
// logged, not returned, so the number is still taken off, save a trunk Stream would not
// delete (see offTrunk).
func (s *Service) unline(ctx context.Context, held store.PhoneNumber, why string, release bool) error {
	app := held.StreamAppPK
	// A call or attach lining the number in another app while this waited for its lock
	// moves it there, and a release follows it.
	for range 3 {
		var moved bool
		group := store.TrunkGroup{CustomerID: held.CustomerID, StreamAppPK: app, Vendor: held.Vendor}
		// Found before the lock: reaching the app may read the database, which under the lock
		// would take a second pooled connection.
		stream, reached := s.streamToRemove(ctx, held.CustomerID, app)
		err := s.inTrunkGroup(ctx, group, func(ctx context.Context, tx *store.TrunkGroupTx) error {
			number, err := tx.Number(held.E164)
			if err != nil {
				return err
			}
			if number.StreamTrunkID != "" && number.StreamAppPK != app {
				moved, app = true, number.StreamAppPK
				return nil
			}
			if number.StreamTrunkID != "" {
				if err := s.offTrunk(ctx, tx, stream, reached, number, why); err != nil {
					return err
				}
			}
			if release {
				return tx.ReleaseNumber(number.E164, time.Now().UTC())
			}
			return nil
		})
		if err != nil || !moved || !release {
			return err
		}
	}
	return stack.Wrap(fmt.Errorf("phone: %s kept moving between apps while it was released", held.E164))
}

// offTrunk takes a number off its trunk and deletes its rule. The trunk is deleted with its
// last number; one the number had to itself, made before trunks were shared, is not a row
// and is deleted with it. A trunk Stream would not delete keeps its row, which holds its
// password and is what a retry deletes it by.
func (s *Service) offTrunk(ctx context.Context, tx *store.TrunkGroupTx, stream *Stream, reached error, number store.PhoneNumber, why string) error {
	if reached != nil {
		stream = nil
		s.logger.Error("could not reach the Stream app a line was made in", "why", why,
			"trunk", number.StreamTrunkID, "route", number.StreamRouteID, "error", reached)
	}
	trunks, err := tx.Trunks()
	if err != nil {
		return err
	}
	i := slices.IndexFunc(trunks, func(t store.StreamTrunk) bool { return t.ID == number.StreamTrunkID })
	if i < 0 {
		if stream != nil {
			s.deleteLines(ctx, stream, number.StreamRouteID, number.StreamTrunkID, why)
		}
		return tx.ClearLine(number.E164)
	}
	if stream != nil {
		if err := stream.DeleteRoute(ctx, number.StreamRouteID); err != nil {
			s.logger.Error("could not delete a routing rule", "why", why, "route", number.StreamRouteID, "error", err)
		}
	}
	trunk := trunks[i]
	trunk.Numbers = slices.DeleteFunc(slices.Clone(trunk.Numbers), func(n string) bool { return n == number.E164 })
	if len(trunk.Numbers) == 0 {
		if stream != nil {
			if err := stream.DeleteTrunk(ctx, trunk.ID); err != nil {
				return err
			}
		}
		if err := tx.DeleteTrunk(trunk.ID); err != nil {
			return err
		}
		return tx.ClearLine(number.E164)
	}
	allowedIPs, err := s.allowlistFor(number.Vendor)
	if err != nil {
		return err
	}
	// Stored as it should be either way: the trunk's next update resends its numbers,
	// which mends one Stream failed to take off.
	if stream != nil {
		err := stream.UpdateTrunk(ctx, trunk.ID, Trunk{
			Name: sharedTrunkName(number.Vendor, number.CustomerID), Numbers: trunk.Numbers, AllowedIPs: allowedIPs,
		})
		if err != nil {
			s.logger.Error("could not take a number off a trunk", "why", why, "trunk", trunk.ID, "error", err)
		}
	}
	if err := tx.SaveTrunk(trunk); err != nil {
		return err
	}
	return tx.ClearLine(number.E164)
}

// Attachment points a number at the agent: every caller lands in a call of their own,
// agent:<uuid>, and the session answering them is opened under that uuid.
type Attachment struct {
	CustomerID string
	E164       string
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

// NumberKey is the custom field an inbound call carries the number it rang under, which is
// how the call that arrives is traced back to the number: its name is the session's.
const NumberKey = "phone_number"

// ErrNoSealer says this deployment has no keyring to seal or open a number's trunk password.
var ErrNoSealer = errors.New("phone: no key to seal a number's trunk password: set ROUTER_AUTH_KEK")

// Attach puts a number on the trunk it shares with the customer's other numbers of its
// vendor, with a rule of its own, and tells the vendor to send calls there. This is what
// turns a bought number into one that reaches an agent. Without a key to seal the trunk's
// password, the number gets a trunk of its own. Attaching a number already attached in this
// app changes nothing and answers what it has.
func (s *Service) Attach(ctx context.Context, attachment Attachment) (Attached, error) {
	if s.store == nil {
		return Attached{}, stack.Wrap(errors.New("phone: attaching a number needs a database to know who holds it"))
	}

	held, err := s.store.Number(ctx, attachment.CustomerID, attachment.E164)
	if err != nil {
		return Attached{}, stack.Wrap(err)
	}
	if held.Vendor == SIPTrunkVendor {
		return Attached{}, stack.Wrap(errors.New("phone: inbound calls to a number on the customer's own sip trunk are not supported"))
	}
	stream, pin, err := s.streamFor(ctx, attachment.CustomerID)
	if errors.Is(err, errNoStream) {
		return Attached{}, stack.Wrap(errors.New("phone: attaching a number needs stream credentials"))
	}
	if err != nil {
		return Attached{}, err
	}
	provider, err := s.registry.Open(held.Vendor)
	if err != nil {
		return Attached{}, stack.Wrap(err)
	}
	declared, _ := s.registry.Lookup(held.Vendor)
	// Refused before anything is made: attached, the number would look fine and never answer.
	if declared.AsksForTrunk && s.sealer == nil {
		return Attached{}, stack.Wrap(fmt.Errorf("phone: %s numbers are answered with their stored trunk: %w",
			held.Vendor, ErrNoSealer))
	}

	// The same request is sent on every run of a program waiting for calls, so a number
	// already attached in this app is answered with what it has.
	if held.AttachedAt != nil && held.StreamTrunkID != "" && held.StreamAppPK == pin {
		return Attached{TrunkID: held.StreamTrunkID, RouteID: held.StreamRouteID,
			Bridge: Bridge{URI: held.BridgeURI}}, nil
	}
	// A number lined in another app is taken off there first: the customer's app is where
	// its calls go now.
	if held.StreamTrunkID != "" && held.StreamAppPK != pin {
		if err := s.unline(ctx, held, "re-attached a number in another app", false); err != nil {
			return Attached{}, err
		}
		held.StreamTrunkID, held.StreamRouteID = "", ""
	}

	if s.sealer != nil {
		line, err := s.line(ctx, stream, pin, held)
		if err == nil {
			attached := Attached{TrunkID: line.TrunkID, RouteID: line.RouteID, Bridge: line.Bridge}
			// The line stays when this fails: the number's calls go out on it.
			if err := s.finishAttach(ctx, provider, attachment, pin, attached); err != nil {
				return Attached{}, err
			}
			return attached, nil
		}
		if !errors.Is(err, store.ErrNoStoredBridge) {
			return Attached{}, stack.Wrap(err)
		}
		// Attached before its trunk's password was kept, so its trunk cannot be shared: it is
		// replaced by one of its own, which keeps it.
		if err := s.unline(ctx, held, "re-attached a number", false); err != nil {
			return Attached{}, err
		}
	} else if held.StreamTrunkID != "" {
		// Still on a trunk of its own (detached, or lined for calls out), and Stream refuses a
		// second trunk naming it.
		if err := s.unline(ctx, held, "re-attached a number", false); err != nil {
			return Attached{}, err
		}
	}

	allowedIPs, err := trunkAllowlist(declared)
	if err != nil {
		return Attached{}, stack.Wrap(err)
	}

	trunkID, bridge, err := stream.CreateTrunk(ctx, Trunk{
		Name:       "phone-" + attachment.E164,
		Numbers:    []string{attachment.E164},
		AllowedIPs: allowedIPs,
	})
	if err != nil {
		return Attached{}, stack.Wrap(err)
	}

	var routeID string
	committed := false
	defer func() {
		if committed {
			return
		}
		// Roll back what this attach created but did not finish wiring up, so a failed
		// attach does not leave a billable Stream trunk behind. Best-effort: a cleanup
		// error is logged, never returned, so it cannot mask the real failure.
		s.deleteLines(ctx, stream, routeID, trunkID, "rolling back a failed attach")
	}()

	// Every caller is a conversation of their own, so each lands in a call named for the
	// session that answers them. The name is only known once Stream renders it, so the
	// number is put on the call, where the hook announcing it finds it.
	routeID, err = stream.CreateRoute(ctx, Route{
		Name:          "phone-" + attachment.E164,
		TrunkIDs:      []string{trunkID},
		CalledNumbers: []string{attachment.E164},
		CallID:        lineCallTemplate,
		CallType:      defaultCallType,
		Custom:        map[string]string{NumberKey: attachment.E164},
	})
	if err != nil {
		return Attached{}, stack.Wrap(err)
	}

	bridge, err = bridge.WithNumber(attachment.E164)
	if err != nil {
		return Attached{}, stack.Wrap(err)
	}
	attached := Attached{TrunkID: trunkID, RouteID: routeID, Bridge: bridge}
	if err := s.finishAttach(ctx, provider, attachment, pin, attached); err != nil {
		return Attached{}, err
	}
	committed = true
	return attached, nil
}

// finishAttach tells the vendor to send the number's calls to its trunk, and records that
// they go there.
func (s *Service) finishAttach(ctx context.Context, provider Provider, attachment Attachment, pin int64, attached Attached) error {
	record := store.NumberAttachment{
		TrunkID: attached.TrunkID, RouteID: attached.RouteID, StreamAppPK: pin, CallType: defaultCallType,
	}
	// The address is not secret and a repeat attach answers it. A vendor that asks per call
	// where to send a caller needs the password then. Without a key it is not kept, rather
	// than kept in the clear.
	record.BridgeURI, record.BridgeUsername = attached.Bridge.URI, attached.Bridge.Username
	if s.sealer != nil {
		sealed, err := s.sealer.SealWithAAD(attached.Bridge.Password, bridgeAAD(attachment.CustomerID, attachment.E164))
		if err != nil {
			return stack.Wrap(err)
		}
		record.BridgePasswordSealed, record.BridgePasswordKEKVersion = sealed, s.sealer.CurrentVersion()
	}

	err := provider.ConfigureInbound(ctx, Inbound{E164: attachment.E164, Bridge: attached.Bridge})
	if err != nil {
		return stack.Wrap(err)
	}
	// The vendor has already been told to send calls to the trunk. There is no primitive to
	// un-configure a provider's inbound routing, so a failure here leaves the vendor
	// pointing at a trunk that, when it is the number's own, is about to be rolled back.
	if err := s.store.AttachNumber(ctx, attachment.CustomerID, attachment.E164, record); err != nil {
		return stack.Wrap(err)
	}
	return nil
}

// InboundBridge is the Stream trunk calls to an attached number of the vendor go to, with
// its password opened.
func (s *Service) InboundBridge(ctx context.Context, vendor, e164 string) (Bridge, error) {
	if s.store == nil {
		return Bridge{}, stack.Wrap(errors.New("phone: finding where a number's calls go needs a database"))
	}
	if s.sealer == nil {
		return Bridge{}, stack.Wrap(ErrNoSealer)
	}
	stored, err := s.store.AttachedBridge(ctx, vendor, e164)
	if err != nil {
		return Bridge{}, err
	}
	if err := s.Suspended(ctx, stored.CustomerID); err != nil {
		return Bridge{}, err
	}
	// The vendor bills the minutes of a call it answers as surely as one placed, and an agent
	// on it is billed work too, so a call to an organization that may not pay is not answered.
	if err := s.gate.Admit(ctx, stored.CustomerID); err != nil {
		return Bridge{}, err
	}
	password, err := s.sealer.OpenWithAADVersion(stored.PasswordSealed, bridgeAAD(stored.CustomerID, e164), stored.KEKVersion)
	if err != nil {
		return Bridge{}, stack.Wrap(fmt.Errorf("phone: open the trunk password of %s: %w", e164, err))
	}
	return Bridge{URI: stored.URI, Username: stored.Username, Password: password}, nil
}

// CallRequest is a call to place from one of the customer's numbers.
type CallRequest struct {
	Owner routing.Owner
	// From is one of the customer's own numbers, which is what the person sees.
	From string
	// To is who to call.
	To string
	// SessionID is the session that holds the call: the answered leg is routed into its
	// call, agent:<session id>. Empty chooses one, because two calls from the same number
	// are two conversations and must not land in the same place.
	SessionID string
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
	// SessionID is the session to open on the call, with voice started: the answered leg
	// is routed into agent:<session id>, and an agent that is not in it hears nothing when
	// the person picks up.
	SessionID string
	// CallID and CallType are that call.
	CallID   string
	CallType string
	// StreamApp is the app that call is in, which the agent has to join it in.
	StreamApp int64
}

// Call places an outbound call and bridges it into a Stream call.
//
// Stream's SIP is inbound only, so the vendor originates the call and connects it to a
// trunk, rather than Stream dialling out. The trunk alone is not enough: without a routing
// rule the answered leg arrives with nothing pointing it at a call, so this creates both and
// pins the rule to the call the agent is waiting in, the way Transfer does. A vendor that
// names the call on its leg goes out on the number's shared line instead, since Stream lets a
// number be on one trunk only.
func (s *Service) Call(ctx context.Context, request CallRequest) (Placed, error) {
	if err := s.Suspended(ctx, request.Owner.CustomerID); err != nil {
		return Placed{}, err
	}
	if request.To != "" {
		if err := Dialable(request.To); err != nil {
			return Placed{}, err
		}
	}
	if err := s.gate.Allow(ctx, request.Owner.CustomerID, dlc.Voice, request.To); err != nil {
		return Placed{}, stack.Wrap(err)
	}
	stream, pin, err := s.streamFor(ctx, request.Owner.CustomerID)
	if errors.Is(err, errNoStream) {
		return Placed{}, stack.Wrap(errors.New("phone: placing a call needs stream credentials"))
	}
	if err != nil {
		return Placed{}, err
	}
	if s.store == nil {
		return Placed{}, stack.Wrap(errors.New("phone: placing a call needs a database to know who holds the number"))
	}
	if request.To == "" {
		return Placed{}, stack.Wrap(errors.New("phone: a call needs someone to call"))
	}

	held, err := s.store.Number(ctx, request.Owner.CustomerID, request.From)
	if err != nil {
		return Placed{}, stack.Wrap(err)
	}
	provider, trunk, err := s.dialVendor(ctx, held)
	if err != nil {
		return Placed{}, stack.Wrap(err)
	}

	outbound := Outbound{
		From:          request.From,
		To:            request.To,
		RingTimeout:   request.RingTimeout,
		InitialDigits: request.InitialDigits,
		Headers:       request.Headers,
		Trunk:         trunk,
	}
	// A term the vendor cannot express is refused rather than dropped: a call placed
	// without the ring timeout that was asked for is not the call that was asked for.
	if missing := outbound.Unsupported(provider); len(missing) > 0 {
		return Placed{}, stack.Wrap(fmt.Errorf("phone: %s cannot place a call with %s",
			held.Vendor, joinFeatures(missing)))
	}

	allowedIPs, err := s.allowlistFor(held.Vendor)
	if err != nil {
		return Placed{}, stack.Wrap(err)
	}

	sessionID := request.SessionID
	if sessionID == "" {
		sessionID = uuid.NewString()
	} else if held, ok := conversation.SessionID(sessionID); ok {
		sessionID = held
	} else {
		return Placed{}, stack.Wrap(fmt.Errorf("phone: the session id %q is not one a session can have", sessionID))
	}
	callID, callType := sessionID, defaultCallType

	line, shared, err := s.sharedLine(ctx, stream, pin, provider, held)
	if err != nil {
		return Placed{}, stack.Wrap(err)
	}
	var trunkID, routeID string
	committed := false
	if shared {
		// The number's rule carries no custom data, so a call that has some is created with
		// it here, before the leg arrives; otherwise the leg creates the call.
		if len(request.Custom) > 0 {
			if err := stream.CreateCall(ctx, callType, callID, request.Custom); err != nil {
				return Placed{}, err
			}
		}
		trunkID, routeID = line.TrunkID, line.RouteID
		outbound.Bridge, outbound.CallID = line.Bridge, callID
	} else {
		var bridge Bridge
		trunkID, bridge, err = stream.CreateTrunk(ctx, Trunk{
			Name:       "call-" + callID,
			Numbers:    []string{request.From},
			AllowedIPs: allowedIPs,
		})
		if err != nil {
			return Placed{}, stack.Wrap(err)
		}

		defer func() {
			if committed {
				return
			}
			// Roll back what this call created but never placed, so a call that did not
			// start does not leave a billable Stream trunk behind. Best-effort: a cleanup
			// error is logged, never returned, so it cannot mask the real failure.
			s.deleteLines(ctx, stream, routeID, trunkID, "rolling back a call that did not place")
		}()

		routeID, err = stream.CreateRoute(ctx, Route{
			Name:          "call-" + callID,
			TrunkIDs:      []string{trunkID},
			CalledNumbers: []string{request.From},
			CallID:        callID,
			CallType:      callType,
			Custom:        request.Custom,
		})
		if err != nil {
			return Placed{}, stack.Wrap(err)
		}

		outbound.Bridge, err = bridge.WithNumber(request.From)
		if err != nil {
			return Placed{}, stack.Wrap(err)
		}
	}
	if err := outbound.Validate(); err != nil {
		return Placed{}, stack.Wrap(err)
	}
	// Three vendors will not take the plan on this request and fetch one on answer, so
	// they get somewhere to fetch it from instead.
	if _, hosted := provider.(AnswerRenderer); hosted {
		outbound.AnswerURL, err = s.park(ctx, held.Vendor, request.Owner.CustomerID, callID, outbound)
		if err != nil {
			return Placed{}, stack.Wrap(err)
		}
	}

	// Only a vendor the router can be reached by can post how the call ended. A call through
	// a customer's own trunk is told by the router's own bridge.
	_, reader := provider.(StatusReader)
	reports := reader && s.publicURL != ""
	token, err := s.outcomes.Placed(ctx, outcome.Call{
		Key: outcome.Key{
			CustomerID: request.Owner.CustomerID, StreamAppPK: pin, CallType: callType, CallID: callID,
			Local: request.From, Remote: request.To,
		},
		Vendor:  held.Vendor,
		Reports: reports || trunk != nil,
	})
	if err != nil {
		return Placed{}, err
	}
	if reports {
		outbound.StatusURL = s.publicURL + "/v1/phone/status/" + token
	}

	if trunk != nil {
		customerID := request.Owner.CustomerID
		outbound.Connected = func(media sipbridge.MediaAddrs) {
			s.recordBridgeMedia(context.WithoutCancel(ctx), customerID, sessionID, media)
		}
	}

	resource := &store.CallResource{
		TrunkID:     trunkID,
		RouteID:     routeID,
		CallType:    callType,
		CallID:      callID,
		CustomerID:  request.Owner.CustomerID,
		StreamAppPK: pin,
		Shared:      shared,
	}
	// A shared line outlives the call, so its row costs nothing to undo when the dial fails,
	// and the call's app (CallPin) is known before its leg arrives.
	if shared {
		if err := s.store.RecordCallResource(ctx, resource); err != nil {
			s.forgetOutcome(ctx, token, callID)
			return Placed{}, stack.Wrap(err)
		}
	}

	dialed, err := routing.Call(ctx, s.dialMeter(held.Vendor, request.Owner), func(ctx context.Context) (Dialed, error) {
		return provider.Dial(ctx, outbound)
	})
	if err != nil {
		if shared {
			s.forgetCallResource(ctx, resource)
		}
		s.forgetOutcome(ctx, token, callID)
		return Placed{}, stack.Wrap(err)
	}

	committed = true
	if err := s.outcomes.Dialed(context.WithoutCancel(ctx), token, dialed.VendorCallID); err != nil {
		s.logger.Error("could not keep a placed call's vendor id", "call", callID,
			"vendor_call_id", dialed.VendorCallID, "error", err)
	}
	// The vendor bills the call from when the person answers, which is when their phone joins
	// the Stream call: the leg rings until then, and one never answered is not billed.
	s.OpenLeg(ctx, request.Owner, store.CallLeg{
		StreamAppPK: pin, CallType: callType, CallID: callID, Vendor: held.Vendor,
		Direction: store.Outbound, LocalNumber: request.From, RemoteNumber: request.To,
	}, false)
	sites := placedSites{streamSIPHost: BridgeHost(outbound.Bridge.URI)}
	if trunk != nil {
		sites.customerTrunkHost = trunk.Host
	}
	s.logger.Info("placed a phone call", append([]any{"session_id", sessionID,
		"vendor_call_id", dialed.VendorCallID, "vendor", held.Vendor}, sites.logged()...)...)
	if !shared {
		if err := s.store.RecordCallResource(ctx, resource); err != nil {
			// The call is placed and connecting; deleting its trunk now would drop a live
			// call. A logged, rare leak beats that. session_ended cannot clean a trunk it
			// has no row for, which is the cost of this failing.
			s.logger.Error("could not record a call's trunk for cleanup; it may leak",
				"trunk", trunkID, "call", callID, "error", err)
		}
	}
	s.recordPlaced(context.WithoutCancel(ctx), request.Owner.CustomerID, sessionID, dialed, held.Vendor, sites)
	return Placed{
		VendorCallID: dialed.VendorCallID,
		Status:       dialed.Status,
		Vendor:       held.Vendor,
		SessionID:    sessionID,
		CallID:       callID,
		CallType:     callType,
		StreamApp:    pin,
	}, nil
}

// forgetOutcome drops the row kept for a call that never rang, which has no outcome. It runs
// on a request that may be gone, and logs a failure rather than mask the one it follows.
func (s *Service) forgetOutcome(ctx context.Context, token, callID string) {
	if err := s.outcomes.Failed(context.WithoutCancel(ctx), token); err != nil {
		s.logger.Error("could not forget a call that did not place", "call", callID, "error", err)
	}
}

// placedSites is where a placed call was sent: the Stream SIP host, and the host of the
// customer's own trunk when the call went out through one. Never the trunk's username or
// password.
type placedSites struct {
	streamSIPHost, customerTrunkHost string
}

func (p placedSites) details() map[string]any {
	details := map[string]any{"stream_sip_host": p.streamSIPHost}
	if p.customerTrunkHost != "" {
		details["customer_trunk_host"] = p.customerTrunkHost
	}
	return details
}

func (p placedSites) logged() []any {
	logged := []any{"stream_sip_host", p.streamSIPHost}
	if p.customerTrunkHost != "" {
		logged = append(logged, "customer_trunk_host", p.customerTrunkHost)
	}
	return logged
}

// recordPlaced keeps where a placed call was sent, found later by its session id. The call
// is already ringing, so a failure is logged and goes no further, and a slow store cannot
// hold the reply for longer than agentLogTimeout.
func (s *Service) recordPlaced(
	ctx context.Context,
	customerID, sessionID string,
	placed Dialed,
	vendor string,
	sites placedSites,
) {
	details := sites.details()
	details["vendor"] = vendor
	details["vendor_call_id"] = placed.VendorCallID
	ctx, cancel := context.WithTimeout(ctx, agentLogTimeout)
	defer cancel()
	err := s.store.RecordAgentLog(ctx, &store.AgentLog{
		CustomerID: customerID,
		SessionID:  sessionID,
		Source:     "system",
		Severity:   "info",
		EventType:  "call_placed",
		Message:    "Call placed",
		Details:    details,
	})
	if err != nil {
		s.logger.Warn("could not record a placed call", "session_id", sessionID, "error", err)
	}
}

// agentLogTimeout bounds writing a call_placed or bridge_media row. The node waits for the
// bridge_media row when it stops.
const agentLogTimeout = 5 * time.Second

// recordBridgeMedia keeps where each side of a call through a customer's own SIP trunk
// sends its audio from, as the router's bridge saw in their SDP. It is the only place the
// Stream side's real address shows: the SIP host's DNS answer depends on who asks.
func (s *Service) recordBridgeMedia(ctx context.Context, customerID, sessionID string, media sipbridge.MediaAddrs) {
	if media.Stream == "" && media.Customer == "" {
		return
	}
	ctx, cancel := context.WithTimeout(ctx, agentLogTimeout)
	defer cancel()
	details := map[string]any{}
	if media.Stream != "" {
		details["stream_media_addr"] = media.Stream
	}
	if media.Customer != "" {
		details["customer_media_addr"] = media.Customer
	}
	err := s.store.RecordAgentLog(ctx, &store.AgentLog{
		CustomerID: customerID,
		SessionID:  sessionID,
		Source:     "system",
		Severity:   "info",
		EventType:  "bridge_media",
		Message:    "Bridge media",
		Details:    details,
	})
	if err != nil {
		s.logger.Warn("could not record a bridged call's media addresses", "session_id", sessionID, "error", err)
	}
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
		return "", stack.Wrap(fmt.Errorf(
			"phone: %s fetches its call plan when the person answers, so it needs a "+
				"public url to fetch it from, and none is configured", vendor))
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

// ErrUnknownStatus is a status post for no call waiting for one: its token is unknown or swept,
// or its vendor reports no status.
var ErrUnknownStatus = errors.New("phone: that call is not waiting for its status")

// ErrUnreadableStatus is a status post its vendor's provider could not read.
var ErrUnreadableStatus = errors.New("phone: that status could not be read")

// ReportStatus takes what a vendor posted to the status URL of the call token was made for.
func (s *Service) ReportStatus(ctx context.Context, token, contentType string, body []byte) error {
	if s.outcomes == nil {
		return stack.Wrap(ErrUnknownStatus)
	}
	call, err := s.outcomes.ByToken(ctx, token)
	if errors.Is(err, outcome.ErrUnknownCall) {
		return stack.Wrap(ErrUnknownStatus)
	}
	if err != nil {
		return err
	}
	provider, err := s.registry.Open(call.Vendor)
	if err != nil {
		return err
	}
	reader, ok := provider.(StatusReader)
	if !ok {
		return stack.Wrap(fmt.Errorf("%w: %s reports no status", ErrUnknownStatus, call.Vendor))
	}
	status, err := reader.ReadStatus(contentType, body)
	if err != nil {
		return stack.Wrap(fmt.Errorf("%w: %w", ErrUnreadableStatus, err))
	}
	_, err = s.report(ctx, token, status)
	return err
}

// report hands a vendor's status for the call token was made for to the keeper, which decides
// what it means.
func (s *Service) report(ctx context.Context, token string, status CallStatus) (outcome.Call, error) {
	reported, err := outcomeOf(status.Result)
	if err != nil {
		return outcome.Call{}, err
	}
	call, err := s.outcomes.Reported(ctx, token, reported, status.Word)
	if errors.Is(err, outcome.ErrUnknownCall) {
		return outcome.Call{}, stack.Wrap(ErrUnknownStatus)
	}
	return call, err
}

// SweepBridges removes the bridges left behind by calls nobody answered.
func (s *Service) SweepBridges(ctx context.Context) (int64, error) {
	if s.store == nil {
		return 0, nil
	}
	return s.store.SweepBridges(ctx)
}

// statusTokenLifetime is how long a placed call's status URL takes posts: past the longest a
// call may last, with an hour for a vendor's late post.
const statusTokenLifetime = MaxCallLength + time.Hour

// SweepStatusTokens retires the status URLs of calls placed longer than statusTokenLifetime
// ago, keeping their outcomes. It reports how many it retired.
func (s *Service) SweepStatusTokens(ctx context.Context) (int64, error) {
	if s.outcomes == nil {
		return 0, nil
	}
	return s.outcomes.Sweep(ctx, s.now().Add(-statusTokenLifetime))
}

// ReleaseCall deletes the per-call Stream trunks and routes an ended call left behind in one
// app, each in the app it was made in. A call's id is only unique within an app, so the
// event that ended it releases only what that app holds; unpinned also releases what was
// made before apps had identities. It is safe to call for any call: one with no per-call
// resources (an inbound or video call) releases nothing. Deletes are best-effort so a
// cleanup failure cannot wedge the event that triggered it.
func (s *Service) ReleaseCall(ctx context.Context, scope store.AppScope, callType, callID string) error {
	if s.store == nil || (s.stream == nil && s.apps == nil) {
		return nil
	}
	resources, err := s.store.ReleaseCallResourcesInApp(ctx, scope, callType, callID)
	if err != nil {
		return fmt.Errorf("phone: release call resources: %w", err)
	}
	for _, resource := range resources {
		if resource.Shared {
			continue
		}
		s.unwire(ctx, resource.CustomerID, resource.StreamAppPK, resource.RouteID, resource.TrunkID, "a call ended")
	}
	return nil
}

// SweepCallResources forgets the calls older than before whose end hook never arrived, and
// unwires their per-call trunks. It returns how many calls' rows it forgot. A shared line
// belongs to its number, not the call, so it is left at Stream.
//
// A per-call row is forgotten only once Stream deleted its trunk, so a trunk Stream kept is
// tried again on the next sweep. Two routers sweeping at once may both delete one trunk,
// and the one that loses only logs it.
func (s *Service) SweepCallResources(ctx context.Context, before time.Time) (int, error) {
	if s.store == nil || (s.stream == nil && s.apps == nil) {
		return 0, nil
	}
	stale, err := s.store.StaleCallResources(ctx, before)
	if err != nil {
		return 0, fmt.Errorf("phone: sweep call resources: %w", err)
	}
	var forgot int
	for _, resource := range stale {
		if !resource.Shared {
			stream, err := s.streamToRemove(ctx, resource.CustomerID, resource.StreamAppPK)
			if err == nil {
				if err := stream.DeleteRoute(ctx, resource.RouteID); err != nil {
					s.logger.Error("could not delete a routing rule", "why", "a call never ended", "route", resource.RouteID, "error", err)
				}
				err = stream.DeleteTrunk(ctx, resource.TrunkID)
			}
			if err != nil {
				s.logger.Error("could not delete the trunk of a call that never ended",
					"trunk", resource.TrunkID, "route", resource.RouteID, "error", err)
				continue
			}
		}
		if err := s.store.ForgetCallResource(ctx, resource); err != nil {
			return forgot, fmt.Errorf("phone: sweep call resources: %w", err)
		}
		forgot++
	}
	return forgot, nil
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
		return nil, stack.Wrap(fmt.Errorf(
			"phone: %s cannot send a password to a trunk, so the trunk has to know its "+
				"signalling addresses, and none are declared for it",
			vendor.Vendor))
	}
	return vendor.Signalling, nil
}

// dialVendor is who places a call from a held number, and the customer's own trunk the
// call goes through when the number is on one.
func (s *Service) dialVendor(ctx context.Context, held store.PhoneNumber) (Provider, *SIPTrunk, error) {
	provider, err := s.provider(held.Vendor)
	if err != nil || held.Vendor != SIPTrunkVendor {
		return provider, nil, err
	}
	if err := s.sipTrunksReady(); err != nil {
		return nil, nil, err
	}
	trunk, err := s.openSIPTrunk(ctx, held.CustomerID, held.SIPTrunkID)
	if err != nil {
		return nil, nil, err
	}
	return provider, trunk, nil
}

// allowlistFor is the addresses a per-call Stream trunk should accept the vendor's leg
// from. The bridge to a customer's own trunk presents the Stream trunk's password, so it
// needs none.
func (s *Service) allowlistFor(vendor string) ([]string, error) {
	if vendor == SIPTrunkVendor {
		return nil, nil
	}
	declared, _ := s.registry.Lookup(vendor)
	return trunkAllowlist(declared)
}

// sharedLine is the line a call from the number goes out on, or false when the call makes a
// trunk and rule of its own: the vendor cannot name the call on its leg, there is no key to
// seal a shared trunk's password with, the number was attached before its password was
// kept, or it is attached in another app. A number lined in another app only to place calls
// moves its line here.
func (s *Service) sharedLine(ctx context.Context, stream *Stream, pin int64, provider Provider, held store.PhoneNumber) (TrunkLine, bool, error) {
	if StreamLegOf(provider) != StreamLegShared {
		return TrunkLine{}, false, nil
	}
	// An attached number stays where its callers are sent. Stream lets a number be on one
	// trunk per app, so the call can still have one of its own here.
	if held.StreamTrunkID != "" && held.StreamAppPK != pin && held.AttachedAt == nil {
		if err := s.unline(ctx, held, "lined a number in another app", false); err != nil {
			return TrunkLine{}, false, err
		}
	}
	line, err := s.line(ctx, stream, pin, held)
	if errors.Is(err, ErrNoSealer) || errors.Is(err, store.ErrNoStoredBridge) || errors.Is(err, errLinedElsewhere) {
		return TrunkLine{}, false, nil
	}
	if err != nil {
		return TrunkLine{}, false, err
	}
	return line, true, nil
}

// forgetCallResource deletes the row of a leg on a shared line that was never placed.
// Best-effort: a failure is logged, never returned, so it cannot mask the dial's own.
func (s *Service) forgetCallResource(ctx context.Context, resource *store.CallResource) {
	if err := s.store.DeleteCallResource(ctx, resource.ID); err != nil {
		s.logger.Error("could not delete the row of a leg that did not place",
			"trunk", resource.TrunkID, "call", resource.CallID, "error", err)
	}
}

// provider is the provider of a vendor name, which for sip_trunk is not in the registry.
func (s *Service) provider(vendor string) (Provider, error) {
	if vendor != SIPTrunkVendor {
		return s.registry.Open(vendor)
	}
	if s.sipTrunks == nil {
		return nil, stack.Wrap(ErrSIPTrunksDisabled)
	}
	return s.sipTrunks, nil
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
	// StreamApp is the app that call is in, the session's pin: zero for the deployment's.
	StreamApp int64
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
// making a second trunk. A vendor that names the call on its leg uses the number's shared
// line instead.
func (s *Service) Transfer(ctx context.Context, request TransferRequest) (Dialed, error) {
	if err := s.Suspended(ctx, request.Owner.CustomerID); err != nil {
		return Dialed{}, err
	}
	if request.To != "" {
		if err := Dialable(request.To); err != nil {
			return Dialed{}, err
		}
	}
	// A transfer dials someone as surely as a call does, so it answers to the same gate; the
	// leg it transfers is already up, and only the human's counts as a new call.
	into := dlc.Call{Type: cmp.Or(request.CallType, defaultCallType), ID: request.CallID}
	if err := s.gate.AllowTransfer(ctx, request.Owner.CustomerID, request.To, into); err != nil {
		return Dialed{}, stack.Wrap(err)
	}
	// The human is routed into the call the agent is on, so the trunk is made in the app
	// that call is in.
	stream, err := s.streamForApp(ctx, request.Owner.CustomerID, request.StreamApp)
	if errors.Is(err, errNoStream) {
		return Dialed{}, stack.Wrap(errors.New("phone: transferring a call needs stream credentials"))
	}
	if err != nil {
		return Dialed{}, err
	}
	if s.store == nil {
		return Dialed{}, stack.Wrap(errors.New("phone: transferring a call needs a database to know who holds the number"))
	}
	if request.CallID == "" {
		return Dialed{}, stack.Wrap(errors.New("phone: a transfer needs the call to transfer into"))
	}
	if request.To == "" {
		return Dialed{}, stack.Wrap(errors.New("phone: a transfer needs someone to transfer to"))
	}

	held, err := s.store.Number(ctx, request.Owner.CustomerID, request.From)
	if err != nil {
		return Dialed{}, stack.Wrap(err)
	}
	provider, trunk, err := s.dialVendor(ctx, held)
	if err != nil {
		return Dialed{}, stack.Wrap(err)
	}

	allowedIPs, err := s.allowlistFor(held.Vendor)
	if err != nil {
		return Dialed{}, stack.Wrap(err)
	}

	// Resolve the call type the same way CreateRoute does, so the row recorded below
	// carries the concrete type that session_ended's call_cid will name. A transfer joins
	// an existing call whose type was fixed at creation; storing the raw empty type would
	// not match "default" at cleanup and would leak the transfer's trunk.
	callType := request.CallType
	if callType == "" {
		callType = defaultCallType
	}

	var line TrunkLine
	var shared bool
	// The number's shared rule routes into defaultCallType only, so a call of another type
	// takes a rule of its own pinned to it.
	if callType == defaultCallType {
		line, shared, err = s.sharedLine(ctx, stream, request.StreamApp, provider, held)
		if err != nil {
			return Dialed{}, stack.Wrap(err)
		}
	}
	var trunkID, routeID string
	var bridge Bridge
	committed := false
	if shared {
		trunkID, routeID, bridge = line.TrunkID, line.RouteID, line.Bridge
	} else {
		trunkID, bridge, err = stream.CreateTrunk(ctx, Trunk{
			Name:       "transfer-" + request.CallID,
			Numbers:    []string{request.From},
			AllowedIPs: allowedIPs,
		})
		if err != nil {
			return Dialed{}, stack.Wrap(err)
		}

		defer func() {
			if committed {
				return
			}
			// Roll back what this transfer created but never placed, so a transfer that did
			// not start does not leave a billable Stream trunk behind. Best-effort: a cleanup
			// error is logged, never returned, so it cannot mask the real failure.
			s.deleteLines(ctx, stream, routeID, trunkID, "rolling back a transfer that did not place")
		}()

		routeID, err = stream.CreateRoute(ctx, Route{
			Name:          "transfer-" + request.CallID,
			TrunkIDs:      []string{trunkID},
			CalledNumbers: []string{request.From},
			CallID:        request.CallID,
			CallType:      callType,
		})
		if err != nil {
			return Dialed{}, stack.Wrap(err)
		}

		bridge, err = bridge.WithNumber(request.From)
		if err != nil {
			return Dialed{}, stack.Wrap(err)
		}
	}

	resource := &store.CallResource{
		TrunkID:     trunkID,
		RouteID:     routeID,
		CallType:    callType,
		CallID:      request.CallID,
		CustomerID:  request.Owner.CustomerID,
		StreamAppPK: request.StreamApp,
		Shared:      shared,
	}
	if shared {
		if err := s.store.RecordCallResource(ctx, resource); err != nil {
			return Dialed{}, stack.Wrap(err)
		}
	}

	outbound := Outbound{From: request.From, To: request.To, Bridge: bridge, Trunk: trunk}
	if shared {
		outbound.CallID = request.CallID
	}
	placed, err := routing.Call(ctx, s.dialMeter(held.Vendor, request.Owner), func(ctx context.Context) (Dialed, error) {
		return provider.Dial(ctx, outbound)
	})
	if err != nil {
		if shared {
			s.forgetCallResource(ctx, resource)
		}
		return Dialed{}, stack.Wrap(err)
	}

	committed = true
	// The human's leg is a leg of the call it joins, billed to whoever transferred from when
	// they answer.
	s.OpenLeg(ctx, request.Owner, store.CallLeg{
		StreamAppPK: request.StreamApp, CallType: callType, CallID: request.CallID, Vendor: held.Vendor,
		Direction: store.Outbound, LocalNumber: request.From, RemoteNumber: request.To,
	}, false)
	if !shared {
		if err := s.store.RecordCallResource(ctx, resource); err != nil {
			// The transfer leg is placed and connecting; deleting its trunk now would drop
			// a live call. A logged, rare leak beats that. session_ended cannot clean a
			// trunk it has no row for, which is the cost of this failing.
			s.logger.Error("could not record a transfer's trunk for cleanup; it may leak",
				"trunk", trunkID, "call", request.CallID, "error", err)
		}
	}
	return placed, nil
}

// SendDigits presses digits on a call this service placed, which is how an agent answers a
// menu it has reached.
//
// The call is named by the id its vendor gave when it was dialled. An inbound call has no
// such id here, because it arrived at Stream rather than being made from this service, so
// only calls placed from here can be pressed at.
func (s *Service) SendDigits(ctx context.Context, vendor, vendorCallID, digits string) error {
	if vendorCallID == "" {
		return stack.Wrap(errors.New("phone: pressing digits needs the call to press them on"))
	}
	if err := ValidateDigits(digits); err != nil {
		return err
	}
	provider, err := s.provider(vendor)
	if err != nil {
		return err
	}
	return provider.SendDigits(ctx, vendorCallID, digits)
}

// Numbers returns what a customer holds.
func (s *Service) Numbers(ctx context.Context, customerID string, includeReleased bool) ([]store.PhoneNumber, error) {
	if s.store == nil {
		return nil, stack.Wrap(errors.New("phone: listing numbers needs a database"))
	}
	return s.store.CustomerNumbers(ctx, customerID, includeReleased)
}

// names renders capabilities for storage, since Postgres holds them as plain text.
func names(capabilities []Capability) []string {
	rendered := make([]string, 0, len(capabilities))
	for _, capability := range capabilities {
		rendered = append(rendered, string(capability))
	}
	return rendered
}
