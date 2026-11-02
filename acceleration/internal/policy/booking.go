package policy

import (
	"context"
	_ "embed"
	"fmt"
	"math"
	"strconv"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// BookingTTL is how long a booking holds its amount unless it is charged or released first, so
// a router that dies mid-purchase lets go of it within the hour.
const BookingTTL = time.Hour

// ChargedTTL is how long a charged booking still counts: past it the usage service has counted
// the charge's rows, so the standing a booking reads already holds them.
const ChargedTTL = time.Minute

// bookingTimeout bounds one call to the shared Redis about bookings, and the standing read.
const bookingTimeout = 2 * time.Second

// bookingPrefix starts the key of an organization's bookings, in the organization's cluster
// slot: booking:{<organization>}.
const bookingPrefix = "booking"

//go:embed booking.lua
var bookingSource string

var bookingScript = rueidis.NewLuaScript(bookingSource)

var (
	bookingsRefused = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "bookings_refused_total",
		Help:      "Purchases refused before their vendor was asked, by why: no_credit, spend_hard_limit, suspended, unchecked",
	}, []string{"reason"})
	bookingsExpired = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "bookings_expired_total",
		Help:      "Bookings that expired neither charged nor released, as a router that stopped mid-purchase leaves them",
	})
)

// errBookingUnchecked is a booking refused because the organization's money could not be read.
var errBookingUnchecked = uncheckedWith(ErrSpendCheckUnavailable,
	"the organization's credit could not be checked, so the purchase was not made; try again shortly")

// Book sets micros of the owner's organization's credit aside for a purchase, before its vendor
// is asked, for at most BookingTTL. It reads the organization's standing from the usage service
// in the call, in every limits mode, and refuses a suspended organization (ErrSuspended), one
// whose credit less its holds and other bookings does not cover the amount (ErrNoCredits), or
// at least a micro of it for a purchase with no known price, and one the amount would take past
// Stream's spend hard limit (ErrSystemLimit). A standing or a Redis that cannot be read refuses
// it as unchecked (ErrSpendCheckUnavailable): a purchase can wait. With limits on, the booking
// is held as work is too, so the rest of the organization's work counts it, and the customer's
// limits refuse it as their mode says.
//
// An owner in no organization, and a router with no usage service, book nothing.
func (e *Enforcer) Book(ctx context.Context, owner routing.Owner, micros int64) (routing.Booked, error) {
	if e == nil || owner.CustomerID == "" || e.settings.Usage == nil {
		return routing.NoBooking, nil
	}
	attributed, err := e.Attribute(ctx, owner.CustomerID)
	if err != nil {
		return e.bookingRefused("", "unchecked", errBookingUnchecked, err)
	}
	organization := attributed.OrganizationID
	if organization == "" {
		return routing.NoBooking, nil
	}
	read, cancel := context.WithTimeout(ctx, bookingTimeout)
	standings, err := e.settings.Usage.Standings(read, usage.StandingQuery{Organizations: []usage.StandingOrganization{
		{OrganizationID: organization, OrganizationTier: attributed.OrganizationTier},
	}})
	cancel()
	if err != nil || len(standings) != 1 {
		return e.bookingRefused(organization, "unchecked", errBookingUnchecked, err)
	}
	standing := standings[0]
	if standing.Status == usage.AccountSuspended {
		return e.bookingRefused(organization, "suspended", ErrSuspended, nil)
	}

	held := e.money.heldFor(organization)
	credit := standing.BalanceMicros - held
	hard := hardLimitRoom(standing, e.settings.Now()) - held
	need := max(micros, 1)
	id := uuid.Must(uuid.NewV7())
	booked, others, expired, err := e.bookings(organization).book(ctx, id, micros, min(credit, hard), need)
	if expired > 0 {
		bookingsExpired.Add(float64(expired))
		e.logger.Warn("bookings expired neither charged nor released, as a router that stopped mid-purchase leaves them",
			append(who(organization), "expired", expired)...)
	}
	if err != nil {
		return e.bookingRefused(organization, "unchecked", errBookingUnchecked, err)
	}
	if !booked {
		if credit-others < need {
			return e.bookingRefused(organization, "no_credit", refusedWith(ErrNoCredits,
				"the organization has %s of credit to spend, and this costs %s; add credit first",
				dollars(max(credit-others, 0)), dollars(need)), nil)
		}
		return e.bookingRefused(organization, "spend_hard_limit", refusedWith(ErrSystemLimit,
			"this costs %s, which would take the organization past its spend hard limit", dollars(need)), nil)
	}

	b := &booking{enforcer: e, organization: organization, id: id}
	if !e.money.off() && micros > 0 {
		if err := e.Reserve(owner, id, micros, BookingTTL); err != nil {
			b.Release()
			return routing.NoBooking, err
		}
		b.reserved = true
	}
	return b, nil
}

// bookingRefused counts and answers a refused booking, logging why one went unchecked.
func (e *Enforcer) bookingRefused(organization, reason string, refusal, cause error) (routing.Booked, error) {
	bookingsRefused.WithLabelValues(reason).Inc()
	if cause != nil && e.money.refusals.due(organization+"\x00booking_"+reason, logEvery) {
		e.logger.Error("could not check an organization's credit for a purchase, so it was refused",
			append(who(organization), "error", cause)...)
	}
	return routing.NoBooking, refusal
}

// heldFor is what the organization's holds set aside, this router's and the others', as far as
// this process knows them: nothing with limits off, which holds nothing. A booking's own hold is
// counted here and as a booking too, which errs towards refusing while it is held.
func (m *money) heldFor(organization string) int64 {
	if m.off() {
		return 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	acct := m.accounts[organization]
	if acct == nil {
		return 0
	}
	return acct.held + acct.remoteOrganization(organization)
}

// hardLimitRoom is what the organization may still spend under Stream's spend hard limit, the
// tighter of the day's and the month's; MaxInt64 when it sets none.
func hardLimitRoom(standing usage.Standing, now time.Time) int64 {
	room := int64(math.MaxInt64)
	for _, interval := range []usage.Interval{usage.Daily, usage.Monthly} {
		limit, ok := standing.Limits.SpendHardLimitMicros(interval)
		if !ok {
			continue
		}
		window := standing.SpendWindows.Daily
		if interval == usage.Monthly {
			window = standing.SpendWindows.Monthly
		}
		start, _ := usage.Window(interval, now)
		spent := int64(0)
		if window.WindowStart.Equal(start) {
			spent = window.SpentMicros
		}
		room = min(room, limit-spent)
	}
	return room
}

// booking is credit Book set aside.
type booking struct {
	enforcer     *Enforcer
	organization string
	id           uuid.UUID
	reserved     bool
	done         sync.Once
}

// Charged keeps the booking counted for ChargedTTL, until the usage service has the purchase's
// rows, and lets go of its hold, since those rows hold for themselves.
func (b *booking) Charged() {
	b.done.Do(func() { b.finish("charge") })
}

// Release frees the booking at once.
func (b *booking) Release() {
	b.done.Do(func() { b.finish("release") })
}

func (b *booking) finish(verb string) {
	e := b.enforcer
	if b.reserved {
		e.Release(b.id)
	}
	if err := e.bookings(b.organization).finish(context.Background(), b.id, verb); err != nil {
		e.logger.Warn("could not settle a booking, which expires on its own",
			append(who(b.organization), "verb", verb, "expires_in", BookingTTL, "error", err)...)
	}
}

// bookingStore counts one organization's bookings: in the shared Redis, or in this process
// when there is none.
type bookingStore interface {
	book(ctx context.Context, id uuid.UUID, micros, room, need int64) (booked bool, others int64, expired int, err error)
	finish(ctx context.Context, id uuid.UUID, verb string) error
}

// bookings is where the organization's bookings are counted.
func (e *Enforcer) bookings(organization string) bookingStore {
	if e.settings.Holds != nil {
		return redisBookings{client: e.settings.Holds, key: bookingPrefix + ":{" + organization + "}"}
	}
	return e.localBookings.of(organization)
}

// redisBookings are an organization's bookings in the shared Redis (booking.lua).
type redisBookings struct {
	client rueidis.Client
	key    string
}

func (r redisBookings) call(ctx context.Context, args ...string) ([]int64, error) {
	call, cancel := context.WithTimeout(ctx, bookingTimeout)
	defer cancel()
	answered, err := bookingScript.Exec(call, r.client, []string{r.key}, args).AsIntSlice()
	if err != nil {
		return nil, err
	}
	if len(answered) != 3 {
		return nil, fmt.Errorf("policy: %d numbers in an answer about bookings, not 3", len(answered))
	}
	return answered, nil
}

func (r redisBookings) book(ctx context.Context, id uuid.UUID, micros, room, need int64) (bool, int64, int, error) {
	answered, err := r.call(ctx, id.String(), "book", strconv.FormatInt(micros, 10),
		strconv.FormatInt(BookingTTL.Milliseconds(), 10), strconv.FormatInt(room, 10), strconv.FormatInt(need, 10))
	if err != nil {
		return false, 0, 0, err
	}
	return answered[0] == 1, answered[1], int(answered[2]), nil
}

func (r redisBookings) finish(ctx context.Context, id uuid.UUID, verb string) error {
	args := []string{id.String(), verb}
	if verb == "charge" {
		args = append(args, strconv.FormatInt(ChargedTTL.Milliseconds(), 10))
	}
	_, err := r.call(ctx, args...)
	return err
}

// localBookings are every organization's bookings in this process, for a router with no
// shared Redis, which then counts only its own.
type localBookings struct {
	mu   sync.Mutex
	held map[string]map[uuid.UUID]localBooking
}

type localBooking struct {
	micros  int64
	expires time.Time
	charged bool
}

func (l *localBookings) of(organization string) bookingStore {
	return localOrganization{bookings: l, organization: organization}
}

// localOrganization is one organization's bookings in this process.
type localOrganization struct {
	bookings     *localBookings
	organization string
}

// live drops the organization's expired bookings and answers its others' sum, how many it
// dropped never charged, and its bookings, under the lock.
func (o localOrganization) live(id uuid.UUID, now time.Time) (int64, int, map[uuid.UUID]localBooking) {
	l := o.bookings
	if l.held == nil {
		l.held = map[string]map[uuid.UUID]localBooking{}
	}
	held := l.held[o.organization]
	if held == nil {
		held = map[uuid.UUID]localBooking{}
		l.held[o.organization] = held
	}
	var others int64
	expired := 0
	for other, b := range held {
		switch {
		case !b.expires.After(now):
			delete(held, other)
			if !b.charged {
				expired++
			}
		case other != id:
			others += b.micros
		}
	}
	return others, expired, held
}

func (o localOrganization) book(_ context.Context, id uuid.UUID, micros, room, need int64) (bool, int64, int, error) {
	o.bookings.mu.Lock()
	defer o.bookings.mu.Unlock()
	now := time.Now()
	others, expired, held := o.live(id, now)
	if room-others < need {
		return false, others, expired, nil
	}
	held[id] = localBooking{micros: micros, expires: now.Add(BookingTTL)}
	return true, others, expired, nil
}

func (o localOrganization) finish(_ context.Context, id uuid.UUID, verb string) error {
	o.bookings.mu.Lock()
	defer o.bookings.mu.Unlock()
	now := time.Now()
	_, _, held := o.live(id, now)
	b, ok := held[id]
	if !ok {
		return nil
	}
	if verb == "release" {
		delete(held, id)
		return nil
	}
	b.charged = true
	if expires := now.Add(ChargedTTL); expires.Before(b.expires) {
		b.expires = expires
	}
	held[id] = b
	return nil
}
