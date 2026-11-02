//go:build integration

package phone

import (
	"context"
	"errors"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// errShort is the refusal of a booking the credit does not cover.
var errShort = errors.New("not enough credit")

// bookings is money that books out of a credit of its own, as the enforcer does, and says what
// was booked, charged and released.
type bookings struct {
	mu       sync.Mutex
	credit   int64
	booked   []int64
	charged  int
	released int
}

func (b *bookings) Admit(context.Context, routing.Owner) (routing.Admission, error) {
	return routing.Admission{}, nil
}

func (b *bookings) Attribute(context.Context, string) (routing.Admission, error) {
	return routing.Admission{}, nil
}

func (b *bookings) Book(_ context.Context, _ routing.Owner, micros int64) (routing.Booked, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if max(micros, 1) > b.credit {
		return nil, errShort
	}
	b.booked = append(b.booked, micros)
	return &booked{of: b}, nil
}

type booked struct {
	of   *bookings
	once sync.Once
}

func (b *booked) Charged() {
	b.once.Do(func() { b.of.mu.Lock(); b.of.charged++; b.of.mu.Unlock() })
}

func (b *booked) Release() {
	b.once.Do(func() { b.of.mu.Lock(); b.of.released++; b.of.mu.Unlock() })
}

// booking has the service's gate book out of credit, with the number to buy offered at a
// dollar a month.
func (s *MeteringSuite) booking(credit int64) (*bookings, string) {
	money := &bookings{credit: credit}
	s.service.gate.Charging(money)
	e164 := aNumber()
	s.vendor.offers = []Available{{E164: e164, MonthlyCostMicros: 1_000_000}}
	return money, e164
}

func (s *MeteringSuite) TestANumberIsBookedItsFirstMonthBeforeItIsBought() {
	money, e164 := s.booking(1_200_000)

	_, err := s.service.Buy(s.ctx, Purchase{Vendor: s.vendor.vendor, E164: e164, Owner: s.owner()})
	s.Require().NoError(err)

	s.Equal([]int64{1_200_000}, money.booked, "the first month as its row is priced: the offer resold at the factor")
	s.Equal(1, money.charged, "charged once the month is billed")
	s.Zero(money.released)
	s.Equal(int64(1_200_000), s.row(ModelNumber).CostMicros)
}

func (s *MeteringSuite) TestANumberTheCreditDoesNotCoverIsNotBought() {
	money, e164 := s.booking(1_199_999)

	_, err := s.service.Buy(s.ctx, Purchase{Vendor: s.vendor.vendor, E164: e164, Owner: s.owner()})
	s.Require().ErrorIs(err, dlc.ErrUnpaid)
	s.Require().ErrorIs(err, errShort)

	s.Zero(s.vendor.ordered, "the vendor is never asked")
	s.Empty(money.booked)
	numbers, err := s.store.CustomerNumbers(s.ctx, s.customer, false)
	s.Require().NoError(err)
	for _, number := range numbers {
		s.NotEqual(e164, number.E164)
	}
}

func (s *MeteringSuite) TestAPurchaseTheVendorRefusesReleasesItsBooking() {
	money, e164 := s.booking(5_000_000)
	s.vendor.err = errors.New("sold out")

	_, err := s.service.Buy(s.ctx, Purchase{Vendor: s.vendor.vendor, E164: e164, Owner: s.owner()})
	s.Require().Error(err)

	s.Len(money.booked, 1)
	s.Equal(1, money.released)
	s.Zero(money.charged)
}

func (s *MeteringSuite) TestANumberWithNoQuoteStillNeedsSomeCredit() {
	money, e164 := s.booking(0)
	s.vendor.offers = nil

	_, err := s.service.Buy(s.ctx, Purchase{Vendor: s.vendor.vendor, E164: e164, Owner: s.owner()})
	s.Require().ErrorIs(err, dlc.ErrUnpaid)
	s.Zero(s.vendor.ordered)

	money.credit = 1
	_, err = s.service.Buy(s.ctx, Purchase{Vendor: s.vendor.vendor, E164: e164, Owner: s.owner()})
	s.Require().NoError(err)
	s.Equal([]int64{0}, money.booked, "no price to book, so only some credit is asked for")
}
