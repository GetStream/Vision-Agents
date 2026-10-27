//go:build integration

package api

import (
	"encoding/json"
	"net/http"
	"slices"
	"strings"
	"time"
)

// readPrices reads the price list as client, with the raw answer.
func (s *PriceListsSuite) readPrices(client *testClient, query string) (Prices, string) {
	status, payload := client.call(http.MethodGet, "/v1/prices?"+query, nil)
	s.Require().Equal(http.StatusOK, status, string(payload))
	var read Prices
	s.Require().NoError(json.Unmarshal(payload, &read))
	return read, string(payload)
}

// priceOfModel is the entry of a t1c model among prices, or false.
func priceOfModel(list []Price, model string) (Price, bool) {
	at := slices.IndexFunc(list, func(price Price) bool { return price.Model == model })
	if at < 0 {
		return Price{}, false
	}
	return list[at], true
}

// cancel calls off a scheduled change.
func (s *PriceListsSuite) cancel(name string) {
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodPost, "/v1/manage/price-lists/changes/"+name+"/cancel",
		CancelPriceChangeInput{Reason: "Only a test."}, nil))
}

// tomorrow is the change scheduled a day from now, created.
func (s *PriceListsSuite) tomorrow(input PriceChangeInput) PriceChange {
	at := time.Now().Add(24 * time.Hour).UTC().Truncate(time.Second)
	input.EffectiveAt = &at
	scheduled := s.created(input)
	s.T().Cleanup(func() { s.cancel(scheduled.Name) })
	return scheduled
}

func (s *PriceListsSuite) TestPricesAreTheListInForce() {
	model := "m" + s.token()
	now := s.created(s.change(entry(model, 1)))
	later := s.change(entry(model, 1.5))
	later.Reason = "The vendor raises it tomorrow."
	scheduled := s.tomorrow(later)

	read, _ := s.readPrices(s.serverClient, "modality=llm")

	inForce, ok := priceOfModel(read.Entries, model)
	s.Require().True(ok)
	s.Equal(now.Name, inForce.Change)
	s.Require().NotNil(inForce.EffectiveAt)
	coming, ok := priceOfModel(read.Upcoming, model)
	s.Require().True(ok, "the change for tomorrow is upcoming only")
	s.Equal(scheduled.Name, coming.Change)
	s.Equal(later.Reason, coming.Description)
	s.Require().NotNil(coming.EffectiveAt)
	s.True(coming.EffectiveAt.Equal(scheduled.EffectiveAt))
	for _, price := range read.Entries {
		s.Equal("llm", price.Modality, "the modality asked for alone")
	}

	counted := s.created(s.change(countedAt(3, 0.3, 15)))
	read, _ = s.readPrices(s.serverClient, "")
	price, ok := priceOfModel(read.Entries, "counted-model")
	s.Require().True(ok)
	s.Equal(counted.Name, price.Change)
	s.runCounted()
	for _, row := range s.countedRows() {
		s.Equal(price.Change, row.PriceVersion, "the rows name the change the read named")
	}
}

func (s *PriceListsSuite) TestPricesAreWhatTheCustomerPays() {
	s.setMarkup(2000)
	model := "m" + s.token()
	s.created(s.change(entry(model, 4)))
	cancelled := s.change(entry(model, 9))
	cancelled.EffectiveAt = pointerTo(time.Now().Add(48 * time.Hour).UTC().Truncate(time.Second))
	called := s.created(cancelled)
	s.cancel(called.Name)
	markup := s.change()
	markup.MarkupBps = pointerTo(int64(3000))
	remarked := s.tomorrow(markup)

	read, _ := s.readPrices(s.serverClient, "")

	inForce, ok := priceOfModel(read.Entries, model)
	s.Require().True(ok)
	s.Equal(4.8, inForce.Rates["input_tokens"].USD, "$4 per million with the list's 20%")
	s.Equal(int64(1_000_000), inForce.Rates["input_tokens"].Per)
	s.False(slices.ContainsFunc(read.Upcoming, func(price Price) bool { return price.Change == called.Name }),
		"a cancelled change is not upcoming")
	moved, ok := priceOfModel(read.Upcoming, model)
	s.Require().True(ok, "a markup change shows as every entry it moves")
	s.Equal(remarked.Name, moved.Change)
	s.Equal(5.2, moved.Rates["input_tokens"].USD)
}

func (s *PriceListsSuite) TestPricesLeaveOutWhatOnlyStaffRead() {
	s.setMarkup(2000)
	model := "m" + s.token()
	written := s.created(s.change(entry(model, 1.25)))

	read, raw := s.readPrices(s.serverClient, "")

	for _, price := range read.Entries {
		s.NotEqual("echo-model", price.Model, "a routed model nobody priced is not a price")
	}
	inForce, ok := priceOfModel(read.Entries, model)
	s.Require().True(ok)
	s.Equal(1.5, inForce.Rates["input_tokens"].USD, "the charged rate, never the vendor's 1.25")
	for _, staffOnly := range []string{"markup", "unpriced", "vendor.example", written.CreatedBy, `"source"`, "created_by", "charged_"} {
		s.False(strings.Contains(raw, staffOnly), "the answer says %q", staffOnly)
	}
}

func (s *PriceListsSuite) TestAUserTokenReadsPrices() {
	s.created(s.change(entry("m"+s.token(), 2)))

	server, _ := s.readPrices(s.serverClient, "")
	device, _ := s.readPrices(s.client, "")

	s.NotEmpty(device.Entries)
	s.Equal(server.Entries, device.Entries)
	s.Equal(server.Upcoming, device.Upcoming)
}
