//go:build integration

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// PriceListsSuite changes the price list through the management API, on a router whose rows
// are priced from the database and read again every suitePriceReload.
//
// The price tables are the deployment's and nothing ever deletes from them, so the suite's
// database keeps every change every run made. A test prices an item of its own, or sets
// what it reads (the counted model's price) before it reads it.
type PriceListsSuite struct {
	RouterSuite
	// staff holds the manage key and names an actor, as Stream's proxy sends a staff
	// member's call.
	staff *testClient
}

func TestPriceListsSuite(t *testing.T) {
	runSuite(t, new(PriceListsSuite))
}

func (s *PriceListsSuite) SetupSuite() {
	s.pricedFromDatabase = true
	s.RouterSuite.SetupSuite()
}

func (s *PriceListsSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.staff = &testClient{suite: &s.RouterSuite, kind: noCredential, header: http.Header{
		manageKeyHeader: {suiteManageKey},
		actorIDHeader:   {"staff:" + s.utils.uuid()},
		actorNameHeader: {"Ada"},
	}}
}

// token is twelve characters of [a-f0-9] no other test has, for a name or an item: the random
// end of a uuid, whose start is the time.
func (s *PriceListsSuite) token() string {
	id := s.utils.uuid()
	return id[len(id)-12:]
}

// millionTokens is a token rate as vendors publish it.
func millionTokens(usd float64) PriceRate { return PriceRate{USD: usd, Per: 1_000_000} }

// entry is an LLM entry of provider t1c for model, its vendor's input rate and its price both
// usd per million tokens.
func entry(model string, usd float64) PriceEntryInput {
	return PriceEntryInput{
		PriceItem: PriceItem{Modality: "llm", Provider: "t1c", Model: model},
		Rates:     map[string]PriceRate{"input_tokens": millionTokens(usd)},
		Prices:    map[string]PriceRate{"input_tokens": millionTokens(usd)},
	}
}

// pricedAt is the entry charged price per million input tokens, whatever its rate.
func pricedAt(entry PriceEntryInput, price float64) PriceEntryInput {
	entry.Prices = map[string]PriceRate{"input_tokens": millionTokens(price)}
	return entry
}

// change is a change of the entries, confirmed, named for the test.
func (s *PriceListsSuite) change(entries ...PriceEntryInput) PriceChangeInput {
	return PriceChangeInput{Name: "test_" + s.token(), Reason: "The vendor published a new rate.",
		Source: "https://vendor.example/pricing, read 2026-10-23", Entries: entries, Confirm: true}
}

// create sends the change under a key of its own and answers the status and the change.
func (s *PriceListsSuite) create(input PriceChangeInput) (int, PriceChange) {
	return s.createUnder(s.utils.uuid(), input)
}

func (s *PriceListsSuite) createUnder(key string, input PriceChangeInput) (int, PriceChange) {
	var written PriceChange
	status := s.staff.with(idempotencyHeader, key).do(http.MethodPost, "/v1/manage/price-lists/changes", input, &written)
	return status, written
}

// created sends the change and requires it written.
func (s *PriceListsSuite) created(input PriceChangeInput) PriceChange {
	status, written := s.create(input)
	s.Require().Equal(http.StatusCreated, status)
	return written
}

// refused sends the change and answers the status and error code it was refused with.
func (s *PriceListsSuite) refused(input PriceChangeInput) (int, string) {
	status, payload := s.staff.with(idempotencyHeader, s.utils.uuid()).call(http.MethodPost, "/v1/manage/price-lists/changes", input)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered), string(payload))
	return status, answered.Error.Code
}

// active is the list in force now.
func (s *PriceListsSuite) active() PriceList {
	var list PriceList
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodGet, "/v1/manage/price-lists/active", nil, &list))
	return list
}

// changeCount is how many price changes the database holds.
func (s *PriceListsSuite) changeCount() int {
	count, err := s.store.DB().NewSelect().Model((*store.PriceChange)(nil)).Count(context.Background())
	s.Require().NoError(err)
	return count
}

// countedRows are the counted model's rows for the test's app, once at least one is written.
func (s *PriceListsSuite) countedRows() []store.Request {
	var rows []store.Request
	s.Require().Eventually(func() bool {
		rows = nil
		err := s.store.ShardDB().NewSelect().Model(&rows).
			Where("customer_id = ? AND provider = 'counted'", s.customerID()).Scan(context.Background())
		return err == nil && len(rows) > 0
	}, settleFor, 20*time.Millisecond)
	return rows
}

// runCounted runs an agent on the counted model, which answers 1,000 input tokens, 400 of them
// cached, and 20 output tokens.
func (s *PriceListsSuite) runCounted() {
	model := "counted/counted-model"
	var config AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name: "agent-" + s.utils.uuid(), Llm: &model, Instructions: pointerTo("be brief"),
	}, &config))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/runs",
		CreateRunRequest{ConfigId: config.Id, Input: "count me"}, nil))
}

// countedAt is the counted model at vendor rates of input, cached and output usd per million
// tokens, priced at factor times each.
func countedAt(input, cached, output, factor float64) PriceEntryInput {
	rates := func(factor float64) map[string]PriceRate {
		return map[string]PriceRate{"input_tokens": millionTokens(input * factor),
			"cached_input_tokens": millionTokens(cached * factor), "output_tokens": millionTokens(output * factor)}
	}
	return PriceEntryInput{
		PriceItem: PriceItem{Modality: "llm", Provider: "counted", Model: "counted-model"},
		Rates:     rates(1), Prices: rates(factor),
	}
}

func (s *PriceListsSuite) TestAChangePricesTheNextRequest() {
	written := s.created(s.change(countedAt(3, 0.3, 15, 1.25)))

	s.runCounted()

	s.Require().Len(written.Entries, 1)
	for _, row := range s.countedRows() {
		// 600 fresh tokens at $3, 400 cached at $0.30 and 20 generated at $15 per million.
		s.EqualValues(2_220, row.BaseCostMicros, "the vendor's cost, for reference")
		s.EqualValues(2_775, row.CostMicros, "at the price the change set")
		s.Equal(written.Name, row.PriceVersion)
		s.Equal(written.Entries[0].SKU, row.SKU)
	}
	s.Equal("llm-counted-counted-model", written.Entries[0].SKU)
}

// TestAQuoteChargesThePriceTheChangeSet: the price is set outright, not worked out from the
// cost, which only says what the work cost.
func (s *PriceListsSuite) TestAQuoteChargesThePriceTheChangeSet() {
	model := "m" + s.token()
	s.created(s.change(pricedAt(entry(model, 1), 1.3)))

	quote := s.prices.Quote(pricing.Item{Modality: "llm", Provider: "t1c", Model: model}, pricing.Quantities{pricing.InputTokens: 1_000_000})

	s.EqualValues(1_300_000, quote.CostMicros)
	s.EqualValues(1_000_000, quote.BaseCostMicros)
	s.Equal("llm-t1c-"+model, quote.SKU)
}

func (s *PriceListsSuite) TestAChangeForTomorrowDoesNotPriceToday() {
	model := "m" + s.token()
	s.created(s.change(entry(model, 1)))
	tomorrow := time.Now().Add(24 * time.Hour).UTC().Truncate(time.Second)
	later := s.change(entry(model, 1.5))
	later.EffectiveAt = &tomorrow
	scheduled := s.created(later)

	item := pricing.Item{Modality: "llm", Provider: "t1c", Model: model}
	vendor, ok := s.prices.Entry(item)
	s.Require().True(ok)
	s.Equal(1.0, vendor.Rates[pricing.InputTokens].USD)
	list := s.active()
	at := slices.IndexFunc(list.Upcoming, func(upcoming UpcomingPriceEntry) bool { return upcoming.Version == scheduled.Name })
	s.Require().GreaterOrEqual(at, 0, "the change is upcoming")
	s.True(list.Upcoming[at].EffectiveAt.Equal(tomorrow))
	s.Equal(later.Reason, list.Upcoming[at].Reason)
	s.Equal(1.5, list.Upcoming[at].Rates["input_tokens"].USD)
	s.Equal(1.5, list.Upcoming[at].Prices["input_tokens"].USD)
	s.Equal("llm-t1c-"+model, list.Upcoming[at].SKU, "the item keeps its SKU")
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodPost, "/v1/manage/price-lists/changes/"+scheduled.Name+"/cancel",
		CancelPriceChangeInput{Reason: "Only a test."}, nil))
}

func (s *PriceListsSuite) TestAQuotedCostIsResoldAtTheFactor() {
	customer := s.customerID()
	recorder := routing.NewRecorder(routing.Phone, s.store, nil, s.prices, slog.New(slog.DiscardHandler))

	recorder.Record(routing.ProviderConfig{Provider: "telnyx", Model: "number"}, routing.Stat{
		Owner:     routing.Owner{CustomerID: customer, Attribution: routing.Attribution{APIKey: s.app.key}},
		StartedAt: time.Now(), CostMicros: 1_000_000, Success: true,
	})
	recorder.Close()

	var row store.Request
	s.Require().NoError(s.store.ShardDB().NewSelect().Model(&row).Where("customer_id = ?", customer).Scan(context.Background()))
	s.EqualValues(1_200_000, row.CostMicros)
	s.EqualValues(1_000_000, row.BaseCostMicros)
	s.Equal(pricing.QuotedVersion, row.PriceVersion)
}

func (s *PriceListsSuite) TestProvidersShowThePrice() {
	s.created(s.change(countedAt(1, 0.5, 10, 1.25)))

	var providers []Provider
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/llm/providers", nil, &providers))

	at := slices.IndexFunc(providers, func(provider Provider) bool { return provider.Model == "counted-model" })
	s.Require().GreaterOrEqual(at, 0)
	s.Require().NotNil(providers[at].Price)
	s.Equal(pointerTo(1.25), providers[at].Price.PerMillionInputTokens)
	s.Equal(pointerTo(12.5), providers[at].Price.PerMillionOutputTokens)
}

func (s *PriceListsSuite) TestAnInvalidPriceIsRefused() {
	negative := entry("m"+s.token(), -1)
	unknown := entry("m"+s.token(), 1)
	unknown.Rates = map[string]PriceRate{"tokens": millionTokens(1)}
	local := entry("m"+s.token(), 1)
	local.Local = true
	versioned := entry("m"+s.token(), 1)
	versioned.Version = "20261023000000_mine"
	before := s.changeCount()

	unrated := entry("m"+s.token(), 1)
	unrated.Rates = nil
	freePriced := entry("m"+s.token(), 1)
	freePriced.Free, freePriced.Rates = true, nil
	dearPrice := pricedAt(entry("m"+s.token(), 1), 2_000_000_000)

	for _, invalid := range []PriceEntryInput{negative, unknown, local, versioned, unrated, freePriced, dearPrice} {
		status, code := s.refused(s.change(invalid))
		s.Equal(http.StatusBadRequest, status, invalid.Model)
		s.Equal(codeInvalidPrice, code, invalid.Model)
	}

	s.Equal(before, s.changeCount(), "nothing written")
}

// TestAPriceNotInTheShapeOfItsCostIsRefused: a price rates the meters its cost does, and has a
// tier at each of its cost's thresholds with the same meters.
func (s *PriceListsSuite) TestAPriceNotInTheShapeOfItsCostIsRefused() {
	unpriced := entry("m"+s.token(), 1)
	unpriced.Prices = nil
	otherMeters := entry("m"+s.token(), 1)
	otherMeters.Rates = map[string]PriceRate{"input_tokens": millionTokens(1), "output_tokens": millionTokens(4)}
	tiered := func(above ...int64) PriceEntryInput {
		tiered := entry("m"+s.token(), 1)
		tiered.Tiers = []PriceTier{{AboveInputTokens: 200_000, Rates: map[string]PriceRate{"input_tokens": millionTokens(2)}}}
		for _, threshold := range above {
			tiered.PriceTiers = append(tiered.PriceTiers, PriceTier{AboveInputTokens: threshold,
				Rates: map[string]PriceRate{"input_tokens": millionTokens(2.4)}})
		}
		return tiered
	}
	before := s.changeCount()

	for _, mismatched := range []PriceEntryInput{unpriced, otherMeters} {
		status, code := s.refused(s.change(mismatched))
		s.Equal(http.StatusBadRequest, status, mismatched.Model)
		s.Equal(codePriceMetersMismatch, code, mismatched.Model)
	}
	for _, mismatched := range []PriceEntryInput{tiered(), tiered(100_000), tiered(200_000, 300_000)} {
		status, code := s.refused(s.change(mismatched))
		s.Equal(http.StatusBadRequest, status, mismatched.Model)
		s.Equal(codePriceTiersMismatch, code, mismatched.Model)
	}
	s.Equal(before, s.changeCount(), "nothing written")
	s.created(s.change(tiered(200_000)))
}

// TestARenamedModelKeepsItsSKUThroughTheAPI: a new item may take the SKU of one the same change
// withdraws, and of no item in force else.
func (s *PriceListsSuite) TestARenamedModelKeepsItsSKUThroughTheAPI() {
	old, renamed := entry("m"+s.token(), 1), entry("m"+s.token(), 1)
	s.created(s.change(old))
	sku := "llm-t1c-" + old.Model
	renamed.SKU = sku

	status, code := s.refused(s.change(renamed))
	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeSKUInUse, code)
	var preview PricePreview
	s.Require().Equal(http.StatusBadRequest, s.staff.do(http.MethodPost, "/v1/manage/price-lists/changes/preview", s.change(renamed), &preview),
		"a preview refuses it as the change would")

	rename := s.change(renamed, PriceEntryInput{PriceItem: old.PriceItem, Withdrawn: true})
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodPost, "/v1/manage/price-lists/changes/preview", rename, &preview))
	shown := map[string]PricePreviewItem{}
	for _, item := range preview.Items {
		shown[item.Model] = item
	}
	s.Require().NotNil(shown[renamed.Model].After)
	s.Equal(sku, shown[renamed.Model].After.SKU)
	s.Require().NotNil(shown[old.Model].Before)
	s.Equal(sku, shown[old.Model].Before.SKU)
	written := s.created(rename)

	s.Require().Len(written.Entries, 2)
	s.Equal(sku, written.Entries[0].SKU)
	s.Equal(sku, written.Entries[1].SKU)
	quote := s.prices.Quote(pricing.Item{Modality: "llm", Provider: "t1c", Model: renamed.Model}, pricing.Quantities{pricing.InputTokens: 1})
	s.Equal(sku, quote.SKU)
}

func (s *PriceListsSuite) TestAnItemInForceKeepsItsSKUThroughTheAPI() {
	priced := entry("m"+s.token(), 1)
	s.created(s.change(priced))
	priced.SKU = "another-" + s.token()

	status, code := s.refused(s.change(priced))

	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeSKUChanged, code)
}

func (s *PriceListsSuite) TestAMisspeltFieldIsRefused() {
	body := map[string]any{
		"name": "test_" + s.token(), "reason": "A typo.", "source": "a test",
		"entries": []map[string]any{{"modality": "llm", "provider": "t1c", "model": "m" + s.token(),
			"rates": map[string]any{"input_tokens": map[string]any{"usdd": 4, "per": 1_000_000}}}},
	}
	before := s.changeCount()

	status := s.staff.with(idempotencyHeader, s.utils.uuid()).do(http.MethodPost, "/v1/manage/price-lists/changes", body, nil)

	s.Equal(http.StatusBadRequest, status, "never a rate of nothing")
	s.Equal(before, s.changeCount())
}

func (s *PriceListsSuite) TestAChangeWithNothingInItIsRefused() {
	status, _ := s.refused(s.change())

	s.Equal(http.StatusBadRequest, status)
}

func (s *PriceListsSuite) TestALargeMoveNeedsConfirmation() {
	model := "m" + s.token()
	s.created(s.change(entry(model, 1)))
	tenfold := s.change(entry(model, 10))
	tenfold.Confirm = false

	var preview PricePreview
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodPost, "/v1/manage/price-lists/changes/preview", tenfold, &preview))
	s.True(preview.ConfirmationRequired)
	s.Equal([]PriceItem{{Modality: "llm", Provider: "t1c", Model: model}}, preview.NeedsConfirmation)
	s.Require().Len(preview.Items, 1)
	s.InDelta(10, preview.Items[0].PriceFactors["input_tokens"], 1e-9)

	status, code := s.refused(tenfold)
	s.Equal(http.StatusConflict, status)
	s.Equal(codeConfirmationRequired, code)

	tenfold.Confirm = true
	status, _ = s.create(tenfold)
	s.Equal(http.StatusCreated, status)
}

// TestACostMovingAloneNeedsNoConfirmation: the rule is on what customers are charged, so a
// vendor's cost moving tenfold under an unchanged price asks for none.
func (s *PriceListsSuite) TestACostMovingAloneNeedsNoConfirmation() {
	model := "m" + s.token()
	s.created(s.change(entry(model, 1)))
	dearer := s.change(pricedAt(entry(model, 10), 1))
	dearer.Confirm = false

	var preview PricePreview
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodPost, "/v1/manage/price-lists/changes/preview", dearer, &preview))
	s.False(preview.ConfirmationRequired)
	s.Require().Len(preview.Items, 1)
	s.Equal(10.0, preview.Items[0].After.Rates["input_tokens"].USD)
	s.Equal(1.0, preview.Items[0].After.Prices["input_tokens"].USD)
	s.Equal(1.0, preview.Items[0].Before.Rates["input_tokens"].USD)
	s.InDelta(1, preview.Items[0].PriceFactors["input_tokens"], 1e-9)
	status, _ := s.create(dearer)
	s.Equal(http.StatusCreated, status)
}

func (s *PriceListsSuite) TestDroppingALongPromptTierNeedsConfirmation() {
	model := "m" + s.token()
	tiered := entry(model, 1)
	tiered.Tiers = []PriceTier{{AboveInputTokens: 200_000, Rates: map[string]PriceRate{"input_tokens": millionTokens(1.5)}}}
	tiered.PriceTiers = tiered.Tiers
	s.created(s.change(tiered))
	untiered := s.change(entry(model, 1))
	untiered.Confirm = false

	var preview PricePreview
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodPost, "/v1/manage/price-lists/changes/preview", untiered, &preview))

	s.True(preview.ConfirmationRequired, "a long prompt would be priced at the short-prompt rate")
	status, code := s.refused(untiered)
	s.Equal(http.StatusConflict, status)
	s.Equal(codeConfirmationRequired, code)
}

func (s *PriceListsSuite) TestDoublingAnIncrementNeedsConfirmation() {
	minutes := PriceEntryInput{PriceItem: PriceItem{Modality: "phone", Provider: "t1c", Model: "m" + s.token()},
		Rates:  map[string]PriceRate{"call_seconds": {USD: 0.01, Per: 60, Increment: 60}},
		Prices: map[string]PriceRate{"call_seconds": {USD: 0.012, Per: 60, Increment: 60}}}
	s.created(s.change(minutes))
	minutes.Prices = map[string]PriceRate{"call_seconds": {USD: 0.012, Per: 60, Increment: 120}}
	doubled := s.change(minutes)
	doubled.Confirm = false

	status, code := s.refused(doubled)

	s.Equal(http.StatusConflict, status)
	s.Equal(codeConfirmationRequired, code)
}

func (s *PriceListsSuite) TestABlankReasonOrSourceIsRefused() {
	blankReason, blankSource := s.change(entry("m"+s.token(), 1)), s.change(entry("m"+s.token(), 1))
	blankReason.Reason, blankSource.Source = "   ", "\t"

	for _, input := range []PriceChangeInput{blankReason, blankSource} {
		status, _ := s.refused(input)
		s.Equal(http.StatusBadRequest, status)
	}
}

func (s *PriceListsSuite) TestARateDearerThanAnyUnitIsRefused() {
	status, code := s.refused(s.change(entry("m"+s.token(), 2_000_000_000)))

	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeInvalidPrice, code)
}

func (s *PriceListsSuite) TestAChangeMoreThanTwoYearsAheadIsRefused() {
	threeYears := time.Now().Add(3 * 365 * 24 * time.Hour)
	input := s.change(entry("m"+s.token(), 1))
	input.EffectiveAt = &threeYears

	status, code := s.refused(input)

	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeValidationFailed, code)
}

// TestAChangeThatWaitsOutTheLockIsBusy: another session holds the lock every write to the price
// list takes. The change gives up after the lock timeout with a 503 to send again later.
func (s *PriceListsSuite) TestAChangeThatWaitsOutTheLockIsBusy() {
	held, err := s.store.DB().BeginTx(context.Background(), nil)
	s.Require().NoError(err)
	defer held.Rollback() //nolint:errcheck // rolled back below; this one is for a failed assertion
	_, err = held.ExecContext(context.Background(), "SELECT pg_advisory_xact_lock(hashtextextended('router.prices', 0))")
	s.Require().NoError(err)
	before := s.changeCount()

	request, err := http.NewRequest(http.MethodPost, s.server.URL+"/v1/manage/price-lists/changes",
		strings.NewReader(mustJSON(s.T(), s.change(entry("m"+s.token(), 1)))))
	s.Require().NoError(err)
	request.Header = s.staff.header.Clone()
	request.Header.Set(idempotencyHeader, s.utils.uuid())
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()

	s.Equal(http.StatusServiceUnavailable, response.StatusCode)
	s.Equal("5", response.Header.Get("Retry-After"))
	var answered ErrorResponse
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&answered))
	s.Equal(codePriceListBusy, answered.Error.Code)
	s.Require().NoError(held.Rollback())
	s.Equal(before, s.changeCount())
}

// mustJSON is value as JSON.
func mustJSON(t *testing.T, value any) string {
	encoded, err := json.Marshal(value)
	require.NoError(t, err)
	return string(encoded)
}

func (s *PriceListsSuite) TestASmallMoveNeedsNoConfirmation() {
	model := "m" + s.token()
	s.created(s.change(entry(model, 1)))
	raised := s.change(entry(model, 1.5))
	raised.Confirm = false

	status, _ := s.create(raised)

	s.Equal(http.StatusCreated, status)
}

func (s *PriceListsSuite) TestWithdrawingARoutedModelNeedsConfirmation() {
	noted := PriceItem{Modality: "llm", Provider: "noted", Model: "noted-model"}
	s.created(s.change(PriceEntryInput{PriceItem: noted, Rates: map[string]PriceRate{"input_tokens": millionTokens(1)},
		Prices: map[string]PriceRate{"input_tokens": millionTokens(1)}}))
	withdrawal := s.change(PriceEntryInput{PriceItem: noted, Withdrawn: true})
	withdrawal.Confirm = false

	var preview PricePreview
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodPost, "/v1/manage/price-lists/changes/preview", withdrawal, &preview))
	s.True(preview.ConfirmationRequired)
	s.Contains(preview.NeedsConfirmation, noted)
	s.Require().Len(preview.Items, 1)
	s.True(preview.Items[0].Unpriced)
	status, code := s.refused(withdrawal)
	s.Equal(http.StatusConflict, status)
	s.Equal(codeConfirmationRequired, code)

	withdrawal.Confirm = true
	status, _ = s.create(withdrawal)
	s.Equal(http.StatusCreated, status)
	s.Contains(s.active().Unpriced, noted)
}

func (s *PriceListsSuite) TestWithdrawingTheDefaultLeavesTheOtherEntries() {
	model := "m" + s.token()
	flex := entry(model, 0.5)
	flex.When = map[string]string{"service_tier": "flex"}
	s.created(s.change(entry(model, 1), flex))

	s.created(s.change(PriceEntryInput{PriceItem: PriceItem{Modality: "llm", Provider: "t1c", Model: model}, Withdrawn: true}))

	used := pricing.Quantities{pricing.InputTokens: 1_000_000}
	atFlex := s.prices.Quote(pricing.Item{Modality: "llm", Provider: "t1c", Model: model, Attributes: flex.When}, used)
	s.EqualValues(500_000, atFlex.BaseCostMicros, "the flex entry still prices")
	s.True(s.prices.Quote(pricing.Item{Modality: "llm", Provider: "t1c", Model: model}, used).Unpriced)
}

func (s *PriceListsSuite) TestAPastEffectiveTimeIsRefused() {
	tenMinutesAgo, twoMinutesAgo := time.Now().Add(-10*time.Minute), time.Now().Add(-2*time.Minute)
	past := s.change(entry("m"+s.token(), 1))
	past.EffectiveAt = &tenMinutesAgo
	recent := s.change(entry("m"+s.token(), 1))
	recent.EffectiveAt = &twoMinutesAgo

	status, _ := s.refused(past)
	s.Equal(http.StatusBadRequest, status)

	written := s.created(recent)
	s.True(written.EffectiveAt.Equal(written.CreatedAt), "two minutes ago is taken as now")
}

func (s *PriceListsSuite) TestAResubmitWritesOnce() {
	key, input := s.utils.uuid(), s.change(entry("m"+s.token(), 1))

	status, first := s.createUnder(key, input)
	s.Require().Equal(http.StatusCreated, status)
	status, again := s.createUnder(key, input)

	s.Equal(http.StatusOK, status)
	s.Equal(first.Name, again.Name)
	s.Len(s.auditOf(first.Name), 1)
}

func (s *PriceListsSuite) TestAReusedKeyWithAnotherBodyIsRefused() {
	key, model := s.utils.uuid(), "m"+s.token()
	input := s.change(entry(model, 1))
	status, _ := s.createUnder(key, input)
	s.Require().Equal(http.StatusCreated, status)
	input.Entries = []PriceEntryInput{entry(model, 1.1)}

	status, payload := s.staff.with(idempotencyHeader, key).call(http.MethodPost, "/v1/manage/price-lists/changes", input)

	s.Equal(http.StatusConflict, status)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered))
	s.Equal(codeIdempotencyKeyReused, answered.Error.Code)
}

func (s *PriceListsSuite) TestAChangeNeedsAnIdempotencyKey() {
	for _, key := range []string{"", strings.Repeat("k", 256), "key\twith a tab"} {
		status := s.staff.withRaw(idempotencyHeader, key).
			do(http.MethodPost, "/v1/manage/price-lists/changes", s.change(entry("m"+s.token(), 1)), nil)
		s.Equal(http.StatusBadRequest, status, "%q", key)
	}
}

func (s *PriceListsSuite) TestAScheduledChangeCanBeCancelled() {
	tomorrow := time.Now().Add(24 * time.Hour)
	input := s.change(entry("m"+s.token(), 1))
	input.EffectiveAt = &tomorrow
	scheduled := s.created(input)
	path := "/v1/manage/price-lists/changes/" + scheduled.Name + "/cancel"

	var cancelled PriceChange
	s.Require().Equal(http.StatusOK, s.staff.with(idempotencyHeader, s.utils.uuid()).
		do(http.MethodPost, path, CancelPriceChangeInput{Reason: "The vendor withdrew it."}, &cancelled))
	s.Require().NotNil(cancelled.CancelledAt)
	s.Equal(s.staff.header.Get(actorIDHeader), cancelled.CancelledBy)
	s.Equal("The vendor withdrew it.", cancelled.CancelReason)
	s.Len(cancelled.Entries, 1)

	other := s.staff.with(actorIDHeader, "staff:"+s.utils.uuid())
	var again PriceChange
	s.Require().Equal(http.StatusOK, other.do(http.MethodPost, path, CancelPriceChangeInput{Reason: "Again."}, &again))
	s.Equal(cancelled.CancelledBy, again.CancelledBy, "a cancel sent again changes nothing")
	s.False(slices.ContainsFunc(s.active().Upcoming, func(upcoming UpcomingPriceEntry) bool { return upcoming.Version == scheduled.Name }))
	audit := s.auditOf(scheduled.Name)
	s.Require().Len(audit, 2)
	s.Equal(store.AuditCancelled, audit[1].Action)
}

func (s *PriceListsSuite) TestAChangeInForceCannotBe() {
	written := s.created(s.change(entry("m"+s.token(), 1)))

	status, message := s.staff.failure(http.MethodPost, "/v1/manage/price-lists/changes/"+written.Name+"/cancel",
		CancelPriceChangeInput{Reason: "Too late."})

	s.Equal(http.StatusConflict, status)
	s.Contains(message, "write a new change instead")
}

func (s *PriceListsSuite) TestAnUnknownChangeIsNotFound() {
	status, payload := s.staff.call(http.MethodGet, "/v1/manage/price-lists/changes/20261023000000_nobody", nil)

	s.Equal(http.StatusNotFound, status)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered))
	s.Equal(codePriceChangeNotFound, answered.Error.Code)
}

func (s *PriceListsSuite) TestTheActiveListSaysWhatIsUnpriced() {
	list := s.active()

	s.Contains(list.Unpriced, PriceItem{Modality: "llm", Provider: "echo", Model: "echo-model"},
		"a routed model no change has priced")
	s.NotEmpty(list.Entries, "the seed")
}

func (s *PriceListsSuite) TestTheActiveListShowsTheSKUTheCostAndThePrice() {
	model := "m" + s.token()
	written := s.created(s.change(pricedAt(entry(model, 1.1), 1.375)))

	list := s.active()

	at := slices.IndexFunc(list.Entries, func(entry PriceEntry) bool { return entry.Model == model })
	s.Require().GreaterOrEqual(at, 0)
	s.Equal("llm-t1c-"+model, list.Entries[at].SKU)
	s.Equal(1.1, list.Entries[at].Rates["input_tokens"].USD)
	s.Equal(1.375, list.Entries[at].Prices["input_tokens"].USD)
	s.Equal(written.Name, list.Entries[at].Version)
}

func (s *PriceListsSuite) TestAPriceChangeCanBeReadBack() {
	model := "m" + s.token()
	written := s.created(s.change(entry(model, 1)))

	var read PriceChange
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodGet, "/v1/manage/price-lists/changes/"+written.Name, nil, &read))
	s.Equal(written.Name, read.Name)
	s.Equal(PriceChangeKind(store.PriceChangeAPI), read.Kind)
	s.Require().Len(read.Entries, 1)
	s.Equal(model, read.Entries[0].Model)
	s.Equal(1.0, read.Entries[0].Rates["input_tokens"].USD)
	s.Equal(1.0, read.Entries[0].Prices["input_tokens"].USD)
	s.Equal("llm-t1c-"+model, read.Entries[0].SKU)

	var page PriceChangePage
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodGet, "/v1/manage/price-lists/changes?kind=api&limit=5", nil, &page))
	s.Equal(written.Name, page.Items[0].Name, "newest first")
	s.Empty(page.Items[0].Entries, "a list of changes leaves the entries out")
}

func (s *PriceListsSuite) TestAChangeIsAudited() {
	written := s.created(s.change(entry("m"+s.token(), 1)))

	audit := s.auditOf(written.Name)
	s.Require().Len(audit, 1)
	s.Equal(store.AuditCreated, audit[0].Action)
	s.Equal(store.AuditSourceManage, audit[0].Source)
	s.Equal(store.AuditResourcePriceList, audit[0].ResourceType)
	s.Equal(s.staff.header.Get(actorIDHeader), audit[0].ActorID)
	s.Equal("Ada", audit[0].ActorName)
	s.Empty(audit[0].CustomerID)
}

func (s *PriceListsSuite) TestThePreviewWritesNothing() {
	before := s.changeCount()
	var preview PricePreview

	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodPost, "/v1/manage/price-lists/changes/preview", s.change(entry("m"+s.token(), 1)), &preview))

	s.Len(preview.Items, 1)
	s.Nil(preview.Items[0].Before, "nothing priced it")
	s.NotNil(preview.Items[0].After)
	s.False(preview.ConfirmationRequired)
	s.Equal(before, s.changeCount())
}

func (s *PriceListsSuite) TestAnAppCredentialDoesNotReachThePriceList() {
	s.Equal(http.StatusUnauthorized, s.serverClient.do(http.MethodGet, "/v1/manage/price-lists/active", nil, nil))
}

// auditOf is the audit rows of a change, oldest first.
func (s *PriceListsSuite) auditOf(name string) []store.AuditEntry {
	var rows []store.AuditEntry
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("resource_id = ?", name).Order("created_at").Scan(context.Background()))
	return rows
}

// withRaw is c sending value in header, even an empty one.
func (c *testClient) withRaw(header, value string) *testClient {
	named := *c
	named.header = c.header.Clone()
	named.header[http.CanonicalHeaderKey(header)] = []string{value}
	return &named
}

// TestAnEmptyListIsNotReady: a router with a price file of its own is not ready on /health
// while its database's price list is empty, and is once a seed is read. Not parallel and not on
// RouterSuite: it needs price tables nothing has written to, so a database of its own, which it
// empties first.
func TestAnEmptyListIsNotReady(t *testing.T) {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("ROUTER_POSTGRES_DSN must be set")
	}
	ctx := context.Background()
	own := testenv.Database(testDatabase(t, dsn), "pricesready")
	require.Contains(t, own, "pricesready")
	require.Contains(t, own, "_test")
	pgStore, err := store.Open(own)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, pgStore.Close()) })
	_, err = pgStore.DB().ExecContext(ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	require.NoError(t, err)
	require.NoError(t, pgStore.Migrate(ctx))
	running, cancel := context.WithCancel(ctx)
	t.Cleanup(cancel)
	fallback, err := prices.Embedded(nil, "test", nil)
	require.NoError(t, err)
	active := pricing.NewActive(running, pricing.ActiveOptions{
		Source: pgStore.PriceSource(), Fallback: fallback, Interval: suitePriceReload, RequireEntries: true,
		Logger: slog.New(slog.DiscardHandler),
	})
	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{routing.LLM: idleInspector{}},
		Store:   pgStore, Prices: active, Logger: slog.New(slog.DiscardHandler),
	})
	require.NoError(t, err)
	listener := httptest.NewServer(server.Handler())
	t.Cleanup(listener.Close)
	health := func() (int, HealthStatus) {
		response, err := http.Get(listener.URL + "/health")
		require.NoError(t, err)
		defer response.Body.Close()
		var status HealthStatus
		require.NoError(t, json.NewDecoder(response.Body).Decode(&status))
		return response.StatusCode, status
	}

	status, body := health()
	require.Equal(t, http.StatusServiceUnavailable, status)
	require.Contains(t, body.Dependencies["prices"], "empty")

	_, _, _, err = pgStore.SyncPrices(ctx, store.PriceLoad{Entries: prices.Vendor().Entries, Source: prices.VendorFile, Release: "test"})
	require.NoError(t, err)

	require.Eventually(t, func() bool {
		status, _ := health()
		return status == http.StatusOK
	}, 5*time.Second, 20*time.Millisecond, fmt.Sprintf("read again within %s", suitePriceReload))
}
