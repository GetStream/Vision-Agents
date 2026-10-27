package api

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// idempotencyHeader carries the key a price change is written once under.
const idempotencyHeader = "Idempotency-Key"

// maxIdempotencyKeyLength is the longest an idempotency key may be, in characters: the usage
// service's limit, so one key a caller makes fits both.
const maxIdempotencyKeyLength = 255

// confirmationFactor is how far a price may move before a change must be confirmed: by this
// factor or more either way, which is a misplaced decimal point rather than a new price.
const confirmationFactor = 2

var (
	errNoPriceList         = notConfigured("this deployment keeps no price list: it has no database")
	errPriceChangeNotFound = APIError{Type: ErrorTypeNotFound, Code: codePriceChangeNotFound,
		Message: "there is no price change of that name"}
	errPriceListNotSeeded = APIError{Type: ErrorTypeUnavailable, Code: codePriceListNotSeeded,
		Message: "the price list is not seeded yet: the router loads its price files into it when it starts, " +
			"and keeps trying until it has; try again shortly"}
	errPriceChangeInForce = APIError{Type: ErrorTypeConflict, Code: codePriceChangeInForce,
		Message: "only a change made through the API that is not yet in force can be cancelled: write a new change instead"}
	errIdempotencyKeyReused = APIError{Type: ErrorTypeConflict, Code: codeIdempotencyKeyReused,
		Message: "the " + idempotencyHeader + " already holds another price change: send a new key for a new change"}
	errPriceChangeExists = APIError{Type: ErrorTypeConflict, Code: codePriceChangeExists,
		Message: "a price change of that name was made this second: send it under another name, or again in a moment"}
	errPriceChangeInPast = invalidRequest(fmt.Sprintf("effective_at is more than %s ago: a change never "+
		"rewrites what was already billed. A time up to then is taken as now", store.PriceChangeGrace))
	errNothingToChange = invalidRequest("a price change sets at least one entry")
)

// priceListBusyRetry is the wait a write refused with price_list_busy is told to take: about
// as long as it waited for the lock. A choice, not a measurement.
const priceListBusyRetry = 5 * time.Second

// priceListBusy refuses a write that waited its lock timeout for another write to the price
// list, or a router's start loading it, with a 503 to send again after Retry-After.
func priceListBusy() error {
	seconds := int(priceListBusyRetry.Seconds())
	return huma.ErrorWithHeaders(APIError{Type: ErrorTypeUnavailable, Code: codePriceListBusy, Message: fmt.Sprintf(
		"another write to the price list is under way: send this again after %ds", seconds)},
		http.Header{"Retry-After": {strconv.Itoa(seconds)}})
}

// invalidPrice is a change whose entries do not make a price list, naming what is wrong.
func invalidPrice(message string) APIError {
	return APIError{Type: ErrorTypeInvalidRequest, Code: codeInvalidPrice, Message: message}
}

// refusedSKU is the 400 a change whose SKUs AssignSKUs refused answers, or nil for any other
// error.
func refusedSKU(err error) error {
	message := func() string { return strings.TrimPrefix(err.Error(), "store: ") }
	switch {
	case errors.Is(err, store.ErrSKUInUse):
		return APIError{Type: ErrorTypeInvalidRequest, Code: codeSKUInUse, Message: message()}
	case errors.Is(err, store.ErrSKUChanged):
		return APIError{Type: ErrorTypeInvalidRequest, Code: codeSKUChanged, Message: message()}
	}
	return nil
}

// checkIdempotencyKey refuses a key the usage service would refuse too: 1 to 255 characters of
// UTF-8 with no control character. An absent key is refused only when required.
func checkIdempotencyKey(key string, required bool) error {
	if key == "" && !required {
		return nil
	}
	if n := utf8.RuneCountInString(key); n == 0 || n > maxIdempotencyKeyLength {
		return invalidRequest(fmt.Sprintf("a price change is made once under its %s: 1 to %d characters",
			idempotencyHeader, maxIdempotencyKeyLength))
	}
	if !utf8.ValidString(key) || strings.ContainsFunc(key, unicode.IsControl) {
		return invalidRequest("the " + idempotencyHeader + " is not UTF-8 or holds a control character")
	}
	return nil
}

// PriceRate is US dollars for every Per units of a meter.
type PriceRate struct {
	USD       float64 `json:"usd" doc:"US dollars for every per units."`
	Per       int64   `json:"per" doc:"The units the rate is for: 1000000 for a rate per million tokens, 1 for one per image."`
	Increment int64   `json:"increment,omitempty" doc:"The quantity is rounded up to a multiple of this before it is priced, such as 60 for a call billed by the started minute."`
	Minimum   int64   `json:"minimum,omitempty" doc:"The least quantity billed once anything is used."`
}

// PriceTier prices a long prompt at rates of its own.
type PriceTier struct {
	AboveInputTokens int64                `json:"above_input_tokens" doc:"A prompt longer than this many input tokens is priced at these rates, for the whole request."`
	Rates            map[string]PriceRate `json:"rates" doc:"By meter."`
}

// PriceItem is what is priced: a model of a provider in a modality, and the attributes that
// choose between its entries.
type PriceItem struct {
	Modality string            `json:"modality" doc:"Such as llm, stt, tts, sms or phone."`
	Provider string            `json:"provider"`
	Model    string            `json:"model"`
	When     map[string]string `json:"when,omitempty" doc:"The item attributes the entry applies to, such as {\"service_tier\": \"flex\"}. Absent is the item's default entry."`
}

// PriceEntry is one item's entry in force: its SKU, the vendor's cost and its price.
type PriceEntry struct {
	PriceItem
	SKU         string               `json:"sku,omitempty" doc:"The item's SKU, the same across its price changes, which every row it prices records. Absent only on a local model no change prices."`
	Free        bool                 `json:"free,omitempty" doc:"The vendor charges nothing for it."`
	Local       bool                 `json:"local,omitempty" doc:"The deployment runs the model itself, as its routing config says. Local work with no price costs nothing."`
	Rates       map[string]PriceRate `json:"rates,omitempty" doc:"The vendor's cost, by meter, for reference: nothing is charged from it."`
	Tiers       []PriceTier          `json:"tiers,omitempty" doc:"The vendor's cost for long prompts, for reference."`
	Prices      map[string]PriceRate `json:"prices,omitempty" doc:"What customers are charged, by the meters of rates."`
	PriceTiers  []PriceTier          `json:"price_tiers,omitempty" doc:"What customers are charged for long prompts, at the thresholds of tiers."`
	Version     string               `json:"version,omitempty" doc:"The price change that wrote the entry, which every row it prices names as its price_version."`
	EffectiveAt *time.Time           `json:"effective_at,omitempty" doc:"When the entry took effect, or takes effect."`
}

// UpcomingPriceEntry is an entry a scheduled change sets.
type UpcomingPriceEntry struct {
	PriceEntry
	Withdrawn bool   `json:"withdrawn,omitempty" doc:"The item under its when stops being priced at effective_at."`
	Reason    string `json:"reason" doc:"The change's reason."`
}

// PriceList is the price list in force at a moment.
type PriceList struct {
	At       time.Time            `json:"at" doc:"The moment the list is read at."`
	Entries  []PriceEntry         `json:"entries" nullable:"false"`
	Unpriced []PriceItem          `json:"unpriced" nullable:"false" doc:"The items the routing config routes whose work is billable and that no entry prices: each records its rows at nothing until a change prices it, or marks it free."`
	Upcoming []UpcomingPriceEntry `json:"upcoming" nullable:"false" doc:"The entries of the changes not cancelled that take effect after at, soonest first."`
}

// PriceChangeKind is what wrote a price change.
type PriceChangeKind string

func (PriceChangeKind) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "PriceChangeKind",
		"What wrote the change: seed is a database's first load from the router's price files, release the "+
			"items a release's price files had that the database lacked, and api a change made through the "+
			"management API.",
		store.PriceChangeSeed, store.PriceChangeRelease, store.PriceChangeAPI)
}

// PriceChangeEntry is an entry a change wrote: its SKU, the vendor's cost and its price.
type PriceChangeEntry struct {
	PriceItem
	SKU        string               `json:"sku" doc:"The item's SKU: the one it is priced under, or the one a withdrawal stopped pricing."`
	Free       bool                 `json:"free,omitempty"`
	Rates      map[string]PriceRate `json:"rates,omitempty" doc:"The vendor's cost, by meter, for reference."`
	Tiers      []PriceTier          `json:"tiers,omitempty" doc:"The vendor's cost for long prompts, for reference."`
	Prices     map[string]PriceRate `json:"prices,omitempty" doc:"What customers are charged, by meter."`
	PriceTiers []PriceTier          `json:"price_tiers,omitempty" doc:"What customers are charged for long prompts."`
	Withdrawn  bool                 `json:"withdrawn,omitempty" doc:"The change stopped pricing the item under its when."`
}

// PriceChange is one write to the price list. None is ever deleted, and none of its entries is
// ever changed: a correction is a newer change.
type PriceChange struct {
	Name         string             `json:"name" doc:"<UTC yyyymmddhhmmss>_<the name it was given>: the price_version of every row its entries price."`
	Kind         PriceChangeKind    `json:"kind"`
	Reason       string             `json:"reason"`
	Source       string             `json:"source"`
	EffectiveAt  time.Time          `json:"effective_at"`
	CreatedBy    string             `json:"created_by" doc:"The actor that made it, or the router release that loaded it."`
	CreatedAt    time.Time          `json:"created_at"`
	CancelledAt  *time.Time         `json:"cancelled_at,omitempty"`
	CancelledBy  string             `json:"cancelled_by,omitempty"`
	CancelReason string             `json:"cancel_reason,omitempty"`
	Entries      []PriceChangeEntry `json:"entries,omitempty" doc:"The entries it wrote. Absent from a list of changes."`
}

// PriceChangePage is one page of price changes, newest first.
type PriceChangePage struct {
	Items      []PriceChange `json:"items" nullable:"false"`
	HasMore    bool          `json:"has_more"`
	NextCursor *string       "json:\"next_cursor,omitempty\" doc:\"Pass as `cursor` for the next page. Absent on the last one.\""
}

// PriceEntryInput is one item's entry as a change writes it, its vendor's cost and its price,
// or its withdrawal.
type PriceEntryInput struct {
	PriceItem
	SKU        string               `json:"sku,omitempty" maxLength:"255" pattern:"^[a-z0-9]+(-[a-z0-9]+)*$" doc:"The item's SKU. Absent keeps the one the item has, or gives a new item one made from its modality, provider, model and when. A new item may take another item's SKU, as a model its vendor renamed keeps the old name's, only when that item is withdrawn in the same change or is no longer in force: else 400 sku_in_use. An item in force keeps its SKU: another is 400 sku_changed."`
	Free       bool                 `json:"free,omitempty" doc:"The vendor charges nothing for it. A free entry has no price."`
	Rates      map[string]PriceRate `json:"rates,omitempty" doc:"The vendor's cost, by meter, kept for reference: input_tokens, cached_input_tokens, cache_write_input_tokens, output_tokens, characters, audio_ms, images, requests, message_segments, call_seconds and the rest pricing knows. Required unless the entry is free or withdrawn. An unknown meter, and a rate of more than $1000 a unit, are refused."`
	Tiers      []PriceTier          `json:"tiers,omitempty" doc:"The vendor's cost for long prompts. Each tier rates every meter the entry does."`
	Prices     map[string]PriceRate `json:"prices,omitempty" doc:"What customers are charged, in the shape of rates: the same meters, or 400 price_meters_mismatch. Required with rates; set outright, never worked out from them. Bounded as a rate is."`
	PriceTiers []PriceTier          `json:"price_tiers,omitempty" doc:"What customers are charged for long prompts: one tier for each of tiers, at its threshold and with its meters, or 400 price_tiers_mismatch."`
	Withdrawn  bool                 `json:"withdrawn,omitempty" doc:"Stop pricing the item under its when from effective_at on. Its entries under another when are untouched. A withdrawal carries no rate, price or SKU."`
	Local      bool                 `json:"local,omitempty" doc:"Refused with invalid_price: a model is local when the routing config says so."`
	Version    string               `json:"version,omitempty" doc:"Refused with invalid_price: an entry's version is the change that writes it."`
}

// PriceChangeInput is one change to the price list: the entries it writes.
type PriceChangeInput struct {
	Name        string            `json:"name" pattern:"^[a-z0-9_]{1,48}$" doc:"1 to 48 characters of a-z, 0-9 and _. The change is named with the UTC time it was made before it."`
	Reason      string            `json:"reason" minLength:"1" maxLength:"2000" doc:"What changed, from what to what. At least one sentence; trimmed, and refused when blank."`
	Source      string            `json:"source" minLength:"1" maxLength:"1000" doc:"The vendor's page and the date it was read, or the operator's own decision. Trimmed, and refused when blank."`
	EffectiveAt *time.Time        `json:"effective_at,omitempty" doc:"When the change takes effect. Absent is now; it may be up to two years ahead, but never more than 5 minutes in the past, and a time up to 5 minutes past is taken as now."`
	Entries     []PriceEntryInput `json:"entries,omitempty" maxItems:"1000" doc:"Each replaces its item's entry under its when whole: its cost, its price and every tier of both."`
	Confirm     bool              `json:"confirm,omitempty" doc:"Required when the preview answers confirmation_required."`
}

// PricedEntry is how an item is priced before or after a change.
type PricedEntry struct {
	SKU        string               `json:"sku,omitempty" doc:"The SKU it is priced under."`
	Free       bool                 `json:"free,omitempty"`
	Local      bool                 `json:"local,omitempty"`
	Rates      map[string]PriceRate `json:"rates,omitempty" doc:"The vendor's cost, for reference."`
	Tiers      []PriceTier          `json:"tiers,omitempty"`
	Prices     map[string]PriceRate `json:"prices,omitempty" doc:"What customers are charged."`
	PriceTiers []PriceTier          `json:"price_tiers,omitempty"`
	Version    string               `json:"version,omitempty" doc:"The change that wrote it. Absent for the change previewed."`
}

// PricePreviewItem is one item a change moves.
type PricePreviewItem struct {
	PriceItem
	Before            *PricedEntry       `json:"before,omitempty" doc:"How it is priced before the change. Absent when nothing prices it."`
	After             *PricedEntry       `json:"after,omitempty" doc:"How it is priced after. Absent when nothing will."`
	PriceFactors      map[string]float64 `json:"price_factors,omitempty" doc:"By meter, the factor its price moves by. Absent for a meter with no price before or after."`
	Unpriced          bool               `json:"unpriced,omitempty" doc:"It is priced before the change and billable with no price after."`
	NeedsConfirmation bool               `json:"needs_confirmation,omitempty"`
}

// PricePreview is what a change would do, written nowhere.
type PricePreview struct {
	EffectiveAt          time.Time          `json:"effective_at" doc:"When the change would take effect."`
	Items                []PricePreviewItem `json:"items" nullable:"false"`
	ConfirmationRequired bool               `json:"confirmation_required" doc:"createPriceChange answers 409 confirmation_required without confirm: true."`
	NeedsConfirmation    []PriceItem        `json:"needs_confirmation,omitempty" doc:"The items that make it so."`
}

// CancelPriceChangeInput says why a change is called off.
type CancelPriceChangeInput struct {
	Reason string `json:"reason" minLength:"1" maxLength:"2000" doc:"Why the change is called off."`
}

// A field a price body does not know is refused rather than dropped, so a misspelt rate is
// a 400 and never a price of nothing.

func (*PriceChangeInput) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.AdditionalProperties = false
	return schema
}

func (*PriceEntryInput) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.AdditionalProperties = false
	return schema
}

func (*PriceRate) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.AdditionalProperties = false
	return schema
}

func (*PriceTier) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.AdditionalProperties = false
	return schema
}

func (*CancelPriceChangeInput) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.AdditionalProperties = false
	return schema
}

type getActivePriceListRequest struct {
	At string `query:"at" format:"date-time" doc:"The moment to read the list at, RFC 3339. Omitted is now."`
}

type priceListResponse struct {
	Body PriceList
}

type listPriceChangesRequest struct {
	Kind   PriceChangeKind `query:"kind" doc:"Only changes of this kind. Omitted is every kind."`
	Limit  int             `query:"limit" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 25."`
	Cursor string          `query:"cursor" doc:"The next_cursor of the previous page, sent with the same kind."`
}

type priceChangePageResponse struct {
	Body PriceChangePage
}

type priceChangeNameRequest struct {
	Name string `path:"name" doc:"The change's name."`
}

type priceChangeResponse struct {
	Status int
	Body   PriceChange
}

type previewPriceChangeRequest struct {
	Body PriceChangeInput
}

type pricePreviewResponse struct {
	Body PricePreview
}

type createPriceChangeRequest struct {
	IdempotencyKey string `header:"Idempotency-Key" doc:"Required: 1 to 255 characters. The change is written once under it."`
	Body           PriceChangeInput
}

type cancelPriceChangeRequest struct {
	Name           string `path:"name" doc:"The change's name."`
	IdempotencyKey string `header:"Idempotency-Key" doc:"Optional: 1 to 255 characters. A change is cancelled once whatever it says; a cancel sent again changes nothing."`
	Body           CancelPriceChangeInput
}

// routedConfig is the routing config of every modality this deployment serves.
func (s *Server) routedConfig() routing.Config {
	config := routing.Config{}
	for modality, router := range s.routers {
		config[modality] = router.Config()
	}
	return config
}

// getActivePriceList answers the list in force at a moment, read from the database rather
// than from this router's memory, so it answers for any moment and for every router at once.
func (s *Server) getActivePriceList(ctx context.Context, request *getActivePriceListRequest) (*priceListResponse, error) {
	if s.store == nil {
		return nil, errNoPriceList
	}
	at := time.Now()
	if request.At != "" {
		parsed, err := time.Parse(time.RFC3339Nano, request.At)
		if err != nil {
			return nil, invalidRequest("at is not an RFC 3339 time")
		}
		at = parsed
	}
	snapshot, err := s.store.PriceListAt(ctx, at)
	if err != nil {
		return nil, err
	}
	rows := snapshot.Entries
	config := s.routedConfig()
	local := routing.LocalItems(config)
	entries := make([]pricing.Entry, 0, len(rows))
	for _, row := range rows {
		entries = append(entries, row.Entry())
	}
	list, err := pricing.NewList(pricing.ActiveVersion, pricing.WithLocal(entries, local), slog.New(slog.DiscardHandler))
	if err != nil {
		return nil, err
	}
	answer := PriceList{At: at.UTC(), Entries: []PriceEntry{}, Unpriced: []PriceItem{}, Upcoming: []UpcomingPriceEntry{}}
	for _, row := range rows {
		entry := row.Entry()
		entry.Local = isLocal(entry, local)
		answer.Entries = append(answer.Entries, renderPriceEntry(entry, row.EffectiveAt))
	}
	for _, item := range list.Unpriced(routing.Items(config)) {
		answer.Unpriced = append(answer.Unpriced, priceItemOf(item))
	}
	for _, row := range snapshot.Upcoming {
		entry := row.Entry()
		entry.Local = isLocal(entry, local)
		answer.Upcoming = append(answer.Upcoming, UpcomingPriceEntry{
			PriceEntry: renderPriceEntry(entry, row.EffectiveAt), Withdrawn: row.Withdrawn, Reason: row.Reason,
		})
	}
	return &priceListResponse{Body: answer}, nil
}

func (s *Server) listPriceChanges(ctx context.Context, request *listPriceChangesRequest) (*priceChangePageResponse, error) {
	if s.store == nil {
		return nil, errNoPriceList
	}
	after, err := decodeCursor[store.PriceChangePosition](&request.Cursor)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	found, err := s.store.PriceChanges(ctx, store.PriceChangeFilter{Kind: string(request.Kind), Limit: request.Limit, After: after})
	if err != nil {
		return nil, err
	}
	kept, more := page(found, store.PriceChangeLimit(request.Limit))
	answer := PriceChangePage{HasMore: more, Items: make([]PriceChange, 0, len(kept))}
	for _, change := range kept {
		answer.Items = append(answer.Items, renderPriceChange(change, nil))
	}
	if more {
		last := kept[len(kept)-1]
		answer.NextCursor = encodeCursor(store.PriceChangePosition{CreatedAt: last.CreatedAt, Name: last.Name})
	}
	return &priceChangePageResponse{Body: answer}, nil
}

func (s *Server) getPriceChange(ctx context.Context, request *priceChangeNameRequest) (*priceChangeResponse, error) {
	if s.store == nil {
		return nil, errNoPriceList
	}
	change, rows, err := s.store.PriceChange(ctx, request.Name)
	if errors.Is(err, store.ErrNoPriceChange) {
		return nil, errPriceChangeNotFound
	}
	if err != nil {
		return nil, err
	}
	return &priceChangeResponse{Status: http.StatusOK, Body: renderPriceChange(change, rows)}, nil
}

func (s *Server) previewPriceChange(ctx context.Context, request *previewPriceChangeRequest) (*pricePreviewResponse, error) {
	if s.store == nil {
		return nil, errNoPriceList
	}
	input, err := normalized(request.Body)
	if err != nil {
		return nil, err
	}
	writes, err := priceEntryWrites(input)
	if err != nil {
		return nil, err
	}
	before, err := s.store.PriceListBefore(ctx, input.EffectiveAt)
	if errors.Is(err, store.ErrPriceChangeInPast) {
		return nil, errPriceChangeInPast
	}
	if err != nil {
		return nil, err
	}
	writes, err = before.Assign(writes)
	if refused := refusedSKU(err); refused != nil {
		return nil, refused
	}
	if err != nil {
		return nil, err
	}
	preview, err := s.previewOf(before, writes)
	if err != nil {
		return nil, err
	}
	return &pricePreviewResponse{Body: preview}, nil
}

func (s *Server) createPriceChange(ctx context.Context, request *createPriceChangeRequest) (*priceChangeResponse, error) {
	if s.store == nil {
		return nil, errNoPriceList
	}
	if err := checkIdempotencyKey(request.IdempotencyKey, true); err != nil {
		return nil, err
	}
	input, err := normalized(request.Body)
	if err != nil {
		return nil, err
	}
	writes, err := priceEntryWrites(input)
	if err != nil {
		return nil, err
	}
	actor := managedBy(ctx)
	change, rows, replayed, err := s.store.CreatePriceChange(ctx, store.PriceChangeWrite{
		Name: input.Name, Reason: input.Reason, Source: input.Source, EffectiveAt: input.EffectiveAt,
		Entries: writes, IdempotencyKey: request.IdempotencyKey, ActorID: actor.ID, ActorName: actor.Name,
	}, func(before store.PriceListAt, assigned []store.PriceEntryWrite) error {
		preview, err := s.previewOf(before, assigned)
		if err != nil {
			return err
		}
		if preview.ConfirmationRequired && !input.Confirm {
			return confirmationRequired(preview.NeedsConfirmation)
		}
		return nil
	})
	if refused := refusedSKU(err); refused != nil {
		return nil, refused
	}
	switch {
	case errors.Is(err, store.ErrPriceListNotSeeded):
		return nil, errPriceListNotSeeded
	case errors.Is(err, store.ErrPriceChangeKeyReused):
		return nil, errIdempotencyKeyReused
	case errors.Is(err, store.ErrPriceChangeExists):
		return nil, errPriceChangeExists
	case errors.Is(err, store.ErrPriceChangeInPast):
		return nil, errPriceChangeInPast
	case errors.Is(err, store.ErrPriceListBusy):
		return nil, priceListBusy()
	case err != nil:
		return nil, err
	}
	status := http.StatusCreated
	if replayed {
		status = http.StatusOK
	} else {
		s.reloadPrices(ctx)
	}
	return &priceChangeResponse{Status: status, Body: renderPriceChange(change, rows)}, nil
}

func (s *Server) cancelPriceChange(ctx context.Context, request *cancelPriceChangeRequest) (*priceChangeResponse, error) {
	if s.store == nil {
		return nil, errNoPriceList
	}
	if err := checkIdempotencyKey(request.IdempotencyKey, false); err != nil {
		return nil, err
	}
	reason := strings.TrimSpace(request.Body.Reason)
	if reason == "" {
		return nil, invalidRequest("a cancel says why: reason is 1 to 2000 characters")
	}
	actor := managedBy(ctx)
	change, rows, cancelled, err := s.store.CancelPriceChange(ctx, request.Name, reason, actor.ID, actor.Name)
	switch {
	case errors.Is(err, store.ErrNoPriceChange):
		return nil, errPriceChangeNotFound
	case errors.Is(err, store.ErrPriceChangeInForce):
		return nil, errPriceChangeInForce
	case errors.Is(err, store.ErrPriceListBusy):
		return nil, priceListBusy()
	case err != nil:
		return nil, err
	}
	if cancelled {
		s.reloadPrices(ctx)
	}
	return &priceChangeResponse{Status: http.StatusOK, Body: renderPriceChange(change, rows)}, nil
}

// reloadPrices reads the price list again after this router wrote to it, so its next request
// is priced by the change. A read that fails is logged by the list, which keeps the one it
// had and reads again within its interval, as every other router does.
func (s *Server) reloadPrices(ctx context.Context) {
	if s.prices != nil {
		_ = s.prices.Reload(context.WithoutCancel(ctx))
	}
}

// confirmationRequired refuses a change that moves a price too far without confirm: true.
func confirmationRequired(items []PriceItem) APIError {
	names := make([]string, 0, len(items))
	for _, item := range items {
		names = append(names, prices.Name(pricing.Entry{Modality: item.Modality, Provider: item.Provider, Model: item.Model, When: item.When}))
	}
	return APIError{Type: ErrorTypeConflict, Code: codeConfirmationRequired, Message: fmt.Sprintf(
		"the change moves a price by a factor of %d or more, or leaves a routed model unpriced, "+
			"so it needs confirm: true: %s", confirmationFactor, strings.Join(names, "; "))}
}

// maxEffectiveAhead is how far ahead a change may be scheduled: past it, a date is a typo in
// its year rather than a vendor's announced price.
const maxEffectiveAhead = 2 * 365 * 24 * time.Hour

// normalized is a change's input as it is checked and stored: its reason and source trimmed.
// One with either blank, or scheduled further ahead than maxEffectiveAhead, is refused.
func normalized(input PriceChangeInput) (PriceChangeInput, error) {
	input.Reason, input.Source = strings.TrimSpace(input.Reason), strings.TrimSpace(input.Source)
	if input.Reason == "" || input.Source == "" {
		return PriceChangeInput{}, invalidRequest("a price change says why, in reason, and from where, in source: neither may be blank")
	}
	if input.EffectiveAt != nil && input.EffectiveAt.After(time.Now().Add(maxEffectiveAhead)) {
		return PriceChangeInput{}, APIError{Type: ErrorTypeInvalidRequest, Code: codeValidationFailed,
			Message: "effective_at is more than two years ahead"}
	}
	return input, nil
}

// priceEntryWrites checks what a change asks for before the database is read: at least one
// entry, and no entry setting what no column holds, writing an item twice, withdrawing with a
// rate, a price or a SKU, or a price that is not in the shape of its vendor cost. Whether its
// entries make a list, and their SKUs, are checked against the list in force.
func priceEntryWrites(input PriceChangeInput) ([]store.PriceEntryWrite, error) {
	if len(input.Entries) == 0 {
		return nil, errNothingToChange
	}
	writes := make([]store.PriceEntryWrite, 0, len(input.Entries))
	seen := map[string]bool{}
	for _, entry := range input.Entries {
		written := pricingEntryOf(entry)
		name := prices.Name(written)
		priced := len(entry.Rates) > 0 || len(entry.Tiers) > 0 || len(entry.Prices) > 0 || len(entry.PriceTiers) > 0
		switch {
		case entry.Local:
			return nil, invalidPrice(name + " sets local: a model is local when the routing config says so")
		case entry.Version != "":
			return nil, invalidPrice(name + " sets a version: an entry's version is the change that writes it")
		case seen[name]:
			return nil, invalidPrice(name + " is in the change twice")
		case entry.Withdrawn && (entry.Free || priced || entry.SKU != ""):
			return nil, invalidPrice(name + " is withdrawn and priced at once: a withdrawal carries no rate, price or SKU")
		case entry.Free && (len(entry.Prices) > 0 || len(entry.PriceTiers) > 0):
			return nil, invalidPrice(name + " is free and has a price: a free entry carries none")
		case !entry.Free && !entry.Withdrawn && len(entry.Rates) == 0:
			return nil, invalidPrice(name + " has no rates: an entry that is not free sets the vendor's cost in rates and its price in prices")
		}
		if !entry.Free && !entry.Withdrawn {
			if err := pricing.Validate(written); errors.Is(err, pricing.ErrPriceMeters) {
				return nil, APIError{Type: ErrorTypeInvalidRequest, Code: codePriceMetersMismatch, Message: strings.TrimPrefix(err.Error(), "pricing: ")}
			} else if errors.Is(err, pricing.ErrPriceTiers) {
				return nil, APIError{Type: ErrorTypeInvalidRequest, Code: codePriceTiersMismatch, Message: strings.TrimPrefix(err.Error(), "pricing: ")}
			}
		}
		seen[name] = true
		writes = append(writes, store.PriceEntryWrite{Entry: written, Withdrawn: entry.Withdrawn})
	}
	return writes, nil
}

// previewOf is what writes, with their SKUs, do to the list a change lands on: every item it
// moves, before and after, and whether it needs confirming. Entries that do not make a list
// with the rest are invalid_price, naming the item and what is wrong.
func (s *Server) previewOf(before store.PriceListAt, writes []store.PriceEntryWrite) (PricePreview, error) {
	config := s.routedConfig()
	local := routing.LocalItems(config)
	discard := slog.New(slog.DiscardHandler)

	current := make([]pricing.Entry, 0, len(before.Entries))
	for _, row := range before.Entries {
		current = append(current, row.Entry())
	}
	beforeList, err := pricing.NewList(pricing.ActiveVersion, pricing.WithLocal(current, local), discard)
	if err != nil {
		return PricePreview{}, err
	}
	next := make([]pricing.Entry, 0, len(current)+len(writes))
	replaced := map[string]bool{}
	for _, write := range writes {
		replaced[prices.Name(write.Entry)] = true
	}
	for _, entry := range current {
		if !replaced[prices.Name(entry)] {
			next = append(next, entry)
		}
	}
	for _, write := range writes {
		if !write.Withdrawn {
			next = append(next, write.Entry)
		}
	}
	afterList, err := pricing.NewList(pricing.ActiveVersion, pricing.WithLocal(next, local), discard)
	if err != nil {
		return PricePreview{}, invalidPrice(strings.TrimPrefix(err.Error(), "pricing: "))
	}

	// The items to show: every one the change writes, and every routed one that becomes
	// unpriced.
	var items []pricing.Item
	seen := map[string]bool{}
	add := func(entry pricing.Entry) {
		if name := prices.Name(entry); !seen[name] {
			seen[name] = true
			items = append(items, pricing.Item{Modality: entry.Modality, Provider: entry.Provider, Model: entry.Model, Attributes: entry.When})
		}
	}
	for _, write := range writes {
		add(write.Entry)
	}
	routed := map[string]bool{}
	for _, item := range routing.Items(config) {
		routed[itemName(item)] = true
	}
	wasUnpriced := map[string]bool{}
	for _, item := range beforeList.Unpriced(routing.Items(config)) {
		wasUnpriced[itemName(item)] = true
	}
	for _, item := range afterList.Unpriced(routing.Items(config)) {
		if !wasUnpriced[itemName(item)] {
			add(pricing.Entry{Modality: item.Modality, Provider: item.Provider, Model: item.Model})
		}
	}

	preview := PricePreview{EffectiveAt: before.At.UTC(), Items: []PricePreviewItem{}}
	for _, item := range items {
		shown := PricePreviewItem{PriceItem: priceItemOf(item)}
		was, wasOK := beforeList.Entry(item)
		now, nowOK := afterList.Entry(item)
		if wasOK {
			shown.Before = pricedEntry(was)
		}
		if nowOK {
			shown.After = pricedEntry(now)
		}
		pricedBefore := len(beforeList.Unpriced([]pricing.Item{item})) == 0
		pricedAfter := len(afterList.Unpriced([]pricing.Item{item})) == 0
		shown.Unpriced = pricedBefore && !pricedAfter
		if shown.Unpriced && routed[itemName(item)] && len(item.Attributes) == 0 {
			shown.NeedsConfirmation = true
		}
		if wasOK && nowOK {
			var moved bool
			if shown.PriceFactors, moved = priceFactors(was, now); moved {
				shown.NeedsConfirmation = true
			}
		}
		if shown.NeedsConfirmation {
			preview.ConfirmationRequired = true
			preview.NeedsConfirmation = append(preview.NeedsConfirmation, shown.PriceItem)
		}
		preview.Items = append(preview.Items, shown)
	}
	slices.SortFunc(preview.Items, func(a, b PricePreviewItem) int {
		return cmp.Compare(priceItemName(a.PriceItem), priceItemName(b.PriceItem))
	})
	slices.SortFunc(preview.NeedsConfirmation, func(a, b PriceItem) int { return cmp.Compare(priceItemName(a), priceItemName(b)) })
	return preview, nil
}

// priceFactors is the factor each meter's price moves by from was to now, the price a meter
// falls back to counting for one with none of its own, and whether any moved far. A meter with
// no price on one side has no factor, and moved far when the other side has one. A price tier
// moved far when its prices did, and when it is on one side only: a long prompt is then
// charged at other prices. The vendor's cost moves no factor.
func priceFactors(was, now pricing.Entry) (map[string]float64, bool) {
	factors := map[string]float64{}
	moved := compareRates(was.Prices, now.Prices, factors)
	tiersOf := func(entry pricing.Entry) map[int64]map[pricing.Meter]pricing.Rate {
		tiers := make(map[int64]map[pricing.Meter]pricing.Rate, len(entry.PriceTiers))
		for _, tier := range entry.PriceTiers {
			tiers[tier.AboveInputTokens] = tier.Rates
		}
		return tiers
	}
	before, after := tiersOf(was), tiersOf(now)
	for above, rates := range after {
		old, ok := before[above]
		if !ok || compareRates(old, rates, nil) {
			moved = true
		}
	}
	for above := range before {
		if _, ok := after[above]; !ok {
			moved = true
		}
	}
	if len(factors) == 0 {
		factors = nil
	}
	return factors, moved
}

func compareRates(was, now map[pricing.Meter]pricing.Rate, factors map[string]float64) bool {
	meters := map[pricing.Meter]bool{}
	for meter := range was {
		meters[meter] = true
	}
	for meter := range now {
		meters[meter] = true
	}
	moved := false
	for _, meter := range slices.Sorted(maps.Keys(meters)) {
		before, after := perUnit(was, meter), perUnit(now, meter)
		if before > 0 && after > 0 && factors != nil {
			factors[string(meter)] = after / before
		}
		// The quantity a rate rounds up to, and the least it bills, move a price as its USD does.
		old, _ := pricing.RateOf(was, meter)
		rate, _ := pricing.RateOf(now, meter)
		if movedFar(before, after) || movedFar(float64(old.Increment), float64(rate.Increment)) ||
			movedFar(float64(old.Minimum), float64(rate.Minimum)) {
			moved = true
		}
	}
	return moved
}

// perUnit is US dollars for one unit of the meter among rates, falling back as pricing does.
func perUnit(rates map[pricing.Meter]pricing.Rate, meter pricing.Meter) float64 {
	rate, ok := pricing.RateOf(rates, meter)
	if !ok || rate.Per <= 0 {
		return 0
	}
	return rate.USD / float64(rate.Per)
}

// movedFar reports whether a figure moved by confirmationFactor or more either way, or from
// nothing to something or back.
func movedFar(before, after float64) bool {
	switch {
	case before == after:
		return false
	case before == 0 || after == 0:
		return true
	}
	return after >= before*confirmationFactor || before >= after*confirmationFactor
}

func isLocal(entry pricing.Entry, local []pricing.Item) bool {
	return slices.ContainsFunc(local, func(item pricing.Item) bool {
		return item.Modality == entry.Modality && item.Provider == entry.Provider && item.Model == entry.Model
	})
}

func itemName(item pricing.Item) string {
	return prices.Name(pricing.Entry{Modality: item.Modality, Provider: item.Provider, Model: item.Model, When: item.Attributes})
}

func priceItemName(item PriceItem) string {
	return prices.Name(pricing.Entry{Modality: item.Modality, Provider: item.Provider, Model: item.Model, When: item.When})
}

func priceItemOf(item pricing.Item) PriceItem {
	return PriceItem{Modality: item.Modality, Provider: item.Provider, Model: item.Model, When: item.Attributes}
}

func priceItemOfEntry(entry pricing.Entry) PriceItem {
	return PriceItem{Modality: entry.Modality, Provider: entry.Provider, Model: entry.Model, When: entry.When}
}

func renderPriceEntry(entry pricing.Entry, effectiveAt time.Time) PriceEntry {
	rendered := PriceEntry{
		PriceItem:  priceItemOfEntry(entry),
		SKU:        entry.SKU,
		Free:       entry.Free,
		Local:      entry.Local,
		Rates:      priceRates(entry.Rates),
		Tiers:      priceTiers(entry.Tiers),
		Prices:     priceRates(entry.Prices),
		PriceTiers: priceTiers(entry.PriceTiers),
		Version:    entry.Version,
	}
	if !effectiveAt.IsZero() {
		at := effectiveAt.UTC()
		rendered.EffectiveAt = &at
	}
	return rendered
}

func pricedEntry(entry pricing.Entry) *PricedEntry {
	return &PricedEntry{
		SKU:        entry.SKU,
		Free:       entry.Free,
		Local:      entry.Local,
		Rates:      priceRates(entry.Rates),
		Tiers:      priceTiers(entry.Tiers),
		Prices:     priceRates(entry.Prices),
		PriceTiers: priceTiers(entry.PriceTiers),
		Version:    entry.Version,
	}
}

func renderPriceChange(change store.PriceChange, rows []store.PriceEntry) PriceChange {
	rendered := PriceChange{
		Name:         change.Name,
		Kind:         PriceChangeKind(change.Kind),
		Reason:       change.Reason,
		Source:       change.Source,
		EffectiveAt:  change.EffectiveAt.UTC(),
		CreatedBy:    change.CreatedBy,
		CreatedAt:    change.CreatedAt.UTC(),
		CancelledBy:  change.CancelledBy,
		CancelReason: change.CancelReason,
	}
	if change.CancelledAt != nil {
		at := change.CancelledAt.UTC()
		rendered.CancelledAt = &at
	}
	for _, row := range rows {
		entry := row.Entry()
		rendered.Entries = append(rendered.Entries, PriceChangeEntry{
			PriceItem:  priceItemOfEntry(entry),
			SKU:        entry.SKU,
			Free:       entry.Free,
			Rates:      priceRates(entry.Rates),
			Tiers:      priceTiers(entry.Tiers),
			Prices:     priceRates(entry.Prices),
			PriceTiers: priceTiers(entry.PriceTiers),
			Withdrawn:  row.Withdrawn,
		})
	}
	return rendered
}

func pricingEntryOf(entry PriceEntryInput) pricing.Entry {
	written := pricing.Entry{
		Modality:   entry.Modality,
		Provider:   entry.Provider,
		Model:      entry.Model,
		SKU:        entry.SKU,
		Free:       entry.Free,
		Rates:      pricingRates(entry.Rates),
		Tiers:      pricingTiers(entry.Tiers),
		Prices:     pricingRates(entry.Prices),
		PriceTiers: pricingTiers(entry.PriceTiers),
	}
	if len(entry.When) > 0 {
		written.When = entry.When
	}
	return written
}

func pricingTiers(tiers []PriceTier) []pricing.Tier {
	var converted []pricing.Tier
	for _, tier := range tiers {
		converted = append(converted, pricing.Tier{AboveInputTokens: tier.AboveInputTokens, Rates: pricingRates(tier.Rates)})
	}
	return converted
}

func priceRates(rates map[pricing.Meter]pricing.Rate) map[string]PriceRate {
	if len(rates) == 0 {
		return nil
	}
	rendered := make(map[string]PriceRate, len(rates))
	for meter, rate := range rates {
		rendered[string(meter)] = PriceRate{USD: rate.USD, Per: rate.Per, Increment: rate.Increment, Minimum: rate.Minimum}
	}
	return rendered
}

func pricingRates(rates map[string]PriceRate) map[pricing.Meter]pricing.Rate {
	if len(rates) == 0 {
		return nil
	}
	converted := make(map[pricing.Meter]pricing.Rate, len(rates))
	for meter, rate := range rates {
		converted[pricing.Meter(meter)] = pricing.Rate{USD: rate.USD, Per: rate.Per, Increment: rate.Increment, Minimum: rate.Minimum}
	}
	return converted
}

func priceTiers(tiers []pricing.Tier) []PriceTier {
	if len(tiers) == 0 {
		return nil
	}
	rendered := make([]PriceTier, 0, len(tiers))
	for _, tier := range tiers {
		rendered = append(rendered, PriceTier{AboveInputTokens: tier.AboveInputTokens, Rates: priceRates(tier.Rates)})
	}
	return rendered
}

// priceListNote ends every price list operation's description.
const priceListNote = "\n\nStream staff only: it needs the manage key, and names its actor in `X-Stream-Actor-Id`. " +
	"On a deployment with no database it answers 400 `not_configured`."

// registerPriceLists declares the price list's management operations: reading the list in
// force and its changes, and changing it, which takes the manage key.
func (s *Server) registerPriceLists(api huma.API) {
	reads := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound}
	writes := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound, http.StatusConflict, http.StatusServiceUnavailable}
	huma.Register(api, huma.Operation{
		OperationID: "getActivePriceList",
		Method:      http.MethodGet,
		Path:        "/v1/manage/price-lists/active",
		Summary:     "Read the price list in force",
		Description: "Every entry in force at `at` (default now): its SKU, which every row it prices records, the " +
			"vendor's cost (`rates`, `tiers`) kept for reference, the price customers are charged (`prices`, " +
			"`price_tiers`), and the change that wrote it, which is the `price_version` of every row it prices. " +
			"`unpriced` names the routed items whose work is billable and that no entry prices, and `upcoming` " +
			"the entries of the changes scheduled after `at`." + priceListNote,
		Tags:      []string{staffTag},
		Security:  manageSecurity,
		Responses: map[string]*huma.Response{"200": {Description: "The price list in force"}},
		Errors:    reads,
	}, s.getActivePriceList)
	huma.Register(api, huma.Operation{
		OperationID: "listPriceChanges",
		Method:      http.MethodGet,
		Path:        "/v1/manage/price-lists/changes",
		Summary:     "List the price changes",
		Description: "Every write to the price list, newest first: the first load, each release's new items, and " +
			"every change made through this API, cancelled ones included." + priceListNote,
		Tags:      []string{staffTag},
		Security:  manageSecurity,
		Responses: map[string]*huma.Response{"200": {Description: "A page of price changes"}},
		Errors:    reads,
	}, s.listPriceChanges)
	huma.Register(api, huma.Operation{
		OperationID: "getPriceChange",
		Method:      http.MethodGet,
		Path:        "/v1/manage/price-lists/changes/{name}",
		Summary:     "Read a price change",
		Description: "The change with the entries it wrote. A name nobody made is a 404 `price_change_not_found`." + priceListNote,
		Tags:        []string{staffTag},
		Security:    manageSecurity,
		Responses:   map[string]*huma.Response{"200": {Description: "The price change"}},
		Errors:      reads,
	}, s.getPriceChange)
	huma.Register(api, huma.Operation{
		OperationID: "previewPriceChange",
		Method:      http.MethodPost,
		Path:        "/v1/manage/price-lists/changes/preview",
		Summary:     "Preview a price change",
		Description: "What `createPriceChange` would do with the same body, writing nothing: for every item it " +
			"moves, its SKU, vendor cost and price before and after, the factor each price moves by, and " +
			"whether it becomes unpriced. `confirmation_required` is true, naming the items, when a price, its " +
			"`increment` or `minimum`, moves by a factor of 2 or more either way (to or from 0 included), a " +
			"long-prompt price tier is added or dropped, or a routed model that is not local would become " +
			"unpriced; the vendor's cost moving asks for none. A body whose entries do not make a price list " +
			"is a 400 `invalid_price`, one whose prices are not in the shape of their cost a 400 " +
			"`price_meters_mismatch` or `price_tiers_mismatch`, and one whose SKUs are refused a 400 " +
			"`sku_in_use` or `sku_changed`, as `createPriceChange` says; a preview never answers 409." + priceListNote,
		Tags:      []string{staffTag},
		Security:  manageSecurity,
		Responses: map[string]*huma.Response{"200": {Description: "What the change would do"}},
		Errors:    reads,
	}, s.previewPriceChange)
	huma.Register(api, huma.Operation{
		OperationID:   "createPriceChange",
		Method:        http.MethodPost,
		Path:          "/v1/manage/price-lists/changes",
		Summary:       "Change the price list",
		DefaultStatus: http.StatusCreated,
		Description: "Writes a change, its entries and an audit row in one transaction. Each entry replaces its " +
			"item's entry under its `when` whole: the vendor's cost in `rates` and `tiers`, for reference, and " +
			"the price customers are charged in `prices` and `price_tiers`, both set outright: nothing is " +
			"worked out from the cost, and no markup is added. Every router prices with it from " +
			"`effective_at`: this one at once, the others within 30 seconds, or at `effective_at` when that is " +
			"later. Nothing ever changes an entry once written: a correction is a newer change, and a " +
			"scheduled one can be cancelled.\n\n" +
			"Every item has one SKU, kept across its price changes, which each row it prices records. An entry " +
			"without `sku` keeps the item's, or gives a new item one made from its modality, provider, model " +
			"and `when`, lowercased with every run of other characters a hyphen and numbered `-2`, `-3` past " +
			"one another item has had. A new item may take another item's SKU, as a model its vendor renamed " +
			"keeps the old name's, when that item is withdrawn by the same change or is no longer in force; " +
			"else, or when a change scheduled later gives it to another item, the change is a 400 " +
			"`sku_in_use`. An item in force given another SKU is a 400 `sku_changed`.\n\n" +
			"An entry that is neither free nor withdrawn needs `rates` and `prices` with the same meters, else " +
			"a 400 `price_meters_mismatch`, and a price tier for each tier, at its threshold and with its " +
			"meters, else a 400 `price_tiers_mismatch`. An entry setting `local` or `version`, or entries that " +
			"do not make a price list, are a 400 `invalid_price` naming the item. A change the preview says needs confirming is a 409 " +
			"`confirmation_required` without `confirm: true`. Before the router's first start has loaded its " +
			"price files the list is not seeded, and every change is a 503 `price_list_not_seeded`. A change " +
			"that waits more than 5 seconds for another write to the price list is a 503 `price_list_busy`, to " +
			"send again after `Retry-After`.\n\n" +
			"It is made once per `" + idempotencyHeader + "` (required, 1 to 255 characters): the same key and " +
			"body again answer 200 with the change already written and write nothing; the same key with another " +
			"body is a 409 `idempotency_key_reused`. A retry sends the same body: the same entries in the same " +
			"order, since entries listed in another order are another body." + priceListNote,
		Tags:     []string{staffTag},
		Security: manageSecurity,
		Responses: map[string]*huma.Response{
			"201": {Description: "The change written"},
			"200": {
				Description: "The change already written under the Idempotency-Key; nothing changed",
				Content:     map[string]*huma.MediaType{"application/json": {Schema: &huma.Schema{Ref: "#/components/schemas/PriceChange"}}},
			},
		},
		Errors: writes,
	}, s.createPriceChange)
	huma.Register(api, huma.Operation{
		OperationID: "cancelPriceChange",
		Method:      http.MethodPost,
		Path:        "/v1/manage/price-lists/changes/{name}/cancel",
		Summary:     "Cancel a scheduled price change",
		Description: "Calls off a change made through this API that is not yet in force, with an audit row, and " +
			"every router stops scheduling it as `createPriceChange` says. A change in force, or one a router's " +
			"start loaded, is a 409 `price_change_in_force`: write a new change instead. A change already " +
			"cancelled is answered as it stands, with who cancelled it and why, and nothing is written, so a " +
			"cancel sent again changes nothing. `" + idempotencyHeader + "` is optional and stores nothing. A " +
			"cancel that waits more than 5 seconds for another write is a 503 `price_list_busy`." + priceListNote,
		Tags:      []string{staffTag},
		Security:  manageSecurity,
		Responses: map[string]*huma.Response{"200": {Description: "The change, cancelled"}},
		Errors:    writes,
	}, s.cancelPriceChange)
}
