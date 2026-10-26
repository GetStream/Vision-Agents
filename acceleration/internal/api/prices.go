package api

import (
	"cmp"
	"context"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// pricesTag puts the price list customers read in the SDKs.
const pricesTag = "Prices"

// Price is what one item costs a customer: the rates every row of its work is debited at.
type Price struct {
	PriceItem
	Free        bool                 `json:"free" doc:"The work costs nothing."`
	Local       bool                 `json:"local" doc:"The deployment runs the model itself; with no rates it costs nothing."`
	Rates       map[string]PriceRate `json:"rates,omitempty" doc:"What a customer is charged, by meter."`
	Tiers       []PriceTier          `json:"tiers,omitempty" doc:"What a customer is charged for long prompts."`
	Change      string               `json:"change,omitempty" doc:"The price change that set it: on an entry in force, the price_version of every row it prices. Absent for a model the deployment runs itself that no change prices."`
	EffectiveAt *time.Time           `json:"effective_at,omitempty" doc:"When it took effect, or takes effect. Absent when no change set it."`
	Withdrawn   bool                 `json:"withdrawn,omitempty" doc:"In upcoming: the item stops being priced at effective_at."`
	Description string               `json:"description,omitempty" doc:"In upcoming: why the change is made."`
}

func (*Price) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What one item costs: a model of a provider in a modality, under the attributes " +
		"in when, at the rates every row of its work is debited at, which X-Stream-Cost-Micros reports."
	return schema
}

// Prices is the price list in force, and what scheduled changes will set later, at the rates
// customers are charged.
type Prices struct {
	At       time.Time `json:"at" doc:"The moment the list is read at."`
	Entries  []Price   `json:"entries" nullable:"false" doc:"Every item priced now, by modality, provider and model."`
	Upcoming []Price   `json:"upcoming" nullable:"false" doc:"What the changes scheduled after at will set, soonest first."`
}

type getPricesRequest struct {
	Modality string `query:"modality" doc:"Only this modality's items, such as llm, stt, tts, sms or phone."`
}

type pricesResponse struct{ Body Prices }

func (s *Server) registerPrices(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "getPrices", Method: http.MethodGet, Path: "/v1/prices",
		Summary: "Read the price list",
		Description: "Every item this router prices, at the rates a customer is charged, which are what every " +
			"answer's X-Stream-Cost-Micros is worked out from: a rate is USD for every per units of its meter, " +
			"and a long prompt is priced whole at its tier's rates. Then what the changes scheduled later will " +
			"set, soonest first, with their reason, an item they withdraw included. The list is the one in " +
			"memory, so reading it costs no query; a change made on another router shows within 30 seconds. " +
			"Prices are the same for everyone, so any caller may read them.",
		Tags:       []string{pricesTag},
		Extensions: map[string]any{clientAccessibleExtension: true},
		Errors:     []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.getPrices)
}

func (s *Server) getPrices(_ context.Context, request *getPricesRequest) (*pricesResponse, error) {
	var list *pricing.List
	var upcoming []pricing.UpcomingEntry
	var markups []pricing.UpcomingMarkup
	switch {
	case s.prices != nil:
		list, upcoming, markups = s.prices.List(), s.prices.Upcoming(), s.prices.UpcomingMarkups()
	default:
		own, ok := s.pricer.(*pricing.List)
		if !ok {
			return nil, errNoPriceList
		}
		list = own
	}
	local := routing.LocalItems(s.routedConfig())
	wanted := func(entry pricing.Entry) bool { return request.Modality == "" || entry.Modality == request.Modality }
	markup := list.MarkupBps()
	answer := Prices{At: time.Now().UTC(), Entries: []Price{}, Upcoming: []Price{}}
	inForce := list.Entries()
	for _, entry := range inForce {
		if wanted(entry) {
			answer.Entries = append(answer.Entries, priceOf(entry, markup, entry.EffectiveAt))
		}
	}
	for _, price := range upcomingPrices(inForce, markup, upcoming, markups, local) {
		if request.Modality == "" || price.Modality == request.Modality {
			answer.Upcoming = append(answer.Upcoming, price)
		}
	}
	return &pricesResponse{Body: answer}, nil
}

// priceOf is an entry at the rates a customer is charged under a list with that markup.
func priceOf(entry pricing.Entry, listMarkupBps int64, effectiveAt time.Time) Price {
	charged := pricing.Charged(entry, listMarkupBps)
	price := Price{
		PriceItem: priceItemOfEntry(entry), Free: entry.Free, Local: entry.Local,
		Rates: priceRates(charged.Rates), Tiers: priceTiers(charged.Tiers), Change: entry.Version,
	}
	if !effectiveAt.IsZero() {
		at := effectiveAt.UTC()
		price.EffectiveAt = &at
	}
	return price
}

// upcomingPrices is what the scheduled changes will set, soonest first, each at the markup in
// force once it takes effect: every entry a change writes or withdraws, and for a change of
// the markup, every entry in force then whose charged rates it moves. Changes taking effect at
// one moment apply in the order of their names, as the list in force reads them.
func upcomingPrices(inForce []pricing.Entry, markup int64, upcoming []pricing.UpcomingEntry, markups []pricing.UpcomingMarkup, local []pricing.Item) []Price {
	current := make(map[string]pricing.Entry, len(inForce))
	for _, entry := range inForce {
		current[prices.Name(entry)] = entry
	}
	moments := map[int64]time.Time{}
	for _, entry := range upcoming {
		moments[entry.EffectiveAt.UnixMicro()] = entry.EffectiveAt
	}
	for _, change := range markups {
		moments[change.EffectiveAt.UnixMicro()] = change.EffectiveAt
	}
	order := slices.Sorted(func(yield func(int64) bool) {
		for moment := range moments {
			if !yield(moment) {
				return
			}
		}
	})
	var answer []Price
	for _, moment := range order {
		at := moments[moment]
		var markedBy *pricing.UpcomingMarkup
		for i := range markups {
			if markups[i].EffectiveAt.UnixMicro() == moment && (markedBy == nil || markups[i].Change > markedBy.Change) {
				markedBy = &markups[i]
			}
		}
		next := markup
		if markedBy != nil {
			next = markedBy.MarkupBps
		}
		var set []Price
		// written is where each item a change writes at this moment is in set: of two changes
		// writing one item, the later name's entry is the one in force.
		written := map[string]int{}
		for _, entry := range upcoming {
			if entry.EffectiveAt.UnixMicro() != moment {
				continue
			}
			name := prices.Name(entry.Entry)
			entry.Local = isLocal(entry.Entry, local)
			price := priceOf(entry.Entry, next, at)
			price.Withdrawn, price.Description = entry.Withdrawn, entry.Reason
			if entry.Withdrawn {
				price.Rates, price.Tiers = nil, nil
				delete(current, name)
			} else {
				current[name] = entry.Entry
			}
			if i, ok := written[name]; ok {
				set[i] = price
				continue
			}
			written[name] = len(set)
			set = append(set, price)
		}
		if next != markup && markedBy != nil {
			for name, entry := range current {
				if _, ok := written[name]; ok || entry.MarkupBps != nil || (len(entry.Rates) == 0 && len(entry.Tiers) == 0) {
					continue
				}
				price := priceOf(entry, next, at)
				price.Change, price.Description = markedBy.Change, markedBy.Reason
				set = append(set, price)
			}
		}
		slices.SortStableFunc(set, func(a, b Price) int {
			return cmp.Or(strings.Compare(a.Modality, b.Modality), strings.Compare(a.Provider, b.Provider),
				strings.Compare(a.Model, b.Model), strings.Compare(priceItemName(a.PriceItem), priceItemName(b.PriceItem)))
		})
		answer = append(answer, set...)
		markup = next
	}
	return answer
}
