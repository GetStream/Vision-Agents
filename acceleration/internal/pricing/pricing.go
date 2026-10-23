// Package pricing turns what a unit of work used into what it costs. Every row the router
// records is priced here, so a price changes in one place and every cost agrees with it.
package pricing

import (
	"maps"
	"math"
	"slices"
)

// Meter is a unit work is measured and priced in.
type Meter string

const (
	// LLM tokens. InputTokens is the prompt neither read from nor written to the vendor's
	// cache; OutputTokens includes reasoning.
	InputTokens           Meter = "input_tokens"
	CachedInputTokens     Meter = "cached_input_tokens"
	CacheWriteInputTokens Meter = "cache_write_input_tokens"
	OutputTokens          Meter = "output_tokens"
	// Speech to speech, by modality. The input ones are fresh, as InputTokens is: a
	// modality's cached tokens are taken out of it. A split replaces its total (Quote).
	InputTextTokens        Meter = "input_text_tokens"
	InputAudioTokens       Meter = "input_audio_tokens"
	InputImageTokens       Meter = "input_image_tokens"
	CachedInputTextTokens  Meter = "cached_input_text_tokens"
	CachedInputAudioTokens Meter = "cached_input_audio_tokens"
	CachedInputImageTokens Meter = "cached_input_image_tokens"
	OutputTextTokens       Meter = "output_text_tokens"
	OutputAudioTokens      Meter = "output_audio_tokens"

	Characters      Meter = "characters"
	AudioMs         Meter = "audio_ms"
	Images          Meter = "images"
	Pixels          Meter = "pixels"
	Requests        Meter = "requests"
	Messages        Meter = "messages"
	MessageSegments Meter = "message_segments"
	CallSeconds     Meter = "call_seconds"
	Seconds         Meter = "seconds"
	SessionSeconds  Meter = "session_seconds"
	VCPUSeconds     Meter = "vcpu_seconds"
	MemoryGBSeconds Meter = "memory_gb_seconds"
	DiskGBSeconds   Meter = "disk_gb_seconds"
	BytesWritten    Meter = "bytes_written"
	BytesQueried    Meter = "bytes_queried"
	BytesReturned   Meter = "bytes_returned"
	Results         Meter = "results"
	ByteHours       Meter = "byte_hours"
	Clones          Meter = "clones"
)

// meters is every meter an entry may rate.
var meters = []Meter{
	InputTokens, CachedInputTokens, CacheWriteInputTokens, OutputTokens,
	InputTextTokens, InputAudioTokens, InputImageTokens,
	CachedInputTextTokens, CachedInputAudioTokens, CachedInputImageTokens,
	OutputTextTokens, OutputAudioTokens,
	Characters, AudioMs, Images, Pixels, Requests, Messages, MessageSegments, CallSeconds,
	Seconds, SessionSeconds, VCPUSeconds, MemoryGBSeconds, DiskGBSeconds,
	BytesWritten, BytesQueried, BytesReturned, Results, ByteHours, Clones,
}

// fallback is the meter whose rate prices a meter its entry has no rate for. A cached or
// cache-written token with no rate of its own costs what a fresh one does, never $0.
var fallback = map[Meter]Meter{
	CachedInputTokens:      InputTokens,
	CacheWriteInputTokens:  InputTokens,
	InputTextTokens:        InputTokens,
	InputAudioTokens:       InputTokens,
	InputImageTokens:       InputTokens,
	CachedInputTextTokens:  CachedInputTokens,
	CachedInputAudioTokens: CachedInputTokens,
	CachedInputImageTokens: CachedInputTokens,
	OutputTextTokens:       OutputTokens,
	OutputAudioTokens:      OutputTokens,
}

// InputMeters are the prompt: what a context tier's threshold is measured on.
var InputMeters = []Meter{InputTokens, CachedInputTokens, CacheWriteInputTokens,
	InputTextTokens, InputAudioTokens, InputImageTokens,
	CachedInputTextTokens, CachedInputAudioTokens, CachedInputImageTokens}

// OutputMeters are what was generated: what Quote.OutputCostMicros prices.
var OutputMeters = []Meter{OutputTokens, OutputTextTokens, OutputAudioTokens}

// inputSplit and outputSplit are the modality meters that replace the totals they split.
var (
	inputSplit = []Meter{InputTextTokens, InputAudioTokens, InputImageTokens,
		CachedInputTextTokens, CachedInputAudioTokens, CachedInputImageTokens}
	outputSplit = []Meter{OutputTextTokens, OutputAudioTokens}
)

// Item is what is priced: a model of a provider in a modality, and the attributes of the
// work that can choose between its entries, such as a service tier or a country.
type Item struct {
	Modality   string
	Provider   string
	Model      string
	Attributes map[string]string
}

// Quantities is how much of each meter one unit of work used.
type Quantities map[Meter]int64

// Zero reports whether nothing was used.
func (q Quantities) Zero() bool {
	for _, quantity := range q {
		if quantity != 0 {
			return false
		}
	}
	return true
}

// Used is the quantities themselves, for work counted by whoever records it rather than
// reported by a result of its own.
func (q Quantities) Used() Quantities { return q }

// Metered is a result that says what its work used, in the units it is priced by. Recording
// a call takes only results that implement it, so a new result type that cannot say what it
// used does not compile where it would be recorded.
type Metered interface {
	Used() Quantities
}

// VendorCosted is a result whose vendor said what the work cost it, in millionths of a
// dollar. It never prices the work: it is kept beside the row to check the price list by.
type VendorCosted interface {
	VendorCostMicros() int64
}

// Rate is USD for every Per units of a meter. Increment rounds the quantity up to a
// multiple of itself, and Minimum is the least billed once anything is used.
type Rate struct {
	USD       float64 `json:"usd" yaml:"usd"`
	Per       int64   `json:"per" yaml:"per"`
	Increment int64   `json:"increment,omitempty" yaml:"increment"`
	Minimum   int64   `json:"minimum,omitempty" yaml:"minimum"`
}

// micros is what quantity costs at this rate, in unrounded millionths of a dollar.
func (r Rate) micros(quantity int64) float64 {
	billed := quantity
	if r.Increment > 0 && billed > 0 && billed%r.Increment != 0 {
		billed += r.Increment - billed%r.Increment
	}
	if r.Minimum > 0 && quantity > 0 {
		billed = max(billed, r.Minimum)
	}
	return r.USD * 1e6 * float64(billed) / float64(r.Per)
}

// Tier prices a prompt longer than AboveInputTokens at Rates, for the whole request, as
// vendors that charge more for a long context do.
type Tier struct {
	AboveInputTokens int64          `json:"above_input_tokens"`
	Rates            map[Meter]Rate `json:"rates"`
}

// Entry is the price of one item. When names the item attributes it applies to; an entry
// without any is the item's default. Free and Local say why an entry with no rate is not
// billed: the vendor charges nothing, or the deployment runs the model itself.
type Entry struct {
	Modality string            `json:"modality"`
	Provider string            `json:"provider"`
	Model    string            `json:"model"`
	When     map[string]string `json:"when,omitempty"`
	Free     bool              `json:"free,omitempty"`
	Local    bool              `json:"local,omitempty"`
	Rates    map[Meter]Rate    `json:"rates,omitempty"`
	Tiers    []Tier            `json:"tiers,omitempty"`
	// MarkupBps is the operator's markup on this entry alone, in basis points (1500 is
	// 15%), replacing the list's: nil takes the list's, and 0 sells at the vendor's rate.
	// Rates stay the vendor's; a markup is kept beside them, never folded into them.
	MarkupBps *int64 `json:"markup_bps,omitempty"`
	// Version names the change to the price list that wrote the entry. Empty takes the
	// list's own version.
	Version string `json:"version,omitempty"`
}

// QuotedVersion is the version of a quote for an amount the vendor quoted outright, such as
// a number's monthly rental: no entry of the list priced it, only its markup.
const QuotedVersion = "quoted"

// Quote is what one unit of work costs, in millionths of a dollar, and the price list
// version that priced it. Billable is false for free and local work, which counts toward
// no spend; Unpriced is billable work no rate priced.
type Quote struct {
	// CostMicros is what the work is charged: its cost at the vendor's rates plus the
	// markup, rounded once.
	CostMicros int64
	// OutputCostMicros is the share of CostMicros the generated tokens cost, markup included.
	OutputCostMicros int64
	// BaseCostMicros is the cost at the vendor's rates, before any markup, so the margin on a
	// row is CostMicros - BaseCostMicros.
	BaseCostMicros int64
	// MarkupBps is the markup CostMicros carries, in basis points.
	MarkupBps int64
	Version   string
	Billable  bool
	Unpriced  bool
	// Rated is true when some meter the work used is one the entry has a rate for, a zero
	// rate included: false for priced work is work that reported none of what it is billed by.
	Rated bool
}

// Pricer prices units of work.
type Pricer interface {
	Quote(item Item, used Quantities) Quote
	// Quoted resells an amount the vendor quoted outright for the item: the amount plus the
	// item's markup, versioned QuotedVersion.
	Quoted(item Item, micros int64) Quote
	Billable(item Item) bool
	// Entry is the item's entry at the vendor's rates.
	Entry(item Item) (Entry, bool)
	// Charged is the item's entry at the rates customers are charged: Charged of Entry
	// with the list's markup.
	Charged(item Item) (Entry, bool)
}

// priced reports whether the entry has a rate above zero.
func (e Entry) priced() bool {
	if anyNonZero(e.Rates) {
		return true
	}
	return slices.ContainsFunc(e.Tiers, func(tier Tier) bool { return anyNonZero(tier.Rates) })
}

func anyNonZero(rates map[Meter]Rate) bool {
	for _, rate := range rates {
		if rate.USD != 0 {
			return true
		}
	}
	return false
}

// Billable reports whether work on this entry counts toward spend: it has a rate, or it
// has none and is neither free nor local.
func (e Entry) Billable() bool {
	return e.priced() || (!e.Free && !e.Local)
}

// Quote prices used at this entry's rates, plus its own markup when it has one. It is how an
// item whose price travels with it, such as a customer's own model, is priced without a list;
// a List matches the entry first and adds its markup to an entry with none of its own.
func (e Entry) Quote(used Quantities) Quote {
	return e.quote(used, markupOf(e, 0))
}

// markupOf is the markup an entry is sold at under a list with listMarkupBps: its own when it
// has one, else the list's.
func markupOf(entry Entry, listMarkupBps int64) int64 {
	if entry.MarkupBps != nil {
		return *entry.MarkupBps
	}
	return listMarkupBps
}

// marked is micros with a markup of markupBps added, unrounded.
func marked(micros float64, markupBps int64) float64 {
	return micros * float64(10_000+markupBps) / 10_000
}

// quote prices used at this entry's rates and adds markupBps once, to the unrounded sum, so
// the markup never multiplies a rounding.
func (e Entry) quote(used Quantities, markupBps int64) Quote {
	if !e.priced() {
		billable := !e.Free && !e.Local
		return Quote{Billable: billable, Unpriced: billable}
	}
	used = split(used)

	rates := e.Rates
	var prompt int64
	for _, meter := range InputMeters {
		prompt += used[meter]
	}
	for _, tier := range e.Tiers {
		if prompt > tier.AboveInputTokens {
			rates = tier.Rates
		}
	}

	// Summed in sorted order, so a cost never depends on map order.
	var micros, outputMicros float64
	rated := false
	for _, meter := range slices.Sorted(maps.Keys(used)) {
		rate, ok := rateOf(rates, meter)
		if !ok {
			continue
		}
		rated = rated || used[meter] != 0
		cost := rate.micros(used[meter])
		micros += cost
		if slices.Contains(OutputMeters, meter) {
			outputMicros += cost
		}
	}
	cost := roundHalfUp(marked(micros, markupBps))
	return Quote{
		CostMicros:       cost,
		OutputCostMicros: min(roundHalfUp(marked(outputMicros, markupBps)), cost),
		BaseCostMicros:   roundHalfUp(micros),
		MarkupBps:        markupBps,
		Billable:         true,
		Rated:            rated,
	}
}

// split drops the totals a modality split replaces, so no token is billed twice: a
// speech-to-speech row carries both, the totals for reporting and the split for pricing.
func split(used Quantities) Quantities {
	has := func(meters []Meter) bool {
		return slices.ContainsFunc(meters, func(meter Meter) bool { return used[meter] != 0 })
	}
	splitInput, splitOutput := has(inputSplit), has(outputSplit)
	if !splitInput && !splitOutput {
		return used
	}
	kept := maps.Clone(used)
	if splitInput {
		delete(kept, InputTokens)
		delete(kept, CachedInputTokens)
	}
	if splitOutput {
		delete(kept, OutputTokens)
	}
	return kept
}

// RateOf is the rate a meter is priced at among rates: its own, or the one its fallback
// leads to, as a cached token with no rate of its own costs a fresh one.
func RateOf(rates map[Meter]Rate, meter Meter) (Rate, bool) {
	return rateOf(rates, meter)
}

// rateOf is the meter's rate, or the one its fallback leads to.
func rateOf(rates map[Meter]Rate, meter Meter) (Rate, bool) {
	for {
		if rate, ok := rates[meter]; ok {
			return rate, true
		}
		next, ok := fallback[meter]
		if !ok {
			return Rate{}, false
		}
		meter = next
	}
}

// roundHalfUp rounds once per row, so fractions of a micro on each meter add up before
// they are rounded rather than being lost one meter at a time.
func roundHalfUp(micros float64) int64 {
	return int64(math.Floor(micros + 0.5))
}
