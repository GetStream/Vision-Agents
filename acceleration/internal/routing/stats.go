package routing

import (
	"context"
	"fmt"
	"log/slog"
	"maps"
	"strconv"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// drainTimeout is how long a recorder's own writer may take to write what is queued when
// the recorder closes.
const drainTimeout = 15 * time.Second

// unnamedTierEvery is how often a service tier the price list does not name is logged for
// one model, so a busy one is not logged on every row. usageLogEvery is the same for an item
// whose rows say its usage is missing, estimated or does not add up.
const (
	unnamedTierEvery = 10 * time.Minute
	usageLogEvery    = 10 * time.Minute
)

// Owner is who a unit of work belongs to and how the customer wants it labelled. It
// travels with a session so every row the session produces is attributed the same way.
type Owner struct {
	// CustomerID owns the request. Every statistic is keyed by it.
	CustomerID string
	// Attribution is who beside the customer the work is billed to and asked for by, the
	// session it ran in included. All of it is recorded on the row.
	Attribution
	// Caller is the end user the work was done for, when it was asked for by a device
	// rather than by the customer's own backend. It is what daily limits are counted
	// against. The user a row records is Attribution's, which leaves out a name nobody
	// checked.
	Caller Caller
	// AgentID is the agent the work was done for. Empty outside a conversation.
	AgentID string
	// CallID is the call the work happened in. Empty outside a conversation.
	CallID string
	// Tags are the customer's own cost labels, carried onto every row so spend can be
	// broken down by whatever the customer cares about.
	Tags Tags
	// Spent adds up what the owner's rows cost for the request or socket that asked for the
	// work, so its answer can say so (SpentFrom). Nil adds nothing up.
	Spent *Spent
}

// Stat is one recorded unit of work. What a unit is depends on the modality: a completed
// turn for speech-to-text, a completed synthesis for text-to-speech, a completed
// completion for an LLM, plus rows for sessions that never got off the ground.
//
// The provider, model, modality and cost are filled in by the Recorder, so a caller
// cannot record a row that disagrees with the routing decision it came from, or with the
// price.
type Stat struct {
	Owner
	// UsageID names the unit of work, set by a caller that may record it more than once so
	// it is written once. Zero gives the row an id of its own.
	UsageID     uuid.UUID
	StartedAt   time.Time
	OperationID string
	Purpose     string
	TurnID      string
	DurationMs  float64
	Usage
	// InputParts is what an LLM's prompt was made of. Zero for every other modality.
	InputParts store.InputParts
	// Meters are quantities Usage has no field for, priced like the rest and stored with
	// the row.
	Meters pricing.Quantities
	// Attributes are what can choose the price, such as the service tier the vendor served
	// at, or explain the row. They are stored with it.
	Attributes map[string]string
	// LatencyMs is how long the customer waited for the work to be useful.
	LatencyMs float64
	// CostMicros overrides the priced amount for work that is not billed by the units in
	// Usage, such as a phone number's monthly charge, where the vendor quotes the price
	// outright rather than a rate to multiply. Zero leaves the pricing to the rates.
	CostMicros int64
	// Free records the work given away: priced as usual, written at no cost and not billed,
	// with what it would have cost in attributes.free_cost_micros.
	Free bool
	// VendorTotalTokens is the vendor's own count of every token the work used, where it
	// sent one. It is not stored: a row whose tokens do not add up to it is counted, since
	// the difference is a part of the bill the adapter missed.
	VendorTotalTokens int64
	Success           bool
	ErrorCode         string
	ErrorMessage      string
	// Fatal marks a failure that ended the session for a reason the provider gives every
	// caller, such as an account it refuses. It makes the provider unavailable for one
	// health window, however much it served before.
	Fatal bool
	// Unmeasured marks a row that bills work without timing a request of its provider's: a
	// voice session's seconds, the model it handed its reasoning to, what was streamed after
	// the last turn. It is billed and counted in the customer's spend, and kept out of the
	// provider's health, whose requests and latencies it would otherwise flatter.
	Unmeasured bool
}

// Use adds what the work used to the stat: the meters with a column of their own into Usage,
// which is what the stats endpoints read, and the rest into Meters. Usage's input is the whole
// prompt, so cached and cache-written tokens are added to it as well.
func (s *Stat) Use(used pricing.Quantities) {
	for meter, quantity := range used {
		if quantity == 0 {
			continue
		}
		switch meter {
		case pricing.InputTokens:
			s.InputTokens += quantity
		case pricing.CachedInputTokens:
			s.InputTokens += quantity
			s.CachedInputTokens += quantity
		case pricing.CacheWriteInputTokens:
			s.InputTokens += quantity
			s.addMeter(meter, quantity)
		case pricing.OutputTokens:
			s.OutputTokens += quantity
		case pricing.AudioMs:
			s.AudioMs += quantity
		case pricing.Characters:
			s.Characters += quantity
		case pricing.Images:
			s.Images += quantity
		case pricing.Pixels:
			s.Pixels += quantity
		case pricing.Requests:
			s.Requests += quantity
		default:
			s.addMeter(meter, quantity)
		}
	}
}

func (s *Stat) addMeter(meter pricing.Meter, quantity int64) {
	if s.Meters == nil {
		s.Meters = pricing.Quantities{}
	}
	s.Meters[meter] += quantity
}

// Attribute sets one attribute on the stat.
func (s *Stat) Attribute(name, value string) {
	if s.Attributes == nil {
		s.Attributes = map[string]string{}
	}
	s.Attributes[name] = value
}

// AttributeUsageEstimated is "true" on a row whose quantities the router counted because the
// vendor never said what the work used.
const AttributeUsageEstimated = "usage_estimated"

// Estimated marks the stat's quantities as the router's own count.
func (s *Stat) Estimated() { s.Attribute(AttributeUsageEstimated, "true") }

// AttributeMeasuredBy names who counted a row's quantities where no vendor event did, and
// MeasuredByRouter is the router: the audio it streamed to a vendor that bills that audio, or
// what it streamed after the last event that reported anything.
const (
	AttributeMeasuredBy = "measured_by"
	MeasuredByRouter    = "router"
)

// Work started to cut latency and possibly thrown away is billed like any other, and tagged so
// it can be found: AttributeSpeculation is the id every speculative reply to one turn of the
// caller's shares, AttributeHedge the id of a request asked of two providers at once, and
// AttributeKept "true" on the attempt whose response was used and "false" on one thrown away.
const (
	AttributeSpeculation = "speculation"
	AttributeHedge       = "hedge"
	AttributeKept        = "kept"
)

// MeasuredByRouter marks the stat's quantities as the router's own measure, which is the bill
// rather than an estimate of it: a zero is the count, not a vendor that said nothing.
func (s *Stat) MeasuredByRouter() { s.Attribute(AttributeMeasuredBy, MeasuredByRouter) }

// VendorCost keeps what the vendor said the work cost, when it said. It never prices the row:
// the price list is checked against it.
func (s *Stat) VendorCost(work any) {
	if costed, ok := work.(pricing.VendorCosted); ok {
		if micros := costed.VendorCostMicros(); micros > 0 {
			s.Attribute(store.VendorCostAttribute, strconv.FormatInt(micros, 10))
		}
	}
}

// AttributeProviderCostMicros is what the work cost the operator under its provider's own
// formula, in micros to three decimal places (1.280), so a day of lookups costing about a
// micro each sums close to the provider's invoice.
const AttributeProviderCostMicros = "provider_cost_micros"

// ProviderCost keeps what the work cost the operator under its provider's formula, when the
// result worked it out. It never prices the row.
func (s *Stat) ProviderCost(work any) {
	if costed, ok := work.(pricing.ProviderCosted); ok {
		if nanos := costed.ProviderCostNanos(); nanos > 0 {
			s.Attribute(AttributeProviderCostMicros, fmt.Sprintf("%d.%03d", nanos/1000, nanos%1000))
		}
	}
}

// ErrorCancelled is the ErrorCode of work its caller gave up on, whether before the provider
// answered or by closing the stream it had opened: a reply superseded by newer words, the
// slower of two hedged requests, or one cut off by the caller speaking. It is not successful,
// but the provider did nothing wrong, so it is kept out of its health. What it generated
// before it was cut off is still billed.
const ErrorCancelled = store.ErrorCancelled

// ErrorTimeout is the ErrorCode of work that ran past a deadline, its caller's or the router's
// own limit on a provider, before it finished. It failed, and what the provider generated
// until then is billed.
const ErrorTimeout = "timeout"

// Recorder prices stats and hands them to a Writer, which writes them to Postgres and Redis
// off the request path. A conversation must never wait on a database, so recording only
// queues.
type Recorder struct {
	modality Modality
	pricer   pricing.Pricer
	logger   *slog.Logger
	// gate is where the tags a customer's policies lay over every row come from. Nil lays
	// none.
	gate Gate

	writer *Writer
	// own is a writer this recorder made, and so closes.
	own bool
	// tiersWarned is when each service tier the price list does not name was last logged,
	// by provider, model and tier.
	tiersWarned throttle
	// missingLogged, estimatedLogged and mismatchLogged are when each item was last logged
	// for billable work with no usage, with an estimate, and with tokens that do not add up.
	missingLogged, estimatedLogged, mismatchLogged throttle
}

// NewRecorder starts a writer of the recorder's own, closed with it. The pricer prices
// every row; nil prices nothing, and logs every row as billable work with no price.
func NewRecorder(modality Modality, pgStore *store.Store, liveClient *live.Client, pricer pricing.Pricer, logger *slog.Logger) *Recorder {
	return newRecorder(modality, pgStore, liveClient, nil, nil, pricer, logger)
}

// NewRecorderOn records through a writer other recorders share, pricing as NewRecorder
// does. Closing the recorder leaves the writer open.
func NewRecorderOn(modality Modality, writer *Writer, pricer pricing.Pricer, logger *slog.Logger) *Recorder {
	return newRecorder(modality, nil, nil, nil, writer, pricer, logger)
}

// newRecorder records through writer, or through one of its own over pgStore and liveClient
// when writer is nil.
func newRecorder(modality Modality, pgStore *store.Store, liveClient *live.Client, gate Gate, writer *Writer, pricer pricing.Pricer, logger *slog.Logger) *Recorder {
	if pricer == nil {
		// An empty list cannot be refused.
		pricer, _ = pricing.NewList("none", nil, 0, logger)
	}
	r := &Recorder{modality: modality, pricer: pricer, logger: logger, gate: gate, writer: writer}
	if writer == nil {
		options := WriterOptions{Store: pgStore, Live: liveClient, Logger: logger}
		// A gate that holds work against its grants hears when the rows count toward them.
		if counter, ok := gate.(interface{ Counted([]Acked) }); ok {
			options.Counted = counter.Counted
		}
		r.writer = NewWriter(options)
		r.own = true
	}
	return r
}

// For is a recorder of another modality writing through this one's writer, gate and prices,
// for work one session does in a modality of its own, such as the text model a voice session
// hands its reasoning to. Closing it leaves the writer open.
func (r *Recorder) For(modality Modality) *Recorder {
	return &Recorder{modality: modality, pricer: r.pricer, logger: r.logger, gate: r.gate, writer: r.writer}
}

// Quote is what that usage of the provider costs, priced as Record prices a row.
func (r *Recorder) Quote(config ProviderConfig, usage Usage) pricing.Quote {
	return r.QuoteStat(config, Stat{Usage: usage})
}

// QuoteStat is what the stat's row costs, its meters and attributes included, priced as
// Record prices it.
func (r *Recorder) QuoteStat(config ProviderConfig, entry Stat) pricing.Quote {
	return quote(r.pricer, r.modality, config, entry.Attributes, row(r.modality, config, entry).Quantities())
}

// Record queues a stat, pricing what its row holds. It never waits: a stat the writer has
// no room for is lost, loudly.
func (r *Recorder) Record(config ProviderConfig, entry Stat) {
	if !r.writer.records() {
		return
	}

	r.warnUnnamedTier(config, entry.Attributes)
	request, priced := r.priced(config, entry)
	if request.UsageID == uuid.Nil {
		request.UsageID = uuid.Must(uuid.NewV7())
	}

	r.watch(config, entry, request, priced)
	entry.Spent.add(request.CostMicros, entry.Attributes[AttributeUsageEstimated] == "true")
	// Before the row is queued, so an ack that comes back fast finds the hold it settles.
	request.LimitGrants = r.spent(entry.Owner, request)
	r.writer.enqueue(queued{request: request, gate: r.gate, fatal: entry.Fatal, unmeasured: entry.Unmeasured})
}

// spent tells the gate what billable work cost the moment it is recorded, so the customer's
// next admission counts it before its row is written, and answers the grants it is drawn from.
// Work that cost nothing lets go of what was reserved for it.
func (r *Recorder) spent(owner Owner, request store.Request) []int64 {
	gate := r.moneyGate()
	if !request.Billable || request.CostMicros <= 0 {
		if reserver, ok := gate.(Reserver); ok {
			reserver.Release(request.UsageID)
		}
		return nil
	}
	if spender, ok := gate.(Spender); ok {
		return spender.Settle(owner, request.UsageID, request.CostMicros)
	}
	return nil
}

// moneyGate is the gate told what work costs: the recorder's, else its writer's.
func (r *Recorder) moneyGate() Gate {
	if r.gate != nil {
		return r.gate
	}
	return r.writer.gate
}

// Reserve sets aside what the estimate costs on the provider, for the owner's unit of work
// under the usage id its row will carry, before the work starts: for at most ttl, unless the
// row is recorded first (which settles it at its cost) or Release lets go of it. It answers at
// once, from the gate's memory. A refusal is the money error to answer, and no vendor is to be
// asked. Work the provider does not bill, an estimate that costs nothing, a recorder whose gate
// holds nothing, and one whose writer records nothing (so no row would settle it) reserve
// nothing.
func (r *Recorder) Reserve(config ProviderConfig, owner Owner, usageID uuid.UUID, estimate pricing.Quantities, ttl time.Duration) error {
	if r == nil || usageID == uuid.Nil || !r.writer.records() || !r.Billable(config) {
		return nil
	}
	reserver, ok := r.moneyGate().(Reserver)
	if !ok {
		return nil
	}
	var entry Stat
	entry.Use(estimate)
	micros := r.QuoteStat(config, entry).CostMicros
	if micros <= 0 {
		return nil
	}
	return reserver.Reserve(owner, usageID, micros, ttl)
}

// Release lets go of a reservation whose work never reached a vendor.
func (r *Recorder) Release(usageID uuid.UUID) {
	if r == nil || usageID == uuid.Nil {
		return
	}
	if reserver, ok := r.moneyGate().(Reserver); ok {
		reserver.Release(usageID)
	}
}

// Billable reports whether work on the provider counts toward spend, as its rows are priced: a
// customer's own model never does, and one the price list has no entry for does.
func (r *Recorder) Billable(config ProviderConfig) bool {
	if config.Provider == CustomProvider {
		return false
	}
	return r.pricer.Billable(ItemOf(r.modality, config, nil))
}

// lose prices the stat as Record would and loses its row at once, logged in full and counted
// as invalid, for work that was done and cannot be written as it is.
func (r *Recorder) lose(config ProviderConfig, entry Stat, cause error) {
	request, _ := r.priced(config, entry)
	r.writer.lose([]store.Request{request}, lostInvalid, cause)
}

// priced is the stat's row priced, and the quote of its quantities. It builds each row Record
// writes or lose loses once, so a user id left off is counted once per row.
func (r *Recorder) priced(config ProviderConfig, entry Stat) (store.Request, pricing.Quote) {
	request := row(r.modality, config, entry)
	request.UserID = userOnRow(entry.CustomerID, entry.UserID)
	priced := quote(r.pricer, r.modality, config, entry.Attributes, request.Quantities())
	request.CostMicros, request.OutputCostMicros, request.Billable = priced.CostMicros, priced.OutputCostMicros, priced.Billable
	request.BaseCostMicros, request.PriceVersion = priced.BaseCostMicros, priced.Version
	if entry.CostMicros != 0 {
		// Quoted outright by the vendor, such as a number's monthly charge, which no rate
		// prices: it is resold with the markup, and what the vendor charged is the base.
		// Generated tokens keep their share of it.
		resold := r.pricer.Quoted(ItemOf(r.modality, config, entry.Attributes), entry.CostMicros)
		request.CostMicros, request.BaseCostMicros, request.PriceVersion = resold.CostMicros, resold.BaseCostMicros, resold.Version
		request.OutputCostMicros = min(priced.OutputCostMicros, resold.CostMicros)
		request.Billable = priced.Billable || resold.Billable
	}
	if entry.Free {
		// Given away: nothing is charged, so every sum of what customers pay leaves it out
		// without asking. What it would have cost is kept beside it, for analysis; what the
		// vendor charged (BaseCostMicros) is never a charge and stays.
		if request.CostMicros != 0 {
			request.Attributes = maps.Clone(request.Attributes)
			if request.Attributes == nil {
				request.Attributes = map[string]string{}
			}
			request.Attributes[AttributeFreeCostMicros] = strconv.FormatInt(request.CostMicros, 10)
		}
		request.CostMicros, request.OutputCostMicros, request.Billable = 0, 0, false
	}
	return request, priced
}

// AttributeFreeCostMicros is what a row given away (Stat.Free) would have cost, in micros: its
// cost is written as zero, so no total of what customers pay counts it.
const AttributeFreeCostMicros = "free_cost_micros"

// warnUnnamedTier logs a service tier the vendor served at that the price list has no entry
// for, which is priced at the default rates: how a vendor renaming a tier shows up. A
// customer's own model is priced from its config, so its config is what must name the tier.
func (r *Recorder) warnUnnamedTier(config ProviderConfig, attributes map[string]string) {
	tier := attributes["service_tier"]
	if tier == "" || tier == "default" {
		return
	}
	if config.Provider == CustomProvider {
		if _, named := config.Price.ServiceTiers[tier]; named {
			return
		}
	} else if entry, ok := r.pricer.Entry(ItemOf(r.modality, config, attributes)); ok && entry.When["service_tier"] == tier {
		return
	}
	if !r.tiersWarned.due([3]string{config.Provider, config.Model, tier}, unnamedTierEvery) {
		return
	}
	r.logger.Warn("the vendor served a tier the price list does not name, priced at the default rates",
		"modality", r.modality, "provider", config.Provider, "model", config.Model, "service_tier", tier)
}

// row is the request a stat records, before it is priced: what it is priced by is what it
// stores.
func row(modality Modality, config ProviderConfig, entry Stat) store.Request {
	request := store.Request{
		UsageID:           entry.UsageID,
		Modality:          string(modality),
		CustomerID:        entry.CustomerID,
		OrganizationID:    entry.OrganizationID,
		OrganizationTier:  entry.OrganizationTier,
		APIKey:            entry.APIKey,
		UserID:            RecordedUser(entry.UserID),
		UsageCategories:   entry.UsageCategories,
		AgentID:           entry.AgentID,
		SessionID:         entry.SessionID,
		CallID:            entry.CallID,
		Tags:              entry.Tags,
		Provider:          config.Provider,
		Model:             config.Model,
		StartedAt:         entry.StartedAt,
		OperationID:       entry.OperationID,
		Purpose:           entry.Purpose,
		TurnID:            entry.TurnID,
		AudioMs:           entry.AudioMs,
		Characters:        entry.Characters,
		InputTokens:       entry.InputTokens,
		CachedInputTokens: entry.CachedInputTokens,
		OutputTokens:      entry.OutputTokens,
		InputParts:        entry.InputParts,
		Images:            entry.Images,
		Meters:            entry.meters(),
		Attributes:        entry.Attributes,
		Success:           entry.Success,
		ErrorCode:         entry.ErrorCode,
		ErrorMessage:      store.SafeLogText(entry.ErrorMessage),
	}
	if entry.CallerKind != "" {
		// A copy: the stat's map may be shared with whatever recorded it.
		request.Attributes = maps.Clone(entry.Attributes)
		if request.Attributes == nil {
			request.Attributes = map[string]string{}
		}
		request.Attributes[CallerAttribute] = entry.CallerKind
	}
	if entry.LatencyMs > 0 {
		latency := entry.LatencyMs
		request.LatencyMs = &latency
	}
	if entry.DurationMs > 0 {
		duration := entry.DurationMs
		request.DurationMs = &duration
	}
	return request
}

// meters is what the row stores beyond its typed columns: the stat's own meters, and the
// usage that has no column.
func (s Stat) meters() pricing.Quantities {
	if len(s.Meters) == 0 && s.Pixels == 0 && s.Requests == 0 {
		return nil
	}
	meters := maps.Clone(s.Meters)
	if meters == nil {
		meters = pricing.Quantities{}
	}
	if s.Pixels != 0 {
		meters[pricing.Pixels] += s.Pixels
	}
	if s.Requests != 0 {
		meters[pricing.Requests] += s.Requests
	}
	return meters
}

// Close writes what is queued and stops the writer when the recorder made it. A writer other
// recorders share is left open.
func (r *Recorder) Close() {
	if !r.own {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), drainTimeout)
	defer cancel()
	r.writer.Close(ctx)
}

// liveUsage is what a request adds to the live counters. A cancelled one is marked so, and the
// counters keep it out of provider health while still counting what it generated.
func liveUsage(request store.Request) live.Usage {
	var latencyMs float64
	if request.LatencyMs != nil {
		latencyMs = *request.LatencyMs
	}
	return live.Usage{
		Modality:          request.Modality,
		CustomerID:        request.CustomerID,
		Provider:          request.Provider,
		Model:             request.Model,
		LatencyMs:         latencyMs,
		AudioMs:           request.AudioMs,
		Characters:        request.Characters,
		InputTokens:       request.InputTokens,
		CachedInputTokens: request.CachedInputTokens,
		OutputTokens:      request.OutputTokens,
		CostMicros:        request.CostMicros,
		Success:           request.Success,
		Cancelled:         !measuresProvider(request),
	}
}

// measuresProvider reports whether a request says anything about its provider's health.
// One its caller cancelled does not: it neither failed nor was served, and how long it ran
// is how long it was left to, not how long the provider took. It is still billed for what
// it generated before it was cut off, so it is kept in the customer's spend and the request
// log.
func measuresProvider(request store.Request) bool {
	return request.ErrorCode != ErrorCancelled
}
