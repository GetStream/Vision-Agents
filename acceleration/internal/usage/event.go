// Package usage is the contract between the producers of usage (the router and its jobs)
// and the usage service: the events they ship, the windows spend is kept in, the subjects it
// is kept for, and the client that talks to the service. It imports nothing but the standard
// library and uuid, so a producer takes on nothing of the service's.
package usage

import (
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/google/uuid"
)

// DefaultTier is the tier of work whose organization's tier nobody named.
const DefaultTier = "default"

// Bounds of one call to the usage service.
const (
	// MaxEvents is the most events one batch carries.
	MaxEvents = 500
	// MaxBodyBytes is the largest body the service reads.
	MaxBodyBytes = 2 << 20
	// MaxSubjects is the most subjects one windows read names.
	MaxSubjects = 500
	// MaxOrganizations is the most organizations one standings read names.
	MaxOrganizations = 500
	// MaxCategories is the most usage categories one event carries, a team included.
	MaxCategories = 10
	// maxCategoryLength is the longest a usage category may be.
	maxCategoryLength = 64
	// MaxIDLength is the most bytes an organization, app, key or user id may hold. They are
	// indexed and keys of the windows, and Postgres fails the whole insert, and so the batch on
	// every retry, for an index row past 2,704 bytes.
	MaxIDLength = 255
	// MaxCostMicros is the most one event may cost, either cost: ten million dollars. It keeps
	// a window's sum far from what a BIGINT holds.
	MaxCostMicros = 10_000_000_000_000
	// maxAhead is how far in the future an event may say it happened, for clocks that
	// disagree.
	maxAhead = time.Hour
)

// Event is one unit of work somebody may be billed for. Its usage id is chosen before the
// work is first written anywhere, so sending it again is a no-op rather than a second charge.
type Event struct {
	UsageID    uuid.UUID `json:"usage_id"`
	OccurredAt time.Time `json:"occurred_at"`
	// Who the work was for. Organization, tier, app and key are never empty: work no key
	// started names its origin instead (hook:chat, phone:inbound, job:<name>). The user is
	// empty when there is none or the caller was anonymous, and is only unique within the
	// app. Categories are the token's, at most 10, the user's team among them as
	// team:<name>.
	OrganizationID   string   `json:"organization_id"`
	OrganizationTier string   `json:"organization_tier"`
	AppID            string   `json:"app_id"`
	APIKey           string   `json:"api_key"`
	UserID           string   `json:"user_id,omitempty"`
	UsageCategories  []string `json:"usage_categories,omitempty"`
	AgentID          string   `json:"agent_id,omitempty"`
	CallID           string   `json:"call_id,omitempty"`
	SessionID        string   `json:"session_id,omitempty"`
	// Meter is the kind of work: a router modality (llm, stt, tts, sts, search, image,
	// detection, decision_model, memory, knowledge, phone) or another producer's (runtime,
	// sandbox). It is a string so a new kind of work needs no deploy of the service, and it
	// is not a pricing meter: those are the keys of Quantities.
	Meter      string            `json:"meter"`
	Provider   string            `json:"provider,omitempty"`
	Model      string            `json:"model,omitempty"`
	Purpose    string            `json:"purpose,omitempty"`
	Quantities map[string]int64  `json:"quantities"`           // input_tokens, audio_ms, seconds, bytes, ...
	Attributes map[string]string `json:"attributes,omitempty"` // country, direction, cpu, memory_gb, ...
	// CostMicros is what the customer is charged, priced by the producer: the service never
	// prices.
	CostMicros int64 `json:"cost_micros"`
	// BaseCostMicros is the same work at the vendor's rates. Never shown to a customer.
	BaseCostMicros int64             `json:"base_cost_micros,omitempty"`
	PriceVersion   string            `json:"price_version,omitempty"`
	Billable       bool              `json:"billable"`
	Success        bool              `json:"success"`
	ErrorCode      string            `json:"error_code,omitempty"`
	Tags           map[string]string `json:"tags,omitempty"`
	// Source names the producer, for finding where an event came from: "router/<shard>/<pod>".
	Source string `json:"source"`
}

// Batch is what a producer ships in one call.
type Batch struct {
	Events []Event `json:"events" nullable:"false"`
}

// Ack is the service's answer to a batch. An event accepted or a duplicate is recorded, and
// the producer may forget it; one rejected is not, and is a producer bug to fix and ship
// again.
type Ack struct {
	Accepted   int         `json:"accepted"`
	Duplicates int         `json:"duplicates"`
	Rejected   []Rejection `json:"rejected"`
	// Mirrored is whether Redis took the totals of every window the batch's recorded events
	// count for, duplicates included, before the answer. False when the service mirrors to no
	// Redis, or the write failed: the service retries it in the background for a minute, and
	// sending the batch again or a read through /v1/windows repairs it. The events are
	// recorded either way.
	Mirrored bool `json:"mirrored"`
}

// Rejection is one event the service refused, and why.
type Rejection struct {
	// Index is the event's position in the batch, or in the events given to Client.Ship. It
	// names the event when its usage id cannot, being missing, malformed or shared.
	Index   int       `json:"index"`
	UsageID uuid.UUID `json:"usage_id"`
	// Reason is one of the Reason constants.
	Reason string `json:"reason"`
	// Message says what exactly was wrong, for the person fixing the producer.
	Message string `json:"message"`
}

// Why an event is rejected. Each is a bounded label of the service's rejected counter.
const (
	ReasonUsageID      = "missing_usage_id"
	ReasonOccurredAt   = "invalid_occurred_at"
	ReasonOrganization = "missing_organization"
	ReasonTier         = "missing_tier"
	ReasonApp          = "missing_app"
	ReasonAPIKey       = "missing_api_key"
	ReasonMeter        = "missing_meter"
	ReasonCost         = "negative_cost"
	ReasonCostTooLarge = "cost_too_large"
	ReasonQuantity     = "negative_quantity"
	ReasonCategories   = "invalid_categories"
	// ReasonSeparator is an id holding |, { or }, which delimit its Redis key.
	ReasonSeparator = "id_contains_separator"
	ReasonIDLength  = "id_too_long"
	ReasonText      = "invalid_text"
	// ReasonMalformed is an event that is not an Event in JSON: a field of the wrong type.
	ReasonMalformed = "malformed"
	// ReasonTooLarge is Client.Ship's own, never the service's: an event bigger than any
	// batch body may be, which it does not send.
	ReasonTooLarge = "too_large"
)

// Check is why the service would reject the event at now, or nil when it would record it.
func (e Event) Check(now time.Time) *Rejection {
	reject := func(reason, format string, args ...any) *Rejection {
		return &Rejection{UsageID: e.UsageID, Reason: reason, Message: fmt.Sprintf(format, args...)}
	}
	switch {
	case e.UsageID == uuid.Nil:
		return reject(ReasonUsageID, "usage_id is required")
	case e.OccurredAt.IsZero():
		return reject(ReasonOccurredAt, "occurred_at is required")
	case e.OccurredAt.After(now.Add(maxAhead)):
		return reject(ReasonOccurredAt, "occurred_at %s is more than %s ahead of the service's clock", e.OccurredAt.UTC().Format(time.RFC3339), maxAhead)
	case e.OrganizationID == "":
		return reject(ReasonOrganization, "organization_id is required")
	case e.OrganizationTier == "":
		return reject(ReasonTier, "organization_tier is required")
	case e.AppID == "":
		return reject(ReasonApp, "app_id is required")
	case e.APIKey == "":
		return reject(ReasonAPIKey, "api_key is required: the key the work was started with, or its origin")
	case e.Meter == "":
		return reject(ReasonMeter, "meter is required")
	case e.CostMicros < 0 || e.BaseCostMicros < 0:
		return reject(ReasonCost, "cost_micros %d and base_cost_micros %d cannot be negative", e.CostMicros, e.BaseCostMicros)
	case e.CostMicros > MaxCostMicros || e.BaseCostMicros > MaxCostMicros:
		return reject(ReasonCostTooLarge, "cost_micros %d and base_cost_micros %d are at most %d each", e.CostMicros, e.BaseCostMicros, int64(MaxCostMicros))
	}
	for name, quantity := range e.Quantities {
		if quantity < 0 {
			return reject(ReasonQuantity, "quantity %s is %d", name, quantity)
		}
	}
	if len(e.UsageCategories) > MaxCategories {
		return reject(ReasonCategories, "usage_categories has %d entries, at most %d", len(e.UsageCategories), MaxCategories)
	}
	for _, category := range e.UsageCategories {
		if !validCategory(category) {
			return reject(ReasonCategories, "usage category %q is not 1 to %d characters of A-Z, a-z, 0-9, _ . : -", category, maxCategoryLength)
		}
	}
	for _, id := range [][2]string{{"organization_id", e.OrganizationID}, {"app_id", e.AppID}, {"api_key", e.APIKey}, {"user_id", e.UserID}} {
		switch IDProblem(id[1]) {
		case ReasonIDLength:
			return reject(ReasonIDLength, "%s is %d bytes, at most %d", id[0], len(id[1]), MaxIDLength)
		case ReasonSeparator:
			found, _ := keyCharacter(id[1])
			return reject(ReasonSeparator, "%s contains %s", id[0], found)
		}
	}
	// Postgres refuses a NUL in text and in JSON, which would fail the whole batch rather
	// than this event.
	if name, found := e.nul(); found {
		return reject(ReasonText, "%s contains a NUL character", name)
	}
	return nil
}

// IDProblem is why an organization, app, key or user id cannot be one an event names, as the
// Reason Check rejects the whole event with (ReasonIDLength, ReasonSeparator or ReasonText),
// or empty when it can. A producer applies it before it records an id it was given.
func IDProblem(id string) string {
	switch {
	case len(id) > MaxIDLength:
		return ReasonIDLength
	case strings.ContainsAny(id, separator+"{}"):
		return ReasonSeparator
	case strings.ContainsRune(id, 0):
		return ReasonText
	}
	return ""
}

// nul finds the first field holding a NUL character.
func (e Event) nul() (string, bool) {
	fields := map[string]string{
		"organization_id": e.OrganizationID, "organization_tier": e.OrganizationTier, "app_id": e.AppID,
		"api_key": e.APIKey, "user_id": e.UserID, "agent_id": e.AgentID, "call_id": e.CallID,
		"session_id": e.SessionID, "meter": e.Meter, "provider": e.Provider, "model": e.Model,
		"purpose": e.Purpose, "price_version": e.PriceVersion, "error_code": e.ErrorCode, "source": e.Source,
	}
	for name, value := range fields {
		if strings.ContainsRune(value, 0) {
			return name, true
		}
	}
	for name := range e.Quantities {
		if strings.ContainsRune(name, 0) {
			return "quantities", true
		}
	}
	for name, labels := range map[string]map[string]string{"attributes": e.Attributes, "tags": e.Tags} {
		for key, value := range labels {
			if strings.ContainsRune(key, 0) || strings.ContainsRune(value, 0) {
				return name, true
			}
		}
	}
	return "", false
}

// validCategory reports whether a usage category is 1 to 64 characters of
// [A-Za-z0-9_.:-].
func validCategory(category string) bool {
	if category == "" || len(category) > maxCategoryLength {
		return false
	}
	for _, r := range category {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '.', r == ':', r == '-':
		default:
			return false
		}
	}
	return true
}

// normalCategories is categories sorted, each once.
func normalCategories(categories []string) []string {
	if len(categories) == 0 {
		return nil
	}
	sorted := slices.Clone(categories)
	slices.Sort(sorted)
	return slices.Compact(sorted)
}

// Normal is the event as the service stores it: its categories sorted, each once.
func (e Event) Normal() Event {
	e.UsageCategories = normalCategories(e.UsageCategories)
	return e
}
