package usage

import (
	"bytes"
	"encoding/json"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
)

// PathAccounts prefixes the customer view of the accounts: what an organization's own
// dashboard reads through its router, on the ingest token and never the manage key.
const PathAccounts = "/v1/accounts"

// StripeActor is the actor of the credit entries the payment processor writes, whose reference
// is its own payment id, which the customer may see. Every other writer's reference is the
// operator's.
const StripeActor = "stripe"

// Where an account's low-balance threshold comes from.
const (
	ThresholdTier    = "tier"
	ThresholdAccount = "account"
)

// Account is an organization's standing as the organization itself may read it: everything a
// standing says but why it is suspended, which is the operator's, with the low-balance
// threshold in force and which settings the operator set for the organization alone.
type Account struct {
	OrganizationID   string        `json:"organization_id"`
	OrganizationTier string        `json:"organization_tier" doc:"The tier string the account was read on, which may have no row."`
	Tier             Tier          `json:"tier" doc:"The row of that tier, or the default tier's when it has none."`
	FallbackTier     bool          `json:"fallback_tier" doc:"The tier read on has no row: tier is the default tier's."`
	Limits           Limits        `json:"limits" doc:"The settings in force: the tier's with the organization's overrides laid over them."`
	Overrides        Limits        `json:"limit_overrides" doc:"The organization's own settings, each replacing the tier's."`
	OwnLimits        []string      `json:"own_limits" doc:"The settings limit_overrides sets, as dotted paths into limits such as channels.voice.concurrent or spend_hard_limit.daily_micros, sorted."`
	Status           AccountStatus `json:"status" enum:"active,suspended"`
	GrantedMicros    int64         `json:"granted_micros" doc:"Every credit entry summed."`
	SpentMicros      int64         `json:"spent_micros" doc:"Every billable event's cost summed."`
	BalanceMicros    int64         `json:"balance_micros" doc:"granted_micros less spent_micros."`
	SpendWindows     SpendWindows  `json:"spend_windows" doc:"The organization's spend in the current UTC day and calendar month, for the spend hard limit."`
	LowBalanceMicros *int64        `json:"low_balance_micros,omitempty" doc:"The balance under which the organization is warned: the account's own when it has one, else the tier's. Absent when neither sets one."`
	LowBalanceSource string        `json:"low_balance_source,omitempty" enum:"tier,account" doc:"Whose low_balance_micros is: the tier's or the account's own. Absent with no threshold."`
}

// CustomerAccount is the standing as its organization may read it.
func CustomerAccount(standing Standing) Account {
	account := Account{
		OrganizationID: standing.OrganizationID, OrganizationTier: standing.OrganizationTier,
		Tier: standing.Tier, FallbackTier: standing.FallbackTier, Limits: standing.Limits, Overrides: standing.Overrides,
		OwnLimits: SetPaths(standing.Overrides), Status: standing.Status, GrantedMicros: standing.GrantedMicros,
		SpentMicros: standing.SpentMicros, BalanceMicros: standing.BalanceMicros, SpendWindows: standing.SpendWindows,
		LowBalanceMicros: standing.Limits.LowBalanceMicros,
	}
	switch {
	case standing.Overrides.LowBalanceMicros != nil:
		account.LowBalanceSource = ThresholdAccount
	case standing.Limits.LowBalanceMicros != nil:
		account.LowBalanceSource = ThresholdTier
	}
	return account
}

// SetPaths names every setting the limits set, as a dotted path to it, sorted: a list such as a
// channel's countries is one setting.
func SetPaths(limits Limits) []string {
	paths := []string{}
	encoded, err := json.Marshal(limits.Compact())
	if err != nil {
		return paths
	}
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	var tree map[string]any
	if decoder.Decode(&tree) != nil {
		return paths
	}
	var walk func(prefix []string, value any)
	walk = func(prefix []string, value any) {
		object, ok := value.(map[string]any)
		if !ok {
			paths = append(paths, strings.Join(prefix, "."))
			return
		}
		for name, child := range object {
			walk(append(slices.Clone(prefix), name), child)
		}
	}
	for name, child := range tree {
		walk([]string{name}, child)
	}
	slices.Sort(paths)
	return paths
}

// CustomerCredit is one credit entry as its organization may read it: without the operator's
// actor and note, and with a reference only on an entry the payment processor wrote.
type CustomerCredit struct {
	ID           int64      `json:"id"`
	Kind         CreditKind `json:"kind" enum:"top_up,grant,refund,dispute,dispute_reversal,adjustment"`
	AmountMicros int64      `json:"amount_micros" doc:"Signed: what the entry added to the organization's credit, in micros."`
	Reference    string     `json:"reference,omitempty" doc:"The payment processor's id of what the entry was written for, such as a top-up's payment. Absent on every entry the operator's staff wrote."`
	CreatedAt    time.Time  `json:"created_at"`
}

// CustomerCreditOf is the entry as its organization may read it.
func CustomerCreditOf(entry CreditEntry) CustomerCredit {
	credit := CustomerCredit{ID: entry.ID, Kind: entry.Kind, AmountMicros: entry.AmountMicros, CreatedAt: entry.CreatedAt.UTC()}
	if entry.Actor == StripeActor {
		credit.Reference = entry.Reference
	}
	return credit
}

// CustomerCreditPage is one page of an organization's credit entries, newest first.
type CustomerCreditPage struct {
	Items      []CustomerCredit `json:"items"`
	HasMore    bool             `json:"has_more"`
	NextCursor string           `json:"next_cursor,omitempty"`
}

// CustomerUsageEvent is one usage event as its organization may read it: without its tier,
// its cost at the vendor's rates, its price version or the producer that sent it.
type CustomerUsageEvent struct {
	UsageID         uuid.UUID         `json:"usage_id"`
	OccurredAt      time.Time         `json:"occurred_at"`
	AppID           string            `json:"app_id"`
	APIKey          string            `json:"api_key" doc:"The public key the work was started with, or the origin of work no key started, such as hook:chat."`
	UserID          string            `json:"user_id,omitempty"`
	UsageCategories []string          `json:"usage_categories" doc:"The token's usage categories, its team among them as team:<name>."`
	AgentID         string            `json:"agent_id,omitempty"`
	CallID          string            `json:"call_id,omitempty"`
	SessionID       string            `json:"session_id,omitempty"`
	Meter           string            `json:"meter" doc:"The kind of work, such as llm or runtime."`
	Provider        string            `json:"provider,omitempty"`
	Model           string            `json:"model,omitempty"`
	Purpose         string            `json:"purpose,omitempty"`
	Quantities      map[string]int64  `json:"quantities"`
	Attributes      map[string]string `json:"attributes,omitempty"`
	SKU             string            `json:"sku,omitempty" doc:"The priced item's SKU, as getPrices lists it: the same across its price changes. Absent for work no entry priced, and for events recorded before SKUs were."`
	CostMicros      int64             `json:"cost_micros" doc:"What the organization is charged for the work, in micros."`
	Billable        bool              `json:"billable"`
	Success         bool              `json:"success"`
	ErrorCode       string            `json:"error_code,omitempty"`
	Tags            map[string]string `json:"tags,omitempty"`
}

// CustomerUsageEventOf is the event as its organization may read it.
func CustomerUsageEventOf(event Event) CustomerUsageEvent {
	categories := event.UsageCategories
	if categories == nil {
		categories = []string{}
	}
	quantities := event.Quantities
	if quantities == nil {
		quantities = map[string]int64{}
	}
	return CustomerUsageEvent{
		UsageID: event.UsageID, OccurredAt: event.OccurredAt.UTC(), AppID: event.AppID, APIKey: event.APIKey,
		UserID: event.UserID, UsageCategories: categories, AgentID: event.AgentID, CallID: event.CallID,
		SessionID: event.SessionID, Meter: event.Meter, Provider: event.Provider, Model: event.Model,
		Purpose: event.Purpose, Quantities: quantities, Attributes: event.Attributes, SKU: event.SKU, CostMicros: event.CostMicros,
		Billable: event.Billable, Success: event.Success, ErrorCode: event.ErrorCode, Tags: event.Tags,
	}
}

// CustomerUsagePage is one page of an organization's usage events, newest first.
type CustomerUsagePage struct {
	Items      []CustomerUsageEvent `json:"items"`
	HasMore    bool                 `json:"has_more"`
	NextCursor string               `json:"next_cursor,omitempty"`
}

// UsageQuery picks an organization's usage events: those of every field set, at most Limit
// after Cursor, the next_cursor of the page before.
type UsageQuery struct {
	// From is inclusive and To exclusive; either may be zero.
	From, To time.Time
	Meter    string
	AppID    string
	APIKey   string
	// UserID needs AppID: a user id is only unique within its app.
	UserID string
	// Team is the category team:<name>, so it is not given with Category.
	Team     string
	Category string
	Limit    int
	Cursor   string
}

// Values is the query as the usage service reads it.
func (q UsageQuery) Values() url.Values {
	values := url.Values{}
	if q.Limit > 0 {
		values.Set("limit", strconv.Itoa(q.Limit))
	}
	for _, instant := range []struct {
		name string
		at   time.Time
	}{{"from", q.From}, {"to", q.To}} {
		if !instant.at.IsZero() {
			values.Set(instant.name, instant.at.UTC().Format(time.RFC3339Nano))
		}
	}
	for name, value := range map[string]string{
		"meter": q.Meter, "app_id": q.AppID, "api_key": q.APIKey, "user_id": q.UserID,
		"team": q.Team, "category": q.Category, "cursor": q.Cursor,
	} {
		if value != "" {
			values.Set(name, value)
		}
	}
	return values
}

// Teams is what an organization's teams spent in their current windows.
type Teams struct {
	AsOf  time.Time   `json:"as_of"`
	Teams []TeamSpend `json:"teams" doc:"Every team whose work was recorded this calendar month (UTC), by name."`
}

// TeamSpend is one team's spend across the organization in each current window, and within
// each app whose work carried it this month.
type TeamSpend struct {
	Team    string         `json:"team" doc:"The team's name, without the team: of its category."`
	Windows []WindowSpend  `json:"windows" doc:"The team's category across the organization, in the current window of every interval, daily first; one nobody spent in at zero."`
	Apps    []TeamAppSpend `json:"apps" doc:"The apps whose work carried the team this month, by id."`
}

// TeamAppSpend is one team's spend within one app.
type TeamAppSpend struct {
	AppID   string        `json:"app_id"`
	Windows []WindowSpend `json:"windows" doc:"The team's category within the app, in the current window of every interval, daily first."`
}
