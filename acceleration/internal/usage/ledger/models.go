package ledger

import (
	"time"

	"github.com/google/uuid"
	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// UsageEvent is one unit of work as a producer shipped it. The usage id is the primary key,
// so an event sent twice is recorded once.
type UsageEvent struct {
	bun.BaseModel `bun:"table:usage_events,alias:ue"`

	UsageID    uuid.UUID `bun:"usage_id,pk,type:uuid"`
	OccurredAt time.Time `bun:"occurred_at,notnull"`
	ReceivedAt time.Time `bun:"received_at,nullzero,notnull,default:current_timestamp"`
	// Who the work was for. UserID and UsageCategories may be empty; the rest never are.
	OrganizationID   string   `bun:"organization_id,notnull"`
	OrganizationTier string   `bun:"organization_tier,notnull"`
	AppID            string   `bun:"app_id,notnull"`
	APIKey           string   `bun:"api_key,notnull"`
	UserID           string   `bun:"user_id,notnull"`
	UsageCategories  []string `bun:"usage_categories,array,nullzero"`
	AgentID          string   `bun:"agent_id,notnull"`
	CallID           string   `bun:"call_id,notnull"`
	SessionID        string   `bun:"session_id,notnull"`
	// Meter is the kind of work, not a pricing meter: the keys of Quantities are those.
	Meter      string            `bun:"meter,notnull"`
	Provider   string            `bun:"provider,notnull"`
	Model      string            `bun:"model,notnull"`
	Purpose    string            `bun:"purpose,notnull"`
	Quantities map[string]int64  `bun:"quantities,type:jsonb,nullzero"`
	Attributes map[string]string `bun:"attributes,type:jsonb,nullzero"`
	// CostMicros is priced by the producer, never by the service. BaseCostMicros is the same
	// work at the vendor's rates.
	CostMicros     int64             `bun:"cost_micros,notnull"`
	BaseCostMicros int64             `bun:"base_cost_micros,notnull"`
	PriceVersion   string            `bun:"price_version,notnull"`
	SKU            string            `bun:"sku,notnull"`
	Billable       bool              `bun:"billable,notnull"`
	Success        bool              `bun:"success,notnull"`
	ErrorCode      string            `bun:"error_code,notnull"`
	Tags           map[string]string `bun:"tags,type:jsonb,nullzero"`
	// Source is the producer: "router/<shard>/<pod>".
	Source string `bun:"source,notnull"`
}

// eventRow is the row an event is stored as.
func eventRow(e usage.Event) UsageEvent {
	return UsageEvent{
		UsageID: e.UsageID, OccurredAt: e.OccurredAt.UTC(),
		OrganizationID: e.OrganizationID, OrganizationTier: e.OrganizationTier, AppID: e.AppID, APIKey: e.APIKey,
		UserID: e.UserID, UsageCategories: e.UsageCategories, AgentID: e.AgentID, CallID: e.CallID, SessionID: e.SessionID,
		Meter: e.Meter, Provider: e.Provider, Model: e.Model, Purpose: e.Purpose,
		Quantities: e.Quantities, Attributes: e.Attributes,
		CostMicros: e.CostMicros, BaseCostMicros: e.BaseCostMicros, PriceVersion: e.PriceVersion, SKU: e.SKU,
		Billable: e.Billable, Success: e.Success, ErrorCode: e.ErrorCode, Tags: e.Tags, Source: e.Source,
	}
}

// Event is the row as the contract has it.
func (r UsageEvent) Event() usage.Event {
	return usage.Event{
		UsageID: r.UsageID, OccurredAt: r.OccurredAt,
		OrganizationID: r.OrganizationID, OrganizationTier: r.OrganizationTier, AppID: r.AppID, APIKey: r.APIKey,
		UserID: r.UserID, UsageCategories: r.UsageCategories, AgentID: r.AgentID, CallID: r.CallID, SessionID: r.SessionID,
		Meter: r.Meter, Provider: r.Provider, Model: r.Model, Purpose: r.Purpose,
		Quantities: r.Quantities, Attributes: r.Attributes,
		CostMicros: r.CostMicros, BaseCostMicros: r.BaseCostMicros, PriceVersion: r.PriceVersion, SKU: r.SKU,
		Billable: r.Billable, Success: r.Success, ErrorCode: r.ErrorCode, Tags: r.Tags, Source: r.Source,
	}
}

// UsageWindow is what one subject spent in one window of one interval: the running total its
// limits are checked against, upserted in the transaction that records the events, so a
// check reads one row (or its Redis mirror) and never sums events. Every event counts for
// every subject it names in every interval, whether or not a limit exists yet, so a limit
// set in the middle of a window starts from the true spend.
type UsageWindow struct {
	bun.BaseModel `bun:"table:usage_windows,alias:uw"`

	OrganizationID string        `bun:"organization_id,pk"`
	Subject        usage.Subject `bun:"subject,pk"`
	// The subject's other ids, empty where it has none: an app has AppID, a key AppID and
	// APIKey, a user AppID and UserID, a category Category, an app's category AppID and
	// Category.
	AppID       string         `bun:"app_id,pk"`
	APIKey      string         `bun:"api_key,pk"`
	UserID      string         `bun:"user_id,pk"`
	Category    string         `bun:"category,pk"`
	Interval    usage.Interval `bun:"interval,pk"`
	WindowStart time.Time      `bun:"window_start,pk"`
	WindowEnd   time.Time      `bun:"window_end,notnull"`
	// SpentMicros sums billable cost; Events counts every event, billable or not.
	SpentMicros int64     `bun:"spent_micros,notnull"`
	Events      int64     `bun:"events,notnull"`
	UpdatedAt   time.Time `bun:"updated_at,nullzero,notnull,default:current_timestamp"`
}

// Ref is the subject the window is for.
func (w UsageWindow) Ref() usage.SubjectRef {
	return usage.SubjectRef{
		OrganizationID: w.OrganizationID, Subject: w.Subject,
		AppID: w.AppID, APIKey: w.APIKey, UserID: w.UserID, Category: w.Category,
	}
}

// Spend is the window as the contract has it.
func (w UsageWindow) Spend() usage.WindowSpend {
	return usage.WindowSpend{
		SubjectRef: w.Ref(), Interval: w.Interval, WindowStart: w.WindowStart.UTC(), WindowEnd: w.WindowEnd.UTC(),
		SpentMicros: w.SpentMicros, Events: w.Events,
	}
}

// windowOf is the empty window of interval holding at, for ref.
func windowOf(ref usage.SubjectRef, interval usage.Interval, at time.Time) UsageWindow {
	start, end := usage.Window(interval, at)
	return UsageWindow{
		OrganizationID: ref.OrganizationID, Subject: ref.Subject,
		AppID: ref.AppID, APIKey: ref.APIKey, UserID: ref.UserID, Category: ref.Category,
		Interval: interval, WindowStart: start, WindowEnd: end,
	}
}

// windowKey is a window's primary key, in the order the table's is.
type windowKey struct {
	organization, subject, app, apiKey, user, category, interval string
	start                                                        int64
}

func (w UsageWindow) key() windowKey {
	return windowKey{
		organization: w.OrganizationID, subject: string(w.Subject), app: w.AppID, apiKey: w.APIKey,
		user: w.UserID, category: w.Category, interval: string(w.Interval), start: w.WindowStart.UnixMicro(),
	}
}

// compareKeys orders windows by their primary key. An upsert takes its row locks in the
// order of its rows, so two batches sorted by this lock in one order and cannot deadlock.
func compareKeys(a, b windowKey) int {
	for _, pair := range [][2]string{
		{a.organization, b.organization}, {a.subject, b.subject}, {a.app, b.app}, {a.apiKey, b.apiKey},
		{a.user, b.user}, {a.category, b.category}, {a.interval, b.interval},
	} {
		if pair[0] != pair[1] {
			if pair[0] < pair[1] {
				return -1
			}
			return 1
		}
	}
	switch {
	case a.start < b.start:
		return -1
	case a.start > b.start:
		return 1
	}
	return 0
}
