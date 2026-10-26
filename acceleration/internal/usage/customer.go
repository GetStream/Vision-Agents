package usage

import (
	"bytes"
	"encoding/json"
	"slices"
	"strings"
	"time"
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
