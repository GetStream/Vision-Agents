package usage

import (
	"fmt"
	"maps"
	"strconv"
	"time"
)

// Bounds of a tier's settings.
const (
	// MaxTierNameLength is the longest a tier name may be.
	MaxTierNameLength = 64
	// maxChannels is the most channels one set of limits names.
	maxChannels = 32
	// maxCountries is the most countries one channel names.
	maxCountries = 250
)

// FailMode is what admission does with billable work while it cannot read an organization's
// balance: refuse it (closed) or admit it (open).
type FailMode string

const (
	FailClosed FailMode = "closed"
	FailOpen   FailMode = "open"
)

// AccountStatus is whether an organization may work at all. A suspended one is refused
// everything, free work included; its credit can still be written.
type AccountStatus string

const (
	AccountActive    AccountStatus = "active"
	AccountSuspended AccountStatus = "suspended"
)

// Sandbox networks, from closed to open.
const (
	NetworkNone      = "none"
	NetworkAllowList = "allow_list"
	NetworkAny       = "any"
)

// Tier is what one tier string decides. The tier string is the producer's: the service
// assigns none, and the DefaultTier row answers any string with no row of its own.
type Tier struct {
	Name     string   `json:"name" doc:"The tier string, as producers send it: 1 to 64 characters of A-Z, a-z, 0-9, _ . : -."`
	FailMode FailMode `json:"fail_mode" enum:"closed,open" doc:"What admission does with billable work while it cannot read the organization's balance: closed refuses it, open admits it."`
	Limits   Limits   `json:"limits" doc:"The tier's settings. A setting left out is no limit."`
}

// BuiltInDefaultTier answers for every tier string while the database has no DefaultTier row:
// fail closed, and no other setting.
func BuiltInDefaultTier() Tier {
	return Tier{Name: DefaultTier, FailMode: FailClosed}
}

// Limits are the settings a tier decides and an organization may override. In a tier a
// setting left out is no limit; in an organization's overrides it is the tier's. A field the
// schema does not know is refused rather than ignored: overrides are replaced whole, so a
// misspelt setting would clear the ones there and set nothing.
type Limits struct {
	_                struct{}                 `additionalProperties:"false"`
	SpendHardLimit   *SpendHardLimit          `json:"spend_hard_limit,omitempty" doc:"The most the organization may spend, whatever its credit and its own limits."`
	Channels         map[string]ChannelLimits `json:"channels,omitempty" doc:"Limits on each messaging and calling channel, by channel name (sms, whatsapp, rcs, imessage, voice, ...)."`
	Sandbox          *SandboxLimits           `json:"sandbox,omitempty" doc:"Bounds of the organization's sandboxes."`
	LowBalanceMicros *int64                   `json:"low_balance_micros,omitempty" minimum:"0" doc:"The balance, in micros, under which the organization is warned."`
}

// SpendHardLimit is the most one organization may spend, whatever its credit and whatever
// limits it set itself: per UTC day and per calendar month (UTC). Only the operator sets it.
// 0 or absent is no limit for that interval.
type SpendHardLimit struct {
	_             struct{} `additionalProperties:"false"`
	DailyMicros   *int64   `json:"daily_micros,omitempty" minimum:"0" doc:"The most spent in one UTC day, in micros; 0 or absent is no limit."`
	MonthlyMicros *int64   `json:"monthly_micros,omitempty" minimum:"0" doc:"The most spent in one calendar month (UTC), in micros; 0 or absent is no limit."`
}

// ChannelLimits bound one channel. A field left out is no limit.
type ChannelLimits struct {
	_                struct{} `additionalProperties:"false"`
	PerDay           *int     `json:"per_day,omitempty" minimum:"0" doc:"How many a UTC day: messages, or minutes for a calling channel."`
	RecipientsPerDay *int     `json:"recipients_per_day,omitempty" minimum:"0" doc:"How many distinct recipients a UTC day."`
	Countries        []string `json:"countries,omitempty" doc:"The destination countries allowed, as ISO 3166-1 alpha-2 codes. Absent or empty allows any; in overrides, a non-empty list replaces the tier's."`
	Concurrent       *int     `json:"concurrent,omitempty" minimum:"0" doc:"How many at once, for a calling channel."`
}

// SandboxLimits bound one organization's sandboxes.
type SandboxLimits struct {
	_             struct{} `additionalProperties:"false"`
	Network       *string  `json:"network,omitempty" enum:"none,allow_list,any" doc:"What a sandbox may reach: nothing, an allow list, or anything."`
	MaxCPU        *int     `json:"max_cpu,omitempty" minimum:"0" doc:"The most vCPUs of one sandbox."`
	MaxMemoryGB   *int     `json:"max_memory_gb,omitempty" minimum:"0" doc:"The most memory of one sandbox, in GiB."`
	MaxDiskGB     *int     `json:"max_disk_gb,omitempty" minimum:"0" doc:"The most disk of one sandbox, in GiB."`
	MaxConcurrent *int     `json:"max_concurrent,omitempty" minimum:"0" doc:"How many sandboxes at once."`
}

// SpendWindows is the organization's billable spend in the current UTC day and calendar
// month, for the spend hard limit. Each is the organization subject's window as
// /v1/windows answers it.
type SpendWindows struct {
	Daily   WindowSpend `json:"daily"`
	Monthly WindowSpend `json:"monthly"`
}

// Over lays every setting the overrides set over these: field by field, channel by channel,
// and within the spend hard limit, a channel and the sandbox field by field.
func (l Limits) Over(overrides Limits) Limits {
	out := l
	if o := overrides.SpendHardLimit; o != nil {
		spend := SpendHardLimit{}
		if l.SpendHardLimit != nil {
			spend = *l.SpendHardLimit
		}
		spend.DailyMicros = overridden(o.DailyMicros, spend.DailyMicros)
		spend.MonthlyMicros = overridden(o.MonthlyMicros, spend.MonthlyMicros)
		out.SpendHardLimit = &spend
	}
	if len(overrides.Channels) > 0 {
		out.Channels = maps.Clone(l.Channels)
		if out.Channels == nil {
			out.Channels = make(map[string]ChannelLimits, len(overrides.Channels))
		}
		for name, o := range overrides.Channels {
			channel := out.Channels[name]
			channel.PerDay = overridden(o.PerDay, channel.PerDay)
			channel.RecipientsPerDay = overridden(o.RecipientsPerDay, channel.RecipientsPerDay)
			channel.Concurrent = overridden(o.Concurrent, channel.Concurrent)
			if len(o.Countries) > 0 {
				channel.Countries = o.Countries
			}
			out.Channels[name] = channel
		}
	}
	if o := overrides.Sandbox; o != nil {
		sandbox := SandboxLimits{}
		if l.Sandbox != nil {
			sandbox = *l.Sandbox
		}
		sandbox.Network = overridden(o.Network, sandbox.Network)
		sandbox.MaxCPU = overridden(o.MaxCPU, sandbox.MaxCPU)
		sandbox.MaxMemoryGB = overridden(o.MaxMemoryGB, sandbox.MaxMemoryGB)
		sandbox.MaxDiskGB = overridden(o.MaxDiskGB, sandbox.MaxDiskGB)
		sandbox.MaxConcurrent = overridden(o.MaxConcurrent, sandbox.MaxConcurrent)
		out.Sandbox = &sandbox
	}
	out.LowBalanceMicros = overridden(overrides.LowBalanceMicros, l.LowBalanceMicros)
	return out
}

// overridden is the override when it is set, else the base.
func overridden[T any](override, base *T) *T {
	if override != nil {
		return override
	}
	return base
}

// Compact is the limits without the settings that set nothing: a spend hard limit, a sandbox
// or a channel with no field, and no channels at all. Two limits that lay the same over a
// tier compact the same, so overrides of {"sandbox": {}} are none.
func (l Limits) Compact() Limits {
	if s := l.SpendHardLimit; s != nil && s.DailyMicros == nil && s.MonthlyMicros == nil {
		l.SpendHardLimit = nil
	}
	if s := l.Sandbox; s != nil && *s == (SandboxLimits{}) {
		l.Sandbox = nil
	}
	channels := make(map[string]ChannelLimits, len(l.Channels))
	for name, channel := range l.Channels {
		if channel.PerDay != nil || channel.RecipientsPerDay != nil || channel.Concurrent != nil || len(channel.Countries) > 0 {
			channels[name] = channel
		}
	}
	l.Channels = nil
	if len(channels) > 0 {
		l.Channels = channels
	}
	return l
}

// IsZero reports whether the limits set nothing.
func (l Limits) IsZero() bool {
	c := l.Compact()
	return c.SpendHardLimit == nil && c.Channels == nil && c.Sandbox == nil && c.LowBalanceMicros == nil
}

// SpendHardLimitMicros is the spend hard limit of interval, Daily or Monthly, and whether
// there is one: 0, absent, or any other interval is none.
func (l Limits) SpendHardLimitMicros(interval Interval) (int64, bool) {
	if l.SpendHardLimit == nil {
		return 0, false
	}
	var amount *int64
	switch interval {
	case Daily:
		amount = l.SpendHardLimit.DailyMicros
	case Monthly:
		amount = l.SpendHardLimit.MonthlyMicros
	}
	if amount == nil || *amount <= 0 {
		return 0, false
	}
	return *amount, true
}

// Problem says what is wrong with the limits, or nothing.
func (l Limits) Problem() string {
	if s := l.SpendHardLimit; s != nil {
		if problem := notNegative("spend_hard_limit.daily_micros", s.DailyMicros); problem != "" {
			return problem
		}
		if problem := notNegative("spend_hard_limit.monthly_micros", s.MonthlyMicros); problem != "" {
			return problem
		}
	}
	if len(l.Channels) > maxChannels {
		return fmt.Sprintf("channels names %d channels, at most %d", len(l.Channels), maxChannels)
	}
	for name, channel := range l.Channels {
		if !validName(name) {
			return fmt.Sprintf("channel %q is not 1 to %d characters of A-Z, a-z, 0-9, _ . : -", name, MaxTierNameLength)
		}
		for field, value := range map[string]*int{"per_day": channel.PerDay, "recipients_per_day": channel.RecipientsPerDay, "concurrent": channel.Concurrent} {
			if problem := notNegative("channels."+name+"."+field, value); problem != "" {
				return problem
			}
		}
		if len(channel.Countries) > maxCountries {
			return fmt.Sprintf("channels.%s.countries names %d countries, at most %d", name, len(channel.Countries), maxCountries)
		}
		for _, country := range channel.Countries {
			if !countryCode(country) {
				return fmt.Sprintf("channels.%s.countries: %q is not an ISO 3166-1 alpha-2 code such as US", name, country)
			}
		}
	}
	if s := l.Sandbox; s != nil {
		if s.Network != nil {
			switch *s.Network {
			case NetworkNone, NetworkAllowList, NetworkAny:
			default:
				return fmt.Sprintf("sandbox.network %q is not one of none, allow_list, any", *s.Network)
			}
		}
		for field, value := range map[string]*int{"max_cpu": s.MaxCPU, "max_memory_gb": s.MaxMemoryGB, "max_disk_gb": s.MaxDiskGB, "max_concurrent": s.MaxConcurrent} {
			if problem := notNegative("sandbox."+field, value); problem != "" {
				return problem
			}
		}
	}
	return notNegative("low_balance_micros", l.LowBalanceMicros)
}

// Problem says what is wrong with the tier, or nothing.
func (t Tier) Problem() string {
	if !ValidTierName(t.Name) {
		return fmt.Sprintf("tier name %q is not 1 to %d characters of A-Z, a-z, 0-9, _ . : -", t.Name, MaxTierNameLength)
	}
	switch t.FailMode {
	case FailClosed, FailOpen:
	case "":
		return "fail_mode is required: closed or open"
	default:
		return fmt.Sprintf("fail_mode %q is not closed or open", t.FailMode)
	}
	return t.Limits.Problem()
}

// ValidTierName reports whether name is 1 to 64 characters of [A-Za-z0-9_.:-].
func ValidTierName(name string) bool {
	return validName(name)
}

// validName reports whether name is 1 to 64 characters of [A-Za-z0-9_.:-].
func validName(name string) bool {
	return name != "" && len(name) <= MaxTierNameLength && validCategory(name)
}

// countryCode reports whether code is two capital letters.
func countryCode(code string) bool {
	return len(code) == 2 && code[0] >= 'A' && code[0] <= 'Z' && code[1] >= 'A' && code[1] <= 'Z'
}

// notNegative says what is wrong with a setting below zero, or nothing.
func notNegative[T int | int64](name string, value *T) string {
	if value != nil && *value < 0 {
		return name + " is " + strconv.FormatInt(int64(*value), 10) + ": it cannot be negative"
	}
	return ""
}

// CreditKind labels a credit entry for whoever reads it, and fixes the sign of its amount;
// nothing else acts on it.
type CreditKind string

const (
	CreditTopUp           CreditKind = "top_up"
	CreditGrant           CreditKind = "grant"
	CreditRefund          CreditKind = "refund"
	CreditDispute         CreditKind = "dispute"
	CreditDisputeReversal CreditKind = "dispute_reversal"
	CreditAdjustment      CreditKind = "adjustment"
)

// AmountProblem says what is wrong with amount for an entry of this kind, or nothing: a top
// up, a grant and a dispute reversal add credit, a refund and a dispute take it away, and an
// adjustment does either. No entry is 0.
func (k CreditKind) AmountProblem(amount int64) string {
	switch k {
	case CreditTopUp, CreditGrant, CreditDisputeReversal:
		if amount <= 0 {
			return fmt.Sprintf("a %s adds credit: amount_micros must be above 0, not %d", k, amount)
		}
	case CreditRefund, CreditDispute:
		if amount >= 0 {
			return fmt.Sprintf("a %s takes credit away: amount_micros must be below 0, not %d", k, amount)
		}
	case CreditAdjustment:
		if amount == 0 {
			return "an adjustment cannot be 0"
		}
	default:
		return fmt.Sprintf("kind %q is not one of top_up, grant, refund, dispute, dispute_reversal, adjustment", k)
	}
	return ""
}

// CreditEntry is one change to an organization's credit, written once under its idempotency
// key.
type CreditEntry struct {
	ID             int64      `json:"id"`
	OrganizationID string     `json:"organization_id"`
	Kind           CreditKind `json:"kind" enum:"top_up,grant,refund,dispute,dispute_reversal,adjustment"`
	AmountMicros   int64      `json:"amount_micros" doc:"Signed: what the entry added to the organization's credit, in micros."`
	Reference      string     `json:"reference,omitempty" doc:"The writer's own reference, such as a payment id. Never read by the service."`
	IdempotencyKey string     `json:"idempotency_key" doc:"The key it was written under: writing again under it changes nothing."`
	Note           string     `json:"note,omitempty"`
	Actor          string     `json:"actor" doc:"Who wrote it, as X-Stream-Actor-Id named them."`
	ActorName      string     `json:"actor_name,omitempty" doc:"Their name, as X-Stream-Actor-Name named it."`
	ReversesID     int64      `json:"reverses_id,omitempty" doc:"The entry this adjustment reverses: a grant it revokes, or the revocation it undoes. Absent on an entry that reverses none."`
	ReversedByID   int64      `json:"reversed_by_id,omitempty" doc:"The entry that reverses this one. Absent while none does."`
	CreatedAt      time.Time  `json:"created_at"`
}
