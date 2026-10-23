package usage

import "time"

// WindowsQuery names the subjects whose current windows are asked for.
type WindowsQuery struct {
	Subjects []SubjectRef `json:"subjects"`
}

// WindowSpend is what one subject spent in one window of one interval.
type WindowSpend struct {
	SubjectRef
	Interval    Interval  `json:"interval"`
	WindowStart time.Time `json:"window_start"`
	WindowEnd   time.Time `json:"window_end"`
	// SpentMicros sums billable cost; Events counts every event, billable or not.
	SpentMicros int64 `json:"spent_micros"`
	Events      int64 `json:"events"`
}

// Windows is the answer to a WindowsQuery: every subject asked about in each of its current
// windows, a window nobody has spent in included at zero.
type Windows struct {
	AsOf    time.Time     `json:"as_of"`
	Windows []WindowSpend `json:"windows"`
}

// StandingQuery names the organizations whose standing is asked for.
type StandingQuery struct {
	Organizations []StandingOrganization `json:"organizations"`
}

// StandingOrganization is one organization a standing is asked for, with the tier the
// producer last saw on its work.
type StandingOrganization struct {
	OrganizationID   string `json:"organization_id"`
	OrganizationTier string `json:"organization_tier,omitempty"`
}

// Standing is what the service knows about one organization that admission reads. An
// organization with no account is active, with nothing granted and nothing spent.
type Standing struct {
	OrganizationID string `json:"organization_id"`
	// OrganizationTier is the tier string the standing was asked for, which may have no row.
	OrganizationTier string `json:"organization_tier"`
	// Tier is the row of the tier asked about, or the default tier's when it has none.
	Tier         Tier `json:"tier"`
	FallbackTier bool `json:"fallback_tier" doc:"The tier asked about has no row: tier is the default tier's."`
	// Limits are the tier's with the organization's overrides laid over them, the settings
	// in force; Overrides are the organization's own.
	Limits        Limits        `json:"limits" doc:"The settings in force: the tier's with the organization's overrides laid over them."`
	Overrides     Limits        `json:"limit_overrides" doc:"The organization's own settings, each replacing the tier's."`
	Status        AccountStatus `json:"status" enum:"active,suspended"`
	StatusReason  string        `json:"status_reason,omitempty" doc:"Why the organization is suspended."`
	GrantedMicros int64         `json:"granted_micros" doc:"Every credit entry summed."`
	SpentMicros   int64         `json:"spent_micros" doc:"Every billable event's cost summed."`
	BalanceMicros int64         `json:"balance_micros" doc:"granted_micros less spent_micros."`
	SpendWindows  SpendWindows  `json:"spend_windows" doc:"The organization's spend in the current UTC day and calendar month, for the spend hard limit."`
}

// Standings is the answer to a StandingQuery, one standing per organization asked about.
type Standings struct {
	AsOf      time.Time  `json:"as_of"`
	Standings []Standing `json:"standings"`
}
