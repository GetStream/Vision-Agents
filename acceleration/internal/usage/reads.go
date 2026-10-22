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

// Standing is what the service knows about one organization that admission reads.
type Standing struct {
	OrganizationID string `json:"organization_id"`
}

// Standings is the answer to a StandingQuery, one standing per organization asked about.
type Standings struct {
	AsOf      time.Time  `json:"as_of"`
	Standings []Standing `json:"standings"`
}
