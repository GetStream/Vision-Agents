// Package person is the only thing the tool knows about a provider: the side of the call a
// person would be on.
package person

import (
	"context"
	"flag"
	"log/slog"
	"time"
)

type Direction int

const (
	Outbound Direction = iota
	Inbound
)

func (d Direction) String() string {
	if d == Inbound {
		return "inbound"
	}
	return "outbound"
}

// LengthOf is how long a call lasted by its record's start and end, zero without both.
func LengthOf(start, end time.Time) time.Duration {
	if start.IsZero() || end.IsZero() {
		return 0
	}
	return end.Sub(start)
}

// Call is what every adapter needs for one call.
type Call struct {
	ID           string // the run's call id, on every log line
	Direction    Direction
	PersonNumber string
	AgentNumber  string
	Talk         time.Duration
	RegionLabel  string       // the free -region label; logged, changes nothing
	Started      time.Time    // when the run's checks passed; outbound Wait finds the person's call from it
	Logger       *slog.Logger // already carries call_id
}

type Person interface {
	Name() string
	// Flags adds the adapter's own input.
	Flags(*flag.FlagSet)
	// Ready says what the adapter is missing for a call in this direction, before any call is
	// placed.
	Ready(Direction) error
	// Dial calls the agent number from the person number with an inline script; returns once
	// placed. Inbound only.
	Dial(ctx context.Context, call Call) error
	// Wait blocks until the person's call has ended, logging chosen fields of the vendor's
	// call record. It returns how long the call lasted by that record, zero when the call was
	// not observed or the record lacks its start or end.
	Wait(ctx context.Context, call Call) (time.Duration, error)
	// Hangup ends the call it placed or found, if it is still live, and forgets it.
	Hangup(ctx context.Context, call Call) error
}
