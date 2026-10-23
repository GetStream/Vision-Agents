package routing

import (
	"context"
	"errors"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// errNoUsageID is Settle refusing work it could bill twice.
var errNoUsageID = errors.New("routing: settled work needs a usage id, so settling it again bills it once")

// Meter is who a call is recorded for and as what.
type Meter struct {
	// Recorder is where the row goes. Nil records nothing, for a router built without one.
	Recorder *Recorder
	Config   ProviderConfig
	Owner    Owner
	// UsageID names work that may be asked for twice, such as preparing the same voice, so it
	// is billed once. Zero gives the call an id of its own, and so does a Call that fails: a
	// failure is not billed, so it must not take the id from the retry that succeeds.
	UsageID uuid.UUID
	// Code turns a failure into the row's error code. Nil records "provider_error".
	Code func(error) string
	// CostMicros sets the cost outright, for a fee the vendor quoted, such as a number's
	// month. Zero prices what the work used from the price list.
	CostMicros int64
	// Free records work that is given away at no cost and not billed, with what it would have
	// cost in attributes.free_cost_micros.
	Free bool
	// Attributes are stored with the row, such as the session a row belongs to.
	Attributes map[string]string
	// OperationID names what the row is part of, such as the voice session whose seconds it
	// bills. Empty names nothing.
	OperationID string
	// Unmeasured keeps the row out of its provider's health, for work settled after the fact
	// that times nothing the provider did (Stat.Unmeasured).
	Unmeasured bool
	// BilledAt dates the row, for work billed for a time of its own, such as an hour of
	// storage sampled after it ends. Zero dates it when the call started.
	BilledAt time.Time
	// Omit reports whether a call that succeeded did no work worth a row, such as an hour of
	// a knowledge base that holds nothing. A failure is always recorded. Nil omits nothing.
	Omit func(used pricing.Quantities) bool
}

// Identified is a result that names the work it did, such as the clone a provider made, so
// the same work reported twice is billed once and other work is billed again. Its id, when
// it has one, replaces the meter's on a success.
type Identified interface {
	UsageID() uuid.UUID
}

// Call makes one request to a vendor and records it however it ends: one row for a result, an
// error or a panic. A failure is recorded without quantities, so it is not billed.
func Call[R pricing.Metered](ctx context.Context, meter Meter, ask func(context.Context) (R, error)) (R, error) {
	started := time.Now()
	stat := meter.stat(started)
	stat.ErrorCode = "panic"
	omitted := false
	// Deferred so a vendor that panics still leaves its row; the panic carries on.
	defer func() {
		stat.LatencyMs = MsSince(started)
		if !stat.Success {
			stat.UsageID = uuid.Nil
		}
		if !omitted {
			meter.record(stat)
		}
	}()

	result, err := ask(ctx)
	if err != nil {
		stat.ErrorCode, stat.ErrorMessage = meter.code(err), err.Error()
		return result, err
	}
	meter.settle(&stat, result)
	omitted = meter.Omit != nil && meter.Omit(result.Used())
	return result, nil
}

// Settle records work billed after the fact, with no vendor call in hand: a call leg that
// ended, a month of a number. Its meter, or the work (Identified), needs a usage id, so
// settling twice bills once: without one the row is lost, logged in full and counted as
// invalid, and the error says why.
func Settle(meter Meter, work pricing.Metered) error {
	stat := meter.stat(time.Now())
	meter.settle(&stat, work)
	if stat.UsageID == uuid.Nil {
		err := stack.Wrap(errNoUsageID)
		if meter.Recorder != nil {
			meter.Recorder.lose(meter.Config, stat, err)
		}
		return err
	}
	meter.record(stat)
	return nil
}

// stat is the row of work the meter records, before it has used anything.
func (m Meter) stat(started time.Time) Stat {
	if !m.BilledAt.IsZero() {
		started = m.BilledAt
	}
	stat := Stat{Owner: m.Owner, UsageID: m.UsageID, StartedAt: started.UTC(), Free: m.Free, OperationID: m.OperationID,
		Unmeasured: m.Unmeasured}
	for name, value := range m.Attributes {
		stat.Attribute(name, value)
	}
	return stat
}

// settle makes stat the row of work that succeeded: what it used, what the vendor said it
// cost or the provider's formula makes it, and the meter's quoted cost, which only a success
// carries since a failure is not billed.
func (m Meter) settle(stat *Stat, work pricing.Metered) {
	stat.Success, stat.ErrorCode, stat.CostMicros = true, "", m.CostMicros
	if identified, ok := work.(Identified); ok && identified.UsageID() != uuid.Nil {
		stat.UsageID = identified.UsageID()
	}
	stat.Use(work.Used())
	stat.VendorCost(work)
	stat.ProviderCost(work)
	stat.Describe(work)
}

// Attributed is work that says more about itself than what it used, such as the size of the
// pictures drawn, which the price list can key on. What it says goes on its row.
type Attributed interface {
	Attributes() map[string]string
}

// Describe adds what the work says about itself to the stat's attributes, when it says.
func (s *Stat) Describe(work any) {
	if attributed, ok := work.(Attributed); ok {
		for name, value := range attributed.Attributes() {
			s.Attribute(name, value)
		}
	}
}

func (m Meter) record(stat Stat) {
	if m.Recorder != nil {
		m.Recorder.Record(m.Config, stat)
	}
}

func (m Meter) code(err error) string {
	if m.Code == nil {
		return "provider_error"
	}
	return m.Code(err)
}
