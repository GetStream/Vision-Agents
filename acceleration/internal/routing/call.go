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
	// is billed once. Zero gives the call an id of its own.
	UsageID uuid.UUID
	// Code turns a failure into the row's error code. Nil records "provider_error".
	Code func(error) string
	// CostMicros sets the cost outright, for a fee the vendor quoted, such as a number's
	// month. Zero prices what the work used from the price list.
	CostMicros int64
	// Free records the work priced as usual and not billed, for work that is given away.
	Free bool
	// Attributes are stored with the row, such as the session a row belongs to.
	Attributes map[string]string
}

// Call makes one request to a vendor and records it however it ends: one row for a result, an
// error or a panic. A failure is recorded without quantities, so it is not billed.
func Call[R pricing.Metered](ctx context.Context, meter Meter, ask func(context.Context) (R, error)) (R, error) {
	started := time.Now()
	stat := meter.stat(started)
	stat.ErrorCode = "panic"
	// Deferred so a vendor that panics still leaves its row; the panic carries on.
	defer func() {
		stat.LatencyMs = MsSince(started)
		meter.record(stat)
	}()

	result, err := ask(ctx)
	if err != nil {
		stat.ErrorCode, stat.ErrorMessage = meter.code(err), err.Error()
		return result, err
	}
	meter.settle(&stat, result)
	return result, nil
}

// Settle records work billed after the fact, with no vendor call in hand: a call leg that
// ended, a month of a number. Its meter needs a UsageID, so settling twice bills once: without
// one the row is lost, logged in full and counted as invalid, and the error says why.
func Settle(meter Meter, work pricing.Metered) error {
	stat := meter.stat(time.Now())
	meter.settle(&stat, work)
	if meter.UsageID == uuid.Nil {
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
	stat := Stat{Owner: m.Owner, UsageID: m.UsageID, StartedAt: started.UTC(), Free: m.Free}
	for name, value := range m.Attributes {
		stat.Attribute(name, value)
	}
	return stat
}

// settle makes stat the row of work that succeeded: what it used, what the vendor said it
// cost, and the meter's quoted cost, which only a success carries since a failure is not
// billed.
func (m Meter) settle(stat *Stat, work pricing.Metered) {
	stat.Success, stat.ErrorCode, stat.CostMicros = true, "", m.CostMicros
	stat.Use(work.Used())
	stat.VendorCost(work)
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
