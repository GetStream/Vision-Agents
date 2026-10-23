package phone

import (
	"context"
	"strconv"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// MaxCallLength is the longest a leg is billed for when nothing says it ended: a lost
// session_ended must not leave a leg billing forever.
const MaxCallLength = 4 * time.Hour

// LegSweepEvery is how often the legs left open past MaxCallLength are ended.
const LegSweepEvery = 10 * time.Minute

// legSweepBatch bounds how many legs one sweep ends, so a backlog is ended over a few sweeps
// rather than in one long statement.
const legSweepBatch = 500

// OpenLeg records that a leg of a call connected, billed to owner when it ends: the request
// that placed it, or the app whose number a caller rang. Its country is where it rang unless
// the leg names its own. An inbound leg already open on the call is not opened again. A leg
// that cannot be recorded is logged, since the call is already up and its minutes are lost.
func (s *Service) OpenLeg(ctx context.Context, owner routing.Owner, leg store.CallLeg) {
	if s.store == nil {
		return
	}
	leg.CustomerID = owner.CustomerID
	leg.OrganizationID, leg.OrganizationTier = owner.OrganizationID, owner.OrganizationTier
	leg.APIKey, leg.UserID, leg.UsageCategories = owner.APIKey, owner.UserID, owner.UsageCategories
	leg.AgentID, leg.Tags = owner.AgentID, owner.Tags
	if leg.Country == "" {
		leg.Country = Country(leg.RemoteNumber)
	}
	leg.StartedAt = s.now()
	if _, err := s.store.OpenCallLeg(context.WithoutCancel(ctx), &leg); err != nil {
		s.logger.Error("could not record a call leg; its minutes are not billed", "call", leg.CallType+":"+leg.CallID,
			"vendor", leg.Vendor, "direction", leg.Direction, "error", err)
	}
}

// EndLegs ends the legs of a call that ended, in the app the event about it came from, and
// bills each for its minutes. A leg already ended, by an earlier delivery of the same event
// or by the sweep, is not billed again.
func (s *Service) EndLegs(ctx context.Context, scope store.AppScope, callType, callID string) error {
	if s.store == nil {
		return nil
	}
	ended, err := s.store.EndCallLegs(ctx, scope, callType, callID, s.now())
	if err != nil {
		return err
	}
	for _, leg := range ended {
		s.settleLeg(ctx, leg)
	}
	return nil
}

// SweepLegs ends the legs of this shard open longer than MaxCallLength, billed to that
// length, and reports how many it ended. Each is logged: a call nobody said ended is either a
// lost event or a call that really ran this long, and both are worth a look.
func (s *Service) SweepLegs(ctx context.Context) (int, error) {
	if s.store == nil {
		return 0, nil
	}
	ended, err := s.store.EndCallLegsOpenSince(ctx, s.now().Add(-MaxCallLength), MaxCallLength, legSweepBatch)
	if err != nil {
		return 0, err
	}
	for _, leg := range ended {
		s.logger.Warn("ended a call leg nothing said had ended, billed to the longest a call is billed for",
			"customer", leg.CustomerID, "call", leg.CallType+":"+leg.CallID, "vendor", leg.Vendor,
			"direction", leg.Direction, "started_at", leg.StartedAt, "billed", MaxCallLength)
		s.settleLeg(ctx, leg)
	}
	return len(ended), nil
}

// RunLegSweep sweeps every interval until ctx ends.
func (s *Service) RunLegSweep(ctx context.Context, every time.Duration) {
	ticker := time.NewTicker(every)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if _, err := s.SweepLegs(ctx); err != nil {
				s.logger.Error("could not end the call legs left open", "error", err)
			}
		}
	}
}

// settleLeg bills an ended leg's minutes to whoever it was opened for, and counts them toward
// the 10DLC sandbox, which gives them away while it holds the app.
func (s *Service) settleLeg(ctx context.Context, leg store.CallLeg) {
	if leg.EndedAt != nil {
		s.gate.Talked(ctx, leg.CustomerID, leg.EndedAt.Sub(leg.StartedAt))
	}
	meter := routing.Meter{
		Recorder: s.recorder,
		Config:   routing.ProviderConfig{Provider: leg.Vendor, Model: ModelCallMinutes},
		Owner:    legOwner(leg),
		UsageID: routing.UsageID("call_leg", leg.CustomerID, leg.CallType, leg.CallID, strconv.FormatInt(leg.ID, 10),
			strconv.FormatInt(leg.StartedAt.UnixMicro(), 10)),
		BilledAt: leg.StartedAt,
		Attributes: map[string]string{
			AttributeCountry:            leg.Country,
			AttributeDirection:          leg.Direction,
			routing.AttributeMeasuredBy: routing.MeasuredByRouter,
		},
		Unmeasured: true,
	}
	if s.sandboxed(ctx, leg.CustomerID) {
		meter.Free = true
		meter.Attributes[AttributeDLCSandbox] = "true"
	}
	if err := routing.Settle(meter, leg); err != nil {
		s.logger.Error("could not bill a call leg", "call", leg.CallType+":"+leg.CallID, "error", stack.Wrap(err))
	}
}

// legOwner is who a leg was opened for, read back from what it kept.
func legOwner(leg store.CallLeg) routing.Owner {
	return routing.Owner{
		CustomerID: leg.CustomerID,
		Attribution: routing.Attribution{
			OrganizationID: leg.OrganizationID, OrganizationTier: leg.OrganizationTier,
			APIKey: leg.APIKey, UserID: leg.UserID, UsageCategories: leg.UsageCategories,
		},
		AgentID: leg.AgentID,
		CallID:  leg.CallID,
		Tags:    leg.Tags,
	}
}

// now is the service's clock, which legs are timed by.
func (s *Service) now() time.Time {
	if s.clock != nil {
		return s.clock()
	}
	return time.Now()
}
