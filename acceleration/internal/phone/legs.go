package phone

import (
	"context"
	"strconv"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// MaxCallLength is the longest a leg may stay open: a leg still open then is ended by the
// sweep, and given away rather than billed, since nothing said how long it really lasted.
const MaxCallLength = 4 * time.Hour

// LegSweepEvery is how often the legs left open past MaxCallLength are ended, and the legs
// that ended lately checked for their row.
const LegSweepEvery = 10 * time.Minute

// legSweepBatch bounds how many legs one sweep ends or checks, so a backlog is worked through
// over a few sweeps rather than in one long statement.
const legSweepBatch = 500

// The legs a sweep checks were billed: those that ended between rebillFrom and rebillTo ago,
// long enough for the writer to have written their row, and recent enough that a process that
// died between ending a leg and writing its row has its legs checked on the next start.
const (
	rebillFrom = 2 * time.Hour
	rebillTo   = 5 * time.Minute
)

// AttributeEndedBy says what ended a leg billed by anything other than its phone leaving or
// the call ending: the sweep.
const AttributeEndedBy = "ended_by"

// OpenLeg records a leg of a call, billed to owner once its phone has joined: the request that
// placed it, or the app whose number a caller rang. An outbound leg rings until its phone joins
// (Connect); pass started for one already connected, as a call that arrived is. Its country is
// where it rang unless the leg names its own. An inbound leg its Stream session already has is
// not opened again. A leg that cannot be recorded is logged, since the call is already up and
// its minutes are lost.
func (s *Service) OpenLeg(ctx context.Context, owner routing.Owner, leg store.CallLeg, started bool) {
	if s.store == nil {
		return
	}
	ctx = context.WithoutCancel(ctx)
	if owner.OrganizationID == "" {
		// Known when the row is written, but the concurrent calls of an organization are counted
		// from its open legs.
		organization, err := s.store.OrganizationOf(ctx, owner.CustomerID)
		if err != nil {
			s.logger.Warn("could not tell whose organization a call leg is", "customer", owner.CustomerID, "error", err)
		}
		owner.OrganizationID = organization
	}
	leg.CustomerID = owner.CustomerID
	leg.OrganizationID, leg.OrganizationTier = owner.OrganizationID, owner.OrganizationTier
	leg.APIKey, leg.UserID, leg.UsageCategories = owner.APIKey, owner.UserID, owner.UsageCategories
	leg.AgentID, leg.Tags = owner.AgentID, owner.Tags
	if leg.Country == "" {
		leg.Country = Country(leg.RemoteNumber)
	}
	leg.DialedAt = s.now()
	if started {
		leg.StartedAt = &leg.DialedAt
	}
	if _, err := s.store.OpenCallLeg(ctx, &leg); err != nil {
		s.logger.Error("could not record a call leg; its minutes are not billed", "call", leg.CallType+":"+leg.CallID,
			"vendor", leg.Vendor, "direction", leg.Direction, "error", err)
	}
}

// Connect starts the leg of a phone that joined a call: the caller of a call that arrived, or
// the person a placed call or a transfer rang, who has answered. Its minutes run from now. A
// phone already given its leg, by an earlier delivery, starts nothing.
func (s *Service) Connect(ctx context.Context, scope store.AppScope, callType, callID, sessionID, participant string) error {
	if s.store == nil || participant == "" {
		return nil
	}
	_, _, err := s.store.ConnectCallLeg(ctx, scope, callType, callID, sessionID, participant, s.now())
	return err
}

// Left ends the leg of a phone that left a call, and bills its minutes.
func (s *Service) Left(ctx context.Context, scope store.AppScope, callType, callID, participant string) error {
	if s.store == nil || participant == "" {
		return nil
	}
	ended, err := s.store.EndParticipantLeg(ctx, scope, callType, callID, participant, s.now())
	if err != nil {
		return err
	}
	s.settleLegs(ctx, ended)
	return nil
}

// EndLegs ends the legs of a call that ended, in the app the event about it came from, and
// bills each whose phone joined for its minutes. A leg already ended, by its phone leaving, an
// earlier delivery of the same event, or the sweep, is not billed again.
func (s *Service) EndLegs(ctx context.Context, scope store.AppScope, callType, callID string) error {
	if s.store == nil {
		return nil
	}
	ended, err := s.store.EndCallLegs(ctx, scope, callType, callID, s.now())
	if err != nil {
		return err
	}
	s.settleLegs(ctx, ended)
	return nil
}

// settleLegs bills the legs that ended whose phone joined. One that never connected was not
// answered, which a vendor does not bill: it is logged, since a call hook that does not deliver
// call.session_participant_joined would leave every placed call looking so.
func (s *Service) settleLegs(ctx context.Context, ended []store.CallLeg) {
	for _, leg := range ended {
		if !leg.Connected() {
			s.logger.Warn("a call leg ended without its phone joining the call; it is not billed",
				"customer", leg.CustomerID, "call", leg.CallType+":"+leg.CallID, "vendor", leg.Vendor,
				"direction", leg.Direction, "hint", "unanswered, or the call hook does not deliver call.session_participant_joined")
			continue
		}
		s.settleLeg(ctx, leg, true)
	}
}

// SweepLegs ends the legs of this shard open longer than MaxCallLength, and checks the legs
// that ended lately have their row, writing again any a process lost between ending the leg and
// writing it. It reports how many legs it ended.
//
// A leg whose phone joined and that nothing ended is logged at ERROR and recorded priced and
// given away at MaxCallLength: a call that ran four hours is rare, a lost session_ended is not,
// and nobody can tell which it was.
func (s *Service) SweepLegs(ctx context.Context) (int, error) {
	if s.store == nil {
		return 0, nil
	}
	now := s.now()
	ended, err := s.store.SweepCallLegs(ctx, now, MaxCallLength, legSweepBatch)
	if err != nil {
		return 0, err
	}
	for _, leg := range ended {
		if !leg.Connected() {
			s.logger.Warn("ended a call leg that rang for the longest a call may last without its phone joining",
				"customer", leg.CustomerID, "call", leg.CallType+":"+leg.CallID, "vendor", leg.Vendor)
			continue
		}
		s.logger.Error("ended a call leg nothing said had ended; it is recorded and not billed",
			"customer", leg.CustomerID, "call", leg.CallType+":"+leg.CallID, "vendor", leg.Vendor,
			"direction", leg.Direction, "started_at", leg.StartedAt, "recorded", MaxCallLength)
		s.settleLeg(ctx, leg, true)
	}
	if err := s.rebill(ctx, now); err != nil {
		return len(ended), err
	}
	return len(ended), nil
}

// rebill writes the row of each leg that ended lately and has none, under the usage id it
// was first meant to have, so a row written meanwhile is not written twice.
func (s *Service) rebill(ctx context.Context, now time.Time) error {
	legs, err := s.store.EndedCallLegs(ctx, now.Add(-rebillFrom), now.Add(-rebillTo), legSweepBatch)
	if err != nil || len(legs) == 0 {
		return err
	}
	ids := make([]uuid.UUID, len(legs))
	for i, leg := range legs {
		ids[i] = legUsageID(leg)
	}
	recorded, err := s.store.RecordedUsage(ctx, ids)
	if err != nil {
		return err
	}
	for i, leg := range legs {
		if recorded[ids[i]] {
			continue
		}
		s.logger.Warn("a call leg that ended had no usage row; it is recorded again",
			"customer", leg.CustomerID, "call", leg.CallType+":"+leg.CallID, "ended_at", leg.EndedAt)
		s.settleLeg(ctx, leg, false)
	}
	return nil
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
				s.logger.Error("could not sweep the call legs", "error", err)
			}
		}
	}
}

// settleLeg bills an ended leg's minutes to whoever it was opened for, and the first time,
// counts them toward the 10DLC sandbox, which gives them away while it holds the app. A leg
// the sweep ended is given away too.
func (s *Service) settleLeg(ctx context.Context, leg store.CallLeg, first bool) {
	if first && leg.EndedAt != nil {
		s.gate.Talked(ctx, leg.CustomerID, leg.EndedAt.Sub(*leg.StartedAt))
	}
	meter := routing.Meter{
		Recorder: s.recorder,
		Config:   routing.ProviderConfig{Provider: leg.Vendor, Model: ModelCallMinutes},
		Owner:    legOwner(leg),
		UsageID:  legUsageID(leg),
		BilledAt: *leg.StartedAt,
		Attributes: map[string]string{
			AttributeCountry:            leg.Country,
			AttributeDirection:          leg.Direction,
			routing.AttributeMeasuredBy: routing.MeasuredByRouter,
		},
		Unmeasured: true,
	}
	if leg.EndReason == store.LegSwept {
		meter.Free = true
		meter.Attributes[AttributeEndedBy] = store.LegSwept
	}
	if s.sandboxed(ctx, leg.CustomerID) {
		meter.Free = true
		meter.Attributes[AttributeDLCSandbox] = "true"
	}
	if err := routing.Settle(meter, leg); err != nil {
		s.logger.Error("could not bill a call leg", "call", leg.CallType+":"+leg.CallID, "error", err)
	}
}

// legUsageID names a leg's row. A leg's id is only unique within its shard, so the call and
// when it was dialled are part of it.
func legUsageID(leg store.CallLeg) uuid.UUID {
	return routing.UsageID("call_leg", leg.CustomerID, leg.CallType, leg.CallID, strconv.FormatInt(leg.ID, 10),
		strconv.FormatInt(leg.DialedAt.UnixMicro(), 10))
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
