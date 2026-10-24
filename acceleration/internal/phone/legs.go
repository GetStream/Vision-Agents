package phone

import (
	"context"
	"maps"
	"slices"
	"strconv"
	"sync"
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

// keepParticipantsFor is how long a phone's join or leave no leg was waiting for is kept for
// the leg that opens after it: far longer than any delivery is late.
const keepParticipantsFor = 24 * time.Hour

// AttributeEndedBy says what ended a leg billed by anything other than its phone leaving or
// the call ending: the sweep.
const AttributeEndedBy = "ended_by"

// OpenLeg records a leg of a call, billed to owner once its phone has joined: the request that
// placed it, or the app whose number a caller rang. An outbound leg rings until its phone joins
// (Connect); pass started for one already connected, as a call that arrived is, from the leg's
// DialedAt, when the event announcing it was made, or now. Its country is where it rang unless
// the leg names its own. An inbound leg its Stream session already has is not opened again. A
// leg that cannot be recorded is logged, since the call is already up and its minutes are lost.
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
	if leg.Direction == store.Outbound {
		s.gate.Dialed(ctx, owner.CustomerID, owner.OrganizationID, leg.RemoteNumber)
	}
	leg.CustomerID = owner.CustomerID
	leg.OrganizationID, leg.OrganizationTier = owner.OrganizationID, owner.OrganizationTier
	leg.APIKey, leg.UserID, leg.UsageCategories = owner.APIKey, owner.UserID, owner.UsageCategories
	leg.AgentID, leg.Tags = owner.AgentID, owner.Tags
	if leg.Country == "" {
		leg.Country = Country(leg.RemoteNumber)
	}
	leg.DialedAt = s.eventAt(leg.DialedAt)
	if started {
		leg.StartedAt = &leg.DialedAt
	}
	opened, err := s.store.OpenCallLeg(ctx, &leg)
	switch {
	case err != nil && !opened:
		s.logger.Error("could not record a call leg; its minutes are not billed", "call", leg.CallType+":"+leg.CallID,
			"vendor", leg.Vendor, "direction", leg.Direction, "error", err)
	case err != nil:
		s.logger.Error("could not give a call leg the phone that joined before it; it starts when the phone's join is delivered again",
			"call", leg.CallType+":"+leg.CallID, "vendor", leg.Vendor, "direction", leg.Direction, "error", err)
	case leg.EndedAt != nil:
		// Its phone joined and left before it was recorded.
		s.settleLegs(ctx, []store.CallLeg{leg})
	}
}

// Connect starts the leg of a phone that joined a call: the caller of a call that arrived, or
// the person a placed call or a transfer rang, who has answered, told apart by the phone's
// number, empty when unknown. Its minutes run from at, when the event saying so was made, or
// now when that is zero. A phone already given its leg, by an earlier delivery, starts nothing.
//
// Stream does not order its call events, so a join no leg is waiting for yet is kept for the
// leg that opens later, and a phone whose leave came first has its leg ended at that leave,
// rather than open, and billed, until the call ends.
func (s *Service) Connect(ctx context.Context, scope store.AppScope, callType, callID, sessionID, participant, number string,
	at time.Time) error {
	if s.store == nil || participant == "" {
		return nil
	}
	at = s.eventAt(at)
	left, err := s.store.ParticipantLeft(ctx, scope, callType, callID, participant)
	if err != nil {
		return err
	}
	if s.connecting != nil {
		s.connecting()
	}
	leg, connection, err := s.store.ConnectCallLeg(ctx, scope, callType, callID, sessionID, participant, number, at, left)
	if err != nil || connection == store.LegKnown {
		return err
	}
	if connection == store.NothingWaiting {
		if err := s.store.RecordParticipant(ctx, scope, store.CallParticipant{
			CallType: callType, CallID: callID, ParticipantSession: participant, Number: number, JoinedAt: &at,
		}, s.now()); err != nil {
			return err
		}
		// A leg opened between the two looks found nothing kept: it is connected here.
		leg, connection, err = s.store.ConnectCallLeg(ctx, scope, callType, callID, sessionID, participant, number, at, left)
		switch {
		case err != nil || connection == store.NothingWaiting:
			return err
		case connection == store.LegKnown:
			return s.store.ForgetParticipant(ctx, scope, callType, callID, participant, nil)
		}
	}
	if leg.EndedAt != nil {
		s.settleLegs(ctx, []store.CallLeg{leg})
	} else if left, err = s.store.ParticipantLeft(ctx, scope, callType, callID, participant); err != nil {
		return err
	} else if left != nil {
		// The phone's leave was kept between the first look and the connect, which did not
		// see it: the leg ends at it, before the leave is forgotten.
		ended, err := s.store.EndParticipantLeg(ctx, scope, callType, callID, participant, *left)
		if err != nil {
			return err
		}
		s.settleLegs(ctx, ended)
	}
	return s.store.ForgetParticipant(ctx, scope, callType, callID, participant, left)
}

// Left ends the leg of a phone that left a call at at (or now, when zero), and bills its
// minutes. A leave no leg has the phone of is kept, for the phone's join delivered after it
// and for the leg that opens later.
func (s *Service) Left(ctx context.Context, scope store.AppScope, callType, callID, participant string, at time.Time) error {
	if s.store == nil || participant == "" {
		return nil
	}
	at = s.eventAt(at)
	ended, err := s.store.EndParticipantLeg(ctx, scope, callType, callID, participant, at)
	if err != nil {
		return err
	}
	if len(ended) > 0 {
		s.settleLegs(ctx, ended)
		return nil
	}
	known, err := s.store.ParticipantHasLeg(ctx, scope, callType, callID, participant)
	if err != nil || known {
		return err
	}
	if err := s.store.RecordParticipant(ctx, scope, store.CallParticipant{
		CallType: callType, CallID: callID, ParticipantSession: participant, LeftAt: &at,
	}, s.now()); err != nil {
		return err
	}
	// A leg given the phone between the two looks did not see the leave: it is ended here.
	if ended, err = s.store.EndParticipantLeg(ctx, scope, callType, callID, participant, at); err != nil {
		return err
	}
	s.settleLegs(ctx, ended)
	return nil
}

// EndLegs ends the legs of a call that ended at at (or now, when zero), in the app the event
// about it came from, and bills each whose phone joined for its minutes. A leg already ended,
// by its phone leaving, an earlier delivery of the same event, or the sweep, is not billed again.
func (s *Service) EndLegs(ctx context.Context, scope store.AppScope, callType, callID string, at time.Time) error {
	if s.store == nil {
		return nil
	}
	ended, err := s.store.EndCallLegs(ctx, scope, callType, callID, s.eventAt(at))
	if err != nil {
		return err
	}
	s.settleLegs(ctx, ended)
	s.checkHooked(ctx, scope, callType, callID, ended)
	return nil
}

// heardRecheck is how long a call found with no outbound leg ringing is not asked about again,
// so a video call with many participants costs a read a minute, and a placed call whose agent
// joined before its leg was written is still heard by the agent's later events.
const heardRecheck = time.Minute

// Heard records that the call hook delivered a participant event of a call, anybody's, which
// is what tells a hook subscribed to them from one that is not (checkHooked). Only a call with
// an outbound leg still ringing is recorded, which is the only one checked: a process records
// each such call once an hour, and asks after any other call at most once every heardRecheck.
func (s *Service) Heard(ctx context.Context, scope store.AppScope, callType, callID string) error {
	if s.store == nil {
		return nil
	}
	key := strconv.FormatInt(scope.App, 10) + ":" + strconv.FormatBool(scope.Unpinned) + ":" + callType + ":" + callID
	now := s.now()
	if !s.heard.first(key, now, heardRecheck) {
		return nil
	}
	ringing, err := s.store.CallRinging(ctx, scope, callType, callID)
	if err != nil || !ringing {
		if err != nil {
			s.heard.forget(key)
		}
		return err
	}
	if err := s.store.HeardCall(ctx, scope, callType, callID, now); err != nil {
		s.heard.forget(key)
		return err
	}
	s.heard.remember(key, now.Add(time.Hour))
	return nil
}

// checkHooked logs at ERROR, once an hour per app, a call that ended with an outbound leg its
// phone never joined and no participant event of it ever delivered. A hook subscribed to them
// delivers the agent's own join on every placed call, so this is an app whose call hook was not
// pointed again (cmd/phone hooks) since the participant events were added to it: its placed
// calls and transfers never connect, and their minutes are not billed.
func (s *Service) checkHooked(ctx context.Context, scope store.AppScope, callType, callID string, ended []store.CallLeg) {
	index := slices.IndexFunc(ended, func(leg store.CallLeg) bool {
		return leg.Direction == store.Outbound && !leg.Connected()
	})
	if index < 0 {
		return
	}
	heard, err := s.store.CallHeard(ctx, scope, callType, callID)
	if err != nil {
		s.logger.Warn("could not tell whether the call hook delivers participant events", "call", callType+":"+callID, "error", err)
		return
	}
	leg := ended[index]
	if heard || !s.unhooked.first(strconv.FormatInt(leg.StreamAppPK, 10), s.now(), time.Hour) {
		return
	}
	s.logger.Error("call hook does not deliver participant events; re-run cmd/phone hooks",
		"stream_app", leg.StreamAppPK, "customer", leg.CustomerID, "call", callType+":"+callID,
		"hint", "a placed call ended unanswered and no participant event of it, not even the agent's join, was delivered: "+
			"placed calls and transfers in this app never connect, and their minutes are not billed")
}

// recently remembers keys until a time each, and at most recentlyMost of them, so a burst of
// new ones forgets the oldest rather than grow without bound.
type recently struct {
	mu    sync.Mutex
	until map[string]time.Time
}

const recentlyMost = 10_000

// first reports whether key is not remembered at now, and remembers it for long if so.
func (r *recently) first(key string, now time.Time, long time.Duration) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if until, ok := r.until[key]; ok && now.Before(until) {
		return false
	}
	r.rememberLocked(key, now, now.Add(long))
	return true
}

// remember remembers key until until.
func (r *recently) remember(key string, until time.Time) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.rememberLocked(key, time.Time{}, until)
}

func (r *recently) rememberLocked(key string, now, until time.Time) {
	if r.until == nil {
		r.until = map[string]time.Time{}
	}
	if _, ok := r.until[key]; !ok && len(r.until) >= recentlyMost {
		maps.DeleteFunc(r.until, func(_ string, until time.Time) bool { return !now.Before(until) })
		if len(r.until) >= recentlyMost {
			clear(r.until)
		}
	}
	r.until[key] = until
}

// forget drops key, so it is first again.
func (r *recently) forget(key string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.until, key)
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
	for {
		deleted, err := s.store.DeleteCallParticipants(ctx, now.Add(-keepParticipantsFor), legSweepBatch)
		if err != nil {
			return len(ended), err
		}
		if deleted < legSweepBatch {
			return len(ended), nil
		}
	}
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
// counts them toward the 10DLC sandbox, which gives them away while it holds the app, and
// toward the organization's voice minutes a day. A leg the sweep ended is given away too, and
// not counted: nothing says how long it really lasted, and four hours of it would use up the
// sandbox's minutes.
func (s *Service) settleLeg(ctx context.Context, leg store.CallLeg, first bool) {
	if first && leg.EndedAt != nil && leg.EndReason != store.LegSwept {
		s.gate.Talked(ctx, leg.CustomerID, leg.OrganizationID, leg.EndedAt.Sub(*leg.StartedAt))
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

// eventLateness is the most an event's own time is trusted to be before it was delivered:
// Stream redelivers within minutes, and a time further back is a misread or a bogus one, which
// would start a leg hours early and have the sweep bill it for four.
const eventLateness = time.Hour

// eventAt is when a call event says it happened, which its legs are timed by, since Stream may
// deliver one late or before one made earlier. An event that says nothing, names a time still to
// come on this clock, or one more than eventLateness before now, happened now.
func (s *Service) eventAt(at time.Time) time.Time {
	now := s.now()
	if at.IsZero() || at.After(now) || now.Sub(at) > eventLateness {
		return now
	}
	return at
}

// now is the service's clock, which legs are timed by when an event says nothing.
func (s *Service) now() time.Time {
	if s.clock != nil {
		return s.clock()
	}
	return time.Now()
}
