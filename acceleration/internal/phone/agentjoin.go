package phone

import (
	"context"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// agentCheckTimeout bounds each Stream request of the agent check: the read of who is in the
// call, and the end of the call.
const agentCheckTimeout = 10 * time.Second

// awaitAgent checks, agentJoinTimeout after the phone of an outbound leg joined, that someone
// besides that phone is in the call. Nothing else makes sure that an agent is in a call the
// router placed. Without this check, the person who answers hears silence until they hang up.
// The check runs off the hook request. A process that stops first does not do the check, and
// the call continues.
func (s *Service) awaitAgent(leg store.CallLeg) {
	if s.agentJoinTimeout <= 0 || leg.Direction != store.Outbound {
		return
	}
	go func() {
		select {
		case <-s.agentWait(s.agentJoinTimeout):
		case <-s.closing:
			return
		}
		s.checkAgent(leg)
	}()
}

// checkAgent asks Stream once who is in the call of an answered outbound leg. Stream lists a
// router agent, a self-hosted agent, an app user and another phone the same way, so any user
// session besides the leg's phone keeps the call. A call with only the phone in it is ended. A
// call whose phone is gone is left alone, because the person hung up. A call that Stream could
// not list is left alone too: the router does not end a call that it could not check. An agent
// that joins between the read and the end is ended with the call. One read cannot prevent
// this, and the time between the two is one Stream request.
func (s *Service) checkAgent(leg store.CallLeg) {
	attributes := []any{"session_id", leg.CallID, "customer", leg.CustomerID, "from", leg.LocalNumber,
		"to", leg.RemoteNumber, "vendor", leg.Vendor, "vendor_call_id", leg.VendorCallID, "waited", s.agentJoinTimeout}
	ctx, cancel := context.WithTimeout(context.Background(), agentCheckTimeout)
	defer cancel()
	participants, err := s.callParticipants(ctx, leg.CustomerID, leg.StreamAppPK, leg.CallType, leg.CallID)
	if err != nil {
		s.logger.Warn("could not ask Stream who is in the call; the call continues", append(attributes, "error", err)...)
		return
	}
	phoneIn, others := false, false
	for _, participant := range participants {
		if participant == leg.ParticipantSession {
			phoneIn = true
		} else {
			others = true
		}
	}
	switch {
	case !phoneIn:
		s.logger.Debug("the phone left the call before the agent check", attributes...)
	case others:
		s.logger.Debug("someone besides the phone is in the call", attributes...)
	default:
		s.logger.Info("no agent joined the call; ending it", attributes...)
		ctx, cancel := context.WithTimeout(context.Background(), agentCheckTimeout)
		defer cancel()
		if err := s.EndCall(ctx, leg.CustomerID, leg.StreamAppPK, leg.CallType, leg.CallID); err != nil {
			s.logger.Warn("could not end a call no agent joined; it continues until the person hangs up",
				append(attributes, "error", err)...)
		}
	}
}

// callParticipants are the user sessions in a call now, in the app the customer's line is in.
func (s *Service) callParticipants(ctx context.Context, customer string, app int64, callType, callID string) ([]string, error) {
	stream, err := s.streamForApp(ctx, customer, app)
	if err != nil {
		return nil, err
	}
	return stream.CallParticipants(ctx, callType, callID)
}
