package api

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// errNoChorusASR refuses stream/chorus-asr on a router with no chorus-asr server to ask.
var errNoChorusASR = notConfigured("floor_control: stream/chorus-asr cannot control the floor on this router, which has " +
	"no chorus-asr server (ROUTER_CHORUS_ASR_URL); leave floor_control out, or empty, for the default flow controller")

// floorControlOr is what a caller sent in place of stored: left out keeps it, sent replaces
// it, and empty is the default flow controller again.
func floorControlOr(stored store.FloorControl, sent *string) store.FloorControl {
	if sent == nil {
		return stored
	}
	return store.FloorControl(*sent)
}

// renderedFloorControl renders a stored choice for the wire, and nothing for the default flow
// controller.
func renderedFloorControl(stored store.FloorControl) *string {
	if stored.IsZero() {
		return nil
	}
	return new(string(stored))
}

// floorControlRefusal is why a floor control sent is refused, if it is: stream/chorus-asr on a router
// without a chorus-asr server. Only what is sent is held to the router: a stored choice a
// write leaves out stays as it is, and its sessions ask nothing of a router without a server.
func (s *Server) floorControlRefusal(sent *string) (APIError, bool) {
	if sent != nil && session.FloorModeOf(store.FloorControl(*sent)).Asks() && s.audioFloor == nil {
		return errNoChorusASR, true
	}
	return APIError{}, false
}

// floorChoice is what an agent chose of chorus-asr's say over its floor: whether it rules it,
// and when it has a turn ended.
type floorChoice struct {
	control   store.FloorControl
	endOfTurn store.EndOfTurn
}

func floorChoiceOf(config store.AgentConfig) floorChoice {
	return floorChoice{control: config.FloorControl, endOfTurn: config.EndOfTurn}
}

// sessionFloorRefusal is why a new session's request is refused for what it sends of
// floor_control and end_of_turn, if it is: held to what a config's write is, its end_of_turn
// under the floor control the session runs by, its own or its config's.
func (s *Server) sessionFloorRefusal(ctx context.Context, request CreateSessionRequest, config *store.AgentConfig) (APIError, bool) {
	if failure, refused := s.floorControlRefusal(request.FloorControl); refused {
		return failure, true
	}
	var configured store.FloorControl
	if config != nil {
		configured = config.FloorControl
	}
	chosen := floorChoice{control: floorControlOr(configured, request.FloorControl), endOfTurn: endOfTurnOr(store.EndOfTurn{}, request.EndOfTurn)}
	if _, message, ok := s.endOfTurnComplaint(ctx, chosen, floorChoice{}); !ok {
		return invalidRequest(message), true
	}
	return APIError{}, false
}
