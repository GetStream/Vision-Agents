package api

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// FloorControl is the model that controls an agent's floor, and how long a caller may wait
// once they stop.
type FloorControl struct {
	Model     *string `json:"model,omitempty" pattern:"^(stream/chorus-asr|flow-controller)$" example:"stream/chorus-asr" doc:"The model that controls the floor, hearing the caller's audio: when they have finished, and what to do about speech over the agent. stream/chorus-asr is the one model there is; flow-controller keeps the flow controller, which rules the floor where no model does, whatever the router defaults an agent that chose none to. Required, unless floor_control is empty. stream/chorus-asr on a router without a chorus-asr server is refused (400 not_configured)."`
	MaxWaitMs *int    `json:"max_wait_ms,omitempty" minimum:"1" example:"800" doc:"The most a caller who has finished may wait for the answer, in milliseconds. It chooses the point of the model's measured trade-off with the fewest cut-ins whose measured p90 wait is within it, and is refused (400) when it is shorter than every point's. Left out, the server's default point. It is stream/chorus-asr's: with flow-controller it is refused (400)."`
}

func (*FloorControl) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The model that controls an agent's floor, and how long a caller may wait once they stop. " +
		"Left out or null, the stored choice stays; an empty object is no choice again, which the router's default rules: " +
		"the flow controller, or stream/chorus-asr at the server's default point where the router defaults every agent to it; " +
		"{model: flow-controller} keeps the flow controller whatever the router's default; " +
		"anything else replaces the stored choice whole."
	schema.AdditionalProperties = false
	return schema
}

// errNoChorusASR refuses stream/chorus-asr on a router with no chorus-asr server to ask.
var errNoChorusASR = notConfigured("floor_control: stream/chorus-asr cannot control the floor on this router, which has " +
	"no chorus-asr server (ROUTER_CHORUS_ASR_URL); leave floor_control out, send {}, or name flow-controller: " +
	"the flow controller rules every floor on this router")

// errEndOfTurn refuses end_of_turn, which floor_control's max_wait_ms replaced.
var errEndOfTurn = invalidRequest("end_of_turn is no longer read: send floor_control: " +
	"{model: stream/chorus-asr, max_wait_ms: N}, where max_wait_ms is how long a caller may wait once they stop")

// floorControlOr is what a caller sent in place of stored: left out keeps it, sent replaces
// it whole, and empty is no choice again, which the router's default rules.
func floorControlOr(stored store.FloorControl, sent *FloorControl) store.FloorControl {
	if sent == nil {
		return stored
	}
	return store.FloorControl{Model: value(sent.Model), MaxWaitMs: value(sent.MaxWaitMs)}
}

// renderedFloorControl renders a stored choice for the wire, the opt-out included, and nothing
// for none chosen: an agent reads back what it stored.
func renderedFloorControl(stored store.FloorControl) *FloorControl {
	if stored.IsZero() {
		return nil
	}
	rendered := &FloorControl{Model: new(stored.Model)}
	if stored.MaxWaitMs != 0 {
		rendered.MaxWaitMs = new(stored.MaxWaitMs)
	}
	return rendered
}

// floorControlRefusal is why a floor control sent is refused, if it is: end_of_turn, which it
// replaced; what the schema says, held here too for the socket's start frame, which no schema
// checks; and stream/chorus-asr on a router without a chorus-asr server. Only what is sent is
// held to the router: a stored choice a write leaves out stays as it is, and its sessions ask
// nothing of a router without a server.
func (s *Server) floorControlRefusal(sent *FloorControl, endOfTurn any) (APIError, bool) {
	if endOfTurn != nil {
		return errEndOfTurn, true
	}
	if sent == nil {
		return APIError{}, false
	}
	chosen := floorControlOr(store.FloorControl{}, sent)
	switch {
	case chosen.Model == "" && sent.MaxWaitMs != nil:
		return invalidRequest("floor_control.model is required: name stream/chorus-asr, or flow-controller, or send {} for the router's default"), true
	case chosen.Model != "" && chosen.Model != store.FloorControlChorusASR && chosen.Model != store.FloorControlFlowController:
		return invalidRequest(fmt.Sprintf("floor_control.model is %q: stream/chorus-asr is the one model there is, and flow-controller keeps the flow controller", chosen.Model)), true
	case chosen.Model == store.FloorControlFlowController && sent.MaxWaitMs != nil:
		return invalidRequest("floor_control.max_wait_ms is stream/chorus-asr's: the flow controller has no wait to choose"), true
	case sent.MaxWaitMs != nil && *sent.MaxWaitMs < 1:
		return invalidRequest("floor_control.max_wait_ms is a number of milliseconds above 0"), true
	case session.FloorModeOf(chosen).Asks() && s.audioFloor == nil:
		return errNoChorusASR, true
	}
	return APIError{}, false
}

// sessionFloorRefusal is why a new session's request is refused for what it sends of
// floor_control, if it is: held to what a config's write is.
func (s *Server) sessionFloorRefusal(ctx context.Context, request CreateSessionRequest) (APIError, bool) {
	if failure, refused := s.floorControlRefusal(request.FloorControl, request.EndOfTurn); refused {
		return failure, true
	}
	if request.FloorControl == nil {
		return APIError{}, false
	}
	if _, message, ok := s.floorControlComplaint(ctx, floorControlOr(store.FloorControl{}, request.FloorControl), store.FloorControl{}); !ok {
		return invalidRequest(message), true
	}
	return APIError{}, false
}

// maxWaitCheck bounds asking the chorus-asr server what it publishes while an agent is
// written, so a server that is down does not hold the write.
const maxWaitCheck = 2 * time.Second

// floorControlComplaint reports what is wrong with a floor control's max_wait_ms, if anything,
// and what a sync should be told about one it keeps that may not do what it says. chosen is
// the agent's floor control after the write, and was before it.
//
// A wait is checked against the frontier the chorus-asr server publishes, so one shorter than
// every measured point's is refused while the agent is written, unless it is unchanged. A wait
// the server could not be asked about is kept, with a warning: the agent is not refused for a
// server that is down. So is one that chooses a point failing the dictation guard, with what it
// was measured to do.
func (s *Server) floorControlComplaint(ctx context.Context, chosen, was store.FloorControl) (warning, message string, ok bool) {
	if chosen.MaxWaitMs == 0 || chosen == was || s.audioFloor == nil || !session.FloorModeOf(chosen).Asks() {
		return "", "", true
	}
	// As the server publishes it now, rather than as the sessions last read it: the agent is
	// written once, and a session reads the policy again as it goes.
	asking, cancel := context.WithTimeout(ctx, maxWaitCheck)
	defer cancel()
	policy, err := s.audioFloor.PolicyNow(asking)
	if err != nil {
		s.logger.Warn("could not check an agent's floor_control.max_wait_ms against the chorus-asr server", "error", err)
		return "floor_control.max_wait_ms is kept unchecked: the chorus-asr server did not say what it publishes, " +
			"and a wait shorter than every point it publishes ends turns by its default", "", true
	}
	ending, err := policy.Resolve(session.EndOfTurnOf(chosen))
	if err != nil {
		return "", err.Error(), false
	}
	return guardWarning(ending), "", true
}

// guardWarning says, of a measured point that fails the dictation guard, what it was measured
// to do to callers reading out a number, an ID or a spelled name: the app asking for it is
// told rather than finding out on a call.
func guardWarning(ending audiofloor.Ending) string {
	point := ending.Point
	if point == nil || point.Measured.DictationGuard {
		return ""
	}
	m := point.Measured
	return fmt.Sprintf("floor_control.max_wait_ms chooses the %d ms point, which fails the dictation guard: it ended %d of %d slow counts before their last number, "+
		"and %s of real pauses inside a dictated number, ID or spelled name; an agent whose callers read those out wants a point that passes",
		point.MaxWaitMs, m.SlowCountsCut, m.SlowCounts, strconv.FormatFloat(100*m.DictationPausesCut, 'f', 1, 64)+"%")
}
