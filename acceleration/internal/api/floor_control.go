package api

import (
	"context"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// FloorControl is an agent's choice of whether chorus-asr rules its floor.
type FloorControl struct {
	Model FloorControlModel `json:"model" doc:"The model that hears the caller's audio and rules the floor."`
	Mode  FloorControlMode  `json:"mode" doc:"What the agent does with the model's rulings."`
}

func (*FloorControl) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Whether a model that hears the caller's audio rules the agent's floor: when " +
		"a caller has finished, and what to do about speech over the agent. An agent opts in for " +
		"itself, and one that never chose is off. shadow and act need the router's chorus-asr server, " +
		"and a write asking for either on a router without one is refused. Off, the flow controller " +
		"rules the floor as it does with no such model, and end_of_turn does nothing; chorus-asr still " +
		"transcribes for an agent whose stt names stream/chorus-asr. Each session runs by its own, so " +
		"agents on one router may choose differently."
	schema.AdditionalProperties = false
	return schema
}

// FloorControlModel is a model that can rule an agent's floor.
type FloorControlModel string

const floorModelChorusASR FloorControlModel = store.FloorModelChorusASR

func (FloorControlModel) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "FloorControlModel", "A model that hears the caller's audio and rules an "+
		"agent's floor. chorus-asr is the one there is.", string(floorModelChorusASR))
}

// FloorControlMode is what an agent does with its floor control model's rulings.
type FloorControlMode string

const (
	floorModeOff    FloorControlMode = store.FloorModeOff
	floorModeShadow FloorControlMode = store.FloorModeShadow
	floorModeAct    FloorControlMode = store.FloorModeAct
)

func (FloorControlMode) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "FloorControlMode", "What the agent does with the model's rulings. off asks "+
		"it nothing. shadow asks it about each turn beside the flow controller and acts on nothing it "+
		"says, to compare the two on real calls. act asks it first and acts on its ruling, and the flow "+
		"controller rules a turn it does not answer in time.",
		string(floorModeOff), string(floorModeShadow), string(floorModeAct))
}

// errNoChorusASR refuses shadow or act on a router with no chorus-asr server to ask.
var errNoChorusASR = notConfigured("floor_control: chorus-asr cannot rule the floor on this router, which has " +
	"no chorus-asr server (ROUTER_CHORUS_ASR_URL); mode off needs none")

// floorControlOr is what a caller sent in place of stored: left out keeps it, and sent replaces
// it.
func floorControlOr(stored store.FloorControl, sent *FloorControl) store.FloorControl {
	if sent == nil {
		return stored
	}
	return store.FloorControl{Model: string(sent.Model), Mode: string(sent.Mode)}
}

// renderedFloorControl renders a stored choice for the wire, and nothing for an agent that
// never chose, which is off.
func renderedFloorControl(stored store.FloorControl) *FloorControl {
	if stored.IsZero() {
		return nil
	}
	return &FloorControl{Model: FloorControlModel(stored.Model), Mode: FloorControlMode(stored.Mode)}
}

// floorControlRefusal is why a floor control sent is refused, if it is: shadow or act on a
// router without a chorus-asr server. Only what is sent is held to the router: a stored choice
// a write leaves out stays as it is, and its sessions ask nothing of a router without a server.
func (s *Server) floorControlRefusal(sent *FloorControl) (APIError, bool) {
	if sent != nil && session.FloorModeOf(floorControlOr(store.FloorControl{}, sent)).Asks() && s.audioFloor == nil {
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
