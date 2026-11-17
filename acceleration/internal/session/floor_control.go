package session

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// FloorModeOf is the mode a stored floor control asks chorus-asr in: stream/chorus-asr acts
// on its rulings, and anything else, the flow controller or none chosen, asks it nothing.
func FloorModeOf(chosen store.FloorControl) audiofloor.Mode {
	if chosen.Model == store.FloorControlChorusASR {
		return audiofloor.Act
	}
	return audiofloor.Off
}

// FloorControlOf is what controls a session's floor: the choice its agent made, or, where it
// made none, the router's default, as config.Agent.FloorControlDefault names it. An agent's own
// choice wins whatever the default is: stream/chorus-asr, with its own max_wait_ms, is kept as
// chosen; flow-controller keeps the flow controller, and is the zero value, which asks nothing.
// None chosen takes the default: stream/chorus-asr at the server's default point on a router
// that has a chorus-asr server (served), and the flow controller on one that does not, where
// the default is warned of once at start and asks nothing of any session.
func FloorControlOf(chosen store.FloorControl, routerDefault string, served bool) store.FloorControl {
	switch {
	case chosen.Model == store.FloorControlFlowController:
		return store.FloorControl{}
	case !chosen.IsZero():
		return chosen
	case routerDefault == store.FloorControlChorusASR && served:
		return store.FloorControl{Model: store.FloorControlChorusASR}
	}
	return store.FloorControl{}
}

// floorControl is what controls a session's floor on this router: FloorControlOf the spec's
// choice, the manager's default and whether it has a server.
func (m *Manager) floorControl(spec Spec) store.FloorControl {
	return FloorControlOf(spec.FloorControl, m.options.FloorControlDefault, m.options.AudioFloor != nil)
}

// floorMode is the mode a session's agent asks chorus-asr in, as what controls its floor says.
// One that chose stream/chorus-asr on a router with no chorus-asr server asks nothing, and the
// flow controller rules its floor alone, as with none chosen.
func (m *Manager) floorMode(id string, ruling store.FloorControl) audiofloor.Mode {
	mode := FloorModeOf(ruling)
	if mode.Asks() && m.options.AudioFloor == nil {
		m.logger.Warn("the agent's floor control is stream/chorus-asr, and this router has no chorus-asr server: "+
			"the flow controller rules its floor alone", "session", id, "floor_control", ruling.Model)
		return audiofloor.Off
	}
	return mode
}

// EndOfTurnOf is when a stored floor control has chorus-asr's Hearer end a turn: the point
// its max_wait_ms chooses, or the server's default.
func EndOfTurnOf(chosen store.FloorControl) audiofloor.EndOfTurn {
	return audiofloor.EndOfTurn{MaxWaitMs: chosen.MaxWaitMs}
}
