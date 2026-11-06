package session

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// FloorModeOf is the mode a stored floor control asks chorus-asr in: stream/chorus-asr acts
// on its rulings, and none chosen asks it nothing, as an agent opts in for itself.
func FloorModeOf(chosen store.FloorControl) audiofloor.Mode {
	if chosen == store.FloorControlChorusASR {
		return audiofloor.Act
	}
	return audiofloor.Off
}

// floorMode is the mode a session's agent asks chorus-asr in, as its floor control says. One
// that chose stream/chorus-asr on a router with no chorus-asr server asks nothing, and the flow
// controller rules its floor alone, as with none chosen.
func (m *Manager) floorMode(id string, spec Spec) audiofloor.Mode {
	mode := FloorModeOf(spec.FloorControl)
	if mode.Asks() && m.options.AudioFloor == nil {
		m.logger.Warn("the agent's floor control is stream/chorus-asr, and this router has no chorus-asr server: "+
			"the flow controller rules its floor alone", "session", id, "floor_control", string(spec.FloorControl))
		return audiofloor.Off
	}
	return mode
}
