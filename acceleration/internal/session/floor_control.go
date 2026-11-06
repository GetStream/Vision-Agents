package session

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// FloorModeOf is the mode a stored floor control asks chorus-asr in: none chosen is off, as
// an agent opts in for itself. chorus-asr is the one model there is.
func FloorModeOf(chosen store.FloorControl) audiofloor.Mode {
	if chosen.IsZero() {
		return audiofloor.Off
	}
	return audiofloor.Mode(chosen.Mode)
}

// floorMode is the mode a session's agent asks chorus-asr in, as its floor control says. One
// that asks on a router with no chorus-asr server asks nothing, and the flow controller rules
// its floor alone, as with floor control off.
func (m *Manager) floorMode(id string, spec Spec) audiofloor.Mode {
	mode := FloorModeOf(spec.FloorControl)
	if mode.Asks() && m.options.AudioFloor == nil {
		m.logger.Warn("the agent's floor control asks chorus-asr, and this router has no chorus-asr server: "+
			"the flow controller rules its floor alone", "session", id, "floor_control", string(mode))
		return audiofloor.Off
	}
	return mode
}
