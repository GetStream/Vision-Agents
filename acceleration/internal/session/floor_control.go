package session

import (
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// FloorDefault is the router's default floor control: what an agent that chose none gets
// (Model, config.Agent.FloorControlDefault), and the languages it is for (Languages,
// config.Agent.FloorControlDefaultLanguages), BCP 47 tags matched by primary subtag. None
// listed is every language.
type FloorDefault struct {
	Model     string
	Languages []string
}

// Covers reports whether the default is for an agent whose callers speak languages, a
// session's language hints: every one's primary subtag is listed, or nothing is. An agent
// naming no language counts as English, which the router's transcribers and chorus-asr fall
// back to for one, so it is covered when en is listed and not otherwise.
func (d FloorDefault) Covers(languages []string) bool {
	if len(d.Languages) == 0 {
		return true
	}
	if len(languages) == 0 {
		languages = []string{"en"}
	}
	for _, spoken := range languages {
		if !slices.ContainsFunc(d.Languages, func(listed string) bool { return primarySubtag(listed) == primarySubtag(spoken) }) {
			return false
		}
	}
	return true
}

// primarySubtag is a BCP 47 tag's language alone, lowercase: en for en-US and en_GB.
func primarySubtag(tag string) string {
	tag = strings.ToLower(strings.TrimSpace(tag))
	if cut := strings.IndexAny(tag, "-_"); cut >= 0 {
		return tag[:cut]
	}
	return tag
}

// FloorModeOf is the mode a stored floor control asks chorus-asr in: stream/chorus-asr acts
// on its rulings, and anything else, the flow controller or none chosen, asks it nothing.
func FloorModeOf(chosen store.FloorControl) audiofloor.Mode {
	if chosen.Model == store.FloorControlChorusASR {
		return audiofloor.Act
	}
	return audiofloor.Off
}

// FloorControlOf is what controls a session's floor: the choice its agent made, or, where it
// made none, the router's default, for an agent whose callers speak languages. An agent's own
// choice wins whatever the default is: stream/chorus-asr, with its own max_wait_ms, is kept as
// chosen; flow-controller keeps the flow controller, and is the zero value, which asks nothing.
// None chosen takes the default where it covers the agent's languages (FloorDefault.Covers):
// stream/chorus-asr at the server's default point on a router that has a chorus-asr server
// (served), and the flow controller on one that does not, where the default is warned of once
// at start and asks nothing of any session.
func FloorControlOf(chosen store.FloorControl, languages []string, d FloorDefault, served bool) store.FloorControl {
	switch {
	case chosen.Model == store.FloorControlFlowController:
		return store.FloorControl{}
	case !chosen.IsZero():
		return chosen
	case d.Model == store.FloorControlChorusASR && served && d.Covers(languages):
		return store.FloorControl{Model: store.FloorControlChorusASR}
	}
	return store.FloorControl{}
}

// floorControl is what controls a session's floor on this router: FloorControlOf the spec's
// choice and languages, the manager's default and whether it has a server.
func (m *Manager) floorControl(spec Spec) store.FloorControl {
	return FloorControlOf(spec.FloorControl, spec.LanguageHints, m.options.FloorDefault, m.options.AudioFloor != nil)
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
