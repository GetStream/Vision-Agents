package api

import (
	"context"
	"fmt"
	"reflect"
	"strconv"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// EndOfTurn is an agent's choice of how eager chorus-asr's turn-taking is.
type EndOfTurn struct {
	MaxWaitMs *int            `json:"max_wait_ms,omitempty" minimum:"1" example:"900" doc:"The most a caller who has finished may wait for the answer, in milliseconds. It chooses the point of chorus-asr's measured trade-off with the fewest cut-ins whose measured p90 wait is within it. Refused when it is shorter than every point's."`
	Preset    *string         `json:"preset,omitempty" minLength:"1" example:"safe" doc:"One of the presets the chorus-asr server publishes, by name, such as safe or fast, each naming one of its measured points. Refused when the server publishes none by that name."`
	Bars      *[]EndOfTurnBar `json:"bars,omitempty" minItems:"1" maxItems:"16" doc:"Bars of the agent's own: the first from 0 ms, each later one from further into the pause. The turn has ended at the first check of a pause where p_eot reaches the last bar to have begun. Unlike a measured point, nobody has measured what they do to callers."`
}

func (*EndOfTurn) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "When chorus-asr has a caller's turn ended, where the agent's floor_control asks it about " +
		"turns: one of max_wait_ms, preset and bars. An empty object is the server's default preset. " +
		"A preset or a wait chooses one of the server's measured points: its bars, and how many checks " +
		"in a row of one pause must reach them. When a pause is checked, how much more a turn's first " +
		"pause asks, and the pause that ends a turn whatever the model says stay the server's. What a " +
		"preset or a wait means is the policy of the model the server runs, read when a pause is " +
		"checked; one that model no longer publishes ends turns by its default."
	schema.AdditionalProperties = false
	return schema
}

// EndOfTurnBar is one of an agent's own bars.
type EndOfTurnBar struct {
	FromMs int     `json:"from_ms" minimum:"0" example:"0" doc:"How far into the pause, in milliseconds, the bar begins."`
	Bar    float64 `json:"bar" example:"0.9" doc:"What p_eot, chorus-asr's probability that the caller has finished, must reach for the turn to have ended: strictly between 0 and 1."`
}

func (*EndOfTurnBar) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What chorus-asr's p_eot must reach for a turn to have ended, at a check " +
		"from from_ms into the pause on."
	schema.AdditionalProperties = false
	return schema
}

// endOfTurnOr is what a caller sent in place of stored: left out keeps it, and sent replaces
// it whole, since a choice is one of three and a wait kept beside a preset sent would be two.
// An empty object is the server's default.
func endOfTurnOr(stored store.EndOfTurn, sent *EndOfTurn) store.EndOfTurn {
	if sent == nil {
		return stored
	}
	chosen := store.EndOfTurn{MaxWaitMs: value(sent.MaxWaitMs), Preset: value(sent.Preset)}
	if sent.Bars != nil {
		chosen.Bars = make([]store.EndOfTurnBar, 0, len(*sent.Bars))
		for _, bar := range *sent.Bars {
			chosen.Bars = append(chosen.Bars, store.EndOfTurnBar{FromMs: bar.FromMs, Bar: bar.Bar})
		}
	}
	return chosen
}

// renderedEndOfTurn renders a stored choice for the wire, and nothing for the default.
func renderedEndOfTurn(stored store.EndOfTurn) *EndOfTurn {
	if stored.IsZero() {
		return nil
	}
	rendered := &EndOfTurn{}
	if stored.MaxWaitMs != 0 {
		rendered.MaxWaitMs = &stored.MaxWaitMs
	}
	if stored.Preset != "" {
		rendered.Preset = &stored.Preset
	}
	if stored.Bars != nil {
		bars := make([]EndOfTurnBar, 0, len(stored.Bars))
		for _, bar := range stored.Bars {
			bars = append(bars, EndOfTurnBar{FromMs: bar.FromMs, Bar: bar.Bar})
		}
		rendered.Bars = &bars
	}
	return rendered
}

// endOfTurnWait bounds asking the chorus-asr server what it publishes while an agent is
// written, so a server that is down does not hold the write.
const endOfTurnWait = 2 * time.Second

// endOfTurnComplaint reports what is wrong with an agent's end_of_turn, if anything, and what
// a sync should be told about one it keeps that may not do what it says. chosen is the agent's
// floor control and end_of_turn after the write, and was before it.
//
// What the choice says is always checked: one of max_wait_ms, preset and bars, and bars that
// begin at 0 ms and rise, each a probability strictly between 0 and 1. What it names is
// checked against the policy the chorus-asr server publishes when the agent's floor control
// asks chorus-asr about its turns, in shadow or acting, so a preset or wait the server doesn't
// have is refused while the agent is written: when it changed, or when the floor control before
// the write asked chorus-asr nothing, as nothing checked it then. A choice the server could not
// be asked about is kept, with a warning: the agent is not refused for a server that is down,
// and a choice that turns out to name nothing the server publishes ends turns by its default.
// So is one that chooses a point failing the dictation guard, with what it was measured to do.
func (s *Server) endOfTurnComplaint(ctx context.Context, chosen, was floorChoice) (warning, message string, ok bool) {
	choice := session.EndOfTurnOf(chosen.endOfTurn)
	if err := choice.Check(); err != nil {
		return "", err.Error(), false
	}
	if chosen.endOfTurn.IsZero() {
		return "", "", true
	}
	if s.audioFloor == nil {
		return "end_of_turn is kept, and does nothing on this router, which has no chorus-asr server", "", true
	}
	if !session.FloorModeOf(chosen.control).Asks() {
		return "end_of_turn is kept, and does nothing unless floor_control is stream/chorus-asr: chorus-asr is asked nothing about the agent's turns", "", true
	}
	if reflect.DeepEqual(chosen.endOfTurn, was.endOfTurn) && session.FloorModeOf(was.control).Asks() {
		return "", "", true
	}
	// As the server publishes it now, rather than as the sessions last read it: the agent is
	// written once, and a session reads the policy again as it goes.
	asking, cancel := context.WithTimeout(ctx, endOfTurnWait)
	defer cancel()
	policy, err := s.audioFloor.PolicyNow(asking)
	if err != nil {
		s.logger.Warn("could not check an agent's end_of_turn against the chorus-asr server", "error", err)
		return "end_of_turn is kept unchecked: the chorus-asr server did not say what it publishes, " +
			"and a preset or wait it does not publish ends turns by its default", "", true
	}
	ending, err := policy.Resolve(choice)
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
	return fmt.Sprintf("end_of_turn chooses the %d ms point, which fails the dictation guard: it ended %d of %d slow counts before their last number, "+
		"and %s of real pauses inside a dictated number, ID or spelled name; an agent whose callers read those out wants a point that passes",
		point.MaxWaitMs, m.SlowCountsCut, m.SlowCounts, strconv.FormatFloat(100*m.DictationPausesCut, 'f', 1, 64)+"%")
}
