package session

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// EndOfTurnOf is a stored end-of-turn choice as chorus-asr's Hearer reads it.
func EndOfTurnOf(stored store.EndOfTurn) audiofloor.EndOfTurn {
	choice := audiofloor.EndOfTurn{MaxWaitMs: stored.MaxWaitMs, Preset: stored.Preset}
	if stored.Bars != nil {
		choice.Bars = make([]audiofloor.PolicyBar, 0, len(stored.Bars))
		for _, bar := range stored.Bars {
			choice.Bars = append(choice.Bars, audiofloor.PolicyBar{FromMs: bar.FromMs, Bar: bar.Bar})
		}
	}
	return choice
}
