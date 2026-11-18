package activity

import (
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Steps are the areas that show how far an organization got with Agents, the furthest last:
// it reads the dashboard, sets an agent up, runs a worker, then holds conversations, by voice
// or video, then on a phone line or SMS. The other areas are used along the way and are no step.
var Steps = []string{AreaDashboardReads, AreaAgentSetup, AreaWorkers, AreaConversations, AreaVoiceVideo, AreaPhoneSMS}

// FurthestStep is the furthest of Steps that areas reached, or "" when they reached none. An
// area counts only with a request answered: one whose every request was refused was tried,
// not reached.
func FurthestStep(areas []store.ActivityArea) string {
	furthest := -1
	for _, area := range areas {
		if area.Requests > area.Refused {
			furthest = max(furthest, slices.Index(Steps, area.Area))
		}
	}
	if furthest < 0 {
		return ""
	}
	return Steps[furthest]
}
