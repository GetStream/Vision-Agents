package activity

import (
	"slices"
	"time"

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

// ReachedStep is a step an organization reached, with the UTC day of its first request
// answered in the step's area.
type ReachedStep struct {
	Step          string
	FirstAnswered time.Time
}

// ReachedSteps are the steps of Steps that areas reached, in the order of Steps, each with the
// earliest day any client of its area had a request answered. As in FurthestStep, an area
// whose every request was refused was not reached.
func ReachedSteps(areas []store.ActivityArea) []ReachedStep {
	firsts := make(map[string]time.Time)
	for _, area := range areas {
		if area.FirstAnswered.IsZero() {
			continue
		}
		if held, found := firsts[area.Area]; !found || area.FirstAnswered.Before(held) {
			firsts[area.Area] = area.FirstAnswered
		}
	}
	var reached []ReachedStep
	for _, step := range Steps {
		if day, found := firsts[step]; found {
			reached = append(reached, ReachedStep{Step: step, FirstAnswered: day})
		}
	}
	return reached
}
