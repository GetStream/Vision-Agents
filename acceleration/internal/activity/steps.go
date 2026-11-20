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

// ReachedSteps are the steps of Steps that first has a day for, in the order of Steps, each with
// that day. The areas of first that are no step are left out.
func ReachedSteps(first []store.ActivityAreaDay) []ReachedStep {
	var reached []ReachedStep
	for _, step := range Steps {
		for _, area := range first {
			if area.Area == step {
				reached = append(reached, ReachedStep{Step: step, FirstAnswered: area.Day})
				break
			}
		}
	}
	return reached
}
