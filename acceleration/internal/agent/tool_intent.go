package agent

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// wantedTool is the tool chorus-asr heard a caller ask for in a turn it ended with the floor
// free: the model's probability of it, of the turn having ended, and the language the server
// heard the turn's words in, empty when it did not say.
type wantedTool struct {
	tool     string
	p        float64
	pEOT     float64
	language string
}

var toolIntents = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "chorus_asr_tool_intents_total",
	Help:      "Turns answered on a chorus-asr ruling that heard the caller ask for one of the agent's tools.",
}, []string{})

// heardIntent reports the tool chorus-asr heard the caller ask for in the turn ready is,
// which is about to be answered.
func (a *Agent) heardIntent(ready candidate) {
	wanted, ok := a.audioFloor.wantedFor(ready.ID)
	if !ok {
		return
	}
	toolIntents.WithLabelValues().Inc()
	a.logger.Info("chorus-asr heard the caller ask for a tool", "turn", ready.ID, "participant", ready.Participant.ID,
		"tool", wanted.tool, "p", wanted.p, "p_eot", wanted.pEOT, "language", wanted.language)
}
