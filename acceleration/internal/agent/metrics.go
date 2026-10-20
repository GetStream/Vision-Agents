package agent

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	speechOverlaps = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "agent_speech_overlaps_total",
		Help:      "Pairs of one agent's utterances whose audio was published interleaved, heard as the agent talking over itself",
	}, []string{"provider"})
	speechOverlapSeconds = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "agent_speech_overlap_seconds_total",
		Help:      "Audio published while another utterance of the same agent was still being published",
	}, []string{"provider"})
)
