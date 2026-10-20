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
	openUtterances = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "agent_open_utterances",
		Help:      "Utterances of one agent with audio published and not finished, each time a chunk is published; above one is the agent talking over itself",
		// One is an agent speaking normally, two an overlap, three or more a pile-up.
		Buckets: []float64{1, 2, 3},
	}, []string{"provider"})
	// The tool name is left out: a caller names its own tools, so it is customer data
	// without a bound.
	toolCallSeconds = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "agent_tool_call_seconds",
		Help:      "How long a tool the model called took to come back, by how it ended",
		// 50 ms to 51.2 s, past the 30 s a caller's own tool is given (session.defaultToolTimeout).
		Buckets: prometheus.ExponentialBuckets(0.05, 2, 11),
	}, []string{"outcome"})
)
