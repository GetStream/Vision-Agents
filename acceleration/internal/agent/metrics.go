package agent

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// turnLatencyBuckets run 200 ms to 10 s: steps of 200 ms up to 1.2 s, the line past which
// Twilio counts a turn as slow (https://www.twilio.com/docs/voice/voice-insights/conversation-relay-insights-dashboard),
// then wide ones for a turn that waited on a slow model. Each bucket is a series per leg and
// pipeline.
var turnLatencyBuckets = []float64{0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.6, 2, 3, 5, 10}

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
	playoutDroppedFrames = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "agent_playout_dropped_frames_total",
		Help:      "Frames of the agent's speech thrown away from the edge's outgoing queue before the call took them, on an interruption",
	}, []string{"provider"})
	playoutQueuePeak = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "agent_playout_queue_peak_seconds",
		Help:      "The most speech the edge's outgoing queue held at once while an utterance was open",
		// 100 ms to 6.4 s. The queue lets a writer go at 400 ms (streamedge.playoutFrames), and
		// a chunk is queued whole on top of that, so a long one is seen as a deep queue.
		Buckets: prometheus.ExponentialBuckets(0.1, 2, 7),
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
	// Unlabelled: the edge knows whose audio it is, which is unbounded, and not which
	// transcriber hears it.
	inboundFrames = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "agent_inbound_audio_frames_total",
		Help:      "Chunks of participants' audio the agent received from the call",
	})
	sttPartials = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "agent_stt_partials_total",
		Help:      "Transcripts with words in them that the speaker may still revise",
	}, []string{"provider"})
	sttFinals = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "agent_stt_finals_total",
		Help:      "Transcripts with words in them that the transcriber settled",
	}, []string{"provider"})
	staleReplies = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "agent_stale_replies_dropped_total",
		Help:      "Replies to a caller's turn dropped before any of their audio went out, because the same caller said more",
	})
	staleQueuedTurns = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "agent_stale_queued_turns_dropped_total",
		Help:      "Caller turns queued behind a reply and dropped, their words folded into the answer to a newer turn of the same caller",
	})
	// kind is the turn kind (turns.go), a fixed set.
	talkedOver = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "agent_replies_started_over_caller_total",
		Help:      "Replies whose first audio went out while a caller was mid-utterance, by what started the reply",
	}, []string{"kind"})
	playedBeforeInterrupt = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "agent_played_before_interrupt_seconds",
		Help:      "How long a reply had been going out, from its first published audio, when an interruption took the floor",
		// 100 ms to 12.8 s: barge-in is ruled on within a second or two of a reply starting.
		Buckets: prometheus.ExponentialBuckets(0.1, 2, 8),
	}, []string{"kind"})
	// leg and pipeline are fixed sets (observeTurnLatency).
	turnLatencySeconds = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "agent_turn_latency_seconds",
		Help:      "How long a caller waited for the reply to what they said, by leg: roundtrip from the last transcript revision to the first audio published (Turn.RoundtripMs), speech_end_to_audio that plus the transcriber's time deciding they had stopped (SpeechEndToAudioMs), speech_end_to_audible the same to the first frame of the reply the call's track took (SpeechEndToAudibleMs, only from an edge that reports it); by pipeline, cascade or native for one speech-to-speech model, the one the turn started on; under native, roundtrip runs from the model's speech-stopped signal and speech_end_to_audio equals roundtrip. A caller turn whose reply said nothing and handed work on, to a tool or the subagent, is timed to the first audio of the first line said for it: a hold or update line while the work runs, or what the work came back with; a filler in its own reply ends the wait before that, so each caller turn is observed once, and speech_end_to_audible counts only the turn's own reply. Replies the agent starts on its own, such as a greeting, and turns a scheduled task or a reminder fires are left out, and so is a leg a turn never reached, such as a reply interrupted before any of it went out or a turn the caller spoke again after, or hung up on, before a line said for it",
		Buckets:   turnLatencyBuckets,
	}, []string{"leg", "pipeline"})
	// cause, ruler and echo are fixed sets (interruption.go, harness.Ruler).
	interruptions = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "agent_interruptions_total",
		Help:      "Replies abandoned, by why, by whose ruling took the floor, and by how much of the words that took it the reply had said itself",
	}, []string{"cause", "ruler", "echo"})
	flowDecisions = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "agent_flow_decisions_total",
		Help:      "Rulings of the flow controller, by disposition and by whose answer was used",
	}, []string{"disposition", "ruler"})
	flowDecisionSeconds = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "agent_flow_decision_seconds",
		Help:      "How long the flow controller took to rule, by whose answer was used",
		// 50 ms to 12.8 s, past the controller's own deadlines (harness.FlowDeadline).
		Buckets: prometheus.ExponentialBuckets(0.05, 2, 9),
	}, []string{"ruler"})
	settledOverAgent = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "agent_settled_over_agent_total",
		Help:      "Settled turns put to the flow controller while the agent was talking, by how many words, whether the last revision of the words was a final, and what came of them",
	}, []string{"words", "final", "action"})
)
