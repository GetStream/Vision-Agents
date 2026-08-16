package agent

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// InboundAudio is a chunk of one participant's speech, arriving from the edge.
type InboundAudio struct {
	Participant stt.Participant
	// Audio must be 16 kHz mono, which is what every speech-to-text provider accepts.
	// Whatever the transport carried is the edge's problem to convert.
	Audio audio.PcmData
}

// Edge is where the conversation happens. It is deliberately four methods wide, so the
// agent's flow can be exercised in-process against a loopback rather than only against a
// real call.
//
// Everything transport-specific -- credentials, codecs, tracks, subscriptions -- lives
// behind this, which is why the agent knows nothing about WebRTC.
type Edge interface {
	Join(ctx context.Context) error
	// Audio carries what the participants said. The channel closes when the edge leaves.
	Audio() <-chan InboundAudio
	// PublishAudio sends a chunk of the agent's speech to the participants. It is called
	// on the text-to-speech provider's goroutine as audio arrives, so it must not block for
	// longer than the chunk it was given represents.
	PublishAudio(pcm audio.PcmData) error
	Leave() error
}
