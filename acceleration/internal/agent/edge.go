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

// Attendance is somebody arriving in or leaving the call.
type Attendance struct {
	Participant stt.Participant
	// Joined is false when they left.
	Joined bool
}

// Playout is an edge that holds a queue of speech on its way to the participants.
//
// Separate from Edge rather than a fifth method on it, because only a transport that paces
// audio out has a queue at all. It matters at both ends of a reply. At the end of one, a
// voice provider streams an utterance far faster than it is spoken, so the provider saying it
// has finished means it sent the last chunk, not that the caller heard it, and leaving on
// that word cuts the agent off mid-word. When the caller takes the floor mid-reply,
// cancelling the provider only stops what it has not synthesised yet, and what it already
// sent goes on being heard until the queue is thrown away.
//
// An edge that cannot say leaves nothing to wait for and nothing to drop, which is what an
// in-process loopback is.
type Playout interface {
	// SpeechPending reports whether audio already published is still waiting to go out.
	SpeechPending() bool
	// DropSpeech throws away audio published but not heard yet, so an interruption stops the
	// agent within a frame rather than at the end of what is already queued.
	DropSpeech()
}

// Roster is an edge that can say who else is in the call.
//
// Separate from Edge rather than a fifth method on it, because it is only answerable by a
// transport with other people in it. It matters for a call the agent did not start: an agent
// answering a phone has to know the caller is there before it says hello, and until somebody
// speaks, audio says nothing about who is listening.
type Roster interface {
	// Attendance reports who comes and goes, starting with whoever was already there when
	// the edge joined. The channel closes when the edge leaves.
	Attendance() <-chan Attendance
}
