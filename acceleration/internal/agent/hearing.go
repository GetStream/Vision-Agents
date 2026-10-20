package agent

import (
	"maps"
	"slices"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// hearing counts what reached the agent from one participant: their audio from the call,
// and what the transcriber made of it. A caller who went quiet, audio that stopped
// arriving and a transcriber that stopped answering all look the same to the
// conversation, and these tell them apart. The edge's consumer writes the audio and the
// participant's transcription consumer the words, while others read, so each is atomic.
type hearing struct {
	frames  atomic.Int64
	audioUs atomic.Int64
	// partials and finals count transcripts with words in them.
	partials atomic.Int64
	finals   atomic.Int64
	// lastFrame and lastWords are Unix nanoseconds, zero until the first.
	lastFrame atomic.Int64
	lastWords atomic.Int64
}

// heardAudio counts one chunk of audio. It runs once per frame, so it only counts.
func (h *hearing) heardAudio(pcm audio.PcmData) {
	inboundFrames.Inc()
	h.frames.Add(1)
	h.audioUs.Add(int64(pcm.DurationMs() * 1000))
	h.lastFrame.Store(time.Now().UnixNano())
}

// heardWords counts one transcript that has words in it, on the router's counters for the
// transcriber that wrote it as well.
func (h *hearing) heardWords(transcript stt.Transcript, partials, finals prometheus.Counter) {
	if transcript.Final() {
		finals.Inc()
		h.finals.Add(1)
	} else {
		partials.Inc()
		h.partials.Add(1)
	}
	h.lastWords.Store(time.Now().UnixNano())
}

// heardFrom is the tally for a participant, started on first asking.
func (a *Agent) heardFrom(participantID string) *hearing {
	a.mu.Lock()
	defer a.mu.Unlock()
	tally, ok := a.hearings[participantID]
	if !ok {
		tally = &hearing{}
		a.hearings[participantID] = tally
	}
	return tally
}

// logHeard reports what has been heard from a participant up to moment, so two lines
// show whether the router heard them in between. A participant never heard is reported
// too: that is the case this is for.
func (a *Agent) logHeard(moment, participantID string, extra ...any) {
	a.mu.Lock()
	tally := a.hearings[participantID]
	a.mu.Unlock()
	if tally == nil {
		tally = &hearing{}
	}
	a.logger.Info("what the agent heard from a participant", append([]any{
		"moment", moment, "participant", participantID,
		"frames", tally.frames.Load(),
		"audio_ms", tally.audioUs.Load() / 1000,
		"partials", tally.partials.Load(),
		"finals", tally.finals.Load(),
		"since_frame_ms", sinceMs(tally.lastFrame.Load()),
		"since_words_ms", sinceMs(tally.lastWords.Load()),
	}, extra...)...)
}

// logHeardFromEveryone reports every participant heard from, in a stable order.
func (a *Agent) logHeardFromEveryone(moment string) {
	a.mu.Lock()
	participants := slices.Sorted(maps.Keys(a.hearings))
	a.mu.Unlock()
	for _, participant := range participants {
		a.logHeard(moment, participant)
	}
}

// sinceMs is how long ago a Unix-nanosecond time was, or -1 for never.
func sinceMs(unixNano int64) int64 {
	if unixNano == 0 {
		return -1
	}
	return time.Since(time.Unix(0, unixNano)).Milliseconds()
}
