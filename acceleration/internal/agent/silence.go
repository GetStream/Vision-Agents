package agent

import (
	"maps"
	"slices"
	"sync/atomic"
	"time"
)

// Silence is a way a voice call can end with audio missing in one direction, from what the
// agent itself saw of the call. A session is in no case unless somebody besides the agent
// was in its call: one nobody joined is a session that never started, not a silent one.
//
// The inbound cases are one of three, the first that holds:
//
//   - SilenceNoInbound: no audio frame arrived from anybody in the call: a media path that
//     carried nothing, or, on the web, a caller who joined with the microphone off and so
//     published no audio track. The two look the same here. A phone line that sends its
//     quiet as frames puts a quiet caller in the next case instead.
//   - SilenceSilentInbound: frames arrived, but not one reached audio.AudibleSample, about
//     -48 dBFS. A line that carries only digital silence and a caller who never made a sound
//     on a very quiet line look the same.
//   - SilenceNoWords: audible frames arrived, but the transcriber wrote no word from them:
//     a transcriber that heard nothing, or a caller who only made noise. Not judged once a
//     speech-to-speech model heard the caller, since it writes no transcript here.
//
// The outbound case stands beside them, so a call with no audio either way is in two:
//
//   - SilenceNoAudibleOutbound: a reply's speech was queued at an edge that says when its
//     track takes it (MarkedPlayout), and the track took no frame of it that was not silence: a
//     track the call never pulled from, or a voice that sent only silence. Speech the caller
//     interrupted before any of it played, with nothing said after, is counted here too.
//     An edge that cannot say is not judged. That the track took a frame does not prove the
//     caller heard it: the router cannot see the far end of the call.
type Silence string

const (
	SilenceNoInbound         Silence = "no_inbound_audio"
	SilenceSilentInbound     Silence = "silent_inbound_audio"
	SilenceNoWords           Silence = "no_words"
	SilenceNoAudibleOutbound Silence = "no_audible_outbound"
)

// Silences is every case, in a fixed order.
var Silences = []Silence{SilenceNoInbound, SilenceSilentInbound, SilenceNoWords, SilenceNoAudibleOutbound}

// CallAudio is what an agent's call carried each way, the signals a Silence is judged on.
type CallAudio struct {
	// Attended is somebody besides the agent having been in the call, by the edge's Roster.
	Attended bool
	// Frames counts the participants' audio frames, and Audible is any of them reaching
	// audio.AudibleSample.
	Frames  int64
	Audible bool
	// Words counts transcripts with words in them, partials and finals.
	Words int64
	// Native is some of the participants' audio having gone to a speech-to-speech model.
	Native bool
	// Published is a reply's speech queued at an edge that marks its playout, and Pulled its
	// track having taken an audible frame of it.
	Published bool
	Pulled    bool
}

// Merge is what two runs of the same session carried between them, as one call.
func (c CallAudio) Merge(other CallAudio) CallAudio {
	return CallAudio{
		Attended:  c.Attended || other.Attended,
		Frames:    c.Frames + other.Frames,
		Audible:   c.Audible || other.Audible,
		Words:     c.Words + other.Words,
		Native:    c.Native || other.Native,
		Published: c.Published || other.Published,
		Pulled:    c.Pulled || other.Pulled,
	}
}

// Silences is the cases the call is in, nil for none.
func (c CallAudio) Silences() []Silence {
	// An edge without a Roster says nobody arrived, and audio from somebody says they did.
	if !c.Attended && c.Frames == 0 {
		return nil
	}
	var cases []Silence
	switch {
	case c.Frames == 0:
		cases = append(cases, SilenceNoInbound)
	case !c.Audible:
		cases = append(cases, SilenceSilentInbound)
	case c.Words == 0 && !c.Native:
		cases = append(cases, SilenceNoWords)
	}
	if c.Published && !c.Pulled {
		cases = append(cases, SilenceNoAudibleOutbound)
	}
	return cases
}

// CallAudio is what the agent's call has carried so far.
func (a *Agent) CallAudio() CallAudio {
	a.mu.Lock()
	tallies := slices.Collect(maps.Values(a.hearings))
	a.mu.Unlock()
	carried := CallAudio{
		Attended:  a.carried.attended.Load(),
		Native:    a.carried.native.Load(),
		Published: a.carried.published.Load(),
		Pulled:    a.carried.pulled.Load(),
	}
	for _, tally := range tallies {
		carried.Frames += tally.frames.Load()
		carried.Audible = carried.Audible || tally.audible.Load()
		carried.Words += tally.partials.Load() + tally.finals.Load()
	}
	return carried
}

// carried is the part of CallAudio that is the agent's rather than one participant's. The
// edge's, the transcribers' and the voice's goroutines write it, so each is atomic.
type carried struct {
	attended  atomic.Bool
	native    atomic.Bool
	published atomic.Bool
	pulled    atomic.Bool
}

// FirstFrameQueued and FirstAudiblePulled run on the publishing and the track's goroutines,
// so they only store.
func (c *carried) FirstFrameQueued(time.Time)   { c.published.Store(true) }
func (c *carried) FirstAudiblePulled(time.Time) { c.pulled.Store(true) }

// marked adds the call's marks to a reply's. Speech the reply asks no report for is left
// unmarked, so the edge is asked about the same speech as without this.
func (c *carried) marked(marks PlayoutMarks) PlayoutMarks {
	if marks == nil {
		return nil
	}
	return bothMarks{marks, c}
}
