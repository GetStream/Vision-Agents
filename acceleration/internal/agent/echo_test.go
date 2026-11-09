package agent

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const echoReply = "Tomorrow in Lisbon it will be sunny and warm, with a high of twenty four degrees."

// The agent's words in a transcript are its echo, as the model heard them, before, after
// and around the caller's own: what is left is the caller's, and an echo alone leaves
// nothing. A short phrase of the caller's inside a long run of the echo, "and in Porto"
// over the reply, is theirs, not a quarter of the run misheard.
func TestTheAgentsOwnWordsAreItsEcho(t *testing.T) {
	for _, c := range []struct {
		heard, left string
		echo        int
	}{
		{"Will it rain in Porto this weekend? Tomorrow in Lisbon it will be sunny and warm.", "Will it rain in Porto this weekend?", 9},
		{"Tomorrow in Lisbon, it will be sunny and warm, with a high of.", "", 13},
		{"with a high of 24 degrees.", "", 7},
		{"With a high of twenty-four degrees.", "", 7},
		{"Uh huh. sunny and warm", "Uh huh.", 3},
		{"Will it rain in Porto this weekend?", "Will it rain in Porto this weekend?", 0},
		{"and warm", "and warm", 0},
		{"Is it warm in Porto?", "Is it warm in Porto?", 0},
		{"Tomorrow in Lisbon it will be sunny and warm, and in Porto high of twenty-four degrees.", "and in Porto", 14},
		{"Tomorrow in Lisbon it will be sunny and warm, and in Porto.", "and in Porto.", 9},
		{"Tomorrow in Lisbon, it will be sunny and warm. No, no, stop, that's not what I asked.", "No, no, stop, that's not what I asked.", 9},
		{"Tomorrow in Lisbon, it will be sunny. No.", "No.", 7},
		// A stop over the reply: "Wait a" lines up with the line's "with a" and "minute" with
		// "high", two words that are not the agent's with one of its words between them, so
		// all three are the caller's.
		{"Tomorrow in Lisbon, it will be sunny and warm. Wait a minute.", "Wait a minute.", 9},
		// One word of the caller's between two runs of the echo reads as a word of the
		// agent's misheard: the checks over the agent see the words as they grow, so the
		// caller's word is the last one when it counts, and two or more are theirs anywhere.
		{"Tomorrow in Lisbon, it will be sunny. No. With a high of twenty four degrees.", "", 15},
	} {
		left, echo := echoed(c.heard, echoReply)
		require.Equal(t, c.left, left, "%q", c.heard)
		require.Equal(t, c.echo, echo, "%q", c.heard)
	}
}

// replayReply is the stop probe's: the replay's reply, about 9 s of the agent's voice, which
// a caller speaks over.
const replayReply = "Tomorrow in Lisbon it will be sunny and warm, with a high of twenty four degrees, " +
	"a light breeze from the north in the afternoon, and a clear and mild evening by the river."

// The caller's words over the echo are theirs, and the agent's words misheard come one at a
// time: two words that are not the agent's, in a row or with one of its words between them,
// end a run of the echo, so a short word of the caller's that is also in the agent's line
// does not carry the run over the words around it. A number the voice says and the model
// writes in digits is the same words. The stop probe's echo line wrote the first eight, the
// replay's S1b the ninth, and the last three are the echo misheard.
func TestTheCallersWordsOverTheEchoAreTheirs(t *testing.T) {
	for _, c := range []struct {
		heard, left string
		echo        int
	}{
		// "Oh, wait a minute" over the reply, as the checks over the agent saw it grow.
		{"Tomorrow in Lisbon, it will be sunny and warm. Wait a minute.", "Wait a minute.", 9},
		{"Tomorrow in Lisbon it will be sunny and warm. Wait a minute! A light breeze from the north in the afternoon.", "Wait a minute!", 18},
		{"Tomorrow in Lisbon it will be sunny and warm. Wait a minute! Twenty-four degrees. A light breeze from.", "Wait a minute!", 16},
		{"Tomorrow in Lisbon it will be sunny and warm. Wait a minute, 24 degrees.", "Wait a minute,", 12},
		// The echo alone, its number in digits.
		{"With a high of 24.", "", 6},
		{"Tomorrow in Lisbon it will be sunny and warm, with a high of 20.", "", 14},
		{"Tomorrow in Lisbon it will be sunny and warm, with a high of 24 degrees. The light breeze.", "", 19},
		// Two words misheard in a row are the caller's, as they were.
		{"Tomorrow in Lisbon, it will be sunny and warm, with a high of 24 degrees. Allow three.", "Allow three.", 16},
		// The end of a caller's question, with the agent's "it will be" in it, said as the
		// reply began (the replay's S1b): theirs, as the model wrote it.
		{"Will it be raining?", "Will it be raining?", 0},
		// A word misheard with only one of the agent's words before it where the transcript
		// begins, or after it where the words end so far, the next not yet written; and one
		// among the agent's words.
		{"And uh clear and mild evening by the river.", "", 9},
		{"Tomorrow in Lisbon it will be sunny and worm with", "", 10},
		{"Tomorrow in Lisbon it will be sunny and warm, with a hi of twenty four degrees, a light breeze.", "", 19},
	} {
		left, echo := echoed(c.heard, replayReply)
		require.Equal(t, c.left, left, "%q", c.heard)
		require.Equal(t, c.echo, echo, "%q", c.heard)
	}
}

// Only chorus-asr's words are weighed, and only those heard while the agent spoke or just
// after: a caller who says the agent's words long after it stopped said them.
func TestOnlyWordsHeardOverTheAgentAreItsEcho(t *testing.T) {
	a := &Agent{history: []llm.Message{{Role: llm.Assistant, Content: echoReply}}}
	heard := stt.Transcript{Provider: audiofloor.ProviderName, Text: "Tomorrow in Lisbon it will be sunny", AudioDurationMs: 1500, ProcessingTimeMs: 300}

	_, echo := a.withoutEcho(heard)
	require.Zero(t, echo, "the agent has said nothing on this call")

	a.voiceOut.from, a.voiceOut.to = time.Now().Add(-3*time.Second), time.Now().Add(-500*time.Millisecond)
	left, echo := a.withoutEcho(heard)
	require.Equal(t, 7, echo)
	require.Empty(t, left)

	other := heard
	other.Provider = "deepgram"
	_, echo = a.withoutEcho(other)
	require.Zero(t, echo, "another transcriber's words are left alone")
	other.Floor = true
	left, echo = a.withoutEcho(other)
	require.Equal(t, 7, echo, "another transcriber's words of an utterance chorus-asr's floor ended, timed by the floor, are weighed")
	require.Empty(t, left)

	a.voiceOut.from, a.voiceOut.to = time.Now().Add(-20*time.Second), time.Now().Add(-10*time.Second)
	_, echo = a.withoutEcho(heard)
	require.Zero(t, echo, "long after the agent stopped")
}

// healthcare.interrupt: the agent's line, and what the caller said once it had stopped.
const (
	movedLine    = "Thanks for clarifying. I won't change your son's appointment using your record. Is Friday your current appointment or the day you'd like to move it to?"
	keptAndMoved = "Keep Maya's Friday ten a.m. with Doctor Chen. Just move it to Tuesday morning."
)

// A caller who says the agent's words once it has stopped said them. In healthcare.interrupt
// the agent asked "…or the day you'd like to move it to?" and went quiet, and the caller
// began 1.24 s after its last audible sample reached them, so at least that long after the
// sample left the router: "…Just move it to Tuesday morning." Taken for the echo, "move it
// to" was dropped, and the reply kept Friday.
func TestACallerWhoRepeatsTheAgentOnceItStoppedIsHeardWhole(t *testing.T) {
	now := time.Now()
	// The final came 120 ms after its audio, which began, the transcriber's 300 ms ahead of
	// the voice included, 1,240 ms after the agent's last audible sample went out.
	heard := stt.Transcript{Provider: audiofloor.ProviderName, Mode: stt.ModeFinal, Text: keptAndMoved,
		AudioDurationMs: 4460, ProcessingTimeMs: 120}
	began := now.Add(-ms(120 + 4460))
	// The voice sent 8,980 ms, audible from 20 ms to 8,500 ms: 480 ms of silence after
	// the last word, as it was.
	a := &Agent{history: []llm.Message{{Role: llm.Assistant, Content: movedLine}}}
	publishLine(a, began.Add(-ms(1240+8500)), spokenLine(ms(8980), ms(20), ms(8500)), ms(400))

	left, echo := a.withoutEcho(heard)
	require.Equal(t, keptAndMoved, left)
	require.Zero(t, echo)
}

// The agent's words heard while its voice goes out, or within echoDelay of its last audible
// sample going out, are its echo: over the agent's question, after it with the line's delay
// (an utterance of the echo that begins as late as echoDelay allows), and around the
// caller's "Uh huh". Words that begin later are the caller's, however much they repeat it.
// What goes out is timed from the edge's pace, not from when the voice sent it.
func TestTheAgentsWordsAreItsEchoOnlyWhileItsVoiceCanBeHeardBack(t *testing.T) {
	for _, c := range []struct {
		name string
		// after is when the transcript's audio began, from the agent's last audible sample
		// going out, and lasts how long it lasts.
		after, lasts time.Duration
		heard, left  string
		echo         int
	}{
		{"over the question", -ms(2500), ms(2600), "Uh huh. the day you'd like to move it to", "Uh huh.", 8},
		{"the echo alone, as it ends", -ms(1800), ms(2200), "the day you'd like to move it to", "", 8},
		{"the echo, as late as it can begin", echoDelay - ms(20), ms(1500), "you'd like to move it to", "", 6},
		{"the caller, just after", echoDelay + ms(20), ms(1500), "you'd like to move it to", "you'd like to move it to", 0},
		{"the caller, 1.24 s after", ms(1240), ms(4460), keptAndMoved, keptAndMoved, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			const processing = 120
			began := time.Now().Add(-ms(processing) - c.lasts)
			a := &Agent{history: []llm.Message{{Role: llm.Assistant, Content: movedLine}}}
			publishLine(a, began.Add(-c.after-ms(8500)), spokenLine(ms(8980), ms(20), ms(8500)), ms(400))
			heard := stt.Transcript{Provider: audiofloor.ProviderName, Mode: stt.ModeFinal, Text: c.heard,
				AudioDurationMs: float64(c.lasts.Milliseconds()), ProcessingTimeMs: processing}
			left, echo := a.withoutEcho(heard)
			require.Equal(t, c.left, left)
			require.Equal(t, c.echo, echo)
		})
	}
}

// When the voice goes out is the edge's pace from when each chunk was queued, after what was
// queued ahead of it, and no sooner than its publish returned: a track that took it late
// returned late. The run spans its audible samples, not the voice's silences around them, and
// a gap of more than echoRun begins another.
func TestTheVoiceGoesOutAtTheEdgesPace(t *testing.T) {
	t0 := time.Now()
	second := func(rate int, from, to time.Duration) audio.PcmData {
		samples := make([]int16, rate)
		for i := int(from * time.Duration(rate) / time.Second); i < int(to*time.Duration(rate)/time.Second); i++ {
			samples[i] = -2000
		}
		return audio.PcmData{Samples: samples, SampleRate: rate, Channels: 1}
	}
	var v voiceOut
	v.published(t0, t0, second(24000, ms(100), ms(900)))
	require.Equal(t, t0.Add(ms(100)), v.from)
	require.Equal(t, t0.Add(ms(900)), v.to)
	require.Equal(t, t0.Add(time.Second), v.until)

	// Queued at once, it goes out after the second ahead of it.
	v.published(t0.Add(ms(10)), t0.Add(ms(10)), second(16000, 0, ms(500)))
	require.Equal(t, t0.Add(ms(100)), v.from, "the same run")
	require.Equal(t, t0.Add(ms(1500)), v.to)
	require.Equal(t, t0.Add(2*time.Second), v.until)

	// Silence goes out and is not heard.
	v.published(t0.Add(ms(20)), t0.Add(ms(20)), second(16000, 0, 0))
	require.Equal(t, t0.Add(ms(1500)), v.to)
	require.Equal(t, t0.Add(3*time.Second), v.until)

	// A track that took it late: the publish returned 2 s after the chunk could have gone out.
	v.published(t0.Add(ms(30)), t0.Add(6*time.Second), second(16000, ms(200), ms(600)))
	require.Equal(t, t0.Add(5*time.Second+ms(200)), v.from, "another run, 3.7 s after the last audible sample")
	require.Equal(t, t0.Add(5*time.Second+ms(600)), v.to)
	require.Equal(t, [2]time.Time{t0.Add(ms(100)), t0.Add(ms(1500))}, v.before)
}

// Speech thrown away at the edge never goes out, to be heard or heard back: the run ends at
// the drop, and a run none of which had gone out leaves the one before it as the last.
func TestDroppedSpeechIsNoEcho(t *testing.T) {
	t0 := time.Now().Add(-10 * time.Second)
	var v voiceOut
	v.published(t0, t0, audio.PcmData{Samples: loudFor(16000, ms(1000)), SampleRate: 16000, Channels: 1})
	v.published(t0.Add(ms(10)), t0.Add(ms(10)), audio.PcmData{Samples: loudFor(16000, ms(5000)), SampleRate: 16000, Channels: 1})
	v.dropped(t0.Add(ms(2000)))
	require.Equal(t, t0, v.from)
	require.Equal(t, t0.Add(ms(2000)), v.to)
	require.Equal(t, t0.Add(ms(2000)), v.until)

	// A reply queued behind 1.5 s of silence and dropped before any of it went out.
	v.published(t0.Add(ms(2100)), t0.Add(ms(2100)), audio.PcmData{Samples: make([]int16, 24000), SampleRate: 16000, Channels: 1})
	v.published(t0.Add(ms(2200)), t0.Add(ms(2200)), audio.PcmData{Samples: loudFor(16000, ms(1000)), SampleRate: 16000, Channels: 1})
	require.Equal(t, t0.Add(ms(3600)), v.from)
	v.dropped(t0.Add(ms(3000)))
	require.Equal(t, t0, v.from)
	require.Equal(t, t0.Add(ms(2000)), v.to)

	// The agent's dropSpeech is the edge's and this.
	edge := newLoopbackEdge()
	a := &Agent{options: Options{Edge: edge}, history: []llm.Message{{Role: llm.Assistant, Content: movedLine}}}
	a.voiceOut = v
	a.voiceOut.to = time.Now().Add(5 * time.Second)
	a.dropSpeech()
	require.False(t, a.voiceOut.to.After(time.Now()))

	// The caller cut the question off 2 s in, and repeats words of it they never heard 900 ms
	// later: theirs. Without the drop, the line would have gone out until it ended.
	heard := stt.Transcript{Provider: audiofloor.ProviderName, Mode: stt.ModeFinal, Text: "you'd like to move it to Tuesday",
		AudioDurationMs: 1500, ProcessingTimeMs: 100}
	began := time.Now().Add(-ms(100 + 1500))
	start := began.Add(-ms(2900))
	a = &Agent{history: []llm.Message{{Role: llm.Assistant, Content: movedLine}}}
	publishLine(a, start, spokenLine(ms(8980), ms(20), ms(8500)), ms(400))
	_, echo := a.withoutEcho(heard)
	require.Equal(t, 6, echo, "the line still going out")
	a.voiceOut.dropped(start.Add(ms(2000)))
	left, echo := a.withoutEcho(heard)
	require.Zero(t, echo)
	require.Equal(t, heard.Text, left)
}

// loudFor is length of samples loud enough to hear, at rate.
func loudFor(rate int, length time.Duration) []int16 {
	samples := make([]int16, int(length*time.Duration(rate)/time.Second))
	for i := range samples {
		samples[i] = 1000
	}
	return samples
}

func ms(n float64) time.Duration { return time.Duration(n * float64(time.Millisecond)) }

// spokenLine is length of a voice's speech, audible from from to to and silent around it, at
// 16 kHz in the 200 ms chunks a voice sends.
func spokenLine(length, from, to time.Duration) []audio.PcmData {
	const rate = 16000
	at := func(d time.Duration) int { return int(d * rate / time.Second) }
	samples := make([]int16, at(length))
	for i := at(from); i < at(to); i++ {
		samples[i] = 3000
	}
	var chunks []audio.PcmData
	for len(samples) > 0 {
		n := min(len(samples), at(200*time.Millisecond))
		chunks = append(chunks, audio.PcmData{Samples: samples[:n], SampleRate: rate, Channels: 1})
		samples = samples[n:]
	}
	return chunks
}

// publishLine has a publish chunks from at as an edge that paces them does, the voice sending
// them faster than they are heard: each is queued once the publish before returned, goes out
// after what is queued ahead of it, and its publish returns once no more than queue of speech
// waits.
func publishLine(a *Agent, at time.Time, chunks []audio.PcmData, queue time.Duration) {
	queued, out := at, at
	for _, chunk := range chunks {
		if out.Before(queued) {
			out = queued
		}
		out = out.Add(ms(chunk.DurationMs()))
		returned := out.Add(-queue)
		if returned.Before(queued) {
			returned = queued
		}
		a.spoke(queued, returned, chunk)
		queued = returned
	}
}
