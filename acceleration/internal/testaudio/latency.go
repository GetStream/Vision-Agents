package testaudio

import (
	"fmt"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// speaker is who the measured audio is attributed to.
var speaker = stt.Participant{ID: "test-user", UserID: "test-user"}

// quietSample is the amplitude, out of a full scale of 32767, below which a chunk counts
// as the room rather than the voice. It is low enough to keep a softly spoken final word
// and high enough to ignore the noise floor of a decoded recording.
const quietSample = 500

// Timing is the wait a caller actually experiences, measured against the clock.
//
// Providers report a latency of their own, but it is generally measured from the last
// audio frame pushed rather than from the speech it describes. Audio streams continuously,
// so that number stays small however long the caller waited, and it cannot answer the only
// question that matters: how long after speaking does the transcript appear.
type Timing struct {
	// SpokeFor is how long there was a voice in the audio.
	SpokeFor time.Duration
	// ToFirstWords is from the start of the speech to the first transcript of any kind.
	// Longer than SpokeFor means nothing appeared until after the caller had stopped.
	ToFirstWords time.Duration
	// ToSettle is from the last word spoken to the settled transcript. The rest of the
	// turn queues behind this, so it is the delay a slow conversation is built on.
	ToSettle time.Duration
	// WhileSpeaking counts the transcripts that arrived before the caller stopped, which
	// is what makes a transcript feel live rather than arriving in a lump.
	WhileSpeaking int
	// Text is the settled transcript.
	Text string
}

// Measure streams speech at the pace a call delivers it, follows it with the silence that
// ends a turn, and reports what the caller waited for. The provider must already be
// started; closing it is left to the caller.
func Measure(provider stt.STT, audio stt.PcmData, chunkMs, silenceMs int, patience time.Duration) (Timing, error) {
	type hearing struct {
		at         time.Time
		transcript stt.Transcript
	}

	heard := make(chan hearing, 128)
	go func() {
		defer close(heard)
		for event := range provider.Events() {
			if transcript, ok := event.(stt.Transcript); ok {
				heard <- hearing{at: time.Now(), transcript: transcript}
			}
		}
	}()

	spoken := Chunks(audio, chunkMs)
	lastSpoken := lastChunkWithSpeech(spoken)

	started := time.Now()
	var stopped time.Time
	for i, chunk := range spoken {
		if wait := time.Until(started.Add(time.Duration(i*chunkMs) * time.Millisecond)); wait > 0 {
			time.Sleep(wait)
		}
		if err := provider.ProcessAudio(chunk, speaker); err != nil {
			return Timing{}, fmt.Errorf("testaudio: sending speech: %w", err)
		}
		if i == lastSpoken {
			stopped = time.Now()
		}
	}

	// The silence goes out from another goroutine so a provider that settles late is still
	// being fed while we wait, the way a call would keep delivering room tone.
	quiet := make(chan error, 1)
	go func() {
		start := time.Now()
		for i, chunk := range Chunks(Silence(silenceMs), chunkMs) {
			if wait := time.Until(start.Add(time.Duration(i*chunkMs) * time.Millisecond)); wait > 0 {
				time.Sleep(wait)
			}
			if err := provider.ProcessAudio(chunk, speaker); err != nil {
				quiet <- err
				return
			}
		}
		quiet <- nil
	}()

	timing := Timing{SpokeFor: stopped.Sub(started)}
	deadline := time.After(patience)
	first := true
	for {
		select {
		case event, open := <-heard:
			if !open {
				return timing, fmt.Errorf("testaudio: the provider stopped before settling a turn")
			}
			if first {
				timing.ToFirstWords = event.at.Sub(started)
				first = false
			}
			if event.at.Before(stopped) {
				timing.WhileSpeaking++
			}
			if event.transcript.Final() {
				timing.ToSettle = event.at.Sub(stopped)
				timing.Text = event.transcript.Text
				return timing, nil
			}
		case err := <-quiet:
			if err != nil {
				return timing, fmt.Errorf("testaudio: sending silence: %w", err)
			}
		case <-deadline:
			return timing, fmt.Errorf("testaudio: no settled transcript within %s", patience)
		}
	}
}

// lastChunkWithSpeech is the index of the final chunk carrying voice rather than the room
// tone a recording trails off into. Timing the wait from the end of the file instead would
// credit the provider for silence it was given for free.
func lastChunkWithSpeech(chunks []stt.PcmData) int {
	last := 0
	for i, chunk := range chunks {
		for _, sample := range chunk.Samples {
			if sample > quietSample || sample < -quietSample {
				last = i
				break
			}
		}
	}
	return last
}
