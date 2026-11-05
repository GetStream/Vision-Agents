package audiofloor

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// interimEvery is the speech after which the words so far are written while the caller
// talks on, as an interim (stt.Transcript.Interim), when no pause has had them written
// meanwhile. An interim sends the whole utterance so far, which a replacement must, and the
// words have no 16 s limit; so that a caller who talks on costs a transcription lane no more
// as they go, the audio an interim sends a second of speech is bounded by Window, what a
// decision hears: past Window of utterance one is asked less often as it grows, every 2 s at
// 32 s and every 3 s at 48 s. On the TPU the words of 16 s take 50 to 250 ms, so a caller
// talking on costs at most a quarter of one lane, whatever the length of their turn.
const interimEvery = time.Second

var interims = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "chorus_asr_interims_total",
	Help:      "Words asked for while the caller talked on, by what became of them: written, late (newer words or the utterance's final went out first), empty, refused (every lane busy: 503, 429 or busy) or failed. None is asked for again.",
}, []string{"outcome"})

// sentWords is the last words of an utterance to have gone out, which later words of it are
// held to: none after its final, which has the last word, and none of less audio than words
// already out, as an interim answered late would be, so that what a watcher sees never goes
// back. Every transcript the session sends goes through it (send), so a final and an interim
// answered as it goes out can't cross.
type sentWords struct {
	mu        sync.Mutex
	utterance int64
	upTo      time.Duration
	done      bool
}

// send sends transcript unless words of its utterance that supersede it have gone out, and
// says whether it did.
func (t *Transcriber) send(transcript stt.Transcript) bool {
	s := &t.sent
	s.mu.Lock()
	defer s.mu.Unlock()
	upTo := time.Duration(transcript.AudioDurationMs * float64(time.Millisecond))
	switch {
	case transcript.Utterance < s.utterance:
		return false
	case transcript.Utterance == s.utterance && (s.done || (!transcript.Final() && upTo <= s.upTo)):
		return false
	}
	s.utterance, s.upTo, s.done = transcript.Utterance, upTo, transcript.Final()
	t.emitter.Send(transcript)
	return true
}

// ended records that utterance is over with nothing more to send: a check ended it on words
// that came back empty, so nothing was heard, and words of it asked for alone are dropped.
func (s *sentWords) ended(utterance int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if utterance >= s.utterance {
		s.utterance, s.done = utterance, true
	}
}

// over says utterance's final has gone out, or it ended with nothing heard: words of it
// still to be asked for or answered are of an utterance that is over.
func (s *sentWords) over(utterance int64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return utterance < s.utterance || (utterance == s.utterance && s.done)
}

// interim asks for the words so far while the caller talks on, once interimEvery of speech
// has come since they were last asked for, by a pause's first check or an interim, less
// often as the utterance grows past Window: one at a time, on a goroutine of its own, so
// that it holds up no pause check and no final. It runs on the session's goroutine, on a
// frame of speech.
func (t *Transcriber) interim(participant stt.Participant, heard int64) {
	if !t.interims || t.interimOut.Load() {
		return
	}
	length := time.Duration(heard-t.start) * frame
	every := time.Duration(float64(interimEvery) * max(1, float64(length)/float64(Window)))
	if time.Duration(t.spoken)*frame < every {
		return
	}
	t.spoken = 0
	t.interimOut.Store(true)
	t.running.Add(1)
	go t.writeInterim(participant, t.utterance+1, t.heard.Last(length))
}

// writeInterim asks for the words of audio, the utterance so far, and sends them as an
// interim of the utterance unless newer words of it, or its final, have gone out meanwhile.
// A server that is busy or fails costs nothing but this request: it is not asked again.
func (t *Transcriber) writeInterim(participant stt.Participant, utterance int64, audio harness.Audio) {
	defer t.running.Done()
	defer t.interimOut.Store(false)
	start := time.Now()
	ctx, cancel := context.WithTimeout(t.ctx, transcribeTimeout+audio.Duration)
	said, err := t.client.Transcribe(ctx, audio, t.languages)
	cancel()
	if t.ctx.Err() != nil {
		return
	}
	if err != nil {
		interims.WithLabelValues(interimFailure(err)).Inc()
		t.logger.Debug("the chorus-asr server did not write an interim, which is not asked for again",
			"participant", participant.ID, "utterance", utterance, "error", err)
		return
	}
	words := strings.TrimSpace(said)
	if words == "" {
		interims.WithLabelValues("empty").Inc()
		return
	}
	sent := t.send(stt.Transcript{
		Participant:      participant,
		Mode:             stt.ModeReplacement,
		Interim:          true,
		Utterance:        utterance,
		Text:             words,
		Provider:         ProviderName,
		Model:            t.model,
		ProcessingTimeMs: float64(time.Since(start).Microseconds()) / 1000,
		AudioDurationMs:  float64(audio.Duration.Microseconds()) / 1000,
	})
	if sent {
		interims.WithLabelValues("written").Inc()
	} else {
		interims.WithLabelValues("late").Inc()
	}
}

// interimFailure is why an interim was not written, as interims counts it: refused by a
// server shedding load, or failed.
func interimFailure(err error) string {
	var transcript *TranscriptError
	var status *StatusError
	switch {
	case errors.As(err, &transcript):
		return "refused"
	case errors.As(err, &status) && (status.Code == http.StatusServiceUnavailable || status.Code == http.StatusTooManyRequests):
		return "refused"
	}
	return "failed"
}
