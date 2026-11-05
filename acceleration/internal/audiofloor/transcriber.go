package audiofloor

import (
	"context"
	"errors"
	"log/slog"
	"math"
	"strings"
	"time"

	"github.com/thesyncim/gopus"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName and DefaultModel name the transcriber in routing.
const (
	ProviderName = "stream"
	DefaultModel = "chorus-asr"
)

// A frame is speech when it is loud enough and the speech detector, Opus's neural speech
// analysis, scores it as speech: the level says when the voice stops, as the detector's
// score lingers for about 200 ms after it, and the score tells speech from noise, hum and
// clicks, however loud.
const (
	// voiceLevel is the root mean square, as a fraction of full scale, from which a frame
	// can be speech: about -48 dBFS, below quiet speech and above a phone line's hiss.
	voiceLevel = 0.004
	// speechScore is the detector's score from which a frame is speech, and detectorWarmup
	// the windows it adapts for first, as the server's speech gate reads them.
	speechScore    = 0.5
	detectorWarmup = 10
	// writeAfter is the pause at which the utterance so far is written, and endAfter the
	// pause that ends it: the end-of-turn policies' first check and cap.
	writeAfter = 200 * time.Millisecond
	endAfter   = 2500 * time.Millisecond
	// preroll is the audio kept ahead of an utterance's first voiced frame, so that a gate
	// that opens late doesn't cut the first sound off.
	preroll = 300 * time.Millisecond
	// maxUtterance bounds one that never pauses, and with it what a session holds: at it,
	// the utterance ends as it is.
	maxUtterance = 60 * time.Second
	// transcribeTimeout bounds one request for words, which on a CPU take seconds.
	transcribeTimeout = 8 * time.Second
	// queued is how many requests may wait for a slow server before the next is dropped.
	queued      = 8
	eventBuffer = 64
)

// Transcriber writes what a participant says with the chorus-asr server's model, so a call
// needs no other speech-to-text provider. It is an stt.STT that writes the whole utterance
// so far at each of its pauses, once the pause reaches writeAfter, as a replacement of the
// words before, and ends the utterance at a pause of endAfter with one final transcript.
// It asks for an utterance's words only when speech has come since it last did, so audio
// is never written twice, and the next utterance begins after the last one ended. When to
// answer is the flow controller's to say, as it is with any transcriber. It asks the
// server for the words alone (POST /v1/transcribe): no agent, decision or policy. The
// server writes nothing of noise, and empty words are nothing heard.
//
// A server that fails to answer is a fatal error, which hands the call to the next
// transcriber the routing target names. The words since the last ones written are lost.
type Transcriber struct {
	client    *Client
	model     string
	languages []string
	logger    *slog.Logger
	emitter   *stt.Emitter
	requests  chan request
	// ctx is the session's lifetime, made in Start, and done closes when transcribe returns.
	ctx    context.Context
	cancel context.CancelFunc
	done   chan struct{}

	// What ProcessAudio keeps, on the session's own goroutine: the frame being filled, the
	// detector that scores each, and the stream of them.
	detector *gopus.SpeechDetector
	frame    [frameSamples]int16
	filled   int
	windows  int
	heard    *Stream
	// The utterance being heard: its audio is heard's from frame start on, and no earlier
	// than the frame ended, where the last one ended. voiced says it has had speech, fresh
	// that speech has come since its words were last asked for, and quiet how long the
	// pause going on has lasted.
	start, ended  int64
	voiced, fresh bool
	quiet         time.Duration
	utterance     int64
}

// request is an utterance's words to write, or with no audio its end on the words last
// written.
type request struct {
	participant stt.Participant
	utterance   int64
	audio       harness.Audio
	final       bool
}

// NewTranscriber returns an unstarted transcriber asking client, for callers who speak
// languages, BCP 47 tags, or any language when there are none.
func NewTranscriber(client *Client, model string, languages []string, logger *slog.Logger) *Transcriber {
	if model == "" {
		model = DefaultModel
	}
	return &Transcriber{client: client, model: model, languages: languages, logger: logger,
		emitter: stt.NewEmitter(eventBuffer), requests: make(chan request, queued), done: make(chan struct{})}
}

// Start opens the session. It asks the server nothing.
func (t *Transcriber) Start(context.Context) error {
	var err error
	if t.heard, err = t.client.NewStream(maxUtterance); err != nil {
		return err
	}
	if t.detector, err = gopus.NewSpeechDetector(SampleRate); err != nil {
		return stack.Wrap(err)
	}
	t.ctx, t.cancel = context.WithCancel(context.Background())
	go t.transcribe()
	t.emitter.Send(stt.Connected{Provider: ProviderName, Model: t.model, At: time.Now()})
	return nil
}

// ProcessAudio keeps the utterance's audio, 20 ms at a time, and asks for its words at its
// pauses.
func (t *Transcriber) ProcessAudio(pcm stt.PcmData, participant stt.Participant) error {
	if t.ctx == nil {
		return stack.Wrap(errors.New("audiofloor: the session is not open"))
	}
	if err := pcm.Validate(stt.SampleRate); err != nil {
		return err
	}
	for samples := pcm.Samples; len(samples) > 0; {
		n := copy(t.frame[t.filled:], samples)
		samples, t.filled = samples[n:], t.filled+n
		if t.filled < frameSamples {
			break
		}
		t.filled = 0
		if err := t.heard.Write(t.frame[:]); err != nil {
			return err
		}
		score, err := t.detector.AnalyzeInt16(t.frame[:])
		if err != nil {
			return stack.Wrap(err)
		}
		t.windows++
		t.step(t.windows > detectorWarmup && score >= speechScore && level(t.frame[:]) >= voiceLevel, participant)
	}
	return nil
}

// step reads one frame, speech or not, and asks for the utterance's words: at a pause after
// speech that came since they were last asked for, and once more at its end.
func (t *Transcriber) step(speech bool, participant stt.Participant) {
	if speech {
		t.voiced, t.fresh, t.quiet = true, true, 0
	} else {
		t.quiet += frame
	}
	heard := t.heard.Frames()
	utterance := func() harness.Audio { return t.heard.Last(time.Duration(heard-t.start) * frame) }
	switch {
	case !t.voiced:
		t.start = max(heard-int64(preroll/frame), t.ended)
	case t.quiet >= endAfter || heard-t.start >= int64(maxUtterance/frame):
		end := request{participant: participant, utterance: t.utterance + 1, final: true}
		if t.fresh {
			end.audio = utterance()
		}
		t.ask(end)
		t.utterance++
		t.start, t.ended, t.voiced, t.fresh, t.quiet = heard, heard, false, false, 0
	case t.quiet >= writeAfter && t.fresh:
		t.ask(request{participant: participant, utterance: t.utterance + 1, audio: utterance()})
		t.fresh = false
	}
}

func (t *Transcriber) ask(r request) {
	select {
	case t.requests <- r:
	default:
		t.logger.Warn("dropped a request for words, the chorus-asr server is behind", "participant", r.participant.ID)
	}
}

// transcribe asks for words one request at a time, so that they arrive in the order said.
// Words that come back empty write nothing, and an utterance ends on the last words that
// didn't.
func (t *Transcriber) transcribe() {
	defer close(t.done)
	var last string
	for {
		var r request
		select {
		case r = <-t.requests:
		case <-t.ctx.Done():
			return
		}
		start := time.Now()
		var words string
		if r.audio.Data != nil {
			ctx, cancel := context.WithTimeout(t.ctx, transcribeTimeout)
			said, err := t.client.Transcribe(ctx, r.audio, t.languages)
			cancel()
			if t.ctx.Err() != nil {
				return
			}
			if err != nil {
				t.emitter.Send(stt.Error{Provider: ProviderName, Model: t.model, Err: err, Context: "transcribe", Fatal: true})
				return
			}
			words = strings.TrimSpace(said)
		}
		if words != "" {
			last = words
		}
		if last == "" || (words == "" && !r.final) {
			continue
		}
		mode := stt.ModeReplacement
		if r.final {
			mode = stt.ModeFinal
		}
		t.emitter.Send(stt.Transcript{
			Participant:      r.participant,
			Mode:             mode,
			Utterance:        r.utterance,
			Text:             last,
			Provider:         ProviderName,
			Model:            t.model,
			ProcessingTimeMs: float64(time.Since(start).Microseconds()) / 1000,
			AudioDurationMs:  float64(r.audio.Duration.Microseconds()) / 1000,
		})
		if r.final {
			last = ""
		}
	}
}

// Events carries the transcripts back.
func (t *Transcriber) Events() <-chan stt.Event { return t.emitter.Events() }

// Close stops a request still out and waits for it.
func (t *Transcriber) Close() error {
	if t.cancel != nil {
		t.cancel()
		<-t.done
	}
	t.emitter.Send(stt.Disconnected{Provider: ProviderName, Model: t.model, Clean: true, At: time.Now()})
	t.emitter.Close()
	return nil
}

// Provider is the stable provider name used in stats.
func (t *Transcriber) Provider() string { return ProviderName }

// Model is the model identifier used in stats.
func (t *Transcriber) Model() string { return t.model }

// level is a frame's root mean square, as a fraction of full scale.
func level(samples []int16) float64 {
	var sum float64
	for _, sample := range samples {
		sum += float64(sample) * float64(sample)
	}
	return math.Sqrt(sum/float64(len(samples))) / math.MaxInt16
}
