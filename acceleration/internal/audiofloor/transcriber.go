package audiofloor

import (
	"context"
	"errors"
	"log/slog"
	"math"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName and DefaultModel name the transcriber in routing.
const (
	ProviderName = "audiofloor"
	DefaultModel = "audio-floor"
)

const (
	// voiceLevel is the root mean square, as a fraction of full scale, from which a frame
	// counts as voice: about -48 dBFS, below quiet speech and above a phone line's hiss.
	// It is the gate the model's reference client uses, not one measured on these calls.
	// It says where the pauses are; what is noise is the server's to say.
	voiceLevel = 0.004
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

// Transcriber writes what a participant says with the audio floor server's model, so a call
// needs no other speech-to-text provider. It is an stt.STT that writes the whole utterance
// so far at each of its pauses, once the pause reaches the policy's first check, as a
// replacement of the words before, and ends the utterance when a pause reaches the policy's
// cap. When to answer is the flow controller's to say, as it is with any transcriber. The
// server writes nothing of noise, and empty words are nothing heard.
//
// A server that fails to answer is a fatal error, which hands the call to the next
// transcriber the routing target names. The words since the last ones written are lost.
type Transcriber struct {
	client   *Client
	model    string
	agent    Agent
	logger   *slog.Logger
	emitter  *stt.Emitter
	requests chan request
	// ctx is the session's lifetime, made in Start, and done closes when transcribe returns.
	ctx    context.Context
	cancel context.CancelFunc
	done   chan struct{}

	// The utterance being heard, kept by ProcessAudio on the session's own goroutine: heard
	// holds its audio from frame start on.
	first, cap time.Duration
	heard      *Stream
	start      int64
	voiced     bool
	quiet      time.Duration
	// asked says the pause going on has had its words asked for.
	asked     bool
	utterance int64
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
	return &Transcriber{client: client, model: model, agent: Agent{Languages: languages}, logger: logger,
		emitter: stt.NewEmitter(eventBuffer), requests: make(chan request, queued), done: make(chan struct{})}
}

// Start reads the server's policy, which says which pauses are written and which end an
// utterance.
func (t *Transcriber) Start(ctx context.Context) error {
	policy, err := t.client.Policy(ctx)
	if err != nil {
		return err
	}
	if t.heard, err = t.client.NewStream(maxUtterance); err != nil {
		return err
	}
	t.first, t.cap = policy.FirstCheck(), policy.Cap()
	t.ctx, t.cancel = context.WithCancel(context.Background())
	go t.transcribe()
	t.emitter.Send(stt.Connected{Provider: ProviderName, Model: t.model, At: time.Now()})
	return nil
}

// ProcessAudio keeps one frame of the utterance, and asks for its words at its pauses.
func (t *Transcriber) ProcessAudio(pcm stt.PcmData, participant stt.Participant) error {
	if t.ctx == nil {
		return stack.Wrap(errors.New("audiofloor: the session is not open"))
	}
	if err := pcm.Validate(stt.SampleRate); err != nil {
		return err
	}
	if err := t.heard.Write(pcm.Samples); err != nil {
		return err
	}
	if level(pcm.Samples) >= voiceLevel {
		t.voiced, t.quiet, t.asked = true, 0, false
	} else {
		t.quiet += time.Duration(pcm.DurationMs() * float64(time.Millisecond))
	}
	heard := t.heard.Frames()
	switch {
	case !t.voiced:
		t.start = heard - int64(preroll/frame)
	case t.quiet >= t.cap || heard-t.start >= int64(maxUtterance/frame):
		end := request{participant: participant, utterance: t.utterance + 1, final: true}
		if !t.asked {
			end.audio = t.heard.Last(time.Duration(heard-t.start) * frame)
		}
		t.ask(end)
		t.utterance++
		// The next utterance begins with the end of this one, quiet as it is.
		t.start, t.voiced, t.quiet, t.asked = heard-int64(preroll/frame), false, 0, false
	case t.quiet >= t.first && !t.asked:
		t.ask(request{participant: participant, utterance: t.utterance + 1, audio: t.heard.Last(time.Duration(heard-t.start) * frame)})
		t.asked = true
	}
	return nil
}

func (t *Transcriber) ask(r request) {
	select {
	case t.requests <- r:
	default:
		t.logger.Warn("dropped a request for words, the audio floor server is behind", "participant", r.participant.ID)
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
			answer, err := t.client.Decide(ctx, t.agent, Turn{}, r.audio, true)
			cancel()
			if t.ctx.Err() != nil {
				return
			}
			if err != nil {
				t.emitter.Send(stt.Error{Provider: ProviderName, Model: t.model, Err: err, Context: "transcribe", Fatal: true})
				return
			}
			words = strings.TrimSpace(answer.Transcript)
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
	if len(samples) == 0 {
		return 0
	}
	var sum float64
	for _, sample := range samples {
		sum += float64(sample) * float64(sample)
	}
	return math.Sqrt(sum/float64(len(samples))) / math.MaxInt16
}
