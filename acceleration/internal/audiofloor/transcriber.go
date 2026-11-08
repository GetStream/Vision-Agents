package audiofloor

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
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
	// pause that ends it, when its words are asked for alone, with no check ruling on the
	// pause: the end-of-turn policies' first check and cap. The cap stands as the fallback
	// as it is: with no model to say otherwise, a shorter silence would cut the pauses a
	// caller leaves inside a number or a spelled name, which the policy's points were fitted
	// to leave alone, and what a check decides it decides sooner than the cap anyway.
	writeAfter = 200 * time.Millisecond
	endAfter   = 2500 * time.Millisecond
	// preroll is the audio kept ahead of an utterance's first voiced frame, so that a gate
	// that opens late doesn't cut the first sound off.
	preroll = 300 * time.Millisecond
	// maxUtterance bounds one that never pauses, and with it what a session holds: at it,
	// the utterance ends as it is.
	maxUtterance = 60 * time.Second
	// transcribeTimeout bounds one request for words, which on a CPU take seconds, with
	// as long again as the audio lasts: the longer the utterance, the longer its words take,
	// though never as long as it took to say.
	transcribeTimeout = 8 * time.Second
	// failoverAfter is how many requests for words in a row the server must not answer
	// (Unreachable), none answered in between and the last at least failoverWindow after the
	// first, before the session gives it up as down for the call: one request a pause and one
	// an utterance's end, so three over the time one request may take is a server that has
	// not written a word in that long, rather than one restarting between two.
	failoverAfter  = 3
	failoverWindow = transcribeTimeout
	// queued is how many requests may wait for a slow server before the next is dropped.
	queued      = 8
	eventBuffer = 64
)

var transcribeFailures = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "chorus_asr_transcribe_failures_total",
	Help:      "Requests for an utterance's words the chorus-asr server did not answer, by why: its status, timeout or connection. Each is asked again at the utterance's next pause or its end.",
}, []string{"reason"})

// Transcriber writes what a participant says with the chorus-asr server's model, so a call
// needs no other speech-to-text provider. It is an stt.STT that ends each utterance when the
// model says the turn has ended, checking each pause from the voice's last frame at the
// server's policy's times (Pauses): on behalf of the agent that acts on the model, when one
// asks (AskOnPauses), and otherwise on its own, under an agent of its own with no tools
// (neutralAgent) and the server's default preset. The first check of a pause asks for the
// decision and the words of the whole utterance in one request, and a turn that ended goes
// out as one final. When to answer is the flow controller's to say, as it is with any
// transcriber. The server writes nothing of noise, and empty words are nothing heard.
//
// Decisions never hold the words up. A pause no check could rule on, because the server did
// not answer or no check ended it within checkGrace of the policy's cap, falls to the words
// alone: the whole utterance so far is written once the pause reaches writeAfter, as a
// replacement of the words before, asked for alone (POST /v1/transcribe), and the utterance
// ends at a pause of endAfter with one final. A server that can't rule at all, with no
// policy for its model, is asked once a session and the words come alone from then on. The
// transcriber asks for an utterance's words only when speech has come since it last did, so
// audio is never written twice, and the next utterance begins after the last one ended.
//
// While the caller talks the words so far are shown, as interims (stt.Transcript.Interim):
// the words of a pause the model waits on, written with its first check anyway, and about
// once a second of speech with no pause in it, asked for alone, one at a time and never
// again when the server is busy (interims.go). They hold up no check and no final, which
// has the last word: an interim answered after it is dropped. The final and the turn's end
// are as they are without them.
//
// A server that is busy, restarting or slow costs a request, not the session: its words are
// asked for again at the utterance's next pause, or at its end. A misconfigured one, which
// will never answer (Misconfigured), is a fatal error at once, and one that answers none of
// failoverAfter requests in a row over failoverWindow (Unreachable) is down for the call and
// a fatal error then: either hands the call to the next transcriber the routing target
// names, rather than leave it without words.
type Transcriber struct {
	client    *Client
	model     string
	languages []string
	logger    *slog.Logger
	emitter   *stt.Emitter
	requests  chan request
	// now is the clock the failover window is measured by: time.Now, but in tests.
	now func() time.Time
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
	// than the frame ended, where the last one ended. voiced says it has had speech, from
	// frame voicedFrom, fresh that speech has come since its words were last asked for, and
	// quiet how long the pause going on has lasted.
	start, ended  int64
	voiced, fresh bool
	voicedFrom    int64
	quiet         time.Duration
	utterance     int64
	// interims says the words so far are written while the caller talks (interim), the
	// client's setting; spoken is the speech since the utterance's words were last asked
	// for, in frames, talked the speech in the utterance so far, and interimOut says an
	// interim is out, at most one at a time.
	interims   bool
	spoken     int64
	talked     int64
	interimOut atomic.Bool
	// sent is the last words of the utterance to have gone out, which every transcript is
	// held to (send): none after the final, and none that go back on words already out.
	sent sentWords
	// toldAt is when the asker was last told the caller is in an utterance (Hearing), in
	// frames.
	toldAt int64
	// stillOf is the utterance the clock of a pause the gate can't hear is of (still).
	stillOf int64

	// The checks of the pause going on, asked for asker, the agent's (AskOnPauses) or the
	// transcriber's own, each run on a goroutine of its own, and how many of the utterance's
	// pauses were asked about. Nil asks about none.
	mu      sync.Mutex
	asker   Asker
	pause   pause
	pauses  int
	over    overChecks
	running sync.WaitGroup
	// stopping are the checks' rulings over the agent to stop or shorten that wait for the
	// words of the pause they were ruled at, by its epoch, to be handed over with them once
	// its first check has written them (await). They are not the pause's: speech that resumes
	// before the words come ends the pause, and not what was ruled on it. The utterance ending
	// forgets them.
	stopping map[int64]Ruled
	// still is the utterance's words as the requests that wrote them with the floor free last
	// read them, which end a turn the speech gate held voiced (still.go), and stillBound how long
	// they must hold still, under the agent's point as its last check resolved it.
	still      stillWords
	stillBound time.Duration
}

// request is an utterance's words to write, or its end. An end carries the whole
// utterance's audio, and fresh says speech came since its words were last asked for: the end
// is written from its audio then, or when the words last asked for were not written, and
// otherwise on the words last written.
type request struct {
	participant stt.Participant
	utterance   int64
	audio       harness.Audio
	final       bool
	fresh       bool
}

// NewTranscriber returns an unstarted transcriber asking client, for callers who speak
// languages, BCP 47 tags, or any language when there are none.
func NewTranscriber(client *Client, model string, languages []string, logger *slog.Logger) *Transcriber {
	if model == "" {
		model = DefaultModel
	}
	return &Transcriber{client: client, model: model, languages: languages, logger: logger, now: time.Now, stillBound: stillFor,
		asker: neutralAgent{hearer: NewHearer(client, EndOfTurn{}, logger)}, interims: client.interims,
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
// speech that came since they were last asked for, once more at its end, and while the
// caller talks on (interim).
func (t *Transcriber) step(speech bool, participant stt.Participant) {
	if at, ok := t.endedOnCheck(); ok {
		// A pause check ended the utterance on its words: the next begins after its audio.
		t.utterance++
		t.start, t.ended, t.voiced, t.fresh, t.quiet, t.spoken, t.talked = at, at, false, false, 0, 0, 0
	}
	if speech {
		if !t.voiced {
			t.voicedFrom = t.heard.Frames()
		}
		t.voiced, t.fresh, t.quiet = true, true, 0
		t.spoken++
		t.talked++
		t.resumed()
	} else {
		t.quiet += frame
	}
	heard := t.heard.Frames()
	if t.voiced && t.stillOf != t.utterance+1 {
		t.newStill()
	}
	utterance := func() harness.Audio { return t.heard.Last(time.Duration(heard-t.start) * frame) }
	switch {
	case !t.voiced:
		t.start = max(heard-int64(preroll/frame), t.ended)
	case t.checksPause(participant, heard):
		// An agent's checks of the pause end the utterance, with its words.
	case t.quiet >= endAfter || heard-t.start >= int64(maxUtterance/frame):
		// Whether the words last asked for were written is transcribe's to know, which may
		// not have asked for them yet, so the end takes the audio either way.
		t.ask(request{participant: participant, utterance: t.utterance + 1, audio: utterance(), final: true, fresh: t.fresh})
		t.utterance++
		t.start, t.ended, t.voiced, t.fresh, t.quiet, t.spoken, t.talked = heard, heard, false, false, 0, 0, 0
		t.mu.Lock()
		t.newUtteranceLocked()
		t.mu.Unlock()
	case t.quiet >= writeAfter && t.fresh:
		t.ask(request{participant: participant, utterance: t.utterance + 1, audio: utterance()})
		t.fresh, t.spoken = false, 0
	}
	if t.voiced {
		t.checksHeld(participant, heard)
	}
	if speech && !t.checksOver(participant, heard) {
		t.interim(participant, heard)
	}
	t.hearing(participant, heard)
}

// hearing tells the asker the caller is in an utterance, every interval of the policy while
// it lasts: the words come at its pauses or its end, and none at all with interims off, so
// without it an agent waiting on a caller who has not finished hears a silent line.
func (t *Transcriber) hearing(participant stt.Participant, heard int64) {
	if !t.voiced {
		return
	}
	_, every, _ := t.checkTimes()
	if t.toldAt > 0 && time.Duration(heard-t.toldAt)*frame < every {
		return
	}
	t.mu.Lock()
	asker := t.asker
	t.mu.Unlock()
	if asker == nil {
		return
	}
	t.toldAt = heard
	asker.Hearing(participant)
}

func (t *Transcriber) ask(r request) {
	select {
	case t.requests <- r:
	default:
		t.logger.Warn("dropped a request for words, the chorus-asr server is behind", "participant", r.participant.ID, "final", r.final)
		if r.final {
			// The utterance is over with its final lost, and nothing more of it is to come.
			t.sent.ended(r.utterance)
		}
	}
}

// transcribe asks for words one request at a time, so that they arrive in the order said.
// Words that come back empty write nothing, and an utterance ends on the last words of its
// own that didn't. Words of an utterance that is over since they were asked for, its final
// gone out on a pause check's words, are dropped, asked or answered: words after a final
// would be taken for the next utterance's.
func (t *Transcriber) transcribe() {
	defer close(t.done)
	var last string
	var lastOf int64
	// failed counts the requests in a row the server did not answer (Unreachable), and
	// failedSince is when the first of them failed. owed is the utterance whose words were
	// last asked for and not written, which its end asks for again: a failure costs a
	// request, and the end is asked for when the pause is, before the pause's words are
	// written or have failed, as on a server that is slow to fail.
	var failed int
	var failedSince time.Time
	var owed int64
	for {
		var r request
		select {
		case r = <-t.requests:
		case <-t.ctx.Done():
			return
		}
		if t.sent.over(r.utterance) {
			continue
		}
		if r.utterance != lastOf {
			// An utterance whose final was dropped leaves its words behind, which are
			// not the next one's.
			last, lastOf = "", r.utterance
		}
		start := time.Now()
		var words string
		if len(r.audio.Data) > 0 && (!r.final || r.fresh || owed == r.utterance) {
			ctx, cancel := context.WithTimeout(t.ctx, transcribeTimeout+r.audio.Duration)
			said, err := t.client.Transcribe(ctx, r.audio, t.languages)
			cancel()
			if t.ctx.Err() != nil {
				return
			}
			if t.sent.over(r.utterance) {
				continue
			}
			switch {
			case err != nil && Misconfigured(err):
				t.emitter.Send(stt.Error{Provider: ProviderName, Model: t.model, Err: err, Context: "transcribe", Fatal: true})
				return
			case err != nil:
				// The failure is counted once it is in the run of them, so that a count of
				// failures is one of failures the failover has weighed.
				var down time.Duration
				if Unreachable(err) {
					if failed++; failed == 1 {
						failedSince = t.now()
					}
					down = t.now().Sub(failedSince)
				}
				transcribeFailures.WithLabelValues(failure(err)).Inc()
				t.logger.Warn("the chorus-asr server did not write an utterance's words, they are asked for again",
					"participant", r.participant.ID, "utterance", r.utterance, "final", r.final, "error", err)
				owed = r.utterance
				if Unreachable(err) && failed >= failoverAfter && down >= failoverWindow {
					t.emitter.Send(stt.Error{Provider: ProviderName, Model: t.model, Context: "transcribe", Fatal: true,
						Err: stack.Wrap(fmt.Errorf("audiofloor: the server answered none of %d requests for words in %s, the last: %w",
							failed, down.Round(time.Second), err))})
					return
				}
			default:
				failed, owed = 0, 0
				words = strings.TrimSpace(said.Text)
				if words != "" {
					t.sent.establish(r.utterance, said.Language)
				}
			}
		}
		if words != "" {
			last = words
		}
		if last == "" || (words == "" && !r.final) {
			if r.final {
				// Over with nothing heard, or its final lost: nothing more of it is to come.
				t.sent.ended(r.utterance)
			}
			continue
		}
		mode := stt.ModeReplacement
		if r.final {
			mode = stt.ModeFinal
		}
		t.send(stt.Transcript{
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
		t.running.Wait()
	}
	t.emitter.Send(stt.Disconnected{Provider: ProviderName, Model: t.model, Clean: true, At: time.Now()})
	t.emitter.Close()
	return nil
}

// Provider is the stable provider name used in stats.
func (t *Transcriber) Provider() string { return ProviderName }

// Model is the model identifier used in stats.
func (t *Transcriber) Model() string { return t.model }

// failure is why a request failed, as transcribeFailures counts it.
func failure(err error) string {
	var status *StatusError
	switch {
	case errors.As(err, &status):
		return strconv.Itoa(status.Code)
	case errors.Is(err, context.DeadlineExceeded):
		return "timeout"
	}
	return "connection"
}

// level is a frame's root mean square, as a fraction of full scale.
func level(samples []int16) float64 {
	var sum float64
	for _, sample := range samples {
		sum += float64(sample) * float64(sample)
	}
	return math.Sqrt(sum/float64(len(samples))) / math.MaxInt16
}
