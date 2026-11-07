//go:build integration

package agent

import (
	"context"
	"math"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// The loop's contract with another transcriber writing the callers' words, chorus-asr ruling on
// their floor alone: each check of TestAudioFloorContract whose words don't depend on which
// transcriber wrote them, played as it is there, against the server ROUTER_CHORUS_ASR_URL names.
// The words are a streaming transcriber's, simulated (streamingSTT): its revisions of the run of
// speech so far while the caller talks, and a final of the run when it hears 500 ms of silence,
// as a vendor's endpointing does, at a breath mid-sentence as readily as at the turn's end.
//
//	ROUTER_CHORUS_ASR_URL=http://127.0.0.1:8080 GOEXPERIMENT=simd \
//	  go test -tags integration -count=1 -timeout 30m -run 'TestAudioFloorContractWithAnotherTranscriber' -v ./internal/agent/
func TestAudioFloorContractWithAnotherTranscriber(t *testing.T) {
	url := os.Getenv("ROUTER_CHORUS_ASR_URL")
	if url == "" {
		t.Skip("needs ROUTER_CHORUS_ASR_URL, a chorus-asr server")
	}
	h := newFloorHarness(t, url)
	h.warm(t)
	// Its own client, so that what it asks the server is not counted as the floor's.
	words := audiofloor.New(url, audiofloor.Opus, &http.Client{}).WithAPIKey(os.Getenv("ROUTER_CHORUS_ASR_API_KEY"))
	h.writer = func() stt.STT { return newStreamingSTT(words, []string{"en"}) }
	k := &contract{floorHarness: h, clips: map[string][]int16{}}
	for _, name := range []string{"mmhmm", "okay", "yeah", "stoponly", "wait", "andlisbon", "dontstop", "memberid", "andporto", "noporto", "makeitsix"} {
		k.clips[name] = fixture(t, name)
	}
	h.lines = map[string][]int16{listReply: fixture(t, "list"), questionReply: fixture(t, "question")}

	for _, name := range []string{"mmhmm", "okay", "yeah"} {
		t.Run("C1 "+name+" over the agent is ignored", func(t *testing.T) { k.ignored(t, name, k.clips[name], nil, 0) })
	}
	t.Run("C1 the agent's own echo at -20 dB is ignored", func(t *testing.T) { k.ignored(t, "", nil, nil, echoAt(-20)) })
	for _, name := range []string{"stoponly", "wait"} {
		t.Run("C2 "+name+" stops the agent, which stays quiet", func(t *testing.T) { k.stopsAndQuiet(t, name) })
	}
	t.Run("C3 and in Lisbon over a forecast is answered", func(t *testing.T) { k.stopsAndAnswered(t, "andlisbon", "lisbon", false) })
	t.Run("C3 wait, make it six… in one breath, stopped on its first words, is answered on all of them", func(t *testing.T) {
		k.correctedAtLength(t, "makeitsix")
	})
	t.Run("C4 a shortened reply stops within 250 ms of the ruling", func(t *testing.T) { k.shortened(t, "andporto") })
	t.Run("C5 don't stop over the agent: one ruling acted on, no second wait", func(t *testing.T) { k.oneWait(t, "dontstop") })
	t.Run("C5 no, Porto over the agent: answered at the point, no second wait", func(t *testing.T) { k.oneWait(t, "noporto") })
	t.Run("C6 no check-in while the caller counts slowly, interims off", k.noCheckInMidTurn)
	t.Run("C6 a member ID dictated with pauses is one turn", k.dictation)
}

const (
	// streamingLevel is the root mean square, as a fraction of full scale, from which the
	// simulated transcriber hears a frame as speech: -40 dBFS.
	streamingLevel = 0.01
	// streamingRevise is the speech after which it writes the run so far again, and
	// streamingEnd the silence that ends a run.
	streamingRevise = 300 * time.Millisecond
	streamingEnd    = 500 * time.Millisecond
)

// streamingSTT stands in for a vendor's streaming transcriber, fed the call's audio as the
// router feeds any: it hears a run of speech by its level, writes the run so far through the
// chorus-asr server's POST /v1/transcribe every streamingRevise of speech while it lasts, as a
// vendor's revisions come, and ends the run at streamingEnd of silence with a final of all of
// it. Its requests go one at a time, so its words come in the order said. It can be told to
// settle the run at once (stt.Finalizer).
type streamingSTT struct {
	client    *audiofloor.Client
	languages []string
	emitter   *stt.Emitter
	jobs      chan streamingJob
	ctx       context.Context
	cancel    context.CancelFunc
	done      chan struct{}

	mu          sync.Mutex
	participant stt.Participant
	// run is the run being heard, none when zero, runs how many there have been, and stream
	// the run's audio, heard of it so far.
	run, runs     int64
	stream        *audiofloor.Stream
	heard         time.Duration
	quiet, unsaid time.Duration
}

// streamingJob is a run's words to write: so far, or final.
type streamingJob struct {
	participant stt.Participant
	run         int64
	audio       harness.Audio
	final       bool
}

func newStreamingSTT(client *audiofloor.Client, languages []string) *streamingSTT {
	ctx, cancel := context.WithCancel(context.Background())
	return &streamingSTT{client: client, languages: languages, emitter: stt.NewEmitter(64), jobs: make(chan streamingJob, 64),
		ctx: ctx, cancel: cancel, done: make(chan struct{})}
}

func (s *streamingSTT) Start(context.Context) error {
	go s.write()
	s.emitter.Send(stt.Connected{Provider: "stub", Model: "stub-model", At: time.Now()})
	return nil
}

func (s *streamingSTT) ProcessAudio(pcm stt.PcmData, participant stt.Participant) error {
	var sum float64
	for _, sample := range pcm.Samples {
		sum += float64(sample) * float64(sample)
	}
	speech := len(pcm.Samples) > 0 && math.Sqrt(sum/float64(len(pcm.Samples)))/math.MaxInt16 >= streamingLevel
	took := time.Duration(len(pcm.Samples)) * time.Second / stt.SampleRate
	s.mu.Lock()
	defer s.mu.Unlock()
	s.participant = participant
	if s.run == 0 {
		if !speech {
			return nil
		}
		stream, err := s.client.NewStream(time.Minute)
		if err != nil {
			return err
		}
		s.runs++
		s.run, s.stream, s.heard, s.quiet, s.unsaid = s.runs, stream, 0, 0, 0
	}
	if err := s.stream.Write(pcm.Samples); err != nil {
		return err
	}
	s.heard += took
	if speech {
		s.quiet, s.unsaid = 0, s.unsaid+took
	} else {
		s.quiet += took
	}
	switch {
	case s.quiet >= streamingEnd:
		s.endLocked()
	case s.unsaid >= streamingRevise:
		s.unsaid = 0
		select {
		case s.jobs <- streamingJob{participant: participant, run: s.run, audio: s.stream.Last(s.heard)}:
		default:
		}
	}
	return nil
}

// endLocked ends the run being heard, its final to be written. s.mu is held.
func (s *streamingSTT) endLocked() {
	if s.run == 0 {
		return
	}
	s.jobs <- streamingJob{participant: s.participant, run: s.run, audio: s.stream.Last(s.heard), final: true}
	s.run = 0
}

// Finalize ends the run being heard at once.
func (s *streamingSTT) Finalize() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.endLocked()
	return nil
}

// write writes each job's words, in turn, and drops what comes of a run after its final.
func (s *streamingSTT) write() {
	defer close(s.done)
	var settled int64
	for {
		var job streamingJob
		select {
		case job = <-s.jobs:
		case <-s.ctx.Done():
			return
		}
		if job.run <= settled {
			continue
		}
		ctx, cancel := context.WithTimeout(s.ctx, 8*time.Second+job.audio.Duration)
		said, err := s.client.Transcribe(ctx, job.audio, s.languages)
		cancel()
		if s.ctx.Err() != nil {
			return
		}
		if job.final {
			settled = job.run
		}
		text := strings.TrimSpace(said.Text)
		if err != nil || text == "" {
			continue
		}
		mode := stt.ModeReplacement
		if job.final {
			mode = stt.ModeFinal
		}
		s.emitter.Send(stt.Transcript{Participant: job.participant, Mode: mode, Utterance: job.run, Text: text,
			Language: "en", Provider: "stub", Model: "stub-model"})
	}
}

func (s *streamingSTT) Events() <-chan stt.Event { return s.emitter.Events() }

func (s *streamingSTT) Close() error {
	s.cancel()
	<-s.done
	s.emitter.Close()
	return nil
}

func (s *streamingSTT) Provider() string { return "stub" }
func (s *streamingSTT) Model() string    { return "stub-model" }

