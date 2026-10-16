// Package audioturn uses the same AudioTurn request for transcription and turn scoring.
package audioturn

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/eotdefaults"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const ProviderName = "audioturn"

// Options selects the service already used for turn detection. Client allows an
// embedding application to share its configured client across transcription sessions.
type Options struct {
	Client *Client
	Model  string
}

// STT adapts rolling AudioTurn windows to ordinary replacement/final transcripts.
// One session belongs to one participant, as with the other streaming providers.
type STT struct {
	client  *Client
	emitter *stt.Emitter
	running sync.WaitGroup

	mu          sync.Mutex
	ctx         context.Context
	cancel      context.CancelFunc
	closed      bool
	audio       []int16
	total       int64
	scored      int64
	participant stt.Participant
}

func New(options Options) (*STT, error) {
	if options.Model != "" && options.Model != DefaultModel {
		return nil, errors.New("audioturn: unsupported model")
	}
	client := options.Client
	if client == nil {
		endpoint, set := os.LookupEnv("ROUTER_EOT_URL")
		if !set {
			endpoint = eotdefaults.HostedDemoEndpoint
		}
		tokenFile := os.Getenv("ROUTER_EOT_ID_TOKEN_FILE")
		var err error
		if eotdefaults.IsHostedDemoEndpoint(endpoint) && strings.TrimSpace(tokenFile) == "" {
			client, err = NewHostedClient()
		} else {
			client, err = NewClient(endpoint, tokenFile)
		}
		if err != nil {
			return nil, err
		}
	}
	return &STT{client: client, emitter: stt.NewEmitter(64)}, nil
}

func (s *STT) Provider() string         { return ProviderName }
func (s *STT) Model() string            { return DefaultModel }
func (s *STT) Events() <-chan stt.Event { return s.emitter.Events() }

func (s *STT) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.closed || s.ctx != nil {
		s.mu.Unlock()
		return errors.New("audioturn: session already started or closed")
	}
	s.ctx, s.cancel = context.WithCancel(ctx)
	s.mu.Unlock()
	// Refuse a decision-only deployment before accepting caller audio.
	_, _, err := s.client.Transcribe(s.ctx, "transcription-preflight", make([]byte, MinSamples*2))
	if err != nil {
		_ = s.Close()
		return fmt.Errorf("audioturn: transcription preflight failed: %w", err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return errors.New("audioturn: session closed")
	}
	s.running.Add(1)
	go s.run()
	return nil
}

func (s *STT) ProcessAudio(pcm stt.PcmData, participant stt.Participant) error {
	if err := pcm.Validate(SampleRate); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.ctx == nil || s.ctx.Err() != nil {
		return errors.New("audioturn: session is not running")
	}
	limit := int64(MaxSamples)
	if s.scored > 0 {
		limit -= SampleRate // keep the last pass's provisional second in context
	}
	if s.total+int64(len(pcm.Samples))-s.scored > limit {
		return errors.New("audioturn: transcription fell behind the audio window")
	}
	s.total += int64(len(pcm.Samples))
	s.participant = participant
	samples := pcm.Samples
	if len(samples) >= MaxSamples {
		s.audio = append(s.audio[:0], samples[len(samples)-MaxSamples:]...)
	} else {
		if overflow := len(s.audio) + len(samples) - MaxSamples; overflow > 0 {
			s.audio = s.audio[:copy(s.audio, s.audio[overflow:])]
		}
		s.audio = append(s.audio, samples...)
	}
	return nil
}

func (s *STT) Close() error {
	s.mu.Lock()
	s.closed = true
	if s.cancel != nil {
		s.cancel()
	}
	s.mu.Unlock()
	s.emitter.Close()
	s.running.Wait()
	return nil
}

func (s *STT) run() {
	defer s.running.Done()
	defer s.emitter.Close()
	defer s.cancel()
	stop := context.AfterFunc(s.ctx, s.emitter.Close)
	defer stop()
	s.emitter.Send(stt.Connected{Provider: ProviderName, Model: DefaultModel, At: time.Now()})
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	var read int64
	var words windowTranscript
	utterance := int64(1)
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
		}
		s.mu.Lock()
		if s.total == read || len(s.audio) < MinSamples {
			s.mu.Unlock()
			continue
		}
		end, participant := s.total, s.participant
		// Do not decode silence before speech or after a settled utterance.
		if len(words.words) == 0 && quietAudio(s.audio[max(0, len(s.audio)-int(end-words.turnStart)):]) {
			read, s.scored = end, end
			s.mu.Unlock()
			continue
		}
		const silence = SampleRate * 600 / 1000
		quiet := len(s.audio) >= silence && quietAudio(s.audio[len(s.audio)-silence:])
		pcm := (stt.PcmData{Samples: s.audio, SampleRate: SampleRate, Channels: 1}).Bytes()
		s.mu.Unlock()

		started := time.Now()
		ctx, cancel := context.WithTimeout(s.ctx, PrimaryLimit)
		score, transcript, err := s.client.Transcribe(ctx, fmt.Sprintf("%s-%d", participant.ID, end), pcm)
		cancel()
		if err != nil {
			if s.ctx.Err() != nil {
				return
			}
			fatal := !IsTransientError(err)
			s.emitter.Send(stt.Error{Provider: ProviderName, Model: DefaultModel, Err: err, Fatal: fatal})
			if fatal {
				return
			}
			continue
		}
		read = end
		s.mu.Lock()
		s.scored = end
		s.mu.Unlock()
		endMS := (end*1000 + SampleRate - 1) / SampleRate
		text := words.update(*transcript, endMS)
		if text == "" {
			continue
		}
		// Word durations can extend into silence. Settle against the audio, while
		// the turn score remains a separate decision about when to answer.
		mode := stt.ModeReplacement
		if quiet {
			mode = stt.ModeFinal
		}
		s.emitter.Send(stt.Transcript{
			Participant: participant, Mode: mode, Utterance: utterance, Text: text,
			Provider: ProviderName, Model: DefaultModel, TurnProbability: &score.Probability,
			ProcessingTimeMs: float64(time.Since(started).Microseconds()) / 1000,
			AudioDurationMs:  float64(end-words.turnStart) * 1000 / SampleRate,
		})
		if quiet {
			utterance++
			words = windowTranscript{turnStart: end}
		}
	}
}

// quietAudio checks for audio below -42 dBFS in 20 ms frames so a short
// word cannot disappear into the average energy of a longer quiet window.
func quietAudio(samples []int16) bool {
	for len(samples) > 0 {
		frame := samples[:min(len(samples), MinSamples)]
		var energy int64
		for _, sample := range frame {
			energy += int64(sample) * int64(sample)
		}
		if energy >= 250*250*int64(len(frame)) {
			return false
		}
		samples = samples[len(frame):]
	}
	return true
}

// Once a turn outgrows the window, keep its prefix and revise the last second.
// Earlier words retain the context they were decoded with.
type windowTranscript struct {
	words     []Word // timestamps on the session clock
	turnStart int64  // samples
	endMS     int
}

func (w *windowTranscript) update(transcript Transcript, endMS int64) string {
	keep := 0
	cutoff := w.endMS - 1000
	if len(w.words) == 0 || endMS-int64(w.words[0].StartMS) <= MaxSamples*1000/SampleRate {
		cutoff = -1
	}
	w.endMS = int(endMS)
	for keep < len(w.words) && w.words[keep].EndMS <= cutoff {
		keep++
	}
	w.words = w.words[:keep]
	seam := int(w.turnStart*1000/SampleRate) - 1
	start := 0
	if keep > 0 {
		last := w.words[keep-1]
		seam = max(seam, last.EndMS-1)
		// Align the last retained word: decoder timestamps can shift between
		// windows. The closest occurrence preserves intentional repetitions.
		distance := 1000
		for i, word := range transcript.Words {
			delta := word.StartMS + int(endMS) - last.StartMS
			if max(delta, -delta) < distance && strings.EqualFold(strings.Trim(word.Text, ".,!?;:"), strings.Trim(last.Text, ".,!?;:")) {
				start, distance = i+1, max(delta, -delta)
				seam = int(w.turnStart*1000/SampleRate) - 1
			}
		}
	}
	for _, word := range transcript.Words[start:] {
		word.StartMS += int(endMS)
		word.EndMS += int(endMS)
		if word.StartMS > seam {
			w.words = append(w.words, word)
		}
	}
	text := make([]string, len(w.words))
	for i, word := range w.words {
		text[i] = word.Text
	}
	return strings.Join(text, " ")
}
