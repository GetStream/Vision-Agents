package simulation

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/loopback"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// What a conversation out loud is given over one in writing. Everything is slower: the
// caller's line has to be synthesised and then played at the speed somebody talks, and the
// agent's reply has to be spoken and then transcribed before there is anything to read.
const (
	spokenSettle = 1500 * time.Millisecond
	spokenWithin = 120 * time.Second
	// speakWithin bounds synthesising one line of the caller's speech.
	speakWithin = 30 * time.Second
)

// The caller's own voice and ears, when the simulation did not name them.
const (
	defaultCallerTTS = "en-low-latency"
	defaultCallerSTT = "en-low-latency"
)

// caller is who the simulated caller is on the call, which is somebody other than the agent
// so the transcriber has two participants to tell apart.
var callerParticipant = stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}

// spoken is a conversation held out loud.
//
// The agent is heard the way a caller hears it: through its own voice, transcribed. That is
// what running the whole pipeline means, and it is what catches a reply that is correct on
// paper and unintelligible in the air.
type spoken struct {
	created *session.Session
	closer  func()
	edge    *loopback.Edge
	mic     *loopback.Microphone
	voice   *ttsrouter.Session
	ears    *sttrouter.Session
	within  time.Duration
	opening string

	mu sync.Mutex
	// heard is what the caller's transcriber made of the agent's reply.
	heard []string
	// intended is what the agent meant to say, kept alongside so a failure caused by the
	// voice rather than the answer can be told apart from one caused by the answer.
	intended []string
	lastAt   time.Time
	failure  error
	gone     bool
}

// converse opens a conversation out loud against the agent under test.
func (r *Runner) converse(
	ctx context.Context,
	spec session.Spec,
	simulation store.Simulation,
	kase store.SimulationCase,
) (transport, error) {
	if r.tts == nil || r.stt == nil {
		return nil, errors.New("simulation: this deployment cannot give the caller a voice or ears")
	}

	held := &spoken{within: spokenWithin, lastAt: time.Now()}

	// The caller's ears are opened first, because the edge publishes into them from the
	// moment the agent says anything.
	ears, err := r.stt.Start(ctx, sttrouter.Request{
		CustomerID: simulation.CustomerID,
		AgentID:    spec.AgentID,
		CallID:     spec.CallID,
		Tags:       spec.Tags,
		Target:     target(simulation.CallerSTT, defaultCallerSTT),
	})
	if err != nil {
		return nil, fmt.Errorf("simulation: give the caller ears: %w", err)
	}
	held.ears = ears
	go held.listen(ears.Events())

	voice, err := r.tts.Start(ctx, ttsrouter.Request{
		CustomerID: simulation.CustomerID,
		AgentID:    spec.AgentID,
		CallID:     spec.CallID,
		Tags:       spec.Tags,
		Target:     target(simulation.CallerTTS, defaultCallerTTS),
		Voice:      simulation.CallerVoice,
	})
	if err != nil {
		held.Close()
		return nil, fmt.Errorf("simulation: give the caller a voice: %w", err)
	}
	held.voice = voice

	held.edge = loopback.New(loopback.Options{Published: held.overhear})
	spec.Edge = held.edge

	greeting := spec.Greeting
	spec.Greeting = ""

	created, err := r.sessions.Create(ctx, spec)
	if err != nil {
		held.Close()
		return nil, fmt.Errorf("simulation: open the conversation: %w", err)
	}
	held.created = created
	held.mic = held.edge.Microphone(callerParticipant, audio.PcmData{})

	events, detach := created.Watch()
	held.closer = detach
	go held.collect(events)

	if greeting != "" {
		if err := created.Say(ctx, greeting); err != nil {
			held.Close()
			return nil, fmt.Errorf("simulation: greet: %w", err)
		}
		held.opening = greeting
		held.settled(ctx)
		held.begin()
	}
	return held, nil
}

func (s *spoken) Session() *session.Session { return s.created }

func (s *spoken) Opening() string { return s.opening }

// Say speaks one line into the call and waits for the whole of what comes back.
func (s *spoken) Say(ctx context.Context, text string) (store.SimulationLine, error) {
	s.begin()

	speech, err := s.speak(ctx, text)
	if err != nil {
		return store.SimulationLine{}, err
	}
	// Playing returns when the caller stopped talking, which is the moment every reply is
	// timed from and the earliest the agent could have started answering.
	s.mic.Play(speech)

	started := time.Now()
	ticker := time.NewTicker(replyPoll)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return store.SimulationLine{}, ctx.Err()
		case <-ticker.C:
			answer, meant, quiet, failure, gone := s.state()
			switch {
			case gone:
				return store.SimulationLine{}, errors.New("simulation: the call ended before the agent answered")
			// Out loud there is a third thing to wait for: the agent may have written the
			// whole reply and still be several seconds from the end of saying it.
			case answer != "" && quiet >= spokenSettle && !s.created.Busy() && !s.edge.Talking():
				return store.SimulationLine{Text: answer, Intended: meant}, nil
			case time.Since(started) >= s.within:
				if failure != nil {
					return store.SimulationLine{}, fmt.Errorf("simulation: the agent did not answer: %w", failure)
				}
				return store.SimulationLine{}, errors.New("simulation: the agent did not answer")
			}
		}
	}
}

// Close hangs up. Leaving stops the microphone as well, and the agent leaves on its own when
// the conversation ends, so this is safe whichever of the two happens first.
func (s *spoken) Close() error {
	if s.edge != nil {
		_ = s.edge.Leave()
	}
	if s.closer != nil {
		s.closer()
	}
	if s.voice != nil {
		_ = s.voice.Close()
	}
	if s.ears != nil {
		_ = s.ears.Close()
	}
	return nil
}

// speak synthesises one line of the caller's speech, at the rate the call carries it.
func (s *spoken) speak(ctx context.Context, text string) (audio.PcmData, error) {
	ctx, cancel := context.WithTimeout(ctx, speakWithin)
	defer cancel()

	id := fmt.Sprintf("caller-%d", time.Now().UnixNano())
	if err := s.voice.Synthesize(tts.Request{ID: id, Text: text, Final: true}); err != nil {
		return audio.PcmData{}, err
	}

	var samples []int16
	rate := stt.SampleRate
	for {
		select {
		case <-ctx.Done():
			return audio.PcmData{}, ctx.Err()
		case event, open := <-s.voice.Events():
			if !open {
				return audio.PcmData{}, errors.New("simulation: the caller's voice closed mid-sentence")
			}
			switch typed := event.(type) {
			case tts.AudioChunk:
				if typed.SynthesisID != id {
					continue
				}
				rate = typed.Audio.SampleRate
				samples = append(samples, typed.Audio.Samples...)
			case tts.SynthesisComplete:
				if typed.SynthesisID != id {
					continue
				}
				if len(samples) == 0 {
					return audio.PcmData{}, errors.New("simulation: the caller's voice said nothing")
				}
				// A provider emits whatever its model produces, and the call carries
				// 16 kHz mono, so this is where the two are reconciled.
				return audio.Resample(
					audio.PcmData{Samples: samples, SampleRate: rate, Channels: 1},
					stt.SampleRate, 1,
				), nil
			case tts.Error:
				if typed.SynthesisID == "" || typed.SynthesisID == id {
					return audio.PcmData{}, typed.Err
				}
			}
		}
	}
}

// overhear is the caller listening to the call. The agent's speech goes into the caller's
// own transcriber, which is what makes this a test of the voice as well as the answer.
func (s *spoken) overhear(pcm audio.PcmData) {
	heard := audio.Resample(pcm, stt.SampleRate, 1)
	if err := s.ears.ProcessAudio(heard, agentParticipant); err != nil {
		s.mu.Lock()
		s.failure = err
		s.mu.Unlock()
	}
}

// listen collects what the caller made of the agent's speech.
func (s *spoken) listen(events <-chan stt.Event) {
	for event := range events {
		switch typed := event.(type) {
		case stt.Transcript:
			s.mu.Lock()
			s.lastAt = time.Now()
			if said := strings.TrimSpace(typed.Text); typed.Final() && said != "" {
				s.heard = append(s.heard, said)
			}
			s.mu.Unlock()
		case stt.Error:
			s.mu.Lock()
			s.lastAt = time.Now()
			s.failure = typed.Err
			s.mu.Unlock()
		}
	}
}

// collect reads what the agent reported. Out loud this is not the answer -- the answer is
// what was heard -- but it is what the agent meant, and it is what says the turn is over.
func (s *spoken) collect(events <-chan session.Event) {
	for event := range events {
		s.mu.Lock()
		s.lastAt = time.Now()
		switch typed := event.(type) {
		case agent.Responded:
			if said := strings.TrimSpace(typed.Text); said != "" {
				s.intended = append(s.intended, said)
			}
		case agent.Error:
			s.failure = typed.Err
		}
		s.mu.Unlock()
	}

	s.mu.Lock()
	s.gone = true
	s.mu.Unlock()
}

func (s *spoken) begin() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.heard = nil
	s.intended = nil
	s.failure = nil
	s.lastAt = time.Now()
}

func (s *spoken) state() (string, string, time.Duration, error, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return strings.Join(s.heard, " "), strings.Join(s.intended, " "),
		time.Since(s.lastAt), s.failure, s.gone
}

// settled waits for the agent to finish whatever it is doing, which is how the greeting is
// kept out of the answer to the first thing the caller says.
func (s *spoken) settled(ctx context.Context) {
	ticker := time.NewTicker(replyPoll)
	defer ticker.Stop()

	deadline := time.Now().Add(s.within)
	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			_, _, quiet, _, _ := s.state()
			if quiet >= spokenSettle && !s.created.Busy() && !s.edge.Talking() {
				return
			}
		}
	}
}

// agentParticipant is who the agent is to the caller's transcriber.
var agentParticipant = stt.Participant{ID: "agent", UserID: "agent", Name: "Agent"}

func target(named, fallback string) string {
	if named != "" {
		return named
	}
	return fallback
}
