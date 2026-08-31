//go:build integration

// Conversation quality, measured the way a caller experiences it.
//
// These are not tests of the agent's parts: every provider is real, the caller's voice is
// synthesised and played into the call at the rate a real call delivers it, and what is
// asserted is what a person on the phone would notice. Whether the agent answered, whether
// it answered once, whether it waited its turn, and how long it kept them waiting.
//
// The thresholds below are the bar the agent is held to rather than a description of what
// it does today. A failure here is a quality regression to explain, not a number to relax.
package agent_test

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testaudio"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// The bar a conversation with this agent is held to.
const (
	// replyGapFloor is the shortest pause after someone stops talking that still sounds
	// like listening rather than cutting in.
	replyGapFloor = 300 * time.Millisecond
	// replyGapCeiling is where waiting to be answered starts to sound like a machine
	// thinking rather than a person replying.
	replyGapCeiling = 700 * time.Millisecond
	// acknowledgedWithin is how long a caller waits to hear anything at all after asking
	// for something the agent has to go away and work out.
	acknowledgedWithin = 2 * time.Second
	// turnSpikeFactor is how much slower than the median turn the slowest turn in a
	// conversation may be. A conversation that is uniformly slow still feels like a
	// conversation; one that stalls every so often does not.
	turnSpikeFactor = 2.5
)

// How long real providers are given, and how a call is paced.
const (
	answerWithin = 30 * time.Second
	lookupWithin = 60 * time.Second
	suiteWithin  = 15 * time.Minute
	// chunkDuration is how much audio a call carries per packet.
	chunkDuration = 20 * time.Millisecond
	// playoutAhead is how much of the agent's speech may be waiting to be heard, which is
	// what the real edge allows before it makes the agent wait.
	playoutAhead = 400 * time.Millisecond
	// overlapWithin is how long the agent is given to decide what to do about somebody
	// talking while it is talking.
	overlapWithin = 5 * time.Second
	// settlePause is how long the conversation has to stay quiet before a test believes
	// the agent has finished having its say.
	settlePause = 2 * time.Second
	// conversationPause is the gap a caller leaves before asking the next thing.
	conversationPause = 4 * time.Second
	// roomAmplitude is how loud the room is under the caller's voice in the noisy call.
	roomAmplitude = 0.12
)

// The instructions each scenario runs the agent under.
const (
	conciseAssistant = "You are a helpful voice assistant. Keep your answers to one or " +
		"two sentences."
	attentiveAssistant = "You are a helpful voice assistant talking to Alex. Keep your " +
		"answers to one or two sentences. Other voices in the room are not talking to you."
	thoroughAssistant = "You are a helpful voice assistant. When asked to explain " +
		"something, take three or four sentences over it, so there is something to " +
		"interrupt."
	delegatingAssistant = "You are a helpful voice assistant. Keep your answers to one " +
		"or two sentences. For anything that has to be worked out or looked up, tell the " +
		"caller you are checking and hand the work to the think skill. Never work it out " +
		"yourself."
	navigatingCaller = "You have called Northwind about a billing error on your account. " +
		"You need their account support team."
)

var (
	caller    = stt.Participant{ID: "caller", UserID: "caller", Name: "Alex"}
	bystander = stt.Participant{ID: "bystander", UserID: "bystander", Name: "Sam"}
)

type ConversationSuite struct {
	suite.Suite
	ctx context.Context

	llm *llmrouter.Router
	stt *sttrouter.Router
	tts *ttsrouter.Router

	// voice synthesises what the caller says, cached because the same sentence is said in
	// more than one scenario.
	voice  *ttsrouter.Session
	mu     sync.Mutex
	spoken map[string]audio.PcmData
	said   int

	// long is a two minute conversation, held once and asserted on by everything that
	// needs one.
	longOnce sync.Once
	long     *longCall
}

func TestConversationSuite(t *testing.T) {
	suite.Run(t, new(ConversationSuite))
}

func (s *ConversationSuite) SetupSuite() {
	// The capability shortcuts below resolve to these, which is also what cmd/agent uses
	// out of the box.
	for _, name := range []string{"DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY", "BASETEN_API_KEY"} {
		if os.Getenv(name) == "" {
			s.T().Skip(name + " not set")
		}
	}
	if !testaudio.HasFFmpeg() {
		s.T().Skip("ffmpeg not available to resample the caller's voice")
	}

	var cancel context.CancelFunc
	s.ctx, cancel = context.WithTimeout(context.Background(), suiteWithin)
	s.T().Cleanup(cancel)
	s.spoken = map[string]audio.PcmData{}

	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	logger := slog.New(slog.DiscardHandler)

	s.llm, err = llmrouter.New(llmrouter.Options{
		Config: config[routing.LLM], Registry: llmrouter.DefaultRegistry(), Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(s.llm.Close)

	s.stt, err = sttrouter.New(sttrouter.Options{
		Config: config[routing.STT], Registry: sttrouter.DefaultRegistry(), Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(s.stt.Close)

	s.tts, err = ttsrouter.New(ttsrouter.Options{
		Config: config[routing.TTS], Registry: ttsrouter.DefaultRegistry(), Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(s.tts.Close)

	s.voice, err = s.tts.Start(s.ctx, ttsrouter.Request{
		CustomerID: "conversation-quality", Target: "en-low-latency",
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = s.voice.Close() })
}

// speech is what a sentence sounds like, at the rate a call delivers it.
func (s *ConversationSuite) speech(text string) audio.PcmData {
	s.mu.Lock()
	defer s.mu.Unlock()

	if cached, ok := s.spoken[text]; ok {
		return cached
	}
	s.said++
	id := fmt.Sprintf("caller-%d", s.said)
	s.Require().NoError(s.voice.Synthesize(tts.Request{ID: id, Text: text, Final: true}))

	var samples []int16
	rate, channels := 0, 1
	for {
		select {
		case event := <-s.voice.Events():
			switch typed := event.(type) {
			case tts.AudioChunk:
				rate, channels = typed.Audio.SampleRate, typed.Audio.Channels
				samples = append(samples, typed.Audio.Samples...)
			case tts.SynthesisComplete:
				spoken, err := testaudio.Resample16kMono(audio.PcmData{
					Samples: samples, SampleRate: rate, Channels: channels,
				})
				s.Require().NoError(err)
				s.Require().NotEmpty(spoken.Samples, "the caller's voice said nothing")
				s.spoken[text] = spoken
				return spoken
			case tts.Error:
				s.FailNowf("the caller's voice failed", "%v", typed.Err)
			}
		case <-s.ctx.Done():
			s.FailNow("the caller's voice timed out")
		}
	}
}

// setup is what a scenario needs of the agent beyond the defaults.
type setup struct {
	instructions string
	// noisy puts a room under the caller's voice.
	noisy bool
	// delegate gives the agent a subagent and the built-in skills, which is what makes
	// going away to work something out possible.
	delegate bool
	// navigating puts the agent on a call it placed, with a keypad to press and somebody
	// to hand the call to.
	navigating bool
}

// call starts a conversation: a real agent on one end and a played caller on the other.
//
// Backchannels are left off, because a murmur is published audio too and every measurement
// here is about when the agent started answering.
func (s *ConversationSuite) call(with setup) *call {
	edge := newPacedEdge()
	options := agent.Options{
		Edge:         edge,
		Instructions: with.instructions,
		CustomerID:   "conversation-quality",
		AgentID:      fmt.Sprintf("conversation-%d", time.Now().UnixNano()),
		CallID:       "conversation-quality",
		LLM:          s.llm,
		LLMTarget:    "llm-fast",
		STT:          s.stt,
		STTTarget:    "en-low-latency",
		TTS:          s.tts,
		TTSTarget:    "en-low-latency",
		Logger:       slog.New(slog.DiscardHandler),
	}
	if with.delegate {
		skills, err := harness.DefaultSkills()
		s.Require().NoError(err)
		options.Skills = skills
		options.SubagentTarget = "en-high-accuracy"
	}
	var keypad *recordingLine
	if with.navigating {
		tools, err := harness.DefaultTools()
		s.Require().NoError(err)
		keypad = &recordingLine{}
		options.Tools = tools
		options.Telephony = keypad
		options.Instructions = agent.NavigatingInstructions + "\n\n" + with.instructions
	}

	voiceAgent, err := agent.New(options)
	s.Require().NoError(err)
	s.Require().NoError(voiceAgent.Join(s.ctx))

	room := audio.PcmData{}
	if with.noisy {
		room = testaudio.Noise(1000, roomAmplitude)
	}
	held := &call{
		suite:     s,
		agent:     voiceAgent,
		edge:      edge,
		log:       newEventLog(voiceAgent),
		room:      room,
		keypad:    keypad,
		primary:   newMicrophone(edge.inbound, caller, room),
		secondary: newMicrophone(edge.inbound, bystander, audio.PcmData{}),
	}
	s.T().Cleanup(func() {
		held.primary.stop()
		held.secondary.stop()
		_ = voiceAgent.Close()
	})
	return held
}

// call is one conversation: what was played into it, and everything the agent did in
// response, with the timings a caller would have felt.
type call struct {
	suite     *ConversationSuite
	agent     *agent.Agent
	edge      *pacedEdge
	log       *eventLog
	room      audio.PcmData
	keypad    *recordingLine
	primary   *microphone
	secondary *microphone
}

// recordingLine is a phone line that keeps what it was asked to do rather than doing it,
// so a scenario can hear a menu through real transcription and still assert on the digit.
type recordingLine struct {
	mu          sync.Mutex
	pressed     []string
	transferred []string
}

func (l *recordingLine) Transfer(_ context.Context, to string) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.transferred = append(l.transferred, to)
	return nil
}

func (l *recordingLine) SendDigits(_ context.Context, digits string) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.pressed = append(l.pressed, digits)
	return nil
}

func (l *recordingLine) keys() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.pressed...)
}

// says plays the caller saying something, and returns the moment they stopped talking,
// which is what every reply is timed from.
func (c *call) says(text string) time.Time {
	speech := c.suite.speech(text)
	if len(c.room.Samples) > 0 {
		speech = testaudio.Mix(speech, c.room, 1)
	}
	return c.primary.play(speech)
}

// bystanderSays plays somebody else in the room, on their own track, the way another
// participant arrives in a real call.
func (c *call) bystanderSays(text string) time.Time {
	return c.secondary.play(c.suite.speech(text))
}

// answer waits for the reply to something said at the given moment.
func (c *call) answer(after time.Time, within time.Duration) agent.Responded {
	var answered agent.Responded
	c.awaits("an answer", within, func() bool {
		replies := since[agent.Responded](c.log, after)
		if len(replies) == 0 {
			return false
		}
		answered = replies[0]
		return true
	})
	return answered
}

// finishes waits for the agent to reach the end of what it had to say, which is not the
// same as the end of the reply: the words are written long before they are all heard.
func (c *call) finishes(after time.Time, within time.Duration) agent.Turn {
	var finished agent.Turn
	c.awaits("the agent to finish talking", within, func() bool {
		spoken := since[agent.Turn](c.log, after)
		if len(spoken) == 0 {
			return false
		}
		finished = spoken[0]
		return true
	})
	return finished
}

// spokeAfter waits for the agent to start making a sound after the given moment.
func (c *call) spokeAfter(after time.Time, within time.Duration) time.Time {
	var spoke time.Time
	c.awaits("the agent to say something", within, func() bool {
		heard, ok := c.edge.spokeAfter(after)
		spoke = heard
		return ok
	})
	return spoke
}

// settles waits for the conversation to go quiet, so what is counted afterwards is
// everything the agent did rather than everything it had done so far.
func (c *call) settles() {
	deadline := time.Now().Add(answerWithin)
	for time.Now().Before(deadline) {
		before := c.log.len()
		time.Sleep(settlePause)
		if c.log.len() == before {
			return
		}
	}
}

func (c *call) awaits(what string, within time.Duration, condition func() bool) {
	c.suite.Require().Eventuallyf(condition, within, 50*time.Millisecond,
		"waited %v for %s\n%s", within, what, c.transcript())
}

// failures are the errors the agent reported, which a caller hears as a turn that went
// nowhere.
func (c *call) failures() []agent.Error {
	return since[agent.Error](c.log, time.Time{})
}

// transcript renders the conversation, which is what makes a quality failure readable.
func (c *call) transcript() string {
	var lines strings.Builder
	for _, entry := range c.log.entries() {
		switch typed := entry.event.(type) {
		case agent.Heard:
			fmt.Fprintf(&lines, "  %s: %s\n", typed.Participant.Name, typed.Text)
		case agent.Responded:
			fmt.Fprintf(&lines, "  agent: %s\n", typed.Text)
		case agent.Delegated:
			fmt.Fprintf(&lines, "  (handed %s to the subagent)\n", typed.Skill)
		case agent.TaskSettled:
			fmt.Fprintf(&lines, "  (%s came back after %.0fms: %s%s)\n",
				typed.Skill, typed.ElapsedMs, typed.Text, typed.Question)
		case agent.TaskCancelled:
			fmt.Fprintf(&lines, "  (%s abandoned: %s)\n", typed.Skill, typed.Reason)
		case agent.Interrupted:
			fmt.Fprintf(&lines, "  (interrupted by %s)\n", typed.Participant.Name)
		case agent.OverlapDecided:
			fmt.Fprintf(&lines, "  (overlap: %s)\n", typed.Action)
		case agent.Error:
			fmt.Fprintf(&lines, "  (%s failed: %v)\n", typed.Context, typed.Err)
		}
	}
	return lines.String()
}

// microphone is one participant's end of the call. It always sends something, because a
// real call carries the room even when nobody is talking, and utterances are spliced into
// that stream.
type microphone struct {
	inbound     chan<- agent.InboundAudio
	participant stt.Participant
	// room is looped between utterances: silence in a quiet call, noise in a busy one.
	room audio.PcmData

	mu      sync.Mutex
	playing *playback
	offset  int

	halt     chan struct{}
	haltOnce sync.Once
	stopped  chan struct{}
}

type playback struct {
	samples  []int16
	at       int
	finished chan time.Time
}

func newMicrophone(
	inbound chan<- agent.InboundAudio,
	participant stt.Participant,
	room audio.PcmData,
) *microphone {
	sending := &microphone{
		inbound:     inbound,
		participant: participant,
		room:        room,
		halt:        make(chan struct{}),
		stopped:     make(chan struct{}),
	}
	go sending.run()
	return sending
}

// play splices an utterance into the stream and returns once all of it has been sent,
// which is the moment the speaker stopped talking.
func (m *microphone) play(pcm audio.PcmData) time.Time {
	finished := make(chan time.Time, 1)
	m.mu.Lock()
	m.playing = &playback{samples: pcm.Samples, finished: finished}
	m.mu.Unlock()

	select {
	case at := <-finished:
		return at
	case <-m.halt:
		return time.Now()
	}
}

// stop hangs up. It returns once the track is really finished, so the call can be torn
// down without a packet arriving on a channel that has been closed.
func (m *microphone) stop() {
	m.haltOnce.Do(func() { close(m.halt) })
	<-m.stopped
}

func (m *microphone) run() {
	ticker := time.NewTicker(chunkDuration)
	defer ticker.Stop()
	defer close(m.stopped)

	for {
		select {
		case <-m.halt:
			return
		case <-ticker.C:
			select {
			case m.inbound <- agent.InboundAudio{Participant: m.participant, Audio: m.next()}:
			case <-m.halt:
				return
			}
		}
	}
}

// next is the next packet of this participant's track.
func (m *microphone) next() audio.PcmData {
	size := stt.SampleRate * int(chunkDuration/time.Millisecond) / 1000
	samples := make([]int16, size)

	m.mu.Lock()
	defer m.mu.Unlock()

	if playing := m.playing; playing != nil {
		copied := copy(samples, playing.samples[playing.at:])
		playing.at += copied
		if playing.at >= len(playing.samples) {
			playing.finished <- time.Now()
			m.playing = nil
		}
	} else if len(m.room.Samples) > 0 {
		for i := range samples {
			samples[i] = m.room.Samples[(m.offset+i)%len(m.room.Samples)]
		}
		m.offset += size
	}
	return audio.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
}

// pacedEdge is a call with no network in it: the microphones write 16 kHz PCM in, and
// whatever the agent says back is timestamped, because when it started talking is the
// measurement.
type pacedEdge struct {
	inbound chan agent.InboundAudio

	mu sync.Mutex
	// playhead is when the speech published so far will have been heard.
	playhead time.Time
	spokenAt []time.Time
	leftOnce sync.Once
}

func newPacedEdge() *pacedEdge {
	return &pacedEdge{inbound: make(chan agent.InboundAudio, 64)}
}

func (e *pacedEdge) Join(context.Context) error { return nil }

func (e *pacedEdge) Audio() <-chan agent.InboundAudio { return e.inbound }

// PublishAudio takes the agent's speech at the rate it is heard rather than the rate it is
// synthesised, which is what the real edge does. An edge that swallowed a reply whole would
// leave the agent believing it had finished talking seconds before the caller had heard it,
// and nothing about taking turns would mean anything.
func (e *pacedEdge) PublishAudio(pcm audio.PcmData) error {
	e.mu.Lock()
	if e.playhead.Before(time.Now()) {
		e.playhead = time.Now()
	}
	heardAt := e.playhead
	e.playhead = e.playhead.Add(time.Duration(pcm.DurationMs()) * time.Millisecond)
	e.spokenAt = append(e.spokenAt, heardAt)
	queued := time.Until(e.playhead) - playoutAhead
	e.mu.Unlock()

	if queued > 0 {
		time.Sleep(queued)
	}
	return nil
}

// Leave hangs up. Closing the channel is what tells the agent the call is over, so the
// microphones have to be stopped before this.
func (e *pacedEdge) Leave() error {
	e.leftOnce.Do(func() { close(e.inbound) })
	return nil
}

// talking reports whether the agent still has speech left to be heard, which is what
// holding the floor means. Chunk timestamps cannot answer this: a provider hands over a
// whole sentence at a time, so a talking agent can be several seconds from its next chunk.
func (e *pacedEdge) talking() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return time.Now().Before(e.playhead)
}

// spokeAfter is when the agent first made a sound after the given moment.
func (e *pacedEdge) spokeAfter(after time.Time) (time.Time, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()

	for _, at := range e.spokenAt {
		if at.After(after) {
			return at, true
		}
	}
	return time.Time{}, false
}

// eventLog is everything the agent reported, and when.
type eventLog struct {
	mu      sync.Mutex
	stamped []entry
}

type entry struct {
	at    time.Time
	event agent.Event
}

func newEventLog(voiceAgent *agent.Agent) *eventLog {
	log := &eventLog{}
	go func() {
		for event := range voiceAgent.Events() {
			log.mu.Lock()
			log.stamped = append(log.stamped, entry{at: time.Now(), event: event})
			log.mu.Unlock()
		}
	}()
	return log
}

func (l *eventLog) entries() []entry {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]entry(nil), l.stamped...)
}

func (l *eventLog) len() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return len(l.stamped)
}

// since returns the events of one kind that arrived after a moment.
func since[E agent.Event](log *eventLog, after time.Time) []E {
	var found []E
	for _, entry := range log.entries() {
		if !entry.at.After(after) {
			continue
		}
		if typed, ok := entry.event.(E); ok {
			found = append(found, typed)
		}
	}
	return found
}

func (s *ConversationSuite) TestAQuestionInAQuietRoomIsAnswered() {
	held := s.call(setup{instructions: conciseAssistant})

	asked := held.says("What is the capital of France?")
	answer := held.answer(asked, answerWithin)

	s.Containsf(strings.ToLower(answer.Text), "paris", "answered instead:\n%s", held.transcript())
	held.settles()
	s.Lenf(since[agent.Responded](held.log, asked), 1,
		"one question earns one answer\n%s", held.transcript())
	s.Emptyf(held.failures(), "a clean exchange should not fail\n%s", held.transcript())
}

func (s *ConversationSuite) TestAQuestionOverRoomNoiseIsAnswered() {
	held := s.call(setup{instructions: conciseAssistant, noisy: true})

	asked := held.says("What is the capital of France?")
	answer := held.answer(asked, answerWithin)

	s.Containsf(strings.ToLower(answer.Text), "paris",
		"a noisy room should not cost the caller their answer\n%s", held.transcript())
}

func (s *ConversationSuite) TestABackgroundConversationIsNotAnswered() {
	held := s.call(setup{instructions: attentiveAssistant})

	started := time.Now()
	held.bystanderSays("So then I told her we would just take the train instead of driving.")
	held.bystanderSays("Yeah, and the tickets turned out to be cheaper anyway.")
	held.settles()

	s.Emptyf(since[agent.Responded](held.log, started),
		"somebody else's conversation is not the agent's to join\n%s", held.transcript())

	// Ignoring the room is only right if the agent is still listening to the caller.
	asked := held.says("What is the capital of France?")
	s.Containsf(strings.ToLower(held.answer(asked, answerWithin).Text), "paris",
		"the agent stopped hearing the caller\n%s", held.transcript())
}

func (s *ConversationSuite) TestLookingSomethingUpDoesNotBlockTheConversation() {
	held := s.call(setup{instructions: delegatingAssistant, delegate: true})

	asked := held.says("Can you work out what fifteen percent of eighty four point two is?")

	// The caller hears something long before the answer exists.
	spoke := held.spokeAfter(asked, answerWithin)
	s.LessOrEqualf(spoke.Sub(asked), acknowledgedWithin,
		"the caller waited %v to hear anything at all\n%s", spoke.Sub(asked), held.transcript())
	held.awaits("the work to be handed over", answerWithin, func() bool {
		return len(since[agent.Delegated](held.log, asked)) > 0
	})
	filler := held.answer(asked, answerWithin)
	s.NotContainsf(filler.Text, "12.6",
		"the agent said the answer it had not worked out yet\n%s", held.transcript())

	// The conversation carries on while the subagent works, which is the point of
	// handing the work over at all.
	interjected := held.says("Take your time.")
	held.answer(interjected, answerWithin)

	held.awaits("the answer to come back", lookupWithin, func() bool {
		for _, settled := range since[agent.TaskSettled](held.log, asked) {
			if settled.Err == nil && strings.Contains(settled.Text, "12.6") {
				return true
			}
		}
		return false
	})
}

func (s *ConversationSuite) TestABriefAcknowledgementDoesNotStopTheAgent() {
	held := s.call(setup{instructions: thoroughAssistant})

	asked := held.says("Explain how the gears on a bicycle work.")
	held.spokeAfter(asked, answerWithin)
	acknowledged := held.says("mm hmm, right")

	// The murmur has to land while the agent is still talking for this to mean anything.
	s.Require().Emptyf(since[agent.Turn](held.log, asked),
		"the agent had already finished answering, so nothing was talked over\n%s", held.transcript())
	time.Sleep(overlapWithin)

	s.Emptyf(since[agent.Interrupted](held.log, acknowledged),
		"a listening noise is not an interruption\n%s", held.transcript())
	s.Truef(held.edge.talking(), "the agent gave up the floor over a murmur\n%s", held.transcript())
}

func (s *ConversationSuite) TestTalkingOverTheAgentWithAChangeOfDirectionStopsIt() {
	held := s.call(setup{instructions: thoroughAssistant})

	asked := held.says("Explain how the gears on a bicycle work.")
	held.spokeAfter(asked, answerWithin)
	changed := held.says("Actually, forget the bicycle. What is the capital of Spain?")

	s.Require().Emptyf(since[agent.Turn](held.log, asked),
		"the agent had already finished answering, so nothing was talked over\n%s", held.transcript())
	held.awaits("the agent to stop and listen", answerWithin, func() bool {
		return len(since[agent.Interrupted](held.log, changed)) > 0
	})
	s.Containsf(strings.ToLower(held.answer(changed, answerWithin).Text), "madrid",
		"the agent stopped talking but answered the wrong question\n%s", held.transcript())
}

func (s *ConversationSuite) TestTheAgentWaitsAHumanIntervalBeforeReplying() {
	held := s.call(setup{instructions: conciseAssistant})

	asked := held.says("What is the capital of France?")
	spoke := held.spokeAfter(asked, answerWithin)

	waited := spoke.Sub(asked)
	s.GreaterOrEqualf(waited, replyGapFloor,
		"replying after %v means the agent cut the caller off", waited)
	s.LessOrEqualf(waited, replyGapCeiling,
		"the caller waited %v to be answered\n%s", waited, held.transcript())
}

func (s *ConversationSuite) TestAMenuOptionIsPressedRatherThanSpoken() {
	// The agent hears the menu through real transcription and has to work out that "press
	// one" means the keypad, not the answer "one".
	held := s.call(setup{instructions: navigatingCaller, navigating: true})

	held.says("Thank you for calling Northwind. For sales, press one. " +
		"For account support, press two. To hear these options again, press nine.")

	held.awaits("the agent to answer the menu", answerWithin, func() bool {
		return len(held.keypad.keys()) > 0
	})
	s.Equalf("2", held.keypad.keys()[0],
		"the agent pressed the wrong option\n%s", held.transcript())
}

func (s *ConversationSuite) TestALongMenuIsHeardOutBeforeAnsweringIt() {
	// A recording is not listening, and one that is talked over starts again from the top,
	// so the agent must wait for the whole menu rather than answering the first option.
	held := s.call(setup{instructions: navigatingCaller, navigating: true})

	started := held.says("Thank you for calling Northwind. Please listen carefully as our " +
		"options have changed. For sales, press one. For account support, press two. " +
		"For anything else, press three.")

	s.Emptyf(since[agent.Interrupted](held.log, started),
		"the agent talked over a recording that cannot hear it\n%s", held.transcript())
	held.awaits("the agent to answer the menu", answerWithin, func() bool {
		return len(held.keypad.keys()) > 0
	})
	s.Lenf(held.keypad.keys(), 1,
		"the agent pressed at every option instead of the one it wanted\n%s", held.transcript())
}

func (s *ConversationSuite) TestAPersonAnsweringIsSpokenToRatherThanPressedAt() {
	// A person cannot hear a keypad, so pressing at one wastes the call.
	held := s.call(setup{instructions: navigatingCaller, navigating: true})

	asked := held.says("Northwind account support, this is Dana, how can I help?")
	answer := held.answer(asked, answerWithin)

	s.Emptyf(held.keypad.keys(), "the agent pressed digits at a person\n%s", held.transcript())
	s.NotEmptyf(answer.Text, "the agent said nothing to the person who answered\n%s",
		held.transcript())
}

func (s *ConversationSuite) TestATwoMinuteConversationStaysCoherent() {
	long := s.longConversation()

	for _, spoken := range long.exchanges {
		s.NotEmptyf(spoken.answered, "%q went unanswered\n%s", spoken.asked, long.transcript)
		if spoken.expect == "" {
			continue
		}
		s.Containsf(strings.ToLower(spoken.answered), spoken.expect,
			"the agent lost track of what was established earlier\n%s", long.transcript)
	}
	s.GreaterOrEqual(long.elapsed, 90*time.Second, "that was not a long conversation")
}

func (s *ConversationSuite) TestLatencyStaysSteadyThroughALongConversation() {
	long := s.longConversation()
	s.Require().NotEmpty(long.roundtrips, "no turn was measured")

	sorted := slices.Clone(long.roundtrips)
	slices.Sort(sorted)
	median := sorted[len(sorted)/2]
	slowest := sorted[len(sorted)-1]
	s.T().Logf("%d turns over %v: median %.0fms, quickest %.0fms, slowest %.0fms",
		len(sorted), long.elapsed.Round(time.Second), median, sorted[0], slowest)

	s.LessOrEqualf(slowest, median*turnSpikeFactor,
		"one turn took %.0fms against a median of %.0fms, which a caller hears as a stall\n%s",
		slowest, median, long.transcript)
}

// exchange is one thing the caller says and what came back.
type exchange struct {
	asked string
	// expect is what the answer has to contain, for the questions that are only
	// answerable from earlier in the conversation.
	expect   string
	answered string
}

// longCall is a two minute conversation, held once because holding one costs two minutes.
type longCall struct {
	exchanges  []exchange
	roundtrips []float64
	elapsed    time.Duration
	transcript string
}

// longConversation holds the conversation on first use. A test that needs one when it
// could not be held fails saying so rather than on a conversation that is not there.
func (s *ConversationSuite) longConversation() *longCall {
	s.longOnce.Do(func() { s.hold() })
	s.Require().NotNil(s.long, "the two minute conversation could not be held")
	return s.long
}

func (s *ConversationSuite) hold() {
	held := s.call(setup{instructions: conciseAssistant})
	script := []exchange{
		{asked: "Hello, my name is Alex and I am planning a trip to Lisbon next month."},
		{asked: "What is the weather usually like there in October?"},
		{asked: "I only have three days. What should I not miss?"},
		{asked: "I would rather skip the museums. What else is worth doing?"},
		{asked: "How much Portuguese should I learn before I go?"},
		{asked: "Remind me, what was my name?", expect: "alex"},
		{asked: "And which city did I say I was going to?", expect: "lisbon"},
	}

	started := time.Now()
	var roundtrips []float64
	for i := range script {
		asked := held.says(script[i].asked)
		script[i].answered = held.answer(asked, answerWithin).Text
		// Talking over the answer would be a different scenario, so the caller hears it
		// out and then leaves the pause a person would leave.
		if finished := held.finishes(asked, answerWithin); finished.RoundtripMs > 0 {
			roundtrips = append(roundtrips, finished.RoundtripMs)
		}
		time.Sleep(conversationPause)
	}
	held.settles()

	s.long = &longCall{
		exchanges:  script,
		roundtrips: roundtrips,
		elapsed:    time.Since(started),
		transcript: held.transcript(),
	}
}
