package agent

import (
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const (
	defaultCadenceGap   = 350 * time.Millisecond
	defaultCadenceRetry = 700 * time.Millisecond
	// defaultCadenceSettle is how long a transcriber that cannot number its utterances is
	// given to restate one the agent has already answered. Both providers repeat the words
	// they settle on, which without this reads as the caller saying the same thing again
	// and earns them a second answer to the first thing they said. It is a guess at how
	// long a transcriber goes over itself for, and only used where there is nothing better.
	defaultCadenceSettle = 2 * time.Second
)

// candidate is a stable transcript revision worth asking the flow controller about.
type candidate struct {
	ID           string
	Participant  stt.Participant
	Text         string
	Language     string
	Confidence   float64
	STTLatencyMs float64
	ReadyAt      time.Time
}

// cadence decides when an evolving transcript has stayed unchanged long enough to act on.
type cadence struct {
	gap    time.Duration
	retry  time.Duration
	settle time.Duration
	logger *slog.Logger
	ready  chan candidate
	done   chan struct{}

	mu       sync.Mutex
	speakers map[string]*cadenceSpeaker
	closed   bool
}

type cadenceSpeaker struct {
	participant stt.Participant
	text        string
	language    string
	confidence  float64
	latencyMs   float64
	candidateID string
	generation  int64
	timer       *time.Timer
	revisedAt   time.Time
	// utterance is the run of speech the words being gathered came from.
	utterance int64
	// committed is the utterance the agent last acted on, kept so the transcriber's own
	// restatement of it is not mistaken for the caller repeating themselves.
	committed string
	// committedUtterance is which run of speech those words were, and committedAt is when
	// they were answered, for transcribers that cannot say which run they are on.
	committedUtterance int64
	committedAt        time.Time
}

func newCadence(gap, retry, settle time.Duration, logger *slog.Logger) *cadence {
	if gap <= 0 {
		gap = defaultCadenceGap
	}
	if retry <= 0 {
		retry = defaultCadenceRetry
	}
	if settle <= 0 {
		settle = defaultCadenceSettle
	}
	if logger == nil {
		logger = slog.Default()
	}
	return &cadence{
		gap:      gap,
		retry:    retry,
		settle:   settle,
		logger:   logger,
		ready:    make(chan candidate, eventBuffer),
		done:     make(chan struct{}),
		speakers: map[string]*cadenceSpeaker{},
	}
}

// Observe records a transcript revision. It returns a controller decision made stale by
// the new words, if there is one.
func (c *cadence) Observe(transcript stt.Transcript) string {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ""
	}
	current := c.speakerFor(transcript.Participant)
	text := transcript.Text
	if transcript.Mode == stt.ModeDelta {
		text = current.text + transcript.Text
	}
	if strings.TrimSpace(text) == "" {
		return ""
	}

	current.participant = transcript.Participant
	current.language = transcript.Language
	current.confidence = transcript.Confidence
	current.latencyMs = transcript.ProcessingTimeMs
	current.utterance = transcript.Utterance
	if sameWords(current.text, text) {
		return ""
	}
	// Nothing new has been said since the agent answered, so these are the words it
	// answered arriving again as the transcriber settles on them.
	if current.text == "" && sameWords(current.committed, text) && c.restating(current, transcript) {
		c.logger.Debug("ignoring the transcriber restating an answered utterance",
			"participant", transcript.Participant.ID, "text", text,
			"utterance", transcript.Utterance, "since", time.Since(current.committedAt))
		return ""
	}

	superseded := current.candidateID
	current.text = text
	current.candidateID = ""
	current.generation++
	current.revisedAt = time.Now()
	c.scheduleLocked(current, c.gap)
	c.logger.Debug("heard more, waiting for the words to stop changing",
		"participant", transcript.Participant.ID, "mode", transcript.Mode, "text", text,
		"confidence", transcript.Confidence, "gap", c.gap, "superseded", superseded)
	return superseded
}

// Resolve records what became of a candidate. Waiting retries the same words after a
// longer pause; every other decision commits them and starts the next utterance cleanly.
func (c *cadence) Resolve(candidateID string, wait bool) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, current := range c.speakers {
		if current.candidateID != candidateID {
			continue
		}
		current.candidateID = ""
		if wait {
			c.logger.Debug("giving the caller longer to finish",
				"participant", current.participant.ID, "candidate", candidateID, "retry", c.retry)
			c.scheduleLocked(current, c.retry)
		} else {
			current.committed = current.text
			current.committedUtterance = current.utterance
			current.committedAt = time.Now()
			current.text = ""
			current.language = ""
			current.confidence = 0
			current.latencyMs = 0
			if current.timer != nil {
				current.timer.Stop()
				current.timer = nil
			}
		}
		return true
	}
	return false
}

// Active reports the most recently heard participant while words are still evolving.
func (c *cadence) Active() (stt.Participant, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	var latest *cadenceSpeaker
	for _, current := range c.speakers {
		if current.text == "" {
			continue
		}
		if latest == nil || current.revisedAt.After(latest.revisedAt) {
			latest = current
		}
	}
	if latest == nil {
		return stt.Participant{}, false
	}
	return latest.participant, true
}

func (c *cadence) Ready() <-chan candidate { return c.ready }

func (c *cadence) Forget(participant stt.Participant) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if current, ok := c.speakers[participant.ID]; ok && current.timer != nil {
		current.timer.Stop()
	}
	delete(c.speakers, participant.ID)
}

func (c *cadence) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return
	}
	c.closed = true
	close(c.done)
	for _, current := range c.speakers {
		if current.timer != nil {
			current.timer.Stop()
		}
	}
}

func (c *cadence) scheduleLocked(current *cadenceSpeaker, delay time.Duration) {
	if current.timer != nil {
		current.timer.Stop()
	}
	generation := current.generation
	participantID := current.participant.ID
	current.timer = time.AfterFunc(delay, func() {
		c.emit(participantID, generation)
	})
}

func (c *cadence) emit(participantID string, generation int64) {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return
	}
	current, ok := c.speakers[participantID]
	if !ok || current.generation != generation || current.text == "" || current.candidateID != "" {
		c.mu.Unlock()
		return
	}
	waited := time.Since(current.revisedAt)
	current.candidateID = replyPrefix + turnStamp()
	current.timer = nil
	ready := candidate{
		ID:           current.candidateID,
		Participant:  current.participant,
		Text:         strings.TrimSpace(current.text),
		Language:     current.language,
		Confidence:   current.confidence,
		STTLatencyMs: current.latencyMs,
		ReadyAt:      time.Now(),
	}
	c.mu.Unlock()

	c.logger.Debug("the words held still, asking whether to answer them",
		"participant", participantID, "candidate", ready.ID, "text", ready.Text, "waited", waited)

	select {
	case c.ready <- ready:
	case <-c.done:
		c.logger.Debug("dropped a settled turn, the agent is closing", "candidate", ready.ID)
	}
}

// restating reports whether words matching the last ones answered are the transcriber
// going over that utterance again rather than the caller saying the same thing twice.
//
// Where the transcriber numbers its utterances the question is already answered: the same
// run of speech cannot be a second hello, however long the provider spends revising it,
// and a new run saying the same word is somebody genuinely repeating themselves and owed
// an answer. Deepgram Flux will restate a settled word for as long as the track is open,
// so any wall clock is a guess that eventually runs out and lets it be answered twice.
//
// A transcriber that cannot say which run it is on leaves the number zero, and there the
// clock is all there is: words that come back quickly are it settling, and words that come
// back later are taken as newly said.
func (c *cadence) restating(current *cadenceSpeaker, transcript stt.Transcript) bool {
	if transcript.Utterance != 0 && current.committedUtterance != 0 {
		return transcript.Utterance == current.committedUtterance
	}
	return time.Since(current.committedAt) < c.settle
}

func (c *cadence) speakerFor(participant stt.Participant) *cadenceSpeaker {
	current, ok := c.speakers[participant.ID]
	if !ok {
		current = &cadenceSpeaker{participant: participant}
		c.speakers[participant.ID] = current
	}
	return current
}
