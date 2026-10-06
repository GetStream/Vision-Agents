package agent

import (
	"log/slog"
	"strings"
	"sync"
	"time"
	"unicode"

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
	// cadenceFinalGap is how long a final transcript waits instead of the usual gap. A
	// transcriber that finalizes has already decided the caller stopped, so waiting the
	// whole gap again for the words to hold still only delays the answer.
	cadenceFinalGap = 60 * time.Millisecond
)

// candidate is a stable transcript revision worth asking the flow controller about.
type candidate struct {
	ID          string
	Participant stt.Participant
	// Speaker is the voice the transcriber heard, for the ones that tell voices apart. It
	// is how a second person at the caller's microphone is told from the caller.
	Speaker      string
	Text         string
	Language     string
	Confidence   float64
	STTLatencyMs float64
	RevisedAt    time.Time
	ReadyAt      time.Time
	// Unfinished says the words are a provisional revision of an utterance still in
	// progress, put to the controller to decide the floor rather than settled.
	Unfinished bool
	// Revision identifies the transcript words the candidate was settled on, independently of
	// the ID, which changes each time the same words are put again after a Wait. It is zero
	// for a candidate that did not come from the cadence.
	Revision uint64
}

type cadenceTimer interface {
	Stop() bool
}

// cadence decides when an evolving transcript has stayed unchanged long enough to act on.
type cadence struct {
	gap    time.Duration
	retry  time.Duration
	settle time.Duration
	logger *slog.Logger
	ready  chan candidate
	done   chan struct{}
	after  func(time.Duration, func()) cadenceTimer

	mu         sync.Mutex
	speakers   map[string]*cadenceSpeaker
	timerEpoch int64
	// revision numbers the words each speaker is heard to settle on, across speakers, so no
	// two revisions share a number.
	revision uint64
	// grace is extra settling time owed to the next turn, whoever says it. It lasts until
	// a turn has been put rather than until the next revision, so every revision of that
	// turn is given it and not only the first.
	grace  time.Duration
	closed bool
}

type cadenceSpeaker struct {
	participant stt.Participant
	// speaker is the diarised voice the words were last heard in.
	speaker     string
	text        string
	language    string
	confidence  float64
	latencyMs   float64
	candidateID string
	generation  int64
	timerEpoch  int64
	timer       cadenceTimer
	revisedAt   time.Time
	// utterance is the run of speech the words being gathered came from.
	utterance int64
	// revision is the number of the words as they stand, which only changes when they do.
	revision uint64
	// carried is what was still unanswered when the transcriber started this utterance.
	// Every revision of the utterance replaces only its own words, so it is put back in
	// front of each one rather than only the first.
	carried string
	// committed is the utterance the agent last acted on, kept so the transcriber's own
	// restatement of it is not mistaken for the caller repeating themselves.
	committed string
	// committedUtterance is which run of speech those words were, and committedAt is when
	// they were answered, for transcribers that cannot say which run they are on.
	committedUtterance int64
	committedAt        time.Time
	emittedGeneration  int64
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
		gap:    gap,
		retry:  retry,
		settle: settle,
		logger: logger,
		ready:  make(chan candidate, eventBuffer),
		done:   make(chan struct{}),
		after: func(delay time.Duration, fn func()) cadenceTimer {
			return time.AfterFunc(delay, fn)
		},
		speakers: map[string]*cadenceSpeaker{},
	}
}

// Observe records a transcript revision.
//
// It returns a controller decision made stale by the new words, if there is one, and what
// the participant is now saying: the whole of it, assembled from the deltas of a provider
// that sends them, and empty when the revision changed nothing.
func (c *cadence) Observe(transcript stt.Transcript) (superseded string, saying string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return "", ""
	}
	current := c.speakerFor(transcript.Participant)
	text := transcript.Text
	if transcript.Mode == stt.ModeDelta {
		text = current.text + transcript.Text
	}
	if strings.TrimSpace(text) == "" {
		return "", ""
	}

	newUtterance := transcript.Utterance != 0 && current.utterance != 0 &&
		transcript.Utterance != current.utterance
	if newUtterance {
		current.carried = ""
		if current.text != "" && !revisesTranscript(current.text, text) {
			// A new utterance that is not a revision of the words in flight, which is how a
			// transcriber splitting "7:30" into "7:00." and "thirty" arrives. Keep both so
			// the next answer is about everything the caller said, not only the tail.
			current.carried = strings.TrimSpace(current.text)
		}
	}
	if current.carried != "" && transcript.Mode != stt.ModeDelta &&
		!strings.HasPrefix(words(text), words(current.carried)) {
		text = current.carried + " " + strings.TrimSpace(text)
	}

	current.participant = transcript.Participant
	// A transcriber names the voice part way through a turn, so the last word on it is
	// the one to keep: an early revision that had nothing to say about who was talking
	// should not erase what a later one worked out.
	if transcript.Speaker != "" {
		current.speaker = transcript.Speaker
	}
	current.language = transcript.Language
	current.confidence = transcript.Confidence
	current.latencyMs = transcript.ProcessingTimeMs
	current.utterance = transcript.Utterance
	final := transcript.Mode == stt.ModeFinal && !incompleteIdentifier(text)
	if sameWords(current.text, text) {
		// The transcriber finalizing words already waited on means they have stopped, so
		// the wait is cut short. It is never lengthened: a final that arrives late must
		// not hold back words that already held still.
		if final && current.timer != nil && current.candidateID == "" {
			c.scheduleLocked(current, c.finalGapLocked())
		}
		return "", ""
	}
	// Nothing new has been said since the agent answered, so these are the words it
	// answered arriving again as the transcriber settles on them.
	if current.text == "" && c.restating(current, transcript, text) {
		c.logger.Debug("ignoring the transcriber restating an answered utterance",
			"participant", transcript.Participant.ID, "text", text,
			"utterance", transcript.Utterance, "since", time.Since(current.committedAt))
		return "", ""
	}

	superseded = current.candidateID
	current.text = text
	current.candidateID = ""
	current.generation++
	c.revision++
	current.revision = c.revision
	current.revisedAt = time.Now()
	delay := c.gap + c.grace
	if final {
		delay = c.finalGapLocked()
	}
	if incompleteIdentifier(text) {
		// Member IDs, PINs and clock times arrive a digit at a time. Answering
		// "ABC12345" 350ms before the last 6 is how verify_identity got the wrong id.
		delay = c.retry
	}
	c.scheduleLocked(current, delay)
	c.logger.Debug("heard more, waiting for the words to stop changing",
		"participant", transcript.Participant.ID, "mode", transcript.Mode, "text", text,
		"confidence", transcript.Confidence, "gap", delay, "superseded", superseded)
	return superseded, strings.TrimSpace(text)
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
			current.carried = ""
			current.speaker = ""
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

// Grace gives the next turn longer than usual to hold still, and is spent on it.
//
// What it is for is the turn after somebody was talked over: the line is running late, so
// the words are still arriving when the usual gap says they have stopped. It is a one-off
// rather than a setting, because a call is not slow for having had one collision in it.
func (c *cadence) Grace(extra time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.grace = extra
}

// Active reports the most recently heard participant while words are still evolving, and
// when their words last changed.
func (c *cadence) Active() (stt.Participant, time.Time, bool) {
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
		return stt.Participant{}, time.Time{}, false
	}
	return latest.participant, latest.revisedAt, true
}

func (c *cadence) Ready() <-chan candidate { return c.ready }

func (c *cadence) matchesCandidate(ready candidate) bool {
	_, ok := c.candidateSnapshot(ready)
	return ok
}

func (c *cadence) candidateSnapshot(ready candidate) (candidate, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	current, ok := c.speakers[ready.Participant.ID]
	if !ok || current.candidateID != ready.ID || strings.TrimSpace(current.text) != ready.Text {
		return candidate{}, false
	}
	ready.Participant = current.participant
	ready.Speaker = current.speaker
	ready.Language = current.language
	ready.Confidence = current.confidence
	ready.STTLatencyMs = current.latencyMs
	ready.RevisedAt = current.revisedAt
	ready.Revision = current.revision
	return ready, true
}

func (c *cadence) currentCandidate(participantID string) (candidate, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	current, ok := c.speakers[participantID]
	if !ok || strings.TrimSpace(current.text) == "" {
		return candidate{}, false
	}
	return candidate{
		Participant:  current.participant,
		Speaker:      current.speaker,
		Text:         strings.TrimSpace(current.text),
		Language:     current.language,
		Confidence:   current.confidence,
		STTLatencyMs: current.latencyMs,
		RevisedAt:    current.revisedAt,
		Revision:     current.revision,
	}, true
}

// ExpediteFinal emits the current transcript immediately when the agent has independently
// checked that a finalized candidate is eligible for primary EOT scoring. The generation
// checks preserve the ordinary cadence retry after a low score and reject stale revisions.
func (c *cadence) ExpediteFinal(transcript stt.Transcript) bool {
	if !transcript.Final() || transcript.Participant.ID == "" {
		return false
	}

	c.mu.Lock()
	current, ok := c.speakers[transcript.Participant.ID]
	if !ok || current.text == "" || current.candidateID != "" || current.timer == nil ||
		current.emittedGeneration == current.generation || c.grace > 0 ||
		incompleteIdentifier(current.text) || !sameWords(current.text, transcript.Text) {
		c.mu.Unlock()
		return false
	}

	current.timer.Stop()
	current.timer = nil
	generation := current.generation
	timerEpoch := c.nextTimerEpochLocked()
	current.timerEpoch = timerEpoch
	participantID := current.participant.ID
	c.mu.Unlock()

	c.emit(participantID, generation, timerEpoch)
	return true
}

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

// finalGapLocked is the wait for words a transcriber has finalized, with any grace still owed
// after an overlap. The caller holds the lock.
func (c *cadence) finalGapLocked() time.Duration {
	return min(c.gap, cadenceFinalGap) + c.grace
}

func (c *cadence) scheduleLocked(current *cadenceSpeaker, delay time.Duration) {
	if current.timer != nil {
		current.timer.Stop()
	}
	generation := current.generation
	timerEpoch := c.nextTimerEpochLocked()
	current.timerEpoch = timerEpoch
	participantID := current.participant.ID
	current.timer = c.after(delay, func() {
		c.emit(participantID, generation, timerEpoch)
	})
}

func (c *cadence) nextTimerEpochLocked() int64 {
	c.timerEpoch++
	return c.timerEpoch
}

func (c *cadence) emit(participantID string, generation, timerEpoch int64) {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return
	}
	current, ok := c.speakers[participantID]
	if !ok || current.generation != generation || current.timerEpoch != timerEpoch ||
		current.text == "" || current.candidateID != "" {
		c.mu.Unlock()
		return
	}
	waited := time.Since(current.revisedAt)
	current.candidateID = replyPrefix + turnStamp()
	current.emittedGeneration = generation
	current.timer = nil
	c.grace = 0
	ready := candidate{
		ID:           current.candidateID,
		Participant:  current.participant,
		Speaker:      current.speaker,
		Text:         strings.TrimSpace(current.text),
		Language:     current.language,
		Confidence:   current.confidence,
		STTLatencyMs: current.latencyMs,
		RevisedAt:    current.revisedAt,
		ReadyAt:      time.Now(),
		Revision:     current.revision,
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
func (c *cadence) restating(current *cadenceSpeaker, transcript stt.Transcript, text string) bool {
	if current.committed == "" {
		return false
	}
	if transcript.Utterance != 0 && current.committedUtterance != 0 {
		if transcript.Utterance != current.committedUtterance {
			return false
		}
		if growsTranscript(current.committed, text) {
			return false
		}
		// The words a transcriber settles on need not be the words it streamed: Gemini
		// writes an order number as "1 2 3" while the caller is talking and "one two
		// three" when it commits. Asking for the same words back would let one reading of
		// the number through as a second turn, and the caller is asked for it twice.
		return transcript.Mode == stt.ModeFinal || sameWords(current.committed, text)
	}
	return sameWords(current.committed, text) && time.Since(current.committedAt) < c.settle
}

func (c *cadence) speakerFor(participant stt.Participant) *cadenceSpeaker {
	current, ok := c.speakers[participant.ID]
	if !ok {
		current = &cadenceSpeaker{participant: participant}
		c.speakers[participant.ID] = current
	}
	return current
}

// revisesTranscript reports whether next is the same run of speech as previous, restated
// or grown, rather than a second utterance that happens to have arrived next.
func revisesTranscript(previous, next string) bool {
	prev := words(previous)
	nxt := words(next)
	if prev == "" || nxt == "" {
		return true
	}
	return strings.HasPrefix(nxt, prev) || strings.HasPrefix(prev, nxt)
}

// growsTranscript reports whether next is previous with more words or a longer last token,
// which is how "ABC12345" becomes "ABC123456" after the agent already answered the short
// form.
func growsTranscript(previous, next string) bool {
	prev := strings.ToLower(words(previous))
	nxt := strings.ToLower(words(next))
	return prev != "" && nxt != prev && strings.HasPrefix(nxt, prev)
}

// incompleteIdentifier reports whether the last token still looks like a PIN, member ID,
// phone fragment, or clock time that the transcriber may grow.
func incompleteIdentifier(text string) bool {
	fields := strings.Fields(words(text))
	if len(fields) == 0 {
		return false
	}
	last := fields[len(fields)-1]
	if len(last) < 2 || len(last) > 16 {
		return false
	}
	hasDigit := false
	for _, symbol := range last {
		if unicode.IsDigit(symbol) {
			hasDigit = true
			continue
		}
		if !unicode.IsLetter(symbol) {
			return false
		}
	}
	return hasDigit
}
