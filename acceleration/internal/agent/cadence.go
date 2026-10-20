package agent

import (
	"log/slog"
	"os"
	"strconv"
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
	// Utterance is the transcriber's identity for this run of speech, when it has one.
	Utterance int64
	// Speaker is the voice the transcriber heard, for the ones that tell voices apart. It
	// is how a second person at the caller's microphone is told from the caller.
	Speaker string
	// Text is the run of speech as the transcriber has it, which is what turn-taking
	// compares revisions against.
	Text string
	// Answered is the start of Text the agent already answered, when the transcriber kept
	// the run open after the answer and went on adding to it. Said is the rest.
	Answered     string
	Language     string
	Confidence   float64
	STTLatencyMs float64
	RevisedAt    time.Time
	ReadyAt      time.Time
	// Unfinished says the words are a provisional revision of an utterance still in
	// progress, put to the controller to decide the floor rather than settled.
	Unfinished bool
	// Ended says the transcriber marked the end of the turn, or never marks one.
	Ended bool
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
	// grace is extra settling time owed to the next turn, whoever says it. It lasts until
	// a turn has been put rather than until the next revision, so every revision of that
	// turn is given it and not only the first.
	grace  time.Duration
	closed bool
	// marksEnds says the transcriber has sent a final, so it is one that says when a turn
	// has ended.
	marksEnds bool
}

type cadenceSpeaker struct {
	participant stt.Participant
	// speaker is the diarised voice the words were last heard in.
	speaker string
	// ended says the words were last sent as final, and askedEnded whether they had been
	// when they were last put.
	ended       bool
	askedEnded  bool
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
	// held is a queued candidate removed from the active transcript until it is answered
	// or superseded by a revision of the same numbered utterance.
	held *heldTranscript
	// carried is what was still unanswered when the transcriber started this utterance.
	// Every revision of the utterance replaces only its own words, so it is put back in
	// front of each one rather than only the first.
	carried string
	// committed is the utterance the agent last acted on, kept so the transcriber's own
	// restatement of it is not mistaken for the caller repeating themselves.
	committed string
	// committedUtterance is which run of speech those words were, and committedAt is when
	// they were answered, for transcribers that cannot say which run they are on.
	committedUtterance  int64
	committedAt         time.Time
	committedGeneration int64
}

// heldTranscript keeps the exact candidate and carried prefix owned by a queue. Its
// generation lets a newer answered utterance keep its deduplication record.
type heldTranscript struct {
	candidate  candidate
	carried    string
	generation int64
	heldAt     time.Time
}

// cadenceGapEnvVar overrides how long unfinished words hold still before they are put. A
// transcriber whose finals come from a voice activity detector rather than a model of
// the turn already ends every pause it trusts, so the gap only has to catch the ones it
// does not, and at 350 ms it answers a caller drawing breath mid-sentence.
const cadenceGapEnvVar = "ROUTER_CADENCE_GAP_MS"

func cadenceGapFromEnv(logger *slog.Logger) time.Duration {
	raw := os.Getenv(cadenceGapEnvVar)
	if raw == "" {
		return 0
	}
	ms, err := strconv.Atoi(raw)
	if err != nil || ms <= 0 {
		logger.Warn("ignoring an unreadable cadence gap", "env", cadenceGapEnvVar, "value", raw)
		return 0
	}
	return time.Duration(ms) * time.Millisecond
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
	if strings.TrimSpace(transcript.Text) == "" && transcript.Mode != stt.ModeDelta {
		return "", ""
	}
	current := c.speakerFor(transcript.Participant)
	var heldRevisionText string
	if held := current.held; held != nil {
		unknownIdentity := transcript.Utterance == 0 || held.candidate.Utterance == 0
		if unknownIdentity && sameWords(held.candidate.Text, transcript.Text) &&
			time.Since(held.heldAt) < c.settle {
			return "", ""
		}
		if transcript.Utterance != 0 && held.candidate.Utterance == transcript.Utterance {
			text := transcript.Text
			if transcript.Mode == stt.ModeDelta {
				text = held.candidate.Text + transcript.Text
			} else if held.carried != "" && !strings.HasPrefix(words(text), words(held.carried)) {
				text = held.carried + " " + strings.TrimSpace(text)
			}
			if sameWords(held.candidate.Text, text) {
				return "", ""
			}
			if current.text != "" && current.utterance != held.candidate.Utterance {
				heldRevisionText = text
			} else {
				current.text = held.candidate.Text
				current.carried = held.carried
				current.speaker = held.candidate.Speaker
				current.language = held.candidate.Language
				current.confidence = held.candidate.Confidence
				current.latencyMs = held.candidate.STTLatencyMs
				current.utterance = held.candidate.Utterance
			}
			current.held = nil
		}
	}
	text := transcript.Text
	mode := transcript.Mode
	if heldRevisionText != "" {
		text = heldRevisionText
		if mode == stt.ModeDelta {
			mode = stt.ModeReplacement
		}
	} else if mode == stt.ModeDelta {
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
	if current.carried != "" && mode != stt.ModeDelta &&
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
	current.ended = transcript.Mode == stt.ModeFinal
	c.marksEnds = c.marksEnds || current.ended
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
			retry := c.retry
			// The end the transcriber marked while the words were being judged is the
			// longer pause the wait was for.
			if current.ended && !current.askedEnded {
				retry = c.finalGapLocked()
			}
			c.logger.Debug("giving the caller longer to finish",
				"participant", current.participant.ID, "candidate", candidateID, "retry", retry)
			c.scheduleLocked(current, retry)
		} else {
			current.committed = current.text
			current.committedUtterance = current.utterance
			current.committedAt = time.Now()
			current.committedGeneration = current.generation
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

// Hold removes a settled candidate from active listening while a reply finishes. The
// candidate stays owned by its queue so a numbered revision can replace it and Waiting
// can commit exactly this turn when it is answered.
func (c *cadence) Hold(ready candidate) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, current := range c.speakers {
		if current.candidateID != ready.ID {
			continue
		}
		if current.timer != nil {
			current.timer.Stop()
			current.timer = nil
		}
		current.held = &heldTranscript{
			candidate:  ready,
			carried:    current.carried,
			generation: current.generation,
			heldAt:     time.Now(),
		}
		current.candidateID = ""
		current.text = ""
		current.carried = ""
		current.speaker = ""
		current.language = ""
		current.confidence = 0
		current.latencyMs = 0
		current.generation++
		return true
	}
	return false
}

// CommitHeld records a queued candidate only if that exact candidate still owns the
// held slot. A later answer cannot be replaced by an older queue draining afterward.
func (c *cadence) CommitHeld(candidateID string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, current := range c.speakers {
		if current.held == nil || current.held.candidate.ID != candidateID {
			continue
		}
		held := current.held
		current.held = nil
		if held.generation >= current.committedGeneration {
			current.committed = held.candidate.Text
			current.committedUtterance = held.candidate.Utterance
			current.committedAt = time.Now()
			current.committedGeneration = held.generation
		}
		return true
	}
	return false
}

// ReleaseHeld drops only the queued candidate that still owns the held slot.
func (c *cadence) ReleaseHeld(candidateID string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, current := range c.speakers {
		if current.held == nil || current.held.candidate.ID != candidateID {
			continue
		}
		current.held = nil
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
	var answered string
	if current.utterance != 0 && current.utterance == current.committedUtterance {
		answered = answeredPrefix(current.committed, current.text)
	}
	current.candidateID = replyPrefix + turnStamp()
	current.timer = nil
	c.grace = 0
	ready := candidate{
		ID:           current.candidateID,
		Participant:  current.participant,
		Utterance:    current.utterance,
		Speaker:      current.speaker,
		Text:         strings.TrimSpace(current.text),
		Answered:     answered,
		Language:     current.language,
		Confidence:   current.confidence,
		STTLatencyMs: current.latencyMs,
		RevisedAt:    current.revisedAt,
		ReadyAt:      time.Now(),
		Ended:        current.ended || !c.marksEnds,
	}
	current.askedEnded = ready.Ended
	c.mu.Unlock()

	c.logger.Debug("the words held still, asking whether to answer them",
		"participant", participantID, "candidate", ready.ID, "text", ready.Text, "waited", waited,
		"ended", ready.Ended)

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

// answeredPrefix is what of text the agent already answered, when text is the answered
// words with more words after them, and empty otherwise. A transcriber that keeps one run of
// speech open across an answer (Deepgram Flux resuming a turn) sends the whole run every
// time, and passing it on as said again repeats every earlier answer to the model and in
// the transcript, more each turn.
//
// Words grown onto something being read out stay whole: "order 1 2" then "3 4" is one
// order number, as are "one two" then "three" and "A B C" then "1 2", and a last token that
// grew ("ABC12345" to "ABC123456") is not new words at all.
func answeredPrefix(answered, text string) string {
	before := strings.ToLower(words(answered))
	now := strings.ToLower(words(text))
	if before == "" || !strings.HasPrefix(now, before+" ") {
		return ""
	}
	fields := strings.Fields(before)
	if readingOut(fields) {
		return ""
	}
	return answered
}

// readingOut reports whether the last of these words is a piece of something read out a piece
// at a time: a number, a number said as a word (which Gemini commits a number as once the
// caller stops), or a letter spelled out. A letter alone is a word ("I", "a"); it is spelled
// out when a piece of the same thing comes before it ("A B C").
func readingOut(words []string) bool {
	last := words[len(words)-1]
	if numberPiece(last) {
		return true
	}
	if len([]rune(last)) != 1 || len(words) < 2 {
		return false
	}
	before := words[len(words)-2]
	return len([]rune(before)) == 1 || numberPiece(before)
}

// numberPiece reports whether a word is a number or a number said as a word.
func numberPiece(word string) bool {
	if strings.ContainsFunc(word, unicode.IsDigit) {
		return true
	}
	_, digit := spokenDigits[word]
	_, teen := spokenTeens[word]
	_, tens := spokenTens[word]
	return digit || teen || tens || word == "hundred" || word == "thousand" || word == "double" || word == "triple"
}

// Said is what the caller said this turn: Text without the words already answered.
func (c candidate) Said() string {
	if c.Answered == "" {
		return c.Text
	}
	skip := len(strings.Fields(words(c.Answered)))
	fields := strings.Fields(c.Text)
	for i, field := range fields {
		if skip == 0 {
			return strings.Join(fields[i:], " ")
		}
		// Punctuation the transcriber set apart is not a word, and is not counted as one.
		if words(field) != "" {
			skip--
		}
	}
	return c.Text
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
