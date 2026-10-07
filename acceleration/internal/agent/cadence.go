package agent

import (
	"log/slog"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

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
	// defaultPreviewDebounce is how long a transcript revision has to hold still before the
	// reply to its words is started, ahead of the wait that decides whether the caller has
	// finished. It is much shorter than that wait, so the model is already working for nearly
	// all of it, and long enough that revisions arriving closer together than this restart it
	// and are previewed once, for the last of them, instead of one revision at a time. It is
	// also no longer than cadenceFinalGap, which is what leaves the words of a finalized
	// transcript to their candidate.
	defaultPreviewDebounce = 60 * time.Millisecond
	// defaultPreviewQuiet is how long the caller's audio has to have been quiet, as well as their
	// words having held still, before a reply is started for them ahead of the wait. Words hold
	// still while a caller breathes, hesitates or makes a sound that is not speech, and a reply
	// started then is thrown away when the next revision arrives.
	defaultPreviewQuiet = 120 * time.Millisecond
	// maxEarlyPreviews is how many replies are started ahead of the wait for one run of a
	// caller's words, counted from the last turn that was answered. The preview quiet already
	// keeps them to real pauses, so this only bounds a caller who pauses again and again: one
	// who speaks in short sentences pauses after each, and a smaller bound would spend itself
	// before the pause that ends the turn. After this many the reply is started with the
	// candidate.
	maxEarlyPreviews = 8
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

	// preview is how long a revision has to hold still before a reply is started for its
	// words, ahead of the wait that makes them a candidate. Zero starts none, which leaves the
	// reply to start with the candidate. previews carries the words it starts one for.
	preview  time.Duration
	previews chan candidate
	// previewQuiet is how long the caller's audio also has to have been quiet, as quietFor
	// measures it, before a reply is started ahead of the wait. Zero looks at the words alone,
	// and so does having no way to measure it. previewing says whether a reply can be started for
	// words at all, which is nothing the cadence knows: nil means it can.
	previewQuiet time.Duration
	quietFor     func(participantID string) time.Duration
	previewing   func() bool

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

	// previewTimer is the debounce that has a reply started for the words if they hold still,
	// and previewEpoch says which timer is the live one.
	previewTimer cadenceTimer
	previewEpoch int64
	// previews is how many replies were started ahead of the wait for the words since the last
	// turn that was answered.
	previews int
	// previewID is the id the last reply announced for the words was started under, and
	// previewRevision the words it was for. The candidate for those words takes the id over, so
	// that what the reply cost is attributed to the turn it became.
	previewID       string
	previewRevision uint64
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
		previews: make(chan candidate, eventBuffer),
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
	unfinished := incompleteIdentifier(text) || visiblyUnfinished(text, transcript.Language)
	final := transcript.Mode == stt.ModeFinal && !unfinished
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
	current.previewID = ""
	current.generation++
	c.revision++
	current.revision = c.revision
	current.revisedAt = time.Now()
	delay := c.gap + c.grace
	if final {
		delay = c.finalGapLocked()
	}
	if unfinished {
		// Member IDs, PINs and clock times arrive a digit at a time. Answering
		// "ABC12345" 350ms before the last 6 is how verify_identity got the wrong id. Words
		// that end on a comma or a joining word are a caller part way through a list or a
		// sentence, and get the same longer wait.
		delay = c.retry
	}
	c.scheduleLocked(current, delay)
	c.schedulePreviewLocked(current, delay, unfinished)
	c.logger.Debug("heard more, waiting for the words to stop changing",
		"participant", transcript.Participant.ID, "mode", transcript.Mode, "text", text,
		"confidence", transcript.Confidence, "gap", delay, "superseded", superseded)
	return superseded, strings.TrimSpace(text)
}

// Resolve records what became of a candidate. Waiting retries the same words after a
// longer pause; every other decision commits them and starts the next utterance cleanly.
func (c *cadence) Resolve(candidateID string, wait bool) bool {
	return c.resolveAfter(candidateID, wait, 0)
}

// resolveAfter is Resolve with the pause before a Wait retries the same words chosen by the
// caller. Zero leaves it at the configured retry.
func (c *cadence) resolveAfter(candidateID string, wait bool, retryAfter time.Duration) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if retryAfter <= 0 {
		retryAfter = c.retry
	}

	for _, current := range c.speakers {
		if current.candidateID != candidateID {
			continue
		}
		current.candidateID = ""
		if wait {
			c.logger.Debug("giving the caller longer to finish",
				"participant", current.participant.ID, "candidate", candidateID, "retry", retryAfter)
			c.scheduleLocked(current, retryAfter)
		} else {
			current.committed = current.text
			current.committedUtterance = current.utterance
			current.committedAt = time.Now()
			current.previews = 0
			current.previewID = ""
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
// Words that are still going, as the cadence holds them or as the final writes them, are not
// expedited, so they keep the longer wait.
func (c *cadence) ExpediteFinal(transcript stt.Transcript) bool {
	if !transcript.Final() || transcript.Participant.ID == "" {
		return false
	}

	c.mu.Lock()
	current, ok := c.speakers[transcript.Participant.ID]
	if !ok || current.text == "" || current.candidateID != "" || current.timer == nil ||
		current.emittedGeneration == current.generation || c.grace > 0 ||
		incompleteIdentifier(current.text) || visiblyUnfinished(current.text, current.language) ||
		visiblyUnfinished(transcript.Text, transcript.Language) || !sameWords(current.text, transcript.Text) {
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

	if current, ok := c.speakers[participant.ID]; ok {
		if current.timer != nil {
			current.timer.Stop()
		}
		c.stopPreviewLocked(current)
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
		c.stopPreviewLocked(current)
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

// timer runs fn after d on the cadence's own clock, so that what is waited on for a caller's
// words, the settling of them and the patience for them, is waited on by one clock.
func (c *cadence) timer(d time.Duration, fn func()) cadenceTimer {
	c.mu.Lock()
	after := c.after
	c.mu.Unlock()
	return after(d, fn)
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
	// A reply already started for these words was asked for under an id of its own, and the model
	// call and the request it was reported as carry it. The turn is known by that id too, so the
	// cost of the reply joins the turn it became. It is spent once: the same words put again after
	// a Wait are a turn of their own.
	if current.previewID != "" && current.previewRevision == current.revision {
		current.candidateID = current.previewID
	}
	current.previewID = ""
	current.emittedGeneration = generation
	current.timer = nil
	// The reply for these words is the candidate's to start from here.
	c.stopPreviewLocked(current)
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

// Previews carries the words of a revision that has held still for the preview debounce, which
// is when a reply to them can be started: ahead of the wait that makes them a candidate, and so
// ahead of any ruling on whether the caller has finished. Each is the words as they stand and
// nothing more, with an id of its own, and a candidate for the same words follows them unless
// they change first.
func (c *cadence) Previews() <-chan candidate { return c.previews }

// previewable reports whether the words a preview was announced for are still the words being
// settled and have not been put to a ruling, which is when the candidate owns what is started
// for them.
func (c *cadence) previewable(early candidate) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	current, ok := c.speakers[early.Participant.ID]
	return ok && current.revision == early.Revision && strings.TrimSpace(current.text) != "" &&
		current.candidateID == "" && current.emittedGeneration != current.generation
}

// schedulePreviewLocked starts the debounce for the words just heard, replacing the one the
// words they replace had running. Words that end visibly unfinished are not previewed, because
// they are about to change, and nor are words whose candidate is due no later than the debounce
// would be, which would only be asked about at the same moment. No debounce is armed at all when
// no reply can be started for the words, while the line is running late after an overlap and is
// owed grace, or once a reply has been started ahead of the wait for this many revisions of the
// caller's words since the last turn that was answered: those are started with the candidate.
// The caller holds the lock.
func (c *cadence) schedulePreviewLocked(current *cadenceSpeaker, candidateDelay time.Duration, unfinished bool) {
	c.stopPreviewLocked(current)
	if c.preview <= 0 || unfinished || candidateDelay <= c.preview || c.grace > 0 ||
		current.previews >= maxEarlyPreviews || (c.previewing != nil && !c.previewing()) {
		return
	}
	c.armPreviewLocked(current, c.preview)
}

// armPreviewLocked has the words as they stand announced after the delay, if they are still
// the words by then. The caller holds the lock.
func (c *cadence) armPreviewLocked(current *cadenceSpeaker, delay time.Duration) {
	generation := current.generation
	epoch := c.nextTimerEpochLocked()
	current.previewEpoch = epoch
	participantID := current.participant.ID
	current.previewTimer = c.after(delay, func() {
		c.emitPreview(participantID, generation, epoch)
	})
}

// stopPreviewLocked cancels a debounce that has not run. One that has already fired and is
// waiting on the lock is cancelled too, because the epoch it was started under no longer
// names a live timer, so it finds nothing to announce: stopping a timer cannot recall a
// callback that has begun. The caller holds the lock.
func (c *cadence) stopPreviewLocked(current *cadenceSpeaker) {
	if current.previewTimer != nil {
		current.previewTimer.Stop()
		current.previewTimer = nil
	}
	current.previewEpoch = 0
}

// emitPreview announces words that held still for the preview debounce, unless they have
// changed, been put to a ruling or been forgotten since it began, or the caller's audio has not
// been quiet for the preview quiet. Words hold still while a caller is still voiced, so that
// is looked at again here rather than when the debounce was armed, and when it is not so yet the
// debounce runs on for the rest of it.
func (c *cadence) emitPreview(participantID string, generation, epoch int64) {
	c.mu.Lock()
	current, ok := c.speakers[participantID]
	if c.closed || !ok || current.generation != generation || current.previewEpoch != epoch ||
		current.text == "" || current.candidateID != "" || current.emittedGeneration == generation ||
		(c.previewing != nil && !c.previewing()) {
		c.mu.Unlock()
		return
	}
	if c.previewQuiet > 0 && c.quietFor != nil {
		if quiet := c.quietFor(participantID); quiet < c.previewQuiet {
			c.armPreviewLocked(current, c.previewQuiet-quiet)
			c.mu.Unlock()
			return
		}
	}
	current.previewTimer = nil
	// Announced once: the words are only announced again if they change and hold still again.
	current.previewEpoch = 0
	current.previews++
	ready := candidate{
		ID:           replyPrefix + turnStamp(),
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
	current.previewID, current.previewRevision = ready.ID, ready.Revision
	c.mu.Unlock()

	c.logger.Debug("the words held still, starting a reply to them",
		"participant", participantID, "preview", ready.ID, "text", ready.Text)

	// A preview that cannot be queued is dropped: the candidate for the same words starts the
	// reply, as it would have without one.
	select {
	case c.previews <- ready:
	default:
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

// continuationWords are the words a speaker leaves a sentence on when more is on its way:
// the conjunctions that join on another clause and the sounds made while finding the next
// word. "so" is not one of them: it ends a sentence as often as it joins one, as "I think so"
// does. They are English, and only a transcript in English is held to them.
var continuationWords = []string{"and", "or", "but", "because", "um", "uh", "er"}

// clauseCommas are the commas a transcriber writes: the Latin one, and the fullwidth,
// ideographic and Arabic ones, so the test holds in any language that is punctuated.
const clauseCommas = ",，、،"

// closingQuotes are the quotation marks a transcriber may close a quoted stretch with, after the
// comma it was in the middle of.
const closingQuotes = "\"'”’»›」』"

// visiblyUnfinished reports whether the words stop where a speaker is plainly about to say
// more: on a comma, a coordinating conjunction, or a filled hesitation.
//
// A pause there is a breath in the middle of a turn, a list being read out or a clause being
// joined on, far more often than the end of one. A transcriber finalizing the words says
// where the audio went quiet, not that the caller is done, so it is no reason to answer.
// The comma is matched as a character, whatever the language and even when a closing quote
// follows it, and the rest as whole words, ignoring case and any punctuation after them, so
// "band" and "summer" are not "and" and "um". The words are English, so they are only looked
// for when the transcript is in English or does not say what it is in: "um" ends a sentence in
// some other languages.
func visiblyUnfinished(text, language string) bool {
	text = strings.TrimRightFunc(text, unicode.IsSpace)
	if last, _ := utf8.DecodeLastRuneInString(strings.TrimRight(text, closingQuotes)); strings.ContainsRune(clauseCommas, last) {
		return true
	}
	if !english(language) {
		return false
	}
	lastWord := text
	if space := strings.LastIndexFunc(text, unicode.IsSpace); space >= 0 {
		_, width := utf8.DecodeRuneInString(text[space:])
		lastWord = text[space+width:]
	}
	lastWord = strings.TrimRightFunc(lastWord, func(symbol rune) bool {
		return !unicode.IsLetter(symbol) && !unicode.IsDigit(symbol)
	})
	for _, word := range continuationWords {
		if strings.EqualFold(lastWord, word) {
			return true
		}
	}
	return false
}

// english reports whether a transcript's language is English, or is not said, which is how a
// transcriber that has not settled on one writes it.
func english(language string) bool {
	language = strings.ToLower(strings.TrimSpace(language))
	return language == "" || language == "en" || strings.HasPrefix(language, "en-")
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
