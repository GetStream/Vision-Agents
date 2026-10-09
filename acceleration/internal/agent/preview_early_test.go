package agent

import (
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/guardrail"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// previewing has the cadence announce words that hold still for d, with the default gap so that
// the candidate for them is due long after, and the timers captured so that nothing runs until
// the test says.
func (s *CadenceSuite) previewing(d time.Duration) *[]*capturedCadenceTimer {
	s.useDefaultCadence()
	s.cadence.preview = d
	return s.captureTimers()
}

// observe is a revision of a participant's words that has not been finalized.
func (s *CadenceSuite) observe(participant stt.Participant, text string) {
	s.cadence.Observe(stt.Transcript{Participant: participant, Mode: stt.ModeReplacement, Text: text})
}

// announced is the words the cadence says a reply can be started for.
func (s *CadenceSuite) announced() candidate {
	select {
	case early := <-s.cadence.Previews():
		return early
	case <-time.After(time.Second):
		s.FailNow("no reply was announced for the words")
		return candidate{}
	}
}

// nothingAnnounced asserts no reply is announced.
func (s *CadenceSuite) nothingAnnounced() {
	select {
	case early := <-s.cadence.Previews():
		s.Failf("nothing should have been announced", "got %q", early.Text)
	case <-time.After(50 * time.Millisecond):
	}
}

// previewClock is the cadence's timers on a clock the test moves, so that revisions a few
// milliseconds apart are replayed exactly. With fireStopped, a timer that was stopped still runs
// when it falls due, which is a timer that had already begun to run when it was stopped.
type previewClock struct {
	fireStopped bool

	mu     sync.Mutex
	now    time.Duration
	timers []*clockTimer
}

type clockTimer struct {
	clock   *previewClock
	due     time.Duration
	run     func()
	stopped bool
	fired   bool
}

func (t *clockTimer) Stop() bool {
	t.clock.mu.Lock()
	defer t.clock.mu.Unlock()
	active := !t.stopped && !t.fired
	t.stopped = true
	return active
}

func (c *previewClock) after(delay time.Duration, run func()) cadenceTimer {
	c.mu.Lock()
	defer c.mu.Unlock()
	timer := &clockTimer{clock: c, due: c.now + delay, run: run}
	c.timers = append(c.timers, timer)
	return timer
}

// advance moves the clock on by d, running the timers that fall due on the way, earliest first.
func (c *previewClock) advance(d time.Duration) {
	c.mu.Lock()
	target := c.now + d
	c.mu.Unlock()
	for {
		c.mu.Lock()
		var next *clockTimer
		for _, timer := range c.timers {
			if timer.fired || timer.due > target || (timer.stopped && !c.fireStopped) {
				continue
			}
			if next == nil || timer.due < next.due {
				next = timer
			}
		}
		if next == nil {
			c.now = target
			c.mu.Unlock()
			return
		}
		c.now, next.fired = next.due, true
		c.mu.Unlock()
		next.run()
	}
}

// armed is how long from now each timer that has neither run nor been stopped has left.
func (c *previewClock) armed() []time.Duration {
	c.mu.Lock()
	defer c.mu.Unlock()
	var left []time.Duration
	for _, timer := range c.timers {
		if !timer.fired && !timer.stopped {
			left = append(left, timer.due-c.now)
		}
	}
	return left
}

// burst is the words of a caller being transcribed a few at a time.
var burst = []string{
	"please", "please find", "please find a", "please find a table", "please find a table for",
	"please find a table for two",
}

// announcedNow is what the cadence has announced so far, without waiting for more.
func (s *CadenceSuite) announcedNow() []candidate {
	var early []candidate
	for {
		select {
		case one := <-s.cadence.Previews():
			early = append(early, one)
		default:
			return early
		}
	}
}

func (s *CadenceSuite) TestABurstOfRevisionsThirtyMillisecondsApartIsAnnouncedOnceForTheFinalWords() {
	s.useDefaultCadence()
	s.cadence.preview = defaultPreviewDebounce
	clock := &previewClock{fireStopped: true}
	s.cadence.after = clock.after
	alice := stt.Participant{ID: "alice"}

	for _, text := range burst {
		s.observe(alice, text)
		clock.advance(30 * time.Millisecond)
		s.Empty(s.announcedNow(), "words that were still changing were announced: "+text)
	}
	clock.advance(defaultPreviewDebounce - 30*time.Millisecond)

	early := s.announcedNow()
	s.Require().Len(early, 1, "a burst of revisions is one preview")
	s.Equal(burst[len(burst)-1], early[0].Text)

	// Every earlier timer, stopped or not, and the candidate for the words come due.
	clock.advance(time.Minute)
	s.Empty(s.announcedNow(), "the words were announced again")
	ready := s.ready()
	s.Equal(early[0].Revision, ready.Revision, "the candidate is for the words that were announced")
}

func (s *CadenceSuite) TestADebounceThatFiresForWordsThatHaveChangedAnnouncesNothing() {
	s.useDefaultCadence()
	s.cadence.preview = defaultPreviewDebounce
	clock := &previewClock{fireStopped: true}
	s.cadence.after = clock.after
	alice := stt.Participant{ID: "alice"}

	s.observe(alice, "book a table")
	s.observe(alice, "book a table and")
	clock.advance(defaultPreviewDebounce)

	s.Empty(s.announcedNow(), "words that were replaced by unfinished ones were announced")

	s.observe(alice, "book a table for two")
	s.cadence.Forget(alice)
	s.observe(stt.Participant{ID: "bob"}, "who is there")
	clock.advance(time.Minute)

	early := s.announcedNow()
	s.Require().Len(early, 1, "words that were forgotten were announced")
	s.Equal("bob", early[0].Participant.ID)
}

func (s *CadenceSuite) TestWordsThatHoldStillForTheDebounceAreAnnouncedAheadOfTheirCandidate() {
	timers := s.previewing(defaultPreviewDebounce)
	alice := stt.Participant{ID: "alice"}

	s.observe(alice, "book a table")

	s.Require().Len(*timers, 2)
	s.Equal(defaultCadenceGap, (*timers)[0].delay)
	s.Equal(defaultPreviewDebounce, (*timers)[1].delay)
	(*timers)[1].fire()
	early := s.announced()
	s.Equal(alice, early.Participant)
	s.Equal("book a table", early.Text)
	s.NotEmpty(early.ID)
	s.NotZero(early.Revision)
	s.quiet()

	(*timers)[0].fire()
	ready := s.ready()
	s.Equal(early.Revision, ready.Revision, "the candidate is for the words that were announced")
	s.Equal(early.ID, ready.ID, "the cost of the reply started for the words joins the turn they become")
}

func (s *CadenceSuite) TestAnAnnouncedIdIsSpentByTheFirstCandidateForTheWordsOnly() {
	timers := s.previewing(defaultPreviewDebounce)
	alice := stt.Participant{ID: "alice"}
	s.observe(alice, "book a table")
	(*timers)[1].fire()
	early := s.announced()
	(*timers)[0].fire()
	first := s.ready()
	s.Equal(early.ID, first.ID)

	// A Wait puts the same words again, and that is a turn of its own.
	s.Require().True(s.cadence.Resolve(first.ID, true))
	(*timers)[len(*timers)-1].fire()
	again := s.ready()
	s.Equal(first.Revision, again.Revision)
	s.NotEqual(early.ID, again.ID)
}

func (s *CadenceSuite) TestACandidateForWordsThatChangedAfterTheAnnouncementIsAnIdOfItsOwn() {
	timers := s.previewing(defaultPreviewDebounce)
	alice := stt.Participant{ID: "alice"}
	s.observe(alice, "book a table")
	(*timers)[1].fire()
	early := s.announced()

	s.observe(alice, "book a table for two")
	(*timers)[2].fire()

	ready := s.ready()
	s.Equal("book a table for two", ready.Text)
	s.NotEqual(early.ID, ready.ID, "the reply for the old words is not the turn for the new ones")
}

func (s *CadenceSuite) TestACandidateWithNoAnnouncementIsAnIdOfItsOwn() {
	timers := s.previewing(0)
	s.observe(stt.Participant{ID: "alice"}, "book a table")
	(*timers)[0].fire()

	s.NotEmpty(s.ready().ID)
}

func (s *CadenceSuite) TestNewWordsRestartTheDebounce() {
	timers := s.previewing(defaultPreviewDebounce)
	alice := stt.Participant{ID: "alice"}

	s.observe(alice, "book a table")
	s.observe(alice, "book a table for two")

	s.Require().Len(*timers, 4)
	s.True((*timers)[1].stopped, "the debounce of the words that were replaced was left running")
	(*timers)[1].fire()
	s.nothingAnnounced()
	(*timers)[3].fire()
	early := s.announced()
	s.Equal("book a table for two", early.Text)
	s.nothingAnnounced()
}

func (s *CadenceSuite) TestTheSameWordsAgainDoNotRestartTheDebounce() {
	timers := s.previewing(defaultPreviewDebounce)
	alice := stt.Participant{ID: "alice"}

	s.observe(alice, "book a table")
	s.observe(alice, "Book a table.")

	s.Require().Len(*timers, 2)
	s.False((*timers)[1].stopped)
}

func (s *CadenceSuite) TestWordsThatEndUnfinishedAreNotAnnounced() {
	timers := s.previewing(defaultPreviewDebounce)
	alice := stt.Participant{ID: "alice"}

	for _, text := range []string{"book a table,", "book a table and", "book a table um", "my member id is ABC12"} {
		s.observe(alice, text)
	}

	s.Require().Len(*timers, 4, "each revision is waited on, and none is announced")
	for _, timer := range *timers {
		s.Equal(defaultCadenceRetry, timer.delay)
	}
}

func (s *CadenceSuite) TestWordsWhoseCandidateIsDueNoLaterAreNotAnnounced() {
	timers := s.previewing(defaultPreviewDebounce)

	s.cadence.Observe(stt.Transcript{Participant: stt.Participant{ID: "alice"}, Mode: stt.ModeFinal, Text: "book a table"})

	s.Require().Len(*timers, 1)
	s.Equal(cadenceFinalGap, (*timers)[0].delay)
}

func (s *CadenceSuite) TestWithoutADebounceNothingIsAnnounced() {
	timers := s.previewing(0)

	s.observe(stt.Participant{ID: "alice"}, "book a table")

	s.Len(*timers, 1)
}

func (s *CadenceSuite) TestACandidateForTheWordsEndsTheDebounce() {
	timers := s.previewing(defaultPreviewDebounce)
	s.observe(stt.Participant{ID: "alice"}, "book a table")

	(*timers)[0].fire()
	ready := s.ready()

	s.True((*timers)[1].stopped)
	(*timers)[1].fire()
	s.nothingAnnounced()
	s.Require().True(s.cadence.Resolve(ready.ID, true))
	(*timers)[1].fire()
	s.nothingAnnounced()
}

func (s *CadenceSuite) TestForgettingAParticipantEndsTheirDebounce() {
	timers := s.previewing(defaultPreviewDebounce)
	alice := stt.Participant{ID: "alice"}
	s.observe(alice, "book a table")

	s.cadence.Forget(alice)

	s.True((*timers)[1].stopped)
	(*timers)[1].fire()
	s.nothingAnnounced()
}

func (s *CadenceSuite) TestClosingEndsEveryDebounce() {
	timers := s.previewing(defaultPreviewDebounce)
	s.observe(stt.Participant{ID: "alice"}, "book a table")

	s.cadence.Close()

	s.True((*timers)[1].stopped)
	(*timers)[1].fire()
	s.nothingAnnounced()
}

// quietAfter has the cadence told, through the pointer it returns, how long the caller's audio has
// been quiet, and asked for the default quiet before a reply is started.
func (s *CadenceSuite) quietAfter(quiet time.Duration) *time.Duration {
	s.cadence.previewQuiet = defaultPreviewQuiet
	s.cadence.quietFor = func(string) time.Duration { return quiet }
	return &quiet
}

func (s *CadenceSuite) TestWordsThatHoldStillWhileTheCallerIsStillVoicedAreNotAnnouncedUntilTheyHaveBeenQuiet() {
	s.useDefaultCadence()
	s.cadence.preview = defaultPreviewDebounce
	clock := &previewClock{fireStopped: true}
	s.cadence.after = clock.after
	voicedFor := new(time.Duration)
	s.cadence.previewQuiet = defaultPreviewQuiet
	s.cadence.quietFor = func(string) time.Duration { return *voicedFor }
	alice := stt.Participant{ID: "alice"}

	for _, text := range burst {
		s.observe(alice, text)
		clock.advance(30 * time.Millisecond)
	}
	clock.advance(defaultPreviewDebounce)
	s.Empty(s.announcedNow(), "words that held still were announced while the caller was still voiced")

	clock.advance(defaultPreviewQuiet)
	s.Empty(s.announcedNow(), "the debounce was not looked at again while the caller was still voiced")

	*voicedFor = defaultPreviewQuiet
	clock.advance(defaultPreviewQuiet)
	early := s.announcedNow()
	s.Require().Len(early, 1, "words that held still once the caller was quiet are one preview")
	s.Equal(burst[len(burst)-1], early[0].Text)

	clock.advance(time.Minute)
	s.Empty(s.announcedNow(), "the words were announced again")
	s.Equal(early[0].Revision, s.ready().Revision, "the candidate is for the words that were announced")
}

func (s *CadenceSuite) TestWordsThatHoldStillOnceTheCallerHasBeenQuietAreAnnouncedOnce() {
	timers := s.previewing(defaultPreviewDebounce)
	s.quietAfter(time.Hour)
	alice := stt.Participant{ID: "alice"}

	s.observe(alice, "book a table")
	(*timers)[1].fire()

	early := s.announcedNow()
	s.Require().Len(early, 1)
	s.Equal("book a table", early[0].Text)
	s.Len(*timers, 2, "a debounce that was satisfied is not armed again")
	(*timers)[1].fire()
	s.Empty(s.announcedNow(), "the same words were announced twice")
}

func (s *CadenceSuite) TestADebounceThatFindsTheCallerPartlyQuietRunsOnForTheRest() {
	timers := s.previewing(defaultPreviewDebounce)
	quiet := s.quietAfter(50 * time.Millisecond)
	alice := stt.Participant{ID: "alice"}

	s.observe(alice, "book a table")
	(*timers)[1].fire()

	s.Empty(s.announcedNow(), "words were announced before the caller had been quiet for the quiet")
	s.Require().Len(*timers, 3)
	s.Equal(defaultPreviewQuiet-50*time.Millisecond, (*timers)[2].delay)

	*quiet = defaultPreviewQuiet
	(*timers)[2].fire()
	s.Require().Len(s.announcedNow(), 1)
}

func (s *CadenceSuite) TestWithoutAQuietTheWordsAloneDecide() {
	timers := s.previewing(defaultPreviewDebounce)
	s.cadence.quietFor = func(string) time.Duration { return 0 }
	alice := stt.Participant{ID: "alice"}

	s.observe(alice, "book a table")
	(*timers)[1].fire()

	s.Require().Len(s.announcedNow(), 1, "a caller who is voiced held a reply back that was asked to look at the words alone")
}

func (s *CadenceSuite) TestEarlyRepliesStopAtTheCapForOneRunOfWords() {
	timers := s.previewing(defaultPreviewDebounce)
	alice := stt.Participant{ID: "alice"}
	revisions := make([]string, maxEarlyPreviews+1)
	words := "book"
	for i := range revisions {
		revisions[i] = words
		words += " again"
	}

	for i, text := range revisions[:maxEarlyPreviews] {
		s.observe(alice, text)
		(*timers)[2*i+1].fire()
		s.Equal(text, s.announced().Text)
	}
	armed := len(*timers)
	s.observe(alice, revisions[maxEarlyPreviews])
	s.Len(*timers, armed+1, "a debounce was armed after the cap")
	s.Equal(defaultCadenceGap, (*timers)[armed].delay, "only the candidate is")

	(*timers)[armed].fire()
	ready := s.ready()
	s.Equal(revisions[maxEarlyPreviews], ready.Text)
	s.Require().True(s.cadence.Resolve(ready.ID, false))

	armed = len(*timers)
	s.observe(alice, "and another thing")
	s.Len(*timers, armed+2, "the count did not start again once the turn was answered")
}

func (s *CadenceSuite) TestNoDebounceIsArmedWhenNoReplyCanBeStartedOrTheLineIsOwedGrace() {
	timers := s.previewing(defaultPreviewDebounce)
	alice := stt.Participant{ID: "alice"}

	s.cadence.previewing = func() bool { return false }
	s.observe(alice, "book a table")
	s.Require().Len(*timers, 1, "a debounce was armed where no reply can be started")
	s.Equal(defaultCadenceGap, (*timers)[0].delay)

	s.cadence.previewing = nil
	s.cadence.Grace(150 * time.Millisecond)
	s.observe(alice, "book a table for two")
	s.Require().Len(*timers, 2, "a debounce was armed while the line is running late")
	s.Equal(defaultCadenceGap+150*time.Millisecond, (*timers)[1].delay)
}

func TestThePreviewQuietDefaultsToOneHundredAndTwentyMillisecondsAndCanBeTurnedOff(t *testing.T) {
	options := func(quiet *time.Duration) Options {
		return Options{
			CustomerID: "acme", Edge: newLoopbackEdge(), LLM: &llmrouter.Router{},
			STT: &sttrouter.Router{}, TTS: &ttsrouter.Router{}, PreviewQuiet: quiet,
		}
	}

	left, err := New(options(nil))
	require.NoError(t, err)
	require.Equal(t, 120*time.Millisecond, left.cadence.previewQuiet)
	require.NotNil(t, left.voiced, "the caller's audio is listened to for it")

	off := time.Duration(0)
	disabled, err := New(options(&off))
	require.NoError(t, err)
	require.Zero(t, disabled.cadence.previewQuiet)

	negative := -time.Millisecond
	_, err = New(options(&negative))
	require.Error(t, err)
}

func TestNoReplyIsStartedAheadOfTheWaitInTextOrSpeechToSpeechMode(t *testing.T) {
	written, err := New(Options{Text: true, CustomerID: "acme", LLM: &llmrouter.Router{}})
	require.NoError(t, err)
	require.False(t, written.previewsEarly())

	spoken, err := New(Options{
		CustomerID: "acme", Edge: newLoopbackEdge(), LLM: &llmrouter.Router{},
		STT: &sttrouter.Router{}, TTS: &ttsrouter.Router{},
	})
	require.NoError(t, err)
	require.True(t, spoken.previewsEarly())
	spoken.nativeMode.Store(true)
	require.False(t, spoken.previewsEarly())

	off := false
	unpreviewed, err := New(Options{
		CustomerID: "acme", Edge: newLoopbackEdge(), LLM: &llmrouter.Router{},
		STT: &sttrouter.Router{}, TTS: &ttsrouter.Router{}, SpeculativeReplies: &off,
	})
	require.NoError(t, err)
	require.False(t, unpreviewed.previewsEarly())
}

func TestThePreviewDebounceDefaultsToSixtyMillisecondsAndCanBeTurnedOff(t *testing.T) {
	options := func(debounce *time.Duration) Options {
		return Options{
			CustomerID: "acme", Edge: newLoopbackEdge(), LLM: &llmrouter.Router{},
			STT: &sttrouter.Router{}, TTS: &ttsrouter.Router{}, PreviewDebounce: debounce,
		}
	}

	left, err := New(options(nil))
	require.NoError(t, err)
	require.Equal(t, 60*time.Millisecond, left.cadence.preview)

	off := time.Duration(0)
	disabled, err := New(options(&off))
	require.NoError(t, err)
	require.Zero(t, disabled.cadence.preview)

	negative := -time.Millisecond
	_, err = New(options(&negative))
	require.Error(t, err)
}

// debouncesFor has the reply to words start once they have held still for d. It is said before
// the agent joins.
func (s *AgentSuite) debouncesFor(d time.Duration) { s.previewDebounce = &d }

// slowGap leaves a revision's words unasked about for d, so that what is started for them
// ahead of the ruling can be seen before there is a ruling.
func (s *AgentSuite) slowGap(d time.Duration) {
	s.agent.cadence.mu.Lock()
	defer s.agent.cadence.mu.Unlock()
	s.agent.cadence.gap = d
}

// asked is what the conversation model was last given as the caller's words.
func (s *AgentSuite) asked(request int) string {
	input := s.model.requests()[request].Input
	return input[len(input)-1].Content
}

// holdsAnEarlyPreview joins an agent that has started a reply for a caller's words and is not
// going to ask about them, and returns the caller and what tells the reply was let go of.
func (s *AgentSuite) holdsAnEarlyPreview() (stt.Participant, watchedPreview) {
	s.join(true)
	s.slowGap(time.Hour)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)
	s.mutters(alice, "please find a table")
	s.eventually(func() bool { return s.keptPreviews() == 1 && len(s.model.requests()) == 1 },
		"no reply was started for the words")
	return alice, s.watchPreview()
}

func (s *AgentSuite) TestWordsThatHoldStillWhileTheCallerIsStillVoicedStartNoReplyUntilTheyHaveBeenQuiet() {
	quiet := 800 * time.Millisecond
	s.previewQuiet = &quiet
	s.join(true)
	s.slowGap(10 * time.Second)
	alice := stt.Participant{ID: "alice"}
	s.speakAloud(alice)

	s.mutters(alice, "please find a table")

	s.Never(func() bool { return len(s.model.requests()) > 0 }, 400*time.Millisecond, 10*time.Millisecond,
		"a reply was started for words that held still while the caller was still voiced")
	s.eventually(func() bool { return len(s.model.requests()) == 1 && s.keptPreviews() == 1 },
		"no reply was started once the caller had been quiet")
	s.Equal("please find a table", s.asked(0))
}

func (s *AgentSuite) TestNoDebounceIsArmedWhenRepliesAreNotPreviewed() {
	off := false
	s.speculation = &off
	s.join(true)
	clock := &previewClock{}
	s.agent.cadence.mu.Lock()
	s.agent.cadence.after = clock.after
	s.agent.cadence.mu.Unlock()
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	s.mutters(alice, "please find a table")

	s.eventually(func() bool { _, heard := s.agent.cadence.currentCandidate(alice.ID); return heard },
		"the words were never heard")
	s.Equal([]time.Duration{defaultCadenceGap}, clock.armed(), "only the candidate is waited for")
}

func (s *AgentSuite) TestAStableRevisionStartsAReplyBeforeItsCandidateAndTheCandidateAdoptsIt() {
	s.join(true)
	s.slowGap(500 * time.Millisecond)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	s.mutters(alice, "please find a table")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "no reply was started for the words")
	s.Empty(s.flow.requests(), "the reply was started before the words were put to a ruling")
	s.Equal(1, s.keptPreviews(), "it is held for the candidate")
	s.Equal(1, s.previewsHeld())
	s.Empty(s.voice.spoken(), "a preview is never spoken before the words are answered")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the words were never answered")
	s.Len(s.model.requests(), 1, "the candidate took the reply over rather than asking the model again")
	s.Len(s.flow.requests(), 1)
	s.Contains(said(s.voice.spoken()), "Hello there.")
	s.Zero(s.previewsHeld())
	s.Zero(s.keptPreviews())
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	responding, _ := firstOf[Responding](s.reported())
	turn, _ := firstOf[Turn](s.reported())
	s.Equal(responding.TurnID, turn.TurnID)

	// What the reply cost is reported under the turn it became.
	s.eventually(func() bool { return len(s.replyCalls()) == 1 }, "the model call was never reported")
	s.Equal(turn.TurnID, s.replyCalls()[0].TurnID, "the model call does not join the turn row")
}

// replyCalls are the model calls reported so far for replies to the caller.
func (s *AgentSuite) replyCalls() []ModelCall {
	var calls []ModelCall
	for _, event := range s.reported() {
		if call, ok := event.(ModelCall); ok && call.Purpose == "reply" {
			calls = append(calls, call)
		}
	}
	return calls
}

// controlsTimers has the cadence's timers, and the patience for the words it settles, run on a
// clock the test moves instead of the one on the wall.
func (s *AgentSuite) controlsTimers() *previewClock {
	clock := &previewClock{fireStopped: true}
	s.agent.cadence.mu.Lock()
	defer s.agent.cadence.mu.Unlock()
	s.agent.cadence.after = clock.after
	return clock
}

// cadenceHolds waits for the cadence to hold the participant's words as they are given.
func (s *AgentSuite) cadenceHolds(participant stt.Participant, text string) {
	s.eventually(func() bool {
		heard, ok := s.agent.cadence.currentCandidate(participant.ID)
		return ok && heard.Text == text
	}, "the words were never heard: "+text)
}

func (s *AgentSuite) TestWordsThatChangeWithinTheDebounceRestartItAndOneReplyIsStartedForTheLastOnes() {
	s.debouncesFor(300 * time.Millisecond)
	s.join(true)
	clock := s.controlsTimers()
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	s.mutters(alice, "please find a table")
	s.cadenceHolds(alice, "please find a table")
	clock.advance(150 * time.Millisecond)
	s.mutters(alice, "please find a table for two")
	s.cadenceHolds(alice, "please find a table for two")
	clock.advance(250 * time.Millisecond)

	s.Empty(s.model.requests(), "a reply was started for words that changed before they held still")
	s.ElementsMatch([]time.Duration{100 * time.Millisecond, 50 * time.Millisecond}, clock.armed(),
		"the debounce for the first words was left running, or the new words had none")
	clock.advance(50 * time.Millisecond)
	s.eventually(func() bool { return len(s.model.requests()) == 1 && s.keptPreviews() == 1 },
		"no reply was started for the last words")
	s.Equal("please find a table for two", s.asked(0))

	// The candidate for the same words takes the reply over rather than starting another.
	clock.advance(100 * time.Millisecond)
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the words were never answered")
	s.Len(s.model.requests(), 1, "more than one reply for one participant")
	s.Zero(s.previewsHeld())
}

func (s *AgentSuite) TestABurstOfRevisionsThirtyMillisecondsApartStartsOneReplyForTheFinalWords() {
	s.join(true)
	clock := s.controlsTimers()
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	for _, text := range burst {
		s.mutters(alice, text)
		s.cadenceHolds(alice, text)
		clock.advance(30 * time.Millisecond)
		s.Empty(s.model.requests(), "a reply was started for words that were still changing: "+text)
		s.Zero(s.previewsHeld())
	}
	clock.advance(defaultPreviewDebounce)

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "no reply was started for the last words")
	s.Equal(burst[len(burst)-1], s.asked(0))
	s.Never(func() bool { return len(s.model.requests()) > 1 || s.previewsHeld() > 1 }, 300*time.Millisecond,
		10*time.Millisecond, "more than one reply for one participant")
	s.Equal(1, s.keptPreviews())
	s.Equal(1, s.previewsHeld())
}

func (s *AgentSuite) TestWordsThatChangeAfterAReplyWasStartedLetGoOfItAndStartAnother() {
	s.join(true)
	s.slowGap(10 * time.Second)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)
	s.mutters(alice, "please find a table")
	s.eventually(func() bool { return s.keptPreviews() == 1 && len(s.model.requests()) == 1 },
		"no reply was started for the words")
	first := s.watchPreview()

	s.mutters(alice, "please find a table for two")

	s.eventually(first.Load, "the reply to the old words was left running")
	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "no reply was started for the new words")
	s.Equal("please find a table for two", s.asked(1))
	agent := s.agent
	s.Never(func() bool {
		agent.mu.Lock()
		defer agent.mu.Unlock()
		return len(agent.previews) > 1
	}, 200*time.Millisecond, 5*time.Millisecond, "two previews for one participant")
	s.Equal(1, s.keptPreviews())
}

func (s *AgentSuite) TestWordsThatLookUnfinishedAreNotPreviewedEarly() {
	s.join(true)
	s.slowGap(10 * time.Second)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	for _, words := range []string{"please find a table,", "please find a table and", "please find a table um",
		"my member id is ABC12"} {
		s.mutters(alice, words)
		s.Never(func() bool { return len(s.model.requests()) > 0 }, 150*time.Millisecond, 10*time.Millisecond,
			"a reply was started for words that end unfinished: "+words)
	}

	s.Zero(s.keptPreviews())
	s.Zero(s.previewsHeld())
}

func (s *AgentSuite) TestWithoutADebounceTheReplyStartsWithTheCandidate() {
	s.debouncesFor(0)
	s.join(true)
	s.slowGap(400 * time.Millisecond)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	s.mutters(alice, "please find a table")

	s.Never(func() bool { return len(s.model.requests()) > 0 }, 250*time.Millisecond, 10*time.Millisecond,
		"a reply was started ahead of the candidate")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the words were never answered")
	s.Len(s.model.requests(), 1)
}

func (s *AgentSuite) TestNoReplyIsStartedEarlyWithoutSpeculativeReplies() {
	off := false
	s.speculation = &off
	s.join(true)
	s.slowGap(10 * time.Second)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	s.mutters(alice, "please find a table")

	s.Never(func() bool { return len(s.model.requests()) > 0 || s.previewsHeld() > 0 },
		250*time.Millisecond, 10*time.Millisecond, "a reply was started for words nobody asked about")
}

func (s *AgentSuite) TestNoReplyIsStartedEarlyWhileTheAgentIsSpeaking() {
	s.join(true)
	s.slowGap(10 * time.Second)
	s.agent.mu.Lock()
	s.agent.utterances = 1
	s.agent.mu.Unlock()
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	s.mutters(alice, "please find a table")

	s.Never(func() bool { return len(s.model.requests()) > 0 || s.previewsHeld() > 0 },
		250*time.Millisecond, 10*time.Millisecond, "a reply was started for words said over the agent")
}

func (s *AgentSuite) TestNoReplyIsStartedEarlyWhenAPolicyMustClearTheWordsFirst() {
	s.screens(guardrail.ModeBlocking)
	s.join(true)
	s.slowGap(10 * time.Second)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	s.mutters(alice, "please find a table")

	s.Never(func() bool { return len(s.model.requests()) > 0 || s.previewsHeld() > 0 },
		250*time.Millisecond, 10*time.Millisecond, "a reply was started for words the policy had not cleared")
}

func (s *AgentSuite) TestNoReplyIsStartedEarlyForAnotherVoiceAtTheMicrophone() {
	s.join(true)
	s.slowGap(10 * time.Second)
	clock := s.controlsTimers()
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)
	s.ears.emitter.Send(stt.Transcript{Participant: alice, Mode: stt.ModeReplacement, Speaker: "voice-1", Text: "please find a table"})
	s.cadenceHolds(alice, "please find a table")
	clock.advance(defaultPreviewDebounce)
	// A preview is kept as soon as it is held, which is before the model has been asked for it. The
	// request is what says it has started: counted from the kept preview alone, it was recorded a
	// moment later and mistaken for the reply to the other voice's words.
	s.eventually(func() bool { return s.keptPreviews() == 1 && len(s.model.requests()) == 1 },
		"the caller's own voice was not previewed")

	s.ears.emitter.Send(stt.Transcript{Participant: alice, Mode: stt.ModeReplacement, Speaker: "voice-2", Text: "who is at the door"})
	s.cadenceHolds(alice, "who is at the door")
	clock.advance(defaultPreviewDebounce)

	s.Never(func() bool { return len(s.model.requests()) > 1 }, 250*time.Millisecond, 10*time.Millisecond,
		"a reply was started for somebody else's words")
	s.Zero(s.keptPreviews())
}

func (s *AgentSuite) TestAnEarlyPreviewIsLetGoWhenTheFloorChanges() {
	alice, started := s.holdsAnEarlyPreview()

	s.agent.perform(Action{Kind: ActInterrupt, Participant: alice})

	s.True(started.Load(), "the floor changed under the reply")
	s.Zero(s.keptPreviews())
}

func (s *AgentSuite) TestAnEarlyPreviewIsLetGoWhenTheCallerLeaves() {
	alice, started := s.holdsAnEarlyPreview()

	s.edge.attending <- Attendance{Participant: alice, Joined: false}

	s.eventually(started.Load, "the reply for somebody who left was left running")
	s.Zero(s.keptPreviews())
}

func (s *AgentSuite) TestClosingTheAgentLetsGoOfAnEarlyPreview() {
	_, started := s.holdsAnEarlyPreview()

	s.Require().NoError(s.agent.Close())

	s.True(started.Load(), "the reply was left running")
	s.Zero(s.keptPreviews())
	s.Zero(s.previewsHeld())
}

func (s *AgentSuite) TestMovingASessionOntoOtherModelsLetsGoOfAnEarlyPreview() {
	w := s.joinSwappable()
	s.agent.cadence.mu.Lock()
	s.agent.cadence.preview = defaultPreviewDebounce
	s.agent.cadence.gap = time.Hour
	s.agent.cadence.mu.Unlock()
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)
	s.eventually(func() bool {
		s.agent.mu.Lock()
		_, listening := s.agent.listeners[alice.ID]
		s.agent.mu.Unlock()
		return listening && len(w.listener().transcribed()) > 0
	}, "nobody listened")
	w.listener().emitter.Send(stt.Transcript{Participant: alice, Mode: stt.ModeReplacement, Text: "please find a table", Language: "en"})
	s.eventually(func() bool { return s.keptPreviews() == 1 }, "no reply was started for the words")
	started := s.watchPreview()

	s.Require().NoError(s.agent.SetSettings(s.ctx, Settings{
		LLMTarget: "other/other-model", STTTarget: "stub/stub-model", TTSTarget: "other/other-model", Voice: "ada",
	}))

	s.True(started.Load(), "the reply was written by the model that was replaced")
	s.Zero(s.keptPreviews())
}

func (s *AgentSuite) TestARulingThatIsNotAnAnswerLetsGoOfAnEarlyPreview() {
	s.join(true)
	s.slowGap(300 * time.Millisecond)
	s.flow.reply = []string{`{"disposition":"ignore","floor":"continue"}`}
	alice := stt.Participant{ID: "child", Name: "Child"}
	s.speak(alice)

	s.mutters(alice, "mom where is my backpack")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "no reply was started for the words")
	s.Empty(s.flow.requests(), "the reply was started before the words were put to a ruling")
	s.eventually(func() bool { return len(s.flow.requests()) == 1 }, "the words were never asked about")
	s.eventually(func() bool { return s.previewsHeld() == 0 && s.keptPreviews() == 0 },
		"words that were ignored kept the reply started for them")
	s.Len(s.model.requests(), 1, "the candidate did not start a second reply")
	s.Zero(countOf[Responding](s.reported()))
	s.Empty(s.voice.spoken())
}

func (s *AgentSuite) TestAnEarlyPreviewIsLetGoWhenNothingComesOfTheWordsBeforeThePatienceRunsOut() {
	s.join(true)
	s.slowGap(time.Hour)
	clock := s.controlsTimers()
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)
	s.mutters(alice, "please find a table")
	s.cadenceHolds(alice, "please find a table")
	clock.advance(defaultPreviewDebounce)
	s.eventually(func() bool { return s.keptPreviews() == 1 }, "no reply was started for the words")
	started := s.watchPreview()
	s.False(started.Load(), "the reply was let go of before the patience for its words ran out")

	clock.advance(s.agent.converse.patienceSpan())

	s.eventually(started.Load, "the reply outlived the patience for its words")
	s.Zero(s.keptPreviews())
	s.Zero(s.previewsHeld())
}

func (s *AgentSuite) TestAnEarlyPreviewAdoptedByAWaitIsLetGoWhenThePatienceRunsOut() {
	s.join(true)
	s.slowGap(200 * time.Millisecond)
	s.flow.reply = []string{`{"disposition":"wait","floor":"continue"}`}
	s.noRetry()
	clock := s.controlsTimers()
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)
	s.mutters(alice, "please find a table")
	s.cadenceHolds(alice, "please find a table")
	clock.advance(defaultPreviewDebounce)
	s.eventually(func() bool { return s.keptPreviews() == 1 && len(s.model.requests()) == 1 },
		"no reply was started for the words")
	clock.advance(200*time.Millisecond - defaultPreviewDebounce)
	s.eventually(func() bool { return len(s.flow.requests()) == 1 }, "the words were never asked about")
	s.eventually(func() bool { return s.keptPreviews() == 1 && s.previewsHeld() == 1 }, "the wait did not keep the reply")
	started := s.watchPreview()
	s.False(started.Load(), "the reply was let go of before the patience for its words ran out")

	clock.advance(s.agent.converse.patienceSpan())

	s.eventually(started.Load, "the reply outlived the patience for its words")
	s.Zero(s.keptPreviews())
	s.Zero(s.previewsHeld())
	s.Len(s.model.requests(), 1, "the wait kept the reply that was started early")
}

func (s *AgentSuite) TestAnEarlyPreviewThatFailedLeavesNothingHeld() {
	s.join(true)
	s.model.refuses = errors.New("the model is down")
	s.slowGap(300 * time.Millisecond)
	clock := s.controlsTimers()
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	s.mutters(alice, "please find a table")
	s.cadenceHolds(alice, "please find a table")
	clock.advance(defaultPreviewDebounce)

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "no reply was started for the words")
	s.Empty(s.flow.requests(), "the reply was started before the words were put to a ruling")
	s.Empty(s.voice.spoken(), "a failed preview must stay silent until the caller is answered")
	clock.advance(300*time.Millisecond - defaultPreviewDebounce)
	s.eventually(func() bool { return countOf[Error](s.reported()) > 0 }, "the failure was never reported")
	s.eventually(func() bool { return s.previewsHeld() == 0 && s.keptPreviews() == 0 },
		"a reply that failed was left held")
	s.eventually(func() bool { return s.spokenText(lostReply) }, "the caller was not told their reply failed")
}
