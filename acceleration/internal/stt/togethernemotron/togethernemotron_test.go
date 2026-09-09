package togethernemotron

import (
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type TogetherNemotronSuite struct {
	suite.Suite
}

func TestTogetherNemotronSuite(t *testing.T) {
	suite.Run(t, new(TogetherNemotronSuite))
}

// newSTT returns a provider that is wired up but never connected, so the event mapping
// can be exercised without touching the network.
func (s *TogetherNemotronSuite) newSTT(options Options) *STT {
	if options.APIKey == "" {
		options.APIKey = "test-key"
	}
	provider, err := New(options)
	s.Require().NoError(err)
	return provider
}

// drain collects the events emitted so far without blocking on an empty channel. A closed
// channel reads forever, so a session that has been hung up ends the drain rather than
// filling the slice with nothing.
func (s *TogetherNemotronSuite) drain(provider *STT) []stt.Event {
	var events []stt.Event
	for {
		select {
		case event, open := <-provider.Events():
			if !open {
				return events
			}
			events = append(events, event)
		default:
			return events
		}
	}
}

// transcripts is the drained events as transcripts, which is what most of these are about.
func (s *TogetherNemotronSuite) transcripts(provider *STT) []stt.Transcript {
	var found []stt.Transcript
	for _, event := range s.drain(provider) {
		transcript, ok := event.(stt.Transcript)
		s.Require().True(ok, "expected only transcripts, got %T", event)
		found = append(found, transcript)
	}
	return found
}

// delta is more of what the caller seems to be saying.
func delta(text string) serverMessage {
	return serverMessage{Type: eventDelta, Delta: text}
}

// completed is the settled utterance.
func completed(text string) serverMessage {
	return serverMessage{Type: eventCompleted, Transcript: text}
}

func (s *TogetherNemotronSuite) TestNewRequiresAPIKey() {
	s.T().Setenv(apiKeyEnvVar, "")

	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *TogetherNemotronSuite) TestNewFallsBackToEnvAPIKey() {
	s.T().Setenv(apiKeyEnvVar, "from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("from-env", provider.options.APIKey)
}

func (s *TogetherNemotronSuite) TestNewRejectsNonWebSocketURL() {
	_, err := New(Options{APIKey: "k", URL: "https://api.together.ai/v1/realtime"})
	s.ErrorContains(err, "url must be ws:// or wss://")
}

func (s *TogetherNemotronSuite) TestProviderAndModelAreReported() {
	provider := s.newSTT(Options{})
	s.Equal(ProviderName, provider.Provider())
	s.Equal(DefaultModel, provider.Model())
}

func (s *TogetherNemotronSuite) TestAnEnglishCallGetsTheEnglishModelUnlessAskedOtherwise() {
	// NVIDIA recommend the English model over the multilingual one for English, so the
	// default is the narrower of the two rather than the one that covers everything.
	s.Equal(DefaultModel, s.newSTT(Options{}).Model())
	s.NotEqual(MultilingualModel, DefaultModel)
}

func (s *TogetherNemotronSuite) TestTheMultilingualModelIsServedByTheSameProvider() {
	// Both models are one socket and one protocol, so asking for the multilingual one is
	// a model choice rather than a second provider.
	provider := s.newSTT(Options{Model: MultilingualModel})

	s.Equal(ProviderName, provider.Provider())
	s.Equal(MultilingualModel, provider.Model())
}

func (s *TogetherNemotronSuite) TestTheProviderIsNamedApartFromTheOtherModelOnTheSameSocket() {
	// Together's Parakeet is the same vendor and the same realtime endpoint, but a
	// different model family, and routing names a provider to pick between them.
	s.NotEqual("together-parakeet", ProviderName)
}

func (s *TogetherNemotronSuite) TestADeltaProducesAReplacementTranscript() {
	provider := s.newSTT(Options{})
	speaker := stt.Participant{ID: "p1", UserID: "u1"}
	provider.participant = speaker

	provider.handleMessage(serverMessage{Type: eventDelta, Delta: "  in a quiet vill  "})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
	s.Equal("in a quiet vill", heard[0].Text, "surrounding whitespace should be trimmed")
	s.Equal(stt.ModeReplacement, heard[0].Mode, "each delta restates the utterance")
	s.False(heard[0].Final())
	s.Equal(speaker, heard[0].Participant)
	s.Equal(ProviderName, heard[0].Provider)
	s.Equal(DefaultModel, heard[0].Model)
}

// texts is the transcripts as their text, for the sequences below where the wording is
// the whole point.
func (s *TogetherNemotronSuite) texts(provider *STT) []string {
	var said []string
	for _, transcript := range s.transcripts(provider) {
		said = append(said, transcript.Text)
	}
	return said
}

// finals is the settled transcripts only, which is what a caller acting on the turn reads.
func (s *TogetherNemotronSuite) finals(provider *STT) []stt.Transcript {
	var settled []stt.Transcript
	for _, transcript := range s.transcripts(provider) {
		if transcript.Final() {
			settled = append(settled, transcript)
		}
	}
	return settled
}

// quiet is the caller falling silent for long enough that the turn is over, which is the
// only end of turn this protocol has.
func (s *TogetherNemotronSuite) quiet(provider *STT) {
	provider.turnOver()
}

// TestTheWholeQuestionSettlesAndNotItsLastWord is the "can you hear me" report, as the
// server actually sends it: it flushes its decoder after "Can you hear", then the deltas
// begin again from nothing and only carry " me". Publishing that flush as a finished turn
// left the last word standing alone as the whole of what the caller had asked.
func (s *TogetherNemotronSuite) TestTheWholeQuestionSettlesAndNotItsLastWord() {
	provider := s.newSTT(Options{})

	provider.handleMessage(delta("Can"))
	provider.handleMessage(delta("Can you"))
	provider.handleMessage(delta("Can you hear"))
	provider.handleMessage(completed("Can you hear"))
	provider.handleMessage(delta(" me"))
	provider.handleMessage(delta(" me?"))
	s.quiet(provider)

	settled := s.finals(provider)
	s.Require().Len(settled, 1)
	s.Equal("Can you hear me?", settled[0].Text,
		"the turn should settle on the whole question, not the fragment the last flush carried")
	for _, final := range settled {
		s.NotEqual("me?", final.Text, "no turn is only the last word of the question")
	}
}

// TestASegmentFlushKeepsTheWordsAlreadyHeard is the same defect on the hypotheses rather
// than the finals. A delta after a flush carries only the new words, so reporting it on
// its own would blank the sentence the caller had been watching build up.
func (s *TogetherNemotronSuite) TestASegmentFlushKeepsTheWordsAlreadyHeard() {
	provider := s.newSTT(Options{})

	provider.handleMessage(delta("Can you hear"))
	provider.handleMessage(completed("Can you hear"))
	provider.handleMessage(delta(" me"))

	s.Equal([]string{"Can you hear", "Can you hear", "Can you hear me"}, s.texts(provider),
		"a hypothesis after a flush should still be the whole utterance")
}

// TestAFlushInTheMiddleOfAWordDoesNotSplitIt is the sequence recorded off the wire: the
// server settled "…seven thirty pat" and the deltas resumed at "io". The join has to be
// exactly what it sent, so trimming either side would spell the word "pat io".
func (s *TogetherNemotronSuite) TestAFlushInTheMiddleOfAWordDoesNotSplitIt() {
	provider := s.newSTT(Options{})

	provider.handleMessage(delta("this Saturday at seven thirty patio"))
	provider.handleMessage(completed("this Saturday at seven thirty pat"))
	provider.handleMessage(delta("io"))
	provider.handleMessage(delta("io."))
	s.quiet(provider)

	settled := s.finals(provider)
	s.Require().Len(settled, 1)
	s.Equal("this Saturday at seven thirty patio.", settled[0].Text)
}

// TestAFlushMidSentenceKeepsTheSpacingTheServerChose is the other half of the join. Here
// the continuation carries the space, so adding one would double it.
func (s *TogetherNemotronSuite) TestAFlushMidSentenceKeepsTheSpacingTheServerChose() {
	provider := s.newSTT(Options{})

	provider.handleMessage(completed("Hi, I'd like to book a table for"))
	provider.handleMessage(completed(" four."))
	s.quiet(provider)

	settled := s.finals(provider)
	s.Require().Len(settled, 1)
	s.Equal("Hi, I'd like to book a table for four.", settled[0].Text)
}

// TestSegmentsOfOneSentenceAreOneUtterance is what stops the router treating the tail of a
// question as a new turn: the flush that split it was the decoder's, not the caller's.
func (s *TogetherNemotronSuite) TestSegmentsOfOneSentenceAreOneUtterance() {
	provider := s.newSTT(Options{})

	provider.handleMessage(completed("Can you hear"))
	provider.handleMessage(delta(" me"))
	provider.handleMessage(completed(" me?"))

	for _, transcript := range s.transcripts(provider) {
		s.Equal(int64(1), transcript.Utterance,
			"one question split by a buffer flush is still one utterance")
	}
}

// TestATurnDoesNotBeginWithTheEndOfTheOneBefore is the "boulder" report, replayed off the
// wire. The server's segment does not end when the caller stops talking: it settles
// "…the big bould" and its next deltas carry "er." forward into the following turn. Left
// on, the caller's last word opens a sentence they never started it with.
func (s *TogetherNemotronSuite) TestATurnDoesNotBeginWithTheEndOfTheOneBefore() {
	provider := s.newSTT(Options{})

	provider.handleMessage(delta("Let us meet by the big"))
	provider.handleMessage(completed("Let us meet by the big bould"))
	provider.handleMessage(delta("er"))
	provider.handleMessage(delta("er."))
	s.quiet(provider)

	provider.handleMessage(delta("er. What"))
	provider.handleMessage(delta("er. What time"))
	s.quiet(provider)

	settled := s.finals(provider)
	s.Require().Len(settled, 2)
	s.Equal("Let us meet by the big boulder.", settled[0].Text)
	s.Equal("What time", settled[1].Text,
		"the word the last turn ended on should not open this one")
}

// TestAWholeTurnIsNotRepeatedIntoTheNext is the same leak at segment scale, which is what
// a turn ending on a word the server did not punctuate used to produce: everything the
// caller had already said, prepended to everything they said next.
func (s *TogetherNemotronSuite) TestAWholeTurnIsNotRepeatedIntoTheNext() {
	provider := s.newSTT(Options{})

	provider.handleMessage(completed("In a quiet village"))
	provider.handleMessage(delta("."))
	s.quiet(provider)

	provider.handleMessage(delta(". Hi"))
	provider.handleMessage(delta(". Hi, I'd like a table"))
	s.quiet(provider)

	settled := s.finals(provider)
	s.Require().Len(settled, 2)
	s.Equal("In a quiet village.", settled[0].Text)
	s.Equal("Hi, I'd like a table", settled[1].Text)
	s.NotContains(settled[1].Text, "quiet village",
		"a turn that settled without a full stop is still over")
}

// TestATurnEndsWhenTheWordsStopRatherThanWhenTheServerPunctuates is the boundary itself.
// The server never says a turn ended and its full stop arrives a flush late, so the only
// thing left to read is the caller having gone quiet.
func (s *TogetherNemotronSuite) TestATurnEndsWhenTheWordsStopRatherThanWhenTheServerPunctuates() {
	provider := s.newSTT(Options{TurnGrace: 20 * time.Millisecond})

	provider.handleMessage(completed("Can you hear me."))
	s.Empty(s.finals(provider), "a full stop is not the caller stopping")

	s.Eventually(func() bool { return len(s.finals(provider)) == 1 },
		time.Second, 10*time.Millisecond, "silence should end the turn")
}

// TestSilenceIsWhatMovesTheUtteranceOn pairs with it: a turn nothing ended is still the
// same turn, however many times the decoder flushed inside it.
func (s *TogetherNemotronSuite) TestSilenceIsWhatMovesTheUtteranceOn() {
	provider := s.newSTT(Options{})

	provider.handleMessage(completed("In a quiet village."))
	provider.handleMessage(delta("Young Mia"))
	s.quiet(provider)
	provider.handleMessage(delta("found a map"))

	heard := s.transcripts(provider)
	s.Require().NotEmpty(heard)
	s.Equal(int64(1), heard[0].Utterance)
	s.Equal(int64(2), heard[len(heard)-1].Utterance, "the turn after a silence is a new one")
	s.Equal("found a map", heard[len(heard)-1].Text)
}

func (s *TogetherNemotronSuite) TestTranscriptsNameTheModelThatHeardThem() {
	// Two models serve this provider, so a transcript that only named the provider would
	// not say which of them produced it.
	provider := s.newSTT(Options{Model: MultilingualModel})

	provider.handleMessage(completed("In a quiet village."))

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
	s.Equal(MultilingualModel, heard[0].Model)
}

func (s *TogetherNemotronSuite) TestACompletedTranscriptRevisesTheTurnRatherThanEndingIt() {
	// A completed frame is the decoder flushing, which it does mid-word. Publishing it as
	// settled hands a caller half a sentence to act on.
	provider := s.newSTT(Options{})

	provider.handleMessage(completed("In a quiet village."))

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
	s.Equal("In a quiet village.", heard[0].Text)
	s.False(heard[0].Final())
}

func (s *TogetherNemotronSuite) TestDeltasShareTheUtteranceOfTheFinalTheyBecome() {
	provider := s.newSTT(Options{})

	provider.handleMessage(delta("in a quiet"))
	provider.handleMessage(delta("in a quiet village"))
	provider.handleMessage(completed("In a quiet village."))
	s.quiet(provider)
	provider.handleMessage(delta("forgotten"))

	heard := s.transcripts(provider)
	s.Require().Len(heard, 5)
	s.Equal(int64(1), heard[0].Utterance)
	s.Equal(int64(1), heard[1].Utterance)
	s.Equal(int64(1), heard[2].Utterance)
	s.Equal(int64(1), heard[3].Utterance, "the end of a run belongs to the run it ends")
	s.True(heard[3].Final())
	s.Equal(int64(2), heard[4].Utterance)
}

func (s *TogetherNemotronSuite) TestHangingUpSettlesWhatTheCallerHadJustSaid() {
	// The grace period has not run out when somebody hangs up mid-sentence, and the words
	// they got out are still owed to whoever was listening.
	provider := s.newSTT(Options{})

	provider.handleMessage(delta("Can you hear"))
	s.Require().NoError(provider.Close())

	settled := s.finals(provider)
	s.Require().Len(settled, 1)
	s.Equal("Can you hear", settled[0].Text)
}

func (s *TogetherNemotronSuite) TestEmptyTranscriptsAreNotEmitted() {
	provider := s.newSTT(Options{})

	provider.handleMessage(delta("   "))
	provider.handleMessage(completed("  "))

	s.Empty(s.drain(provider), "whitespace-only transcripts carry no information")
}

func (s *TogetherNemotronSuite) TestASettledUtteranceReleasesAWaitingClose() {
	provider := s.newSTT(Options{})

	provider.handleMessage(completed("In a quiet village."))

	select {
	case <-provider.settled:
	default:
		s.Fail("a settled utterance should release a Close waiting for the tail")
	}
}

func (s *TogetherNemotronSuite) TestAnEmptyCommitStillReleasesAWaitingClose() {
	// There was nothing left to transcribe. Close has its answer all the same, and waiting
	// out the timeout would spend it on every hangup.
	provider := s.newSTT(Options{})

	provider.handleMessage(completed(""))

	select {
	case <-provider.settled:
	default:
		s.Fail("an empty transcript is still an answer")
	}
}

func (s *TogetherNemotronSuite) TestAFailedUtteranceIsNotFatal() {
	// Together's protocol says the session carries on, so tearing it down would end a call
	// over one utterance that could not be transcribed.
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{Type: eventFailed, Message: "decode failed"})

	events := s.drain(provider)
	s.Require().Len(events, 1)
	failure, ok := events[0].(stt.Error)
	s.Require().True(ok)
	s.False(failure.Fatal)
	s.ErrorContains(failure, "decode failed")
}

func (s *TogetherNemotronSuite) TestAFailedUtteranceReleasesAWaitingClose() {
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{Type: eventFailed, Message: "decode failed"})

	select {
	case <-provider.settled:
	default:
		s.Fail("a hangup should not wait out the timeout for words the server has given up on")
	}
}

func (s *TogetherNemotronSuite) TestServerErrorIsFatal() {
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{Type: eventError, Message: "model not available"})

	events := s.drain(provider)
	s.Require().Len(events, 1)
	failure, ok := events[0].(stt.Error)
	s.Require().True(ok)
	s.True(failure.Fatal)
	s.ErrorContains(failure, "model not available")
}

func (s *TogetherNemotronSuite) TestAnErrorNestedTheRealtimeWayIsStillReported() {
	// Together documents the message at the top level; the realtime protocol they mirror
	// nests it. A failure nobody can read is worse than either.
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{
		Type:  eventError,
		Error: &serverError{Message: "invalid api key"},
	})

	events := s.drain(provider)
	s.Require().Len(events, 1)
	s.ErrorContains(events[0].(stt.Error), "invalid api key")
}

func (s *TogetherNemotronSuite) TestTheSessionCreatedFrameIsNotAnSTTEvent() {
	// The handshake already reported the session as connected.
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{Type: eventSessionCreated})

	s.Empty(s.drain(provider))
}

func (s *TogetherNemotronSuite) TestTheEndpointNamesTheModelAndAsksToTranscribe() {
	endpoint := s.newSTT(Options{}).endpoint()

	s.Contains(endpoint, "intent=transcription")
	s.Contains(endpoint, "model=nvidia%2Fnemotron-3-asr-streaming-0.6b")
	s.Contains(endpoint, "input_audio_format=pcm_s16le_16000")
}

func (s *TogetherNemotronSuite) TestTheEndpointNamesTheMultilingualModelWhenAskedForIt() {
	endpoint := s.newSTT(Options{Model: MultilingualModel}).endpoint()

	s.Contains(endpoint, "model=nvidia%2Fnemotron-3.5-asr-streaming-0.6b")
}

func (s *TogetherNemotronSuite) TestProcessAudioRejectsWrongAudioFormat() {
	provider := s.newSTT(Options{})

	err := provider.ProcessAudio(stt.PcmData{SampleRate: 48000, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "sample rate must be 16000")
}

func (s *TogetherNemotronSuite) TestProcessAudioFailsBeforeStart() {
	provider := s.newSTT(Options{})

	err := provider.ProcessAudio(stt.PcmData{SampleRate: stt.SampleRate, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "not started")
}

func (s *TogetherNemotronSuite) TestProcessAudioFailsAfterClose() {
	provider := s.newSTT(Options{})
	s.Require().NoError(provider.Close())

	err := provider.ProcessAudio(stt.PcmData{SampleRate: stt.SampleRate, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "session closed")
}

func (s *TogetherNemotronSuite) TestCloseIsIdempotentAndClosesEvents() {
	provider := s.newSTT(Options{})

	s.Require().NoError(provider.Close())
	s.Require().NoError(provider.Close())

	_, open := <-provider.Events()
	s.False(open, "closing the session should close the event channel")
}

func (s *TogetherNemotronSuite) TestSatisfiesSTTInterface() {
	var _ stt.STT = s.newSTT(Options{})
}
