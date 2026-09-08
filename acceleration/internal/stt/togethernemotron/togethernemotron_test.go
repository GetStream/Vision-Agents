package togethernemotron

import (
	"testing"

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

// drain collects the events emitted so far without blocking on an empty channel.
func (s *TogetherNemotronSuite) drain(provider *STT) []stt.Event {
	var events []stt.Event
	for {
		select {
		case event := <-provider.Events():
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
	provider.handleMessage(completed(" me?"))

	settled := s.finals(provider)
	s.Require().NotEmpty(settled)
	s.Equal("Can you hear me?", settled[len(settled)-1].Text,
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
	provider.handleMessage(completed("io."))

	settled := s.finals(provider)
	s.Require().NotEmpty(settled)
	s.Equal("this Saturday at seven thirty patio.", settled[len(settled)-1].Text)
}

// TestAFlushMidSentenceKeepsTheSpacingTheServerChose is the other half of the join. Here
// the continuation carries the space, so adding one would double it.
func (s *TogetherNemotronSuite) TestAFlushMidSentenceKeepsTheSpacingTheServerChose() {
	provider := s.newSTT(Options{})

	provider.handleMessage(completed("Hi, I'd like to book a table for"))
	provider.handleMessage(completed(" four."))

	settled := s.finals(provider)
	s.Require().NotEmpty(settled)
	s.Equal("Hi, I'd like to book a table for four.", settled[len(settled)-1].Text)
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

// TestAFinishedSentenceEndsTheUtterance is the boundary that does exist, and the reason
// the accumulation does not run for the length of the call.
func (s *TogetherNemotronSuite) TestAFinishedSentenceEndsTheUtterance() {
	provider := s.newSTT(Options{})

	provider.handleMessage(completed("In a quiet village."))
	provider.handleMessage(delta("Young Mia"))

	heard := s.transcripts(provider)
	s.Require().Len(heard, 2)
	s.Equal(int64(1), heard[0].Utterance)
	s.Equal(int64(2), heard[1].Utterance, "a new sentence is a new utterance")
	s.Equal("Young Mia", heard[1].Text,
		"the sentence that ended should not be carried into the one that follows")
}

// TestAQuestionAndAnExclamationEndAnUtteranceToo guards the other two ways the server ends
// a sentence, since only a full stop is obvious.
func (s *TogetherNemotronSuite) TestAQuestionAndAnExclamationEndAnUtteranceToo() {
	for _, sentence := range []string{"Can you hear me?", "Hello there!"} {
		provider := s.newSTT(Options{})

		provider.handleMessage(completed(sentence))
		provider.handleMessage(delta("Something else"))

		heard := s.transcripts(provider)
		s.Require().Len(heard, 2, sentence)
		s.Equal("Something else", heard[1].Text, sentence)
		s.Equal(int64(2), heard[1].Utterance, sentence)
	}
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

func (s *TogetherNemotronSuite) TestACompletedTranscriptSettlesTheTurn() {
	provider := s.newSTT(Options{})

	provider.handleMessage(completed("In a quiet village."))

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
	s.Equal("In a quiet village.", heard[0].Text)
	s.True(heard[0].Final())
}

func (s *TogetherNemotronSuite) TestDeltasShareTheUtteranceOfTheFinalTheyBecome() {
	provider := s.newSTT(Options{})

	provider.handleMessage(delta("in a quiet"))
	provider.handleMessage(delta("in a quiet village"))
	provider.handleMessage(completed("In a quiet village."))
	provider.handleMessage(delta("forgotten"))

	heard := s.transcripts(provider)
	s.Require().Len(heard, 4)
	s.Equal(int64(1), heard[0].Utterance)
	s.Equal(int64(1), heard[1].Utterance)
	s.Equal(int64(1), heard[2].Utterance, "the end of a run belongs to the run it ends")
	s.Equal(int64(2), heard[3].Utterance)
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
