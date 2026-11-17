package grok

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type GrokSuite struct {
	suite.Suite
}

func TestGrokSuite(t *testing.T) {
	suite.Run(t, new(GrokSuite))
}

// newSTT returns a provider that is wired up but never connected, so the event mapping
// can be exercised without touching the network.
func (s *GrokSuite) newSTT(options Options) *STT {
	if options.APIKey == "" {
		options.APIKey = "test-key"
	}
	provider, err := New(options)
	s.Require().NoError(err)
	return provider
}

// drain collects the events emitted so far without blocking on an empty channel.
func (s *GrokSuite) drain(provider *STT) []stt.Event {
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
func (s *GrokSuite) transcripts(provider *STT) []stt.Transcript {
	var found []stt.Transcript
	for _, event := range s.drain(provider) {
		transcript, ok := event.(stt.Transcript)
		s.Require().True(ok, "expected only transcripts, got %T", event)
		found = append(found, transcript)
	}
	return found
}

// interim is a hypothesis that may still change.
func interim(text string) serverMessage {
	return serverMessage{Type: eventPartial, Text: text}
}

// chunkFinal is text the server has settled while the caller keeps talking.
func chunkFinal(text string) serverMessage {
	return serverMessage{Type: eventPartial, Text: text, IsFinal: true}
}

// utteranceFinal is the settled turn.
func utteranceFinal(text string) serverMessage {
	return serverMessage{Type: eventPartial, Text: text, IsFinal: true, SpeechFinal: true}
}

func (s *GrokSuite) TestNewRequiresAPIKey() {
	s.T().Setenv(apiKeyEnvVar, "")

	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *GrokSuite) TestNewFallsBackToEnvAPIKey() {
	s.T().Setenv(apiKeyEnvVar, "from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("from-env", provider.options.APIKey)
}

func (s *GrokSuite) TestNewRejectsNonWebSocketURL() {
	_, err := New(Options{APIKey: "k", URL: "https://api.x.ai/v1/stt"})
	s.ErrorContains(err, "url must be ws:// or wss://")
}

func (s *GrokSuite) TestNewRejectsMoreKeytermsThanTheApiTakes() {
	terms := make([]string, stt.MaxKeyterms+1)
	for i := range terms {
		terms[i] = "term"
	}

	_, err := New(Options{APIKey: "k", Keyterms: terms})
	s.ErrorContains(err, "at most 100 keyterms")
}

func (s *GrokSuite) TestNewRejectsASmartTurnThresholdThatIsNotAConfidence() {
	_, err := New(Options{APIKey: "k", SmartTurn: 1.5})
	s.ErrorContains(err, "between 0 and 1")
}

func (s *GrokSuite) TestProviderAndModelAreReported() {
	provider := s.newSTT(Options{})
	s.Equal(ProviderName, provider.Provider())
	s.Equal(DefaultModel, provider.Model())
}

func (s *GrokSuite) TestAnInterimProducesAReplacementTranscript() {
	provider := s.newSTT(Options{})
	speaker := stt.Participant{ID: "p1", UserID: "u1"}
	provider.participant = speaker

	provider.handleMessage(serverMessage{
		Type:     eventPartial,
		Text:     "  in a quiet vill  ",
		Duration: 1.5,
		Language: "en",
	})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
	s.Equal("in a quiet vill", heard[0].Text, "surrounding whitespace should be trimmed")
	s.Equal(stt.ModeReplacement, heard[0].Mode)
	s.False(heard[0].Final())
	s.Equal(speaker, heard[0].Participant)
	s.Equal("en", heard[0].Language, "the server reports the language it recognised")
	s.Equal(ProviderName, heard[0].Provider)
	s.Equal(DefaultModel, heard[0].Model)
	s.InDelta(1500.0, heard[0].AudioDurationMs, 0.001, "the audio window is reported in seconds")
}

func (s *GrokSuite) TestAChunkFinalDoesNotSettleTheTurn() {
	// The caller is still talking: the words are locked, the sentence is not over. Settling
	// here would have whoever is listening answer mid-sentence.
	provider := s.newSTT(Options{})

	provider.handleMessage(chunkFinal("in a quiet village"))

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
	s.Equal(stt.ModeReplacement, heard[0].Mode)
	s.False(heard[0].Final())
}

func (s *GrokSuite) TestAnUtteranceFinalSettlesTheTurn() {
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{
		Type:                eventPartial,
		Text:                "In a quiet village.",
		IsFinal:             true,
		SpeechFinal:         true,
		Duration:            9.8,
		EndOfTurnConfidence: 0.983,
	})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
	s.Equal("In a quiet village.", heard[0].Text)
	s.True(heard[0].Final())
	s.InDelta(0.983, heard[0].Confidence, 0.001)
	s.InDelta(9800.0, heard[0].AudioDurationMs, 0.001)
}

func (s *GrokSuite) TestEveryHypothesisIsTakenAsTheWholeUtterance() {
	// The server restates the utterance from its start in every frame, chunk finals
	// included. Assembling them by appending would spell the sentence out several times.
	provider := s.newSTT(Options{})

	provider.handleMessage(interim("in a quiet"))
	provider.handleMessage(chunkFinal("in a quiet village"))
	provider.handleMessage(interim("in a quiet village nestled between"))

	heard := s.transcripts(provider)
	s.Require().Len(heard, 3)
	s.Equal("in a quiet", heard[0].Text)
	s.Equal("in a quiet village", heard[1].Text)
	s.Equal("in a quiet village nestled between", heard[2].Text)
}

func (s *GrokSuite) TestAWordTheServerRevisesIsNotKeptAlongsideItsReplacement() {
	// The server changes its mind about words it has already reported as the rest of the
	// sentence arrives. Only the latest reading of the utterance is the transcript.
	provider := s.newSTT(Options{})

	provider.handleMessage(interim("young Mia"))
	provider.handleMessage(chunkFinal("young meyer discovered a map"))
	provider.handleMessage(interim("young Mira discovered a map leading to"))
	provider.handleMessage(utteranceFinal("Young Mia discovered a map leading to forgotten treasures."))

	heard := s.transcripts(provider)
	s.Require().Len(heard, 4)
	s.Equal("young Mira discovered a map leading to", heard[2].Text)
	s.Equal("Young Mia discovered a map leading to forgotten treasures.", heard[3].Text)
}

func (s *GrokSuite) TestOneRunOfSpeechKeepsOneUtteranceNumber() {
	provider := s.newSTT(Options{})

	provider.handleMessage(interim("in a quiet"))
	provider.handleMessage(chunkFinal("in a quiet village"))
	provider.handleMessage(interim("nestled between"))
	provider.handleMessage(utteranceFinal("In a quiet village nestled between."))

	for _, transcript := range s.transcripts(provider) {
		s.Equal(int64(1), transcript.Utterance)
	}
}

func (s *GrokSuite) TestASecondRunOfSpeechIsNumberedApartFromTheFirst() {
	provider := s.newSTT(Options{})

	provider.handleMessage(interim("in a quiet"))
	provider.handleMessage(utteranceFinal("In a quiet village."))
	provider.handleMessage(interim("forgotten"))

	heard := s.transcripts(provider)
	s.Require().Len(heard, 3)
	s.Equal(int64(1), heard[0].Utterance)
	s.Equal(int64(1), heard[1].Utterance, "the end of a run belongs to the run it ends")
	s.Equal(int64(2), heard[2].Utterance)
}

func (s *GrokSuite) TestEmptyTranscriptsAreNotEmitted() {
	provider := s.newSTT(Options{})

	provider.handleMessage(interim("   "))

	s.Empty(s.drain(provider), "whitespace-only transcripts carry no information")
}

func (s *GrokSuite) TestSilenceAfterASettledTurnDoesNotOpenAnother() {
	// The server keeps reporting empty transcripts for as long as nobody is talking. Each
	// one would otherwise start a run of speech that never happened.
	provider := s.newSTT(Options{})

	provider.handleMessage(utteranceFinal("In a quiet village."))
	provider.handleMessage(interim(""))
	provider.handleMessage(chunkFinal(""))
	provider.handleMessage(interim("forgotten treasures"))

	heard := s.transcripts(provider)
	s.Require().Len(heard, 2)
	s.Equal(int64(1), heard[0].Utterance)
	s.Equal(int64(2), heard[1].Utterance, "the quiet between two turns is one boundary, not four")
}

func (s *GrokSuite) TestClosingSettlesTheTailOfACallThatWasCutOff() {
	// The caller hung up mid-sentence, so no silence ever ended the turn and this frame is
	// the only place those words arrive.
	provider := s.newSTT(Options{})

	provider.handleMessage(interim("in a quiet"))
	provider.handleMessage(serverMessage{Type: eventDone, Text: "In a quiet village.", Duration: 4})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 2)
	s.True(heard[1].Final())
	s.Equal("In a quiet village.", heard[1].Text)
	s.Equal(int64(1), heard[1].Utterance, "the tail belongs to the run of speech it ends")
}

func (s *GrokSuite) TestClosingASettledTurnDoesNotReportItTwice() {
	// After an ordinary turn this frame says the same words over again, and a caller
	// answered twice is worse than one not answered at all.
	provider := s.newSTT(Options{})

	provider.handleMessage(utteranceFinal("In a quiet village."))
	provider.handleMessage(serverMessage{Type: eventDone, Text: "In a quiet village.", Duration: 9.8})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
}

func (s *GrokSuite) TestTheFlushIsReleasedByTheLastTranscript() {
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{Type: eventDone, Text: "In a quiet village."})

	select {
	case <-provider.finished:
	default:
		s.Fail("transcript.done should release a Close that is waiting for the tail")
	}
}

func (s *GrokSuite) TestServerErrorIsFatal() {
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{Type: eventError, Message: "sample_rate must be 16000"})

	events := s.drain(provider)
	s.Require().Len(events, 1)
	failure, ok := events[0].(stt.Error)
	s.Require().True(ok)
	s.True(failure.Fatal)
	s.ErrorContains(failure, "sample_rate must be 16000")
}

func (s *GrokSuite) TestTheReadyFrameIsNotAnSTTEvent() {
	// The handshake already reported the session as connected, so a second event here
	// would say it twice.
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{Type: eventCreated})

	s.Empty(s.drain(provider))
}

func (s *GrokSuite) TestTheEndpointCarriesTheSessionConfiguration() {
	provider := s.newSTT(Options{
		Language:           "de",
		EndpointingMs:      600,
		SmartTurn:          0.7,
		SmartTurnTimeoutMs: 3000,
		Keyterms:           []string{"Vision Agents", "  ", "Stream"},
	})

	endpoint := provider.endpoint()
	s.Contains(endpoint, "sample_rate=16000")
	s.Contains(endpoint, "encoding=pcm")
	s.Contains(endpoint, "interim_results=true")
	s.Contains(endpoint, "language=de")
	s.Contains(endpoint, "endpointing=600")
	s.Contains(endpoint, "smart_turn=0.7")
	s.Contains(endpoint, "smart_turn_timeout=3000")
	s.Contains(endpoint, "keyterm=Vision+Agents")
	s.Contains(endpoint, "keyterm=Stream")
}

func (s *GrokSuite) TestTheEndpointLeavesUnsetOptionsToTheServer() {
	endpoint := s.newSTT(Options{}).endpoint()

	s.NotContains(endpoint, "language=")
	s.NotContains(endpoint, "endpointing=")
	s.NotContains(endpoint, "smart_turn")
	s.NotContains(endpoint, "keyterm")
	s.NotContains(endpoint, "diarize")
}

func (s *GrokSuite) TestDiarizationIsAskedForOnTheSocket() {
	s.Contains(s.newSTT(Options{Diarize: true}).endpoint(), "diarize=true")
}

func (s *GrokSuite) TestTheTurnIsReportedInTheVoiceMostOfItWasSpokenIn() {
	// The server labels each word, and a word of somebody else's speech caught at the
	// edge of a turn should not rename the whole of it.
	provider := s.newSTT(Options{Diarize: true})

	provider.handleMessage(serverMessage{
		Type: eventPartial, Text: "thanks for calling how can I help", IsFinal: true, SpeechFinal: true,
		Words: []word{
			{Text: "thanks", Speaker: 1},
			{Text: "for", Speaker: 0},
			{Text: "calling", Speaker: 0},
			{Text: "how", Speaker: 0},
			{Text: "can", Speaker: 0},
			{Text: "I", Speaker: 0},
			{Text: "help", Speaker: 0},
		},
	})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
	s.Equal("0", heard[0].Speaker)
}

func (s *GrokSuite) TestATranscriptNamesNoVoiceWhenDiarizationIsOff() {
	// Zero is a label the server uses, so the number alone cannot say whether anybody was
	// named. A transcript that claimed voice "0" would have the agent believe it could
	// tell two people at one microphone apart when it was never asked to listen for it.
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{
		Type: eventPartial, Text: "hello", IsFinal: true, SpeechFinal: true,
		Words: []word{{Text: "hello", Speaker: 0}},
	})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
	s.Empty(heard[0].Speaker)
}

func (s *GrokSuite) TestASmartTurnTimeoutWithoutSmartTurnIsNotSent() {
	endpoint := s.newSTT(Options{SmartTurnTimeoutMs: 3000}).endpoint()

	s.NotContains(endpoint, "smart_turn_timeout",
		"the server only honours the timeout when Smart Turn is on")
}

func (s *GrokSuite) TestProcessAudioRejectsWrongAudioFormat() {
	provider := s.newSTT(Options{})

	err := provider.ProcessAudio(stt.PcmData{SampleRate: 48000, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "sample rate must be 16000")
}

func (s *GrokSuite) TestProcessAudioFailsBeforeStart() {
	provider := s.newSTT(Options{})

	err := provider.ProcessAudio(stt.PcmData{SampleRate: stt.SampleRate, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "not started")
}

func (s *GrokSuite) TestProcessAudioFailsAfterClose() {
	provider := s.newSTT(Options{})
	s.Require().NoError(provider.Close())

	err := provider.ProcessAudio(stt.PcmData{SampleRate: stt.SampleRate, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "session closed")
}

func (s *GrokSuite) TestCloseIsIdempotentAndClosesEvents() {
	provider := s.newSTT(Options{})

	s.Require().NoError(provider.Close())
	s.Require().NoError(provider.Close())

	_, open := <-provider.Events()
	s.False(open, "closing the session should close the event channel")
}

func (s *GrokSuite) TestSatisfiesSTTInterface() {
	var _ stt.STT = s.newSTT(Options{})
}
