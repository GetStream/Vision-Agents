package gemini

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type GeminiSuite struct {
	suite.Suite
}

func TestGeminiSuite(t *testing.T) {
	suite.Run(t, new(GeminiSuite))
}

// newProvider returns a provider that is wired up but never connected, so the event
// mapping can be exercised without touching the network.
func (s *GeminiSuite) newProvider() *STS {
	provider, err := New(Options{APIKey: "test-key", Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	return provider
}

// drain collects the events emitted so far without blocking on an empty channel.
func (s *GeminiSuite) drain(provider *STS) []sts.Event {
	var events []sts.Event
	for {
		select {
		case event := <-provider.Events():
			events = append(events, event)
		default:
			return events
		}
	}
}

// completed is the one reply settled so far.
func (s *GeminiSuite) completed(provider *STS) sts.ResponseComplete {
	var completes []sts.ResponseComplete
	for _, event := range s.drain(provider) {
		if typed, ok := event.(sts.ResponseComplete); ok {
			completes = append(completes, typed)
		}
	}
	s.Require().Len(completes, 1)
	return completes[0]
}

func heard(text string) serverMessage {
	return serverMessage{ServerContent: &serverContent{InputTranscription: &transcription{Text: text}}}
}

func spoke(samples int) serverMessage {
	return serverMessage{ServerContent: &serverContent{ModelTurn: &content{Role: "model", Parts: []part{{
		InlineData: &blob{
			Data:     base64.StdEncoding.EncodeToString(make([]byte, samples*2)),
			MimeType: "audio/pcm;rate=24000",
		},
	}}}}}
}

func turnComplete() serverMessage {
	return serverMessage{ServerContent: &serverContent{TurnComplete: true}}
}

func (s *GeminiSuite) TestNewRequiresAnAPIKey() {
	s.T().Setenv("GOOGLE_API_KEY", "")
	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *GeminiSuite) TestWhatIsTakenAtSetupCannotChangeAfterwards() {
	provider := s.newProvider()
	s.ErrorIs(provider.SetInstructions("x"), sts.ErrInstructionsFixed)
	s.ErrorIs(provider.SetTools([]llm.Tool{{Name: "x"}}), sts.ErrToolsFixed)
	s.False(provider.Capabilities().InstructionsMidSession)
	s.True(provider.Capabilities().Resumable)
	s.Equal(maxDuration, provider.Capabilities().MaxDuration)
	s.Equal(OutputSampleRate, provider.SampleRate())
}

func (s *GeminiSuite) TestTheCallerIsTakenToHaveStoppedAtTheLastWordHeard() {
	provider := s.newProvider()
	provider.participant = sts.Participant{ID: "alice"}

	provider.handleMessage(heard("hello "))
	provider.handleMessage(heard("world"))
	time.Sleep(20 * time.Millisecond)
	provider.handleMessage(spoke(2400))
	provider.handleMessage(turnComplete())

	events := s.drain(provider)
	s.Require().Len(events, 8)

	started, ok := events[0].(sts.SpeechStarted)
	s.Require().True(ok, "the first word heard is the only sign the caller began")
	s.Equal("alice", started.Participant.ID)
	first, ok := events[1].(sts.InputTranscript)
	s.Require().True(ok)
	s.Equal(stt.ModeDelta, first.Mode)
	s.Equal("hello ", first.Text)
	_, ok = events[2].(sts.InputTranscript)
	s.Require().True(ok, "the second piece is another delta, not another start")

	settled, ok := events[3].(sts.InputTranscript)
	s.Require().True(ok)
	s.Equal(stt.ModeFinal, settled.Mode)
	s.Equal("hello world", settled.Text, "the model beginning to speak is what settles what it heard")
	_, ok = events[4].(sts.SpeechStopped)
	s.Require().True(ok)

	begun, ok := events[5].(sts.ResponseStarted)
	s.Require().True(ok)
	s.NotEmpty(begun.ResponseID, "this API has no response ids, so the reply gets a generated ID")
	chunk, ok := events[6].(sts.AudioChunk)
	s.Require().True(ok)
	s.Equal(OutputSampleRate, chunk.Audio.SampleRate)
	s.Len(chunk.Audio.Samples, 2400)

	complete, ok := events[7].(sts.ResponseComplete)
	s.Require().True(ok)
	s.Equal(begun.ResponseID, complete.ResponseID)
	s.False(complete.Interrupted)
	s.GreaterOrEqual(complete.TimeToFirstByteMs, 20.0, "the wait is timed from the last word heard")
	s.InDelta(100, complete.AudioDurationMs, 0.001)
}

// The words of a reply can come before its audio. They still find the caller's turn
// settled first, so each reply's wait starts at its own caller's last word.
func (s *GeminiSuite) TestAReplyWhoseWordsComeFirstStillFollowsTheCallerStopping() {
	provider := s.newProvider()
	provider.participant = sts.Participant{ID: "alice"}

	for _, words := range []string{"my favorite color is blue", "and green"} {
		provider.handleMessage(heard(words))
		time.Sleep(20 * time.Millisecond)
		provider.handleMessage(serverMessage{ServerContent: &serverContent{OutputTranscription: &transcription{Text: "Nice."}}})
		provider.handleMessage(spoke(2400))
		provider.handleMessage(turnComplete())

		events := s.drain(provider)
		var kinds []string
		for _, event := range events {
			kinds = append(kinds, fmt.Sprintf("%T", event))
		}
		s.Require().Equal([]string{
			"sts.SpeechStarted", "sts.InputTranscript", "sts.InputTranscript", "sts.SpeechStopped",
			"sts.ResponseStarted", "sts.OutputTranscript", "sts.AudioChunk", "sts.ResponseComplete",
		}, kinds, "the caller stops before the reply to them starts")

		settled := events[2].(sts.InputTranscript)
		s.Equal(stt.ModeFinal, settled.Mode)
		s.Equal(words, settled.Text, "what the reply answers is known when it starts")
		stopped := events[3].(sts.SpeechStopped)
		begun := events[4].(sts.ResponseStarted)
		s.False(begun.At.Before(stopped.At))

		complete := events[7].(sts.ResponseComplete)
		s.GreaterOrEqual(complete.TimeToFirstByteMs, 20.0, "the wait is timed from this caller's last word")
		s.Less(complete.TimeToFirstByteMs, 100.0, "not from the turn before")
		time.Sleep(100 * time.Millisecond)
	}
}

func (s *GeminiSuite) TestTheServerInterruptingSettlesTheReplyAsCutOff() {
	provider := s.newProvider()

	provider.handleMessage(spoke(240))
	provider.handleMessage(serverMessage{ServerContent: &serverContent{Interrupted: true}})
	provider.handleMessage(spoke(240))

	events := s.drain(provider)
	var completes []sts.ResponseComplete
	var starts int
	for _, event := range events {
		switch typed := event.(type) {
		case sts.ResponseComplete:
			completes = append(completes, typed)
		case sts.ResponseStarted:
			starts++
		}
	}
	s.Require().Len(completes, 1)
	s.True(completes[0].Interrupted)
	s.Equal(2, starts, "audio after the interruption is the next reply")
}

// A caller who barges in is still being heard when the rest of the cut-off reply arrives.
// None of it may settle their turn or open a reply.
func (s *GeminiSuite) TestWhatRemainsOfACutOffReplyDoesNotEndTheBargeIn() {
	remainders := map[string]serverMessage{
		"words": {ServerContent: &serverContent{OutputTranscription: &transcription{Text: "and the rest"}}},
		"audio": spoke(240),
	}
	for name, remainder := range remainders {
		s.Run(name, func() {
			provider := s.newProvider()
			provider.participant = sts.Participant{ID: "alice"}

			provider.handleMessage(spoke(240))
			s.Require().NoError(provider.Interrupt(0))
			s.drain(provider)
			provider.handleMessage(heard("wait "))
			provider.handleMessage(remainder)

			var kinds []string
			for _, event := range s.drain(provider) {
				kinds = append(kinds, fmt.Sprintf("%T", event))
			}
			s.Equal([]string{"sts.SpeechStarted", "sts.InputTranscript"}, kinds, "the caller is still speaking")
		})
	}
}

func (s *GeminiSuite) TestALocalInterruptMutesTheRestOfTheTurn() {
	provider := s.newProvider()

	provider.handleMessage(spoke(240))
	s.Require().NoError(provider.Interrupt(0))
	// The model has not heard about it and keeps talking; none of this may get through.
	provider.handleMessage(spoke(240))
	provider.handleMessage(spoke(240))
	provider.handleMessage(turnComplete())
	// The next turn is a new reply and is heard.
	provider.handleMessage(spoke(240))

	events := s.drain(provider)
	var chunks []sts.AudioChunk
	var completes []sts.ResponseComplete
	for _, event := range events {
		switch typed := event.(type) {
		case sts.AudioChunk:
			chunks = append(chunks, typed)
		case sts.ResponseComplete:
			completes = append(completes, typed)
		}
	}
	s.Require().Len(completes, 1)
	s.True(completes[0].Interrupted)
	s.Require().Len(chunks, 2, "one before the interrupt, one from the next reply")
	s.Equal(1, chunks[0].Generation)
	s.Equal(2, chunks[1].Generation)
}

func (s *GeminiSuite) TestWhatTheModelSaidIsForwardedAsDeltas() {
	provider := s.newProvider()

	provider.handleMessage(serverMessage{ServerContent: &serverContent{OutputTranscription: &transcription{Text: "Hello "}}})
	provider.handleMessage(serverMessage{ServerContent: &serverContent{OutputTranscription: &transcription{Text: "there."}}})

	events := s.drain(provider)
	s.Require().Len(events, 3, "a reply is opened by the first thing the model says about it")
	started, ok := events[0].(sts.ResponseStarted)
	s.Require().True(ok)
	said, ok := events[1].(sts.OutputTranscript)
	s.Require().True(ok)
	s.Equal(stt.ModeDelta, said.Mode)
	s.Equal("Hello ", said.Text)
	s.Equal(started.ResponseID, said.ResponseID)
}

func (s *GeminiSuite) TestAToolCallRemembersTheNameItsAnswerNeeds() {
	provider := s.newProvider()

	provider.handleMessage(serverMessage{ToolCall: &toolCall{FunctionCalls: []functionCall{
		{ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{"city":"Paris"}`)},
		{ID: "call_2", Name: "get_time"},
	}}})

	events := s.drain(provider)
	var calls []sts.ToolCall
	for _, event := range events {
		if typed, ok := event.(sts.ToolCall); ok {
			calls = append(calls, typed)
		}
	}
	s.Require().Len(calls, 2)
	s.Equal("call_1", calls[0].CallID)
	s.Equal("get_weather", calls[0].Name)
	s.JSONEq(`{"city":"Paris"}`, calls[0].Arguments)
	s.Equal("{}", calls[1].Arguments, "a call with no arguments still carries an object")
	s.Equal("get_weather", provider.calls["call_1"], "the answer has to carry the name back")

	s.ErrorContains(provider.Answer("call_9", "x", nil), "no tool call")

	provider.handleMessage(serverMessage{ToolCallCancellation: &toolCallCancellation{IDs: []string{"call_2"}}})
	cancelled := s.drain(provider)
	s.Require().Len(cancelled, 1)
	s.Equal([]string{"call_2"}, cancelled[0].(sts.ToolCancel).CallIDs)
	s.NotContains(provider.calls, "call_2")
}

func (s *GeminiSuite) TestUsageIsAttachedToTheReplyItBelongsTo() {
	provider := s.newProvider()

	provider.handleMessage(spoke(240))
	provider.handleMessage(serverMessage{UsageMetadata: &usageMetadata{
		PromptTokenCount:      120,
		ResponseTokenCount:    60,
		PromptTokensDetails:   []modalityTokens{{Modality: "AUDIO", TokenCount: 100}, {Modality: "TEXT", TokenCount: 20}},
		ResponseTokensDetails: []modalityTokens{{Modality: "AUDIO", TokenCount: 60}},
	}})
	provider.handleMessage(turnComplete())

	events := s.drain(provider)
	complete, ok := events[len(events)-1].(sts.ResponseComplete)
	s.Require().True(ok)
	s.Equal(sts.Usage{InputTokens: 120, OutputTokens: 60, InputTextTokens: 20, InputAudioTokens: 100, OutputAudioTokens: 60},
		complete.Usage)
}

// TestUsageSplitsByModalityAndCountsThinking uses a typed turn as
// gemini-3.8-live-extended-thinking reported it: the details fall 111 tokens short of the
// prompt, and the thinking is outside the response count.
func (s *GeminiSuite) TestUsageSplitsByModalityAndCountsThinking() {
	provider := s.newProvider()

	provider.handleMessage(spoke(240))
	provider.handleMessage(serverMessage{UsageMetadata: &usageMetadata{
		PromptTokenCount:      3162,
		ResponseTokenCount:    54,
		ThoughtsTokenCount:    60,
		PromptTokensDetails:   []modalityTokens{{Modality: "TEXT", TokenCount: 2829}, {Modality: "AUDIO", TokenCount: 222}},
		ResponseTokensDetails: []modalityTokens{{Modality: "AUDIO", TokenCount: 54}},
	}})
	provider.handleMessage(turnComplete())

	s.Equal(sts.Usage{
		InputTokens:       3162,
		OutputTokens:      114,
		InputTextTokens:   2829,
		InputAudioTokens:  222,
		OutputTextTokens:  60,
		OutputAudioTokens: 54,
		ReasoningTokens:   60,
	}, s.completed(provider).Usage, "Google bill thinking as text output")
}

// TestAResponseTheDetailsLeaveOutIsBilledAsText uses gemini-3.8-live's answer after a tool
// call: 12 response tokens, the call, with no details, and 49 of thinking.
func (s *GeminiSuite) TestAResponseTheDetailsLeaveOutIsBilledAsText() {
	provider := s.newProvider()

	provider.handleMessage(spoke(240))
	provider.handleMessage(serverMessage{UsageMetadata: &usageMetadata{
		PromptTokenCount:   1362,
		ResponseTokenCount: 12,
		ThoughtsTokenCount: 49,
		PromptTokensDetails: []modalityTokens{
			{Modality: "TEXT", TokenCount: 468}, {Modality: "AUDIO", TokenCount: 516}, {Modality: "IMAGE", TokenCount: 256},
		},
	}})
	provider.handleMessage(turnComplete())

	usage := s.completed(provider).Usage
	s.EqualValues(61, usage.OutputTextTokens)
	s.EqualValues(61, usage.OutputTokens)
	s.EqualValues(256, usage.InputImageTokens)
}

// TestAReplyOverSeveralTurnsSumsEveryField uses gemini-3.8-live-extended-thinking's tool
// round trip: a filler while the tool ran, then the answer, billed as one reply.
func (s *GeminiSuite) TestAReplyOverSeveralTurnsSumsEveryField() {
	provider := s.newProvider()

	provider.handleMessage(spoke(240))
	provider.handleMessage(serverMessage{
		ServerContent: &serverContent{TurnComplete: true, InteractionStatus: "IN_PROGRESS"},
		UsageMetadata: &usageMetadata{
			PromptTokenCount:        3898,
			CachedContentTokenCount: 1024,
			ResponseTokenCount:      47,
			ThoughtsTokenCount:      86,
			PromptTokensDetails: []modalityTokens{
				{Modality: "TEXT", TokenCount: 2943}, {Modality: "AUDIO", TokenCount: 516}, {Modality: "IMAGE", TokenCount: 256},
			},
			ResponseTokensDetails: []modalityTokens{{Modality: "AUDIO", TokenCount: 47}},
		},
	})
	provider.handleMessage(spoke(240))
	provider.handleMessage(serverMessage{
		ServerContent: &serverContent{TurnComplete: true, InteractionStatus: "IDLE"},
		UsageMetadata: &usageMetadata{
			PromptTokenCount:        7319,
			CachedContentTokenCount: 2048,
			ResponseTokenCount:      119,
			ThoughtsTokenCount:      696,
			PromptTokensDetails: []modalityTokens{
				{Modality: "TEXT", TokenCount: 6199}, {Modality: "IMAGE", TokenCount: 352}, {Modality: "AUDIO", TokenCount: 563},
			},
			ResponseTokensDetails: []modalityTokens{{Modality: "TEXT", TokenCount: 27}, {Modality: "AUDIO", TokenCount: 92}},
		},
	})

	s.Equal(sts.Usage{
		InputTokens:       11217,
		CachedInputTokens: 3072,
		OutputTokens:      948,
		InputTextTokens:   9142,
		InputAudioTokens:  1079,
		InputImageTokens:  608,
		OutputTextTokens:  809,
		OutputAudioTokens: 139,
		ReasoningTokens:   782,
	}, s.completed(provider).Usage)
}

// TestATurnWithNoPromptDetailsIsStillBilled sums a turn whose prompt has details with one
// whose prompt has none. The split replaces the prompt total in the price, so the second
// turn's prompt is billed at the audio rate rather than left out of it.
func (s *GeminiSuite) TestATurnWithNoPromptDetailsIsStillBilled() {
	provider := s.newProvider()

	provider.handleMessage(spoke(240))
	provider.handleMessage(serverMessage{
		ServerContent: &serverContent{TurnComplete: true, InteractionStatus: "IN_PROGRESS"},
		UsageMetadata: &usageMetadata{
			PromptTokenCount:      600,
			ResponseTokenCount:    40,
			PromptTokensDetails:   []modalityTokens{{Modality: "TEXT", TokenCount: 400}, {Modality: "AUDIO", TokenCount: 150}},
			ResponseTokensDetails: []modalityTokens{{Modality: "AUDIO", TokenCount: 40}},
		},
	})
	provider.handleMessage(spoke(240))
	provider.handleMessage(serverMessage{
		ServerContent: &serverContent{TurnComplete: true, InteractionStatus: "IDLE"},
		UsageMetadata: &usageMetadata{PromptTokenCount: 1000, ResponseTokenCount: 30},
	})

	usage := s.completed(provider).Usage
	s.EqualValues(1600, usage.InputTokens)
	s.EqualValues(400, usage.InputTextTokens)
	s.EqualValues(1150, usage.InputAudioTokens, "150 described, and the 1,000 the second turn did not describe")
}

func (s *GeminiSuite) TestVideoIsBilledAsImages() {
	provider := s.newProvider()

	provider.handleMessage(spoke(240))
	provider.handleMessage(serverMessage{UsageMetadata: &usageMetadata{
		PromptTokenCount: 900,
		PromptTokensDetails: []modalityTokens{
			{Modality: "TEXT", TokenCount: 300}, {Modality: "IMAGE", TokenCount: 256}, {Modality: "VIDEO", TokenCount: 344},
		},
	}})
	provider.handleMessage(turnComplete())

	usage := s.completed(provider).Usage
	s.EqualValues(600, usage.InputImageTokens, "Google price video at the image rate")
	s.EqualValues(300, usage.InputTextTokens)
}

// TestCachedTokensStayInTheirModality reads a cached prompt. Google publish no cached
// rate for Live, so the cached tokens stay in their modality, at its input rate, and the
// cached count is kept only as the total.
func (s *GeminiSuite) TestCachedTokensStayInTheirModality() {
	provider := s.newProvider()

	provider.handleMessage(spoke(240))
	provider.handleMessage(serverMessage{UsageMetadata: &usageMetadata{
		PromptTokenCount:        2600,
		CachedContentTokenCount: 2048,
		PromptTokensDetails:     []modalityTokens{{Modality: "TEXT", TokenCount: 2400}, {Modality: "AUDIO", TokenCount: 200}},
	}})
	provider.handleMessage(turnComplete())

	s.Equal(sts.Usage{InputTokens: 2600, CachedInputTokens: 2048, InputTextTokens: 2400, InputAudioTokens: 200},
		s.completed(provider).Usage)
}

func (s *GeminiSuite) TestAFillerSpokenWhileTheModelThinksIsPartOfTheSameReply() {
	provider := s.newProvider()

	provider.handleMessage(spoke(240))
	provider.handleMessage(serverMessage{
		ServerContent: &serverContent{TurnComplete: true, InteractionStatus: "IN_PROGRESS"},
		UsageMetadata: &usageMetadata{PromptTokenCount: 3193, ResponseTokenCount: 47},
	})
	provider.handleMessage(spoke(240))
	provider.handleMessage(serverMessage{
		ServerContent: &serverContent{TurnComplete: true, InteractionStatus: "IDLE"},
		UsageMetadata: &usageMetadata{PromptTokenCount: 3335, ResponseTokenCount: 2736},
	})

	events := s.drain(provider)
	var chunks []sts.AudioChunk
	var completes []sts.ResponseComplete
	for _, event := range events {
		switch typed := event.(type) {
		case sts.AudioChunk:
			chunks = append(chunks, typed)
		case sts.ResponseComplete:
			completes = append(completes, typed)
		}
	}
	s.Require().Len(completes, 1, "the reply ends when the interaction does")
	s.False(completes[0].Interrupted)
	s.Equal(sts.Usage{InputTokens: 6528, OutputTokens: 2783, InputAudioTokens: 6528, OutputTextTokens: 2783}, completes[0].Usage,
		"each turn is billed on its own, so the reply is their sum")
	s.Require().Len(chunks, 2)
	s.Equal(chunks[0].Generation, chunks[1].Generation)
}

func (s *GeminiSuite) TestALocalInterruptInTheThinkingPauseCutsTheReplyOff() {
	provider := s.newProvider()

	provider.handleMessage(spoke(240))
	provider.handleMessage(serverMessage{ServerContent: &serverContent{TurnComplete: true, InteractionStatus: "IN_PROGRESS"}})
	s.Require().NoError(provider.Interrupt(0))
	// The answer the filler promised is the reply that was cut off, and none of it plays.
	provider.handleMessage(spoke(240))
	provider.handleMessage(serverMessage{ServerContent: &serverContent{TurnComplete: true, InteractionStatus: "IDLE"}})
	provider.handleMessage(spoke(240))

	events := s.drain(provider)
	var chunks []sts.AudioChunk
	var completes []sts.ResponseComplete
	for _, event := range events {
		switch typed := event.(type) {
		case sts.AudioChunk:
			chunks = append(chunks, typed)
		case sts.ResponseComplete:
			completes = append(completes, typed)
		}
	}
	s.Require().Len(completes, 1)
	s.True(completes[0].Interrupted)
	s.Require().Len(chunks, 2, "the filler, then the next reply")
	s.Equal(1, chunks[0].Generation)
	s.Equal(2, chunks[1].Generation)
}

func (s *GeminiSuite) TestAModelThatWillNotOpenWithoutAThinkingLevelIsGivenItsFloor() {
	for model, want := range map[string]string{
		"gemini-3.8-live-extended-thinking": "LOW",
		"gemini-3.8-live":                   "",
		"gemini-3.1-flash-live-preview":     "",
	} {
		provider, err := New(Options{APIKey: "test-key", Model: model})
		s.Require().NoError(err)
		thinking := provider.setup("").GenerationConfig.ThinkingConfig
		if want == "" {
			s.Nil(thinking, "%s rejects or does not need a thinking level nobody asked for", model)
			continue
		}
		s.Require().NotNil(thinking, model)
		s.Equal(want, thinking.ThinkingLevel, model)
	}

	provider, err := New(Options{APIKey: "test-key", Model: "gemini-3.8-live-extended-thinking", ThinkingLevel: "HIGH"})
	s.Require().NoError(err)
	s.Equal("HIGH", provider.setup("").GenerationConfig.ThinkingConfig.ThinkingLevel, "a level the caller asked for is kept")
}

func (s *GeminiSuite) TestTheServerWarnsBeforeItHangsUpAndLeavesAHandle() {
	provider := s.newProvider()

	provider.handleMessage(serverMessage{SessionResumptionUpdate: &resumptionUpdate{NewHandle: "h-1", Resumable: true}})
	provider.handleMessage(serverMessage{SessionResumptionUpdate: &resumptionUpdate{NewHandle: "h-2", Resumable: false}})
	provider.handleMessage(serverMessage{GoAway: &goAway{TimeLeft: "30s"}})

	s.Equal("h-1", provider.handle, "a handle that cannot be resumed from is not worth keeping")
	events := s.drain(provider)
	s.Require().Len(events, 1)
	s.Equal(30*time.Second, events[0].(sts.SessionExpiring).TimeLeft)
}

func (s *GeminiSuite) TestCloseSettlesAReplyInFlight() {
	provider := s.newProvider()
	provider.handleMessage(spoke(240))

	s.Require().NoError(provider.Close())

	var events []sts.Event
	for event := range provider.Events() {
		events = append(events, event)
	}
	complete, ok := events[len(events)-1].(sts.ResponseComplete)
	s.Require().True(ok)
	s.True(complete.Interrupted)
}

func (s *GeminiSuite) TestAnUnreadableEndOfSpeechLeavesTheLastWordHeard() {
	provider := s.newProvider()
	provider.sentMs, provider.sentAt = 1000, time.Now().Add(-time.Second)
	provider.handleMessage(heard("hello"))
	heardBy := time.Now()
	provider.handleMessage(serverMessage{VoiceActivity: &voiceActivity{Type: "ACTIVITY_END", AudioOffset: "soon"}})
	provider.handleMessage(spoke(2400))

	var stopped sts.SpeechStopped
	for _, event := range s.drain(provider) {
		if typed, ok := event.(sts.SpeechStopped); ok {
			stopped = typed
		}
	}
	s.False(stopped.At.After(heardBy))
	s.True(stopped.At.After(heardBy.Add(-100*time.Millisecond)), "not the start of the audio stream")
}

func (s *GeminiSuite) TestAnEndOfSpeechIsNotKeptForTheTurnAfter() {
	provider := s.newProvider()
	provider.sentMs, provider.sentAt = 1000, time.Now().Add(-time.Second)
	provider.handleMessage(heard("hello"))
	provider.handleMessage(serverMessage{VoiceActivity: &voiceActivity{Type: "ACTIVITY_END", AudioOffset: "0.500s"}})
	provider.handleMessage(spoke(2400))
	provider.handleMessage(turnComplete())
	s.drain(provider)

	provider.handleMessage(heard("again"))
	heardBy := time.Now()
	provider.handleMessage(spoke(2400))

	var stopped sts.SpeechStopped
	for _, event := range s.drain(provider) {
		if typed, ok := event.(sts.SpeechStopped); ok {
			stopped = typed
		}
	}
	s.True(stopped.At.After(heardBy.Add(-100*time.Millisecond)), "the second turn is timed from its own last word, not the first turn's end")
}

func (s *GeminiSuite) TestAnActivityOfAnotherTypeLeavesTheLastWordHeard() {
	provider := s.newProvider()
	provider.sentMs, provider.sentAt = 1000, time.Now().Add(-time.Second)
	provider.handleMessage(heard("hello"))
	heardBy := time.Now()
	provider.handleMessage(serverMessage{VoiceActivity: &voiceActivity{Type: "TYPE_UNSPECIFIED", AudioOffset: "0.500s"}})
	provider.handleMessage(spoke(2400))

	var stopped sts.SpeechStopped
	for _, event := range s.drain(provider) {
		if typed, ok := event.(sts.SpeechStopped); ok {
			stopped = typed
		}
	}
	s.True(stopped.At.After(heardBy.Add(-100*time.Millisecond)), "only an ACTIVITY_END places the stop")
}

func (s *GeminiSuite) TestAStopHeardDuringAReplyIsNotKeptForTheNext() {
	provider := s.newProvider()
	provider.handleMessage(heard("hello"))
	provider.handleMessage(spoke(2400))
	provider.handleMessage(heard("wait"))
	provider.handleMessage(spoke(2400))
	provider.handleMessage(turnComplete())
	time.Sleep(50 * time.Millisecond)
	provider.handleMessage(spoke(2400))
	provider.handleMessage(turnComplete())

	var completes []sts.ResponseComplete
	for _, event := range s.drain(provider) {
		if typed, ok := event.(sts.ResponseComplete); ok {
			completes = append(completes, typed)
		}
	}
	s.Require().Len(completes, 2)
	s.Zero(completes[1].TimeToFirstByteMs, "nobody was waiting on the next reply")
}
