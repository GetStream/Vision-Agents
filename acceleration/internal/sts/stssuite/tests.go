//go:build integration

package stssuite

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testaudio"
)

// briefly is what every session is opened with, so a reply is a sentence rather than an
// essay and the tests spend seconds rather than minutes listening.
const briefly = "You are a helpful assistant on a phone call. Answer in one short sentence."

// weatherTool is a function any model that calls tools should reach for when asked about
// the weather.
var weatherTool = llm.Tool{
	Name:        "get_weather",
	Description: "Get the current weather in a city.",
	Parameters: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{"type": "string", "description": "The city to look up."},
		},
		"required": []string{"city"},
	},
}

func (s *Suite) TestTheModelHearsTheCallerAndAnswersInItsOwnVoice() {
	provider := s.Started(Ask{Instructions: briefly})
	defer s.Hangup(provider)

	s.Speak(provider)
	reply := s.Spoken(provider)

	// A server turn detector may take a pause mid-sentence for the end of the turn and open
	// a reply that the rest of the sentence cuts off. That reply has to settle silent.
	for _, event := range reply.AllEvents {
		if complete, ok := event.(sts.ResponseComplete); ok && complete.ResponseID != reply.Complete.ResponseID {
			s.True(complete.Interrupted, "a reply before the spoken one should have been cut off")
			s.Zero(complete.AudioDurationMs, "a reply the caller cut off should have said nothing")
		}
	}

	s.Require().NotEmpty(reply.Chunks, "the model should have spoken")
	for _, chunk := range reply.Chunks {
		s.Equal(provider.SampleRate(), chunk.Audio.SampleRate, "audio should arrive at the rate the provider reports")
		s.Equal(1, chunk.Audio.Channels)
	}
	s.False(reply.Complete.Interrupted)
	s.Greater(reply.Complete.AudioDurationMs, 0.0)
	s.InDelta(reply.AudioMs(), reply.Complete.AudioDurationMs, 1.0,
		"the model's own count of what it said should match the chunks")
	if reply.Complete.TimeToFirstByteMs > 0 {
		s.LessOrEqual(reply.Complete.TimeToFirstByteMs, s.MaxTimeToFirstByte,
			"the caller waited too long to hear the reply begin")
	}

	if provider.Capabilities().InputTranscript {
		heard := reply.HeardText()
		s.Require().NotEmpty(heard, "the model should have written down what it heard")
		s.GreaterOrEqual(testaudio.Accuracy(s.Reference, heard), s.MinAccuracy,
			"heard %q, wanted %q", heard, s.Reference)
	}
	if provider.Capabilities().OutputTranscript {
		s.NotEmpty(reply.SaidText(), "the model should have written down what it said")
	}
}

func (s *Suite) TestTheModelHearsAndAnswersWhileTheCallerIsStillSpeaking() {
	capabilities := s.New(Ask{}).Capabilities()
	if !capabilities.FullDuplex {
		s.T().Skip("provider does not declare full-duplex audio")
	}
	if !capabilities.Text || !capabilities.InputTranscript {
		s.T().Skip("full-duplex proof needs prompts and input transcript observability")
	}

	provider := s.Started(Ask{Instructions: "Keep telling a long story about a lighthouse keeper in a storm, even when the caller speaks."})
	defer s.Hangup(provider)
	s.Require().NoError(provider.Prompt("Tell me a story about a lighthouse keeper watching a storm approach."))

	chunks := testaudio.Chunks(s.Audio, s.ChunkMs)
	quiet := testaudio.Silence(s.ChunkMs)
	chunkDuration := time.Duration(s.ChunkMs) * time.Millisecond
	ticker := time.NewTicker(chunkDuration)
	defer ticker.Stop()
	timeout := time.NewTimer(s.Timeout)
	defer timeout.Stop()

	var originalID string
	originalGeneration, lastIndex := 0, -1
	originalCompleted, fixtureStarted := false, false
	var fixtureEnd time.Time
	freshTranscript, continued := false, false
	var overlapFirst, overlapLast time.Time
	overlapChunks := 0
	overlapAudioMs := 0.0
	var finalizedTranscript, pendingTranscript string
	accuracy := 0.0
	for {
		select {
		case event, open := <-provider.Events():
			if !open {
				s.FailNow("provider closed its event stream during full-duplex proof")
			}
			at := time.Now()
			activeFixture := fixtureStarted && (fixtureEnd.IsZero() || at.Before(fixtureEnd))
			switch typed := event.(type) {
			case sts.Error:
				s.FailNowf("provider error during full-duplex proof", "fatal=%t context=%q", typed.Fatal, typed.Context)
			case sts.Disconnected:
				s.FailNowf("provider disconnected during full-duplex proof", "reason=%q", typed.Reason)
			case sts.InputTranscript:
				if fixtureStarted {
					switch typed.Mode {
					case stt.ModeDelta:
						pendingTranscript += typed.Text
					case stt.ModeReplacement:
						pendingTranscript = typed.Text
					case stt.ModeFinal:
						finalizedTranscript = strings.TrimSpace(finalizedTranscript + " " + typed.Text)
						pendingTranscript = ""
					default:
						s.FailNowf("unexpected input transcript mode", "%q", typed.Mode)
					}
					accuracy = testaudio.Accuracy(s.Reference, strings.TrimSpace(finalizedTranscript+" "+pendingTranscript))
					if !freshTranscript && originalGeneration != 0 && !originalCompleted && activeFixture && strings.TrimSpace(typed.Text) != "" {
						freshTranscript = true
					}
				}
			case sts.AudioChunk:
				nonSilent := slices.ContainsFunc(typed.Audio.Samples, func(sample int16) bool { return sample != 0 })
				if originalGeneration == 0 && nonSilent {
					originalID, originalGeneration, lastIndex = typed.ResponseID, typed.Generation, typed.Index
				}
				if !originalCompleted && typed.ResponseID == originalID && typed.Generation == originalGeneration && typed.Index > lastIndex {
					lastIndex = typed.Index
					if nonSilent && activeFixture {
						overlapChunks++
						if overlapFirst.IsZero() {
							overlapFirst = at
						}
						overlapLast = at
						overlapAudioMs += typed.Audio.DurationMs()
						if freshTranscript {
							continued = true
						}
					}
				}
			case sts.ResponseComplete:
				if typed.Generation == originalGeneration && typed.ResponseID == originalID {
					if !continued {
						s.FailNow("the original response completed before it continued after caller speech")
					}
					originalCompleted = true
				}
			}
		case <-ticker.C:
			if originalGeneration == 0 {
				continue
			}
			var chunk stt.PcmData
			sendingFixture := len(chunks) > 0
			if sendingFixture {
				chunk, chunks = chunks[0], chunks[1:]
			} else {
				if !fixtureEnd.IsZero() && time.Now().Before(fixtureEnd) {
					continue
				}
				chunk = quiet
			}
			s.Require().NoError(provider.ProcessAudio(chunk, speaker))
			if sendingFixture {
				fixtureStarted = true
				if len(chunks) == 0 {
					fixtureEnd = time.Now().Add(chunkDuration)
				}
			}
		case <-timeout.C:
			s.FailNowf("timed out waiting for full-duplex evidence", "overlap=%s audio=%.0fms chunks=%d accuracy=%.3f",
				overlapLast.Sub(overlapFirst), overlapAudioMs, overlapChunks, accuracy)
		}

		fixtureDone := !fixtureEnd.IsZero() && !time.Now().Before(fixtureEnd)
		if !continued || !fixtureDone || accuracy < s.MinAccuracy {
			continue
		}
		s.Require().GreaterOrEqual(overlapChunks, 2, "need multiple non-silent original-response chunks during fixture streaming")
		s.Require().GreaterOrEqual(overlapLast.Sub(overlapFirst), 500*time.Millisecond,
			"overlap lasted %s across %d chunks", overlapLast.Sub(overlapFirst), overlapChunks)
		s.Require().GreaterOrEqual(overlapAudioMs, 500.0,
			"original response supplied only %.0fms of non-silent audio during the fixture", overlapAudioMs)
		s.T().Logf("full-duplex proof: overlap=%s non-silent-output=%.0fms chunks=%d transcript-accuracy=%.3f",
			overlapLast.Sub(overlapFirst), overlapAudioMs, overlapChunks, accuracy)
		return
	}
}

func (s *Suite) TestCuttingInStopsTheReplyAndSettlesItOnce() {
	provider := s.Started(Ask{Instructions: "You are a storyteller. When asked, tell a long story of at least ten sentences."})
	defer s.Hangup(provider)

	s.Speak(provider)
	// Wait for the reply to be under way, so there is something to cut off.
	s.Collect(provider, func(event sts.Event) bool {
		_, spoke := event.(sts.AudioChunk)
		return spoke
	})
	s.Require().NoError(provider.Interrupt(0))

	events := s.Collect(provider, func(event sts.Event) bool {
		done, ok := event.(sts.ResponseComplete)
		return ok && done.Interrupted
	})
	reply := replyOf(events)
	s.True(reply.Complete.Interrupted, "the reply that was cut off should say so")

	// Whatever arrives afterwards belongs to a later reply: nothing from the interrupted
	// one may play after it was settled.
	settled := reply.Complete.Generation
	deadline := time.After(2 * time.Second)
	for {
		select {
		case event, open := <-provider.Events():
			if !open {
				return
			}
			if chunk, ok := event.(sts.AudioChunk); ok {
				s.NotEqual(settled, chunk.Generation, "audio from the interrupted reply arrived after it settled")
			}
		case <-deadline:
			return
		}
	}
}

func (s *Suite) TestATypedTurnIsAnswered() {
	if !s.New(Ask{}).Capabilities().Text {
		s.T().Skip("this model takes no typed turns")
	}
	provider := s.Started(Ask{Instructions: briefly})
	defer s.Hangup(provider)

	s.Require().NoError(provider.SendText("Say the word pineapple and nothing else.", speaker))
	reply := s.Settled(provider)

	s.NotEmpty(reply.Chunks, "the model should have spoken its answer")
	s.Zero(reply.Complete.TimeToFirstByteMs, "nobody was waiting for a typed turn to be answered")
}

func (s *Suite) TestATypedTurnIsRefusedWhereTheModelTakesNone() {
	if s.New(Ask{}).Capabilities().Text {
		s.T().Skip("this model takes typed turns")
	}
	provider := s.Started(Ask{Instructions: briefly})
	defer s.Hangup(provider)

	s.ErrorIs(provider.SendText("hello", speaker), sts.ErrNoText)
}

func (s *Suite) TestAToolIsCalledAndItsAnswerCarriesTheConversationOn() {
	if !s.New(Ask{}).Capabilities().Tools {
		s.T().Skip("this model calls no tools")
	}
	provider := s.Started(Ask{
		Instructions: briefly + " Always use the get_weather tool when asked about the weather.",
		Tools:        []llm.Tool{weatherTool},
	})
	defer s.Hangup(provider)

	s.Require().NoError(provider.SendText("What is the weather in Paris right now?", speaker))
	asked := s.Asked(provider)
	s.Require().NotEmpty(asked.ToolCalls)
	call := asked.ToolCalls[0]
	s.Equal(weatherTool.Name, call.Name)
	s.NotEmpty(call.CallID)

	s.Require().NoError(provider.Answer(call.CallID, `{"forecast":"sunny","temperature_c":21}`, nil))
	reply := s.Spoken(provider)
	s.NotEmpty(reply.Chunks, "the model should have spoken the forecast")
}

func (s *Suite) TestInstructionsChangeOnlyWhereTheModelAllowsIt() {
	provider := s.Started(Ask{Instructions: briefly})
	defer s.Hangup(provider)

	err := provider.SetInstructions(briefly + " Always end with the word goodbye.")
	if provider.Capabilities().InstructionsMidSession {
		s.NoError(err)
		return
	}
	s.ErrorIs(err, sts.ErrInstructionsFixed, "a model that cannot change its instructions has to say so")
}

func (s *Suite) TestAPromptMakesTheModelSpeakUnasked() {
	if !s.New(Ask{}).Capabilities().Text {
		s.T().Skip("this model takes no prompts")
	}
	provider := s.Started(Ask{Instructions: briefly})
	defer s.Hangup(provider)

	s.Require().NoError(provider.Prompt("Greet the caller warmly in one sentence."))
	reply := s.Settled(provider)
	s.NotEmpty(reply.Chunks, "the model should have greeted the caller")
	s.False(reply.Complete.Interrupted)
}

func (s *Suite) TestAModelToldHowAnOpenCallIsGoingSaysSoAndCallsNothing() {
	updater, ok := s.New(Ask{Tools: []llm.Tool{weatherTool}}).(sts.CallUpdater)
	if !ok || !updater.UpdatesCalls() {
		s.T().Skip("this model cannot be told how an open call is going")
	}
	provider := s.Started(Ask{
		Instructions: briefly + " Always use the get_weather tool when asked about the weather.",
		Tools:        []llm.Tool{weatherTool},
	})
	defer s.Hangup(provider)
	s.Require().NoError(provider.SendText("What is the weather in Paris right now?", speaker))
	call := s.Asked(provider).ToolCalls[0]

	// Reykjavik is in the status alone, so a reply naming it is one that heard the update.
	s.Require().NoError(provider.(sts.CallUpdater).UpdateCall(sts.CallUpdate{
		CallID: call.CallID, Status: "waiting on the backup weather station in Reykjavik", Instructions: briefly,
		Guidance: "The weather lookup is still running and reports: waiting on the backup weather station in Reykjavik. " +
			"Tell the caller in one short sentence where it has got to, naming the station's city. Do not call any tool.",
	}))
	var said strings.Builder
	var update Reply
	for !strings.Contains(strings.ToLower(said.String()), "reykjavik") {
		update = s.Spoken(provider)
		s.Require().Empty(update.ToolCalls, "a reply about an open call must call nothing")
		for _, transcript := range update.Said {
			said.WriteString(transcript.Text)
		}
	}

	s.NotEmpty(update.Chunks, "the model should have said where the lookup has got to")
	s.Require().NoError(provider.Answer(call.CallID, `{"forecast":"sunny","temperature_c":21}`, nil))
	s.NotEmpty(s.Spoken(provider).Chunks, "the open call should still take its answer")
}

// bookTool is a function that changes something, so the agent asks the caller first.
var bookTool = llm.Tool{
	Name:        "book_flight",
	Description: "Book a flight for the caller by its flight number.",
	Parameters: map[string]any{
		"type":       "object",
		"properties": map[string]any{"flight_id": map[string]any{"type": "string", "description": "The flight number, such as BA 117."}},
		"required":   []string{"flight_id"},
	},
}

// confirmTool and approvalAnswer are confirm_tool_call and the answer to a call that needs
// the caller's yes, as the agent offers and words them on a native call (agent/approval.go).
var confirmTool = llm.Tool{
	Name: "confirm_tool_call",
	Description: "Give the caller's answer to a question a tool asked before running. Call it only once " +
		"they have answered, with the tool_call_id the question came with.",
	Parameters: map[string]any{
		"type":     "object",
		"required": []string{"tool_call_id", "allowed"},
		"properties": map[string]any{
			"tool_call_id": map[string]any{"type": "string", "description": "The id the question came with."},
			"allowed":      map[string]any{"type": "boolean", "description": "True only for a yes."},
		},
	},
}

const approvalAnswer = "Not run yet: %s runs only once the caller says yes. Ask them now, in these words: %q. " +
	"When they answer, call confirm_tool_call with tool_call_id %q and allowed true for a yes, or false for " +
	"anything else. Do not call %s again for this and do not say it was done."

func (s *Suite) TestAModelAsksTheCallerFirstAndConfirmsTheirAnswer() {
	if !s.New(Ask{}).Capabilities().Tools {
		s.T().Skip("this model calls no tools")
	}
	for _, c := range []struct {
		name    string
		answer  string
		allowed bool
	}{
		{"yes", "Yes, please go ahead.", true},
		{"no", "No, don't book it.", false},
	} {
		s.Run(c.name, func() {
			provider := s.Started(Ask{
				Instructions: briefly + " Always use the book_flight tool when asked to book a flight.",
				Tools:        []llm.Tool{bookTool, confirmTool},
			})
			defer s.Hangup(provider)
			s.Require().NoError(provider.SendText("Book flight BA 117 for me, please.", speaker))
			asked := s.Asked(provider).ToolCalls
			s.Require().NotEmpty(asked)
			call := asked[0]
			s.Require().Equal(bookTool.Name, call.Name)

			question := "Shall I book flight BA 117?"
			s.Require().NoError(provider.Answer(call.CallID, fmt.Sprintf(approvalAnswer, call.Name, question, call.CallID, call.Name), nil))
			// The flight number is in the question alone, said as digits or as words. A model's
			// transcript can trail the end of its reply, so it is read off the events as they come.
			var said strings.Builder
			for _, event := range s.Collect(provider, func(event sts.Event) bool {
				if transcript, ok := event.(sts.OutputTranscript); ok {
					said.WriteString(transcript.Text)
				}
				_, called := event.(sts.ToolCall)
				heard := strings.ToLower(said.String())
				return called || strings.Contains(heard, "117") || strings.Contains(heard, "seventeen")
			}) {
				_, called := event.(sts.ToolCall)
				s.Require().False(called, "the model called again before asking the caller")
			}

			s.Require().NoError(provider.SendText(c.answer, speaker))
			confirmed := s.Asked(provider).ToolCalls
			s.Require().NotEmpty(confirmed)
			s.Require().Equal(confirmTool.Name, confirmed[0].Name, "the caller's answer goes to confirm_tool_call")
			var answer struct {
				ToolCallID string `json:"tool_call_id"`
				Allowed    bool   `json:"allowed"`
			}
			s.Require().NoError(json.Unmarshal([]byte(confirmed[0].Arguments), &answer))
			s.Equal(c.allowed, answer.Allowed)
			s.Equal(call.CallID, answer.ToolCallID)
		})
	}
}
