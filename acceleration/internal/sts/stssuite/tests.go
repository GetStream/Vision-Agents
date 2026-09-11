//go:build integration

package stssuite

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
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
	reply := s.Settled(provider)

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
