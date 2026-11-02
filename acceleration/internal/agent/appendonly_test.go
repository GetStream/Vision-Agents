package agent

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// startsWith reports whether input opens with every message of sent, unchanged.
func startsWith(sent, input []llm.Message) bool {
	if len(sent) > len(input) {
		return false
	}
	for i, message := range sent {
		if !llm.SameMessage(message, input[i]) {
			return false
		}
	}
	return true
}

func (s *AgentSuite) TestEachRequestOpensWithTheOneBeforeItAcrossADeliveredAnswer() {
	// What a provider caches is the prefix it read, so every request carries the one before
	// it unchanged and the prompt every turn shares stays the same: the answer coming back
	// is added after the conversation, and kept there.
	s.delegates()
	s.join(true)
	s.model.reply = []string{`Let me check. <ask skill="think">15% of 84.20</ask>`}
	s.model.then = []string{"That comes to 12.63."}
	s.subagent.reply = []string{"It is 12.63."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "what is 15% of 84.20")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 2 }, "the answer was never spoken")
	s.quiet()
	s.says(participant, "thanks")
	s.eventually(func() bool { return len(s.model.requests()) == 3 }, "the caller was never answered")

	requests := s.model.requests()
	for i := 1; i < len(requests); i++ {
		s.Equal(requests[0].Instructions, requests[i].Instructions, "request %d", i)
		s.True(startsWith(requests[i-1].Input, requests[i].Input), "request %d opens with request %d", i, i-1)
	}
	s.True(knows(requests[2], "It is 12.63."), "what came back stays in the conversation")
}

func (s *AgentSuite) TestAnExchangeThatCarriedANoteIsStillTheLastExchange() {
	history := []llm.Message{
		{Role: llm.User, Content: "what is 15%"},
		{Role: llm.User, Content: `{"harness":"colleague_result","text":"It is 12.63."}`},
		{Role: llm.System, Content: harness.Notes()[0] + "\nTell the caller."},
		{Role: llm.Assistant, Content: "It is 12.63."},
	}

	s.Equal([]llm.Message{history[0], history[3]}, lastExchange(history))
}
