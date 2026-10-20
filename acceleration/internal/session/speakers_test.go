package session

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// replyTo is the request the conversation model got for a turn, from whichever model the
// session opened for it: a swap opens new ones.
func (s *SessionSuite) replyTo(text string) (llm.ResponseParams, bool) {
	s.othersMu.Lock()
	defer s.othersMu.Unlock()
	for _, model := range s.others {
		for _, request := range model.requests() {
			if request.Input[len(request.Input)-1].Content == text {
				return request, true
			}
		}
	}
	return llm.ResponseParams{}, false
}

func (s *SessionSuite) TestThePeopleHeardInACallAreStillLabelledOnceVoiceStops() {
	s.manages()
	created := s.joins(Spec{})
	s.Require().Eventually(func() bool { return len(s.edges) == 1 }, settleFor, 10*time.Millisecond)
	alice := stt.Participant{ID: "alice", Name: "Alice"}
	bob := stt.Participant{ID: "bob", Name: "Bob"}
	pcm := audio.PcmData{Samples: make([]int16, 320), SampleRate: 16_000, Channels: 1}
	s.edges[0].inbound <- agent.InboundAudio{Participant: alice, Audio: pcm}
	s.edges[0].inbound <- agent.InboundAudio{Participant: bob, Audio: pcm}
	for i, turn := range []struct {
		participant stt.Participant
		text        string
	}{{alice, "hello"}, {bob, "hi there"}} {
		s.ears.emitter.Send(stt.Transcript{Participant: turn.participant, Mode: stt.ModeFinal, Text: turn.text, Language: "en", Confidence: 1})
		s.Require().Eventually(func() bool { return len(s.model.requests()) == i+1 }, settleFor, 10*time.Millisecond, "turn %d was never asked", i)
		s.Require().Eventually(func() bool { return !created.Busy() }, settleFor, 10*time.Millisecond)
	}

	s.Require().NoError(created.StopVoice(s.ctx))
	_, err := created.Respond(s.ctx, "and now?", nil, options.LLM{})
	s.Require().NoError(err)

	var asked llm.ResponseParams
	s.Require().Eventually(func() bool {
		var ok bool
		asked, ok = s.replyTo("and now?")
		return ok
	}, settleFor, 10*time.Millisecond, "the typed turn was never asked")
	s.Contains(asked.Instructions, "Several people are talking in this call.")
	var heard []string
	for _, message := range asked.Input {
		heard = append(heard, message.Content)
	}
	s.Equal([]string{"hello", "Hello.", "Bob: hi there", "Hello.", "and now?"}, heard)
}
