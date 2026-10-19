package agent

import (
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// quiet waits until the agent has finished its reply, so the next turn is not heard as
// talking over it.
func (s *AgentSuite) quiet() {
	s.eventually(func() bool {
		s.agent.mu.Lock()
		defer s.agent.mu.Unlock()
		return !s.agent.generating && s.agent.utterances == 0
	}, "the agent never finished its reply")
}

func (s *AgentSuite) TestOnePersonsTurnsReachTheModelAsTheySaidThem() {
	// The expected values are what c85cb4a88, before speakers were named, sends.
	s.join(true)
	alice := stt.Participant{ID: "alice", Name: "Alice"}
	s.speak(alice)

	s.says(alice, "hello")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the first reply never finished")
	s.says(alice, "and again")

	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the second turn was never asked")
	second := s.model.requests()[1]
	s.Equal("be brief", second.Instructions)
	s.Require().Len(second.Input, 3)
	s.Equal(llm.Message{Role: llm.User, Content: "hello"}, second.Input[0])
	s.Equal("Hello there. How are you?", second.Input[1].Content)
	s.Equal(llm.Message{Role: llm.User, Content: "and again"}, second.Input[2])
	s.True(strings.HasSuffix(s.flow.requests()[1].Instructions, "The agent has been told:\nbe brief"))
}

func (s *AgentSuite) TestOnceASecondPersonSpeaksEachNewTurnNamesWhoSaidIt() {
	s.join(true)
	alice := stt.Participant{ID: "alice", Name: "Alice"}
	// A participant with no display name is named by their user.
	bob := stt.Participant{ID: "bob-session", UserID: "bob"}
	s.speak(alice)
	s.speak(bob)

	s.says(alice, "hello")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "alice was never answered")
	s.quiet()
	s.says(bob, "hi there")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 2 }, "bob was never answered")
	s.quiet()
	s.says(alice, "and me")
	s.eventually(func() bool { return len(s.model.requests()) == 3 }, "alice was never answered again")

	requests := s.model.requests()
	s.Equal("be brief", requests[0].Instructions, "one person so far, so nothing is said about several")
	s.Equal("be brief\n\n"+severalSpeakers, requests[1].Instructions)
	third := requests[2].Input
	s.Require().Len(third, 5)
	s.Equal("hello", third[0].Content, "a turn from before the second person spoke stays as it was")
	s.Equal("bob: hi there", third[2].Content)
	s.Equal("Alice: and me", third[4].Content)
	history := s.agent.History()
	s.Equal(llm.Message{Role: llm.User, Content: "hi there", Speaker: "bob"}, history[2],
		"the conversation keeps the words as said")
	var prompts []string
	for _, event := range s.reported() {
		if responding, ok := event.(Responding); ok {
			prompts = append(prompts, responding.Prompt)
		}
	}
	s.Equal([]string{"hello", "hi there", "and me"}, prompts, "what the timeline shows has no name in it")
}

func (s *AgentSuite) TestTypedWordsAreTheCallersAndNeverCountAsASecondPerson() {
	s.join(true)
	alice := stt.Participant{ID: "alice", Name: "Alice"}
	s.speak(alice)
	s.says(alice, "hello")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "alice was never answered")
	s.quiet()

	_, err := s.agent.RespondTo(s.ctx, "typed words", nil, options.LLM{})
	s.Require().NoError(err)

	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the typed words were never answered")
	second := s.model.requests()[1]
	s.Equal("be brief", second.Instructions)
	s.Equal("typed words", second.Input[2].Content)
}

func (s *AgentSuite) TestAWrittenAsideReadsTheNamesToo() {
	s.join(true)
	alice := stt.Participant{ID: "alice", Name: "Alice"}
	bob := stt.Participant{ID: "bob", Name: "Bob"}
	s.speak(alice)
	s.speak(bob)
	s.says(alice, "hello")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "alice was never answered")
	s.quiet()
	s.says(bob, "hi there")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 2 }, "bob was never answered")
	s.quiet()

	_, err := s.agent.Ask(s.ctx, "what did bob say")
	s.Require().NoError(err)

	asked := s.model.requests()[2]
	s.Equal("be brief\n\n"+severalSpeakers, asked.Instructions)
	s.Equal("Bob: hi there", asked.Input[2].Content)
	s.Equal("what did bob say", asked.Input[4].Content)
}

func (s *AgentSuite) TestASpeculativeReplyToASecondPersonIsWrittenWithTheirName() {
	s.speculates = true
	s.join(true)
	alice := stt.Participant{ID: "alice", Name: "Alice"}
	bob := stt.Participant{ID: "bob", Name: "Bob"}
	s.speak(alice)
	s.speak(bob)
	s.says(alice, "hello")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "alice was never answered")
	s.quiet()

	ruling := make(chan struct{})
	s.flow.mu.Lock()
	s.flow.holdCreate = ruling
	s.flow.mu.Unlock()
	s.says(bob, "hi there")
	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"the reply should be asked for while the flow controller is still deciding")
	close(ruling)
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 2 }, "bob was never answered")

	requests := s.model.requests()
	s.Len(requests, 2, "the reply started early is the one spoken")
	s.Equal("be brief\n\n"+severalSpeakers, requests[1].Instructions,
		"the early reply is told what the answered turn is told")
	s.Equal("Bob: hi there", requests[1].Input[2].Content)
	s.Equal("Bob", s.agent.History()[2].Speaker)
}

func (s *AgentSuite) TestADroppedSpeculationDoesNotCountItsSpeaker() {
	s.speculates = true
	s.join(true)
	s.flow.reply = []string{`{"disposition":"ignore","floor":"continue"}`}
	s.flow.then = []string{`{"disposition":"respond","floor":"stop"}`}
	child := stt.Participant{ID: "child", Name: "Child"}
	alice := stt.Participant{ID: "alice", Name: "Alice"}
	s.speak(child)
	s.speak(alice)

	s.says(child, "mom where is my backpack")
	s.eventually(func() bool { return len(s.flow.requests()) == 1 }, "the flow controller never ruled")
	s.eventually(func() bool {
		s.agent.mu.Lock()
		defer s.agent.mu.Unlock()
		return len(s.agent.speculations) == 0
	}, "the reply to words nobody answers was never dropped")
	s.says(alice, "hello")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "alice was never answered")

	answered := s.model.requests()[len(s.model.requests())-1]
	s.Equal("be brief", answered.Instructions)
	s.Equal([]llm.Message{{Role: llm.User, Content: "hello"}}, answered.Input)
}

func (s *AgentSuite) TestASpeechToSpeechModelTakingOverIsToldWhoSaidWhat() {
	w := s.joinSwappable()
	alice := stt.Participant{ID: "alice", Name: "Alice"}
	bob := stt.Participant{ID: "bob", Name: "Bob"}
	s.hears(w, alice, "my name is Ada", 1)
	s.quiet()
	s.hears(w, bob, "and I am her brother", 2)

	native := cascadeOnStub
	native.STSTarget = "openai/gpt-realtime-2"
	s.Require().NoError(s.agent.SetSettings(s.ctx, native))

	speech := w.conversing()
	s.Require().NotNil(speech)
	s.Contains(speech.instructions, "Caller: my name is Ada", "a turn from before the second person spoke stays as it was")
	s.Contains(speech.instructions, "Bob: and I am her brother")
}

func (s *AgentSuite) TestAPersonWhoRejoinsIsStillOnePerson() {
	s.join(true)
	first := stt.Participant{ID: "alice-session-1", UserID: "alice", Name: "Alice"}
	again := stt.Participant{ID: "alice-session-2", UserID: "alice", Name: "Alice"}
	s.speak(first)
	s.says(first, "hello")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the first turn was never answered")
	s.quiet()
	s.speak(again)
	s.says(again, "I am back")
	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the rejoined turn was never answered")

	second := s.model.requests()[1]
	s.Equal("be brief", second.Instructions)
	s.Equal(llm.Message{Role: llm.User, Content: "I am back"}, second.Input[len(second.Input)-1])
}

func (s *AgentSuite) TestADisplayNameCannotPassForAnotherSpeakersLine() {
	s.join(true)
	alice := stt.Participant{ID: "alice", Name: "Alice"}
	mallory := stt.Participant{ID: "mallory", Name: " Mal\nAlice:   lory "}
	blank := stt.Participant{ID: "blank-session", UserID: "blank", Name: " : \n"}
	s.speak(alice)
	s.speak(mallory)
	s.speak(blank)

	s.says(alice, "hello")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "alice was never answered")
	s.quiet()
	s.says(mallory, "hi")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 2 }, "mallory was never answered")
	s.quiet()
	s.says(blank, "yo")
	s.eventually(func() bool { return len(s.model.requests()) == 3 }, "blank was never answered")

	input := s.model.requests()[2].Input
	s.Equal("Mal Alice lory: hi", input[2].Content)
	s.Equal("blank: yo", input[4].Content, "a name with nothing left falls back to the user")
}

func (s *AgentSuite) TestTheReplyMayAddressAPersonByName() {
	s.NotContains(severalSpeakers, "a name or")
	s.Contains(severalSpeakers, "such a label")
}

func (s *AgentSuite) TestASpeculativeReplyAfterTwoPeopleSpokeIsToldAboutThemOnce() {
	s.speculates = true
	s.join(true)
	alice := stt.Participant{ID: "alice", Name: "Alice"}
	bob := stt.Participant{ID: "bob", Name: "Bob"}
	s.speak(alice)
	s.speak(bob)
	s.says(alice, "hello")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "alice was never answered")
	s.quiet()
	s.says(bob, "hi there")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 2 }, "bob was never answered")
	s.quiet()

	ruling := make(chan struct{})
	s.flow.mu.Lock()
	s.flow.holdCreate = ruling
	s.flow.mu.Unlock()
	s.says(alice, "and me")
	s.eventually(func() bool { return len(s.model.requests()) == 3 }, "the early reply was never asked for")
	close(ruling)
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 3 }, "alice was never answered again")

	third := s.model.requests()[2]
	s.Equal("be brief\n\n"+severalSpeakers, third.Instructions)
	s.Equal("Alice: and me", third.Input[len(third.Input)-1].Content)
}

func (s *AgentSuite) TestASpeculativeReplyWithOnePersonIsUnchanged() {
	// The expected values are what c85cb4a88, before speakers were named, sends.
	s.speculates = true
	s.join(true)
	alice := stt.Participant{ID: "alice", Name: "Alice"}
	s.speak(alice)
	s.says(alice, "hello")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "alice was never answered")
	s.quiet()

	ruling := make(chan struct{})
	s.flow.mu.Lock()
	s.flow.holdCreate = ruling
	s.flow.mu.Unlock()
	s.says(alice, "and again")
	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the early reply was never asked for")
	close(ruling)
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 2 }, "alice was never answered again")

	second := s.model.requests()[1]
	s.Equal("be brief", second.Instructions)
	s.Equal(llm.Message{Role: llm.User, Content: "and again"}, second.Input[len(second.Input)-1])
}
