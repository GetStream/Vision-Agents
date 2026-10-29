package agent

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// AI-2014: a thread channel several people write in, such as an iMessage group chat. The
// author of each turn reaches RespondTo on its ctx (WithAuthor), and the turns read back from
// the channel carry theirs (llm.Message.Author).

// writes has the agent answer text as written by author, "" for a typed turn nobody in the
// thread wrote, and waits for the reply.
func (s *AgentSuite) writes(author, text string) llm.ResponseParams {
	ctx := s.ctx
	if author != "" {
		ctx = WithAuthor(ctx, author)
	}
	before := len(s.model.requests())
	_, err := s.agent.RespondTo(ctx, text, nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return countOf[Responded](s.reported()) == before+1 }, "the turn was never answered")
	return s.model.requests()[before]
}

func contentsOf(input []llm.Message) []string {
	contents := make([]string, 0, len(input))
	for _, message := range input {
		contents = append(contents, message.Content)
	}
	return contents
}

// The repro of AI-2014 (staging, 2026-10-27): each turn opens the conversation again from the
// thread channel, so who wrote what comes from the turns read back.
func (s *AgentSuite) TestAThreadReadBackWithTwoAuthorsNamesEachOfThem() {
	s.joinText()
	s.agent.RestoreHistory([]llm.Message{
		{Role: llm.User, Content: "Hi all, I'm Kanat", Author: "linq-kanat"},
		{Role: llm.Assistant, Content: "Hi Kanat!"},
		{Role: llm.User, Content: "I'm Ana. What is your return policy?", Author: "linq-ana"},
		{Role: llm.Assistant, Content: "Hi Ana! Thirty days."},
	})

	asked := s.writes("linq-kanat", "What are our names?")

	s.Equal("be brief\n\n"+severalWriters, withoutClock(asked.Instructions))
	s.Equal([]string{
		"Speaker 1: Hi all, I'm Kanat", "Hi Kanat!",
		"Speaker 2: I'm Ana. What is your return policy?", "Hi Ana! Thirty days.",
		"Speaker 1: What are our names?",
	}, contentsOf(asked.Input))
	s.Equal("Speaker 1: hi", contentsOf(s.writes("linq-kanat", "hi").Input)[6], "Kanat is still Speaker 1")
	for _, event := range s.reported() {
		if responding, ok := event.(Responding); ok {
			s.Equal(stt.Participant{ID: typedCaller}, responding.Participant, "events and tools still see the typed caller")
		}
	}
}

// A session kept open on the thread names a second author from their first turn, as a call
// names a second speaker: the first author's earlier turns stay as they were.
func (s *AgentSuite) TestASecondAuthorInAnOpenThreadIsNamedFromTheirFirstTurn() {
	s.joinText()

	first := s.writes("slack-alice", "is the build green?")
	second := s.writes("slack-bob", "and the deploy?")
	third := s.writes("slack-alice", "thanks")

	s.Equal("be brief", withoutClock(first.Instructions), "one author so far")
	s.Equal([]string{"is the build green?"}, contentsOf(first.Input))
	s.Equal("be brief\n\n"+severalWriters, withoutClock(second.Instructions))
	s.Equal("Speaker 2: and the deploy?", contentsOf(second.Input)[2])
	s.Equal("is the build green?", contentsOf(third.Input)[0], "a turn from before the second author stays as it was")
	s.Equal("Speaker 1: thanks", contentsOf(third.Input)[4])
}

// A typed turn nobody in the thread wrote, such as a scheduled task's or a login's follow-up,
// is the caller's: never named and never counted.
func (s *AgentSuite) TestATypedTurnWithoutAnAuthorInAThreadIsNotNamed() {
	s.joinText()
	s.agent.RestoreHistory([]llm.Message{
		{Role: llm.User, Content: "remind us at noon", Author: "linq-kanat"},
		{Role: llm.Assistant, Content: "Will do."},
	})

	asked := s.writes("", "It is noon.")

	s.Equal("be brief", withoutClock(asked.Instructions))
	s.Equal("It is noon.", contentsOf(asked.Input)[2])
}

// A typed turn nobody wrote in a thread two people wrote in keeps them named, and the note
// says they write.
func (s *AgentSuite) TestATypedTurnWithoutAnAuthorAfterTwoAuthorsKeepsThemNamed() {
	s.joinText()
	s.agent.RestoreHistory([]llm.Message{
		{Role: llm.User, Content: "remind us at noon", Author: "linq-kanat"},
		{Role: llm.Assistant, Content: "Will do."},
		{Role: llm.User, Content: "me too", Author: "linq-ana"},
		{Role: llm.Assistant, Content: "Both of you, then."},
	})

	asked := s.writes("", "It is noon.")

	s.Equal("be brief\n\n"+severalWriters, withoutClock(asked.Instructions))
	s.Equal([]string{"Speaker 1: remind us at noon", "Will do.", "Speaker 2: me too", "Both of you, then.", "It is noon."},
		contentsOf(asked.Input))
}

// The control: a 1:1 chat on a thread channel and a session in writing that names no author
// (dashboard, SDK) send what base af17ef057 sends for the same turns. The expected request is
// the one a probe on af17ef057 printed for this history and turn, with no authors.
func (s *AgentSuite) TestOnePersonInWritingReachesTheModelAsAtBase() {
	for _, author := range []string{"linq-kanat", ""} {
		s.joinText()
		s.agent.RestoreHistory([]llm.Message{
			{Role: llm.User, Content: "Hi all, I'm Kanat", Author: author},
			{Role: llm.Assistant, Content: "Hi Kanat!"},
		})

		asked := s.writes(author, "What is my name?")

		s.Equal("be brief", withoutClock(asked.Instructions), author)
		sent := make([]llm.Message, 0, len(asked.Input))
		for _, message := range asked.Input {
			// No provider reads Author (llm.Message).
			message.Author = ""
			sent = append(sent, message)
		}
		s.Equal([]llm.Message{
			{Role: llm.User, Content: "Hi all, I'm Kanat"},
			{Role: llm.Assistant, Content: "Hi Kanat!"},
			{Role: llm.User, Content: "What is my name?"},
		}, sent, author)
	}
}

func (s *AgentSuite) TestTheWritersNoteIsForWriting() {
	s.NotContains(severalWriters, "call")
	s.Contains(severalWriters, "such a label")
}
