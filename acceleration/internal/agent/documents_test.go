package agent

import (
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

func (s *AgentSuite) TestAFileSentWithATurnGoesAheadOfThatReplyAndEveryOneAfter() {
	s.joinText()
	lease := Document{ID: "d-1", Name: "lease.docx", Text: "The deposit is two months of rent."}

	_, err := s.agent.RespondTo(WithDocument(s.ctx, lease), "What is the deposit?", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.model.requests()) == 1 && !s.agent.Busy() }, "the first turn never finished")
	_, err = s.agent.RespondTo(s.ctx, "And the notice period?", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the second turn never reached the model")

	for _, asked := range s.model.requests() {
		s.Require().True(asked.Input[0].Document, "the file opens what every reply is sent")
		s.Contains(asked.Input[0].Content, lease.Text)
	}
	for _, kept := range s.agent.History() {
		s.False(kept.Document, "the file is not part of the conversation's history")
	}
}

func (s *AgentSuite) TestCompactionNeverSumsUpAFile() {
	s.joinText()
	manual := Document{ID: "d-1", Name: "manual.pdf", Pages: 3, Text: "Hold the reset button for ten seconds."}
	_, err := s.agent.RespondTo(WithDocument(s.ctx, manual), "How do I reset it?", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.model.requests()) == 1 && !s.agent.Busy() }, "the first turn never finished")

	s.agent.applyCompaction(harness.Compacted{Prefix: s.agent.History(), Summary: "They asked how to reset it."})
	_, err = s.agent.RespondTo(s.ctx, "And how long does it take?", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the turn after compaction never reached the model")

	s.Contains(s.model.requests()[1].Input[0].Content, manual.Text, "the file is read whole after the history is summed up")
}

func (s *AgentSuite) TestTheFilesCarryOnToTheAgentThatTakesOver() {
	s.joinText()
	menu := Document{ID: "d-1", Name: "menu.docx", Text: "Margherita 9 EUR."}
	_, err := s.agent.RespondTo(WithDocument(s.ctx, menu), "What does the margherita cost?", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.model.requests()) == 1 && !s.agent.Busy() }, "the first turn never finished")
	old := s.agent

	s.joinText()
	s.agent.Continue(old)
	_, err = s.agent.RespondTo(s.ctx, "And with extra cheese?", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the new agent never answered")

	s.Contains(s.model.requests()[0].Input[0].Content, menu.Text, "a chat that goes to voice still reads its files")
}

func (s *AgentSuite) TestAFileSentWithATurnThatDoesNotStartIsNotKept() {
	s.joinText()

	_, err := s.agent.RespondTo(WithDocument(s.ctx, Document{ID: "d-1", Name: "a.pdf", Text: "text"}),
		"What is in it?", []llm.ImagePart{{}}, options.LLM{})

	s.Require().Error(err, "an image with nothing in it refuses the turn")
	s.Empty(s.agent.Documents(), "a refused turn leaves no file behind for the next one")
}

func (s *AgentSuite) TestAFileSentWithNoWordsIsAskedAbout() {
	s.joinText()

	_, err := s.agent.RespondTo(WithDocument(s.ctx, Document{ID: "d-1", Name: "a.pdf", Text: "text"}), "  ", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the turn never reached the model")

	said := harness.WithoutNotes(s.model.requests()[0].Input)
	s.True(strings.HasSuffix(said[len(said)-1].Content, documentAlone))
}

func (s *AgentSuite) TestTheRoomForAFileIsWhatTheFilesPinnedLeave() {
	s.joinText()
	s.agent.PinDocuments(Document{ID: "d-1", Name: "a.pdf", Text: strings.Repeat("x", 100_000)})

	s.Equal(MaxDocumentRunes-100_000, s.agent.DocumentRoom())
	s.agent.PinDocuments(Document{ID: "d-1", Name: "a.pdf", Text: strings.Repeat("x", 100_000)})
	s.Equal(MaxDocumentRunes-100_000, s.agent.DocumentRoom(), "a file pinned again is counted once")
}
