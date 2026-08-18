package agent

import (
	"context"
	"errors"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// stubKnowledge is a knowledge base with no search engine in it: it answers with whatever
// the test wrote down, and records what it was asked.
type stubKnowledge struct {
	documents []knowledge.Document
	err       error

	mu     sync.Mutex
	asked  []knowledge.Query
	closed bool
}

func (k *stubKnowledge) Search(_ context.Context, query knowledge.Query) ([]knowledge.Document, error) {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.asked = append(k.asked, query)
	if k.err != nil {
		return nil, k.err
	}
	return k.documents, nil
}

func (k *stubKnowledge) Provider() string { return "stub" }

func (k *stubKnowledge) Close() error {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.closed = true
	return nil
}

func (k *stubKnowledge) queries() []knowledge.Query {
	k.mu.Lock()
	defer k.mu.Unlock()
	return append([]knowledge.Query(nil), k.asked...)
}

// reads gives the agent a knowledge base and the tools that reach it.
func (s *AgentSuite) reads(documents ...knowledge.Document) {
	s.knows = &stubKnowledge{documents: documents}
	s.namespace = "handbook"
	tools, err := harness.DefaultTools()
	s.Require().NoError(err)
	s.tools = tools
}

func (s *AgentSuite) TestALookupAnswersOutOfTheKnowledgeBase() {
	s.reads(knowledge.Document{Text: "Delivery is free over $50.", Source: "shipping.md"})
	s.join(false)
	s.model.reply = []string{"Let me check that."}
	s.asksFor("lookup", `{"query":"what does delivery cost"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "how much is delivery")

	ran := s.awaitToolRan()
	s.Equal("lookup", ran.Tool)
	s.NoError(ran.Err)
	s.Contains(ran.Result, "Delivery is free over $50.")
	s.Contains(ran.Result, "shipping.md")

	asked := s.knows.queries()
	s.Require().Len(asked, 1)
	s.Equal("handbook", asked[0].Namespace, "an agent must only read its own knowledge base")
	s.Equal("what does delivery cost", asked[0].Text)
}

func (s *AgentSuite) TestAQuestionTheKnowledgeBaseDoesNotCoverIsAnsweredInWords() {
	// An empty answer would have the model invent one, which on a support line is worse
	// than admitting the handbook says nothing.
	s.reads()
	s.join(false)
	s.model.reply = []string{"One moment."}
	s.asksFor("lookup", `{"query":"do you deliver to mars"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "do you deliver to mars")

	ran := s.awaitToolRan()
	s.NoError(ran.Err, "an unanswered question is not a broken tool")
	s.Contains(ran.Result, "rather than guessing")
}

func (s *AgentSuite) TestASearchThatFailedIsToldToTheModelInWords() {
	s.reads()
	s.knows.err = errors.New("the index is rebuilding")
	s.join(false)
	s.model.reply = []string{"One moment."}
	s.asksFor("lookup", `{"query":"opening hours"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "when do you open")

	ran := s.awaitToolRan()
	s.Require().Error(ran.Err)
	s.Contains(ran.Result, "did not work")
}

func (s *AgentSuite) TestLookupIsNotOfferedWithoutAKnowledgeBase() {
	// A model told it can search and then refused would promise the caller an answer it
	// has no way of getting.
	tools, err := harness.DefaultTools()
	s.Require().NoError(err)
	s.tools = tools
	s.join(false)

	_, offered := s.agent.availableTools().Lookup("lookup")

	s.False(offered)
}
