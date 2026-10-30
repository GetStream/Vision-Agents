package session

import (
	"context"
	"log/slog"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/stretchr/testify/suite"
)

type TitleSuite struct {
	suite.Suite
}

func TestTitleSuite(t *testing.T) {
	suite.Run(t, new(TitleSuite))
}

func (s *TitleSuite) TestTheGreetingIsNotAnAnswerToNameTheConversationAfter() {
	// The greeting is the same on every call, so naming after it would call every
	// conversation the same thing until the next renaming.
	session := &Session{naming: &naming{}}
	// A naming in flight stops this from starting one, which needs a model.
	session.naming.running.Store(true)

	session.name(agent.Responded{TurnID: "say-1", Text: "Hi, thanks for calling Acme, how can I help?"})
	s.Zero(session.naming.answered)

	session.name(agent.Responded{TurnID: "turn-1", Text: "We open at nine."})
	s.Equal(1, session.naming.answered)
}

func (s *TitleSuite) TestATitleIsReadOutOfWhatTheModelWrote() {
	named, err := parseTitle("```json\n{\"title\":\"\\\"Add push to an Android app.\\\"\",\"description\":\" They asked how. \"}\n```")

	s.Require().NoError(err)
	s.Equal("Add push to an Android app", named.Title)
	s.Equal("They asked how.", named.Description)
}

// The namer runs on llm-fast, whose failover models think before they answer, out of the same
// cap as the title.
func (s *TitleSuite) TestAModelThatThinksBeforeItNamesStillWritesTheWholeTitle() {
	reasoning := llmrouter.NewRegistry()
	reasoning.Register("stub", func(routing.Spec) (llmrouter.Provider, error) {
		return &llmtest.Thinker{Thinks: 1300, Answer: `{"title":"Add push to an Android app","description":"They asked how."}`}, nil
	})
	config := stubConfig()
	config.Aliases["llm-fast"] = routing.Alias{Languages: []string{"en"}, RequireRealtime: true}
	router, err := llmrouter.New(llmrouter.Options{Config: config, Registry: reasoning, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)

	named, err := (&titler{llm: router}).ask(context.Background(), Spec{CustomerID: "customer-named"}, "title-thought-over",
		[]spoken{{text: "how do I add push?"}})

	s.Require().NoError(err)
	s.Equal("Add push to an Android app", named.Title)
}

func (s *TitleSuite) TestNoTitleIsNotOne() {
	_, err := parseTitle(`{"title":"  ","description":"something"}`)

	s.Require().Error(err)
}

func (s *TitleSuite) TestALongTitleIsCut() {
	named, err := parseTitle(`{"title":"` + strings.Repeat("word ", 40) + `"}`)

	s.Require().NoError(err)
	s.LessOrEqual(utf8.RuneCountInString(named.Title), titleRunes)
	s.True(strings.HasSuffix(named.Title, "…"))
}

func (s *TitleSuite) TestTheConversationIsQuotedFromItsBeginningWithBothSides() {
	said := []spoken{{text: "how do I add push?"}, {agent: true, text: "register the device first."}}
	for range titleLimit {
		said = append(said, spoken{text: "and then?"})
	}

	written := quoted(said)

	s.Contains(written, "Person: how do I add push?")
	s.Contains(written, "Assistant: register the device first.")
	s.Equal(titleLimit, strings.Count(written, "\n")-2)
}

func (s *TitleSuite) TestResumedHistoryKeepsOnlyWhatWasSaid() {
	said := spokenOf([]llm.Message{
		{Role: llm.System, Content: "be helpful"},
		{Role: llm.User, Content: "hello"},
		{Role: llm.Assistant, Content: ""},
		{Role: llm.Assistant, Content: "hi"},
	})

	s.Equal([]spoken{{text: "hello"}, {agent: true, text: "hi"}}, said)
}
