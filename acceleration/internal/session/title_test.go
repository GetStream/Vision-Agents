package session

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/stretchr/testify/suite"
)

type TitleSuite struct {
	suite.Suite
}

func TestTitleSuite(t *testing.T) {
	suite.Run(t, new(TitleSuite))
}

func (s *TitleSuite) TestATitleIsReadOutOfWhatTheModelWrote() {
	named, err := parseTitle("```json\n{\"title\":\"\\\"Add push to an Android app.\\\"\",\"description\":\" They asked how. \"}\n```")

	s.Require().NoError(err)
	s.Equal("Add push to an Android app", named.Title)
	s.Equal("They asked how.", named.Description)
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
