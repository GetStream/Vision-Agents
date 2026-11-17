package campaign

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

type VariablesSuite struct {
	suite.Suite
}

func TestVariablesSuite(t *testing.T) {
	suite.Run(t, new(VariablesSuite))
}

func (s *VariablesSuite) contact() map[string]string {
	return map[string]string{"first_name": "Anna", "plan": "gold"}
}

func (s *VariablesSuite) TestAVariableNameIsLowercaseAndAtMost64Characters() {
	s.True(VariableName.MatchString("first_name"))
	s.True(VariableName.MatchString("a2"))
	s.False(VariableName.MatchString("First"))
	s.False(VariableName.MatchString("2nd"))
	s.False(VariableName.MatchString("first-name"))
	s.False(VariableName.MatchString(strings.Repeat("a", 65)))
}

func (s *VariablesSuite) TestAPlaceholderIsFilledWithTheContactsValue() {
	s.Equal("Hi Anna", Render("Hi {{first_name}}", s.contact()))
	s.Equal("Hi Anna", Render("Hi {{ first_name }}", s.contact()))
}

func (s *VariablesSuite) TestAPlaceholderTheContactLacksBecomesEmpty() {
	s.Equal("Plan !", Render("Plan {{ city }}!", s.contact()))
}

func (s *VariablesSuite) TestBracesThatAreNotAVariableAreLeftAlone() {
	s.Equal("{{ 1+1 }}", Render("{{ 1+1 }}", s.contact()))
	s.Equal("{x}", Render("{x}", s.contact()))
}

func (s *VariablesSuite) TestVariablesTheConfigNeverMentionsAreBriefed() {
	brief := Brief([]string{"Hi {{first_name}}"}, s.contact())

	s.Equal("About the person you are calling:\n- plan: gold", brief)
}

func (s *VariablesSuite) TestBriefedVariablesAreSortedByName() {
	variables := map[string]string{"zip": "1", "age": "2", "plan": "3"}

	s.Equal("About the person you are calling:\n- age: 2\n- plan: 3\n- zip: 1", Brief(nil, variables))
}

func (s *VariablesSuite) TestNoBriefWhenEverythingIsMentioned() {
	s.Empty(Brief([]string{"{{first_name}} is on {{ plan }}"}, s.contact()))
}

func (s *VariablesSuite) TestAVariableMentionedInAnyTextIsNotBriefed() {
	s.Empty(Brief([]string{"Hi.", "You are on {{plan}}, {{first_name}}."}, s.contact()))
}
