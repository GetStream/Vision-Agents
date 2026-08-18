package knowledge

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type KnowledgeSuite struct {
	suite.Suite
}

func TestKnowledgeSuite(t *testing.T) {
	suite.Run(t, new(KnowledgeSuite))
}

func (s *KnowledgeSuite) TestThePromptCarriesThePassagesAndWhereTheyCameFrom() {
	prompt := Prompt([]Document{
		{Text: "Delivery is free over $50.", Source: "shipping.md"},
		{Text: "We open at nine."},
	})

	s.Contains(prompt, "Delivery is free over $50.")
	s.Contains(prompt, "shipping.md")
	s.Contains(prompt, "We open at nine.")
}

func (s *KnowledgeSuite) TestFindingNothingTellsTheModelToSaySo() {
	// A model handed an empty answer invents one, which on a support line is worse than
	// admitting the handbook does not cover it.
	prompt := Prompt(nil)

	s.NotEmpty(prompt)
	s.Contains(prompt, "rather than guessing")
}
