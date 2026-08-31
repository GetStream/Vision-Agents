package search

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type SearchSuite struct {
	suite.Suite
}

func TestSearchSuite(t *testing.T) {
	suite.Run(t, new(SearchSuite))
}

func (s *SearchSuite) TestAQuestionWithNothingInItIsRefused() {
	s.ErrorContains(Query{Text: "   "}.Validate(), "nothing to look for")
	s.NoError(Query{Text: "traffic on I-70"}.Validate())
}

func (s *SearchSuite) TestNothingFoundIsSaidInWords() {
	// The model is about to speak either way. Told plainly that the search found nothing,
	// it says so; handed an empty string it reads as a broken tool and gets apologised for.
	rendered := Prompt(Result{})

	s.Contains(rendered, "could not find out")
}

func (s *SearchSuite) TestWhatWasFoundIsRenderedForTheModel() {
	rendered := Prompt(Result{
		Answer: "I-70 is clear through the tunnel.",
		Documents: []Document{
			{Title: "COtrip", URL: "https://cotrip.org", Text: "No closures reported."},
		},
	})

	s.Contains(rendered, "I-70 is clear through the tunnel.")
	s.Contains(rendered, "No closures reported.")
	s.Contains(rendered, "COtrip")
	s.Contains(rendered, "prefer it over", "the model is told today beats what it remembers")
}

func (s *SearchSuite) TestASourceWithNoTitleIsNamedByItsAddress() {
	rendered := Prompt(Result{Documents: []Document{
		{URL: "https://cotrip.org", Text: "No closures reported."},
	}})

	s.Contains(rendered, "https://cotrip.org")
}

func (s *SearchSuite) TestASourceWithNothingToReadIsLeftOut() {
	rendered := Prompt(Result{
		Answer:    "It is clear.",
		Documents: []Document{{Title: "Empty", Text: "   "}},
	})

	s.NotContains(rendered, "Empty")
}
