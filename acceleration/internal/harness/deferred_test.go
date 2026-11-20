package harness

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

// DeferredToolsSuite covers how the model finds and loads the tools held back from it.
type DeferredToolsSuite struct {
	suite.Suite
	deferred []Tool
}

func TestDeferredToolsSuite(t *testing.T) {
	suite.Run(t, new(DeferredToolsSuite))
}

func (s *DeferredToolsSuite) SetupTest() {
	s.deferred = []Tool{
		{Name: "lookup_booking", Description: "Find a booking by its reference.\n\nThe reference is six letters."},
		{Name: "refund_booking", Description: "Give the traveller their money back for a cancelled trip."},
		{Name: "void_ticket", Description: "Void a ticket issued today.", Parameters: map[string]any{"type": "object"}},
	}
}

// found are the names a search_tool answer lists, in the order it lists them.
func (s *DeferredToolsSuite) found(answer string) []string {
	var listed struct {
		Tools []struct{ Name, Description string } `json:"tools"`
	}
	s.Require().NoError(json.Unmarshal([]byte(answer), &listed), answer)
	var names []string
	for _, tool := range listed.Tools {
		names = append(names, tool.Name)
	}
	return names
}

func (s *DeferredToolsSuite) TestASearchFindsAToolByAWordOfItsDescription() {
	for _, query := range []string{"money back", "the MONEY", "trip got cancelled, money?"} {
		found := s.found(SearchDeferred(s.deferred, query))
		s.Require().NotEmpty(found, query)
		s.Equal("refund_booking", found[0], query)
	}
}

func (s *DeferredToolsSuite) TestASearchRanksAToolNamedByTheWordFirst() {
	found := s.found(SearchDeferred(s.deferred, "booking"))

	s.Equal([]string{"lookup_booking", "refund_booking"}, found)
}

func (s *DeferredToolsSuite) TestASearchListsEachToolByItsFirstLine() {
	answer := SearchDeferred(s.deferred, "reference")

	s.Contains(answer, "Find a booking by its reference.")
	s.NotContains(answer, "six letters")
}

func (s *DeferredToolsSuite) TestASearchThatMatchesNothingSaysSo() {
	answer := SearchDeferred(s.deferred, "weather")

	s.True(strings.HasPrefix(answer, `No tool matches "weather".`), answer)
}

func (s *DeferredToolsSuite) TestALoadHandsOverTheWholeToolAndNamesWhatItCouldNotFind() {
	answer, found := LoadDeferred(s.deferred, []string{"void_ticket", "book_hotel", "void_ticket"})

	s.Equal([]string{"void_ticket"}, found)
	var loaded struct {
		Loaded []struct {
			Name        string         `json:"name"`
			InputSchema map[string]any `json:"input_schema"`
		} `json:"loaded"`
		Unknown []string `json:"unknown"`
	}
	s.Require().NoError(json.Unmarshal([]byte(answer), &loaded))
	s.Len(loaded.Loaded, 1)
	s.Equal(map[string]any{"type": "object"}, loaded.Loaded[0].InputSchema)
	s.Equal([]string{"book_hotel"}, loaded.Unknown)
}

func (s *DeferredToolsSuite) TestAToolBothOfferedAndDeferredIsRefused() {
	tools := Tools{Tools: s.deferred[:1], Deferred: s.deferred}

	s.ErrorContains(tools.Validate(), "lookup_booking is declared twice")
}
