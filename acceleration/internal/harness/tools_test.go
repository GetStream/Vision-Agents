package harness

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/suite"
)

// testTools are two tools whose schemas are the shape a real one has, so a test about what
// reaches the model is testing something the model could act on.
func testTools() Tools {
	return Tools{Tools: []Tool{
		{
			Name:        "transfer",
			Description: "hand the caller to a human",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"to": map[string]any{"type": "string"}},
				"required":   []any{"to"},
			},
		},
		{
			Name:        "press",
			Description: "press digits at a menu",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"digits": map[string]any{"type": "string"}},
				"required":   []any{"digits"},
			},
		},
	}}
}

type ToolsSuite struct {
	suite.Suite
}

func TestToolsSuite(t *testing.T) {
	suite.Run(t, new(ToolsSuite))
}

func (s *ToolsSuite) TestTheBuiltInSetIsUsable() {
	tools, err := DefaultTools()

	s.Require().NoError(err)
	s.NotEmpty(tools.Tools, "an agent with telephony should work without an external file")

	transfer, known := tools.Lookup("transfer")
	s.Require().True(known)
	s.NotEmpty(transfer.Description)
	s.NotEmpty(transfer.Parameters, "the model has to be told what a transfer needs")

	_, known = tools.Lookup("press")
	s.True(known)
}

func (s *ToolsSuite) TestLoadFallsBackToTheBuiltInSet() {
	loaded, err := LoadTools("")
	s.Require().NoError(err)

	builtIn, err := DefaultTools()
	s.Require().NoError(err)
	s.Equal(builtIn, loaded)
}

func (s *ToolsSuite) TestLoadReportsAFileItCannotRead() {
	_, err := LoadTools(filepath.Join(s.T().TempDir(), "nothing.yaml"))

	s.ErrorContains(err, "read tools")
}

func (s *ToolsSuite) TestAToolWithoutADescriptionIsRefused() {
	// A tool the model is told nothing about is one it can never know when to reach for.
	err := Tools{Tools: []Tool{{Name: "transfer"}}}.Validate()

	s.ErrorContains(err, "description")
}

func (s *ToolsSuite) TestAToolWithoutANameIsRefused() {
	err := Tools{Tools: []Tool{{Description: "does something"}}}.Validate()

	s.ErrorContains(err, "name")
}

func (s *ToolsSuite) TestTwoToolsWithOneNameAreRefused() {
	// Which one ran would depend on lookup order, and the model has no way to say.
	err := Tools{Tools: []Tool{
		{Name: "transfer", Description: "one way"},
		{Name: "transfer", Description: "another way"},
	}}.Validate()

	s.ErrorContains(err, "twice")
}

func (s *ToolsSuite) TestAnEmptySetOffersNothingRatherThanAnEmptyList() {
	s.Nil(Tools{}.Requests())
}

func (s *ToolsSuite) TestRequestsCarryTheSchemaTheModelFillsIn() {
	requests := testTools().Requests()

	s.Require().Len(requests, 2)
	s.Equal("transfer", requests[0].Name)
	s.Equal("hand the caller to a human", requests[0].Description)
	s.Equal("object", requests[0].Parameters["type"])
}

func (s *ToolsSuite) TestLookupMissesWhatWasNeverDeclared() {
	_, known := testTools().Lookup("hang_up")

	s.False(known)
}
