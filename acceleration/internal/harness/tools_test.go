package harness

import (
	"path/filepath"
	"strings"
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

func (s *ToolsSuite) TestOnlyAModelWithToolsIsToldHowToUseThem() {
	s.Empty(Tools{}.Prompt(), "a harness without tools adds nothing to the system prompt")
	s.Equal(usePolicy+" "+sayDo, testTools().Prompt())
	s.Empty(Tools{}.TextPrompt())
	s.Equal(textUsePolicy+" "+sayDo, testTools().TextPrompt())
}

func (s *ToolsSuite) TestAWrittenConversationFillsNoPauseBeforeACall() {
	// In writing every sentence before a call stays on the page, so a chain of calls opened
	// the answer with one hold phrase per link. What the instructions have the agent say
	// before acting, such as a read-back, still comes first.
	s.NotContains(textUsePolicy, "say one short sentence")
	s.Contains(textUsePolicy, "without announcing it")
	s.Contains(textUsePolicy, "write nothing before a call or between calls")
	s.Contains(textUsePolicy, "what your instructions ask you to write before acting")
	s.Contains(textUsePolicy, "reading the caller's details back")
	s.Contains(textUsePolicy, "Do not collect optional arguments")
	s.Contains(textUsePolicy, "require confirmation first, follow them")
}

func (s *ToolsSuite) TestTheUsePolicyKeepsTheReadBackThatPrecedesAnAction() {
	// Acting at once must not turn into acting in silence: what the operator has the agent say
	// before a tool, and in its absence what the agent is doing, comes first, in the same turn.
	s.Contains(usePolicy, "Before calling a tool, say one short sentence")
	s.Contains(usePolicy, "what your instructions ask you to say before acting")
	s.Contains(usePolicy, "reading the caller's details back")
	s.Contains(usePolicy, "or else what you are doing")
	s.Contains(usePolicy, "never in place of a required read-back")
	s.Contains(usePolicy, "call the tool in the same turn")
}

func (s *ToolsSuite) TestAGoAheadIsActedOnWithoutAskingForOptionalDetails() {
	for _, prompt := range []string{testTools().Prompt(), testTools().TextPrompt()} {
		s.Contains(prompt, "A yes or go-ahead after a read-back means call the tool now with what "+
			"you have, without asking for anything optional again.")
	}
}

func (s *ToolsSuite) TestAReadBackThatWaitsForAgreementEndsTheTurn() {
	// Voicebench operators say to read details back and wait for agreement before acting,
	// and Gemma read them back and called the tool in the same breath.
	for _, prompt := range []string{testTools().Prompt(), testTools().TextPrompt()} {
		s.Contains(prompt, "If your instructions say to wait for agreement after a read-back, "+
			"end your turn after it, with no hold phrase and no call, until they agree.")
	}
}

func (s *ToolsSuite) TestTheSentenceBeforeACallOpensWithTheHoldPhraseAndRunsOnIntoTheReadBack() {
	// A read-back takes seconds to say, so a hold phrase after it, or after the result, is
	// heard once the wait is over. It opens the sentence and the sentence carries on, because
	// a one-word sentence of its own leaves a pause an interruption falls into.
	s.Contains(usePolicy, "one short sentence that opens with a brief hold phrase")
	s.Contains(usePolicy, "and goes straight on, with no full stop between, into what")
	s.Contains(usePolicy, "in your own words, never the same twice")
	s.NotContains(usePolicy, `"`, "an example hold phrase is the one every reply opens with")
	s.Contains(usePolicy, "never as a sentence of its own")
	s.Contains(usePolicy, "never after the call or after a result")
}

func (s *ToolsSuite) TestAResultIsAnsweredFromWithoutAnotherHoldPhrase() {
	// The wait has one hold phrase. The tools a result calls for are called in the same reply,
	// together when they do not need each other, so a request with several steps is not one
	// spoken sentence and one model turn per tool.
	s.Contains(usePolicy, "After a result, answer from it")
	s.Contains(usePolicy, "call them straight away without a word, together when independent")
}

func (s *ToolsSuite) TestTheUsePolicyIsShortAndNamesNoToolOrDeployment() {
	built, err := DefaultTools()
	s.Require().NoError(err)
	for _, policy := range []string{usePolicy, textUsePolicy} {
		s.LessOrEqual(len(strings.Fields(policy)), 160)
		s.GreaterOrEqual(len(strings.Fields(policy)), 60)
		for _, tool := range append(built.Tools, testTools().Tools...) {
			s.NotContains(strings.ToLower(policy), strings.ToLower(tool.Name),
				"the policy is about any tool, not one of them")
		}
	}
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

func (s *ToolsSuite) TestSourcesAreOrderedSkillThenKnowledgeThenWeb() {
	skills := Skills{Skills: []Skill{{Name: "sales", Description: "pricing", Instructions: "quote"}}}
	built, err := DefaultTools()
	s.Require().NoError(err)
	lookup, _ := built.Lookup("lookup")
	search, _ := built.Lookup("search")

	// Declared in any order, the sources are still told skill, knowledge base, web.
	order := Tools{Tools: []Tool{search, lookup, skills.LoadTool()}}.SourceOrder()
	s.Contains(order, "a matching skill (load_skill), then the knowledge base (lookup), then the web (search)")
	s.Contains(order, "one at a time")
	s.Contains(order, "never let a later source contradict an earlier one")
	s.Contains(order, "Load a skill only when its description covers the request")
	s.Contains(order, "Always ask the knowledge base before the web")

	withoutSkills := Tools{Tools: []Tool{search, lookup}}.SourceOrder()
	s.Contains(withoutSkills, "the knowledge base (lookup), then the web (search)")
	s.NotContains(withoutSkills, "Load a skill")

	withoutKnowledge := Tools{Tools: []Tool{skills.LoadTool(), search}}.SourceOrder()
	s.Contains(withoutKnowledge, "a matching skill (load_skill), then the web (search)")
	s.NotContains(withoutKnowledge, "knowledge base before the web")
}

func (s *ToolsSuite) TestOneSourceHasNothingToOrder() {
	built, err := DefaultTools()
	s.Require().NoError(err)
	search, _ := built.Lookup("search")

	s.Empty(Tools{Tools: []Tool{search}}.SourceOrder())
	s.Empty(testTools().SourceOrder())
}

func (s *ToolsSuite) TestTheBuiltInSourcesDeferToTheOnesAboveThem() {
	built, err := DefaultTools()
	s.Require().NoError(err)
	lookup, _ := built.Lookup("lookup")
	search, _ := built.Lookup("search")

	s.Contains(lookup.Description, "after a matching skill and before searching the web")
	s.Contains(search.Description, "only when a matching skill and the knowledge base have nothing")
}

func (s *ToolsSuite) TestLookupMissesWhatWasNeverDeclared() {
	_, known := testTools().Lookup("hang_up")

	s.False(known)
}

func (s *ToolsSuite) TestAToolWithOptionalArgumentsNamesTheOnesItNeeds() {
	tools := Tools{Tools: []Tool{{
		Name:        "create_reservation",
		Description: "Book the table.",
		Parameters: map[string]any{
			"type":       "object",
			"properties": map[string]any{"time": map[string]any{}, "name": map[string]any{}, "phone": map[string]any{}},
			"required":   []any{"time", "name"},
		},
	}, {
		Name:        "check_outage",
		Description: "Check for an outage.",
		Parameters:  map[string]any{"type": "object", "properties": map[string]any{}},
	}}}

	requests := tools.Requests()
	s.Equal("Book the table. Only time, name are needed to call it; ask for the others only if "+
		"your instructions say to collect or verify them, and pass every one the caller gave.",
		requests[0].Description)
	s.Equal("Check for an outage.", requests[1].Description, "a tool with nothing optional says nothing more")
}
