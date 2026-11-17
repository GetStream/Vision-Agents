package contracttest

import (
	"context"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// The tools a SourceSubject's provider offers, by the names the source lists them under.
const (
	// ToolEcho takes {"text": string}, text required, and answers with the text.
	ToolEcho = "echo"
	// ToolLarge takes no arguments and answers with more than core.MaxResultBytes of text that
	// is not ASCII, so a cut can land inside a character.
	ToolLarge = "large"
	// ToolBroken takes no arguments and reports an error whose only content is an image.
	ToolBroken = "broken"
)

// SourceSubject is one tool source under SourceContract, with a provider the subject runs that
// offers ToolEcho, ToolLarge and ToolBroken.
type SourceSubject struct {
	Source core.ToolSource
	// Binding reaches the provider through the connection's client. Its Binding.Name is the
	// alias Open offers the tools under.
	Binding core.ResolvedBinding
	// Calls is how many calls of tool reached the provider.
	Calls func(tool string) int
	// ChangeSchema makes the provider offer ToolEcho with another input schema from now on.
	ChangeSchema func()
}

// SourceContract is the suite every core.ToolSource passes. Run it with suite.Run and a New
// that builds a fresh SourceSubject for each test.
type SourceContract struct {
	suite.Suite
	New func(t *testing.T) SourceSubject

	ctx     context.Context
	subject SourceSubject
}

func (s *SourceContract) SetupTest() {
	s.ctx = context.Background()
	s.subject = s.New(s.T())
	s.Require().NotNil(s.subject.Source, "SourceSubject.Source")
	s.Require().NotNil(s.subject.Calls, "SourceSubject.Calls")
	s.Require().NotNil(s.subject.ChangeSchema, "SourceSubject.ChangeSchema")
}

func (s *SourceContract) TestDiscoverGivesTheSameDigestForTheSameSchema() {
	first := s.digests()
	second := s.digests()

	s.Equal(first, second)
	for _, tool := range []string{ToolEcho, ToolLarge, ToolBroken} {
		s.NotEmpty(first[tool], tool)
	}
	s.NotEqual(first[ToolEcho], first[ToolLarge], "tools with different schemas have different digests")
}

func (s *SourceContract) TestAChangedSchemaHasAnotherDigest() {
	before := s.digests()[ToolEcho]
	s.subject.ChangeSchema()

	s.NotEqual(before, s.digests()[ToolEcho])
}

func (s *SourceContract) TestOnlyTheGrantedToolsAreOffered() {
	names := s.names()
	set := s.open(ToolEcho, ToolBroken)

	var offered []string
	for _, tool := range set.Tools() {
		offered = append(offered, tool.Name)
	}
	s.ElementsMatch([]string{names[ToolEcho], names[ToolBroken]}, offered)
}

func (s *SourceContract) TestAnUngrantedToolIsNeverDispatched() {
	names := s.names()
	set := s.open(ToolEcho)

	_, err := set.Call(s.ctx, llm.ToolCall{Name: names[ToolLarge], Arguments: "{}"})

	s.Error(err)
	s.Zero(s.subject.Calls(ToolLarge), "the call never reached the provider")
}

func (s *SourceContract) TestAToolWhoseSchemaChangedSinceItWasGrantedIsHidden() {
	names := s.names()
	grants := s.grants(ToolEcho)
	s.subject.ChangeSchema()

	set, err := s.subject.Source.Open(s.ctx, s.subject.Binding, grants)
	s.Require().NoError(err)
	s.T().Cleanup(set.Close)
	_, called := set.Call(s.ctx, llm.ToolCall{Name: names[ToolEcho], Arguments: `{"text":"hi"}`})

	s.Empty(set.Tools())
	s.Error(called)
	s.Zero(s.subject.Calls(ToolEcho))
}

func (s *SourceContract) TestArgumentsTheSchemaRefusesNeverReachTheProvider() {
	set := s.open(ToolEcho)
	name := set.Tools()[0].Name

	for _, arguments := range []string{`{}`, `{"text":3}`, `[]`, `not json`} {
		_, err := set.Call(s.ctx, llm.ToolCall{Name: name, Arguments: arguments})
		s.Error(err, arguments)
	}
	s.Zero(s.subject.Calls(ToolEcho))
}

func (s *SourceContract) TestAGrantedToolAnswers() {
	set := s.open(ToolEcho)

	result, err := set.Call(s.ctx, llm.ToolCall{Name: set.Tools()[0].Name, Arguments: `{"text":"hello there"}`})

	s.Require().NoError(err)
	s.Equal("hello there", llm.TextOf(result.Parts))
	s.Equal(1, s.subject.Calls(ToolEcho))
}

func (s *SourceContract) TestAResultOverTheCapIsCutWithTheMarker() {
	set := s.open(ToolLarge)

	result, err := set.Call(s.ctx, llm.ToolCall{Name: set.Tools()[0].Name, Arguments: "{}"})

	s.Require().NoError(err)
	text := llm.TextOf(result.Parts)
	s.LessOrEqual(len(text), core.MaxResultBytes)
	s.True(strings.HasSuffix(text, core.TruncatedMarker), "ends with the marker")
	s.True(utf8.ValidString(text), "no character is split")
}

func (s *SourceContract) TestAnErrorWithNoTextIsStillAnError() {
	set := s.open(ToolBroken)

	_, err := set.Call(s.ctx, llm.ToolCall{Name: set.Tools()[0].Name, Arguments: "{}"})

	var reported *core.ToolError
	s.Require().ErrorAs(err, &reported)
	s.NotEmpty(reported.Message, "the model is told something")
}

// digests is each tool's digest, by its name at the provider.
func (s *SourceContract) digests() map[string]string {
	specs, err := s.subject.Source.Discover(s.ctx, s.subject.Binding)
	s.Require().NoError(err)
	digests := map[string]string{}
	for _, spec := range specs {
		digests[spec.Name] = spec.SchemaDigest
	}
	return digests
}

// grants grants the tools at the schema they have now.
func (s *SourceContract) grants(tools ...string) []core.ToolGrant {
	digests := s.digests()
	grants := make([]core.ToolGrant, 0, len(tools))
	for _, tool := range tools {
		s.Require().NotEmpty(digests[tool], tool)
		grants = append(grants, core.ToolGrant{Name: tool, SchemaDigest: digests[tool]})
	}
	return grants
}

// open is a toolset of the tools granted at the schema they have now.
func (s *SourceContract) open(tools ...string) core.Toolset {
	set, err := s.subject.Source.Open(s.ctx, s.subject.Binding, s.grants(tools...))
	s.Require().NoError(err)
	s.T().Cleanup(set.Close)
	return set
}

// names is the name the model is offered each tool under, by its name at the provider, each
// read from a toolset that grants that tool alone. A test that calls a name learns it here, so
// the contract assumes nothing about how a source names what it offers.
func (s *SourceContract) names() map[string]string {
	names := map[string]string{}
	for _, tool := range []string{ToolEcho, ToolLarge, ToolBroken} {
		offered := s.open(tool).Tools()
		s.Require().Len(offered, 1, tool)
		names[tool] = offered[0].Name
	}
	return names
}
