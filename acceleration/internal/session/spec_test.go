package session

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type SpecSuite struct {
	suite.Suite
}

func TestSpecSuite(t *testing.T) {
	suite.Run(t, new(SpecSuite))
}

// spec is a session that normalizes, so a test only has to say the part it is about.
func (s *SpecSuite) spec(keyterms []string) Spec {
	return Spec{CallID: "call-1", CustomerID: "acme", Keyterms: keyterms}
}

func (s *SpecSuite) TestAConfigsKeytermsBecomeTheSessions() {
	spec := FromConfig(store.AgentConfig{
		CustomerID: "acme",
		Keyterms:   []string{"Vision Agents", "Stream"},
	})

	s.Equal([]string{"Vision Agents", "Stream"}, spec.Keyterms)
}

func (s *SpecSuite) TestAConfigsPluginsBecomeTheSessions() {
	spec := FromConfig(store.AgentConfig{
		CustomerID: "acme",
		Plugins:    []string{"slack", "calendly"},
	})

	s.Equal([]string{"slack", "calendly"}, spec.Plugins)
}

func (s *SpecSuite) TestAConfigsSandboxBecomesTheSessions() {
	spec := FromConfig(store.AgentConfig{
		CustomerID: "acme",
		Sandbox:    daytonaProvider,
	})

	s.Equal(daytonaProvider, spec.Sandbox)
}

func (s *SpecSuite) TestKeytermsAreTidiedOnTheWayIn() {
	spec := s.spec([]string{" Vision Agents ", "", "Stream"})

	s.Require().NoError(spec.Normalize())

	s.Equal([]string{"Vision Agents", "Stream"}, spec.Keyterms)
}

func (s *SpecSuite) TestMoreKeytermsThanAProviderTakesIsRefused() {
	// A list no transcriber would accept is worth refusing here, rather than opening the
	// call and failing on the connection the caller cannot see.
	many := make([]string, stt.MaxKeyterms+1)
	for i := range many {
		many[i] = fmt.Sprintf("term-%d", i)
	}
	spec := s.spec(many)

	err := spec.Normalize()

	s.ErrorContains(err, "keyterms")
}

func (s *SpecSuite) TestTheLargestListAProviderTakesIsAllowed() {
	many := make([]string, stt.MaxKeyterms)
	for i := range many {
		many[i] = fmt.Sprintf("term-%d", i)
	}
	spec := s.spec(many)

	s.Require().NoError(spec.Normalize())

	s.Len(spec.Keyterms, stt.MaxKeyterms)
}

func (s *SpecSuite) TestNativeSessionsKeepDelegatedWork() {
	for _, spec := range []Spec{
		{SkillNames: []string{"think"}},
		{Skills: &harness.Skills{Skills: []harness.Skill{{Name: "think"}}}},
		{SubagentTarget: "llm-thinking"},
	} {
		spec.STSTarget = "openai/gpt-realtime-2"
		spec.CallID = "call-1"
		spec.CustomerID = "acme"
		s.NoError(spec.Normalize())
	}
	spec := s.spec(nil)
	spec.STSTarget = "openai/gpt-realtime-2"
	s.NoError(spec.Normalize())
	s.Empty(spec.LLMTarget)
	s.Empty(spec.STTTarget)
	s.Empty(spec.TTSTarget)
}

func (s *SpecSuite) TestAConfigsNameIsCarriedOnTheSession() {
	spec := FromConfig(store.AgentConfig{CustomerID: "acme", ID: "cfg-1", Name: "docs"})

	s.Equal("docs", spec.AgentName)
	s.Equal("cfg-1", spec.ConfigID)
}

func (s *SpecSuite) TestModelOverwritesBeatTheConfigsRoutes() {
	spec := FromConfig(store.AgentConfig{CustomerID: "acme", LLM: "llm-flow", STT: "stt-fast"})
	spec.CallID = "call-1"
	spec.ModelOverwrites = store.ModelOverwrites{LLM: "llm-thinking"}

	s.Require().NoError(spec.Normalize())

	s.Equal("llm-thinking", spec.LLMTarget)
	s.Equal("stt-fast", spec.STTTarget, "a route nobody overwrote is still the config's")
}

func (s *SpecSuite) TestThinkingBecomesTheReasoningEffort() {
	spec := s.spec(nil)
	spec.ModelOverwrites = store.ModelOverwrites{Thinking: "high"}

	s.Require().NoError(spec.Normalize())

	// Routing is untouched: how hard to think is a per-request option, not a different model.
	s.Empty(spec.ModelOverwrites.LLM)
	s.Equal("high", spec.LLMOverwrites().ReasoningEffort)
}

func (s *SpecSuite) TestOverwritingTheSubagentBeatsTheConfigs() {
	spec := FromConfig(store.AgentConfig{CustomerID: "acme", Subagent: "llm-flow"})
	spec.CallID = "call-1"
	spec.ModelOverwrites = store.ModelOverwrites{Subagent: "llm-thinking"}

	s.Require().NoError(spec.Normalize())

	s.Equal("llm-thinking", spec.SubagentTarget)
}

func (s *SpecSuite) TestIncognitoRecordsNothing() {
	spec := Spec{CustomerID: "acme", Text: true, Incognito: true}
	spec.PersistConversation = true
	spec.ConversationID = "agent:something"

	s.Require().NoError(spec.Normalize())

	// Honoured once, here, rather than at each of the places that records something: one
	// place that forgot would be a conversation kept against its caller's wishes.
	s.False(spec.PersistConversation)
	s.Empty(spec.ConversationID)
	s.True(spec.NoReview)
}

func (s *SpecSuite) TestAProjectIsAlsoACostLabel() {
	spec := s.spec(nil)
	spec.Project = "Health"

	s.Require().NoError(spec.Normalize())

	s.Equal("Health", spec.Tags["project"], "billing should not need the caller to say it twice")
}

func (s *SpecSuite) TestAnExplicitProjectTagWins() {
	spec := s.spec(nil)
	spec.Project = "Health"
	spec.Tags = routing.Tags{"project": "spelled-out"}

	s.Require().NoError(spec.Normalize())

	s.Equal("spelled-out", spec.Tags["project"], "somebody who wrote the tag meant the tag")
	s.Equal("Health", spec.Project)
}

func (s *SpecSuite) TestLLMOverwritesCarryOnlyTheSafeKnobs() {
	temperature := 0.2
	tokens := 2048
	spec := s.spec(nil)
	spec.ModelOverwrites = store.ModelOverwrites{
		LLM: "llm-thinking", Thinking: "low",
		Temperature: &temperature, MaxOutputTokens: &tokens, Verbosity: "high",
	}

	over := spec.LLMOverwrites()

	s.Equal("low", over.ReasoningEffort)
	s.Equal(&temperature, over.Temperature)
	s.Equal(&tokens, over.MaxOutputTokens)
	s.Equal("high", over.Verbosity)
	// Instructions belong to the agent. A caller able to rewrite them could make a session
	// impersonate a different agent, so they are not something a session may overwrite.
	s.Empty(over.Instructions)
	s.Empty(over.Target)
}
