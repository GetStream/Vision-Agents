package session

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
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
		{Subagents: map[string]string{"vision": "vlm"}},
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
