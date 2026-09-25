package openweights

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

type OpenWeightsSuite struct {
	suite.Suite
}

func TestOpenWeightsSuite(t *testing.T) {
	suite.Run(t, new(OpenWeightsSuite))
}

// fields runs the switch the way openaicompat would.
func (s *OpenWeightsSuite) fields(model string, thinking bool, effort string) map[string]any {
	build := RequestFields(model, thinking)
	s.Require().NotNil(build, "%s should have a switch", model)
	return build(llm.ResponseParams{}, effort)
}

func (s *OpenWeightsSuite) TestTheSameWeightsAreRecognisedHoweverAHostSpellsThem() {
	for _, id := range []string{"zai-org/GLM-5.3-Flash", "glm:5.3-flash", "z-ai-glm-5-3-flash", "GLM-5.2"} {
		s.Equal(FamilyGLM, FamilyOf(id), id)
	}
	for _, id := range []string{"deepseek-ai/DeepSeek-V4.1-Flash", "deepseek:v4-flash-0731", "deepseek-v4-1-flash"} {
		s.Equal(FamilyDeepSeek, FamilyOf(id), id)
	}
	for _, id := range []string{"deepseek-ai/DeepSeek-V4-Pro-0813", "deepseek:v4-pro"} {
		s.Equal(FamilyDeepSeekPro, FamilyOf(id), id)
	}
	s.Equal(FamilyQwen, FamilyOf("Qwen/Qwen3.8-27B"))
	s.Equal(FamilyKimi, FamilyOf("moonshotai/Kimi-K3"))
	s.Equal(FamilyMiniMax, FamilyOf("MiniMaxAI/MiniMax-M3"))
}

func (s *OpenWeightsSuite) TestAModelWithNothingToSwitchIsAskedForNothing() {
	for _, id := range []string{"google/gemma-4-26B-A4B-it", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B", "XiaomiMiMo/MiMo-V2.5", "tencent/Hy3"} {
		s.Equal(FamilyNone, FamilyOf(id), id)
		s.False(Thinks(id), id)
		s.Nil(RequestFields(id, false), id)
	}
}

func (s *OpenWeightsSuite) TestEachFamilyIsToldNotToThinkTheWayItsOwnTemplateReadsIt() {
	s.Equal(
		map[string]any{"thinking": map[string]any{"type": "disabled"}},
		s.fields("zai-org/GLM-5.3-Flash", false, ""),
	)
	s.Equal(
		map[string]any{"thinking": map[string]any{"type": "disabled"}},
		s.fields("MiniMaxAI/MiniMax-M3", false, ""),
	)
	s.Equal(
		map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}},
		s.fields("Qwen/Qwen3.8-2.4T-A95B", false, ""),
	)
	s.Equal(
		map[string]any{"chat_template_kwargs": map[string]any{"thinking": false}},
		s.fields("moonshotai/Kimi-K3", false, ""),
	)
	s.Equal(
		map[string]any{"chat_template_kwargs": map[string]any{"thinking": false}},
		s.fields("deepseek-ai/DeepSeek-V4.1-Flash", false, ""),
	)
}

func (s *OpenWeightsSuite) TestTurningThinkingOnSaysSoRatherThanSayingNothing() {
	s.Equal(
		map[string]any{"thinking": map[string]any{"type": "enabled"}},
		s.fields("zai-org/GLM-5.3", true, ""),
	)
	s.Equal(
		map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": true}},
		s.fields("Qwen/Qwen3.8-27B", true, ""),
	)
}

func (s *OpenWeightsSuite) TestOnlyDeepSeekProCarriesAReasoningEffort() {
	s.Equal(
		map[string]any{
			"thinking":         map[string]any{"type": "enabled"},
			"reasoning_effort": "high",
		},
		s.fields("deepseek-ai/DeepSeek-V4-Pro-0813", true, "high"),
	)
	s.Equal([]string{"low", "medium", "high"}, Efforts("deepseek-ai/DeepSeek-V4-Pro-0813"))

	// A family with no ladder is sent no effort even when one is resolved for it.
	s.Equal(
		map[string]any{"thinking": map[string]any{"type": "enabled"}},
		s.fields("zai-org/GLM-5.3", true, "high"),
	)
	s.Empty(Efforts("zai-org/GLM-5.3"))
}

func (s *OpenWeightsSuite) TestCapabilitiesFollowWhetherTheModelIsThinking() {
	quiet := Capabilities("zai-org/GLM-5.3-Flash", false, "")
	s.False(quiet.StreamsReasoning)
	s.Empty(quiet.ReasoningEfforts)

	loud := Capabilities("deepseek-ai/DeepSeek-V4-Pro-0813", true, "medium")
	s.True(loud.StreamsReasoning)
	s.Equal([]string{"low", "medium", "high"}, loud.ReasoningEfforts)
	s.Equal("medium", loud.DefaultEffort)
}
