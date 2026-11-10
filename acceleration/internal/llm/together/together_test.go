package together

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type TogetherSuite struct {
	hosttest.Unit
}

func TestTogetherSuite(t *testing.T) {
	suite.Run(t, &TogetherSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "MiniMaxAI/MiniMax-M3",
		ThinkingModel: "Qwen/Qwen3.8-2.4T-A95B",
	}})
}

func TestTheCacheKeyIsSentAsPromptCacheKey(t *testing.T) {
	build := func(baseURL string) (*openaicompat.LLM, error) {
		return New(openweights.Options{APIKey: "k", Model: "zai-org/GLM-5.3", BaseURL: baseURL})
	}
	provider, err := build("http://unused")
	require.NoError(t, err)
	require.True(t, provider.Capabilities().PromptCacheKey)

	sent := hosttest.Sent(t, build, llm.ResponseParams{
		Input:          []llm.Message{{Role: llm.User, Content: "hi"}},
		PromptCacheKey: "agent-config-7",
	})
	require.Equal(t, "agent-config-7", sent["prompt_cache_key"])

	sent = hosttest.Sent(t, build, llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Content: "hi"}}})
	require.NotContains(t, sent, "prompt_cache_key", "no key, no field")
}
