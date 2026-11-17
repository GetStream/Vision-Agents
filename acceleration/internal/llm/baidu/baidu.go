// Package baidu reaches open-weight models on Baidu AI Cloud's Qianfan platform.
//
// Qianfan has two endpoints. qianfan.baidubce.com is the China-domestic one and answers 403
// from Beijing to everyone else, so this is the international one. It names a model without
// its publisher, and suffixes -intl on the models it serves outside China but not on the
// rest: glm-5.3-flash-intl, deepseek-v4-flash-0731, kimi-k2.6.
package baidu

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "baidu"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "QIANFAN_API_KEY",
	BaseURLEnvVar:  "QIANFAN_BASE_URL",
	DefaultBaseURL: "https://api.baiduqianfan.ai/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
