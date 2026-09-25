// Package cloudflare reaches open-weight models on Cloudflare Workers AI.
//
// Cloudflare is the one host here whose endpoint is not the same for everybody: the
// account is part of the path rather than of the token, so there is no default base URL
// and CLOUDFLARE_ACCOUNT_ID is as necessary as the key. Model ids carry an @cf/ prefix,
// as in @cf/zai-org/glm-5.3-flash.
package cloudflare

import (
	"errors"
	"os"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "cloudflare"

// accountEnvVar holds the account the endpoint is addressed by.
const accountEnvVar = "CLOUDFLARE_ACCOUNT_ID"

// Host is where this provider's requests go and what pays for them. It has no default
// base URL because the account has to be in it, so New always works one out.
var Host = openweights.Host{
	Provider:      ProviderName,
	APIKeyEnvVar:  "CLOUDFLARE_API_KEY",
	BaseURLEnvVar: "CLOUDFLARE_BASE_URL",
}

// New builds the provider, reading the API key and the account from the environment when
// Options does not carry them.
func New(options openweights.Options) (*openaicompat.LLM, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(Host.APIKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New(ProviderName + ": " + Host.APIKeyEnvVar + " is required")
	}
	if options.BaseURL == "" {
		options.BaseURL = os.Getenv(Host.BaseURLEnvVar)
	}
	if options.BaseURL == "" {
		account := os.Getenv(accountEnvVar)
		if account == "" {
			return nil, errors.New(ProviderName + ": " + accountEnvVar + " is required")
		}
		options.BaseURL = "https://api.cloudflare.com/client/v4/accounts/" + account + "/ai/v1"
	}
	return Host.New(options)
}
