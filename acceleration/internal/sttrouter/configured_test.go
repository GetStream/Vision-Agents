package sttrouter

import (
	"os"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// configuration is every variable a speech-to-text factory here reads its key or endpoint
// from: the apiKeyEnvVar and endpoint constants of each package under internal/stt, and the
// names deepgram, parakeet and microsoft pass to keypool.FromEnv and os.Getenv.
var configuration = map[string]string{
	"ASSEMBLYAI_API_KEY": "key",
	"AZURE_MAI_API_KEY":  "key",
	"BASETEN_API_KEY":    "key",
	"CARTESIA_API_KEY":   "key",
	"DEEPGRAM_API_KEY":   "key",
	"ELEVENLABS_API_KEY": "key",
	"GOOGLE_API_KEY":     "key",
	"INWORLD_API_KEY":    "key",
	"META_API_KEY":       "key",
	"TOGETHER_API_KEY":   "key",
	"XAI_API_KEY":        "key",
	"AZURE_MAI_ENDPOINT": "https://mai.example.invalid",
	"PARAKEET_WS_URL":    "wss://parakeet.example.invalid",
}

// configure sets every variable in configuration but the ones left out, which it clears
// along with their numbered siblings (NAME_1 and on), since keypool reads those too.
func (s *STTRouterSuite) configure(leftOut ...string) {
	for name, value := range configuration {
		s.T().Setenv(name, value)
	}
	for _, name := range leftOut {
		s.T().Setenv(name, "")
		for _, pair := range os.Environ() {
			if key, _, _ := strings.Cut(pair, "="); strings.HasPrefix(key, name+"_") {
				s.T().Setenv(key, "")
			}
		}
	}
}

func (s *STTRouterSuite) TestEveryModelInTheCatalogIsReadyOnceItsConfigIsSet() {
	s.configure()
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	live := DefaultRegistry()
	RegisterMock(live)
	recorded := DefaultTranscriberRegistry()

	for _, provider := range config[routing.STT].Providers {
		s.NoError(live.Ready(provider.Provider, provider.Model), provider.Name())
		s.NoError(recorded.Ready(provider.Provider, provider.Model), provider.Name())
	}
}

func (s *STTRouterSuite) TestAModelWithoutItsKeyIsListedUnavailableAndRefusedNamingTheKey() {
	s.configure("AZURE_MAI_API_KEY")
	router := s.newRouter()

	available := map[string]bool{}
	for _, candidate := range router.Providers(s.ctx) {
		available[candidate.Config.Name()] = candidate.Health.Available
	}
	s.False(available["microsoft/MAI-Transcribe-2-Streaming"])
	s.True(available["deepgram/flux-general-en"], "a model whose key is set stays available")

	_, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "microsoft/MAI-Transcribe-2-Streaming"})
	s.ErrorContains(err, "microsoft: api key is required (set AZURE_MAI_API_KEY)")
}

func (s *STTRouterSuite) TestABatchModelWithoutItsKeyIsNotReady() {
	s.configure("DEEPGRAM_API_KEY")

	s.ErrorContains(DefaultTranscriberRegistry().Ready("deepgram", "nova-3"),
		"deepgram: api key is required (set DEEPGRAM_API_KEY)")
}
