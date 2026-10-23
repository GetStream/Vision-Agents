package ttsrouter

import (
	"os"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// configuration is every variable a text-to-speech factory here reads its key or endpoint
// from: the names each package under internal/tts passes to keypool.FromEnv and os.Getenv.
var configuration = map[string]string{
	"BASETEN_API_KEY":    "key",
	"CARTESIA_API_KEY":   "key",
	"ELEVENLABS_API_KEY": "key",
	"FISH_API_KEY":       "key",
	"GOOGLE_API_KEY":     "key",
	"INWORLD_API_KEY":    "key",
	"SPEECHIFY_API_KEY":  "key",
	"BREEZE_WS_URL":      "wss://breeze.example.invalid",
	"S2PRO_WS_URL":       "wss://s2pro.example.invalid",
}

// configure sets every variable in configuration but the ones left out, which it clears
// along with their numbered siblings (NAME_1 and on), since keypool reads those too.
func (s *TTSRouterSuite) configure(leftOut ...string) {
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

func (s *TTSRouterSuite) TestEveryVoiceInTheCatalogIsReadyOnceItsConfigIsSet() {
	s.configure()
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	live := DefaultRegistry()
	RegisterMock(live)
	recorded := DefaultRecorderRegistry()

	for _, provider := range config[routing.TTS].Providers {
		s.NoError(live.Ready(provider.Provider, provider.Model), provider.Name())
		s.NoError(recorded.Ready(provider.Provider, provider.Model), provider.Name())
	}
}

func (s *TTSRouterSuite) TestAVoiceWithoutItsEndpointIsListedUnavailableAndRefusedNamingIt() {
	s.configure("BREEZE_WS_URL")
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	router, err := New(Options{Config: config[routing.TTS], Registry: DefaultRegistry()})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)

	available := map[string]bool{}
	for _, candidate := range router.Providers(s.ctx) {
		available[candidate.Config.Name()] = candidate.Health.Available
	}
	s.False(available["breeze/breeze-tts-2"])
	s.True(available["elevenlabs/eleven_flash_v2_5"], "a voice whose key is set stays available")

	_, err = router.Start(s.ctx, Request{CustomerID: "acme", Target: "breeze/breeze-tts-2"})
	s.ErrorContains(err, "breeze: websocket url is required (set BREEZE_WS_URL)")
}
