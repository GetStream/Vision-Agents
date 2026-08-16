package routing

import (
	"embed"
	"errors"
	"fmt"
	"maps"
	"os"
	"regexp"
	"slices"
	"strings"

	"gopkg.in/yaml.v3"
)

// defaultConfigFS carries the built-in capability config so the router works without an
// external file.
//
//go:embed router.yaml
var defaultConfigFS embed.FS

// Modality is the kind of model a provider serves. Every statistic and health key is
// scoped by it, so the same provider can serve two modalities without their numbers
// mixing.
type Modality string

const (
	STT Modality = "stt"
	TTS Modality = "tts"
	LLM Modality = "llm"
	// Memory and Phone are recorded but not routed: there is one memory store and one
	// vendor per number, so there is nothing to choose between. They are modalities so
	// what they cost shows up in the same reporting as the models.
	Memory Modality = "memory"
	Phone  Modality = "phone"
)

// Tier separates models tuned for latency from models tuned for output quality. It is
// what lets a shortcut ask for the fastest option rather than the best one.
type Tier string

const (
	// LowLatency models answer quickly, which is what a live conversation needs.
	LowLatency Tier = "low-latency"
	// HighQuality models sound or score better but take longer.
	HighQuality Tier = "high-quality"
)

// Price is what a provider charges, in US dollars. Providers bill by different units, so
// a model sets whichever rates apply to it and leaves the rest at zero.
type Price struct {
	// PerMillionChars prices synthesised text.
	PerMillionChars float64 `yaml:"per_million_chars"`
	// PerAudioHour prices audio, whether transcribed or produced.
	PerAudioHour float64 `yaml:"per_audio_hour"`
	// PerMillionInputTokens prices the prompt the model read.
	PerMillionInputTokens float64 `yaml:"per_million_input_tokens"`
	// PerMillionCachedInputTokens prices prompt tokens served from the provider's cache,
	// which is cheaper than reading them afresh.
	PerMillionCachedInputTokens float64 `yaml:"per_million_cached_input_tokens"`
	// PerMillionOutputTokens prices what the model generated.
	PerMillionOutputTokens float64 `yaml:"per_million_output_tokens"`
}

// Usage is what one unit of work consumed. A modality fills in the units it bills by and
// leaves the rest at zero: audio for speech-to-text, characters for text-to-speech,
// tokens for an LLM.
type Usage struct {
	// AudioMs is billable audio, transcribed or produced.
	AudioMs int64
	// Characters is billable text.
	Characters int64
	// InputTokens is the whole prompt, including any part of it that was cached.
	InputTokens int64
	// CachedInputTokens is the part of the prompt the provider served from its cache. It
	// is priced at the cheaper cached rate and excluded from the standard input rate.
	CachedInputTokens int64
	// OutputTokens is everything generated, reasoning included.
	OutputTokens int64
}

// CostMicros returns what one request cost in millionths of a dollar. Micros keep the
// arithmetic in integers, since a fraction of a cent per request is normal.
func (p Price) CostMicros(usage Usage) int64 {
	// Cached tokens are a subset of the prompt, so they are billed once at their own rate
	// rather than twice.
	freshInputTokens := usage.InputTokens - usage.CachedInputTokens
	if freshInputTokens < 0 {
		freshInputTokens = 0
	}

	dollars := p.PerMillionChars*float64(usage.Characters)/1_000_000 +
		p.PerAudioHour*float64(usage.AudioMs)/3_600_000 +
		p.PerMillionInputTokens*float64(freshInputTokens)/1_000_000 +
		p.PerMillionCachedInputTokens*float64(usage.CachedInputTokens)/1_000_000 +
		p.PerMillionOutputTokens*float64(usage.OutputTokens)/1_000_000
	return int64(dollars * 1_000_000)
}

// tagLimit caps how many labels one request may carry, and tagValueLimit how long each
// one may be. Every label becomes a row in the tag rollups, so an unbounded map would
// turn the aggregates back into one row per request.
const (
	tagLimit      = 16
	tagValueLimit = 256
)

// tagKeyPattern is what a label key may look like.
var tagKeyPattern = regexp.MustCompile(`^[a-zA-Z0-9_.-]{1,64}$`)

// Tags are the customer's own labels for a request, for example
// {"customer_id": "123", "project": "moderation", "environment": "dev"}. The keys mean
// whatever the customer wants them to mean; nothing here interprets them.
type Tags map[string]string

// Validate reports the first label the rollups could not carry.
func (t Tags) Validate() error {
	if len(t) > tagLimit {
		return fmt.Errorf("routing: at most %d tags are allowed, got %d", tagLimit, len(t))
	}
	// Sorted so a request with two bad labels always reports the same one.
	for _, key := range slices.Sorted(maps.Keys(t)) {
		if !tagKeyPattern.MatchString(key) {
			return fmt.Errorf("routing: tag key %q must match %s", key, tagKeyPattern)
		}
		if len(t[key]) > tagValueLimit {
			return fmt.Errorf("routing: tag %q is longer than %d characters", key, tagValueLimit)
		}
	}
	return nil
}

// ProviderConfig declares what one provider and model combination can do.
type ProviderConfig struct {
	Provider string `yaml:"provider"`
	Model    string `yaml:"model"`
	// Languages are the ISO codes the model handles.
	Languages []string `yaml:"languages"`
	// Realtime is false for models that only make sense off the live path.
	Realtime bool `yaml:"realtime"`
	// Tier is what the model optimises for. It defaults to low-latency, since a model
	// that says nothing is assumed to be usable in a conversation.
	Tier  Tier  `yaml:"tier"`
	Price Price `yaml:"price"`
}

// Name is the registry key, for example "deepgram/flux-general-en".
func (p ProviderConfig) Name() string { return p.Provider + "/" + p.Model }

// Multilingual reports whether the model covers more than one language.
func (p ProviderConfig) Multilingual() bool { return len(p.Languages) > 1 }

// Speaks reports whether the model covers every requested language.
func (p ProviderConfig) Speaks(languages []string) bool {
	for _, wanted := range languages {
		if !slices.Contains(p.Languages, strings.ToLower(wanted)) {
			return false
		}
	}
	return true
}

// tier returns the declared tier, defaulting to low-latency.
func (p ProviderConfig) tier() Tier {
	if p.Tier == "" {
		return LowLatency
	}
	return p.Tier
}

// Alias is a capability shortcut such as en-low-latency. It describes the requirements a
// provider must meet, so the candidate list follows from the config rather than from a
// hand-maintained list of names.
type Alias struct {
	// Languages every candidate must cover.
	Languages []string `yaml:"languages"`
	// Multilingual requires candidates that handle more than one language.
	Multilingual bool `yaml:"multilingual"`
	// RequireRealtime excludes models that are not suitable for the live path.
	RequireRealtime bool `yaml:"require_realtime"`
	// Tier restricts candidates to one tier. Empty accepts any.
	Tier Tier `yaml:"tier"`
}

// matches reports whether a provider satisfies the alias.
func (a Alias) matches(provider ProviderConfig) bool {
	if a.RequireRealtime && !provider.Realtime {
		return false
	}
	if a.Multilingual && !provider.Multilingual() {
		return false
	}
	if a.Tier != "" && provider.tier() != a.Tier {
		return false
	}
	return provider.Speaks(a.Languages)
}

// ModalityConfig is the capability configuration for one modality.
type ModalityConfig struct {
	Providers []ProviderConfig `yaml:"providers"`
	Aliases   map[string]Alias `yaml:"aliases"`
}

// Provider returns the declaration for a "provider/model" name.
func (c ModalityConfig) Provider(name string) (ProviderConfig, bool) {
	for _, provider := range c.Providers {
		if provider.Name() == name {
			return provider, true
		}
	}
	return ProviderConfig{}, false
}

// Validate reports the first problem that would make routing decisions meaningless.
func (c ModalityConfig) Validate() error {
	if err := c.validate(); err != nil {
		return fmt.Errorf("routing: %w", err)
	}
	return nil
}

func (c ModalityConfig) validate() error {
	if len(c.Providers) == 0 {
		return errors.New("config must declare at least one provider")
	}

	seen := make(map[string]struct{}, len(c.Providers))
	for _, provider := range c.Providers {
		if provider.Provider == "" || provider.Model == "" {
			return errors.New("every provider needs a provider and a model")
		}
		if len(provider.Languages) == 0 {
			return fmt.Errorf("%s declares no languages", provider.Name())
		}
		if provider.Tier != "" && provider.Tier != LowLatency && provider.Tier != HighQuality {
			return fmt.Errorf("%s declares unknown tier %q", provider.Name(), provider.Tier)
		}
		if _, duplicate := seen[provider.Name()]; duplicate {
			return fmt.Errorf("%s is declared twice", provider.Name())
		}
		seen[provider.Name()] = struct{}{}
	}

	for _, name := range slices.Sorted(maps.Keys(c.Aliases)) {
		if !slices.ContainsFunc(c.Providers, c.Aliases[name].matches) {
			return fmt.Errorf("alias %s matches no provider", name)
		}
	}
	return nil
}

// Config is the whole capability configuration, one section per modality. A modality the
// deployment does not serve is simply absent.
type Config map[Modality]ModalityConfig

// DefaultConfig returns the built-in configuration.
func DefaultConfig() (Config, error) {
	raw, err := defaultConfigFS.ReadFile("router.yaml")
	if err != nil {
		return nil, fmt.Errorf("routing: read default config: %w", err)
	}
	return parseConfig(raw)
}

// LoadConfig reads a configuration file, or the built-in default when path is empty.
func LoadConfig(path string) (Config, error) {
	if path == "" {
		return DefaultConfig()
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("routing: read config %s: %w", path, err)
	}
	return parseConfig(raw)
}

func parseConfig(raw []byte) (Config, error) {
	var config Config
	if err := yaml.Unmarshal(raw, &config); err != nil {
		return nil, fmt.Errorf("routing: parse config: %w", err)
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	return config, nil
}

// Validate checks every modality section.
func (c Config) Validate() error {
	if len(c) == 0 {
		return errors.New("routing: config must declare at least one modality")
	}
	// Sorted so a config with two broken sections always reports the same one.
	for _, modality := range slices.Sorted(maps.Keys(c)) {
		if err := c[modality].validate(); err != nil {
			return fmt.Errorf("routing: %s: %w", modality, err)
		}
	}
	return nil
}
