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

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
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
	// Search is routed like the three above: several providers will answer the same
	// question, they differ in what they cost and how long they take, and which one is
	// worth asking changes with their health.
	Search Modality = "search"
	// LCM is a large classifier model: it answers a question about a piece of text with a
	// typed value and the probability behind it, rather than with prose. It is routed
	// rather than called directly for the same reasons the others are: it is on the live
	// path, so which provider is worth asking changes with their health, and what each
	// judgement cost is worth reporting beside what the conversation cost.
	//
	// It is its own modality rather than a mode of LLM because nothing about it is a
	// language model's shape: there is no stream, no generated text and no token budget,
	// and a caller asks for named questions instead of a prompt.
	LCM Modality = "lcm"
	// STS is speech-to-speech: one native audio model that hears the caller and speaks
	// back, in place of the three above. It is its own modality rather than a flag on a
	// language model because it is served over a different protocol, billed in different
	// units and asked for different things.
	STS Modality = "sts"
	// Image is a picture drawn from a prompt. It is routed like search: several providers
	// will draw the same prompt, they differ in price and in how long a picture takes, and
	// one call is one unit of work with nothing arriving in pieces.
	Image Modality = "image"
	// Memory, Knowledge and Phone are recorded but not routed: there is one memory store,
	// one knowledge base and one vendor per number, so there is nothing to choose
	// between. They are modalities so what they cost shows up in the same reporting as
	// the models.
	Memory    Modality = "memory"
	Knowledge Modality = "knowledge"
	Phone     Modality = "phone"
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

// Benchmark holds the Artificial Analysis numbers for a model. A field the model was
// not measured on is zero.
type Benchmark struct {
	// Elo is the speech arena rating of a voice model.
	Elo int `yaml:"elo"`
	// CharactersPerSecond is how fast a voice model synthesises on the vendor's API.
	CharactersPerSecond float64 `yaml:"characters_per_second"`
	// WordErrorRate is the streaming AA-WER of a transcriber, from 0 to 1.
	WordErrorRate float64 `yaml:"word_error_rate"`
	// LatencyMs is how long a transcriber takes to its final transcript after speech ends.
	LatencyMs int `yaml:"latency_ms"`
	// SearchIndex is the Artificial Analysis Search Index of a search provider, 0 to 100.
	SearchIndex int `yaml:"search_index"`
	// CostPerTask is what one task of that benchmark cost in US dollars, the searches and
	// the answering model's tokens together.
	CostPerTask float64 `yaml:"cost_per_task"`
}

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
	// PerThousandRequests prices work a provider bills by the call rather than by what it
	// read or wrote, which is how a search API charges. The rate is not applied here,
	// because nothing in Usage counts calls: a caller billed this way sets Stat.CostMicros
	// from it, the way a phone number's monthly charge is set outright.
	PerThousandRequests float64 `yaml:"per_thousand_requests"`
	// PerImage prices each picture a model drew.
	PerImage float64 `yaml:"per_image"`
	// PerMegapixel prices the pixels a model drew, a million to the megapixel, for a
	// vendor whose bill grows with the size of the picture rather than the count.
	PerMegapixel float64 `yaml:"per_megapixel"`
}

// RequestMicros is what one call to a provider billed by the call costs, in millionths of
// a dollar.
func (p Price) RequestMicros() int64 {
	return int64(p.PerThousandRequests * 1_000)
}

// Usage is what one unit of work consumed. A modality fills in the units it bills by and
// leaves the rest at zero: audio for speech-to-text, characters for text-to-speech,
// tokens for an LLM, pictures for image generation.
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
	// Images is how many pictures were drawn.
	Images int64
	// Pixels is how many pixels those pictures hold between them, counted on what came
	// back rather than what was asked for. It prices a model billed by the megapixel and
	// is not stored: the picture count is what a customer reads.
	Pixels int64
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
		p.PerMillionOutputTokens*float64(usage.OutputTokens)/1_000_000 +
		p.PerImage*float64(usage.Images) +
		p.PerMegapixel*float64(usage.Pixels)/1_000_000
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
	// Thinking enables provider-specific reasoning for this concrete model.
	Thinking bool `yaml:"thinking"`
	// ReasoningEffort selects the provider-specific reasoning budget.
	ReasoningEffort string `yaml:"reasoning_effort"`
	// Description is one sentence for someone choosing a model: what it is good at, and
	// what it costs them in speed or money to get it.
	Description string `yaml:"description"`
	// Languages are the ISO codes the model handles.
	Languages []string `yaml:"languages"`
	// Realtime is false for models that only make sense off the live path.
	Realtime bool `yaml:"realtime"`
	// Tier is what the model optimises for. It defaults to low-latency, since a model
	// that says nothing is assumed to be usable in a conversation.
	Tier Tier `yaml:"tier"`
	// Terms are the optional terms this model can express - diarization, a speaking
	// speed, a domain filter. A request asking for one is only routed here if it is
	// declared, so a term is either honoured or the request is refused.
	Terms []options.Term `yaml:"supports"`
	// DataPolicy is what happens to what this model is sent: whether the provider trains
	// on it and how long they keep it. Like the price, it is what this deployment's
	// contract and account settings amount to rather than what a vendor's documentation
	// says, because a retention window depends on which plan we are on and whether an
	// admin switched zero retention on. A request naming a data policy is only routed to
	// a model that meets it.
	DataPolicy options.DataHandling `yaml:"data_policy"`
	Price      Price                `yaml:"price"`
	// Benchmark is what Artificial Analysis measured, for someone choosing a model.
	// Routing never reads it.
	Benchmark Benchmark `yaml:"benchmark"`
	// InputModalities are extra input kinds this model accepts, e.g. "image". Empty
	// means text only, and a request carrying anything else is not routed here.
	InputModalities []string `yaml:"input_modalities"`
}

// Supports reports whether this model can express every term a request named.
func (p ProviderConfig) Supports(terms []options.Term) bool {
	return options.Expressible(p.Terms, terms)
}

// Permits reports whether this model may serve a request asking for that data policy. A
// model that has declared nothing permits nothing, since the alternative is answering a
// request that asked to stay away from providers whose policy is unknown.
func (p ProviderConfig) Permits(policy options.DataPolicy) bool {
	return policy.SatisfiedBy(p.DataPolicy)
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

// Sees reports whether the model accepts every requested input kind besides text.
func (p ProviderConfig) Sees(modalities []string) bool {
	for _, wanted := range modalities {
		if wanted == "" || wanted == "text" {
			continue
		}
		if !slices.Contains(p.InputModalities, wanted) {
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
	// Title is what the shortcut is called where someone picks one, and Description says
	// what it is for. A shortcut with no title is plumbing, such as the one the flow
	// controller runs on, and is not offered as a choice.
	Title       string `yaml:"title"`
	Description string `yaml:"description"`

	RequireInputModalities []string `yaml:"require_input_modalities"`
	// Only names the candidates outright, for the shortcut whose members have nothing
	// declarable in common. It is the exception to everything above: a name here is a
	// judgement about which models are wanted rather than a fact about what they can do,
	// so it goes stale in a way the requirements do not, and a shortcut that can say what
	// it means in requirements should. Empty leaves the choice to them.
	Only []string `yaml:"only"`
	// Languages every candidate must cover.
	Languages []string `yaml:"languages"`
	// Multilingual requires candidates that handle more than one language.
	Multilingual bool `yaml:"multilingual"`
	// RequireRealtime excludes models that are not suitable for the live path.
	RequireRealtime bool `yaml:"require_realtime"`
	// RequireRecorded is the other way round: it excludes the live models, so a whole
	// recording routes to the batch endpoint that is cheaper and more accurate than the
	// same vendor's streaming one rather than being streamed at a socket.
	RequireRecorded bool `yaml:"require_recorded"`
	// Tier restricts candidates to one tier. Empty accepts any.
	Tier Tier `yaml:"tier"`
	// Prefer is a "provider/model" that goes ahead of the ranking for as long as it is
	// available. It is how a shortcut names the model it means while keeping the rest of
	// the tier as failover; empty leaves the order to health alone.
	Prefer string `yaml:"prefer"`
}

// matches reports whether a provider satisfies the alias.
func (a Alias) matches(provider ProviderConfig) bool {
	if len(a.Only) > 0 && !slices.Contains(a.Only, provider.Name()) {
		return false
	}
	if a.RequireRealtime && !provider.Realtime {
		return false
	}
	if a.RequireRecorded && provider.Realtime {
		return false
	}
	if a.Multilingual && !provider.Multilingual() {
		return false
	}
	if a.Tier != "" && provider.tier() != a.Tier {
		return false
	}
	return provider.Speaks(a.Languages) && provider.Sees(a.RequireInputModalities)
}

// ModalityConfig is the capability configuration for one modality.
type ModalityConfig struct {
	Providers []ProviderConfig `yaml:"providers"`
	Aliases   map[string]Alias `yaml:"aliases"`
	// aliasOrder is the order the aliases were written in, which a map forgets.
	aliasOrder []string
}

// UnmarshalYAML decodes the section and keeps the order its aliases were written in.
func (c *ModalityConfig) UnmarshalYAML(node *yaml.Node) error {
	type plain ModalityConfig
	if err := node.Decode((*plain)(c)); err != nil {
		return err
	}
	for i := 0; i+1 < len(node.Content); i += 2 {
		if node.Content[i].Value != "aliases" {
			continue
		}
		aliases := node.Content[i+1].Content
		for j := 0; j < len(aliases); j += 2 {
			c.aliasOrder = append(c.aliasOrder, aliases[j].Value)
		}
	}
	return nil
}

// Offered returns the shortcuts that carry a title, in the order the config wrote them.
// Those are the ones someone picking a model is shown.
func (c ModalityConfig) Offered() []string {
	order := c.aliasOrder
	// A config built in code has no written order, so it is offered alphabetically.
	if len(order) == 0 {
		order = slices.Sorted(maps.Keys(c.Aliases))
	}
	var offered []string
	for _, name := range order {
		if c.Aliases[name].Title != "" {
			offered = append(offered, name)
		}
	}
	return offered
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

// Declares reports whether a vendor is configured here at all, whichever of their models
// it is. It is what tells a priority list naming a vendor from one with a typo in it.
func (c ModalityConfig) Declares(vendor string) bool {
	return slices.ContainsFunc(c.Providers, func(provider ProviderConfig) bool {
		return provider.Provider == vendor
	})
}

// Names reports whether this is something a priority list may hold: a vendor, one of
// their models, or a capability shortcut.
func (c ModalityConfig) Names(target string) bool {
	if _, ok := c.Aliases[target]; ok {
		return true
	}
	if _, ok := c.Provider(target); ok {
		return true
	}
	return c.Declares(target)
}

// Meets reports whether any model here could serve a request asking for that data policy.
// A policy nothing meets is worth refusing when it is written down, since every request
// made under it afterwards would be refused anyway.
func (c ModalityConfig) Meets(policy options.DataPolicy) bool {
	if !policy.Asks() {
		return true
	}
	return slices.ContainsFunc(c.Providers, func(provider ProviderConfig) bool {
		return provider.Permits(policy)
	})
}

// Expresses reports whether any model here can express every one of those terms, which is
// the same question serving() asks of a request's candidates.
func (c ModalityConfig) Expresses(terms []options.Term) bool {
	if len(terms) == 0 {
		return true
	}
	return slices.ContainsFunc(c.Providers, func(provider ProviderConfig) bool {
		return provider.Supports(terms)
	})
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
		if provider.DataPolicy != (options.DataHandling{}) && !provider.DataPolicy.Valid() {
			return fmt.Errorf("%s declares a data policy of %q and %q, which a request cannot be compared against",
				provider.Name(), provider.DataPolicy.TrainsOnData, provider.DataPolicy.Retention)
		}
		if _, duplicate := seen[provider.Name()]; duplicate {
			return fmt.Errorf("%s is declared twice", provider.Name())
		}
		seen[provider.Name()] = struct{}{}
	}

	for _, name := range slices.Sorted(maps.Keys(c.Aliases)) {
		alias := c.Aliases[name]
		if !slices.ContainsFunc(c.Providers, alias.matches) {
			return fmt.Errorf("alias %s matches no provider", name)
		}
		// A name nobody answers to would quietly shrink the shortcut instead of failing,
		// which is the one way naming candidates is worse than describing them.
		for _, only := range alias.Only {
			named, ok := c.Provider(only)
			if !ok {
				return fmt.Errorf("alias %s names %s, which is not declared", name, only)
			}
			if !alias.matches(named) {
				return fmt.Errorf("alias %s names %s, which does not meet its own requirements", name, only)
			}
		}
		if alias.Prefer == "" {
			continue
		}
		// A pin that no candidate answers to would be silently ignored, which is the one
		// way this config can lie about where a request goes.
		preferred, ok := c.Provider(alias.Prefer)
		if !ok {
			return fmt.Errorf("alias %s prefers %s, which is not declared", name, alias.Prefer)
		}
		if !alias.matches(preferred) {
			return fmt.Errorf("alias %s prefers %s, which does not meet its own requirements", name, alias.Prefer)
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
		// Speech in, speech out and the models that do both are the modalities a data
		// policy can be asked of, so they are the ones where every model has to have said
		// what happens to what it is sent. An undeclared model would not answer such a
		// request anyway; failing here is how that is found out when the provider is
		// added rather than when a customer asks.
		if modality != STT && modality != TTS && modality != STS {
			continue
		}
		for _, provider := range c[modality].Providers {
			if !provider.DataPolicy.Declared() {
				return fmt.Errorf("routing: %s: %s declares no data_policy", modality, provider.Name())
			}
		}
	}
	return nil
}
