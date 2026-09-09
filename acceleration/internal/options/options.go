// Package options is what a caller asks of a provider beyond a target and a language.
//
// It is its own package, and a leaf, because three layers need the same vocabulary: the
// store keeps these blocks as a router config, routing narrows candidates by what each
// model declared it can express, and a provider reads its own block. One definition means
// what is stored, what is routed and what reaches the provider cannot drift apart.
package options

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
	"time"
)

// Term is one optional thing a request asks a provider for beyond a target and a
// language: diarization, a redacted transcript, a speaking speed.
//
// Providers disagree about all of them. A model declares in config which terms it can
// express, and a request naming one is only routed to a model that declared it, the same
// way a language hint already narrows the candidates. That is what keeps a caller from
// being handed a transcript that was quietly not diarized: nothing here pretends, and a
// term no provider can serve is an error rather than a silence.
type Term string

const (
	DetectLanguage Term = "detect_language"
	Interim        Term = "interim"
	Endpointing    Term = "endpointing"
	Diarize        Term = "diarize"
	MaxSpeakers    Term = "max_speakers"
	Keyterms       Term = "keyterms"
	Format         Term = "format"
	Redact         Term = "redact"
	Events         Term = "events"
	Channels       Term = "channels"
	Words          Term = "words"
	Summary        Term = "summary"
	Entities       Term = "entities"
	// Verbatim and Smart are the two halves of Mode, and they are two terms rather than
	// one because almost nobody can do both. Verbatim is a transcript with the ums and
	// the false starts left in, which three of the six providers can be asked for; Smart
	// is disfluencies removed, grammar tidied and the result formatted, which one can. A
	// single term would have made asking to keep the fillers route to the only provider
	// that can also rewrite them.
	Verbatim        Term = "verbatim"
	Smart           Term = "smart"
	ProfanityFilter Term = "profanity_filter"
	Speed          Term = "speed"
	Volume         Term = "volume"
	Emotion        Term = "emotion"
	Stability      Term = "stability"
	Pronunciations Term = "pronunciations"
	ChunkSchedule  Term = "chunk_schedule"
	Domains        Term = "domains"
	Category       Term = "category"
	Recency        Term = "recency"
	Location       Term = "location"
	Contents       Term = "contents"
	OutputSchema   Term = "output_schema"
)

// Transcription modes. Verbatim keeps what was said; Smart tidies it.
const (
	ModeVerbatim = "verbatim"
	ModeSmart    = "smart"
)

// Claim is a yes, a no, or nobody having said.
//
// Three states rather than two because a vendor that has published nothing about what it
// does with audio is not the same as one that has published a no, and a data policy that
// treated them alike would be worth nothing. Unknown satisfies no requirement.
type Claim string

const (
	ClaimYes     Claim = "yes"
	ClaimNo      Claim = "no"
	ClaimUnknown Claim = "unknown"
)

// Retention is how long what was sent is kept: none, a duration such as 30d or 24h, or an
// admission that this is not known. Unspecified is a provider that stores something and
// does not say for how long; unknown is one that has published nothing either way.
type Retention string

const (
	RetentionNone        Retention = "none"
	RetentionUnspecified Retention = "unspecified"
	RetentionUnknown     Retention = "unknown"
)

// Window is how long this retention keeps something, and whether it says at all.
func (r Retention) Window() (time.Duration, bool) {
	switch r {
	case RetentionNone:
		return 0, true
	case "", RetentionUnspecified, RetentionUnknown:
		return 0, false
	}

	// Days are the unit a retention policy is written in and the one unit
	// time.ParseDuration does not have.
	if digits, ok := strings.CutSuffix(string(r), "d"); ok {
		days, err := strconv.Atoi(digits)
		if err != nil || days < 0 {
			return 0, false
		}
		return time.Duration(days) * 24 * time.Hour, true
	}

	window, err := time.ParseDuration(string(r))
	if err != nil || window < 0 {
		return 0, false
	}
	return window, true
}

// Valid reports whether this is a retention anything can be concluded from.
func (r Retention) Valid() bool {
	if r == "" || r == RetentionUnspecified || r == RetentionUnknown {
		return true
	}
	_, ok := r.Window()
	return ok
}

// DataPolicy is what a caller requires of what a provider does with their audio once it
// has been transcribed. It is a requirement, not a description: a request naming one is
// only routed to a model whose declared handling meets it, and a policy nothing meets is
// an error rather than a transcript that went somewhere it should not have.
type DataPolicy struct {
	// AllowTraining false requires a provider that has said it does not train on what it
	// is sent. Nil asks nothing, and true permits it.
	AllowTraining *bool `json:"allow_training,omitempty"`
	// Retention is the longest a provider may keep this audio. Empty asks nothing.
	Retention Retention `json:"retention,omitempty"`
}

// Asks reports whether this policy requires anything at all.
func (p DataPolicy) Asks() bool {
	return p.AllowTraining != nil || p.Retention != ""
}

// Valid reports whether this policy is one a provider could be measured against.
func (p DataPolicy) Valid() bool {
	if p.Retention == RetentionUnspecified || p.Retention == RetentionUnknown {
		return false
	}
	return p.Retention.Valid()
}

// SatisfiedBy reports whether a provider that handles data this way may serve a request
// that asked for this policy.
func (p DataPolicy) SatisfiedBy(handling DataHandling) bool {
	if p.AllowTraining != nil && !*p.AllowTraining && handling.TrainsOnData != ClaimNo {
		return false
	}
	if p.Retention == "" {
		return true
	}
	// Keeping nothing meets every ceiling, including a request for no retention at all.
	if handling.Retention == RetentionNone {
		return true
	}
	allowed, ok := p.Retention.Window()
	if !ok {
		return false
	}
	kept, ok := handling.Retention.Window()
	if !ok {
		return false
	}
	return kept <= allowed
}

// DataHandling is what one model does with what it is sent, declared per model in the
// router's own config.
//
// It is config rather than fact for the same reason a price is: what a vendor does with
// our audio depends on which plan we are on, which flags we send and whether an admin
// switched zero retention on, none of which a vendor's documentation knows. A deployment
// that changes its contract edits this.
type DataHandling struct {
	// TrainsOnData is whether this model's provider trains on what it is sent.
	TrainsOnData Claim `yaml:"trains_on_data" json:"trains_on_data,omitempty"`
	// Retention is how long the provider keeps it.
	Retention Retention `yaml:"retention" json:"retention,omitempty"`
}

// Declared reports whether anything was said here. An undeclared handling is not a
// permissive one: it satisfies no policy, which is what keeps a new provider from
// answering a request that asked to be kept away from providers like it.
func (h DataHandling) Declared() bool {
	return h.TrainsOnData != "" && h.Retention != ""
}

// Valid reports whether both halves say something a policy can be compared against.
func (h DataHandling) Valid() bool {
	switch h.TrainsOnData {
	case ClaimYes, ClaimNo, ClaimUnknown:
	default:
		return false
	}
	return h.Retention != "" && h.Retention.Valid()
}

// STT is how a caller wants speech transcribed, whether live or from a recording.
//
// The pointers are what tells "say nothing about this" from "turn this off": a config that
// diarizes and a call that asks not to are different requests, and a plain bool cannot
// hold the difference. Fields that mean nothing to one of the two forms are ignored by it
// rather than refused, since a config describes both.
type STT struct {
	Target string `json:"target,omitempty"`
	// Providers is a priority list of where to try, in the order given: a bare provider
	// name, a "provider/model", or a capability shortcut. It is what Target cannot say,
	// which is "this one, and this one after it": a shortcut ranks its members by health,
	// and a caller who has decided that one vendor comes first wants to be asked second
	// only when the first is down. Empty leaves the choice to Target.
	Providers       []string `json:"providers,omitempty"`
	Languages       []string `json:"languages,omitempty"`
	DetectLanguage  *bool    `json:"detect_language,omitempty"`
	SampleRate      *int     `json:"sample_rate,omitempty"`
	Interim         *bool    `json:"interim,omitempty"`
	Endpointing     string   `json:"endpointing,omitempty"`
	SilenceMs       *int     `json:"silence_ms,omitempty"`
	UtteranceEndMs  *int     `json:"utterance_end_ms,omitempty"`
	Diarize         *bool    `json:"diarize,omitempty"`
	MaxSpeakers     *int     `json:"max_speakers,omitempty"`
	Keyterms        []string `json:"keyterms,omitempty"`
	Format          *bool    `json:"format,omitempty"`
	Redact          *bool    `json:"redact,omitempty"`
	Events          *bool    `json:"events,omitempty"`
	Channels        *int     `json:"channels,omitempty"`
	Words           *bool    `json:"words,omitempty"`
	Output          string   `json:"output,omitempty"`
	Summary         *bool    `json:"summary,omitempty"`
	Entities        *bool    `json:"entities,omitempty"`
	ProfanityFilter *bool    `json:"profanity_filter,omitempty"`
	// Mode is verbatim or smart. Empty leaves it to the provider's own default.
	Mode string `json:"mode,omitempty"`
	// DataPolicy is what the caller requires of what happens to the audio afterwards.
	DataPolicy DataPolicy `json:"data_policy,omitempty"`
	// Overwrites are settings for one provider that the vocabulary above has no word for,
	// keyed by provider name. It is the escape hatch, and it is deliberately narrow: the
	// provider it names parses its own block into a typed struct and rejects a field it
	// does not have, so an overwrite is either sent or refused rather than accepted and
	// dropped. An overwrite for a provider that is not a candidate at all is a typo, and
	// reported as one.
	Overwrites map[string]json.RawMessage `json:"overwrites,omitempty"`
}

// Merge returns these options with everything the other one names written over them.
func (o STT) Merge(over STT) STT {
	merged := o
	overwrite(&merged.Target, over.Target)
	overwriteSlice(&merged.Providers, over.Providers)
	overwriteSlice(&merged.Languages, over.Languages)
	overwritePointer(&merged.DetectLanguage, over.DetectLanguage)
	overwritePointer(&merged.SampleRate, over.SampleRate)
	overwritePointer(&merged.Interim, over.Interim)
	overwrite(&merged.Endpointing, over.Endpointing)
	overwritePointer(&merged.SilenceMs, over.SilenceMs)
	overwritePointer(&merged.UtteranceEndMs, over.UtteranceEndMs)
	overwritePointer(&merged.Diarize, over.Diarize)
	overwritePointer(&merged.MaxSpeakers, over.MaxSpeakers)
	overwriteSlice(&merged.Keyterms, over.Keyterms)
	overwritePointer(&merged.Format, over.Format)
	overwritePointer(&merged.Redact, over.Redact)
	overwritePointer(&merged.Events, over.Events)
	overwritePointer(&merged.Channels, over.Channels)
	overwritePointer(&merged.Words, over.Words)
	overwrite(&merged.Output, over.Output)
	overwritePointer(&merged.Summary, over.Summary)
	overwritePointer(&merged.Entities, over.Entities)
	overwritePointer(&merged.ProfanityFilter, over.ProfanityFilter)
	overwrite(&merged.Mode, over.Mode)
	overwritePointer(&merged.DataPolicy.AllowTraining, over.DataPolicy.AllowTraining)
	overwrite((*string)(&merged.DataPolicy.Retention), string(over.DataPolicy.Retention))
	// Per provider rather than wholesale, so a call can change one vendor's setting
	// without restating what the config said about the others.
	if len(over.Overwrites) > 0 {
		merged.Overwrites = make(map[string]json.RawMessage, len(o.Overwrites)+len(over.Overwrites))
		maps.Copy(merged.Overwrites, o.Overwrites)
		maps.Copy(merged.Overwrites, over.Overwrites)
	}
	return merged
}

// Validate reports the first thing about these options a provider could not be asked for.
//
// It is here rather than at each caller because a stored config and a start frame are the
// same options, and a mode nothing recognises should be refused when it is written rather
// than ignored once a socket is open.
func (o STT) Validate() error {
	if o.Mode != "" && o.Mode != ModeVerbatim && o.Mode != ModeSmart {
		return fmt.Errorf("options: mode is %s or %s, not %q", ModeVerbatim, ModeSmart, o.Mode)
	}
	// Smart rewrites what was said, and a word cannot be timed or attributed to a speaker
	// once it may not be the word that was spoken. Refusing beats returning timings that
	// point into a transcript nobody said.
	if o.Mode == ModeSmart {
		if on(o.Diarize) || o.MaxSpeakers != nil {
			return errors.New("options: smart mode cannot diarize, since it rewrites what was said")
		}
		if on(o.Words) {
			return errors.New("options: smart mode has no word timings, since it rewrites what was said")
		}
	}
	if !o.DataPolicy.Valid() {
		return fmt.Errorf("options: retention is none or a duration such as 30d, not %q", o.DataPolicy.Retention)
	}
	for provider, block := range o.Overwrites {
		if provider == "" {
			return errors.New("options: an overwrite has to name the provider it is for")
		}
		if !json.Valid(block) {
			return fmt.Errorf("options: the overwrites for %s are not valid JSON", provider)
		}
	}
	return nil
}

// Terms is what these options ask of a provider. Only what is turned on counts: a caller
// asking not to be diarized rules nothing out, since a model that cannot diarize was
// never going to.
func (o STT) Terms() []Term {
	var asked []Term
	asked = appendIf(asked, DetectLanguage, on(o.DetectLanguage))
	asked = appendIf(asked, Interim, on(o.Interim))
	asked = appendIf(asked, Endpointing, o.Endpointing != "" || o.SilenceMs != nil || o.UtteranceEndMs != nil)
	asked = appendIf(asked, Diarize, on(o.Diarize) || o.MaxSpeakers != nil)
	// Capping the speakers is its own term, because a provider that diarizes and cannot
	// be told when to stop would otherwise return however many it thought it heard.
	asked = appendIf(asked, MaxSpeakers, o.MaxSpeakers != nil)
	asked = appendIf(asked, Keyterms, len(o.Keyterms) > 0)
	asked = appendIf(asked, Format, on(o.Format))
	asked = appendIf(asked, Redact, on(o.Redact))
	asked = appendIf(asked, Events, on(o.Events))
	asked = appendIf(asked, Channels, o.Channels != nil && *o.Channels > 1)
	// Subtitles are not a term: they are words and timings grouped into lines, so asking
	// for them asks the provider for the timings and nothing more.
	asked = appendIf(asked, Words, on(o.Words) || (o.Output != "" && o.Output != "json"))
	asked = appendIf(asked, Summary, on(o.Summary))
	asked = appendIf(asked, Entities, on(o.Entities))
	asked = appendIf(asked, ProfanityFilter, on(o.ProfanityFilter))
	asked = appendIf(asked, Verbatim, o.Mode == ModeVerbatim)
	asked = appendIf(asked, Smart, o.Mode == ModeSmart)
	return asked
}

// TTS is how a caller wants text spoken.
type TTS struct {
	Target         string            `json:"target,omitempty"`
	Voice          string            `json:"voice,omitempty"`
	Languages      []string          `json:"languages,omitempty"`
	Speed          *float64          `json:"speed,omitempty"`
	Volume         *float64          `json:"volume,omitempty"`
	Emotion        string            `json:"emotion,omitempty"`
	Style          string            `json:"style,omitempty"`
	Stability      *float64          `json:"stability,omitempty"`
	Similarity     *float64          `json:"similarity,omitempty"`
	Format         string            `json:"format,omitempty"`
	Pronunciations map[string]string `json:"pronunciations,omitempty"`
	ChunkSchedule  []int             `json:"chunk_schedule,omitempty"`
}

// Merge returns these options with everything the other one names written over them.
func (o TTS) Merge(over TTS) TTS {
	merged := o
	overwrite(&merged.Target, over.Target)
	overwrite(&merged.Voice, over.Voice)
	overwriteSlice(&merged.Languages, over.Languages)
	overwritePointer(&merged.Speed, over.Speed)
	overwritePointer(&merged.Volume, over.Volume)
	overwrite(&merged.Emotion, over.Emotion)
	overwrite(&merged.Style, over.Style)
	overwritePointer(&merged.Stability, over.Stability)
	overwritePointer(&merged.Similarity, over.Similarity)
	overwrite(&merged.Format, over.Format)
	if len(over.Pronunciations) > 0 {
		merged.Pronunciations = maps.Clone(over.Pronunciations)
	}
	overwriteSlice(&merged.ChunkSchedule, over.ChunkSchedule)
	return merged
}

// Terms is what these options ask of a voice.
func (o TTS) Terms() []Term {
	var asked []Term
	asked = appendIf(asked, Speed, o.Speed != nil)
	asked = appendIf(asked, Volume, o.Volume != nil)
	asked = appendIf(asked, Emotion, o.Emotion != "" || o.Style != "")
	asked = appendIf(asked, Stability, o.Stability != nil || o.Similarity != nil)
	asked = appendIf(asked, Format, o.Format != "")
	asked = appendIf(asked, Pronunciations, len(o.Pronunciations) > 0)
	asked = appendIf(asked, ChunkSchedule, len(o.ChunkSchedule) > 0)
	return asked
}

// LLM is how a caller wants a model to answer. The names are the response
// parameters the providers already speak rather than a second vocabulary for the same
// things, so nothing here has to be translated on the way through.
type LLM struct {
	Target          string            `json:"target,omitempty"`
	Instructions    string            `json:"instructions,omitempty"`
	MaxOutputTokens *int              `json:"max_output_tokens,omitempty"`
	Temperature     *float64          `json:"temperature,omitempty"`
	ReasoningEffort string            `json:"reasoning_effort,omitempty"`
	Format          string            `json:"format,omitempty"`
	Verbosity       string            `json:"verbosity,omitempty"`
	ToolChoice      string            `json:"tool_choice,omitempty"`
	Store           *bool             `json:"store,omitempty"`
	PromptCacheKey  string            `json:"prompt_cache_key,omitempty"`
	Metadata        map[string]string `json:"metadata,omitempty"`
}

// Merge returns these options with everything the other one names written over them.
func (o LLM) Merge(over LLM) LLM {
	merged := o
	overwrite(&merged.Target, over.Target)
	overwrite(&merged.Instructions, over.Instructions)
	overwritePointer(&merged.MaxOutputTokens, over.MaxOutputTokens)
	overwritePointer(&merged.Temperature, over.Temperature)
	overwrite(&merged.ReasoningEffort, over.ReasoningEffort)
	overwrite(&merged.Format, over.Format)
	overwrite(&merged.Verbosity, over.Verbosity)
	overwrite(&merged.ToolChoice, over.ToolChoice)
	overwritePointer(&merged.Store, over.Store)
	overwrite(&merged.PromptCacheKey, over.PromptCacheKey)
	if len(over.Metadata) > 0 {
		merged.Metadata = maps.Clone(over.Metadata)
	}
	return merged
}

// Terms is what these options ask of a model. Nothing, as it happens: every provider here
// speaks the whole of the response parameters, and one that cannot honour a parameter says
// so itself when the response is created.
func (o LLM) Terms() []Term { return nil }

// Search is how a caller wants a question answered.
type Search struct {
	Target         string   `json:"target,omitempty"`
	Depth          string   `json:"depth,omitempty"`
	Results        *int     `json:"results,omitempty"`
	IncludeDomains []string `json:"include_domains,omitempty"`
	ExcludeDomains []string `json:"exclude_domains,omitempty"`
	Category       string   `json:"category,omitempty"`
	MaxAgeHours    *int     `json:"max_age_hours,omitempty"`
	Location       string   `json:"location,omitempty"`
	Contents       []string `json:"contents,omitempty"`
	OutputSchema   string   `json:"output_schema,omitempty"`
}

// Merge returns these options with everything the other one names written over them.
func (o Search) Merge(over Search) Search {
	merged := o
	overwrite(&merged.Target, over.Target)
	overwrite(&merged.Depth, over.Depth)
	overwritePointer(&merged.Results, over.Results)
	overwriteSlice(&merged.IncludeDomains, over.IncludeDomains)
	overwriteSlice(&merged.ExcludeDomains, over.ExcludeDomains)
	overwrite(&merged.Category, over.Category)
	overwritePointer(&merged.MaxAgeHours, over.MaxAgeHours)
	overwrite(&merged.Location, over.Location)
	overwriteSlice(&merged.Contents, over.Contents)
	overwrite(&merged.OutputSchema, over.OutputSchema)
	return merged
}

// Terms is what these options ask of a search provider.
//
// Depth is not among them: how much work a search is worth decides which tier answers it,
// so it picks the target rather than being asked of whichever provider was picked.
func (o Search) Terms() []Term {
	var asked []Term
	asked = appendIf(asked, Domains, len(o.IncludeDomains) > 0 || len(o.ExcludeDomains) > 0)
	asked = appendIf(asked, Category, o.Category != "")
	asked = appendIf(asked, Recency, o.MaxAgeHours != nil)
	asked = appendIf(asked, Location, o.Location != "")
	asked = appendIf(asked, Contents, len(o.Contents) > 0)
	asked = appendIf(asked, OutputSchema, o.OutputSchema != "")
	return asked
}

// Route is where a search should go. A named target wins; otherwise the depth decides,
// since asking for a deep answer is asking for the tier that reads pages before it
// answers rather than the one that returns them to be read.
func (o Search) Route() string {
	switch {
	case o.Target != "":
		return o.Target
	case o.Depth == "standard" || o.Depth == "deep":
		return "multilingual-high-accuracy"
	default:
		return "search-fast"
	}
}

// Expressible reports whether every term asked for is among the ones declared. A model
// that declares nothing serves the requests that ask for nothing, which is every request
// that was being made before terms existed.
func Expressible(declared, asked []Term) bool {
	for _, term := range asked {
		if !slices.Contains(declared, term) {
			return false
		}
	}
	return true
}

func on(flag *bool) bool { return flag != nil && *flag }

func appendIf(terms []Term, term Term, asked bool) []Term {
	if !asked {
		return terms
	}
	return append(terms, term)
}

func overwrite(field *string, over string) {
	if over != "" {
		*field = over
	}
}

func overwritePointer[T any](field **T, over *T) {
	if over != nil {
		value := *over
		*field = &value
	}
}

func overwriteSlice[T any](field *[]T, over []T) {
	if len(over) > 0 {
		*field = slices.Clone(over)
	}
}
