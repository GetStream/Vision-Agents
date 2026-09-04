// Package options is what a caller asks of a provider beyond a target and a language.
//
// It is its own package, and a leaf, because three layers need the same vocabulary: the
// store keeps these blocks as a router config, routing narrows candidates by what each
// model declared it can express, and a provider reads its own block. One definition means
// what is stored, what is routed and what reaches the provider cannot drift apart.
package options

import (
	"maps"
	"slices"
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

// STT is how a caller wants speech transcribed, whether live or from a recording.
//
// The pointers are what tells "say nothing about this" from "turn this off": a config that
// diarizes and a call that asks not to are different requests, and a plain bool cannot
// hold the difference. Fields that mean nothing to one of the two forms are ignored by it
// rather than refused, since a config describes both.
type STT struct {
	Target         string   `json:"target,omitempty"`
	Languages      []string `json:"languages,omitempty"`
	DetectLanguage *bool    `json:"detect_language,omitempty"`
	SampleRate     *int     `json:"sample_rate,omitempty"`
	Interim        *bool    `json:"interim,omitempty"`
	Endpointing    string   `json:"endpointing,omitempty"`
	SilenceMs      *int     `json:"silence_ms,omitempty"`
	UtteranceEndMs *int     `json:"utterance_end_ms,omitempty"`
	Diarize        *bool    `json:"diarize,omitempty"`
	MaxSpeakers    *int     `json:"max_speakers,omitempty"`
	Keyterms       []string `json:"keyterms,omitempty"`
	Format         *bool    `json:"format,omitempty"`
	Redact         *bool    `json:"redact,omitempty"`
	Events         *bool    `json:"events,omitempty"`
	Channels       *int     `json:"channels,omitempty"`
	Words          *bool    `json:"words,omitempty"`
	Output         string   `json:"output,omitempty"`
	Summary        *bool    `json:"summary,omitempty"`
	Entities       *bool    `json:"entities,omitempty"`
}

// Merge returns these options with everything the other one names written over them.
func (o STT) Merge(over STT) STT {
	merged := o
	overwrite(&merged.Target, over.Target)
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
	return merged
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
