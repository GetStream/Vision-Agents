package api

import (
	"encoding/json"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/danielgtaylor/huma/v2"
)

// Reading the wire's option blocks into the router's own, and writing them back out.
//
// The two shapes say the same things and differ in how they say nothing: the generated
// types make every field a pointer because the spec marks them all optional, while the
// router keeps a pointer only where "leave this alone" and "turn this off" are different
// requests. A speed of nought is not "the voice's own speed", so speed is a pointer both
// sides; a target of "" is nothing anybody meant, so a target is a plain string here.

func sttOptionsOf(sent *SttOptions) options.STT {
	if sent == nil {
		return options.STT{}
	}
	held := options.STT{
		Target:          value(sent.Target),
		Providers:       value(sent.Providers),
		Languages:       value(sent.Languages),
		DetectLanguage:  sent.DetectLanguage,
		SampleRate:      sent.SampleRate,
		Interim:         sent.Interim,
		Endpointing:     string(value(sent.Endpointing)),
		SilenceMs:       sent.SilenceMs,
		UtteranceEndMs:  sent.UtteranceEndMs,
		EagerEndOfTurn:  sent.EagerEndOfTurn,
		Diarize:         sent.Diarize,
		MaxSpeakers:     sent.MaxSpeakers,
		Keyterms:        value(sent.Keyterms),
		Format:          sent.Format,
		Redact:          sent.Redact,
		Events:          sent.Events,
		Channels:        sent.Channels,
		Words:           sent.Words,
		Output:          string(value(sent.Output)),
		Summary:         sent.Summary,
		Entities:        sent.Entities,
		ProfanityFilter: sent.ProfanityFilter,
		Mode:            string(value(sent.Mode)),
	}
	held.DataPolicy = dataPolicyOf(sent.DataPolicy)
	held.Overwrites = overwritesOf(sent.Overwrites)
	return held
}

func sttOptionsFor(held options.STT) *SttOptions {
	sent := &SttOptions{
		Target:          optional(held.Target),
		Providers:       list(held.Providers),
		Languages:       list(held.Languages),
		DetectLanguage:  held.DetectLanguage,
		SampleRate:      held.SampleRate,
		Interim:         held.Interim,
		SilenceMs:       held.SilenceMs,
		UtteranceEndMs:  held.UtteranceEndMs,
		EagerEndOfTurn:  held.EagerEndOfTurn,
		Diarize:         held.Diarize,
		MaxSpeakers:     held.MaxSpeakers,
		Keyterms:        list(held.Keyterms),
		Format:          held.Format,
		Redact:          held.Redact,
		Events:          held.Events,
		Channels:        held.Channels,
		Words:           held.Words,
		Summary:         held.Summary,
		Entities:        held.Entities,
		ProfanityFilter: held.ProfanityFilter,
	}
	if held.Endpointing != "" {
		endpointing := Endpointing(held.Endpointing)
		sent.Endpointing = &endpointing
	}
	if held.Output != "" {
		output := TranscriptFormat(held.Output)
		sent.Output = &output
	}
	if held.Mode != "" {
		mode := TranscriptionMode(held.Mode)
		sent.Mode = &mode
	}
	sent.DataPolicy = dataPolicyFor(held.DataPolicy)
	sent.Overwrites = overwritesFor(held.Overwrites)
	return sent
}

func ttsOptionsOf(sent *TtsOptions) options.TTS {
	if sent == nil {
		return options.TTS{}
	}
	return options.TTS{
		Target:         value(sent.Target),
		Providers:      value(sent.Providers),
		Voice:          value(sent.Voice),
		Languages:      value(sent.Languages),
		Speed:          wider(sent.Speed),
		Volume:         wider(sent.Volume),
		Emotion:        value(sent.Emotion),
		Style:          value(sent.Style),
		Stability:      wider(sent.Stability),
		Similarity:     wider(sent.Similarity),
		Format:         value(sent.Format),
		Pronunciations: value(sent.Pronunciations),
		ChunkSchedule:  value(sent.ChunkSchedule),
		DataPolicy:     dataPolicyOf(sent.DataPolicy),
		Overwrites:     overwritesOf(sent.Overwrites),
	}
}

func ttsOptionsFor(held options.TTS) *TtsOptions {
	sent := &TtsOptions{
		Target:        optional(held.Target),
		Providers:     list(held.Providers),
		Voice:         optional(held.Voice),
		Languages:     list(held.Languages),
		Speed:         narrower(held.Speed),
		Volume:        narrower(held.Volume),
		Emotion:       optional(held.Emotion),
		Style:         optional(held.Style),
		Stability:     narrower(held.Stability),
		Similarity:    narrower(held.Similarity),
		Format:        optional(held.Format),
		ChunkSchedule: list(held.ChunkSchedule),
		DataPolicy:    dataPolicyFor(held.DataPolicy),
		Overwrites:    overwritesFor(held.Overwrites),
	}
	if len(held.Pronunciations) > 0 {
		pronunciations := held.Pronunciations
		sent.Pronunciations = &pronunciations
	}
	return sent
}

func llmOptionsOf(sent *LlmOptions) options.LLM {
	if sent == nil {
		return options.LLM{}
	}
	return options.LLM{
		Target:          value(sent.Target),
		Providers:       value(sent.Providers),
		MaxOutputTokens: sent.MaxOutputTokens,
		Temperature:     wider(sent.Temperature),
		ReasoningEffort: string(value(sent.ReasoningEffort)),
		Format:          string(value(sent.Format)),
		Verbosity:       string(value(sent.Verbosity)),
		ToolChoice:      value(sent.ToolChoice),
		Store:           sent.Store,
		PromptCacheKey:  value(sent.PromptCacheKey),
		Metadata:        value(sent.Metadata),
	}
}

func llmOptionsFor(held options.LLM) *LlmOptions {
	sent := &LlmOptions{
		Target:          optional(held.Target),
		Providers:       list(held.Providers),
		MaxOutputTokens: held.MaxOutputTokens,
		Temperature:     narrower(held.Temperature),
		ToolChoice:      optional(held.ToolChoice),
		Store:           held.Store,
		PromptCacheKey:  optional(held.PromptCacheKey),
	}
	if held.ReasoningEffort != "" {
		effort := LlmOptionsReasoningEffort(held.ReasoningEffort)
		sent.ReasoningEffort = &effort
	}
	if held.Format != "" {
		format := LlmOptionsFormat(held.Format)
		sent.Format = &format
	}
	if held.Verbosity != "" {
		verbosity := LlmOptionsVerbosity(held.Verbosity)
		sent.Verbosity = &verbosity
	}
	if len(held.Metadata) > 0 {
		metadata := held.Metadata
		sent.Metadata = &metadata
	}
	return sent
}

func stsOptionsOf(sent *StsOptions) options.STS {
	if sent == nil {
		return options.STS{}
	}
	return options.STS{
		Target:            value(sent.Target),
		Providers:         value(sent.Providers),
		Instructions:      value(sent.Instructions),
		Voice:             value(sent.Voice),
		Languages:         value(sent.Languages),
		TurnDetection:     string(value(sent.TurnDetection)),
		SilenceMs:         sent.SilenceMs,
		PrefixPaddingMs:   sent.PrefixPaddingMs,
		InterruptResponse: sent.InterruptResponse,
		InputTranscript:   sent.InputTranscript,
		OutputTranscript:  sent.OutputTranscript,
		Tools:             sent.Tools,
		Text:              sent.Text,
		Images:            sent.Images,
		DataPolicy:        dataPolicyOf(sent.DataPolicy),
		Overwrites:        overwritesOf(sent.Overwrites),
	}
}

func stsOptionsFor(held options.STS) *StsOptions {
	sent := &StsOptions{
		Target:            optional(held.Target),
		Providers:         list(held.Providers),
		Instructions:      optional(held.Instructions),
		Voice:             optional(held.Voice),
		Languages:         list(held.Languages),
		SilenceMs:         held.SilenceMs,
		PrefixPaddingMs:   held.PrefixPaddingMs,
		InterruptResponse: held.InterruptResponse,
		InputTranscript:   held.InputTranscript,
		OutputTranscript:  held.OutputTranscript,
		Tools:             held.Tools,
		Text:              held.Text,
		Images:            held.Images,
		DataPolicy:        dataPolicyFor(held.DataPolicy),
		Overwrites:        overwritesFor(held.Overwrites),
	}
	if held.TurnDetection != "" {
		turns := StsOptionsTurnDetection(held.TurnDetection)
		sent.TurnDetection = &turns
	}
	return sent
}

func searchOptionsOf(sent *SearchOptions) options.Search {
	if sent == nil {
		return options.Search{}
	}
	held := options.Search{
		Target:         value(sent.Target),
		Providers:      value(sent.Providers),
		Depth:          string(value(sent.Depth)),
		Results:        sent.Results,
		IncludeDomains: value(sent.IncludeDomains),
		ExcludeDomains: value(sent.ExcludeDomains),
		Category:       value(sent.Category),
		MaxAgeHours:    sent.MaxAgeHours,
		Location:       value(sent.Location),
	}
	for _, want := range value(sent.Contents) {
		held.Contents = append(held.Contents, string(want))
	}
	// The schema is carried as text rather than as a decoded object because nothing here
	// reads inside it: it is handed to whichever provider was asked for it.
	if sent.OutputSchema != nil {
		if encoded, err := json.Marshal(*sent.OutputSchema); err == nil {
			held.OutputSchema = string(encoded)
		}
	}
	return held
}

func searchOptionsFor(held options.Search) *SearchOptions {
	sent := &SearchOptions{
		Target:         optional(held.Target),
		Providers:      list(held.Providers),
		Results:        held.Results,
		IncludeDomains: list(held.IncludeDomains),
		ExcludeDomains: list(held.ExcludeDomains),
		Category:       optional(held.Category),
		MaxAgeHours:    held.MaxAgeHours,
		Location:       optional(held.Location),
	}
	if held.Depth != "" {
		depth := SearchDepth(held.Depth)
		sent.Depth = &depth
	}
	if len(held.Contents) > 0 {
		contents := make([]SearchOptionsContents, 0, len(held.Contents))
		for _, want := range held.Contents {
			contents = append(contents, SearchOptionsContents(want))
		}
		sent.Contents = &contents
	}
	if held.OutputSchema != "" {
		var schema map[string]any
		if err := json.Unmarshal([]byte(held.OutputSchema), &schema); err == nil {
			sent.OutputSchema = &schema
		}
	}
	return sent
}

// list carries a slice only when there is one, which is how an unset field stays unset on
// the way back out.
func list[T any](items []T) *[]T {
	if len(items) == 0 {
		return nil
	}
	return &items
}

// dataPolicyOf and dataPolicyFor move a policy between the two shapes. Both modalities
// that can be asked for one say it the same way, so they read it the same way too.
func dataPolicyOf(sent *DataPolicy) options.DataPolicy {
	if sent == nil {
		return options.DataPolicy{}
	}
	return options.DataPolicy{
		AllowTraining: sent.AllowTraining,
		Retention:     options.Retention(value(sent.Retention)),
	}
}

func dataPolicyFor(held options.DataPolicy) *DataPolicy {
	if !held.Asks() {
		return nil
	}
	return &DataPolicy{
		AllowTraining: held.AllowTraining,
		Retention:     optional(string(held.Retention)),
	}
}

// overwritesOf carries an overwrite as text rather than as a decoded object because
// nothing here reads inside it: it is handed to whichever provider it names, which is the
// only thing that knows what its own settings are called.
func overwritesOf(sent *map[string]any) map[string]json.RawMessage {
	blocks := value(sent)
	if len(blocks) == 0 {
		return nil
	}

	held := make(map[string]json.RawMessage, len(blocks))
	for provider, block := range blocks {
		encoded, err := json.Marshal(block)
		if err != nil {
			continue
		}
		held[provider] = encoded
	}
	return held
}

func overwritesFor(held map[string]json.RawMessage) *map[string]any {
	if len(held) == 0 {
		return nil
	}

	overwrites := make(map[string]any, len(held))
	for provider, block := range held {
		var decoded any
		if err := json.Unmarshal(block, &decoded); err != nil {
			continue
		}
		overwrites[provider] = decoded
	}
	return &overwrites
}

// wider and narrower move between the float32 the spec's "format: float" generates and
// the float64 everything else here is written in.
func wider(value *float32) *float64 {
	if value == nil {
		return nil
	}
	widened := float64(*value)
	return &widened
}

func narrower(value *float64) *float32 {
	if value == nil {
		return nil
	}
	narrowed := float32(*value)
	return &narrowed
}

// Endpointing What decides a turn is over: a long enough pause, or a model reading the words and judging the sentence finished.
type Endpointing string

// Defines values for Endpointing.
const (
	EndpointingSemantic Endpointing = "semantic"
	EndpointingSilence  Endpointing = "silence"
)

// Valid indicates whether the value is a known member of the Endpointing enum.
func (e Endpointing) Valid() bool {
	switch e {
	case EndpointingSemantic:
		return true
	case EndpointingSilence:
		return true
	default:
		return false
	}
}

func (Endpointing) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "Endpointing", "What decides a turn is over: a long enough pause, or a model reading the words and judging the sentence finished.", "silence", "semantic")
}

// LlmOptionsFormat is the LlmOptionsFormat schema.
type LlmOptionsFormat string

// Defines values for LlmOptionsFormat.
const (
	LlmOptionsFormatJsonObject LlmOptionsFormat = "json_object"
	LlmOptionsFormatText       LlmOptionsFormat = "text"
)

// Valid indicates whether the value is a known member of the LlmOptionsFormat enum.
func (e LlmOptionsFormat) Valid() bool {
	switch e {
	case LlmOptionsFormatJsonObject:
		return true
	case LlmOptionsFormatText:
		return true
	default:
		return false
	}
}

// LlmOptionsReasoningEffort is the LlmOptionsReasoningEffort schema.
type LlmOptionsReasoningEffort string

// Defines values for LlmOptionsReasoningEffort.
const (
	LlmOptionsReasoningEffortHigh    LlmOptionsReasoningEffort = "high"
	LlmOptionsReasoningEffortLow     LlmOptionsReasoningEffort = "low"
	LlmOptionsReasoningEffortMedium  LlmOptionsReasoningEffort = "medium"
	LlmOptionsReasoningEffortMinimal LlmOptionsReasoningEffort = "minimal"
)

// Valid indicates whether the value is a known member of the LlmOptionsReasoningEffort enum.
func (e LlmOptionsReasoningEffort) Valid() bool {
	switch e {
	case LlmOptionsReasoningEffortHigh:
		return true
	case LlmOptionsReasoningEffortLow:
		return true
	case LlmOptionsReasoningEffortMedium:
		return true
	case LlmOptionsReasoningEffortMinimal:
		return true
	default:
		return false
	}
}

// LlmOptionsVerbosity is the LlmOptionsVerbosity schema.
type LlmOptionsVerbosity string

// Defines values for LlmOptionsVerbosity.
const (
	LlmOptionsVerbosityHigh   LlmOptionsVerbosity = "high"
	LlmOptionsVerbosityLow    LlmOptionsVerbosity = "low"
	LlmOptionsVerbosityMedium LlmOptionsVerbosity = "medium"
)

// Valid indicates whether the value is a known member of the LlmOptionsVerbosity enum.
func (e LlmOptionsVerbosity) Valid() bool {
	switch e {
	case LlmOptionsVerbosityHigh:
		return true
	case LlmOptionsVerbosityLow:
		return true
	case LlmOptionsVerbosityMedium:
		return true
	default:
		return false
	}
}

// SearchDepth How much work a search is worth. instant answers from the index in a few hundred milliseconds; deep crawls and reasons over what it finds and can take tens of seconds. Providers offer different ladders, so each one maps these four onto its own.
type SearchDepth string

// Defines values for SearchDepth.
const (
	Deep     SearchDepth = "deep"
	Fast     SearchDepth = "fast"
	Instant  SearchDepth = "instant"
	Standard SearchDepth = "standard"
)

// Valid indicates whether the value is a known member of the SearchDepth enum.
func (e SearchDepth) Valid() bool {
	switch e {
	case Deep:
		return true
	case Fast:
		return true
	case Instant:
		return true
	case Standard:
		return true
	default:
		return false
	}
}

func (SearchDepth) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "SearchDepth", "How much work a search is worth. instant answers from the index in a few hundred milliseconds; deep crawls and reasons over what it finds and can take tens of seconds. Providers offer different ladders, so each one maps these four onto its own.", "instant", "fast", "standard", "deep")
}

// SearchOptions How this config finds out today's answers.
type SearchOptions struct {
	Category       *string                  `json:"category,omitempty" doc:"The kind of source to prefer - news, papers, company, github - for the providers that classify their index."`
	Contents       *[]SearchOptionsContents `json:"contents,omitempty" doc:"What to return alongside each hit."`
	Depth          *SearchDepth             `json:"depth,omitempty"`
	ExcludeDomains *[]string                `json:"exclude_domains,omitempty"`
	IncludeDomains *[]string                `json:"include_domains,omitempty" doc:"Only answer from these domains."`
	Location       *string                  `json:"location,omitempty" doc:"Country or region to answer from, for queries whose answer depends on where."`
	MaxAgeHours    *int                     `json:"max_age_hours,omitempty" doc:"How stale a cached page may be. Zero forces a live crawl, which is slower and costs more." minimum:"0"`
	OutputSchema   *map[string]interface{}  `json:"output_schema,omitempty" doc:"A JSON schema the answer must fit, for the providers that can be asked to structure what they found."`
	Providers      *[]string                `json:"providers,omitempty" doc:"A priority list of where to try, in the order given, which wins over target and depth when it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, expanded where it stands. A search that fails is asked of the next entry that will have it."`
	Results        *int                     `json:"results,omitempty" doc:"How many hits to return." minimum:"1"`
	Target         *string                  `json:"target,omitempty" doc:"A provider/model or a capability shortcut." example:"search-fast"`
}

func (*SearchOptions) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["max_age_hours"].Format = ""
	schema.Properties["results"].Format = ""
	schema.Properties["contents"].Items.Enum = []any{"text", "highlights", "summary"}
	schema.Description = "How this config finds out today's answers."
	return schema
}

// SearchOptionsContents is the SearchOptionsContents schema.
type SearchOptionsContents string

// Defines values for SearchOptionsContents.
const (
	SearchOptionsContentsHighlights SearchOptionsContents = "highlights"
	SearchOptionsContentsSummary    SearchOptionsContents = "summary"
	SearchOptionsContentsText       SearchOptionsContents = "text"
)

// Valid indicates whether the value is a known member of the SearchOptionsContents enum.
func (e SearchOptionsContents) Valid() bool {
	switch e {
	case SearchOptionsContentsHighlights:
		return true
	case SearchOptionsContentsSummary:
		return true
	case SearchOptionsContentsText:
		return true
	default:
		return false
	}
}

// StsOptionsTurnDetection is the StsOptionsTurnDetection schema.
type StsOptionsTurnDetection string

// Defines values for StsOptionsTurnDetection.
const (
	StsOptionsTurnDetectionNone      StsOptionsTurnDetection = "none"
	StsOptionsTurnDetectionSemantic  StsOptionsTurnDetection = "semantic"
	StsOptionsTurnDetectionServerVad StsOptionsTurnDetection = "server_vad"
)

// Valid indicates whether the value is a known member of the StsOptionsTurnDetection enum.
func (e StsOptionsTurnDetection) Valid() bool {
	switch e {
	case StsOptionsTurnDetectionNone:
		return true
	case StsOptionsTurnDetectionSemantic:
		return true
	case StsOptionsTurnDetectionServerVad:
		return true
	default:
		return false
	}
}

// TranscriptFormat What a finished transcript is rendered as. json carries the words and speakers; srt and vtt are subtitle files. Recording only.
type TranscriptFormat string

// Defines values for TranscriptFormat.
const (
	Json TranscriptFormat = "json"
	Srt  TranscriptFormat = "srt"
	Vtt  TranscriptFormat = "vtt"
)

// Valid indicates whether the value is a known member of the TranscriptFormat enum.
func (e TranscriptFormat) Valid() bool {
	switch e {
	case Json:
		return true
	case Srt:
		return true
	case Vtt:
		return true
	default:
		return false
	}
}

func (TranscriptFormat) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "TranscriptFormat", "What a finished transcript is rendered as. json carries the words and speakers; srt and vtt are subtitle files. Recording only.", "json", "srt", "vtt")
}

// TranscriptionMode How faithfully the transcript follows what was said. verbatim keeps the ums, the repetitions and the false starts; smart removes them, tidies the grammar and formats the result, which is why it cannot also diarize or time the words - they may no longer be the words that were spoken. Almost no provider offers both, so this narrows where a request can go.
type TranscriptionMode string

// Defines values for TranscriptionMode.
const (
	Smart    TranscriptionMode = "smart"
	Verbatim TranscriptionMode = "verbatim"
)

// Valid indicates whether the value is a known member of the TranscriptionMode enum.
func (e TranscriptionMode) Valid() bool {
	switch e {
	case Smart:
		return true
	case Verbatim:
		return true
	default:
		return false
	}
}

func (TranscriptionMode) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "TranscriptionMode", "How faithfully the transcript follows what was said. verbatim keeps the ums, the repetitions and the false starts; smart removes them, tidies the grammar and formats the result, which is why it cannot also diarize or time the words - they may no longer be the words that were spoken. Almost no provider offers both, so this narrows where a request can go.", "verbatim", "smart")
}
