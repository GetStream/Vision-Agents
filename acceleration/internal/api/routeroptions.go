package api

import (
	"encoding/json"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
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
	return options.STT{
		Target:         value(sent.Target),
		Languages:      value(sent.Languages),
		DetectLanguage: sent.DetectLanguage,
		SampleRate:     sent.SampleRate,
		Interim:        sent.Interim,
		Endpointing:    string(value(sent.Endpointing)),
		SilenceMs:      sent.SilenceMs,
		UtteranceEndMs: sent.UtteranceEndMs,
		Diarize:        sent.Diarize,
		MaxSpeakers:    sent.MaxSpeakers,
		Keyterms:       value(sent.Keyterms),
		Format:         sent.Format,
		Redact:         sent.Redact,
		Events:         sent.Events,
		Channels:       sent.Channels,
		Words:          sent.Words,
		Output:         string(value(sent.Output)),
		Summary:        sent.Summary,
		Entities:       sent.Entities,
	}
}

func sttOptionsFor(held options.STT) *SttOptions {
	sent := &SttOptions{
		Target:         optional(held.Target),
		Languages:      list(held.Languages),
		DetectLanguage: held.DetectLanguage,
		SampleRate:     held.SampleRate,
		Interim:        held.Interim,
		SilenceMs:      held.SilenceMs,
		UtteranceEndMs: held.UtteranceEndMs,
		Diarize:        held.Diarize,
		MaxSpeakers:    held.MaxSpeakers,
		Keyterms:       list(held.Keyterms),
		Format:         held.Format,
		Redact:         held.Redact,
		Events:         held.Events,
		Channels:       held.Channels,
		Words:          held.Words,
		Summary:        held.Summary,
		Entities:       held.Entities,
	}
	if held.Endpointing != "" {
		endpointing := Endpointing(held.Endpointing)
		sent.Endpointing = &endpointing
	}
	if held.Output != "" {
		output := TranscriptFormat(held.Output)
		sent.Output = &output
	}
	return sent
}

func ttsOptionsOf(sent *TtsOptions) options.TTS {
	if sent == nil {
		return options.TTS{}
	}
	return options.TTS{
		Target:         value(sent.Target),
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
	}
}

func ttsOptionsFor(held options.TTS) *TtsOptions {
	sent := &TtsOptions{
		Target:        optional(held.Target),
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
		Instructions:    value(sent.Instructions),
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
		Instructions:    optional(held.Instructions),
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

func searchOptionsOf(sent *SearchOptions) options.Search {
	if sent == nil {
		return options.Search{}
	}
	held := options.Search{
		Target:         value(sent.Target),
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
