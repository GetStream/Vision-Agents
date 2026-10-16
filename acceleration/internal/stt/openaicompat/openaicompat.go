// Package openaicompat implements batch transcription through a configured
// OpenAI-compatible audio transcription endpoint.
package openaicompat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const (
	// ProviderName is the stable provider identifier used by routing and stats.
	ProviderName = "openai-compatible"
	// DefaultModel is the conventional transcription alias used when no routing model
	// is supplied.
	DefaultModel = "whisper-1"

	baseURLEnv       = "OPENAI_STT_BASE_URL"
	apiKeyEnv        = "OPENAI_STT_API_KEY"
	maxAudioBytes    = 25 << 20
	maxResponseBytes = 2 << 20
	requestTimeout   = 2 * time.Minute
)

var errUnsupportedInput = errors.New("recording option is not supported")

// TranscriptionSettings are the standard multipart fields the endpoint understands.
// Unknown fields in a provider overwrite are rejected by routing.Spec.Settings.
type TranscriptionSettings struct {
	Prompt                 string          `json:"prompt,omitempty"`
	Language               string          `json:"language,omitempty"`
	Languages              []string        `json:"languages,omitempty"`
	Keywords               []string        `json:"keywords,omitempty"`
	Temperature            *float64        `json:"temperature,omitempty"`
	ResponseFormat         string          `json:"response_format,omitempty"`
	TimestampGranularities []string        `json:"timestamp_granularities,omitempty"`
	Include                []string        `json:"include,omitempty"`
	ChunkingStrategy       json.RawMessage `json:"chunking_strategy,omitempty"`
	KnownSpeakerNames      []string        `json:"known_speaker_names,omitempty"`
	KnownSpeakerReferences []string        `json:"known_speaker_references,omitempty"`
	Delay                  string          `json:"delay,omitempty"`
	Stream                 *bool           `json:"stream,omitempty"`
}

// Options configures one OpenAI-compatible transcription endpoint.
type Options struct {
	BaseURL string
	APIKey  string
	Model   string
	// Language and Languages remain available to realtime callers and are batch fallbacks
	// when neither typed transcription settings nor per-recording hints were supplied.
	Language   string
	Languages  []string
	Keyterms   []string
	Settings   TranscriptionSettings
	HTTPClient *http.Client
}

// Transcriber sends a whole recording to an explicitly configured endpoint.
type Transcriber struct {
	endpoint  string
	apiKey    string
	model     string
	language  string
	languages []string
	keyterms  []string
	settings  TranscriptionSettings
	client    *http.Client
}

// endpointCredentials applies process configuration only when the endpoint itself came
// from the environment. A caller's explicit endpoint must never inherit an unrelated key.
func endpointCredentials(options Options) (baseURL, apiKey string) {
	baseURL = strings.TrimSpace(options.BaseURL)
	apiKey = strings.TrimSpace(options.APIKey)
	if baseURL != "" {
		return baseURL, apiKey
	}
	baseURL = strings.TrimSpace(os.Getenv(baseURLEnv))
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv(apiKeyEnv))
	}
	return baseURL, apiKey
}

// New validates the endpoint configuration. An API key is optional for local and
// otherwise keyless compatible endpoints.
func New(options Options) (*Transcriber, error) {
	baseURL, apiKey := endpointCredentials(options)
	if baseURL == "" {
		return nil, stack.Wrap(errors.New("openai-compatible: base URL is required (set OPENAI_STT_BASE_URL)"))
	}

	endpoint, err := transcriptionEndpoint(baseURL)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("openai-compatible: configure endpoint: %w", err))
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if len(options.Model) > 128 || strings.ContainsAny(options.Model, "\r\n") || strings.TrimSpace(options.Model) == "" {
		return nil, stack.Wrap(errors.New("openai-compatible: model must be a non-empty identifier of at most 128 bytes"))
	}

	client := &http.Client{Timeout: requestTimeout}
	if options.HTTPClient != nil {
		clientCopy := *options.HTTPClient
		client = &clientCopy
		if client.Timeout == 0 {
			client.Timeout = requestTimeout
		}
	}
	// Do not forward a bearer token through an operator-configured redirect.
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }

	return &Transcriber{
		endpoint:  endpoint,
		apiKey:    apiKey,
		model:     options.Model,
		language:  options.Language,
		languages: append([]string(nil), options.Languages...),
		keyterms:  append([]string(nil), options.Keyterms...),
		settings:  options.Settings,
		client:    client,
	}, nil
}

// Start satisfies stt.Transcriber. The HTTP client has no persistent session to open.
func (p *Transcriber) Start(context.Context) error { return nil }

// Close satisfies stt.Transcriber. The shared HTTP transport is owned by net/http.
func (p *Transcriber) Close() error { return nil }

// Provider is the stable provider identifier used in routing and stats.
func (p *Transcriber) Provider() string { return ProviderName }

// Model is the configured routing alias sent to the transcription endpoint.
func (p *Transcriber) Model() string { return p.model }

// Transcribe submits one recording and consumes the endpoint's JSON, text, subtitle or
// event-stream response into the router's transcript contract.
func (p *Transcriber) Transcribe(ctx context.Context, recording stt.Recording) (stt.Transcription, error) {
	if err := recording.Validate(); err != nil {
		return stt.Transcription{}, stack.Wrap(err)
	}
	if err := validateRecordingOptions(recording); err != nil {
		return stt.Transcription{}, stack.Wrap(err)
	}
	if len(recording.Audio) > maxAudioBytes {
		return stt.Transcription{}, stack.Wrap(fmt.Errorf("openai-compatible: audio exceeds %d bytes", maxAudioBytes))
	}

	settings, err := p.requestSettings(recording)
	if err != nil {
		return stt.Transcription{}, stack.Wrap(fmt.Errorf("openai-compatible: build request: %w", err))
	}
	request, err := p.newRequest(ctx, recording, settings)
	if err != nil {
		return stt.Transcription{}, stack.Wrap(fmt.Errorf("openai-compatible: build request: %w", err))
	}
	response, err := p.client.Do(request)
	if err != nil {
		return stt.Transcription{}, stack.Wrap(fmt.Errorf("openai-compatible: transcribe: %w", err))
	}
	defer response.Body.Close()

	body, err := io.ReadAll(io.LimitReader(response.Body, maxResponseBytes+1))
	if err != nil {
		return stt.Transcription{}, stack.Wrap(fmt.Errorf("openai-compatible: read transcript: %w", err))
	}
	if len(body) > maxResponseBytes {
		return stt.Transcription{}, stack.Wrap(fmt.Errorf("openai-compatible: transcript response exceeds %d bytes", maxResponseBytes))
	}
	if response.StatusCode != http.StatusOK {
		return stt.Transcription{}, stack.Wrap(fmt.Errorf("openai-compatible: transcribe: %w", transcriptionAPIError(response.StatusCode, body)))
	}

	transcription, _, err := parseTranscription(body, response.Header.Get("Content-Type"), settings.responseFormat)
	if err != nil {
		return stt.Transcription{}, stack.Wrap(err)
	}
	if transcription.Language == "" {
		if settings.language != "" {
			transcription.Language = settings.language
		} else if len(settings.languages) == 1 {
			transcription.Language = settings.languages[0]
		}
	}
	return transcription, nil
}

func validateRecordingOptions(recording stt.Recording) error {
	if recording.URL != "" {
		return fmt.Errorf("openai-compatible: URL sources are not supported: %w", errUnsupportedInput)
	}
	if recording.MaxSpeakers != 0 || recording.Redact || recording.Summary || recording.Entities ||
		recording.Channels < 0 || recording.Channels > 1 || recording.ProfanityFilter || recording.FillerWords {
		return fmt.Errorf("openai-compatible: requested transcription feature is not supported: %w", errUnsupportedInput)
	}
	return nil
}
