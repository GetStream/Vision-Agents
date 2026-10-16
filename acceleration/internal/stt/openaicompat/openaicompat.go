// Package openaicompat implements batch transcription through a configured
// OpenAI-compatible audio transcription endpoint.
package openaicompat

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
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
	maxAudioSeconds  = 60
	maxWAVChunks     = 1024
	requestTimeout   = 2 * time.Minute
)

var (
	errInvalidWAV       = errors.New("invalid PCM WAV")
	errUnsupportedWAV   = errors.New("unsupported PCM WAV format")
	errAudioTooLong     = errors.New("audio exceeds 60 seconds")
	errUnsupportedInput = errors.New("recording option is not supported")
)

// Options configures one OpenAI-compatible transcription endpoint.
type Options struct {
	BaseURL  string
	APIKey   string
	Model    string
	Language string
}

// Transcriber sends whole WAV files to an explicitly configured endpoint.
type Transcriber struct {
	endpoint string
	apiKey   string
	model    string
	client   *http.Client
}

// New validates the endpoint and credentials. There is no default endpoint: the
// operator must configure both OPENAI_STT_BASE_URL and OPENAI_STT_API_KEY.
func New(options Options) (*Transcriber, error) {
	if options.BaseURL == "" {
		options.BaseURL = os.Getenv(baseURLEnv)
	}
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnv)
	}
	options.BaseURL = strings.TrimSpace(options.BaseURL)
	options.APIKey = strings.TrimSpace(options.APIKey)
	if options.BaseURL == "" {
		return nil, stack.Wrap(errors.New("openai-compatible: base URL is required (set OPENAI_STT_BASE_URL)"))
	}
	if options.APIKey == "" {
		return nil, stack.Wrap(errors.New("openai-compatible: API key is required (set OPENAI_STT_API_KEY)"))
	}

	endpoint, err := transcriptionEndpoint(options.BaseURL)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("openai-compatible: configure endpoint: %w", err))
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if len(options.Model) > 128 || strings.ContainsAny(options.Model, "\r\n") || strings.TrimSpace(options.Model) == "" {
		return nil, stack.Wrap(errors.New("openai-compatible: model must be a non-empty identifier of at most 128 bytes"))
	}

	return &Transcriber{
		endpoint: endpoint,
		apiKey:   options.APIKey,
		model:    options.Model,
		client: &http.Client{
			Timeout: requestTimeout,
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
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

// Transcribe submits one supported recording and returns its transcript as plain text.
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

	durationMs, err := inspectWAV(ctx, recording.Audio)
	if err != nil {
		return stt.Transcription{}, stack.Wrap(fmt.Errorf("openai-compatible: validate audio: %w", err))
	}
	request, err := p.newRequest(ctx, recording)
	if err != nil {
		return stt.Transcription{}, stack.Wrap(fmt.Errorf("openai-compatible: build request: %w", err))
	}

	response, err := p.client.Do(request)
	if err != nil {
		return stt.Transcription{}, stack.Wrap(fmt.Errorf("openai-compatible: transcribe: %w", err))
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return stt.Transcription{}, stack.Wrap(fmt.Errorf("openai-compatible: transcribe: endpoint returned HTTP %d", response.StatusCode))
	}

	body, err := io.ReadAll(io.LimitReader(response.Body, maxResponseBytes+1))
	if err != nil {
		return stt.Transcription{}, stack.Wrap(fmt.Errorf("openai-compatible: read transcript: %w", err))
	}
	if len(body) > maxResponseBytes {
		return stt.Transcription{}, stack.Wrap(fmt.Errorf("openai-compatible: transcript response exceeds %d bytes", maxResponseBytes))
	}
	text, err := transcriptText(body)
	if err != nil {
		return stt.Transcription{}, stack.Wrap(err)
	}

	language := ""
	if len(recording.Languages) == 1 {
		language = recording.Languages[0]
	}
	return stt.Transcription{
		Text:            text,
		Language:        language,
		AudioDurationMs: durationMs,
	}, nil
}

func validateRecordingOptions(recording stt.Recording) error {
	if recording.URL != "" {
		return fmt.Errorf("openai-compatible: URL sources are not supported: %w", errUnsupportedInput)
	}
	if len(recording.Languages) > 1 {
		return fmt.Errorf("openai-compatible: only one language hint is supported: %w", errUnsupportedInput)
	}
	if len(recording.Languages) == 1 && !validLanguage(recording.Languages[0]) {
		return errors.New("openai-compatible: language must be a two-letter lowercase ISO 639-1 code")
	}
	if recording.Diarize || recording.MaxSpeakers != 0 || recording.Words || recording.Format || recording.Redact ||
		recording.Summary || recording.Entities || len(recording.Keyterms) > 0 || recording.Channels < 0 || recording.Channels > 1 ||
		recording.ProfanityFilter || recording.FillerWords {
		return fmt.Errorf("openai-compatible: requested transcription feature is not supported: %w", errUnsupportedInput)
	}
	return nil
}

func validLanguage(language string) bool {
	if len(language) != 2 {
		return false
	}
	for _, character := range language {
		if character < 'a' || character > 'z' {
			return false
		}
	}
	return true
}

func (p *Transcriber) newRequest(ctx context.Context, recording stt.Recording) (*http.Request, error) {
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	file, err := form.CreateFormFile("file", "audio.wav")
	if err != nil {
		return nil, err
	}
	if _, err := file.Write(recording.Audio); err != nil {
		return nil, err
	}
	if err := form.WriteField("model", p.model); err != nil {
		return nil, err
	}
	if err := form.WriteField("response_format", "json"); err != nil {
		return nil, err
	}
	if len(recording.Languages) == 1 {
		if err := form.WriteField("language", recording.Languages[0]); err != nil {
			return nil, err
		}
	}
	if err := form.Close(); err != nil {
		return nil, err
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint, &body)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Authorization", "Bearer "+p.apiKey)
	request.Header.Set("Content-Type", form.FormDataContentType())
	return request, nil
}

func transcriptionEndpoint(base string) (string, error) {
	parsed, err := parseBaseURL(base)
	if err != nil {
		return "", err
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/") + "/audio/transcriptions"
	parsed.RawPath = ""
	return parsed.String(), nil
}

func transcriptText(body []byte) (string, error) {
	decoder := json.NewDecoder(bytes.NewReader(body))
	var response map[string]json.RawMessage
	if err := decoder.Decode(&response); err != nil || response == nil {
		return "", errors.New("openai-compatible: decode transcript: response must be a JSON object")
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return "", errors.New("openai-compatible: decode transcript: response has trailing JSON")
	}
	rawText, ok := response["text"]
	if !ok {
		return "", errors.New("openai-compatible: decode transcript: response is missing text")
	}
	var text *string
	if err := json.Unmarshal(rawText, &text); err != nil || text == nil {
		return "", errors.New("openai-compatible: decode transcript: text must be a string")
	}
	return *text, nil
}

// inspectWAV validates the same bounded PCM16 WAV subset accepted by the private
// transcription endpoint, without decoding or resampling samples in the router.
func inspectWAV(ctx context.Context, data []byte) (int64, error) {
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return 0, errInvalidWAV
	}
	if uint64(binary.LittleEndian.Uint32(data[4:8]))+8 != uint64(len(data)) {
		return 0, errInvalidWAV
	}

	var sampleRate uint32
	var haveFormat, haveData bool
	var dataSize uint64
	chunks := 0
	for offset := 12; offset < len(data); {
		if chunks >= maxWAVChunks || len(data)-offset < 8 {
			return 0, errInvalidWAV
		}
		if chunks%128 == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		chunks++
		chunkID := string(data[offset : offset+4])
		chunkSize := uint64(binary.LittleEndian.Uint32(data[offset+4 : offset+8]))
		offset += 8
		end := uint64(offset) + chunkSize
		if end > uint64(len(data)) {
			return 0, errInvalidWAV
		}
		chunk := data[offset:int(end)]
		switch chunkID {
		case "fmt ":
			if haveFormat || len(chunk) < 16 {
				return 0, errInvalidWAV
			}
			format := binary.LittleEndian.Uint16(chunk[0:2])
			channels := binary.LittleEndian.Uint16(chunk[2:4])
			sampleRate = binary.LittleEndian.Uint32(chunk[4:8])
			byteRate := binary.LittleEndian.Uint32(chunk[8:12])
			blockAlign := binary.LittleEndian.Uint16(chunk[12:14])
			bits := binary.LittleEndian.Uint16(chunk[14:16])
			if format != 1 || channels != 1 || bits != 16 || blockAlign != 2 ||
				(sampleRate != 16000 && sampleRate != 24000) || byteRate != sampleRate*2 {
				return 0, errUnsupportedWAV
			}
			haveFormat = true
		case "data":
			if !haveFormat || haveData || chunkSize == 0 || chunkSize%2 != 0 {
				return 0, errInvalidWAV
			}
			if chunkSize > uint64(sampleRate)*2*maxAudioSeconds {
				return 0, errAudioTooLong
			}
			dataSize = chunkSize
			haveData = true
		}
		offset = int(end)
		if chunkSize%2 != 0 {
			if offset >= len(data) {
				return 0, errInvalidWAV
			}
			offset++
		}
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if !haveFormat || !haveData {
		return 0, errInvalidWAV
	}

	frames := dataSize / 2
	return int64((frames*1000 + uint64(sampleRate) - 1) / uint64(sampleRate)), nil
}
