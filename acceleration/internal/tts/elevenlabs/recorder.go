package elevenlabs

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// RecorderModel is what a recording is spoken by. It is the quality model rather than
// flash: flash exists to get the first sound out quickly, which is worth nothing to a file
// nobody is waiting on.
const RecorderModel = "eleven_multilingual_v2"

// DefaultRecordingFormat is a good MP3 rather than PCM, since this is a file to keep
// rather than samples to play into a call.
const DefaultRecordingFormat = "mp3_44100_128"

// recorderURL is the non-streaming synthesis endpoint.
const recorderURL = "https://api.elevenlabs.io/v1/text-to-speech"

// recorderTimeout bounds one recording. A chapter is a minute or two of request.
const recorderTimeout = 10 * time.Minute

// RecorderOptions configures the batch voice. APIKey falls back to ELEVENLABS_API_KEY and
// VoiceID to ELEVENLABS_VOICE_ID.
type RecorderOptions struct {
	APIKey  string
	VoiceID string
	Model   string
	// BaseURL overrides the endpoint, for a proxy or a test server.
	BaseURL string
	Logger  *slog.Logger
}

// Recorder speaks whole texts into single files.
//
// The streaming socket the rest of this package uses returns audio as it is generated,
// which is what a conversation needs and what makes an audiobook arrive in a thousand
// pieces to be stitched back together. This asks for the file instead.
type Recorder struct {
	options RecorderOptions
	client  *http.Client
	logger  *slog.Logger
}

// NewRecorder validates the options and returns a recorder.
func NewRecorder(options RecorderOptions) (*Recorder, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv("ELEVENLABS_API_KEY")
	}
	if options.APIKey == "" {
		return nil, errors.New("elevenlabs: api key is required (set ELEVENLABS_API_KEY)")
	}
	if options.VoiceID == "" {
		options.VoiceID = os.Getenv("ELEVENLABS_VOICE_ID")
	}
	if options.VoiceID == "" {
		options.VoiceID = DefaultVoiceID
	}
	if options.Model == "" {
		options.Model = RecorderModel
	}
	if options.BaseURL == "" {
		options.BaseURL = recorderURL
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &Recorder{
		options: options,
		client:  &http.Client{Timeout: recorderTimeout},
		logger:  logger.With("provider", ProviderName, "model", options.Model),
	}, nil
}

// Start is what routing.Provider asks for. There is no connection to open.
func (r *Recorder) Start(context.Context) error { return nil }

// Close is what routing.Provider asks for.
func (r *Recorder) Close() error { return nil }

// Provider is the stable provider name used in stats.
func (r *Recorder) Provider() string { return ProviderName }

// Model is the model identifier used in stats.
func (r *Recorder) Model() string { return r.options.Model }

// Record speaks the text and returns the file.
func (r *Recorder) Record(ctx context.Context, recording tts.Recording) (tts.Recorded, error) {
	if err := recording.Validate(); err != nil {
		return tts.Recorded{}, err
	}

	voice := recording.Voice
	if voice == "" {
		voice = r.options.VoiceID
	}
	format := recording.Format
	if format == "" {
		format = DefaultRecordingFormat
	}

	body, err := json.Marshal(r.payload(recording))
	if err != nil {
		return tts.Recorded{}, fmt.Errorf("elevenlabs: encode request: %w", err)
	}

	endpoint := fmt.Sprintf("%s/%s?output_format=%s",
		r.options.BaseURL, url.PathEscape(voice), url.QueryEscape(format))
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return tts.Recorded{}, fmt.Errorf("elevenlabs: build request: %w", err)
	}
	request.Header.Set("xi-api-key", r.options.APIKey)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "audio/*")

	response, err := r.client.Do(request)
	if err != nil {
		return tts.Recorded{}, fmt.Errorf("elevenlabs: record: %w", err)
	}
	defer response.Body.Close()

	audio, err := io.ReadAll(response.Body)
	if err != nil {
		return tts.Recorded{}, fmt.Errorf("elevenlabs: read audio: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		return tts.Recorded{}, fmt.Errorf("elevenlabs: record: %s: %s",
			response.Status, strings.TrimSpace(string(audio)))
	}

	return tts.Recorded{
		Audio:      audio,
		Format:     format,
		Characters: int64(len([]rune(recording.Text))),
	}, nil
}

// recorderRequest is the body of one synthesis. Fields are omitted when unset because the
// server distinguishes an absent field from a zero one, and a stability of nought is a
// setting rather than a silence.
type recorderRequest struct {
	Text          string                 `json:"text"`
	ModelID       string                 `json:"model_id"`
	LanguageCode  string                 `json:"language_code,omitempty"`
	VoiceSettings *recorderVoiceSettings `json:"voice_settings,omitempty"`
	Pronunciation []pronunciation        `json:"pronunciation_dictionary_locators,omitempty"`
}

type recorderVoiceSettings struct {
	Stability       *float64 `json:"stability,omitempty"`
	SimilarityBoost *float64 `json:"similarity_boost,omitempty"`
	Speed           *float64 `json:"speed,omitempty"`
	Style           *float64 `json:"style,omitempty"`
}

// pronunciation is a dictionary the account already holds. ElevenLabs takes pronunciations
// by reference rather than inline, which is why a caller's word list cannot be honoured
// here and the provider does not declare that term.
type pronunciation struct {
	DictionaryID string `json:"pronunciation_dictionary_id"`
	VersionID    string `json:"version_id"`
}

func (r *Recorder) payload(recording tts.Recording) recorderRequest {
	payload := recorderRequest{
		Text:    recording.Text,
		ModelID: r.options.Model,
	}
	if recording.Language != "" && slices.Contains(multilingualModels, r.options.Model) {
		payload.LanguageCode = strings.ToLower(recording.Language)
	}

	settings := recorderVoiceSettings{}
	named := false
	if recording.Stability > 0 {
		settings.Stability = &recording.Stability
		named = true
	}
	if recording.Similarity > 0 {
		settings.SimilarityBoost = &recording.Similarity
		named = true
	}
	if recording.Speed > 0 {
		settings.Speed = &recording.Speed
		named = true
	}
	if named {
		payload.VoiceSettings = &settings
	}
	return payload
}
