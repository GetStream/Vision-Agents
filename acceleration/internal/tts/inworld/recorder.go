package inworld

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
	"unicode/utf16"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// DefaultRecordingFormat is the one ElevenLabs' recorder defaults to, so a recording that
// asks for no format is the same kind of file whichever vendor spoke it. Inworld's own
// default is MP3 at 128 kbps
// (https://docs.inworld.ai/api-reference/ttsAPI/texttospeech/synthesize-speech, read 2026-10-19).
const DefaultRecordingFormat = "mp3_44100_128"

// recorderURL is the endpoint that returns the whole file
// (https://docs.inworld.ai/api-reference/ttsAPI/texttospeech/synthesize-speech, read 2026-10-19).
const recorderURL = "https://api.inworld.ai/tts/v1/voice"

// recorderTimeout bounds one recording, as Cartesia's does.
const recorderTimeout = 10 * time.Minute

// MaxRecordingText is the most text /tts/v1/voice takes, in UTF-16 code units
// (https://docs.inworld.ai/api-reference/ttsAPI/texttospeech/synthesize-speech, read 2026-10-19).
const MaxRecordingText = 2_000

// batchSuffix ends the model name of the router's batch entry. The endpoint takes the same
// model ids the live socket does, and a model is declared once in router.yaml, so the batch
// entry is the live id with this suffix and the suffix is not sent.
const batchSuffix = "-batch"

// ErrFormat says a recording asked for a file Inworld cannot produce. The recorder is
// refused when built rather than handing back a different file than was asked for.
var ErrFormat = errors.New("inworld: format is not one Inworld records")

// ErrTextTooLong says a recording's text is more than one request to /tts/v1/voice takes.
var ErrTextTooLong = errors.New("inworld: text is longer than one recording takes")

// minRecordingSampleRate and maxRecordingSampleRate bound sampleRateHertz
// (https://docs.inworld.ai/api-reference/ttsAPI/texttospeech/synthesize-speech, read 2026-10-19).
const (
	minRecordingSampleRate = 8_000
	maxRecordingSampleRate = 48_000
)

// recordingEncodings maps a format name's codec to Inworld's audioEncoding. The names are
// ElevenLabs' codec_samplerate[_kbps] (tts.Recording.Format). pcm is raw little-endian
// 16-bit, which is how the live voice already reads Inworld's PCM (inworld.go handleAudio);
// Inworld's PCM is headerless and its WAV has a header in this response. Its encodings are
// LINEAR16, MP3, OGG_OPUS, ALAW, MULAW, FLAC, PCM and WAV
// (https://docs.inworld.ai/api-reference/ttsAPI/texttospeech/synthesize-speech, read 2026-10-19).
// opus is left out: ElevenLabs does not say which container its opus_* is in, and OGG_OPUS
// would be a different file if it is not Ogg.
var recordingEncodings = map[string]string{
	"mp3":  "MP3",
	"pcm":  "PCM",
	"wav":  "WAV",
	"ulaw": "MULAW",
	"alaw": "ALAW",
}

// RecorderOptions configures the batch voice. APIKey falls back to INWORLD_API_KEY and
// VoiceID to INWORLD_VOICE_ID.
type RecorderOptions struct {
	APIKey  string
	VoiceID string
	// Model is an Inworld model id, or the router's batch entry for one.
	Model string
	// Language is a base ISO code such as "en", for a recording that names none. Empty lets
	// the model detect it.
	Language string
	// DeliveryMode is STABLE, BALANCED or CREATIVE, as the live voice takes it. Empty leaves
	// it to the server.
	DeliveryMode string
	// Format is the format the recording will be asked for, empty for the default. It is
	// only checked here, so a format Inworld cannot produce is refused when the recorder is
	// built; what is sent comes from the recording.
	Format string
	// BaseURL overrides the endpoint, for a proxy or a test server.
	BaseURL string
	Logger  *slog.Logger
}

// Recorder speaks whole texts into single files on /tts/v1/voice, where the rest of this
// package streams sentences over a socket.
type Recorder struct {
	options RecorderOptions
	client  *http.Client
	logger  *slog.Logger
}

// NewRecorder validates the options and returns a recorder.
func NewRecorder(options RecorderOptions) (*Recorder, error) {
	keys := keypool.FromEnv(options.APIKey, "INWORLD_API_KEY")
	if keys.Len() == 0 {
		return nil, stack.Wrap(errors.New("inworld: api key is required (set INWORLD_API_KEY)"))
	}
	options.APIKey = keys.Peek()
	if options.VoiceID == "" {
		options.VoiceID = os.Getenv("INWORLD_VOICE_ID")
	}
	if options.VoiceID == "" {
		options.VoiceID = DefaultVoiceID
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if _, err := recordingFormat(options.Format); err != nil {
		return nil, err
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
		// The key is Inworld's Base64 credential, sent as is after Basic, as the live
		// voice does.
		client: keys.WrapClient(&http.Client{Timeout: recorderTimeout}, keypool.Header("Authorization", "Basic ")),
		logger: logger.With("provider", ProviderName, "model", options.Model),
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

// Voice is the speaker of a recording that names none, which makes the recorder a
// tts.Voiced: the recording router reads it to report the voice that spoke.
func (r *Recorder) Voice() string { return r.options.VoiceID }

// Record speaks the text and returns the file.
func (r *Recorder) Record(ctx context.Context, recording tts.Recording) (tts.Recorded, error) {
	if err := recording.Validate(); err != nil {
		return tts.Recorded{}, stack.Wrap(err)
	}
	// One request is one file, and files are not joined: mp3 and wav cannot be cut
	// together byte for byte, so a longer text goes to a vendor that takes it.
	if units := utf16Length(recording.Text); units > MaxRecordingText {
		return tts.Recorded{}, stack.Wrap(fmt.Errorf("%w: %d UTF-16 code units, at most %d",
			ErrTextTooLong, units, MaxRecordingText))
	}
	name := recording.Format
	if name == "" {
		name = DefaultRecordingFormat
	}
	format, err := recordingFormat(name)
	if err != nil {
		return tts.Recorded{}, err
	}
	voice := recording.Voice
	if voice == "" {
		voice = r.options.VoiceID
	}
	language := recording.Language
	if language == "" {
		language = r.options.Language
	}

	body, err := json.Marshal(recorderRequest{
		Text:         recording.Text,
		VoiceID:      voice,
		ModelID:      strings.TrimSuffix(r.options.Model, batchSuffix),
		AudioConfig:  format,
		Language:     inworldLanguage(language),
		DeliveryMode: r.options.DeliveryMode,
		Temperature:  recordingTemperature,
	})
	if err != nil {
		return tts.Recorded{}, stack.Wrap(fmt.Errorf("inworld: encode request: %w", err))
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, r.options.BaseURL, bytes.NewReader(body))
	if err != nil {
		return tts.Recorded{}, stack.Wrap(fmt.Errorf("inworld: build request: %w", err))
	}
	request.Header.Set("Authorization", "Basic "+r.options.APIKey)
	request.Header.Set("Content-Type", "application/json")

	response, err := r.client.Do(request)
	if err != nil {
		return tts.Recorded{}, stack.Wrap(fmt.Errorf("inworld: record: %w", err))
	}
	defer response.Body.Close()

	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return tts.Recorded{}, stack.Wrap(fmt.Errorf("inworld: read response: %w", err))
	}
	if response.StatusCode != http.StatusOK {
		return tts.Recorded{}, stack.Wrap(fmt.Errorf("inworld: record: %s: %s",
			response.Status, strings.TrimSpace(string(raw))))
	}
	var answer recorderResponse
	if err := json.Unmarshal(raw, &answer); err != nil {
		return tts.Recorded{}, stack.Wrap(fmt.Errorf("inworld: decode response: %w", err))
	}
	audio, err := base64.StdEncoding.DecodeString(answer.AudioContent)
	if err != nil {
		return tts.Recorded{}, stack.Wrap(fmt.Errorf("inworld: decode audio: %w", err))
	}

	return tts.Recorded{
		Audio:      audio,
		Format:     name,
		Characters: int64(len([]rune(recording.Text))),
	}, nil
}

// recorderRequest is the body of one synthesis on /tts/v1/voice
// (https://docs.inworld.ai/api-reference/ttsAPI/texttospeech/synthesize-speech, read 2026-10-19).
// speakingRate is not sent: the models do not declare speed in router.yaml, so the router
// refuses a recording that asks for one.
type recorderRequest struct {
	Text         string      `json:"text"`
	VoiceID      string      `json:"voiceId"`
	ModelID      string      `json:"modelId"`
	AudioConfig  audioConfig `json:"audioConfig"`
	Language     string      `json:"language,omitempty"`
	DeliveryMode string      `json:"deliveryMode,omitempty"`
	Temperature  float64     `json:"temperature"`
}

// recordingTemperature is what the live socket sends too (inworld.go, createConfig). The docs
// say inworld-tts-2 ignores it and the other models default to 1
// (https://docs.inworld.ai/api-reference/ttsAPI/texttospeech/synthesize-speech, read 2026-10-20).
const recordingTemperature = 1.1

// recorderResponse is what /tts/v1/voice answers: the whole file, base64.
type recorderResponse struct {
	AudioContent string `json:"audioContent"`
}

// recordingFormat reads a format name into Inworld's audioConfig, refusing one it has no
// encoding or sample rate for. Empty is the default.
func recordingFormat(name string) (audioConfig, error) {
	if name == "" {
		name = DefaultRecordingFormat
	}
	refused := stack.Wrap(fmt.Errorf("%w: %s", ErrFormat, name))
	parts := strings.Split(name, "_")
	encoding, known := recordingEncodings[parts[0]]
	if !known || len(parts) < 2 {
		return audioConfig{}, refused
	}
	rate, err := strconv.Atoi(parts[1])
	if err != nil || rate < minRecordingSampleRate || rate > maxRecordingSampleRate {
		return audioConfig{}, refused
	}
	format := audioConfig{AudioEncoding: encoding, SampleRateHertz: rate}
	// Only mp3 names a bit rate, in kbps.
	if encoding != "MP3" {
		if len(parts) != 2 {
			return audioConfig{}, refused
		}
		return format, nil
	}
	if len(parts) != 3 {
		return audioConfig{}, refused
	}
	kbps, err := strconv.Atoi(parts[2])
	if err != nil || kbps <= 0 {
		return audioConfig{}, refused
	}
	format.BitRate = kbps * 1000
	return format, nil
}

// inworldLanguage is the code Inworld is sent for a recording's language: lower case, and
// fil for tl. Probed 2026-10-20 14:19Z on inworld-tts-2 with the repo key: of the 42 codes
// of the batch entry only tl is refused (400 "language 'tl' could not be resolved to a
// BCP-47 tag or known alias"); fil is 200 and tl-PH is 400.
func inworldLanguage(language string) string {
	language = strings.ToLower(language)
	if language == "tl" {
		return "fil"
	}
	return language
}

// utf16Length is how /tts/v1/voice counts text: JavaScript's text.length.
func utf16Length(text string) int {
	units := 0
	for _, r := range text {
		units += utf16.RuneLen(r)
	}
	return units
}
