package cartesia

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// DefaultRecordingFormat is the one ElevenLabs' recorder defaults to, so a recording that
// asks for no format is the same kind of file whichever vendor spoke it. It is also
// Cartesia's own mp3 default: 44100 Hz at 128 kbps
// (https://docs.cartesia.ai/api-reference/tts/bytes, read 2026-10-19).
const DefaultRecordingFormat = "mp3_44100_128"

// recorderURL is the endpoint that returns the whole file
// (https://docs.cartesia.ai/api-reference/tts/bytes, read 2026-10-19).
const recorderURL = "https://api.cartesia.ai/tts/bytes"

// recorderTimeout bounds one recording. A chapter is a minute or two of request.
const recorderTimeout = 10 * time.Minute

// ErrFormat says a recording asked for a file Cartesia cannot produce. The recorder is
// refused when built rather than handing back a different file than was asked for.
var ErrFormat = errors.New("cartesia: format is not one Cartesia records")

// recordingSampleRates and recordingBitRates are the values output_format takes on
// /tts/bytes (https://docs.cartesia.ai/api-reference/tts/bytes, read 2026-10-19).
var (
	recordingSampleRates = []int{8_000, 16_000, 22_050, 24_000, 44_100, 48_000}
	recordingBitRates    = []int{32_000, 64_000, 96_000, 128_000, 192_000}
)

// recordingEncodings maps a format name's codec to Cartesia's container and encoding.
// The names are ElevenLabs' codec_samplerate[_kbps] (tts.Recording.Format,
// https://elevenlabs.io/docs/api-reference/text-to-speech/convert, read 2026-10-19), whose
// pcm is raw little-endian 16-bit: the router reads it so (elevenlabs.go emits pcm_* through
// audio.FromBytes). Cartesia's containers are raw, wav and mp3, and the encodings of the
// first two pcm_s16le, pcm_f32le, pcm_mulaw and pcm_alaw
// (https://docs.cartesia.ai/api-reference/tts/bytes, read 2026-10-19).
var recordingEncodings = map[string]fileFormat{
	"pcm":  {Container: "raw", Encoding: "pcm_s16le"},
	"wav":  {Container: "wav", Encoding: "pcm_s16le"},
	"ulaw": {Container: "raw", Encoding: "pcm_mulaw"},
	"alaw": {Container: "raw", Encoding: "pcm_alaw"},
	"mp3":  {Container: "mp3"},
}

// RecorderOptions configures the batch voice. APIKey falls back to CARTESIA_API_KEY and
// VoiceID to CARTESIA_VOICE_ID.
type RecorderOptions struct {
	APIKey  string
	VoiceID string
	Model   string
	// Language is a base ISO code such as "en", for a recording that names none. Empty lets
	// the model infer it.
	Language string
	// Format is the format the recording will be asked for, empty for the default. It is
	// only checked here, so a format Cartesia cannot produce is refused when the recorder
	// is built; what is sent comes from the recording.
	Format string
	// BaseURL overrides the endpoint, for a proxy or a test server.
	BaseURL string
	Logger  *slog.Logger
}

// Recorder speaks whole texts into single files on /tts/bytes, where the rest of this
// package streams sentences over a socket.
type Recorder struct {
	options RecorderOptions
	client  *http.Client
	logger  *slog.Logger
}

// NewRecorder validates the options and returns a recorder.
func NewRecorder(options RecorderOptions) (*Recorder, error) {
	keys := keypool.FromEnv(options.APIKey, "CARTESIA_API_KEY")
	if keys.Len() == 0 {
		return nil, stack.Wrap(errors.New("cartesia: api key is required (set CARTESIA_API_KEY)"))
	}
	options.APIKey = keys.Peek()
	if options.VoiceID == "" {
		options.VoiceID = os.Getenv("CARTESIA_VOICE_ID")
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
		client:  keys.WrapClient(&http.Client{Timeout: recorderTimeout}, keypool.Bearer),
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

// Voice is the speaker of a recording that names none, which makes the recorder a
// tts.Voiced: the recording router reads it to report the voice that spoke.
func (r *Recorder) Voice() string { return r.options.VoiceID }

// Record speaks the text and returns the file.
func (r *Recorder) Record(ctx context.Context, recording tts.Recording) (tts.Recorded, error) {
	if err := recording.Validate(); err != nil {
		return tts.Recorded{}, stack.Wrap(err)
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
		ModelID:      r.options.Model,
		Transcript:   recording.Text,
		Voice:        recorderVoice{ID: voice},
		Language:     language,
		OutputFormat: format,
	})
	if err != nil {
		return tts.Recorded{}, stack.Wrap(fmt.Errorf("cartesia: encode request: %w", err))
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, r.options.BaseURL, bytes.NewReader(body))
	if err != nil {
		return tts.Recorded{}, stack.Wrap(fmt.Errorf("cartesia: build request: %w", err))
	}
	request.Header.Set("Authorization", "Bearer "+r.options.APIKey)
	request.Header.Set("Cartesia-Version", apiVersion)
	request.Header.Set("Content-Type", "application/json")

	response, err := r.client.Do(request)
	if err != nil {
		return tts.Recorded{}, stack.Wrap(fmt.Errorf("cartesia: record: %w", err))
	}
	defer response.Body.Close()

	audio, err := io.ReadAll(response.Body)
	if err != nil {
		return tts.Recorded{}, stack.Wrap(fmt.Errorf("cartesia: read audio: %w", err))
	}
	if response.StatusCode != http.StatusOK {
		return tts.Recorded{}, stack.Wrap(fmt.Errorf("cartesia: record: %s: %s",
			response.Status, strings.TrimSpace(string(audio))))
	}

	return tts.Recorded{
		Audio:      audio,
		Format:     name,
		Characters: int64(len([]rune(recording.Text))),
	}, nil
}

// recorderRequest is the body of one synthesis on /tts/bytes
// (https://docs.cartesia.ai/api-reference/tts/bytes, read 2026-10-19). Speed, volume and
// emotion would go in generation_config, which this does not send: the model does not
// declare them in router.yaml, so the router refuses a recording that asks for one.
type recorderRequest struct {
	ModelID      string        `json:"model_id"`
	Transcript   string        `json:"transcript"`
	Voice        recorderVoice `json:"voice"`
	Language     string        `json:"language,omitempty"`
	OutputFormat fileFormat    `json:"output_format"`
}

type recorderVoice struct {
	ID string `json:"id"`
}

// fileFormat is output_format for a file: mp3 takes a bit rate and no encoding, raw and
// wav an encoding and no bit rate.
type fileFormat struct {
	Container  string `json:"container"`
	Encoding   string `json:"encoding,omitempty"`
	SampleRate int    `json:"sample_rate"`
	BitRate    int    `json:"bit_rate,omitempty"`
}

// recordingFormat reads a format name into Cartesia's output_format, refusing one it has no
// container, encoding, sample rate or bit rate for. Empty is the default.
func recordingFormat(name string) (fileFormat, error) {
	if name == "" {
		name = DefaultRecordingFormat
	}
	refused := stack.Wrap(fmt.Errorf("%w: %s", ErrFormat, name))
	parts := strings.Split(name, "_")
	format, known := recordingEncodings[parts[0]]
	if !known || len(parts) < 2 {
		return fileFormat{}, refused
	}
	rate, err := strconv.Atoi(parts[1])
	if err != nil || !slices.Contains(recordingSampleRates, rate) {
		return fileFormat{}, refused
	}
	format.SampleRate = rate
	// Only mp3 names a bit rate, in kbps.
	if format.Container != "mp3" {
		if len(parts) != 2 {
			return fileFormat{}, refused
		}
		return format, nil
	}
	if len(parts) != 3 {
		return fileFormat{}, refused
	}
	kbps, err := strconv.Atoi(parts[2])
	if err != nil || !slices.Contains(recordingBitRates, kbps*1000) {
		return fileFormat{}, refused
	}
	format.BitRate = kbps * 1000
	return format, nil
}
