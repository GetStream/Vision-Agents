package deepgram

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
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// PrerecordedModel is the batch model. It is not one of the Flux models: Flux exists to
// decide where a turn ended in a live conversation, which a recording does not need.
const PrerecordedModel = "nova-3"

// prerecordedURL is the batch listen endpoint.
const prerecordedURL = "https://api.deepgram.com/v1/listen"

// prerecordedTimeout bounds one transcription. Deepgram transcribes far faster than
// real time, but an hour of audio is still minutes of request, so this is generous.
const prerecordedTimeout = 30 * time.Minute

// PrerecordedOptions configures the batch transcriber. Only APIKey is required, and it
// falls back to DEEPGRAM_API_KEY.
type PrerecordedOptions struct {
	APIKey string
	Model  string
	// BaseURL overrides the endpoint, for a proxy or a test server.
	BaseURL string
	Logger  *slog.Logger
}

// Prerecorded transcribes whole recordings through Deepgram's batch endpoint.
//
// Flux, which the rest of this package implements, is the live path: a model whose job is
// deciding where a turn ended while somebody is still talking. This is a plain HTTP
// request against a different model, and it is both cheaper and more accurate because it
// has the whole recording in front of it.
type Prerecorded struct {
	options PrerecordedOptions
	client  *http.Client
	logger  *slog.Logger
}

// NewPrerecorded validates the options and returns a transcriber.
func NewPrerecorded(options PrerecordedOptions) (*Prerecorded, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv("DEEPGRAM_API_KEY")
	}
	if options.APIKey == "" {
		return nil, errors.New("deepgram: api key is required (set DEEPGRAM_API_KEY)")
	}
	if options.Model == "" {
		options.Model = PrerecordedModel
	}
	if options.BaseURL == "" {
		options.BaseURL = prerecordedURL
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &Prerecorded{
		options: options,
		client:  &http.Client{Timeout: prerecordedTimeout},
		logger:  logger.With("provider", ProviderName, "model", options.Model),
	}, nil
}

// Start is what routing.Provider asks for. There is no connection to open.
func (p *Prerecorded) Start(context.Context) error { return nil }

// Close is what routing.Provider asks for.
func (p *Prerecorded) Close() error { return nil }

// Provider is the stable provider name used in stats.
func (p *Prerecorded) Provider() string { return ProviderName }

// Model is the model identifier used in stats.
func (p *Prerecorded) Model() string { return p.options.Model }

// Transcribe sends the recording and reads back the whole transcript.
func (p *Prerecorded) Transcribe(ctx context.Context, recording stt.Recording) (stt.Transcription, error) {
	if err := recording.Validate(); err != nil {
		return stt.Transcription{}, err
	}

	request, err := p.request(ctx, recording)
	if err != nil {
		return stt.Transcription{}, err
	}

	response, err := p.client.Do(request)
	if err != nil {
		return stt.Transcription{}, fmt.Errorf("deepgram: transcribe: %w", err)
	}
	defer response.Body.Close()

	body, err := io.ReadAll(response.Body)
	if err != nil {
		return stt.Transcription{}, fmt.Errorf("deepgram: read transcript: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		return stt.Transcription{}, fmt.Errorf("deepgram: transcribe: %s: %s",
			response.Status, strings.TrimSpace(string(body)))
	}

	var decoded prerecordedResponse
	if err := json.Unmarshal(body, &decoded); err != nil {
		return stt.Transcription{}, fmt.Errorf("deepgram: decode transcript: %w", err)
	}
	return decoded.transcription(), nil
}

// request builds the call. A URL is handed over as JSON for Deepgram to fetch itself;
// audio is sent as the body, with no content type, since the API sniffs the container and
// naming the wrong one is worse than naming none.
func (p *Prerecorded) request(ctx context.Context, recording stt.Recording) (*http.Request, error) {
	endpoint := p.options.BaseURL + "?" + p.query(recording).Encode()

	var body io.Reader
	contentType := ""
	if recording.URL != "" {
		encoded, err := json.Marshal(map[string]string{"url": recording.URL})
		if err != nil {
			return nil, fmt.Errorf("deepgram: encode source: %w", err)
		}
		body = bytes.NewReader(encoded)
		contentType = "application/json"
	} else {
		body = bytes.NewReader(recording.Audio)
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, body)
	if err != nil {
		return nil, fmt.Errorf("deepgram: build request: %w", err)
	}
	request.Header.Set("Authorization", "Token "+p.options.APIKey)
	if contentType != "" {
		request.Header.Set("Content-Type", contentType)
	}
	return request, nil
}

// query turns the recording's terms into the parameters Deepgram names them by.
func (p *Prerecorded) query(recording stt.Recording) url.Values {
	query := url.Values{}
	query.Set("model", p.options.Model)

	if len(recording.Languages) > 0 {
		query.Set("language", recording.Languages[0])
	} else {
		// Nova-3 identifies the language itself when told to, which is what a caller
		// who did not say has asked for.
		query.Set("detect_language", "true")
	}
	if recording.Format {
		query.Set("smart_format", "true")
	}
	if recording.Diarize {
		query.Set("diarize", "true")
	}
	if recording.Redact {
		query.Set("redact", "pii")
	}
	if recording.Summary {
		query.Set("summarize", "v2")
	}
	if recording.Entities {
		query.Set("detect_entities", "true")
	}
	if recording.Channels > 1 {
		query.Set("multichannel", "true")
		query.Set("channels", strconv.Itoa(recording.Channels))
	}
	for _, term := range recording.Keyterms {
		query.Add("keyterm", term)
	}
	return query
}

// prerecordedResponse is the part of Deepgram's batch answer worth reading. A batch
// transcript arrives as one alternative per channel, with the words under it.
type prerecordedResponse struct {
	Metadata struct {
		// Duration is in seconds, fractional.
		Duration float64 `json:"duration"`
	} `json:"metadata"`
	Results struct {
		Channels []struct {
			DetectedLanguage string `json:"detected_language"`
			Alternatives     []struct {
				Transcript string `json:"transcript"`
				Words      []struct {
					Word           string  `json:"word"`
					PunctuatedWord string  `json:"punctuated_word"`
					Start          float64 `json:"start"`
					End            float64 `json:"end"`
					Confidence     float64 `json:"confidence"`
					Speaker        *int    `json:"speaker"`
				} `json:"words"`
				Entities []struct {
					Label     string  `json:"label"`
					Value     string  `json:"value"`
					StartWord int     `json:"start_word"`
					EndWord   int     `json:"end_word"`
					StartTime float64 `json:"start_time"`
					EndTime   float64 `json:"end_time"`
				} `json:"entities"`
			} `json:"alternatives"`
		} `json:"channels"`
		Summary struct {
			Short string `json:"short"`
		} `json:"summary"`
	} `json:"results"`
}

// transcription reads the answer into the shape the router speaks. A response with no
// channels or no alternatives is a recording with nothing audible in it, which is an empty
// transcript rather than an error.
func (r prerecordedResponse) transcription() stt.Transcription {
	transcription := stt.Transcription{
		Summary:         strings.TrimSpace(r.Results.Summary.Short),
		AudioDurationMs: int64(r.Metadata.Duration * 1000),
	}
	if len(r.Results.Channels) == 0 || len(r.Results.Channels[0].Alternatives) == 0 {
		return transcription
	}

	transcription.Language = r.Results.Channels[0].DetectedLanguage
	best := r.Results.Channels[0].Alternatives[0]
	transcription.Text = best.Transcript

	seen := map[string]struct{}{}
	for _, word := range best.Words {
		text := word.PunctuatedWord
		if text == "" {
			text = word.Word
		}
		speaker := ""
		if word.Speaker != nil {
			speaker = "speaker_" + strconv.Itoa(*word.Speaker)
			if _, known := seen[speaker]; !known {
				seen[speaker] = struct{}{}
				transcription.Speakers = append(transcription.Speakers, speaker)
			}
		}
		transcription.Words = append(transcription.Words, stt.Word{
			Text:       text,
			StartMs:    int64(word.Start * 1000),
			EndMs:      int64(word.End * 1000),
			Confidence: word.Confidence,
			Speaker:    speaker,
		})
	}
	for _, entity := range best.Entities {
		transcription.Entities = append(transcription.Entities, stt.Entity{
			Type:    entity.Label,
			Text:    entity.Value,
			StartMs: int64(entity.StartTime * 1000),
			EndMs:   int64(entity.EndTime * 1000),
		})
	}
	return transcription
}
