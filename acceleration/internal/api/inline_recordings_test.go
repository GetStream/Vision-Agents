package api

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
	"github.com/stretchr/testify/require"
)

type inlineProvider struct{}

func (inlineProvider) Start(context.Context) error { return nil }
func (inlineProvider) Close() error                { return nil }
func (inlineProvider) Provider() string            { return "inline-test" }
func (inlineProvider) Model() string               { return "model" }
func (inlineProvider) Record(_ context.Context, r tts.Recording) (tts.Recorded, error) {
	return tts.Recorded{Audio: []byte("audio"), Format: "wav", Characters: int64(len(r.Text))}, nil
}
func (inlineProvider) Transcribe(_ context.Context, r stt.Recording) (stt.Transcription, error) {
	return stt.Transcription{Text: "editable transcript", Language: "en"}, nil
}
func TestInlineAudioRequiresNoDatabase(t *testing.T) {
	logger := slog.New(slog.DiscardHandler)
	config := routing.ModalityConfig{Providers: []routing.ProviderConfig{{Provider: "inline-test", Model: "model", Languages: []string{"en"}}}}
	voices := ttsrouter.NewRecorderRegistry()
	voices.Register("inline-test", func(routing.Spec) (tts.Recorder, error) { return inlineProvider{}, nil })
	speech, err := ttsrouter.NewRecordings(ttsrouter.Options{Config: config, Recorders: voices, Logger: logger})
	require.NoError(t, err)
	defer speech.Close()
	ears := sttrouter.NewTranscriberRegistry()
	ears.Register("inline-test", func(routing.Spec) (stt.Transcriber, error) { return inlineProvider{}, nil })
	transcripts, err := sttrouter.NewRecordings(sttrouter.Options{Config: config, Transcribers: ears, Logger: logger})
	require.NoError(t, err)
	defer transcripts.Close()
	api, err := NewServer(Options{Routers: map[routing.Modality]routing.Inspector{routing.STT: transcripts, routing.TTS: speech}, Streams: &Streams{Speech: speech, Transcriptions: transcripts}, Logger: logger})
	require.NoError(t, err)
	send := func(path, body string) *httptest.ResponseRecorder {
		r := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
		r.Header.Set("Content-Type", "application/json")
		r.Header.Set(CustomerHeader, "owner")
		w := httptest.NewRecorder()
		api.Handler().ServeHTTP(w, r)
		return w
	}
	w := send("/v1/tts/recordings", `{"inline":true,"text":"hello","options":{"target":"inline-test/model"}}`)
	require.Equal(t, 202, w.Code, w.Body.String())
	var spoken Speech
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &spoken))
	require.Equal(t, RecordingStatusCompleted, spoken.Status)
	require.Equal(t, []byte("audio"), *spoken.Audio)
	w = send("/v1/stt/recordings", `{"inline":true,"source":{"audio":"YXVkaW8="},"options":{"target":"inline-test/model"}}`)
	require.Equal(t, 202, w.Code, w.Body.String())
	var transcript Transcription
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &transcript))
	require.Equal(t, "editable transcript", *transcript.Text)
	w = send("/v1/tts/recordings", `{"text":"hello"}`)
	require.Equal(t, 400, w.Code)
	require.Contains(t, w.Body.String(), "no database")
	w = send("/v1/tts/recordings", `{"inline":true,"text":"hello","callback":"https://example.com"}`)
	require.Equal(t, 400, w.Code)
	require.Contains(t, w.Body.String(), "callback")
	w = send("/v1/stt/recordings", `{"inline":true,"source":{"url":"https://example.com/audio.wav"}}`)
	require.Equal(t, 400, w.Code)
	require.Contains(t, w.Body.String(), "no URL")
}
