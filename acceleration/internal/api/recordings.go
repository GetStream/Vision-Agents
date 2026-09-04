package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// noRecordings is what the recording paths say on a deployment that does not run them.
// They are jobs, so they need somewhere to keep one as well as something to route it to.
const noRecordings = "this deployment does not run recordings"

// noRecordingStore is what they say without a database. A job whose result nobody could
// come back for is worse than a refusal.
const noRecordingStore = "recordings are not available: no database configured"

// recordingDeadline bounds one job. Transcription runs far faster than real time, but a
// feature-length recording is still minutes of work, and a job that hangs is a row that
// stays queued forever.
const recordingDeadline = 45 * time.Minute

// callbackTimeout bounds telling a caller their job is done.
const callbackTimeout = 30 * time.Second

// TranscribeRecording accepts a recording and transcribes it off the live path.
func (s *Server) TranscribeRecording(ctx context.Context, request TranscribeRecordingRequestObject) (TranscribeRecordingResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return TranscribeRecording401JSONResponse{missingCustomer()}, nil
	}
	if s.streams == nil || s.streams.Transcriptions == nil {
		return TranscribeRecording404JSONResponse{NotFoundJSONResponse{Error: noRecordings}}, nil
	}
	if s.store == nil {
		return TranscribeRecording400JSONResponse{badRequest(noRecordingStore)}, nil
	}
	if request.Body == nil {
		return TranscribeRecording400JSONResponse{badRequest("a request body is required")}, nil
	}

	config, err := s.routerOptions(ctx, customerID, value(request.Body.ConfigId))
	if err != nil {
		return TranscribeRecording400JSONResponse{badRequest(err.Error())}, nil
	}
	held := config.STT.Merge(sttOptionsOf(request.Body.Options))
	if held.Target == "" {
		held.Target = recordedTarget(held.Languages)
	}

	source := stt.Recording{
		URL:         value(request.Body.Source.Url),
		Audio:       value(request.Body.Source.Audio),
		Languages:   held.Languages,
		Diarize:     truthy(held.Diarize) || held.MaxSpeakers != nil,
		MaxSpeakers: count(held.MaxSpeakers),
		Words:       truthy(held.Words) || subtitled(held.Output),
		Format:      truthy(held.Format),
		Redact:      truthy(held.Redact),
		Summary:     truthy(held.Summary),
		Entities:    truthy(held.Entities),
		Keyterms:    held.Keyterms,
		Channels:    count(held.Channels),
	}
	if err := source.Validate(); err != nil {
		return TranscribeRecording400JSONResponse{badRequest(err.Error())}, nil
	}
	if _, err := stt.Subtitles(stt.Transcription{}, held.Output); err != nil {
		return TranscribeRecording400JSONResponse{badRequest(err.Error())}, nil
	}

	tags := tagsUnder(config, request.Body.Tags)
	if err := tags.Validate(); err != nil {
		return TranscribeRecording400JSONResponse{badRequest(err.Error())}, nil
	}

	job := store.Recording{
		CustomerID: customerID,
		Modality:   string(Stt),
		Source:     source.URL,
		STT:        held,
		Callback:   value(request.Body.Callback),
		Tags:       tags,
	}
	if err := s.store.CreateRecording(ctx, &job); err != nil {
		return nil, err
	}

	// The job outlives the request that asked for it, so it runs under a context of its
	// own: a caller that has been handed an id and hung up is still owed a transcript.
	go s.transcribe(job, sttrouter.Recording{
		CustomerID: customerID,
		Tags:       tags,
		Options:    held,
		Source:     source,
	})

	return TranscribeRecording202JSONResponse(transcriptionOf(job)), nil
}

// GetTranscription returns one transcription job, and its transcript once it has one.
func (s *Server) GetTranscription(ctx context.Context, request GetTranscriptionRequestObject) (GetTranscriptionResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetTranscription401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return GetTranscription400JSONResponse{badRequest(noRecordingStore)}, nil
	}

	job, err := s.store.Recording(ctx, customerID, request.Id)
	if err != nil || job.Modality != string(Stt) {
		return GetTranscription404JSONResponse{NotFoundJSONResponse{Error: "no such transcription"}}, nil
	}
	return GetTranscription200JSONResponse(transcriptionOf(job)), nil
}

// RecordSpeech accepts a text and speaks the whole of it into one file.
func (s *Server) RecordSpeech(ctx context.Context, request RecordSpeechRequestObject) (RecordSpeechResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return RecordSpeech401JSONResponse{missingCustomer()}, nil
	}
	if s.streams == nil || s.streams.Speech == nil {
		return RecordSpeech404JSONResponse{NotFoundJSONResponse{Error: noRecordings}}, nil
	}
	if s.store == nil {
		return RecordSpeech400JSONResponse{badRequest(noRecordingStore)}, nil
	}
	if request.Body == nil {
		return RecordSpeech400JSONResponse{badRequest("a request body is required")}, nil
	}
	if strings.TrimSpace(request.Body.Text) == "" {
		return RecordSpeech400JSONResponse{badRequest("there is nothing to say")}, nil
	}

	config, err := s.routerOptions(ctx, customerID, value(request.Body.ConfigId))
	if err != nil {
		return RecordSpeech400JSONResponse{badRequest(err.Error())}, nil
	}
	held := config.TTS.Merge(ttsOptionsOf(request.Body.Options))
	if held.Target == "" {
		held.Target = recordedTarget(held.Languages)
	}

	tags := tagsUnder(config, request.Body.Tags)
	if err := tags.Validate(); err != nil {
		return RecordSpeech400JSONResponse{badRequest(err.Error())}, nil
	}

	job := store.Recording{
		CustomerID: customerID,
		Modality:   string(Tts),
		Text:       request.Body.Text,
		TTS:        held,
		Callback:   value(request.Body.Callback),
		Tags:       tags,
	}
	if err := s.store.CreateRecording(ctx, &job); err != nil {
		return nil, err
	}

	go s.record(job, ttsrouter.Recording{
		CustomerID: customerID,
		Tags:       tags,
		Options:    held,
		Text:       request.Body.Text,
	})

	return RecordSpeech202JSONResponse(speechOf(job)), nil
}

// GetSpeech returns one speech job, and its audio once it has some.
func (s *Server) GetSpeech(ctx context.Context, request GetSpeechRequestObject) (GetSpeechResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetSpeech401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return GetSpeech400JSONResponse{badRequest(noRecordingStore)}, nil
	}

	job, err := s.store.Recording(ctx, customerID, request.Id)
	if err != nil || job.Modality != string(Tts) {
		return GetSpeech404JSONResponse{NotFoundJSONResponse{Error: "no such speech job"}}, nil
	}
	return GetSpeech200JSONResponse(speechOf(job)), nil
}

// transcribe runs one transcription job to its end and writes down what happened.
func (s *Server) transcribe(job store.Recording, recording sttrouter.Recording) {
	ctx, cancel := context.WithTimeout(context.Background(), recordingDeadline)
	defer cancel()

	transcription, config, err := s.streams.Transcriptions.Transcribe(ctx, recording)
	if err != nil {
		s.finished(ctx, job, config.Provider, config.Model, nil, err)
		return
	}

	subtitles, err := stt.Subtitles(transcription, job.STT.Output)
	if err != nil {
		s.finished(ctx, job, config.Provider, config.Model, nil, err)
		return
	}

	encoded, err := json.Marshal(transcriptResult{
		Language:        transcription.Language,
		Text:            transcription.Text,
		Words:           wordsOf(transcription.Words),
		Speakers:        transcription.Speakers,
		Subtitles:       subtitles,
		Summary:         transcription.Summary,
		Entities:        entitiesOf(transcription.Entities),
		AudioDurationMs: transcription.AudioDurationMs,
	})
	if err != nil {
		s.finished(ctx, job, config.Provider, config.Model, nil, err)
		return
	}
	s.finished(ctx, job, config.Provider, config.Model, encoded, nil)
}

// record runs one speech job to its end and writes down what happened.
func (s *Server) record(job store.Recording, recording ttsrouter.Recording) {
	ctx, cancel := context.WithTimeout(context.Background(), recordingDeadline)
	defer cancel()

	recorded, config, err := s.streams.Speech.Record(ctx, recording)
	if err != nil {
		s.finished(ctx, job, config.Provider, config.Model, nil, err)
		return
	}

	encoded, err := json.Marshal(speechResult{
		Audio:           recorded.Audio,
		Format:          recorded.Format,
		AudioDurationMs: recorded.AudioDurationMs,
		Characters:      recorded.Characters,
	})
	if err != nil {
		s.finished(ctx, job, config.Provider, config.Model, nil, err)
		return
	}
	s.finished(ctx, job, config.Provider, config.Model, encoded, nil)
}

// finished writes the result down and, if the caller asked to be told rather than to
// poll, tells them.
func (s *Server) finished(ctx context.Context, job store.Recording, provider, model string, result json.RawMessage, failure error) {
	if failure != nil {
		s.logger.Error("a recording failed", "recording", job.ID, "modality", job.Modality, "error", failure)
	}
	if err := s.store.FinishRecording(ctx, job.ID, provider, model, result, failure); err != nil {
		s.logger.Error("could not write down a finished recording", "recording", job.ID, "error", err)
		return
	}
	if job.Callback == "" {
		return
	}

	finished, err := s.store.Recording(ctx, job.CustomerID, job.ID)
	if err != nil {
		s.logger.Error("could not read back a finished recording", "recording", job.ID, "error", err)
		return
	}
	var body any = transcriptionOf(finished)
	if finished.Modality == string(Tts) {
		body = speechOf(finished)
	}
	s.callBack(ctx, job.Callback, body)
}

// callBack tells a caller their job is done. A callback that cannot be delivered is
// logged and let go: the result is written down either way, so the caller can still ask
// for it, and retrying somebody else's endpoint from here would be a queue of its own.
func (s *Server) callBack(ctx context.Context, url string, body any) {
	encoded, err := json.Marshal(body)
	if err != nil {
		s.logger.Error("could not encode a recording callback", "url", url, "error", err)
		return
	}

	ctx, cancel := context.WithTimeout(ctx, callbackTimeout)
	defer cancel()

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(encoded))
	if err != nil {
		s.logger.Error("could not build a recording callback", "url", url, "error", err)
		return
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := http.DefaultClient.Do(request)
	if err != nil {
		s.logger.Error("could not deliver a recording callback", "url", url, "error", err)
		return
	}
	defer response.Body.Close()
	if response.StatusCode >= http.StatusBadRequest {
		s.logger.Error("a recording callback was refused", "url", url, "status", response.Status)
	}
}

// transcriptResult is a finished transcript as it is stored. It has its own field names
// rather than the provider's types so that what is in the column stays readable when the
// contract behind it moves on.
type transcriptResult struct {
	Language        string             `json:"language,omitempty"`
	Text            string             `json:"text,omitempty"`
	Words           []TranscriptWord   `json:"words,omitempty"`
	Speakers        []string           `json:"speakers,omitempty"`
	Subtitles       string             `json:"subtitles,omitempty"`
	Summary         string             `json:"summary,omitempty"`
	Entities        []TranscriptEntity `json:"entities,omitempty"`
	AudioDurationMs int64              `json:"audio_duration_ms,omitempty"`
}

// speechResult is finished audio as it is stored.
type speechResult struct {
	Audio           []byte `json:"audio,omitempty"`
	URL             string `json:"url,omitempty"`
	Format          string `json:"format,omitempty"`
	AudioDurationMs int64  `json:"audio_duration_ms,omitempty"`
	Characters      int64  `json:"characters,omitempty"`
}

// transcriptionOf renders a transcription job for the wire, result and all when it has
// one.
func transcriptionOf(job store.Recording) Transcription {
	rendered := Transcription{
		Id:        job.ID,
		Status:    RecordingStatus(job.Status),
		Provider:  optional(job.Provider),
		Model:     optional(job.Model),
		Error:     optional(job.Error),
		CreatedAt: job.CreatedAt,
		UpdatedAt: job.UpdatedAt,
	}
	rendered.CompletedAt = job.CompletedAt

	var result transcriptResult
	if len(job.Result) == 0 || json.Unmarshal(job.Result, &result) != nil {
		return rendered
	}
	rendered.Language = optional(result.Language)
	rendered.Text = optional(result.Text)
	rendered.Words = list(result.Words)
	rendered.Speakers = list(result.Speakers)
	rendered.Subtitles = optional(result.Subtitles)
	rendered.Summary = optional(result.Summary)
	rendered.Entities = list(result.Entities)
	if result.AudioDurationMs > 0 {
		duration := result.AudioDurationMs
		rendered.AudioDurationMs = &duration
	}
	return rendered
}

// speechOf renders a speech job for the wire, audio and all when it has some.
func speechOf(job store.Recording) Speech {
	rendered := Speech{
		Id:        job.ID,
		Status:    RecordingStatus(job.Status),
		Provider:  optional(job.Provider),
		Model:     optional(job.Model),
		Error:     optional(job.Error),
		CreatedAt: job.CreatedAt,
		UpdatedAt: job.UpdatedAt,
	}
	rendered.CompletedAt = job.CompletedAt

	var result speechResult
	if len(job.Result) == 0 || json.Unmarshal(job.Result, &result) != nil {
		return rendered
	}
	rendered.Format = optional(result.Format)
	rendered.Url = optional(result.URL)
	if len(result.Audio) > 0 {
		audio := result.Audio
		rendered.Audio = &audio
	}
	if result.AudioDurationMs > 0 {
		duration := result.AudioDurationMs
		rendered.AudioDurationMs = &duration
	}
	if result.Characters > 0 {
		characters := result.Characters
		rendered.Characters = &characters
	}
	return rendered
}

func wordsOf(words []stt.Word) []TranscriptWord {
	rendered := make([]TranscriptWord, 0, len(words))
	for _, word := range words {
		confidence := float32(word.Confidence)
		rendered = append(rendered, TranscriptWord{
			Text:       word.Text,
			StartMs:    word.StartMs,
			EndMs:      word.EndMs,
			Confidence: &confidence,
			Speaker:    optional(word.Speaker),
		})
	}
	return rendered
}

func entitiesOf(entities []stt.Entity) []TranscriptEntity {
	rendered := make([]TranscriptEntity, 0, len(entities))
	for _, entity := range entities {
		start, end := entity.StartMs, entity.EndMs
		rendered = append(rendered, TranscriptEntity{
			Type:    entity.Type,
			Text:    entity.Text,
			StartMs: &start,
			EndMs:   &end,
		})
	}
	return rendered
}

// truthy and count read an option that was not necessarily named, where unset means the
// provider's own behaviour.
func truthy(flag *bool) bool { return flag != nil && *flag }

func count(number *int) int {
	if number == nil {
		return 0
	}
	return *number
}

// subtitled reports whether an output format has to be rendered from timings.
func subtitled(output string) bool {
	return output != "" && output != "json"
}
