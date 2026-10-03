package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/danielgtaylor/huma/v2"
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
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

// transcribeRecording accepts a recording and transcribes it off the live path.
func (s *Server) transcribeRecording(ctx context.Context, request *transcribeRecordingRequest) (*transcribeRecordingResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.streams == nil || s.streams.Transcriptions == nil {
		return nil, huma.Error404NotFound(noRecordings)
	}
	if s.store == nil && (request.Body == nil || !truthy(request.Body.Inline)) {
		return nil, huma.Error400BadRequest(noRecordingStore)
	}
	if request.Body == nil {
		return nil, huma.Error400BadRequest("a request body is required")
	}

	config, err := s.routerOptions(ctx, customerID, value(request.Body.ConfigId))
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	held := config.STT.Merge(sttOptionsOf(request.Body.Options))
	if err := held.Validate(); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
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

		ProfanityFilter: truthy(held.ProfanityFilter),
		FillerWords:     held.Mode == options.ModeVerbatim,
	}
	if err := source.Validate(); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	if _, err := stt.Subtitles(stt.Transcription{}, held.Output); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}

	tags := tagsSent(request.Body.Tags)
	if err := tags.Validate(); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}

	job := store.Recording{
		CustomerID: customerID,
		Modality:   string(Stt),
		Source:     source.URL,
		STT:        held,
		Callback:   value(request.Body.Callback),
		Tags:       tags,
	}
	if truthy(request.Body.Inline) {
		if job.Callback != "" || len(source.Audio) > 8*1024*1024 || source.URL != "" {
			return nil, huma.Error400BadRequest("inline transcription requires at most 8 MiB of audio and no URL or callback")
		}
		ctx, cancel := context.WithTimeout(ctx, 90*time.Second)
		defer cancel()
		transcript, provider, failure := s.streams.Transcriptions.Transcribe(ctx, sttrouter.Recording{CustomerID: customerID, Tags: tags, Options: held, Source: source})
		subtitles, subErr := stt.Subtitles(transcript, held.Output)
		if failure == nil {
			failure = subErr
		}
		raw, _ := json.Marshal(transcriptResult{Text: transcript.Text, Language: transcript.Language, Words: wordsOf(transcript.Words), Speakers: transcript.Speakers, Subtitles: subtitles, Summary: transcript.Summary, Entities: entitiesOf(transcript.Entities), AudioDurationMs: transcript.AudioDurationMs})
		return &transcribeRecordingResponse{Body: transcriptionOf(inlineResult(job, provider.Provider, provider.Model, raw, failure))}, nil
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

	return &transcribeRecordingResponse{Body: transcriptionOf(job)}, nil
}

// getTranscription returns one transcription job, and its transcript once it has one.
func (s *Server) getTranscription(ctx context.Context, request *getTranscriptionRequest) (*getTranscriptionResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noRecordingStore)
	}

	job, err := s.store.Recording(ctx, customerID, request.Id)
	if err != nil || job.Modality != string(Stt) {
		return nil, huma.Error404NotFound("no such transcription")
	}
	return &getTranscriptionResponse{Body: transcriptionOf(job)}, nil
}

// recordSpeech accepts a text and speaks the whole of it into one file.
func (s *Server) recordSpeech(ctx context.Context, request *recordSpeechRequest) (*recordSpeechResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.streams == nil || s.streams.Speech == nil {
		return nil, huma.Error404NotFound(noRecordings)
	}
	if s.store == nil && (request.Body == nil || !truthy(request.Body.Inline)) {
		return nil, huma.Error400BadRequest(noRecordingStore)
	}
	if request.Body == nil {
		return nil, huma.Error400BadRequest("a request body is required")
	}
	if strings.TrimSpace(request.Body.Text) == "" {
		return nil, huma.Error400BadRequest("there is nothing to say")
	}

	config, err := s.routerOptions(ctx, customerID, value(request.Body.ConfigId))
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	held := config.TTS.Merge(ttsOptionsOf(request.Body.Options))
	if held.Target == "" {
		held.Target = recordedTarget(held.Languages)
	}

	tags := tagsSent(request.Body.Tags)
	if err := tags.Validate(); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}

	job := store.Recording{
		CustomerID: customerID,
		Modality:   string(Tts),
		Text:       request.Body.Text,
		TTS:        held,
		Callback:   value(request.Body.Callback),
		Tags:       tags,
	}
	if truthy(request.Body.Inline) {
		if job.Callback != "" || utf8.RuneCountInString(job.Text) > 16000 {
			return nil, huma.Error400BadRequest("inline speech requires at most 16000 characters and no callback")
		}
		ctx, cancel := context.WithTimeout(ctx, 90*time.Second)
		defer cancel()
		audio, provider, failure := s.streams.Speech.Record(ctx, ttsrouter.Recording{CustomerID: customerID, Tags: tags, Options: held, Text: job.Text})
		raw, _ := json.Marshal(speechResult{Audio: audio.Audio, Format: audio.Format, Characters: audio.Characters, AudioDurationMs: audio.AudioDurationMs})
		return &recordSpeechResponse{Body: speechOf(inlineResult(job, provider.Provider, provider.Model, raw, failure))}, nil
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

	return &recordSpeechResponse{Body: speechOf(job)}, nil
}

// getSpeech returns one speech job, and its audio once it has some.
func (s *Server) getSpeech(ctx context.Context, request *getSpeechRequest) (*getSpeechResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noRecordingStore)
	}

	job, err := s.store.Recording(ctx, customerID, request.Id)
	if err != nil || job.Modality != string(Tts) {
		return nil, huma.Error404NotFound("no such speech job")
	}
	return &getSpeechResponse{Body: speechOf(job)}, nil
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

// inlineResult exists only for this HTTP response; no recording, audio or transcript is stored.
func inlineResult(job store.Recording, provider, model string, result json.RawMessage, failure error) store.Recording {
	now := time.Now().UTC()
	job.ID = uuid.NewString()
	job.Provider = provider
	job.Model = model
	job.Result = result
	job.Status = "completed"
	job.CreatedAt = now
	job.UpdatedAt = now
	job.CompletedAt = &now
	if failure != nil {
		job.Status = "failed"
		job.Error = failure.Error()
	}
	return job
}

// registerRecordings declares the operations served in recordings.go.
func (s *Server) registerRecordings(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "transcribeRecording",
		Method:      http.MethodPost,
		Path:        "/v1/stt/recordings",
		Summary:     "Transcribe a recording, off the live path",
		Description: "The non-realtime half of speech-to-text: a whole recording in, a whole transcript out. " +
			"It is a job rather than a response because an hour of audio takes minutes to " +
			"transcribe, so this returns immediately with an id to poll, or calls a callback when it " +
			"is done.\n" +
			"Routing works as it does everywhere else, except that the candidates are the providers " +
			"registered as not realtime - the batch APIs, which are cheaper and more accurate than " +
			"the same vendor's streaming model.",
		DefaultStatus: http.StatusAccepted,
		Responses: map[string]*huma.Response{
			"202": {Description: "The job was accepted"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.transcribeRecording)
	huma.Register(api, huma.Operation{
		OperationID: "getTranscription",
		Method:      http.MethodGet,
		Path:        "/v1/stt/recordings/{id}",
		Summary:     "One transcription job, and its transcript once it has one",
		Responses: map[string]*huma.Response{
			"200": {Description: "The job as it now stands"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getTranscription)
	huma.Register(api, huma.Operation{
		OperationID: "recordSpeech",
		Method:      http.MethodPost,
		Path:        "/v1/tts/recordings",
		Summary:     "Speak a whole text into one audio file, off the live path",
		Description: "The non-realtime half of text-to-speech: a chapter in, a file out. A job for the same " +
			"reason transcription is - an audiobook is not a conversation, and nothing is waiting to " +
			"hear the first chunk.",
		DefaultStatus: http.StatusAccepted,
		Responses: map[string]*huma.Response{
			"202": {Description: "The job was accepted"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.recordSpeech)
	huma.Register(api, huma.Operation{
		OperationID: "getSpeech",
		Method:      http.MethodGet,
		Path:        "/v1/tts/recordings/{id}",
		Summary:     "One speech job, and its audio once it has some",
		Responses: map[string]*huma.Response{
			"200": {Description: "The job as it now stands"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getSpeech)
}

type transcribeRecordingRequest struct {
	Body *TranscriptionRequest `required:"true"`
}

type transcribeRecordingResponse struct {
	Body Transcription
}

type getTranscriptionRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type getTranscriptionResponse struct {
	Body Transcription
}

type recordSpeechRequest struct {
	Body *SpeechRequest `required:"true"`
}

type recordSpeechResponse struct {
	Body Speech
}

type getSpeechRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type getSpeechResponse struct {
	Body Speech
}

// RecordingStatus Where a job has got to. A failed job carries the reason in `error`, and a completed one carries its result.
type RecordingStatus string

// Defines values for RecordingStatus.
const (
	RecordingStatusCompleted RecordingStatus = "completed"
	RecordingStatusFailed    RecordingStatus = "failed"
	RecordingStatusQueued    RecordingStatus = "queued"
	RecordingStatusRunning   RecordingStatus = "running"
)

// Valid indicates whether the value is a known member of the RecordingStatus enum.
func (e RecordingStatus) Valid() bool {
	switch e {
	case RecordingStatusCompleted:
		return true
	case RecordingStatusFailed:
		return true
	case RecordingStatusQueued:
		return true
	case RecordingStatusRunning:
		return true
	default:
		return false
	}
}

func (RecordingStatus) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "RecordingStatus", "Where a job has got to. A failed job carries the reason in `error`, and a completed one carries its result.", "queued", "running", "completed", "failed")
}

// SpeechRequest is the SpeechRequest schema.
type SpeechRequest struct {
	Callback *string            `json:"callback,omitempty" doc:"A URL the finished job is POSTed to, so a caller does not have to poll."`
	ConfigId *string            `json:"config_id,omitempty" doc:"A stored router config to take the options from. Anything named here as well overrides that one field of it."`
	Inline   *bool              `json:"inline,omitempty" doc:"Complete this short request synchronously without storing a recording job or audio. The 202 response contains the completed or failed result and its ephemeral ID cannot be retrieved later. No database is required. Cancelling the request cancels the work. Incompatible with callback; deadline 90 seconds. Maximum 8 MiB of input audio or 16000 characters of speech text. Default false retains asynchronous stored jobs." default:"false"`
	Options  *TtsOptions        `json:"options,omitempty"`
	Tags     *map[string]string `json:"tags,omitempty"`
	Text     string             `json:"text" doc:"What to say. Whole paragraphs rather than the sentence at a time a socket takes."`
}

// TranscriptEntity Something the recording named, for the providers that pick them out.
type TranscriptEntity struct {
	EndMs   *int64 `json:"end_ms,omitempty"`
	StartMs *int64 `json:"start_ms,omitempty"`
	Text    string `json:"text"`
	Type    string `json:"type" example:"person"`
}

func (*TranscriptEntity) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Something the recording named, for the providers that pick them out."
	return schema
}

// TranscriptWord is the TranscriptWord schema.
type TranscriptWord struct {
	Confidence *float32 `json:"confidence,omitempty"`
	EndMs      int64    `json:"end_ms"`
	Speaker    *string  `json:"speaker,omitempty" doc:"Who said it, when diarization was asked for."`
	StartMs    int64    `json:"start_ms"`
	Text       string   `json:"text"`
}

// Transcription is the Transcription schema.
type Transcription struct {
	AudioDurationMs *int64              `json:"audio_duration_ms,omitempty" doc:"How long the recording was, which is what it was billed on."`
	CompletedAt     *time.Time          `json:"completed_at,omitempty"`
	CreatedAt       time.Time           `json:"created_at"`
	Entities        *[]TranscriptEntity `json:"entities,omitempty"`
	Error           *string             `json:"error,omitempty" doc:"Why the job failed, if it did."`
	Id              string              `json:"id"`
	Language        *string             `json:"language,omitempty" doc:"What was spoken, whether it was asked for or detected."`
	Model           *string             `json:"model,omitempty"`
	Provider        *string             `json:"provider,omitempty"`
	Speakers        *[]string           `json:"speakers,omitempty" doc:"The speakers diarization found, in the order they first spoke."`
	Status          RecordingStatus     `json:"status"`
	Subtitles       *string             `json:"subtitles,omitempty" doc:"The transcript as an SRT or VTT file, when one of those was asked for."`
	Summary         *string             `json:"summary,omitempty"`
	Text            *string             `json:"text,omitempty" doc:"The whole transcript as prose."`
	UpdatedAt       time.Time           `json:"updated_at"`
	Words           *[]TranscriptWord   `json:"words,omitempty" doc:"Present when word-level timestamps were asked for."`
}

// TranscriptionRequest is the TranscriptionRequest schema.
type TranscriptionRequest struct {
	Callback *string            `json:"callback,omitempty" doc:"A URL the finished job is POSTed to, so a caller does not have to poll. The body is the same Transcription this returns."`
	ConfigId *string            `json:"config_id,omitempty" doc:"A stored router config to take the options from. Anything named here as well overrides that one field of it."`
	Inline   *bool              `json:"inline,omitempty" doc:"Complete this short request synchronously without storing a recording job or audio. The 202 response contains the completed or failed result and its ephemeral ID cannot be retrieved later. No database is required. Cancelling the request cancels the work. Incompatible with callback; deadline 90 seconds. Maximum 8 MiB of input audio or 16000 characters of speech text. Default false retains asynchronous stored jobs." default:"false"`
	Options  *SttOptions        `json:"options,omitempty"`
	Source   RecordingSource    `json:"source"`
	Tags     *map[string]string `json:"tags,omitempty" doc:"Cost labels for this job."`
}
