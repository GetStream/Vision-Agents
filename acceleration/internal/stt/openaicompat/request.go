package openaicompat

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

func (p *Transcriber) newRequest(ctx context.Context, recording stt.Recording, settings requestSettings) (*http.Request, error) {
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	filename, contentType, err := audioMetadata(recording)
	if err != nil {
		return nil, err
	}
	header := make(textproto.MIMEHeader)
	header.Set("Content-Disposition", mime.FormatMediaType("form-data", map[string]string{
		"name": "file", "filename": filename,
	}))
	header.Set("Content-Type", contentType)
	file, err := form.CreatePart(header)
	if err != nil {
		return nil, err
	}
	if _, err := file.Write(recording.Audio); err != nil {
		return nil, err
	}
	fields := []multipartField{
		{"model", p.model},
		{"response_format", settings.responseFormat},
	}
	if settings.prompt != "" {
		fields = append(fields, multipartField{"prompt", settings.prompt})
	}
	if settings.language != "" {
		fields = append(fields, multipartField{"language", settings.language})
	}
	if settings.temperature != nil {
		fields = append(fields, multipartField{"temperature", fmt.Sprint(*settings.temperature)})
	}
	for _, value := range settings.languages {
		fields = append(fields, multipartField{"languages[]", value})
	}
	for _, value := range settings.keywords {
		fields = append(fields, multipartField{"keywords[]", value})
	}
	for _, value := range settings.timestampGranularities {
		fields = append(fields, multipartField{"timestamp_granularities[]", value})
	}
	for _, value := range settings.include {
		fields = append(fields, multipartField{"include[]", value})
	}
	for _, value := range settings.knownSpeakerNames {
		fields = append(fields, multipartField{"known_speaker_names[]", value})
	}
	for _, value := range settings.knownSpeakerReferences {
		fields = append(fields, multipartField{"known_speaker_references[]", value})
	}
	if len(settings.chunkingStrategy) > 0 {
		chunkingFields, err := chunkingStrategyFields(settings.chunkingStrategy)
		if err != nil {
			return nil, err
		}
		fields = append(fields, chunkingFields...)
	}
	if settings.stream != nil {
		fields = append(fields, multipartField{"stream", fmt.Sprint(*settings.stream)})
	}
	for _, field := range fields {
		if err := form.WriteField(field.name, field.value); err != nil {
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
	if p.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	request.Header.Set("Content-Type", form.FormDataContentType())
	return request, nil
}

type requestSettings struct {
	responseFormat         string
	prompt                 string
	language               string
	languages              []string
	keywords               []string
	temperature            *float64
	timestampGranularities []string
	include                []string
	chunkingStrategy       json.RawMessage
	knownSpeakerNames      []string
	knownSpeakerReferences []string
	stream                 *bool
}

func (p *Transcriber) requestSettings(recording stt.Recording) (requestSettings, error) {
	settings := p.settings
	if settings.Delay != "" {
		return requestSettings{}, errors.New("openai-compatible: delay is only supported by realtime transcription")
	}
	if settings.Language != "" && len(settings.Languages) != 0 {
		return requestSettings{}, errors.New("openai-compatible: language and languages overwrites cannot both be set")
	}

	resolved := requestSettings{
		responseFormat:         settings.ResponseFormat,
		prompt:                 settings.Prompt,
		language:               settings.Language,
		languages:              append([]string(nil), settings.Languages...),
		keywords:               append([]string(nil), settings.Keywords...),
		temperature:            settings.Temperature,
		timestampGranularities: append([]string(nil), settings.TimestampGranularities...),
		include:                append([]string(nil), settings.Include...),
		chunkingStrategy:       settings.ChunkingStrategy,
		knownSpeakerNames:      append([]string(nil), settings.KnownSpeakerNames...),
		knownSpeakerReferences: append([]string(nil), settings.KnownSpeakerReferences...),
		stream:                 settings.Stream,
	}
	if resolved.responseFormat == "" {
		switch {
		case recording.Diarize:
			resolved.responseFormat = "diarized_json"
		case recording.Words:
			resolved.responseFormat = "verbose_json"
		default:
			resolved.responseFormat = "json"
		}
	}
	if recording.Words && !recording.Diarize {
		hasWordGranularity := false
		for _, granularity := range resolved.timestampGranularities {
			if granularity == "word" {
				hasWordGranularity = true
				break
			}
		}
		if !hasWordGranularity {
			resolved.timestampGranularities = append(resolved.timestampGranularities, "word")
		}
	}
	if resolved.language == "" && len(resolved.languages) == 0 {
		languageHints := recording.Languages
		if len(languageHints) == 0 {
			languageHints = p.languages
		}
		switch len(languageHints) {
		case 1:
			resolved.language = languageHints[0]
		case 0:
			resolved.language = p.language
		default:
			resolved.languages = append([]string(nil), languageHints...)
		}
	}
	resolved.prompt = VocabularyPrompt(resolved.prompt, mergeTerms(p.keyterms, recording.Keyterms))
	return resolved, nil
}

// VocabularyPrompt adds shared transcription terms using the generic prompt mechanism,
// which is accepted by Whisper and other compatible models without model-specific fields.
func VocabularyPrompt(prompt string, terms []string) string {
	if len(terms) == 0 {
		return prompt
	}
	if prompt != "" {
		prompt += "\n"
	}
	return prompt + "Vocabulary: " + strings.Join(terms, ", ") + "."
}

func mergeTerms(groups ...[]string) []string {
	var merged []string
	seen := make(map[string]struct{})
	for _, group := range groups {
		for _, term := range group {
			if _, exists := seen[term]; exists {
				continue
			}
			seen[term] = struct{}{}
			merged = append(merged, term)
		}
	}
	return merged
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

func audioMetadata(recording stt.Recording) (string, string, error) {
	ext := audioExtension(recording.Audio)
	filename := strings.TrimSpace(recording.Filename)
	if filename != "" {
		filename = filepath.Base(strings.ReplaceAll(filename, "\\", "/"))
	}
	contentType := strings.TrimSpace(recording.ContentType)
	if contentType != "" {
		mediaType, _, err := mime.ParseMediaType(contentType)
		if err != nil || mediaType == "" {
			return "", "", errors.New("openai-compatible: content type must be a valid media type")
		}
		contentType = mediaType
		if ext == ".bin" {
			ext = audioExtensionForType(mediaType)
		}
	}
	if filename == "" || filename == "." {
		filename = "audio" + ext
	} else if filepath.Ext(filename) == "" && ext != ".bin" {
		filename += ext
	}
	if contentType == "" {
		switch strings.ToLower(filepath.Ext(filename)) {
		case ".wav":
			contentType = "audio/wav"
		case ".mp3", ".mpeg", ".mpga":
			contentType = "audio/mpeg"
		case ".mp4", ".m4a":
			contentType = "audio/mp4"
		case ".webm":
			contentType = "audio/webm"
		case ".ogg":
			contentType = "audio/ogg"
		case ".flac":
			contentType = "audio/flac"
		default:
			if byExtension := mime.TypeByExtension(filepath.Ext(filename)); byExtension != "" {
				if mediaType, _, err := mime.ParseMediaType(byExtension); err == nil {
					contentType = mediaType
				}
			}
		}
	}
	if contentType == "" {
		contentType = http.DetectContentType(recording.Audio)
	}
	return filename, contentType, nil
}

func audioExtensionForType(mediaType string) string {
	switch strings.ToLower(mediaType) {
	case "audio/wav", "audio/x-wav":
		return ".wav"
	case "audio/mpeg":
		return ".mp3"
	case "audio/mp4":
		return ".m4a"
	case "audio/webm":
		return ".webm"
	case "audio/ogg":
		return ".ogg"
	case "audio/flac", "audio/x-flac":
		return ".flac"
	default:
		return ".bin"
	}
}

type multipartField struct {
	name  string
	value string
}

func chunkingStrategyFields(raw json.RawMessage) ([]multipartField, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, errors.New("openai-compatible: chunking_strategy must be valid JSON")
	}
	fields := make([]multipartField, 0)
	if err := flattenMultipartValue("chunking_strategy", value, &fields); err != nil {
		return nil, err
	}
	return fields, nil
}

func flattenMultipartValue(name string, value any, fields *[]multipartField) error {
	switch value := value.(type) {
	case map[string]any:
		keys := make([]string, 0, len(value))
		for key := range value {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			if err := flattenMultipartValue(name+"["+key+"]", value[key], fields); err != nil {
				return err
			}
		}
	case []any:
		for _, item := range value {
			if err := flattenMultipartValue(name+"[]", item, fields); err != nil {
				return err
			}
		}
	case nil:
		return nil
	case string:
		*fields = append(*fields, multipartField{name: name, value: value})
	case json.Number:
		*fields = append(*fields, multipartField{name: name, value: value.String()})
	case bool:
		*fields = append(*fields, multipartField{name: name, value: strconv.FormatBool(value)})
	default:
		return errors.New("openai-compatible: chunking_strategy contains an unsupported value")
	}
	return nil
}

func audioExtension(audio []byte) string {
	switch {
	case len(audio) >= 12 && string(audio[:4]) == "RIFF" && string(audio[8:12]) == "WAVE":
		return ".wav"
	case len(audio) >= 4 && string(audio[:4]) == "fLaC":
		return ".flac"
	case len(audio) >= 4 && string(audio[:4]) == "OggS":
		return ".ogg"
	case len(audio) >= 3 && string(audio[:3]) == "ID3":
		return ".mp3"
	case len(audio) >= 8 && string(audio[4:8]) == "ftyp":
		return ".mp4"
	case len(audio) >= 4 && string(audio[:4]) == "\x1aE\xdf\xa3":
		return ".webm"
	case len(audio) >= 2 && audio[0] == 0xff && audio[1]&0xe0 == 0xe0:
		return ".mp3"
	default:
		return ".bin"
	}
}
