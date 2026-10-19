package openaicompat

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"mime"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type transcriptionResponse struct {
	Text        string                  `json:"text"`
	TextPresent bool                    `json:"-"`
	Language    string                  `json:"language"`
	Languages   []transcriptionLanguage `json:"languages"`
	Duration    *float64                `json:"duration"`
	Usage       transcriptionUsage      `json:"usage"`
	Words       []transcriptionWord     `json:"words"`
	Segments    []transcriptionSegment  `json:"segments"`
}

type transcriptionLanguage struct {
	Code string `json:"code"`
}

type transcriptionUsage struct {
	Seconds *float64 `json:"seconds"`
}

type transcriptionWord struct {
	Text       string   `json:"word"`
	Start      *float64 `json:"start"`
	End        *float64 `json:"end"`
	Confidence *float64 `json:"confidence"`
	Speaker    string   `json:"speaker"`
	SpeakerID  string   `json:"speaker_id"`
}

type transcriptionSegment struct {
	Text    string              `json:"text"`
	Start   *float64            `json:"start"`
	End     *float64            `json:"end"`
	Speaker string              `json:"speaker"`
	Words   []transcriptionWord `json:"words"`
}

func parseTranscription(body []byte, contentType, responseFormat string) (stt.Transcription, bool, error) {
	mediaType, _, _ := mime.ParseMediaType(contentType)
	if mediaType == "text/event-stream" {
		response, err := parseTranscriptionStream(body)
		if err != nil {
			return stt.Transcription{}, false, err
		}
		return transcriptionOf(response)
	}
	format := strings.ToLower(responseFormat)
	if format == "text" {
		return stt.Transcription{Text: string(body)}, false, nil
	}
	if format == "srt" || format == "vtt" || mediaType == "text/vtt" || mediaType == "application/x-subrip" {
		return stt.Transcription{Text: subtitleText(string(body))}, false, nil
	}
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return stt.Transcription{}, false, errors.New("openai-compatible: decode transcript: response is empty")
	}
	if trimmed[0] == '"' {
		var text string
		if err := json.Unmarshal(trimmed, &text); err != nil {
			return stt.Transcription{}, false, fmt.Errorf("openai-compatible: decode transcript: %w", err)
		}
		return stt.Transcription{Text: text}, false, nil
	}
	if trimmed[0] != '{' {
		return stt.Transcription{Text: string(body)}, false, nil
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &fields); err != nil {
		return stt.Transcription{}, false, fmt.Errorf("openai-compatible: decode transcript: %w", err)
	}
	textPresent := false
	if rawText, exists := fields["text"]; exists {
		if bytes.Equal(bytes.TrimSpace(rawText), []byte("null")) {
			return stt.Transcription{}, false, errors.New("openai-compatible: decode transcript: text must be a string")
		}
		var text string
		if err := json.Unmarshal(rawText, &text); err != nil {
			return stt.Transcription{}, false, errors.New("openai-compatible: decode transcript: text must be a string")
		}
		textPresent = true
	}
	decoder := json.NewDecoder(bytes.NewReader(trimmed))
	var response transcriptionResponse
	if err := decoder.Decode(&response); err != nil {
		return stt.Transcription{}, false, fmt.Errorf("openai-compatible: decode transcript: %w", err)
	}
	response.TextPresent = textPresent
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return stt.Transcription{}, false, errors.New("openai-compatible: decode transcript: response has trailing JSON")
	}
	return transcriptionOf(response)
}

func transcriptionOf(response transcriptionResponse) (stt.Transcription, bool, error) {
	text := response.Text
	if text == "" && !response.TextPresent && len(response.Segments) != 0 {
		parts := make([]string, 0, len(response.Segments))
		for _, segment := range response.Segments {
			if segment.Text != "" {
				parts = append(parts, segment.Text)
			}
		}
		text = strings.Join(parts, " ")
	}
	if !response.TextPresent && len(response.Segments) == 0 && len(response.Words) == 0 {
		return stt.Transcription{}, false, errors.New("openai-compatible: decode transcript: response is missing text")
	}
	language := response.Language
	if language == "" && len(response.Languages) == 1 {
		language = response.Languages[0].Code
	}
	transcription := stt.Transcription{Text: text, Language: language}
	duration, durationKnown, err := responseDuration(response)
	if err != nil {
		return stt.Transcription{}, false, err
	}
	transcription.AudioDurationMs = duration
	var sourceWords []transcriptionWord
	if len(response.Words) != 0 {
		sourceWords = response.Words
	} else {
		for _, segment := range response.Segments {
			for _, word := range segment.Words {
				if word.Speaker == "" && word.SpeakerID == "" {
					word.Speaker = segment.Speaker
				}
				sourceWords = append(sourceWords, word)
			}
		}
	}
	for _, source := range sourceWords {
		if source.Start == nil || source.End == nil {
			continue
		}
		if *source.Start < 0 || *source.End < *source.Start {
			return stt.Transcription{}, false, errors.New("openai-compatible: decode transcript: word timestamps are invalid")
		}
		start, err := secondsToMilliseconds(*source.Start)
		if err != nil {
			return stt.Transcription{}, false, err
		}
		end, err := secondsToMilliseconds(*source.End)
		if err != nil {
			return stt.Transcription{}, false, err
		}
		confidence := float64(0)
		if source.Confidence != nil {
			confidence = *source.Confidence
		}
		speaker := source.Speaker
		if speaker == "" {
			speaker = source.SpeakerID
		}
		transcription.Words = append(transcription.Words, stt.Word{
			Text: source.Text, StartMs: start, EndMs: end, Confidence: confidence, Speaker: speaker,
		})
	}
	seenSpeakers := map[string]bool{}
	addSpeaker := func(speaker string) {
		if speaker != "" && !seenSpeakers[speaker] {
			transcription.Speakers = append(transcription.Speakers, speaker)
			seenSpeakers[speaker] = true
		}
	}
	for _, segment := range response.Segments {
		addSpeaker(segment.Speaker)
	}
	for _, word := range transcription.Words {
		addSpeaker(word.Speaker)
	}
	return transcription, durationKnown, nil
}

func responseDuration(response transcriptionResponse) (int64, bool, error) {
	seconds := response.Duration
	if seconds == nil {
		seconds = response.Usage.Seconds
	}
	if seconds == nil {
		return 0, false, nil
	}
	if *seconds < 0 {
		return 0, false, errors.New("openai-compatible: decode transcript: duration is negative")
	}
	milliseconds, err := secondsToMilliseconds(*seconds)
	return milliseconds, true, err
}

func secondsToMilliseconds(seconds float64) (int64, error) {
	if math.IsNaN(seconds) || math.IsInf(seconds, 0) || seconds < 0 || seconds > float64(math.MaxInt64)/1000 {
		return 0, errors.New("openai-compatible: decode transcript: timestamp is out of range")
	}
	return int64(math.Round(seconds * 1000)), nil
}

func subtitleText(value string) string {
	lines := strings.Split(strings.ReplaceAll(value, "\r\n", "\n"), "\n")
	text := make([]string, 0, len(lines))
	inVTTHeader := len(lines) > 0 && strings.TrimSpace(lines[0]) == "WEBVTT"
	for index, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			inVTTHeader = false
			continue
		}
		if inVTTHeader || line == "WEBVTT" || strings.Contains(line, "-->") {
			continue
		}
		if isCueIdentifier(line) && nextLineHasTimestamp(lines, index+1) {
			continue
		}
		text = append(text, line)
	}
	return strings.Join(text, " ")
}

func isCueIdentifier(line string) bool {
	if line == "" {
		return false
	}
	for _, character := range line {
		if character < '0' || character > '9' {
			return false
		}
	}
	return true
}

func nextLineHasTimestamp(lines []string, start int) bool {
	return start < len(lines) && strings.Contains(strings.TrimSpace(lines[start]), "-->")
}

func parseTranscriptionStream(body []byte) (transcriptionResponse, error) {
	var response transcriptionResponse
	var finalText string
	finalSeen := false
	eventName := ""
	data := strings.Builder{}
	flush := func() error {
		if data.Len() == 0 {
			return nil
		}
		payload := []byte(strings.TrimSpace(data.String()))
		data.Reset()
		if string(payload) == "[DONE]" {
			return nil
		}
		var event struct {
			Type      string                  `json:"type"`
			Text      *string                 `json:"text"`
			Duration  *float64                `json:"duration"`
			Usage     transcriptionUsage      `json:"usage"`
			Speaker   string                  `json:"speaker"`
			Languages []transcriptionLanguage `json:"languages"`
		}
		if err := json.Unmarshal(payload, &event); err != nil {
			return fmt.Errorf("openai-compatible: decode transcript stream: %w", err)
		}
		kind := event.Type
		if kind == "" {
			kind = eventName
		}
		switch kind {
		case "transcript.text.done":
			if event.Text != nil {
				finalText, finalSeen = *event.Text, true
				response.TextPresent = true
				response.Languages = event.Languages
			}
		case "transcript.text.segment":
			var segment transcriptionSegment
			if err := json.Unmarshal(payload, &segment); err != nil {
				return fmt.Errorf("openai-compatible: decode transcript segment: %w", err)
			}
			response.Segments = append(response.Segments, segment)
		case "error":
			return errors.New("openai-compatible: transcription stream returned an error event")
		}
		if event.Duration != nil {
			response.Duration = event.Duration
		}
		if event.Usage.Seconds != nil {
			response.Usage = event.Usage
		}
		if len(event.Languages) != 0 {
			response.Languages = event.Languages
		}
		return nil
	}
	for _, line := range bytes.Split(body, []byte("\n")) {
		line = bytes.TrimSuffix(line, []byte("\r"))
		if len(line) == 0 {
			if err := flush(); err != nil {
				return transcriptionResponse{}, err
			}
			eventName = ""
			continue
		}
		if bytes.HasPrefix(line, []byte("event:")) {
			eventName = strings.TrimSpace(string(bytes.TrimPrefix(line, []byte("event:"))))
			continue
		}
		if bytes.HasPrefix(line, []byte("data:")) {
			if data.Len() != 0 {
				data.WriteByte('\n')
			}
			data.Write(bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:"))))
		}
	}
	if err := flush(); err != nil {
		return transcriptionResponse{}, err
	}
	if !finalSeen {
		return transcriptionResponse{}, errors.New("openai-compatible: decode transcript stream: response is missing transcript.text.done")
	}
	response.Text = finalText
	return response, nil
}

type endpointAPIError struct {
	status int
	typeID string
	code   string
	param  string
}

func (e endpointAPIError) Error() string {
	parts := []string{fmt.Sprintf("endpoint returned HTTP %d", e.status)}
	if e.typeID != "" {
		parts = append(parts, "type="+e.typeID)
	}
	if e.code != "" {
		parts = append(parts, "code="+e.code)
	}
	if e.param != "" {
		parts = append(parts, "param="+e.param)
	}
	return strings.Join(parts, " ")
}

func transcriptionAPIError(status int, body []byte) error {
	var response struct {
		Error struct {
			Type  string `json:"type"`
			Code  string `json:"code"`
			Param string `json:"param"`
		} `json:"error"`
	}
	_ = json.Unmarshal(body, &response)
	return endpointAPIError{
		status: status,
		typeID: safeErrorField(response.Error.Type),
		code:   safeErrorField(response.Error.Code),
		param:  safeErrorField(response.Error.Param),
	}
}

func safeErrorField(value string) string {
	value = strings.TrimSpace(value)
	if len(value) > 128 {
		value = value[:128]
	}
	return strings.Map(func(character rune) rune {
		if character < 0x20 || character == 0x7f {
			return -1
		}
		return character
	}, value)
}
