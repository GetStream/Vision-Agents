package openaicompat

import (
	"encoding/json"
	"errors"
	"net/url"
	"strings"
)

type realtimeEvent struct {
	Type       string `json:"type"`
	ItemID     string `json:"item_id"`
	Transcript string `json:"transcript"`
	Code       string `json:"code"`
	Error      struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func realtimeEndpoint(base, model string) (string, error) {
	parsed, err := parseBaseURL(base)
	if err != nil {
		return "", err
	}
	if parsed.Scheme == "https" {
		parsed.Scheme = "wss"
	} else {
		parsed.Scheme = "ws"
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/") + "/realtime"
	query := url.Values{}
	query.Set("intent", "transcription")
	query.Set("model", model)
	parsed.RawQuery = query.Encode()
	return parsed.String(), nil
}

func parseBaseURL(base string) (*url.URL, error) {
	parsed, err := url.Parse(base)
	if err != nil {
		return nil, errors.New("base URL is invalid")
	}
	if (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Hostname() == "" || parsed.User != nil ||
		parsed.Opaque != "" || parsed.RawPath != "" || parsed.ForceQuery || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, errors.New("base URL must be an absolute http or https URL without credentials, query, or fragment")
	}
	return parsed, nil
}

func sessionUpdate(model, language string) map[string]any {
	transcription := map[string]any{"model": model}
	if language != "" {
		transcription["language"] = language
	}
	return map[string]any{
		"type": "session.update",
		"session": map[string]any{
			"type": "transcription",
			"audio": map[string]any{
				"input": map[string]any{
					"format":          map[string]any{"type": "audio/pcm", "rate": 24000},
					"transcription":   transcription,
					"turn_detection":  nil,
					"noise_reduction": nil,
				},
			},
		},
	}
}

func eventErrorCode(event realtimeEvent) string {
	if event.Error.Code != "" {
		return event.Error.Code
	}
	return event.Code
}

func decodeRealtimeEvent(data []byte) (realtimeEvent, error) {
	var event realtimeEvent
	if err := json.Unmarshal(data, &event); err != nil {
		return realtimeEvent{}, errors.New("openai-compatible: realtime event is not valid JSON")
	}
	if event.Type == "" {
		return realtimeEvent{}, errors.New("openai-compatible: realtime event has no type")
	}
	return event, nil
}
