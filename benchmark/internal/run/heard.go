package run

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
)

type heardEvent struct {
	Kind string `json:"kind"`
	Said string `json:"said,omitempty"`
	At   string `json:"at,omitempty"`
}

// captureAgentHeard writes what the acceleration agent settled on, per utterance, and
// counts what it did with it. Deepgram's view of the caller leg is already in
// transcript.json; this is the agent's own STT, which is what hid the 7:30 split.
//
// The counts are what tell a noise failure apart: no utterances at all is STT losing the
// caller, and utterances the flow controller ignored is the agent deciding they were not
// meant for it.
func captureAgentHeard(cfg Config, callID, callDir string, m *score.Metrics) error {
	if cfg.TargetName != "accelerated" && cfg.TargetName != "acceleration" {
		return nil
	}
	base := strings.TrimRight(envOr("STREAM_ACCELERATION_URL", "http://127.0.0.1:8080"), "/")
	customer := envOr("STREAM_ACCELERATION_CUSTOMER_ID", "voicebench")
	events, err := fetchAgentHeard(base, customer, callID)
	if err != nil {
		return err
	}
	for _, event := range events {
		switch event.Kind {
		case "ask":
			m.HeardUtterances++
		case "ignore":
			m.HeardIgnored++
		}
	}
	return writeJSON(filepath.Join(callDir, "heard.json"), events)
}

func fetchAgentHeard(base, customer, streamCallID string) ([]heardEvent, error) {
	sessionID, err := resolveHeardCallID(base, customer, streamCallID)
	if err != nil {
		return nil, err
	}
	body, err := getJSON(base+"/v1/agents/calls/"+sessionID+"/events", customer)
	if err != nil {
		return nil, err
	}
	var raw []struct {
		Kind string  `json:"kind"`
		Said *string `json:"said"`
		At   string  `json:"at"`
	}
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, err
	}
	var events []heardEvent
	for _, event := range raw {
		switch event.Kind {
		case "ask", "answer", "settle", "wait", "interrupt", "ignore":
		default:
			continue
		}
		// A ruling with no words is still a ruling. Dropping those hid exactly the
		// case worth reading: an utterance the agent decided was not for it.
		said := ""
		if event.Said != nil {
			said = strings.TrimSpace(*event.Said)
		}
		events = append(events, heardEvent{Kind: event.Kind, Said: said, At: event.At})
	}
	return events, nil
}

// resolveHeardCallID maps the Stream call id the harness joined to the session
// id the events API holds the row by.
func resolveHeardCallID(base, customer, streamCallID string) (string, error) {
	body, err := getJSON(base+"/v1/agents/calls?limit=50", customer)
	if err != nil {
		return "", err
	}
	var listed []struct {
		ID     string `json:"id"`
		CallID string `json:"call_id"`
	}
	if err := json.Unmarshal(body, &listed); err != nil {
		return "", err
	}
	for _, call := range listed {
		if call.CallID == streamCallID || call.ID == streamCallID {
			return call.ID, nil
		}
	}
	return "", fmt.Errorf("heard: no call with stream id %s", streamCallID)
}

func getJSON(url, customer string) ([]byte, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-Customer-Id", customer)
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("heard: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	return body, nil
}
