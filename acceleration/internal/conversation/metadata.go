package conversation

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	supportMessageVersion  = 1
	maxSupportMessageBytes = 32 << 10
	maxDisplayItems        = 20
)

var displayID = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,128}$`)
var artifactID = regexp.MustCompile(`^[A-Za-z0-9_-]{1,80}$`)

const maxArtifactAttachments = 32

type ArtifactAttachment struct {
	Type       string `json:"type"`
	ArtifactID string `json:"artifact_id"`
	Revision   int    `json:"revision"`
	Title      string `json:"title"`
	Alt        string `json:"alt,omitempty"`
}

type supportMessage struct {
	SchemaVersion int             `json:"schema_version"`
	Sequence      int             `json:"sequence"`
	State         string          `json:"state"`
	Tools         []displayTool   `json:"attachments,omitempty"`
	Sources       []displaySource `json:"sources,omitempty"`
}

type displayTool struct {
	ID      string `json:"tool_call_id"`
	Name    string `json:"name"`
	Status  string `json:"status"`
	Summary string `json:"display_summary,omitempty"`
}

type displaySource struct {
	ID       string `json:"id"`
	Title    string `json:"title"`
	URL      string `json:"url"`
	Citation string `json:"citation,omitempty"`
}

type runtimeMessage struct {
	CommandID      string     `json:"command_id,omitempty"`
	TurnID         string     `json:"turn_id,omitempty"`
	QuestionID     string     `json:"question_id,omitempty"`
	Role           string     `json:"role"`
	State          string     `json:"state"`
	StartedAt      time.Time  `json:"response_started_at"`
	StateStartedAt time.Time  `json:"state_started_at"`
	FinishedAt     *time.Time `json:"finished_at,omitempty"`
	DurationMS     int64      `json:"duration_ms"`
}

func metadataOf(message Message) (supportMessage, error) {
	metadata := supportMessage{
		SchemaVersion: supportMessageVersion,
		Sequence:      message.Sequence,
		State:         message.State,
		Tools:         []displayTool{},
		Sources:       []displaySource{},
	}
	if message.Sequence < 0 || !observableState(message.State) {
		return supportMessage{}, errors.New("invalid observable message state")
	}
	for _, tool := range message.Tools {
		display, ok := displayToolOf(tool)
		if !ok {
			continue
		}
		if len(metadata.Tools) == maxDisplayItems {
			break
		}
		metadata.Tools = append(metadata.Tools, display)
	}
	seen := map[string]struct{}{}
	for _, source := range message.Sources {
		if len(metadata.Sources) == maxDisplayItems {
			break
		}
		if !validSource(source) {
			continue
		}
		if _, exists := seen[source.ID]; exists {
			continue
		}
		seen[source.ID] = struct{}{}
		metadata.Sources = append(metadata.Sources, displaySource(source))
	}
	encoded, err := json.Marshal(metadata)
	if err != nil || len(encoded) > maxSupportMessageBytes {
		return supportMessage{}, errors.New("observable message metadata is too large")
	}
	return metadata, nil
}

func decodeMetadata(raw any) (supportMessage, error) {
	encoded, err := json.Marshal(raw)
	if err != nil || len(encoded) == 0 || len(encoded) > maxSupportMessageBytes {
		return supportMessage{}, errors.New("invalid observable message metadata")
	}
	var metadata supportMessage
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&metadata) != nil || !errors.Is(decoder.Decode(&struct{}{}), io.EOF) {
		return supportMessage{}, errors.New("invalid observable message metadata")
	}
	if metadata.SchemaVersion != supportMessageVersion || metadata.Sequence < 0 ||
		!observableState(metadata.State) || len(metadata.Tools) > maxDisplayItems ||
		len(metadata.Sources) > maxDisplayItems {
		return supportMessage{}, errors.New("invalid observable message metadata")
	}
	seenTools := map[string]struct{}{}
	for _, tool := range metadata.Tools {
		if !displayID.MatchString(tool.ID) || tool.Name != "athena_resource_metadata" ||
			tool.Status == "running" && tool.Summary != "" ||
			tool.Status == "completed" && tool.Summary != "Conversation metadata checked." ||
			tool.Status == "failed" && tool.Summary != "Conversation metadata unavailable." &&
				tool.Summary != "Stopped before completion." ||
			tool.Status != "running" && tool.Status != "completed" && tool.Status != "failed" {
			return supportMessage{}, errors.New("invalid observable message metadata")
		}
		if _, exists := seenTools[tool.ID]; exists {
			return supportMessage{}, errors.New("invalid observable message metadata")
		}
		seenTools[tool.ID] = struct{}{}
	}
	seenSources := map[string]struct{}{}
	for _, source := range metadata.Sources {
		if !validSource(Source(source)) {
			return supportMessage{}, errors.New("invalid observable message metadata")
		}
		if _, exists := seenSources[source.ID]; exists {
			return supportMessage{}, errors.New("invalid observable message metadata")
		}
		seenSources[source.ID] = struct{}{}
	}
	return metadata, nil
}

func decodeRuntime(raw any) (runtimeMessage, error) {
	encoded, err := json.Marshal(raw)
	if err != nil || len(encoded) > 8<<10 {
		return runtimeMessage{}, errors.New("invalid runtime message metadata")
	}
	var runtime runtimeMessage
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&runtime) != nil || !errors.Is(decoder.Decode(&struct{}{}), io.EOF) ||
		(runtime.Role != "user" && runtime.Role != "assistant") ||
		!observableState(runtime.State) || runtime.StartedAt.IsZero() || runtime.StateStartedAt.IsZero() {
		return runtimeMessage{}, errors.New("invalid runtime message metadata")
	}
	return runtime, nil
}

func messageFromWire(id, text string, custom map[string]any) (Message, error) {
	runtimeRaw, runtimeOK := custom["support_runtime"]
	metadataRaw, metadataOK := custom["support_message"]
	if !runtimeOK || !metadataOK {
		return Message{}, errors.New("schema-v1 message metadata is unavailable")
	}
	runtime, err := decodeRuntime(runtimeRaw)
	if err != nil {
		return Message{}, err
	}
	metadata, err := decodeMetadata(metadataRaw)
	if err != nil || runtime.State != metadata.State ||
		id == "" || len(id) > 128 || len(runtime.CommandID) > 128 || len(runtime.TurnID) > 128 {
		return Message{}, errors.New("invalid schema-v1 message metadata")
	}
	message := Message{
		CommandID: runtime.CommandID, TurnID: runtime.TurnID, ID: id,
		QuestionID: runtime.QuestionID, Role: runtime.Role, Text: text,
		State: runtime.State, StartedAt: runtime.StartedAt,
		StateStartedAt: runtime.StateStartedAt, FinishedAt: runtime.FinishedAt,
		DurationMS: runtime.DurationMS, Sequence: metadata.Sequence, Saved: true,
		Tools: []Tool{}, Sources: []Source{},
	}
	for _, display := range metadata.Tools {
		message.Tools = append(message.Tools, Tool{
			Type: "tool_calling", ID: display.ID, Name: display.Name, Title: title(display.Name),
			Status: display.Status, Phase: display.Status, Summary: display.Summary,
		})
	}
	for _, display := range metadata.Sources {
		message.Sources = append(message.Sources, Source(display))
	}
	return message, nil
}

func displayToolOf(tool Tool) (displayTool, bool) {
	if tool.Name != "athena_resource_metadata" || !displayID.MatchString(tool.ID) {
		return displayTool{}, false
	}
	display := displayTool{ID: tool.ID, Name: tool.Name}
	switch tool.Status {
	case "running", "queued":
		display.Status = "running"
	case "completed":
		display.Status = "completed"
		display.Summary = "Conversation metadata checked."
	case "failed":
		display.Status = "failed"
		display.Summary = "Conversation metadata unavailable."
	case "cancelled":
		display.Status = "failed"
		display.Summary = "Stopped before completion."
	default:
		return displayTool{}, false
	}
	return display, true
}

func sourcesOf(tool, result string) []Source {
	if tool != "search_docs" && tool != "search_web" || len(result) > maxSupportMessageBytes {
		return nil
	}
	var payload struct {
		Status    string   `json:"status"`
		Citations []Source `json:"citations"`
	}
	decoder := json.NewDecoder(strings.NewReader(result))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&payload) != nil || !errors.Is(decoder.Decode(&struct{}{}), io.EOF) ||
		payload.Status != "answered" {
		return nil
	}
	sources := make([]Source, 0, min(len(payload.Citations), maxDisplayItems))
	seen := map[string]struct{}{}
	for _, source := range payload.Citations {
		if len(sources) == maxDisplayItems {
			break
		}
		if !validSource(source) {
			continue
		}
		if _, exists := seen[source.ID]; exists {
			continue
		}
		seen[source.ID] = struct{}{}
		sources = append(sources, source)
	}
	return sources
}

func mergeSources(existing, additions []Source) []Source {
	merged := append([]Source{}, existing...)
	seen := map[string]struct{}{}
	for _, source := range merged {
		seen[source.ID] = struct{}{}
	}
	for _, source := range additions {
		if len(merged) == maxDisplayItems {
			break
		}
		if _, exists := seen[source.ID]; exists {
			continue
		}
		seen[source.ID] = struct{}{}
		merged = append(merged, source)
	}
	return merged
}

// StoredArtifacts accepts only validated stored artifact receipts, never model prose.
// Voice transcripts use the same receipt format as persistent text conversations.
func StoredArtifacts(result string) []ArtifactAttachment {
	return artifactsOf(result)
}

func artifactsOf(result string) []ArtifactAttachment {
	if result == "" || len(result) > maxSupportMessageBytes {
		return nil
	}
	var payload struct {
		SchemaVersion int    `json:"schema_version"`
		Status        string `json:"status"`
		Publication   string `json:"publication"`
		Attachment    struct {
			Type       string `json:"type"`
			ArtifactID string `json:"artifact_id"`
			Revision   int    `json:"revision"`
			Title      string `json:"title"`
			Alt        string `json:"alt"`
			SHA256     string `json:"sha256"`
		} `json:"attachment"`
	}
	decoder := json.NewDecoder(strings.NewReader(result))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&payload) != nil || !errors.Is(decoder.Decode(&struct{}{}), io.EOF) ||
		payload.SchemaVersion != 1 || payload.Status != "stored" || payload.Publication != "pending" {
		return nil
	}
	artifact := ArtifactAttachment{
		Type: payload.Attachment.Type, ArtifactID: payload.Attachment.ArtifactID,
		Revision: payload.Attachment.Revision, Title: payload.Attachment.Title,
		Alt: payload.Attachment.Alt,
	}
	if !validArtifact(artifact) {
		return nil
	}
	return []ArtifactAttachment{artifact}
}

func validArtifact(artifact ArtifactAttachment) bool {
	switch artifact.Type {
	case "athena_image":
		if !boundedDisplayText(artifact.Alt, 500) {
			return false
		}
	case "athena_pdf", "athena_canvas", "athena_file", "athena_site":
		if artifact.Alt != "" {
			return false
		}
	default:
		return false
	}
	return artifactID.MatchString(artifact.ArtifactID) &&
		artifact.Revision >= 1 && artifact.Revision <= 2147483647 &&
		boundedDisplayText(artifact.Title, 200)
}

func mergeArtifacts(existing, additions []ArtifactAttachment) []ArtifactAttachment {
	merged := append([]ArtifactAttachment{}, existing...)
	seen := map[string]struct{}{}
	for _, artifact := range merged {
		seen[artifact.ArtifactID+":"+strconv.Itoa(artifact.Revision)] = struct{}{}
	}
	for _, artifact := range additions {
		if len(merged) == maxArtifactAttachments {
			break
		}
		key := artifact.ArtifactID + ":" + strconv.Itoa(artifact.Revision)
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		merged = append(merged, artifact)
	}
	return merged
}

func streamAttachments(artifacts []ArtifactAttachment) []map[string]any {
	if len(artifacts) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(artifacts))
	for _, artifact := range artifacts {
		item := map[string]any{
			"type": artifact.Type, "artifact_id": artifact.ArtifactID,
			"revision": artifact.Revision, "title": artifact.Title,
		}
		if artifact.Alt != "" {
			item["alt"] = artifact.Alt
		}
		out = append(out, item)
	}
	return out
}

func runtimeOf(message Message) runtimeMessage {
	runtime := runtimeMessage{
		CommandID: message.CommandID, TurnID: message.TurnID, QuestionID: message.QuestionID,
		Role: message.Role, State: message.State, StartedAt: message.StartedAt,
		StateStartedAt: message.StateStartedAt, DurationMS: message.DurationMS,
	}
	if message.FinishedAt != nil {
		finished := *message.FinishedAt
		runtime.FinishedAt = &finished
	}
	return runtime
}

func observableState(state string) bool {
	switch state {
	case "queued", "thinking", "writing", "tools", "completed", "failed", "cancelled", "interrupted":
		return true
	default:
		return false
	}
}

func validSource(source Source) bool {
	parsed, err := url.Parse(source.URL)
	rawHost := strings.ToLower(parsed.Hostname())
	host := strings.TrimSuffix(rawHost, ".")
	return err == nil && displayID.MatchString(source.ID) &&
		boundedDisplayText(source.Title, 200) && boundedOptionalText(source.Citation, 500) &&
		parsed.Scheme == "https" && parsed.User == nil && host != "" &&
		(parsed.Port() == "" || parsed.Port() == "443") && host == rawHost &&
		net.ParseIP(host) == nil && strings.Contains(host, ".") &&
		host != "localhost" && !strings.HasSuffix(host, ".localhost") &&
		!strings.HasSuffix(host, ".local") && !strings.HasSuffix(host, ".internal") &&
		!strings.HasSuffix(host, ".home")
}

func boundedDisplayText(value string, maximum int) bool {
	return strings.TrimSpace(value) != "" && utf8.ValidString(value) &&
		utf8.RuneCountInString(value) <= maximum && !strings.ContainsRune(value, '\x00')
}

func boundedOptionalText(value string, maximum int) bool {
	return value == "" || boundedDisplayText(value, maximum)
}
