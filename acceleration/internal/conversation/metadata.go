package conversation

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/url"
	"path"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	getstream "github.com/GetStream/getstream-go/v5"
)

const (
	supportMessageVersion  = 1
	maxSupportMessageBytes = 32 << 10
	maxDisplayItems        = 20
)

var displayID = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,128}$`)

// DefaultVisibleTools are the tools whose steps end users see when an agent config names none.
var DefaultVisibleTools = []string{"search", "web_search"}

// ValidVisibleTool reports whether pattern is a tool name or a path.Match pattern a config
// may name in visible_tools.
func ValidVisibleTool(pattern string) bool {
	_, err := path.Match(pattern, "")
	return err == nil && pattern != "" && len(pattern) <= 128
}

// ToolVisible reports whether a tool's name and outcome may reach chat clients. Anything a
// config does not name, connector operations and arbitrary skills above all, stays hidden.
func ToolVisible(patterns []string, name string) bool {
	if len(patterns) == 0 {
		patterns = DefaultVisibleTools
	}
	for _, pattern := range patterns {
		if matched, _ := path.Match(pattern, name); matched {
			return true
		}
	}
	return false
}

// supportMessage is what a Chat message carries as custom.support_message: the reply's state
// and the steps and sources an end user may see. Arguments and results are never part of it.
type supportMessage struct {
	SchemaVersion int             `json:"schema_version"`
	Sequence      int             `json:"sequence"`
	State         string          `json:"state"`
	Role          string          `json:"role,omitempty"`
	CommandID     string          `json:"command_id,omitempty"`
	QuestionID    string          `json:"question_id,omitempty"`
	Tools         []displayTool   `json:"attachments,omitempty"`
	Sources       []displaySource `json:"sources,omitempty"`
}

type displayTool struct {
	ID      string `json:"tool_call_id"`
	Name    string `json:"name"`
	Status  string `json:"status"`
	Summary string `json:"display_summary,omitempty"`
	// Timing is shown so end users can see where a reply spent its time.
	StartedAt  *time.Time `json:"started_at,omitempty"`
	FinishedAt *time.Time `json:"finished_at,omitempty"`
	DurationMS int64      `json:"duration_ms,omitempty"`
}

type displaySource struct {
	ID       string `json:"id"`
	Title    string `json:"title"`
	URL      string `json:"url"`
	Citation string `json:"citation,omitempty"`
}

// runtimeMessage is custom.support_runtime: who wrote a message, for which command and
// turn, where its answer starts and how long it took.
type runtimeMessage struct {
	TextLayout     int        `json:"text_layout,omitempty"`
	AnswerStart    int        `json:"answer_start"`
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

func metadataOf(message Message, visible []string) (supportMessage, error) {
	metadata := supportMessage{
		SchemaVersion: supportMessageVersion,
		Sequence:      message.Sequence,
		State:         message.State,
		Role:          message.Role,
		CommandID:     message.CommandID,
		QuestionID:    message.QuestionID,
		Tools:         []displayTool{},
		Sources:       []displaySource{},
	}
	if message.Sequence < 0 || !observableState(message.State) {
		return supportMessage{}, errors.New("invalid observable message state")
	}
	for _, tool := range message.Tools {
		if !ToolVisible(visible, tool.Name) {
			continue
		}
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

// decodeMetadata reads a stored support_message back, refusing anything a writer of this
// version would not have written.
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
		expected, ok := displayToolOf(Tool{ID: tool.ID, Name: tool.Name, Status: tool.Status})
		// Timing is the one part a client cannot derive from the name and status.
		expected.StartedAt, expected.FinishedAt, expected.DurationMS = tool.StartedAt, tool.FinishedAt, tool.DurationMS
		if !ok || !sameDisplay(tool, expected) && !(tool.Status == "failed" && tool.Summary == "Stopped before completion.") || !validTiming(tool) {
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

// messageFromWire rebuilds a message from what a schema-v1 Chat message carries.
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
		runtime.TextLayout < 0 || runtime.TextLayout > 1 || runtime.AnswerStart < 0 || runtime.AnswerStart > utf8.RuneCountInString(text) ||
		id == "" || len(id) > 128 || len(runtime.CommandID) > 128 || len(runtime.TurnID) > 128 {
		return Message{}, errors.New("invalid schema-v1 message metadata")
	}
	message := Message{
		CommandID: runtime.CommandID, TurnID: runtime.TurnID, ID: id,
		QuestionID: runtime.QuestionID, Role: runtime.Role, Text: text,
		TextLayout: runtime.TextLayout, AnswerStart: runtime.AnswerStart,
		State: runtime.State, StartedAt: runtime.StartedAt,
		StateStartedAt: runtime.StateStartedAt, FinishedAt: runtime.FinishedAt,
		DurationMS: runtime.DurationMS, Sequence: metadata.Sequence, Saved: true,
		Tools: []Tool{}, Sources: []Source{},
	}
	for _, display := range metadata.Tools {
		message.Tools = append(message.Tools, Tool{
			Type: "tool_calling", ID: display.ID, Name: display.Name, Title: title(display.Name),
			Status: display.Status, Phase: display.Status, Summary: display.Summary,
			StartedAt: startedAt(display), FinishedAt: display.FinishedAt, DurationMS: display.DurationMS,
		})
	}
	for _, display := range metadata.Sources {
		message.Sources = append(message.Sources, Source(display))
	}
	return message, nil
}

// displayToolOf is what an end user may see of a step: its name, a fixed summary of how it
// ended and its timing.
func displayToolOf(tool Tool) (displayTool, bool) {
	if !displayID.MatchString(tool.Name) || !displayID.MatchString(tool.ID) {
		return displayTool{}, false
	}
	display := displayTool{ID: tool.ID, Name: tool.Name, FinishedAt: tool.FinishedAt, DurationMS: tool.DurationMS}
	if !tool.StartedAt.IsZero() {
		started := tool.StartedAt
		display.StartedAt = &started
	}
	switch tool.Status {
	case "running", "queued":
		display.Status = "running"
	case "completed":
		display.Status = "completed"
	case "failed":
		display.Status = "failed"
		display.Summary = "Didn’t complete."
	case "cancelled":
		display.Status = "failed"
		display.Summary = "Stopped before completion."
	default:
		return displayTool{}, false
	}
	return display, true
}

// sourcesOf reads the sources a tool result cites. The convention is a result that is
// exactly {"status":"answered","citations":[{"id","title","url","citation"}]}: an id of
// letters, digits, ".", "_" or "-", a title of at most 200 characters, a public https URL
// and an optional citation of at most 500. Anything else in the result cites nothing.
func sourcesOf(result string) []Source {
	if len(result) > maxSupportMessageBytes {
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
	return mergeSources(nil, payload.Citations)
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
		if !validSource(source) {
			continue
		}
		if _, exists := seen[source.ID]; exists {
			continue
		}
		seen[source.ID] = struct{}{}
		merged = append(merged, source)
	}
	return merged
}

var (
	artifactID   = regexp.MustCompile(`^[A-Za-z0-9_-]{1,80}$`)
	artifactType = regexp.MustCompile(`^[a-z][a-z0-9_]{0,63}$`)
)

const maxArtifactAttachments = 32

// ArtifactAttachment is a stored artifact a reply links to. It reaches Chat as an attachment
// whose type, title and custom artifact_id, revision and alt a client opens it by.
type ArtifactAttachment struct {
	Type       string `json:"type"`
	ArtifactID string `json:"artifact_id"`
	Revision   int    `json:"revision"`
	Title      string `json:"title"`
	Alt        string `json:"alt,omitempty"`
}

// StoredArtifacts reads the artifact a tool result says it stored, never model prose. The
// convention is a result that is exactly {"schema_version":1,"status":"stored",
// "publication":"pending","attachment":{"type","artifact_id","revision","title","alt",
// "sha256"}}, publication and alt and sha256 being optional: a type of lowercase letters,
// digits and "_", an artifact_id of at most 80 letters, digits, "_" or "-", a positive
// revision, a title of at most 200 characters and alt text of at most 500. The sha256 is
// accepted and never shown. Anything else in the result stores nothing.
func StoredArtifacts(result string) []ArtifactAttachment {
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
		payload.SchemaVersion != 1 || payload.Status != "stored" ||
		payload.Publication != "" && payload.Publication != "pending" {
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
	return artifactType.MatchString(artifact.Type) && artifactID.MatchString(artifact.ArtifactID) &&
		artifact.Revision >= 1 && artifact.Revision <= 2147483647 &&
		boundedDisplayText(artifact.Title, 200) && boundedOptionalText(artifact.Alt, 500)
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

// ChatAttachments is how artifacts are written onto a Chat message.
func ChatAttachments(artifacts []ArtifactAttachment) []getstream.Attachment {
	attachments := make([]getstream.Attachment, 0, len(artifacts))
	for _, artifact := range artifacts {
		custom := map[string]any{"artifact_id": artifact.ArtifactID, "revision": artifact.Revision}
		if artifact.Alt != "" {
			custom["alt"] = artifact.Alt
		}
		attachments = append(attachments, getstream.Attachment{Type: &artifact.Type, Title: &artifact.Title, Custom: custom})
	}
	return attachments
}

func runtimeOf(message Message) runtimeMessage {
	runtime := runtimeMessage{
		CommandID: message.CommandID, TurnID: message.TurnID, QuestionID: message.QuestionID,
		Role: message.Role, State: message.State, StartedAt: message.StartedAt,
		TextLayout: message.TextLayout, AnswerStart: message.AnswerStart,
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
	if err != nil {
		return false
	}
	rawHost := strings.ToLower(parsed.Hostname())
	host := strings.TrimSuffix(rawHost, ".")
	return displayID.MatchString(source.ID) &&
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

func startedAt(display displayTool) time.Time {
	if display.StartedAt == nil {
		return time.Time{}
	}
	return *display.StartedAt
}

func sameDisplay(a, b displayTool) bool {
	return a.ID == b.ID && a.Name == b.Name && a.Status == b.Status && a.Summary == b.Summary &&
		a.StartedAt == b.StartedAt && a.FinishedAt == b.FinishedAt && a.DurationMS == b.DurationMS
}

func validTiming(tool displayTool) bool {
	if tool.DurationMS < 0 {
		return false
	}
	return tool.StartedAt == nil || tool.FinishedAt == nil || !tool.FinishedAt.Before(*tool.StartedAt)
}
