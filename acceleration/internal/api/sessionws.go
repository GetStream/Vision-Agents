package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"
	"github.com/gorilla/websocket"
	"github.com/oapi-codegen/runtime"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// writeWait bounds one frame write, so a reader that stopped reading cannot hold the
// fan-out goroutine open for the rest of the call.
const writeWait = 10 * time.Second

// pingEvery keeps the socket alive through whatever is between the two ends. A control
// channel can be quiet for minutes while the conversation carries on.
const pingEvery = 30 * time.Second

// pongWait is how long a silent peer is given before it is treated as gone. It is longer
// than the ping interval so one lost ping is not a disconnection.
const pongWait = 90 * time.Second

// newUpgrader accepts the origins the deployment named, and any request that names none.
type SessionRespondCommandType string

const (
	Respond SessionRespondCommandType = "respond"
)

// Valid indicates whether the value is a known member of the SessionRespondCommandType enum.
func (e SessionRespondCommandType) Valid() bool {
	switch e {
	case Respond:
		return true
	default:
		return false
	}
}

type SessionRespondCommand struct {
	Type      SessionRespondCommandType `json:"type" enum:"respond"`
	Text      string                    `json:"text"`
	CommandId *string                   `json:"command_id,omitempty" doc:"Required for personal persistent text conversations; reuse on retries. Text only when present." pattern:"^[A-Za-z0-9_-]{1,128}$"`
	Images    *[]ImageSource            `json:"images,omitempty"`
}

type ToolResultCommandType string

const (
	ToolResultCommandTypeToolResult ToolResultCommandType = "tool_result"
)

// Valid indicates whether the value is a known member of the ToolResultCommandType enum.
func (e ToolResultCommandType) Valid() bool {
	switch e {
	case ToolResultCommandTypeToolResult:
		return true
	default:
		return false
	}
}

type ToolResultCommand struct {
	Type       ToolResultCommandType `json:"type" enum:"tool_result"`
	ToolCallId string                `json:"tool_call_id"`
	CommandId  *string               `json:"command_id,omitempty"`
	TurnId     *string               `json:"turn_id,omitempty"`
	Output     *MessageContent       `json:"output,omitempty"`
	Error      *string               `json:"error,omitempty"`
}

// MessageContent defines model for MessageContent.
type MessageContent struct {
	union json.RawMessage
}

// MessageContent0 defines model for MessageContent.0.
type MessageContent0 = string

// MessageContent1 defines model for MessageContent.1.
type MessageContent1 = []ContentPart

// AsMessageContent0 returns the union data inside the MessageContent as a MessageContent0
func (t MessageContent) AsMessageContent0() (MessageContent0, error) {
	var body MessageContent0
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromMessageContent0 overwrites any union data inside the MessageContent as the provided MessageContent0
func (t *MessageContent) FromMessageContent0(v MessageContent0) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeMessageContent0 performs a merge with any union data inside the MessageContent, using the provided MessageContent0
func (t *MessageContent) MergeMessageContent0(v MessageContent0) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

// AsMessageContent1 returns the union data inside the MessageContent as a MessageContent1
func (t MessageContent) AsMessageContent1() (MessageContent1, error) {
	var body MessageContent1
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromMessageContent1 overwrites any union data inside the MessageContent as the provided MessageContent1
func (t *MessageContent) FromMessageContent1(v MessageContent1) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeMessageContent1 performs a merge with any union data inside the MessageContent, using the provided MessageContent1
func (t *MessageContent) MergeMessageContent1(v MessageContent1) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

func (t MessageContent) MarshalJSON() ([]byte, error) {
	b, err := t.union.MarshalJSON()
	return b, err
}

func (t *MessageContent) UnmarshalJSON(b []byte) error {
	err := t.union.UnmarshalJSON(b)
	return err
}

func (MessageContent) Schema(registry huma.Registry) *huma.Schema {
	return namedUnion(registry, "MessageContent", "",
		&huma.Schema{Type: huma.TypeString}, &huma.Schema{Type: huma.TypeArray, Items: &huma.Schema{Ref: "#/components/schemas/ContentPart"}})
}

// ContentPart defines model for ContentPart.
type ContentPart struct {
	union json.RawMessage
}

// AsTextContentPart returns the union data inside the ContentPart as a TextContentPart
func (t ContentPart) AsTextContentPart() (TextContentPart, error) {
	var body TextContentPart
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromTextContentPart overwrites any union data inside the ContentPart as the provided TextContentPart
func (t *ContentPart) FromTextContentPart(v TextContentPart) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeTextContentPart performs a merge with any union data inside the ContentPart, using the provided TextContentPart
func (t *ContentPart) MergeTextContentPart(v TextContentPart) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

// AsImageContentPart returns the union data inside the ContentPart as a ImageContentPart
func (t ContentPart) AsImageContentPart() (ImageContentPart, error) {
	var body ImageContentPart
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromImageContentPart overwrites any union data inside the ContentPart as the provided ImageContentPart
func (t *ContentPart) FromImageContentPart(v ImageContentPart) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeImageContentPart performs a merge with any union data inside the ContentPart, using the provided ImageContentPart
func (t *ContentPart) MergeImageContentPart(v ImageContentPart) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

func (t ContentPart) MarshalJSON() ([]byte, error) {
	b, err := t.union.MarshalJSON()
	return b, err
}

func (t *ContentPart) UnmarshalJSON(b []byte) error {
	err := t.union.UnmarshalJSON(b)
	return err
}

func (ContentPart) Schema(registry huma.Registry) *huma.Schema {
	return namedUnion(registry, "ContentPart", "",
		&huma.Schema{Ref: "#/components/schemas/TextContentPart"}, &huma.Schema{Ref: "#/components/schemas/ImageContentPart"})
}

type TextContentPartType string

const (
	TextContentPartTypeText TextContentPartType = "text"
)

// Valid indicates whether the value is a known member of the TextContentPartType enum.
func (e TextContentPartType) Valid() bool {
	switch e {
	case TextContentPartTypeText:
		return true
	default:
		return false
	}
}

type TextContentPart struct {
	Type TextContentPartType `json:"type" enum:"text"`
	Text string              `json:"text"`
}

type ImageContentPartType string

const (
	ImageUrl ImageContentPartType = "image_url"
)

// Valid indicates whether the value is a known member of the ImageContentPartType enum.
func (e ImageContentPartType) Valid() bool {
	switch e {
	case ImageUrl:
		return true
	default:
		return false
	}
}

type ImageContentPart struct {
	Type     ImageContentPartType `json:"type" enum:"image_url"`
	ImageUrl ImageSource          `json:"image_url"`
}

// registerSocketFrames declares the frames the session socket takes, which no operation
// names, so the spec and the clients describe them.
func registerSocketFrames(registry huma.Registry) {
	for _, frame := range []reflect.Type{reflect.TypeFor[SessionRespondCommand](), reflect.TypeFor[ToolResultCommand](), reflect.TypeFor[MessageContent](), reflect.TypeFor[ContentPart](), reflect.TypeFor[TextContentPart](), reflect.TypeFor[ImageContentPart]()} {
		registry.Schema(frame, true, "")
	}
}

// A request without an Origin header did not come from a browser, so there is no session
// for another site to ride on and nothing for this check to protect. One that does carry
// an origin is held to the same list as an ordinary cross-origin request: a socket that
// accepted every origin would be the way around the check the rest of the API makes.
func newUpgrader(allowed []string) websocket.Upgrader {
	permitted := make(map[string]struct{}, len(allowed))
	for _, origin := range allowed {
		permitted[strings.TrimSpace(origin)] = struct{}{}
	}
	_, anywhere := permitted["*"]

	return websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			origin := r.Header.Get("Origin")
			if origin == "" || anywhere {
				return true
			}
			_, named := permitted[origin]
			return named
		},
	}
}

// frame is one message in either direction. The type names the event and the rest of the
// object is that event's own fields, flattened rather than nested so a reader can switch
// on the type and decode once.
type frame map[string]any

// watchSession streams a conversation and takes the caller's answers to its tool calls.
//
// The socket is the only path where traffic runs both ways: everything else the caller can
// do is a request. It is here rather than in the generated server because an upgrade
// returns a connection and a strict handler has to return a response.
func (s *Server) watchSession(w http.ResponseWriter, r *http.Request) {
	if _, ok := CustomerFrom(r.Context()); !ok {
		writeError(w, http.StatusUnauthorized, "the "+CustomerHeader+" header is required")
		return
	}
	if s.sessions == nil {
		writeError(w, http.StatusNotFound, noSessions)
		return
	}
	found, ok := s.sessions.Get(r.PathValue("id"), OwnerFrom(r.Context()))
	if !ok || !canReadSession(r.Context(), found.Spec()) {
		writeError(w, http.StatusNotFound, unknownSession)
		return
	}

	// Watching starts before the upgrade so nothing said between the two is missed.
	watch := found.Watch
	if r.URL.Query().Get("replay_pending_tools") == "true" {
		watch = found.WatchPendingVoiceTools
	}
	events, detach := watch()
	defer detach()

	connection, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		// Upgrade has already written its own response, so there is nothing to say here
		// that the caller would see.
		s.logger.Debug("could not upgrade the session socket", "error", err)
		return
	}
	defer connection.Close()
	connection.SetReadLimit(maxSocketMessage)

	// Reading and writing each own the connection in one direction, which is what gorilla
	// requires: two goroutines writing to one socket interleave frames.
	owner := OwnerFrom(r.Context())
	gone := make(chan struct{})
	go func() {
		defer close(gone)
		s.readCommands(connection, found, owner)
	}()
	s.writeEvents(connection, events, watching(r), gone)
}

// wanted says which of the frequent frames this watcher asked for.
//
// Interim transcripts and decisions arrive several times a second, which is what a person
// watching a call wants and what an SDK holding a conversation would only have to throw
// away. Neither is worth sending to somebody who is not reading it.
type wanted struct {
	interim   bool
	decisions bool
}

func watching(r *http.Request) wanted {
	asked := r.URL.Query()
	return wanted{
		interim:   asked.Get("interim") == "true",
		decisions: asked.Get("decisions") != "false",
	}
}

func (w wanted) takes(event session.Event) bool {
	switch event.(type) {
	case agent.Hearing:
		return w.interim
	case agent.Decided:
		return w.decisions
	default:
		return true
	}
}

// writeEvents pushes the conversation to the caller until the session ends or the socket
// breaks.
func (s *Server) writeEvents(connection *websocket.Conn, events <-chan session.Event, asked wanted, gone <-chan struct{}) {
	ping := time.NewTicker(pingEvery)
	defer ping.Stop()

	for {
		select {
		case <-gone:
			return
		case event, open := <-events:
			if !open {
				connection.SetWriteDeadline(time.Now().Add(writeWait))
				connection.WriteMessage(websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, "the session ended"))
				return
			}
			if !asked.takes(event) {
				continue
			}
			encoded, ok := frameOf(event)
			if !ok {
				continue
			}
			connection.SetWriteDeadline(time.Now().Add(writeWait))
			if err := connection.WriteJSON(encoded); err != nil {
				s.logger.Debug("session socket write failed", "error", err)
				return
			}

		case <-ping.C:
			connection.SetWriteDeadline(time.Now().Add(writeWait))
			if err := connection.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// readCommands applies what the caller sends, which is tool results and the handful of
// things it can do to the conversation.
//
// A frame it cannot read is reported and skipped rather than closing the socket: dropping
// the connection over one bad message would take the tool calls in flight with it.
func (s *Server) readCommands(connection *websocket.Conn, found *session.Session, owner session.Owner) {
	connection.SetReadDeadline(time.Now().Add(pongWait))
	connection.SetPongHandler(func(string) error {
		return connection.SetReadDeadline(time.Now().Add(pongWait))
	})

	for {
		var command struct {
			Type string `json:"type"`
			// ToolCallID names the call a tool_result answers.
			ToolCallID string `json:"tool_call_id"`
			CommandID  string `json:"command_id"`
			TurnID     string `json:"turn_id"`
			// Output is a string or a parts array, which is what a tool that returns an
			// image sends.
			Output json.RawMessage `json:"output"`
			// Error is what to tell the model instead, when the tool did not work.
			Error string `json:"error"`
			// Text carries say and respond.
			Text string `json:"text"`
			// Images attach to a respond command, and become image parts on that turn.
			Images []wireImage `json:"images"`
			// Instructions carries the instructions command.
			Instructions string `json:"instructions"`
		}
		if err := connection.ReadJSON(&command); err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				s.logger.Debug("session socket read failed", "error", err)
			}
			return
		}
		connection.SetReadDeadline(time.Now().Add(pongWait))

		switch command.Type {
		case "tool_result":
			parts, err := parseToolOutput(command.Output)
			if err != nil {
				found.Report(err, "tool")
				if !resolveTool(found, command.ToolCallID, command.CommandID, command.TurnID, nil, err.Error()) {
					s.logger.Debug("a tool result answered nothing",
						"session", found.ID(), "call", command.ToolCallID)
				}
				continue
			}
			if !resolveTool(found, command.ToolCallID, command.CommandID, command.TurnID, parts, command.Error) {
				s.logger.Debug("a tool result answered nothing",
					"session", found.ID(), "call", command.ToolCallID)
			}

		case "say":
			if err := found.Say(context.Background(), command.Text); err != nil {
				s.logger.Debug("could not say it", "session", found.ID(), "error", err)
			}

		case "respond":
			if leftToDispatch(found, owner.Kind) {
				if len(command.Images) > 0 {
					found.Report(errors.New("this agent hands what is written to its server, which takes text only"), "dispatch")
					continue
				}
				if _, err := s.dispatchText(context.Background(), found, dispatch.Message{
					Text: command.Text, CommandID: command.CommandID, UserID: owner.UserID,
				}, ""); err != nil {
					found.Report(err, "dispatch")
				}
				continue
			}
			if command.CommandID != "" {
				if len(command.Images) > 0 {
					found.Report(fmt.Errorf("durable commands currently support text only"), "llm")
					continue
				}
				if _, _, err := found.RespondCommand(context.Background(), command.CommandID, command.Text, ""); err != nil {
					found.Report(err, "llm")
				}
				continue
			}
			images, err := imagesFromWire(command.Images)
			if err != nil {
				found.Report(err, "llm")
				continue
			}
			if _, err := found.Respond(context.Background(), command.Text, images); err != nil {
				found.Report(err, "llm")
			}

		case "interrupt":
			// A stop naming a command stops that command wherever it got to. Without a
			// name it stops whatever is being said now, which is what a voice caller
			// talking over the agent means.
			if command.CommandID != "" {
				if _, err := found.InterruptCommand(command.CommandID); err != nil {
					found.Report(err, "command")
				}
				continue
			}
			found.Interrupt()

		case "instructions":
			found.SetInstructions(command.Instructions)

		case "close":
			// Through the manager rather than found.Close(), which ends the conversation
			// and leaves it in the manager's map: a session closed this way stayed listed
			// as live for the rest of the process's life. Every SDK closes over the socket
			// when it is holding one, so that was every conversation, and the resource
			// surface made it visible — a query returns live sessions ahead of the rows,
			// so conversations that were over came back as running ones.
			if _, err := s.sessions.Close(found.ID(), owner); err != nil {
				s.logger.Debug("could not close the session", "session", found.ID(), "error", err)
			}
			return

		default:
			found.Report(fmt.Errorf("unknown command %q", command.Type), "command")
			s.logger.Debug("ignoring an unknown command",
				"session", found.ID(), "type", command.Type)
		}
	}
}

func resolveTool(found *session.Session, callID, commandID, turnID string, parts []llm.ContentPart, failure string) bool {
	if commandID != "" {
		return found.ResolveCommandTool(callID, commandID, turnID, parts, failure)
	}
	if turnID != "" {
		return found.ResolveTurnTool(callID, turnID, parts, failure)
	}
	return found.ResolveToolParts(callID, parts, failure)
}

// frameOf renders one event for the wire, reporting false for anything with no
// representation.
//
// The mapping is written out rather than reflected because the wire format is a contract
// with the SDKs: a field renamed in Go should break this function, not a client.
func frameOf(event session.Event) (frame, bool) {
	switch typed := event.(type) {
	case session.ToolCall:
		if typed.Cancel {
			return frame{"type": "tool_cancel", "id": typed.ID, "command_id": typed.CommandID, "turn_id": typed.TurnID}, true
		}
		return frame{
			"type":       "tool_call",
			"id":         typed.ID,
			"name":       typed.Name,
			"arguments":  typed.Arguments,
			"command_id": typed.CommandID,
			"turn_id":    typed.TurnID,
		}, true

	case agent.Joined:
		return frame{"type": "joined", "at": typed.At}, true

	case agent.ParticipantJoined:
		return frame{
			"type":        "participant_joined",
			"participant": participantOf(typed.Participant),
			"at":          typed.At,
		}, true

	case agent.ParticipantLeft:
		return frame{
			"type":        "participant_left",
			"participant": participantOf(typed.Participant),
			"at":          typed.At,
		}, true

	case agent.Hearing:
		return frame{
			"type":        "hearing",
			"participant": participantOf(typed.Participant),
			"text":        typed.Text,
			"language":    typed.Language,
		}, true

	case agent.Heard:
		return frame{
			"type":        "heard",
			"participant": participantOf(typed.Participant),
			"text":        typed.Text,
			"language":    typed.Language,
		}, true

	case agent.Decided:
		return frame{
			"type":        "decision",
			"at":          typed.At,
			"kind":        typed.Kind,
			"reason":      typed.Reason,
			"turn_id":     typed.TurnID,
			"participant": participantOf(typed.Participant),
			"said":        typed.Text,
			"latency_ms":  typed.LatencyMs,
		}, true

	case agent.Responding:
		return frame{
			"type":        "responding",
			"turn_id":     typed.TurnID,
			"participant": participantOf(typed.Participant),
			"prompt":      typed.Prompt,
		}, true

	case agent.ResponseDelta:
		return frame{"type": "response_delta", "turn_id": typed.TurnID, "text": typed.Text}, true

	case agent.Responded:
		return frame{
			"pending_work":           typed.PendingWork,
			"type":                   "responded",
			"turn_id":                typed.TurnID,
			"text":                   typed.Text,
			"time_to_first_token_ms": typed.TimeToFirstTokenMs,
		}, true

	case agent.Blocked:
		return frame{
			"type":        "blocked",
			"turn_id":     typed.TurnID,
			"reason":      typed.Reason,
			"probability": typed.Probability,
			"held_ms":     typed.HeldMs,
		}, true

	case agent.Spoke:
		return frame{
			"type":                  "spoke",
			"turn_id":               typed.TurnID,
			"audio_duration_ms":     typed.AudioDurationMs,
			"time_to_first_byte_ms": typed.TimeToFirstByteMs,
		}, true

	case agent.Turn:
		return frame{
			"type":                   "turn",
			"turn_id":                typed.TurnID,
			"participant":            participantOf(typed.Participant),
			"started_at":             typed.StartedAt,
			"stt_latency_ms":         typed.STTLatencyMs,
			"cadence_ms":             typed.CadenceMs,
			"decision_ms":            typed.DecisionMs,
			"model_to_first_text_ms": typed.ModelToFirstTextMs,
			"text_to_tts_ms":         typed.TextToTTSMs,
			"tts_to_audio_ms":        typed.TTSToAudioMs,
			"llm_ttft_ms":            typed.LLMTTFTMs,
			"tts_ttfb_ms":            typed.TTSTTFBMs,
			"roundtrip_ms":           typed.RoundtripMs,
			"speech_end_to_audio_ms": typed.SpeechEndToAudioMs,
			"audio_out_ms":           typed.AudioOutMs,
			"interrupted":            typed.Interrupted,
		}, true

	case agent.ModelCall:
		return frame{
			"type": "model_call", "operation_id": typed.OperationID,
			"purpose": typed.Purpose, "turn_id": typed.TurnID,
			"provider": typed.Provider, "model": typed.Model,
			"ttft_ms": typed.TTFTMs, "duration_ms": typed.DurationMs,
			"success": typed.Success,
		}, true

	case agent.Delegated:
		return frame{
			"type":    "delegated",
			"task_id": typed.TaskID,
			"skill":   typed.Skill,
			"prompt":  typed.Prompt,
			"turn_id": typed.TurnID,
		}, true

	case agent.TaskSettled:
		return frame{
			"type":       "task_settled",
			"evidence":   typed.Evidence,
			"task_id":    typed.TaskID,
			"skill":      typed.Skill,
			"text":       typed.Text,
			"question":   typed.Question,
			"elapsed_ms": typed.ElapsedMs,
			"error":      errorText(typed.Err),
		}, true

	case agent.TaskCancelled:
		return frame{
			"type":    "task_cancelled",
			"task_id": typed.TaskID,
			"skill":   typed.Skill,
			"reason":  typed.Reason,
		}, true

	case conversation.CommandReceipt:
		return frame{"type": "command_accepted", "command": typed}, true
	case session.CommandStopped:
		return frame{"type": "command_stopped", "command": typed.CommandReceipt}, true
	case conversation.Updated:
		return frame{"type": "conversation_updated", "conversation_id": typed.CID, "message": typed.Message}, true
	case agent.ToolStarted:
		return frame{"type": "tool_started", "tool_call_id": typed.ID, "tool": typed.Tool, "turn_id": typed.TurnID, "started_at": typed.StartedAt}, true
	case agent.ToolRan:
		return frame{
			"type":         "tool_ran",
			"tool_call_id": typed.ID,
			"turn_id":      typed.TurnID,
			"tool":         typed.Tool,
			"arguments":    typed.Arguments,
			"result":       typed.Result,
			"error":        errorText(typed.Err),
		}, true

	case agent.Transferred:
		return frame{
			"type":    "transferred",
			"turn_id": typed.TurnID,
			"to":      typed.To,
			"summary": typed.Summary,
		}, true

	case agent.Pressed:
		return frame{"type": "pressed", "turn_id": typed.TurnID, "digits": typed.Digits}, true

	case agent.LookedUp:
		return frame{
			"type":      "looked_up",
			"turn_id":   typed.TurnID,
			"query":     typed.Query,
			"documents": typed.Documents,
		}, true

	case agent.Backchannel:
		return frame{
			"type":        "backchannel",
			"participant": participantOf(typed.Participant),
			"text":        typed.Text,
		}, true

	case agent.Interrupted:
		return frame{
			"type":        "interrupted",
			"turn_id":     typed.TurnID,
			"participant": participantOf(typed.Participant),
		}, true

	case agent.OverlapDecided:
		return frame{
			"type":        "overlap_decided",
			"turn_id":     typed.TurnID,
			"participant": participantOf(typed.Participant),
			"action":      typed.Action,
		}, true

	case agent.ConversationCompacted:
		return frame{
			"type":    "conversation_compacted",
			"before":  typed.Before,
			"after":   typed.After,
			"summary": typed.Summary,
		}, true

	case agent.Error:
		return frame{"type": "error", "context": typed.Context, "error": errorText(typed.Err)}, true

	case agent.Left:
		return frame{"type": "left", "at": typed.At}, true

	case agent.ModelsChanged:
		mode := "cascade"
		if typed.Native {
			mode = "native"
		}
		return frame{
			"type":     "models_changed",
			"at":       typed.At,
			"mode":     mode,
			"llm":      typed.LLM,
			"stt":      typed.STT,
			"tts":      typed.TTS,
			"sts":      typed.STS,
			"subagent": typed.Subagent,
			"voice":    typed.Voice,
		}, true

	default:
		return nil, false
	}
}

func participantOf(participant stt.Participant) frame {
	return frame{
		"id":      participant.ID,
		"user_id": participant.UserID,
		"name":    participant.Name,
	}
}

// errorText renders a failure as the empty string when there was none, so a client can
// read one field rather than checking whether it is there.
func errorText(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

// writeError reports a failure that happened before the upgrade, in the same shape as the
// rest of the API so a client has one error format to read.
func writeError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(Error{Error: message})
}
