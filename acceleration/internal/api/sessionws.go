package api

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
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

// upgrader accepts any origin, because this is a server-to-server API reached with a
// customer header rather than a browser session cookie.
var upgrader = websocket.Upgrader{
	CheckOrigin: func(*http.Request) bool { return true },
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
	customerID, ok := CustomerFrom(r.Context())
	if !ok {
		writeError(w, http.StatusUnauthorized, "the "+CustomerHeader+" header is required")
		return
	}
	if s.sessions == nil {
		writeError(w, http.StatusNotFound, noSessions)
		return
	}
	found, ok := s.sessions.Get(r.PathValue("id"), customerID)
	if !ok {
		writeError(w, http.StatusNotFound, unknownSession)
		return
	}

	// Watching starts before the upgrade so nothing said between the two is missed.
	events, detach := found.Watch()
	defer detach()

	connection, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		// Upgrade has already written its own response, so there is nothing to say here
		// that the caller would see.
		s.logger.Debug("could not upgrade the session socket", "error", err)
		return
	}
	defer connection.Close()

	// Reading and writing each own the connection in one direction, which is what gorilla
	// requires: two goroutines writing to one socket interleave frames.
	go s.readCommands(connection, found)
	s.writeEvents(connection, events)
}

// writeEvents pushes the conversation to the caller until the session ends or the socket
// breaks.
func (s *Server) writeEvents(connection *websocket.Conn, events <-chan session.Event) {
	ping := time.NewTicker(pingEvery)
	defer ping.Stop()

	for {
		select {
		case event, open := <-events:
			if !open {
				connection.SetWriteDeadline(time.Now().Add(writeWait))
				connection.WriteMessage(websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, "the session ended"))
				return
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
func (s *Server) readCommands(connection *websocket.Conn, found *session.Session) {
	connection.SetReadDeadline(time.Now().Add(pongWait))
	connection.SetPongHandler(func(string) error {
		return connection.SetReadDeadline(time.Now().Add(pongWait))
	})

	for {
		var command struct {
			Type string `json:"type"`
			// ToolCallID names the call a tool_result answers.
			ToolCallID string `json:"tool_call_id"`
			// Output is what the tool returned, in words the model can use.
			Output string `json:"output"`
			// Error is what to tell the model instead, when the tool did not work.
			Error string `json:"error"`
			// Text carries say and respond.
			Text string `json:"text"`
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
			if !found.ResolveTool(command.ToolCallID, command.Output, command.Error) {
				// The commonest reason is a result for a call that already timed out,
				// which is worth a line in a log and nothing more.
				s.logger.Debug("a tool result answered nothing",
					"session", found.ID(), "call", command.ToolCallID)
			}

		case "say":
			// The work these commands start belongs to the conversation rather than to
			// the frame that asked for it, so it is not tied to this socket's lifetime.
			if err := found.Say(context.Background(), command.Text); err != nil {
				s.logger.Debug("could not say it", "session", found.ID(), "error", err)
			}

		case "respond":
			if err := found.Respond(context.Background(), command.Text); err != nil {
				s.logger.Debug("could not answer it", "session", found.ID(), "error", err)
			}

		case "interrupt":
			found.Interrupt()

		case "instructions":
			found.SetInstructions(command.Instructions)

		case "close":
			found.Close()
			return

		default:
			s.logger.Debug("ignoring an unknown command",
				"session", found.ID(), "type", command.Type)
		}
	}
}

// frameOf renders one event for the wire, reporting false for anything with no
// representation.
//
// The mapping is written out rather than reflected because the wire format is a contract
// with the SDKs: a field renamed in Go should break this function, not a client.
func frameOf(event session.Event) (frame, bool) {
	switch typed := event.(type) {
	case session.ToolCall:
		return frame{
			"type":      "tool_call",
			"id":        typed.ID,
			"name":      typed.Name,
			"arguments": typed.Arguments,
		}, true

	case agent.Joined:
		return frame{"type": "joined", "at": typed.At}, true

	case agent.Heard:
		return frame{
			"type":        "heard",
			"participant": participantOf(typed.Participant),
			"text":        typed.Text,
			"language":    typed.Language,
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
			"type":                   "responded",
			"turn_id":                typed.TurnID,
			"text":                   typed.Text,
			"time_to_first_token_ms": typed.TimeToFirstTokenMs,
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
			"llm_ttft_ms":            typed.LLMTTFTMs,
			"tts_ttfb_ms":            typed.TTSTTFBMs,
			"roundtrip_ms":           typed.RoundtripMs,
			"speech_end_to_audio_ms": typed.SpeechEndToAudioMs,
			"audio_out_ms":           typed.AudioOutMs,
			"interrupted":            typed.Interrupted,
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

	case agent.ToolRan:
		return frame{
			"type":      "tool_ran",
			"turn_id":   typed.TurnID,
			"tool":      typed.Tool,
			"arguments": typed.Arguments,
			"result":    typed.Result,
			"error":     errorText(typed.Err),
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
