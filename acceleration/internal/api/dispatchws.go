package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
)

// defaultCapacity is how many calls a worker that did not say is assumed to take at once.
// Small, because a worker that has not told us what it can handle is one we should not pile
// calls onto.
const defaultCapacity = 4

// maxCapacity is the most work one worker may say it takes at once. The router queues that
// much for it up front, so an unbounded number is memory the router cannot have. A worker's
// capacity is the agents its own process runs at once. The whole router holds 512 voice
// agents on two full nodes (performance_benchmark/findings.md, "What 512 is"), so this leaves
// one worker process twice that, where the SDKs ask for 4.
const maxCapacity = 1024

// dispatchCalls hands arriving calls to a worker waiting for one.
//
// The direction is the point: the worker connects here and waits, because the agent runs in
// the customer's own process which this service cannot reach. It is a socket rather than
// long polling because a call that is ringing has seconds, not a poll interval.
//
// Server-side only, and this is the clearest case of why: a worker is offered other
// people's callers, so anything that can open this socket can answer for the whole app.
// socketRoutes already refuses a client-side caller in the middleware; the same question
// is asked again here because this is the socket where being wrong is worst, and a second
// check costs one comparison.
func (s *Server) dispatchCalls(w http.ResponseWriter, r *http.Request) {
	customerID, ok := CustomerFrom(r.Context())
	if !ok {
		writeError(w, errMissingCustomer)
		return
	}
	if s.refuseClientSide(w, r) {
		return
	}
	if s.dispatch == nil {
		writeError(w, notFound("this deployment does not dispatch calls"))
		return
	}

	registration, failure := registrationOf(r)
	if failure != "" {
		writeError(w, invalidRequest(failure))
		return
	}

	// Registering before the upgrade would put a worker in the rotation that cannot be
	// written to yet, so a call arriving in between would be dropped rather than queued.
	connection, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		s.logger.Debug("could not upgrade the dispatch socket", "error", err)
		return
	}
	defer connection.Close()

	worker, release := s.dispatch.Register(customerID, registration)
	defer func() {
		hosted := s.dispatch.AgentsHostedBy(worker)
		release()
		if s.workers == nil {
			return
		}
		for _, agentID := range hosted {
			s.workers.Offer(r.Context(), customerID, agentID)
		}
	}()
	if s.directory != nil {
		defer s.directory.Wait(r.Context(), customerID)()
	}

	s.logger.Info("a dispatch worker is waiting for calls",
		"customer", customerID, "worker", worker.ID, "capacity", registration.Capacity,
		"carried", registration.Active, "handles", registration.Handles)

	// A worker that goes away is noticed by the reader, and the writer is asleep on a
	// channel until a call arrives. Without being told, it would sit there until the next
	// ping failed and hold the worker's place in the rotation for that whole time, which
	// is calls handed to a socket nobody is on the other end of.
	gone := make(chan error, 1)
	pongs := make(chan float64)
	replies := make(chan frame)
	writerDone := make(chan struct{})
	stopTaking := sync.OnceFunc(func() { close(writerDone) })
	defer stopTaking()
	go func() {
		gone <- s.readWorker(connection, worker, pongs, replies, writerDone)
	}()
	s.writeCalls(connection, worker, gone, pongs, replies, stopTaking)

	s.logger.Info("a dispatch worker stopped waiting", "worker", worker.ID)
}

// registrationOf reads what a worker says about itself off its query string, returning why
// it cannot be registered if anything in there is not something a worker can be.
//
// Saying `active` at all, even as zero, is what says the worker reports each piece of work
// finished. A worker that says nothing is one built against an older router, which can only
// be held to its queue.
func registrationOf(r *http.Request) (dispatch.Registration, string) {
	asked := r.URL.Query()
	registration := dispatch.Registration{Capacity: defaultCapacity}

	if capacity := asked.Get("capacity"); capacity != "" {
		parsed, err := strconv.Atoi(capacity)
		if err != nil || parsed < 1 {
			return registration, "capacity has to be a positive number of calls"
		}
		if parsed > maxCapacity {
			return registration, "capacity can be at most " + strconv.Itoa(maxCapacity) + " calls"
		}
		registration.Capacity = parsed
	}
	if asked.Has("active") {
		parsed, err := strconv.Atoi(asked.Get("active"))
		if err != nil || parsed < 0 {
			return registration, "active has to be the amount of work this worker is already holding"
		}
		registration.Active = parsed
		registration.Tracking = true
	}
	if asked.Has("handles") {
		// Present but naming nothing is a worker that takes no work at all, which a process
		// hosting tools and answering neither calls nor messages is. Absent is a worker that
		// takes everything, which is what one built against a router that never asked means.
		registration.Handles = []dispatch.Kind{}
		for kind := range strings.SplitSeq(asked.Get("handles"), ",") {
			switch dispatch.Kind(kind) {
			case "":
			case dispatch.Calls, dispatch.Messages:
				registration.Handles = append(registration.Handles, dispatch.Kind(kind))
			default:
				return registration, "handles takes " + string(dispatch.Calls) + ", " + string(dispatch.Messages) + ", or both"
			}
		}
	}
	return registration, ""
}

// writeCalls pushes calls and messages to the worker until it goes away, is released, or
// the socket breaks. gone carries why the reader stopped, and stopTaking tells
// the reader the writer takes no more pongs or replies from it.
func (s *Server) writeCalls(connection *websocket.Conn, worker *dispatch.Worker, gone <-chan error, pongs <-chan float64, replies <-chan frame, stopTaking func()) {
	// A reader left handing over a pong would never read the worker's answer to the close.
	end := func(code int, reason string) {
		stopTaking()
		closeBeforeReader(connection, code, reason, gone)
	}
	connection.SetWriteDeadline(time.Now().Add(writeWait))
	ready := frame{"type": "ready", "worker_id": worker.ID}
	if err := connection.WriteJSON(ready); err != nil {
		s.logger.Debug("dispatch socket write failed", "worker", worker.ID, "error", err)
		return
	}

	ping := time.NewTicker(pingEvery)
	defer ping.Stop()

	for {
		select {
		case at := <-pongs:
			connection.SetWriteDeadline(time.Now().Add(writeWait))
			if err := connection.WriteJSON(frame{"type": "pong", "at": at}); err != nil {
				return
			}

		case reply := <-replies:
			connection.SetWriteDeadline(time.Now().Add(writeWait))
			if err := connection.WriteJSON(reply); err != nil {
				return
			}

		case call := <-worker.ToolCalls():
			connection.SetWriteDeadline(time.Now().Add(writeWait))
			if err := connection.WriteJSON(toolCallFrame(call)); err != nil {
				// The session waiting on it is failed when the worker is released, which
				// returning here leads to.
				s.logger.Error("could not hand a tool call to a worker",
					"worker", worker.ID, "tool", call.Name, "error", err)
				return
			}

		case call := <-worker.Calls():
			connection.SetWriteDeadline(time.Now().Add(writeWait))
			if err := connection.WriteJSON(callFrame(call)); err != nil {
				// The call is already out of the pool, so there is nobody else it can be
				// given to. Losing it is worth an error rather than a debug line.
				s.logger.Error("could not hand a call to a worker",
					"worker", worker.ID, "call", call.CallID, "error", err)
				return
			}

		case message := <-worker.Messages():
			connection.SetWriteDeadline(time.Now().Add(writeWait))
			if err := connection.WriteJSON(messageFrame(message)); err != nil {
				s.logger.Error("could not hand a message to a worker",
					"worker", worker.ID, "channel", message.ChannelID, "error", err)
				return
			}

		case <-ping.C:
			connection.SetWriteDeadline(time.Now().Add(writeWait))
			if err := connection.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}

		case <-s.draining:
			end(websocket.CloseGoingAway, "router restarting")
			return

		case stopped := <-gone:
			if errors.Is(stopped, errUnreadableFrame) {
				refuseUnreadable(connection)
			}
			return
		}
	}
}

// readWorker applies what the worker sends back, which is how it is doing and whether it
// took the call.
//
// A message of a type it does not know is reported and skipped rather than closing the
// socket: dropping a worker over one bad message would take the calls it is already in with
// it. A frame that is not JSON at all returns errUnreadableFrame.
func (s *Server) readWorker(connection *websocket.Conn, worker *dispatch.Worker, pongs chan<- float64, replies chan<- frame, writerDone <-chan struct{}) error {
	connection.SetReadDeadline(time.Now().Add(pongWait))
	connection.SetPongHandler(func(string) error {
		return connection.SetReadDeadline(time.Now().Add(pongWait))
	})

	for {
		var report struct {
			Type string `json:"type"`
			// ActiveAgents, CPUPercent, MemoryPercent and LatencyMs carry load.
			ActiveAgents  int     `json:"active_agents"`
			CPUPercent    float64 `json:"cpu_percent"`
			MemoryPercent float64 `json:"memory_percent"`
			LatencyMs     float64 `json:"latency_ms"`
			// CallID names the call accepted or rejected answers for.
			CallID string `json:"call_id"`
			// WorkID names the piece of work done reports finished.
			WorkID string `json:"work_id"`
			// Reason is why a call was rejected, in words for a log.
			Reason string `json:"reason"`
			// At echoes back a ping, so the worker can measure the round trip itself.
			At float64 `json:"at"`
			// AgentID, Tools and TimeoutMs are what a worker offers to run for every
			// session opened under one agent id.
			AgentID   string       `json:"agent_id"`
			Tools     []hostedTool `json:"tools"`
			TimeoutMs int          `json:"timeout_ms"`
			// ToolCallID, Output and Error answer one hosted tool call, as a session's
			// tool_result does: Output is a string or a parts array. ID is ToolCallID under
			// the name workers sent it by before.
			ToolCallID string          `json:"tool_call_id"`
			ID         string          `json:"id"`
			Output     json.RawMessage `json:"output"`
			Error      string          `json:"error"`
			// Text and Verbatim say how a running hosted call is going, as a session's
			// tool_update does.
			Text     string `json:"text"`
			Verbatim bool   `json:"verbatim"`
		}
		_, payload, err := connection.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				s.logger.Debug("dispatch socket read failed", "worker", worker.ID, "error", err)
			}
			return nil
		}
		connection.SetReadDeadline(time.Now().Add(pongWait))
		if json.Unmarshal(payload, &report) != nil {
			return errUnreadableFrame
		}

		switch report.Type {
		case "load":
			worker.Report(dispatch.Load{
				ActiveAgents:  report.ActiveAgents,
				CPUPercent:    report.CPUPercent,
				MemoryPercent: report.MemoryPercent,
				LatencyMs:     report.LatencyMs,
			})

		case "ping":
			// The worker times its own round trip, because the network it is on is the one
			// that will carry the audio. All this does is send the timestamp back.
			// Keep all data writes in writeCalls, including their deadlines.
			select {
			case pongs <- report.At:
			case <-writerDone:
				return nil
			}

		case "host_tools":
			reply := s.hostTools(worker, report.AgentID, report.Tools, report.TimeoutMs)
			select {
			case replies <- reply:
			case <-writerDone:
				return nil
			}

		case "tool_result":
			id := report.ToolCallID
			if id == "" {
				id = report.ID
			}
			result := dispatch.ToolResult{Failure: report.Error}
			if parts, err := parseToolOutput(report.Output); err != nil {
				result.Failure = "the tool's output could not be read: " + err.Error()
			} else {
				result.Output = parts
			}
			if !worker.Resolve(id, result) {
				s.logger.Debug("a hosted tool answered a call nobody is waiting on",
					"worker", worker.ID, "id", id)
			}

		case "tool_update":
			worker.Update(report.ToolCallID, report.Text, report.Verbatim)

		case "tool_hold":
			// The conversation may be on another node, so the answer is a round trip that
			// the worker's other frames are not held up behind.
			go func(id string) {
				reply := frame{"type": "tool_held", "id": id}
				if sessionID, held := worker.Hold(id); !held {
					reply = frame{"type": "tool_cancel", "id": id, "session_id": sessionID}
				}
				select {
				case replies <- reply:
				case <-writerDone:
				}
			}(report.ToolCallID)

		case "done":
			// What the worker is holding is what it has been handed and not yet reported,
			// so this is where its room for the next piece of work comes back.
			if report.Error != "" {
				s.logger.Error("a worker could not do the work it was handed",
					"worker", worker.ID, "work", report.WorkID, "error", report.Error)
			}
			worker.Done(report.WorkID)

		case "accepted":
			s.logger.Debug("a worker took a call", "worker", worker.ID, "call", report.CallID)

		case "rejected":
			// Nothing is re-assigned here. A worker that says no has already had the call
			// taken out of the pool, and giving it to somebody else after the caller has
			// been waiting is worse than the log line saying so.
			s.logger.Error("a worker refused a call",
				"worker", worker.ID, "call", report.CallID, "reason", report.Reason)

		default:
			s.logger.Debug("ignoring an unknown dispatch message",
				"worker", worker.ID, "type", report.Type)
		}
	}
}

// hostedTool is one tool as a worker declares it.
type hostedTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
	// Interruptible and TimeoutMs mean what they do on a session's own tool.
	Interruptible bool `json:"interruptible"`
	TimeoutMs     int  `json:"timeout_ms"`
	// Approval is the question a voice caller is asked before each call runs, as on a
	// session's own tool. Only its title is asked.
	Approval *hostedApproval `json:"approval"`
}

// hostedApproval is the question a hosted tool asks.
type hostedApproval struct {
	Title string `json:"title"`
}

// hostTools records what a worker offers to run for sessions under an agent id, and says
// whether it was taken.
//
// The offer is scoped to the worker's own customer, which is what makes it safe: a worker
// is trusted with its customer's conversations already, and naming an agent is not a way
// into anybody else's. The agent id need not be stored anywhere — the sessions this is for
// are the ones a browser opens naming an agent and no config.
func (s *Server) hostTools(worker *dispatch.Worker, agentID string, declared []hostedTool, timeoutMs int) frame {
	refuse := func(reason string) frame {
		s.logger.Warn("refused a worker's hosted tools", "worker", worker.ID, "agent", agentID, "reason", reason)
		return frame{"type": "hosting_refused", "agent_id": agentID, "reason": reason}
	}
	tools := make([]dispatch.Tool, 0, len(declared))
	names := make([]string, 0, len(declared))
	for _, tool := range declared {
		if tool.Name == "" || tool.Description == "" {
			return refuse("every hosted tool needs a name and a description")
		}
		if tool.TimeoutMs < 0 {
			return refuse(tool.Name + "'s timeout_ms cannot be negative")
		}
		hosted := dispatch.Tool{
			Name: tool.Name, Description: tool.Description, Parameters: tool.Parameters,
			Interruptible: tool.Interruptible, Timeout: time.Duration(tool.TimeoutMs) * time.Millisecond,
		}
		if tool.Approval != nil {
			if hosted.Approval = strings.TrimSpace(tool.Approval.Title); hosted.Approval == "" {
				return refuse(tool.Name + "'s approval needs a title")
			}
		}
		tools = append(tools, hosted)
		names = append(names, tool.Name)
	}
	if err := s.dispatch.Host(worker, agentID, tools, time.Duration(timeoutMs)*time.Millisecond); err != nil {
		return refuse(err.Error())
	}
	if s.workers != nil {
		s.workers.Offer(context.Background(), worker.CustomerID, agentID)
	}
	s.logger.Info("a worker hosts tools", "worker", worker.ID, "agent", agentID, "tools", names)
	return frame{"type": "hosting", "agent_id": agentID, "tools": names}
}

// toolCallFrame renders one hosted call, or the cancelling of one, for the wire.
func toolCallFrame(call dispatch.ToolCall) frame {
	if call.Cancel {
		return frame{"type": "tool_cancel", "id": call.ID, "session_id": call.SessionID}
	}
	rendered := frame{
		"type":       "tool_call",
		"id":         call.ID,
		"session_id": call.SessionID,
		"turn_id":    call.TurnID,
		"name":       call.Name,
		"arguments":  call.Arguments,
	}
	if call.User != nil {
		rendered["user"] = toolUserFrame(call.User.ID, call.User.Name)
	}
	if call.Conversation != nil {
		rendered["conversation"] = toolConversationFrame(call.Conversation.CID, call.Conversation.Custom)
	}
	if call.Approved {
		rendered["approved"] = true
	}
	return rendered
}

// toolUserFrame is who a tool call was made for, as both sockets carry it.
func toolUserFrame(id, name string) frame {
	user := frame{"id": id}
	if name != "" {
		user["name"] = name
	}
	return user
}

// toolConversationFrame is the conversation a tool call was made in, as both sockets carry
// it. custom is always an object, so a client reads an empty one rather than a missing one.
func toolConversationFrame(cid string, custom map[string]any) frame {
	if custom == nil {
		custom = map[string]any{}
	}
	conversation := frame{"custom": custom}
	if cid != "" {
		conversation["cid"] = cid
	}
	return conversation
}

// callFrame renders one arriving call for the wire.
//
// Written out rather than reflected for the same reason the session frames are: this is a
// contract with the SDKs, and a field renamed in Go should break this function rather than a
// client.
func callFrame(call dispatch.Call) frame {
	custom := call.Custom
	if custom == nil {
		// A client reading an absent field and a client reading an empty object should not
		// have to differ.
		custom = map[string]string{}
	}
	return frame{
		"type":          "call",
		"work_id":       call.WorkID,
		"call_id":       call.CallID,
		"call_type":     call.CallType,
		"session_id":    call.SessionID,
		"called_number": call.CalledNumber,
		"caller_number": call.CallerNumber,
		"custom":        custom,
		"at":            call.At,
	}
}

// messageFrame renders one arriving message for the wire, spelled out for the same reason
// callFrame is.
func messageFrame(message dispatch.Message) frame {
	custom := message.Custom
	if custom == nil {
		custom = map[string]string{}
	}
	return frame{
		"type":         "message",
		"work_id":      message.WorkID,
		"channel_type": message.ChannelType,
		"channel_id":   message.ChannelID,
		"agent_id":     message.AgentID,
		"config_id":    message.ConfigID,
		"session_id":   message.SessionID,
		"request_id":   message.CommandID,
		"custom":       custom,
		"text":         message.Text,
		"message_id":   message.MessageID,
		"user_id":      message.UserID,
		"user_name":    message.UserName,
		"at":           message.At,
	}
}
