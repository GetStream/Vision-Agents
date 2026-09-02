package api

import (
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
)

// defaultCapacity is how many calls a worker that did not say is assumed to take at once.
// Small, because a worker that has not told us what it can handle is one we should not pile
// calls onto.
const defaultCapacity = 4

// dispatchCalls hands arriving calls to a worker waiting for one.
//
// The direction is the point: the worker connects here and waits, because the agent runs in
// the customer's own process which this service cannot reach. It is a socket rather than
// long polling because a call that is ringing has seconds, not a poll interval.
func (s *Server) dispatchCalls(w http.ResponseWriter, r *http.Request) {
	customerID, ok := CustomerFrom(r.Context())
	if !ok {
		writeError(w, http.StatusUnauthorized, "the "+CustomerHeader+" header is required")
		return
	}
	if s.dispatch == nil {
		writeError(w, http.StatusNotFound, "this deployment does not dispatch calls")
		return
	}

	capacity := defaultCapacity
	if asked := r.URL.Query().Get("capacity"); asked != "" {
		parsed, err := strconv.Atoi(asked)
		if err != nil || parsed < 1 {
			writeError(w, http.StatusBadRequest, "capacity has to be a positive number of calls")
			return
		}
		capacity = parsed
	}

	// Registering before the upgrade would put a worker in the rotation that cannot be
	// written to yet, so a call arriving in between would be dropped rather than queued.
	connection, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		s.logger.Debug("could not upgrade the dispatch socket", "error", err)
		return
	}
	defer connection.Close()

	worker, release := s.dispatch.Register(customerID, capacity)
	defer release()

	s.logger.Info("a dispatch worker is waiting for calls",
		"customer", customerID, "worker", worker.ID, "capacity", capacity)

	// A worker that goes away is noticed by the reader, and the writer is asleep on a
	// channel until a call arrives. Without being told, it would sit there until the next
	// ping failed and hold the worker's place in the rotation for that whole time, which
	// is calls handed to a socket nobody is on the other end of.
	gone := make(chan struct{})
	go func() {
		defer close(gone)
		s.readWorker(connection, worker)
	}()
	s.writeCalls(connection, worker, gone)

	s.logger.Info("a dispatch worker stopped waiting", "worker", worker.ID)
}

// writeCalls pushes calls to the worker until it goes away, is released, or the socket
// breaks.
func (s *Server) writeCalls(connection *websocket.Conn, worker *dispatch.Worker, gone <-chan struct{}) {
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
		case call, open := <-worker.Calls():
			if !open {
				connection.SetWriteDeadline(time.Now().Add(writeWait))
				connection.WriteMessage(websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, "dispatch stopped"))
				return
			}
			connection.SetWriteDeadline(time.Now().Add(writeWait))
			if err := connection.WriteJSON(callFrame(call)); err != nil {
				// The call is already out of the pool, so there is nobody else it can be
				// given to. Losing it is worth an error rather than a debug line.
				s.logger.Error("could not hand a call to a worker",
					"worker", worker.ID, "call", call.CallID, "error", err)
				return
			}

		case <-ping.C:
			connection.SetWriteDeadline(time.Now().Add(writeWait))
			if err := connection.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}

		case <-gone:
			return
		}
	}
}

// readWorker applies what the worker sends back, which is how it is doing and whether it
// took the call.
//
// A frame it cannot read is reported and skipped rather than closing the socket: dropping a
// worker over one bad message would take the calls it is already in with it.
func (s *Server) readWorker(connection *websocket.Conn, worker *dispatch.Worker) {
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
			// Reason is why a call was rejected, in words for a log.
			Reason string `json:"reason"`
			// At echoes back a ping, so the worker can measure the round trip itself.
			At float64 `json:"at"`
		}
		if err := connection.ReadJSON(&report); err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				s.logger.Debug("dispatch socket read failed", "worker", worker.ID, "error", err)
			}
			return
		}
		connection.SetReadDeadline(time.Now().Add(pongWait))

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
			connection.SetWriteDeadline(time.Now().Add(writeWait))
			if err := connection.WriteJSON(frame{"type": "pong", "at": report.At}); err != nil {
				return
			}

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
		"call_id":       call.CallID,
		"call_type":     call.CallType,
		"called_number": call.CalledNumber,
		"caller_number": call.CallerNumber,
		"custom":        custom,
		"at":            call.At,
	}
}
