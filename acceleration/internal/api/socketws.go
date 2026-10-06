package api

import (
	"encoding/json"
	"errors"
	"net/http"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/socketedge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// socketStart is the first frame on a session socket.
type socketStart struct {
	Type string `json:"type"`
	// SampleRate is the rate of the PCM16 mono audio carried both ways. Zero means 16 kHz.
	SampleRate int `json:"sample_rate"`
	// Session is what POST /v1/agents/sessions takes. call_id may be left out: there is no
	// call to join, so one is made up for the records.
	Session CreateSessionRequest `json:"session"`
}

// openSocketSession holds a voice conversation over the socket itself rather than a call.
//
// The caller sends a `start` frame naming the session, the way POST /v1/agents/sessions
// does, and is answered with a `session` frame carrying it. From then on binary frames are
// PCM16 mono at the start frame's rate in both directions: the caller's audio in, and the
// agent's speech out at the pace it would be heard on a call. A `cleared` frame says speech
// already sent was thrown away because the caller cut in. Tool calls and everything else the
// conversation emits go over the session's own events socket, as they would for a call.
//
// The session lasts as long as the socket: closing it, or sending `stop`, ends the
// conversation, and a conversation that ends closes the socket.
//
// It is a socket rather than a field on POST /v1/agents/sessions because the audio has to
// arrive on the connection that holds the session open.
func (s *Server) openSocketSession(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		writeError(w, missingCustomer)
		return
	}
	if s.sessions == nil {
		writeError(w, noSessions)
		return
	}

	connection, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		s.logger.Debug("could not upgrade the session socket", "error", err)
		return
	}
	defer connection.Close()
	connection.SetReadLimit(maxSocketMessage)

	// gorilla allows one writer at a time, and the agent's speech, the cleared notices and
	// this handler all write.
	var writing sync.Mutex
	write := func(kind int, payload []byte) error {
		writing.Lock()
		defer writing.Unlock()
		_ = connection.SetWriteDeadline(time.Now().Add(writeWait))
		return stack.Wrap(connection.WriteMessage(kind, payload))
	}
	writeFrame := func(f frame) error {
		payload, err := json.Marshal(f)
		if err != nil {
			return stack.Wrap(err)
		}
		return stack.Wrap(write(websocket.TextMessage, payload))
	}
	refuse := func(message string) {
		_ = writeFrame(frame{"type": "error", "error": message})
		_ = write(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.ClosePolicyViolation, ""))
	}

	_ = connection.SetReadDeadline(time.Now().Add(startWait))
	var start socketStart
	if err := connection.ReadJSON(&start); err != nil || start.Type != "start" {
		refuse("the first frame must be a start frame naming the session")
		return
	}
	_ = connection.SetReadDeadline(time.Time{})

	config, failure := s.configFor(ctx, customerID, start.Session.ConfigId, start.Session.Agent)
	if failure != nil {
		refuse(failure.Error())
		return
	}
	if value(start.Session.Text) {
		refuse("a socket session carries audio, so it cannot be a text session")
		return
	}
	if value(start.Session.CallId) == "" {
		callID := "socket-" + uuid.NewString()
		start.Session.CallId = &callID
	}

	edge := socketedge.New(socketedge.Options{
		SampleRate: start.SampleRate,
		Caller:     stt.Participant{ID: "caller", Name: "Caller"},
		Send:       func(pcm []byte) error { return write(websocket.BinaryMessage, pcm) },
		Cleared:    func() { _ = writeFrame(frame{"type": "cleared"}) },
	})
	spec := specOf(start.Session, customerID, config)
	spec.Caller = CallerFrom(ctx)
	spec.CallerKind = KindFrom(ctx)
	spec.Edge = edge

	created, err := s.sessions.Create(ctx, spec)
	if errors.Is(err, session.ErrSessionExists) || err != nil {
		refuse(err.Error())
		return
	}
	owner := OwnerFrom(ctx)
	defer func() {
		if _, err := s.sessions.Close(created.ID(), owner); err != nil {
			s.logger.Debug("could not close the socket session", "session", created.ID(), "error", err)
		}
	}()

	rate := start.SampleRate
	if rate <= 0 {
		rate = stt.SampleRate
	}
	if err := writeFrame(frame{"type": "session", "session": sessionOf(created), "sample_rate": rate}); err != nil {
		return
	}

	// The conversation ending is the other way the socket closes: the agent hangs up, and
	// reading stops when the connection does.
	events, detach := created.Watch()
	defer detach()
	go func() {
		for range events {
		}
		_ = write(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, "the conversation is over"))
		_ = connection.Close()
	}()

	for {
		kind, payload, err := connection.ReadMessage()
		if err != nil {
			return
		}
		switch kind {
		case websocket.BinaryMessage:
			if err := edge.Hear(payload); err != nil {
				return
			}
		case websocket.TextMessage:
			var command struct {
				Type string `json:"type"`
			}
			if json.Unmarshal(payload, &command) == nil && command.Type == "stop" {
				return
			}
		}
	}
}
