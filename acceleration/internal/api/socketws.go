package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/socketedge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// socketStart is the first frame on a session socket.
type socketStart struct {
	Type string `json:"type"`
	// SampleRate is the rate of the PCM16 mono audio carried both ways, from
	// socketedge.MinSampleRate to socketedge.MaxSampleRate. Zero means 16 kHz.
	SampleRate int `json:"sample_rate"`
	// Session is what POST /v1/agents/sessions takes. call_id may be left out: there is no
	// call to join, so one is made up for the records.
	Session CreateSessionRequest `json:"session"`
}

// errSocketCall refuses a call named in a socket session's start frame: the socket carries
// the audio, so the session joins no Stream call.
var errSocketCall = invalidRequest("a socket session carries its own audio and joins no Stream call, so it takes no call")

// errSocketSampleRate refuses a start frame's sample_rate that the socket does not carry.
var errSocketSampleRate = invalidRequest(fmt.Sprintf(
	"sample_rate must be from %d to %d, or left out for 16000", socketedge.MinSampleRate, socketedge.MaxSampleRate))

// errSocketVoiceEnded refuses voice on a socket session whose voice was stopped: the socket
// was its call, and stopping voice hung that call up.
var errSocketVoiceEnded = invalidRequest("voice on a socket session cannot be started again once stopped: open a new socket")

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
		writeError(w, errMissingCustomer)
		return
	}
	if s.sessions == nil {
		writeError(w, errNoSessions)
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
	// A refusal ends the socket with closeSocket: nothing else reads it yet, and the caller
	// is held to answer the close so the error frame before it is not lost.
	refuse := func(message string) {
		_ = writeFrame(frame{"type": "error", "error": message})
		closeSocket(connection, websocket.ClosePolicyViolation, "")
	}
	// refuseWith also names the failure's code and type, which are what
	// POST /v1/agents/sessions answers the same failure with.
	refuseWith := func(failure APIError) {
		_ = writeFrame(frame{"type": "error", "error": failure.Message, "code": failure.Code, "error_type": failure.Type})
		closeSocket(connection, websocket.ClosePolicyViolation, "")
	}

	_ = connection.SetReadDeadline(time.Now().Add(startWait))
	var start socketStart
	if err := connection.ReadJSON(&start); err != nil || start.Type != "start" {
		refuse("the first frame must be a start frame naming the session")
		return
	}
	_ = connection.SetReadDeadline(time.Time{})

	// The start frame carries POST /v1/agents/sessions's body, so a request is refused here
	// as it is refused there.
	if failure, refused := refuseSessionRequest(ctx, start.Session); refused {
		refuseWith(failure)
		return
	}
	if start.Session.Call != nil {
		refuseWith(errSocketCall)
		return
	}

	edge, err := socketedge.New(socketedge.Options{
		SampleRate: start.SampleRate,
		Caller:     stt.Participant{ID: "caller", Name: "Caller"},
		Send:       func(pcm []byte) error { return write(websocket.BinaryMessage, pcm) },
		Cleared:    func() { _ = writeFrame(frame{"type": "cleared"}) },
	})
	if err != nil {
		refuseWith(errSocketSampleRate)
		return
	}

	config, failure := s.configFor(ctx, customerID, start.Session.ConfigId, start.Session.Agent)
	if failure != nil {
		refuse(failure.Error())
		return
	}
	spec := specOf(start.Session, customerID, config)
	// A socket session carries audio, so voice is started on it whatever the frame says.
	spec.Text = false
	ctx = s.threadConversation(ctx, customerID, &spec)
	spec.Caller = CallerFrom(ctx)
	spec.CallerKind = KindFrom(ctx)
	spec.Edge = edge

	created, err := s.sessions.Create(ctx, spec)
	if answer, refused := moneyError(err); refused {
		refuseWith(answer)
		return
	}
	if errors.Is(err, session.ErrSessionExists) || err != nil {
		refuse(err.Error())
		return
	}
	s.rememberTimezone(ctx, customerID, spec)
	owner := OwnerFrom(ctx)
	defer func() {
		if _, err := s.sessions.Close(created.ID(), owner); err != nil {
			s.logger.Debug("could not close the socket session", "session", created.ID(), "error", err)
		}
	}()

	if err := writeFrame(frame{"type": "session", "session": sessionOf(ctx, created), "sample_rate": edge.SampleRate()}); err != nil {
		return
	}

	// The conversation ending is the other way the socket closes: the agent hangs up, and
	// reading stops when the connection does. The close waits for the read loop below to take
	// the caller's answer, and this handler waits for it before the connection is closed.
	// When the read loop returns first, on a stop or a broken socket, detaching closes events,
	// so the close is still sent; closeBeforeReader bounds the wait for it to closeWait.
	events, detach := created.Watch()
	readDone := make(chan struct{})
	hungUp := make(chan struct{})
	defer func() {
		close(readDone)
		detach()
		<-hungUp
	}()
	go func() {
		defer close(hungUp)
		for range events {
		}
		closeBeforeReader(connection, websocket.CloseNormalClosure, "the conversation is over", readDone)
		// A caller that never answers leaves the read loop reading; closing ends it.
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
