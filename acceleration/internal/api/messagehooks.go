package api

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
)

// askTimeout bounds answering one written message. It is generous compared with a spoken
// turn because nobody is sitting in silence waiting for it, and short enough that a stuck
// provider does not hold a goroutine for the life of the process.
const askTimeout = 2 * time.Minute

// messageEvent is the part of a message event this hook acts on.
//
// Decoded into a struct of its own rather than the SDK's, for the reason
// receiveCallEvent gives: the SDK's timestamps read one format and write another, so a
// delivery would be refused as unsigned over a field nothing here reads.
type messageEvent struct {
	ChannelType string `json:"channel_type"`
	ChannelID   string `json:"channel_id"`
	// ChannelCustom is whatever the channel was created with. One field of it is read
	// here, for a channel no agent has ever run on, where it is the only thing that says
	// who should answer; see ownerOf. The rest is carried to the worker unread.
	ChannelCustom map[string]any `json:"channel_custom"`
	Message       struct {
		ID   string `json:"id"`
		Text string `json:"text"`
		User struct {
			ID   string `json:"id"`
			Name string `json:"name"`
		} `json:"user"`
		// Custom is whatever the writer put on the message. What matters here is the one
		// field the agent writes on everything it stores.
		Custom map[string]any `json:"custom"`
	} `json:"message"`
}

// receiveMessageEvent takes the message events Stream sends and turns a message written to
// an agent into an answer.
//
// This is what makes an agent reachable in writing. A channel outlives the call that filled
// it, so the same channel reaches the session still running on it and, once that has ended,
// starts a new one.
//
// Like the call hook it carries no customer header, because Stream is not a customer, and is
// authenticated by signature instead. Every outcome short of something that did not come
// from Stream is a 200: Stream retries a non-2xx, and a message nobody can answer is not
// answerable on the second delivery either.
func (s *Server) receiveMessageEvent(w http.ResponseWriter, r *http.Request) {
	if s.streamSecret == "" {
		// Refusing is the only safe answer: without the secret there is no way to tell
		// Stream from anyone who found the URL, and this path starts agents.
		http.Error(w, "message events are not configured", http.StatusNotFound)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "could not read that message event", http.StatusBadRequest)
		return
	}
	// Deliveries may be compressed, and the signature is over what is inside.
	payload, err := getstream.GunzipPayload(body)
	if err != nil {
		s.logger.Warn("rejected a message event", "error", err)
		http.Error(w, "that is not a message event from Stream", http.StatusUnauthorized)
		return
	}
	if !getstream.VerifySignature(payload, r.Header.Get(signatureHeader), s.streamSecret) {
		s.logger.Warn("rejected a message event with a bad signature", "bytes", len(payload))
		http.Error(w, "that is not a message event from Stream", http.StatusUnauthorized)
		return
	}

	eventType := getstream.GetEventType(payload)
	if eventType == "" {
		http.Error(w, "could not read that message event", http.StatusBadRequest)
		return
	}

	if eventType == getstream.EventTypeMessageNew {
		var event messageEvent
		if err := json.Unmarshal(payload, &event); err != nil {
			http.Error(w, "could not read that message event", http.StatusBadRequest)
			return
		}
		s.routeArrivingMessage(r, event)
	} else {
		s.logger.Debug("ignoring a message event", "type", eventType)
	}
	w.WriteHeader(http.StatusOK)
}

// addressed reports whether a message is one somebody wrote to an agent.
//
// Three things it is not. A message outside an agent's own channel: every message in the app
// is delivered here, and a team's channel has nothing on the other end of it. A message
// carrying a source: everything the agent stores does, whether it is speech it answered as
// it was said or its own reply, and answering either is the agent talking to itself. And a
// message with nothing written in it, which an attachment on its own is.
func addressed(event messageEvent) bool {
	if event.ChannelType != chatlog.ChannelType || event.ChannelID == "" {
		return false
	}
	if event.Message.Text == "" {
		return false
	}
	_, written := event.Message.Custom[chatlog.SourceField]
	return !written
}

// routeArrivingMessage answers a message from the session running on its channel, or hands
// it to a worker to start one.
func (s *Server) routeArrivingMessage(r *http.Request, event messageEvent) {
	if !addressed(event) {
		return
	}

	if s.sessions != nil {
		if found, running := s.sessions.ByAgent(event.ChannelID); running {
			// On its own goroutine because a model call takes seconds and Stream is
			// waiting on this delivery. The answer goes back to the channel rather than
			// in a response, so there is nothing here to wait for.
			go s.answerMessage(found, event)
			return
		}
	}

	if s.store == nil || s.dispatch == nil {
		return
	}

	// Nothing is running, so this has to be given to a worker, and that needs to know
	// whose channel it is and which agent answers in it.
	customerID, configID, found := s.ownerOf(r.Context(), event)
	if !found {
		return
	}

	message := dispatch.Message{
		ChannelType: event.ChannelType,
		ChannelID:   event.ChannelID,
		AgentID:     event.ChannelID,
		ConfigID:    configID,
		Custom:      customOf(event.ChannelCustom),
		Text:        event.Message.Text,
		MessageID:   event.Message.ID,
		UserID:      event.Message.User.ID,
		UserName:    event.Message.User.Name,
		At:          time.Now().UTC(),
	}

	worker, err := s.dispatch.AssignMessage(customerID, message)
	if err != nil {
		// Somebody has written to an agent that nothing is going to answer, which is the
		// most useful error this service can report.
		s.logger.Error("nobody could answer an arriving message",
			"channel", event.ChannelID, "customer", customerID, "error", err)
		return
	}
	s.logger.Info("handed an arriving message to a worker",
		"channel", event.ChannelID, "customer", customerID,
		"config", configID, "worker", worker.ID)
}

// ConfigField is the custom field on an agent channel naming the agent config that answers
// in it.
//
// It is how a conversation that starts in writing is reachable at all. A channel a call
// left behind is claimed by the row that call wrote; a channel created for somebody opening
// a support chat has no such row, and without this there is nothing to say which agent they
// have written to.
//
// The customer is not read from the channel. Whoever creates a channel decides what is on
// it, so a customer id there would be a claim rather than a fact, and acting on it would let
// one app's channel be answered by another app's workers. A config id is not a claim: the
// store says who owns that config, and a channel naming one nobody owns is answered by
// nobody.
const ConfigField = "agent_config_id"

// ownerOf works out whose channel an arriving message was written in, and which agent config
// answers there.
//
// The row the last conversation left is asked first, because a channel that has held one is
// the ordinary case and its row is what actually ran. A channel with no row falls back to
// what the channel itself declares.
func (s *Server) ownerOf(ctx context.Context, event messageEvent) (customerID, configID string, found bool) {
	if previous, err := s.store.CallByAgent(ctx, event.ChannelID); err == nil {
		return previous.CustomerID, previous.ConfigID, true
	}

	declared, _ := event.ChannelCustom[ConfigField].(string)
	if declared == "" {
		s.logger.Debug("no agent has ever run on an arriving message's channel, and it names no config",
			"channel", event.ChannelID, "field", ConfigField)
		return "", "", false
	}

	config, err := s.store.AgentConfigOwner(ctx, declared)
	if err != nil {
		s.logger.Error("an arriving message's channel names a config nobody holds",
			"channel", event.ChannelID, "config", declared, "error", err)
		return "", "", false
	}
	return config.CustomerID, config.ID, true
}

// answerMessage answers from a session that is already running.
//
// The answer is written rather than spoken. A session on a call has somebody listening to
// it, and saying a reply to something they never said would interrupt them with an answer
// to somebody else's question.
func (s *Server) answerMessage(found *session.Session, event messageEvent) {
	ctx, cancel := context.WithTimeout(context.Background(), askTimeout)
	defer cancel()

	if _, err := found.Ask(ctx, event.Message.Text); err != nil {
		s.logger.Error("could not answer a message",
			"channel", event.ChannelID, "session", found.ID(), "error", err)
		return
	}
	s.logger.Debug("answered a message in writing",
		"channel", event.ChannelID, "session", found.ID())
}
