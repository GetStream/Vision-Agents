package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
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
	if !s.hooksConfigured() {
		// Refusing is the only safe answer: without the secret there is no way to tell
		// Stream from anyone who found the URL, and this path starts agents.
		writeError(w, notFound("message events are not configured"))
		return
	}

	payload, ok := readHook(w, r, "message event")
	if !ok {
		s.logger.Warn("rejected a message event before reading its signature")
		return
	}
	origin, ok := s.verifyHook(w, r, payload, "message event")
	if !ok {
		return
	}

	eventType := getstream.GetEventType(payload)
	if eventType == "" {
		writeError(w, invalidRequest("could not read that message event"))
		return
	}

	if eventType == getstream.EventTypeMessageNew {
		var event messageEvent
		if err := json.Unmarshal(payload, &event); err != nil {
			writeError(w, invalidRequest("could not read that message event"))
			return
		}
		// Every message in the app arrives here; only one written to an agent, or an agent's
		// reply bound for an external thread, is worth recording as delivered.
		switch {
		case addressed(event):
			if s.acting(r.Context(), origin, eventType, payload) {
				s.routeArrivingMessage(r, origin, payload, event)
			}
		case replied(event):
			s.handOffReply(r.Context(), origin, eventType, payload, event)
		}
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
	// This namespace belongs to durable session commands, even on older channels
	// without trigger metadata or when a client omits/forges the source marker.
	if conversation.SessionCommandChannel(event.ChannelType, event.ChannelID) {
		return false
	}
	if event.ChannelType != chatlog.ChannelType || event.ChannelID == "" {
		return false
	}
	if event.Message.Text == "" {
		return false
	}
	_, written := event.Message.Custom[chatlog.SourceField]
	return !written
}

// replied reports whether a message is an agent's finished written reply in an agent
// channel: the one message chatlog.Log.Reply writes for an answer, with its text, source agent
// and generating false (internal/chatlog/chatlog.go, writer.send). A reply still being
// written (generating true) is left alone, so a half-written one never leaves.
func replied(event messageEvent) bool {
	if event.ChannelType != chatlog.ChannelType || event.ChannelID == "" || event.Message.Text == "" {
		return false
	}
	if generating, _ := event.Message.Custom[chatlog.GeneratingField].(bool); generating {
		return false
	}
	return event.Message.Custom[chatlog.SourceField] == chatlog.SourceAgent
}

// handOffReply gives the channel bridge an agent's reply in a thread channel linked to an
// external thread, such as a Slack thread, for the bridge to send there (channels.md on
// connectors/planning, «Who moves messages: the channel bridge», step 7). A channel linked to
// nothing is a plain Stream Chat conversation, and nothing leaves it. A thread is acted on
// only from a hook of the app it is pinned to, and only once per delivery.
func (s *Server) handOffReply(ctx context.Context, origin hookOrigin, eventType string, payload []byte, event messageEvent) {
	if s.store == nil {
		return
	}
	thread, err := s.store.ChannelThread(ctx, event.ChannelID)
	if errors.Is(err, store.ErrNoChannelThread) {
		return
	}
	if err != nil {
		s.logger.Error("could not tell whether a reply goes to an external thread", "channel", event.ChannelID, "error", err)
		return
	}
	if !origin.owns(thread.CustomerID, thread.StreamAppPK) {
		s.logger.Info("ignoring a reply in a thread channel of another app", "channel", event.ChannelID, "stream_app", origin.app)
		return
	}
	if !s.acting(ctx, origin, eventType, payload) {
		return
	}
	s.channelBridge.Reply(ctx, thread, event.Message.Text)
}

// routeArrivingMessage answers a message from the session running on its channel, or hands
// it to a worker to start one.
//
// The body is carried in rather than read again because the session may be running on
// another node, which has to be handed the delivery exactly as Stream signed it.
func (s *Server) routeArrivingMessage(r *http.Request, origin hookOrigin, body []byte, event messageEvent) {
	if s.sessions != nil {
		// A session running on a channel of the same name in another app is somebody
		// else's conversation.
		if found, running := s.sessions.ByAgentWhere(event.ChannelID, origin.owns); running {
			if found.Spec().PersistConversation {
				return
			}
			if found.Spec().DispatchText {
				_, err := s.dispatchText(r.Context(), found, dispatch.Message{
					ChannelType: event.ChannelType,
					ChannelID:   event.ChannelID,
					Custom:      customOf(event.ChannelCustom),
					Text:        event.Message.Text,
					MessageID:   event.Message.ID,
					UserID:      event.Message.User.ID,
					UserName:    event.Message.User.Name,
				}, "")
				if err != nil {
					s.logger.Error("nobody could answer an arriving message",
						"channel", event.ChannelID, "session", found.ID(), "error", err)
				}
				return
			}
			// On its own goroutine because a model call takes seconds and Stream is
			// waiting on this delivery. The answer goes back to the channel rather than
			// in a response, so there is nothing here to wait for.
			go s.answerMessage(found, event)
			return
		}
	}

	if s.forwardedMessage(r, body, event) {
		return
	}

	if s.store == nil || s.dispatch == nil {
		return
	}

	// Nothing is running, so this has to be given to a worker, and that needs to know
	// whose channel it is and which agent answers in it.
	customerID, configID, found := s.ownerOf(r.Context(), origin, event)
	if !found {
		return
	}
	if !s.mayWrite(r.Context(), origin, customerID) {
		s.logger.Info("not starting work in the deployment's app for a customer no longer writing there",
			"channel", event.ChannelID, "customer", customerID)
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
	s.pinHook(origin, customerID, chatlog.ChannelType+":"+event.ChannelID)
	s.logger.Info("handed an arriving message to a worker",
		"channel", event.ChannelID, "customer", customerID,
		"config", configID, "worker", worker.ID, "stream_app", origin.app)
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
//
// Both are looked for only within the app the hook came from: the channel's last call has to
// have been in that app, and a config the channel names has to be the app's own customer's.
func (s *Server) ownerOf(ctx context.Context, origin hookOrigin, event messageEvent) (customerID, configID string, found bool) {
	if previous, err := s.store.CallByAgentInApp(ctx, origin.scope(), event.ChannelID); err == nil &&
		(origin.deployment || previous.CustomerID == origin.customer) {
		return previous.CustomerID, previous.ConfigID, true
	}

	declared, _ := event.ChannelCustom[ConfigField].(string)
	if declared == "" {
		s.logger.Debug("no agent has ever run on an arriving message's channel, and it names no config",
			"channel", event.ChannelID, "field", ConfigField)
		return "", "", false
	}

	var config store.AgentConfig
	var err error
	if owner, scoped := s.configOwnerOf(origin); scoped {
		config, err = s.store.AgentConfig(ctx, owner, declared)
	} else {
		config, err = s.store.AgentConfigOwner(ctx, declared)
	}
	if err != nil {
		s.logger.Error("an arriving message's channel names a config nobody in its app holds",
			"channel", event.ChannelID, "config", declared, "stream_app", origin.app, "error", err)
		return "", "", false
	}
	return config.CustomerID, config.ID, true
}

// configOwnerOf is the only customer whose configs a channel in the hook's app may name: the
// registered app's own customer, or in app mode the deployment's own customer for a hook
// from the deployment's app, so a channel there never starts a fallback tenant's agent. In
// deployment mode every customer shares the app, and a channel may name any config.
func (s *Server) configOwnerOf(origin hookOrigin) (string, bool) {
	switch {
	case !origin.deployment:
		return origin.customer, true
	case s.stream != nil && s.stream.PerApp():
		return streamapp.CustomerOf(origin.app), true
	}
	return "", false
}

// leftToDispatch reports whether text this kind of caller sent a session goes to the
// customer's dispatch worker rather than the model. Only an end user's does: the server is
// how the worker answers, so what it sends always reaches the model.
func leftToDispatch(found *session.Session, kind auth.Kind) bool {
	return found.Spec().DispatchText && kind != auth.KindServer
}

// dispatchText hands text an end user wrote to a session to one of the customer's dispatch
// workers, instead of the model. A durable command is accepted first, so the message is
// recorded and shown as being answered, and withdrawn again if no worker can take it.
func (s *Server) dispatchText(ctx context.Context, found *session.Session, message dispatch.Message, clientID string) (conversation.CommandReceipt, error) {
	if s.dispatch == nil {
		return conversation.CommandReceipt{}, errors.New("this agent leaves text to a dispatch worker, and this deployment has none")
	}
	var receipt conversation.CommandReceipt
	if message.CommandID != "" {
		accepted, err := found.AwaitCommand(ctx, message.CommandID, message.Text, clientID)
		if err != nil || accepted.Duplicate {
			return accepted, err
		}
		receipt = accepted
	}

	spec := found.Spec()
	message.AgentID = spec.AgentID
	message.ConfigID = spec.ConfigID
	message.SessionID = found.ID()
	message.At = time.Now().UTC()
	worker, err := s.dispatch.AssignMessage(spec.CustomerID, message)
	if err != nil {
		if message.CommandID != "" {
			if _, stopped := found.InterruptCommand(message.CommandID); stopped != nil {
				s.logger.Error("could not withdraw a command no worker took",
					"session", found.ID(), "command", message.CommandID, "error", stopped)
			}
		}
		return conversation.CommandReceipt{}, err
	}
	s.logger.Info("handed a message to a worker",
		"session", found.ID(), "customer", spec.CustomerID, "worker", worker.ID)
	return receipt, nil
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
