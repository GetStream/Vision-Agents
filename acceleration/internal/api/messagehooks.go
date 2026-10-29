package api

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
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
		// Source is the one field of what the writer put on the message that matters here:
		// chatlog.SourceField, which the agent writes on everything it stores. A hook carries
		// a message's custom fields at the top level of the message, beside its own, not
		// under "custom" as the v2 API returns them (Stream Chat docs, «Webhook Events»,
		// message.new; the classic GET /messages/{id} of an episode card shows the same).
		Source json.RawMessage `json:"source"`
		// Attachments are what was sent with the message; a photo on a thread channel is
		// answered (photos). Read by photos, one at a time, so an attachment of a shape
		// nothing here expects costs only itself, not the message's text.
		Attachments json.RawMessage `json:"attachments"`
	} `json:"message"`
}

// messageAttachment is the part of a Stream Chat attachment that says whether it is a photo
// and where it is. An image carries its URL in image_url, as this router writes it
// (conversation.chatFileAttachments), or in asset_url, as Stream's upload docs write it
// (https://getstream.io/chat/docs/javascript/file_uploads/).
type messageAttachment struct {
	Type     string `json:"type"`
	ImageURL string `json:"image_url"`
	AssetURL string `json:"asset_url"`
	// OGScrapeURL is set on the preview Stream adds for a link in the text, which may be an
	// image but is no photo anyone sent
	// (https://getstream.io/chat/docs/javascript/message_format/, «URL Attachment Fields»).
	OGScrapeURL string `json:"og_scrape_url"`
}

// photos are the images sent with a message, at most maxRunImages, as many as a run takes.
func (e messageEvent) photos() []llm.ImagePart {
	var attachments []json.RawMessage
	if json.Unmarshal(e.Message.Attachments, &attachments) != nil {
		return nil
	}
	var images []llm.ImagePart
	for _, raw := range attachments {
		var attachment messageAttachment
		if json.Unmarshal(raw, &attachment) != nil {
			continue
		}
		source := cmp.Or(attachment.ImageURL, attachment.AssetURL)
		if attachment.Type == "image" && source != "" && attachment.OGScrapeURL == "" && len(images) < maxRunImages {
			images = append(images, llm.ImagePart{URL: source})
		}
	}
	return images
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
		// Every message in the app arrives here; only one written to an agent is worth
		// recording as delivered.
		if addressed(event) && s.acting(r.Context(), origin, eventType, payload) {
			origin.handOver = s.replayed(r, payload)
			s.routeArrivingMessage(r.Context(), origin, event)
		}
	} else {
		s.logger.Debug("ignoring a message event", "type", eventType)
	}
	w.WriteHeader(http.StatusOK)
}

// chatHookAttribution owns the work a message starts outside a session a key opened: no key
// started it, and its sender is a user of the app Stream signed the delivery for.
func chatHookAttribution(customerID string, event messageEvent) routing.Attribution {
	attribution := routing.Attribution{APIKey: routing.OriginHookChat, UserID: event.Message.User.ID}
	if routing.RecordedUser(attribution.UserID) != "" {
		attribution.CallerKind = routing.CallerUser
	}
	return attribution
}

// addressed reports whether a message is one somebody wrote to an agent.
//
// Three things it is not. A message outside an agent's own channel: every message in the app
// is delivered here, and a team's channel has nothing on the other end of it. A message
// carrying a source: everything the agent stores does, whether it is speech it answered as
// it was said or its own reply, and answering either is the agent talking to itself. And a
// message with nothing written in it, which an attachment on its own is, except a photo on a
// thread channel: the person in an external thread sent it to the agent (answerThread).
func addressed(event messageEvent) bool {
	// This namespace belongs to durable session commands, even on older channels
	// without trigger metadata or when a client omits/forges the source marker.
	if conversation.SessionCommandChannel(event.ChannelType, event.ChannelID, event.ChannelCustom) {
		return false
	}
	if event.ChannelType != chatlog.ChannelType || event.ChannelID == "" {
		return false
	}
	if event.Message.Text == "" && (!strings.HasPrefix(event.ChannelID, conversation.ThreadChannelPrefix) || len(event.photos()) == 0) {
		return false
	}
	return len(event.Message.Source) == 0
}

// routeArrivingMessage answers a message from the session running on its channel, or hands
// it to a worker to start one.
//
// The session may be running on another node, which origin hands the message to. settled is
// false when nothing took the message and a later try may: the queue's worker tries again.
func (s *Server) routeArrivingMessage(ctx context.Context, origin hookOrigin, event messageEvent) (settled bool) {
	// A thread channel holds an external thread, such as a Slack thread, which the Router
	// answers itself: nobody watches it to hand the message to a worker.
	thread, linked, err := s.linkedThread(ctx, origin, event.ChannelID)
	if err != nil {
		s.logger.Error("not answering a message in a channel that may hold an external thread",
			"channel", event.ChannelID, "error", err)
		return false
	}
	if linked {
		origin.answer(func() { s.answerThread(origin, thread, event) })
		return true
	}
	// Only a linked thread answers a photo sent without text (addressed).
	if event.Message.Text == "" {
		return true
	}
	if s.sessions != nil {
		// A session running on a channel of the same name in another app is somebody
		// else's conversation.
		if found, running := s.sessions.ByAgentWhere(event.ChannelID, origin.owns); running {
			if found.Spec().PersistConversation {
				return true
			}
			if answer, settled := s.answerOnce(ctx, origin, event); !answer {
				return settled
			}
			if found.Spec().DispatchText {
				_, err := s.dispatchText(ctx, found, dispatch.Message{
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
					s.unanswered(ctx, origin)
				}
				return err == nil
			}
			// The answer goes back to the channel rather than in a response, so there is
			// nothing here to wait for.
			origin.answer(func() { s.answerMessage(found, event) })
			return true
		}
	}

	if elsewhere, taken := s.forwardedMessage(ctx, origin, event); elsewhere {
		return taken
	}

	if s.store == nil || s.dispatch == nil {
		return true
	}

	// Nothing is running, so this has to be given to a worker, and that needs to know
	// whose channel it is and which agent answers in it.
	customerID, configID, found, err := s.ownerOf(ctx, origin, event)
	if err != nil {
		return false
	}
	if !found {
		return true
	}
	if may, err := s.mayStillWrite(ctx, origin, customerID); !may {
		if err != nil && origin.queued {
			s.logger.Error("not starting work for a customer whose app could not be read",
				"channel", event.ChannelID, "customer", customerID, "error", err)
			return false
		}
		// A hook is not tried again, so it ends here as it always has.
		s.logger.Info("not starting work in the deployment's app for a customer no longer writing there",
			"channel", event.ChannelID, "customer", customerID)
		return true
	}

	if elsewhere, taken := s.forwardedToWorkers(ctx, origin.handOver, customerID); elsewhere {
		return taken
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

	if answer, settled := s.answerOnce(ctx, origin, event); !answer {
		return settled
	}
	worker, err := s.dispatch.AssignMessage(customerID, message)
	if err != nil {
		// Somebody has written to an agent that nothing is going to answer, which is the
		// most useful error this service can report.
		s.logger.Error("nobody could answer an arriving message",
			"channel", event.ChannelID, "customer", customerID, "error", err)
		s.unanswered(ctx, origin)
		return false
	}
	s.pinHook(origin, customerID, chatlog.ChannelType+":"+event.ChannelID)
	s.logger.Info("handed an arriving message to a worker",
		"channel", event.ChannelID, "customer", customerID,
		"config", configID, "worker", worker.ID, "stream_app", origin.app)
	return true
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
//
// err is a read that failed, which a later try may get through: the queue's worker tries again.
func (s *Server) ownerOf(ctx context.Context, origin hookOrigin, event messageEvent) (customerID, configID string, found bool, err error) {
	previous, err := s.store.CallByAgentInApp(ctx, origin.scope(), event.ChannelID)
	if err == nil && (origin.deployment || previous.CustomerID == origin.customer) {
		return previous.CustomerID, previous.ConfigID, true, nil
	}
	// A hook answers from what the channel declares when its last call could not be read, as it
	// always has. A queued delivery is tried again instead, so it is answered by the agent that
	// actually ran there.
	if origin.queued && err != nil && !errors.Is(err, store.ErrNoCall) && !errors.Is(err, store.ErrAmbiguousHook) {
		s.logger.Error("could not read the last call on a queued message's channel", "channel", event.ChannelID, "error", err)
		return "", "", false, err
	}

	declared, _ := event.ChannelCustom[ConfigField].(string)
	if declared == "" {
		s.logger.Debug("no agent has ever run on an arriving message's channel, and it names no config",
			"channel", event.ChannelID, "field", ConfigField)
		return "", "", false, nil
	}

	var config store.AgentConfig
	if owner, scoped := s.configOwnerOf(origin); scoped {
		config, err = s.store.AgentConfig(ctx, owner, declared)
	} else {
		config, err = s.store.AgentConfigOwner(ctx, declared)
	}
	// A config nobody in the app holds is no failure here: an app whose message hooks deliver
	// to two routers, such as a local one and staging, sends each the other's channels too.
	if errors.Is(err, store.ErrNoAgentConfig) {
		s.logger.Info("an arriving message's channel names a config nobody in its app holds",
			"channel", event.ChannelID, "config", declared, "stream_app", origin.app, "customer", origin.customer)
		return "", "", false, nil
	}
	if err != nil {
		s.logger.Error("could not read the config an arriving message's channel names",
			"channel", event.ChannelID, "config", declared, "stream_app", origin.app, "error", err)
		return "", "", false, err
	}
	return config.CustomerID, config.ID, true, nil
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
	workerID, err := s.assignMessage(ctx, spec.CustomerID, message)
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
		"session", found.ID(), "customer", spec.CustomerID, "worker", workerID)
	return receipt, nil
}

// assignMessage hands a message to one of the customer's workers, wherever in the
// deployment it is connected, and names the worker it went to.
func (s *Server) assignMessage(ctx context.Context, customerID string, message dispatch.Message) (string, error) {
	if s.workers != nil {
		return s.workers.AssignMessage(ctx, customerID, message)
	}
	worker, err := s.dispatch.AssignMessage(customerID, message)
	if err != nil {
		return "", err
	}
	return worker.ID, nil
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
