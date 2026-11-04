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
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
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
	// Channel is the channel as Stream sends it with the message, its members included
	// (members). Raw, and read only for a chat episode, so a message under a config without
	// episode cards costs nothing more, and a value of another shape costs only the card.
	Channel json.RawMessage `json:"channel"`
	Message struct {
		ID   string `json:"id"`
		Text string `json:"text"`
		// CreatedAt is when Stream made the message (created). Raw, as Channel is.
		CreatedAt json.RawMessage `json:"created_at"`
		User      struct {
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
		// Bridged is channelbridge.BridgedField, which the channel bridge sets on a message it
		// wrote and hands to the agent itself (bridged). Raw, so a value of another type,
		// which anyone writing in the app can set, costs only itself.
		Bridged json.RawMessage `json:"stream_agents_bridged"`
	} `json:"message"`
}

// bridged is whether the channel bridge wrote the message: its custom field
// channelbridge.BridgedField is JSON true. Any other value, or none, is a message somebody
// else wrote.
func (e messageEvent) bridged() bool {
	var bridged bool
	return json.Unmarshal(e.Message.Bridged, &bridged) == nil && bridged
}

// members are the user ids of the channel's members as Stream sends them with the message, in
// the channel of the event, as the app's webhook carries it: «For message and channel events
// the webhook request body will also include the list of channel members»
// (https://getstream.io/chat/docs/go-golang/webhook_events/). The docs say «truncated up to 50
// members»; the delivery loads up to 100 (DefaultMemberLimit, chat send_message.go:418,
// channel_cache.go:39). A list cut short is longer than two, so omnichannel.Between reads it as
// the channel of more than one person it is. A channel sent with no members, or none sent, has none, and so no person.
func (e messageEvent) members() []string {
	var channel struct {
		Members []struct {
			UserID string `json:"user_id"`
		} `json:"members"`
	}
	if json.Unmarshal(e.Channel, &channel) != nil {
		return nil
	}
	ids := make([]string, 0, len(channel.Members))
	for _, member := range channel.Members {
		ids = append(ids, member.UserID)
	}
	return ids
}

// created is when Stream made the message, as its created_at says, in RFC 3339 as Stream
// writes it (2020-03-30T07:54:46.255628Z); zero when it says no time.
func (e messageEvent) created() time.Time {
	var at time.Time
	if json.Unmarshal(e.Message.CreatedAt, &at) != nil {
		return time.Time{}
	}
	return at
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

// recording is where the voice recording sent with a message is: the first attachment a Stream
// Chat client plays as one, voiceRecording, or sends as a sound file, audio
// (stream-chat-js isVoiceRecordingAttachment, isAudioAttachment), at its asset_url. sent is
// false for a message with none.
func (e messageEvent) recording() (source string, sent bool) {
	var attachments []json.RawMessage
	if json.Unmarshal(e.Message.Attachments, &attachments) != nil {
		return "", false
	}
	for _, raw := range attachments {
		var attachment messageAttachment
		if json.Unmarshal(raw, &attachment) == nil && (attachment.Type == "voiceRecording" || attachment.Type == "audio") &&
			attachment.AssetURL != "" {
			return attachment.AssetURL, true
		}
	}
	return "", false
}

// dispatchedPhotos are the photos sent with a message as a worker is handed them: their URLs,
// which it fetches itself. None is nil, so a frame without photos carries no images.
func (e messageEvent) dispatchedPhotos() []dispatch.Image {
	var images []dispatch.Image
	for _, photo := range e.photos() {
		images = append(images, dispatch.Image{URL: photo.URL})
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
		if addressed(event) && s.awaitingOwnApp(r.Context(), origin, eventType) {
			// Stream retries, and by then the deployment will know which app is its own.
			writeStreamWaiting(w)
			return
		}
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
// it was said or its own reply, and answering either is the agent talking to itself, the
// voice recording it shows on a person's message included. And a message with no text, photo
// or voice recording, such as a file or a video on its own, which nothing that answers here
// reads.
func addressed(event messageEvent) bool {
	// This namespace belongs to durable session commands, even on older channels
	// without trigger metadata or when a client omits/forges the source marker.
	if conversation.SessionCommandChannel(event.ChannelType, event.ChannelID, event.ChannelCustom) {
		return false
	}
	if event.ChannelType != chatlog.ChannelType || event.ChannelID == "" {
		return false
	}
	if _, recorded := event.recording(); event.Message.Text == "" && len(event.photos()) == 0 && !recorded {
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
		// The worker that wrote it hands it to the agent (AnswerQueued): settled here.
		if event.bridged() {
			s.logger.Info("leaving a message the channel bridge wrote to its worker", "channel", thread.ChannelID, "message", event.Message.ID)
			return true
		}
		// A thread is answered from what was written and the photos, as before voice recordings
		// were heard in a plain channel: a recording alone has nothing in it to answer there.
		if event.Message.Text == "" && len(event.photos()) == 0 {
			return true
		}
		origin.answer(func() { s.answerThread(origin, thread, event) })
		return true
	}
	// Only a linked thread answers a photo sent without text on a thread channel (addressed).
	_, recorded := event.recording()
	if event.Message.Text == "" && strings.HasPrefix(event.ChannelID, conversation.ThreadChannelPrefix) {
		return true
	}
	if s.sessions != nil {
		found, running, err := s.liveSessionOn(ctx, origin, event)
		if errors.Is(err, store.ErrAmbiguousHook) {
			s.logger.Warn("not answering a message in a channel more than one customer's sessions run on",
				"channel", event.ChannelID, "stream_app", origin.app, "error", err)
			return true
		}
		if err != nil {
			return false
		}
		// A session that started with a person's episode cards answers that person alone: a
		// message by anybody else gets a session of its own, handed out below, which reads
		// none of the first person's cards. The first session takes no new message from then
		// on, and closes once it has answered what it was handed (retire).
		if running && found.CardsFor() != "" && (found.CardsFor() != event.Message.User.ID || found.Retiring()) {
			s.retire(found)
			running = false
		}
		if running {
			s.chatEpisode(ctx, origin, event, found.Spec())
			// The session a worker opened to answer this channel answers in it, as a thread's
			// does: the message is already there, so only the reply is written.
			if found.Spec().Dispatched {
				if answer, settled := s.answerOnce(ctx, origin, event); !answer {
					return settled
				}
				owed := s.owing.owe(found.ID())
				origin.answer(func() {
					defer owed()
					s.followUpMessage(origin, found, event)
				})
				return true
			}
			if found.Spec().PersistConversation {
				return true
			}
			if answer, settled := s.answerOnce(ctx, origin, event); !answer {
				return settled
			}
			if found.Spec().DispatchText {
				return s.handOn(ctx, origin, recorded, func(ctx context.Context) bool {
					_, err := s.dispatchText(ctx, found, dispatch.Message{
						ChannelType: event.ChannelType,
						ChannelID:   event.ChannelID,
						Custom:      customOf(event.ChannelCustom),
						Text:        s.heardText(ctx, found.Spec(), origin, event),
						Images:      event.dispatchedPhotos(),
						MessageID:   event.Message.ID,
						UserID:      event.Message.User.ID,
						UserName:    event.Message.User.Name,
					}, "")
					if err != nil {
						s.logger.Error("nobody could answer an arriving message",
							"channel", event.ChannelID, "session", found.ID(), "error", err)
						if errors.Is(err, node.ErrNoReply) {
							// The other node may have placed it before its reply was lost: queued
							// again, it would be placed twice.
							s.logger.Warn("a queued chat inbound message another node may have placed is left settled",
								"channel", event.ChannelID, "message", origin.queuedMessage, "stream_app", origin.app)
						} else {
							s.unanswered(ctx, origin)
						}
					}
					return err == nil
				})
			}
			// The answer goes back to the channel rather than in a response, so there is
			// nothing here to wait for.
			origin.answer(func() { s.answerMessage(origin, found, event) })
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
	customerID, configID, declared, found, err := s.ownerOf(ctx, origin, event)
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
		Images:      event.dispatchedPhotos(),
		MessageID:   event.Message.ID,
		UserID:      event.Message.User.ID,
		UserName:    event.Message.User.Name,
		At:          time.Now().UTC(),
	}
	var heardFor session.Spec
	if recorded {
		// Read before the delivery is settled, so a read that failed is tried again.
		if heardFor, err = s.unopenedSpec(ctx, customerID, configID, event); err != nil {
			s.logger.Error("could not read the config a voice recording would be heard for",
				"channel", event.ChannelID, "customer", customerID, "config", configID, "error", err)
			return false
		}
	}
	s.chatEpisodeUnopened(ctx, origin, event, declared, heardFor, recorded)

	if answer, settled := s.answerOnce(ctx, origin, event); !answer {
		return settled
	}
	return s.handOn(ctx, origin, recorded, func(ctx context.Context) bool {
		// Only a worker connected here is handed it (forwardedToWorkers): with none, nothing
		// would answer what the recording says, so it is not heard.
		if recorded && len(s.dispatch.Workers(customerID)) > 0 {
			message.Text = s.heardText(ctx, heardFor, origin, event)
		}
		// Said before the worker has the message, so the session it opens for it finds it said.
		s.handOut(ctx, origin, customerID, chatlog.ChannelType+":"+event.ChannelID, event.Message.User.ID)
		worker, err := s.dispatch.AssignMessage(customerID, message)
		if err != nil {
			// Somebody has written to an agent that nothing is going to answer, which is the
			// most useful error this service can report.
			s.logger.Error("nobody could answer an arriving message",
				"channel", event.ChannelID, "customer", customerID, "error", err)
			s.unanswered(ctx, origin)
			return false
		}
		s.logger.Info("handed an arriving message to a worker",
			"channel", event.ChannelID, "customer", customerID,
			"config", configID, "worker", worker.ID, "stream_app", origin.app)
		return true
	})
}

// handOn hands a message on with hand, which says whether it was taken. A voice recording on a
// hook is heard off the request, as hookOrigin.answer answers one, because hearing it takes up
// to inlineDeadline and Stream is waiting on the delivery: the hook is taken then. Anything
// else is handed on in line, as it always was: a queued delivery too, here or on the peer it
// was handed to, whose answer says whether something took it (answerChatInboundForPeer).
func (s *Server) handOn(ctx context.Context, origin hookOrigin, recorded bool, hand func(context.Context) bool) bool {
	if !recorded || origin.queuedMessage != "" {
		return hand(ctx)
	}
	origin.answer(func() {
		ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), askTimeout)
		defer cancel()
		hand(ctx)
	})
	return true
}

// unopenedSpec is what a voice recording sent where no session runs is heard for, as a session
// opened for it would hear it: the config answering the channel, else its customer alone, for
// the channel's agent, under the attribution of the person who sent it.
// A config nobody holds any more is the customer alone; err is a read that failed.
func (s *Server) unopenedSpec(ctx context.Context, customerID, configID string, event messageEvent) (session.Spec, error) {
	spec := session.Spec{CustomerID: customerID}
	if configID != "" {
		config, err := s.store.AgentConfig(ctx, customerID, configID)
		switch {
		case err == nil:
			spec = session.FromConfig(config)
		case !errors.Is(err, store.ErrNoAgentConfig):
			return session.Spec{}, err
		}
	}
	spec.AgentID = event.ChannelID
	spec.Attribution = chatHookAttribution(customerID, event)
	return spec, nil
}

// heardText is a message's text with what its voice recording says after it, heard for spec as a
// voice note sent with a question is (hearVoiceNote). A recording that could not be heard leaves
// the note that says so instead, as a photo that could not be seen does (withPhotoNote), so the
// reply says so rather than the person hearing nothing.
//
// A queued delivery's recording one try heard is not heard again: what it said is kept on its
// row, and a try after one nothing took, here or on a peer, hands that on (origin.heard). A deployment that hears
// nothing answers a message with text or photos from those alone, as before recordings were
// heard.
func (s *Server) heardText(ctx context.Context, spec session.Spec, origin hookOrigin, event messageEvent) string {
	source, sent := event.recording()
	if !sent {
		return event.Message.Text
	}
	if origin.heard != nil {
		return withVoiceNote(event.Message.Text, *origin.heard)
	}
	if (s.streams == nil || s.streams.Transcriptions == nil) && (event.Message.Text != "" || len(event.photos()) > 0) {
		return event.Message.Text
	}
	note, err := s.hearVoiceNote(ctx, spec, source, "")
	if err != nil {
		// The code and host alone: what fetching it failed with names its URL, which may carry
		// the signature that lets anyone fetch it.
		var refused APIError
		errors.As(err, &refused)
		s.logger.Warn("could not hear the voice recording of a message",
			"channel", event.ChannelID, "host", hostOf(source), "code", refused.Code)
		return withPhotoNote(event.Message.Text, event.photos(), unheardRecording, unheardRecordingWithMore)
	}
	if origin.queuedMessage != "" {
		if err := s.store.KeepChatInboundHeard(ctx, origin.app, origin.queuedMessage, note.Heard); err != nil {
			s.logger.Warn("could not keep what the voice recording of a queued chat inbound message said",
				"message", origin.queuedMessage, "stream_app", origin.app, "error", err)
		}
	}
	return withVoiceNote(event.Message.Text, note.Heard)
}

// liveSessionOn is the session running on a message's channel that answers it. err is
// store.ErrAmbiguousHook for a channel that is more than one customer's, which is answered by
// none of them, or a read that failed, which a later try may get through.
//
// A session running on a channel of the same name in another app is somebody else's
// conversation. In the deployment's app only the session of the channel's own customer answers
// a message there: the customer a channel the router created names (conversation.CustomerField),
// or for a channel naming nobody the customer ownerOf reads from the store: its last calls,
// else the customer holding the config it names. Unlike the customer ownerOf works out, this
// one only ever narrows: it admits only sessions the hook already admits. A channel the store
// names no customer for is answered by a session only while the sessions on it are one
// customer's, as a hook matching rows of more than one customer is acted on for nobody.
func (s *Server) liveSessionOn(ctx context.Context, origin hookOrigin, event messageEvent) (*session.Session, bool, error) {
	if !origin.deployment {
		found, running := s.sessions.ByAgentWhere(event.ChannelID, origin.owns)
		return found, running, nil
	}
	owner, stamped := event.ChannelCustom[conversation.CustomerField]
	if !stamped {
		// The store is read only when something runs on the channel, so a message nothing
		// runs for costs no read here.
		found, running := s.sessions.ByAgentWhere(event.ChannelID, origin.owns)
		if !running {
			return nil, false, nil
		}
		known := false
		if s.store != nil {
			previous, held, err := s.lastCallOn(ctx, origin, event.ChannelID)
			if err != nil {
				return nil, false, err
			}
			owner, known = previous.CustomerID, held
			if !held {
				config, named, err := s.declaredConfigOf(ctx, origin, event)
				if err != nil {
					return nil, false, err
				}
				owner, known = config.CustomerID, named
			}
		}
		if !known {
			theirs := found.Spec().CustomerID
			if _, another := s.sessions.ByAgentWhere(event.ChannelID, func(customer string, app int64) bool {
				return customer != theirs && origin.owns(customer, app)
			}); another {
				return nil, false, store.ErrAmbiguousHook
			}
			return found, true, nil
		}
	}
	found, running := s.sessions.ByAgentWhere(event.ChannelID, func(customer string, app int64) bool {
		return customer == owner && origin.owns(customer, app)
	})
	return found, running, nil
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
// declared is the config, when it was read: the one the channel names, and for a channel with
// a row, that config again when the row ran under it (heldConfigOf).
//
// err is a read that failed, which a later try may get through: the queue's worker tries again.
func (s *Server) ownerOf(ctx context.Context, origin hookOrigin, event messageEvent) (customerID, configID string,
	declared *store.AgentConfig, found bool, err error) {
	previous, held, err := s.lastCallOn(ctx, origin, event.ChannelID)
	if err != nil && !errors.Is(err, store.ErrAmbiguousHook) {
		return "", "", nil, false, err
	}
	if held && (origin.deployment || previous.CustomerID == origin.customer) {
		return previous.CustomerID, previous.ConfigID, s.heldConfigOf(ctx, origin, event, previous), true, nil
	}

	config, named, err := s.declaredConfigOf(ctx, origin, event)
	if err != nil || !named {
		return "", "", nil, false, err
	}
	return config.CustomerID, config.ID, &config, true, nil
}

// heldConfigOf is the config a channel with a row names, when it is the one the row ran under,
// so a chat the worker's session answered before goes on writing episode cards
// (chatEpisodeUnopened). It is the one read of the config such a message makes, as a channel
// with no row makes in ownerOf: base c9a61fc90 made none here, and whether the cards are on is
// not known without it. A channel naming no config, or another one than the row's, is not
// read, nor one with a voice recording, which unopenedSpec reads. A read that fails costs the
// card, not the answer: nil.
func (s *Server) heldConfigOf(ctx context.Context, origin hookOrigin, event messageEvent, previous store.Call) *store.AgentConfig {
	if _, recorded := event.recording(); recorded {
		return nil
	}
	if named, _ := event.ChannelCustom[ConfigField].(string); named != previous.ConfigID {
		return nil
	}
	config, named, err := s.declaredConfigOf(ctx, origin, event)
	if err != nil || !named {
		return nil
	}
	return &config
}

// declaredConfigOf is the config a channel names, when the store says a customer of the hook's
// app holds it.
func (s *Server) declaredConfigOf(ctx context.Context, origin hookOrigin, event messageEvent) (config store.AgentConfig, found bool, err error) {
	declared, _ := event.ChannelCustom[ConfigField].(string)
	if declared == "" {
		s.logger.Debug("an arriving message's channel has held no call, and it names no config",
			"channel", event.ChannelID, "field", ConfigField)
		return store.AgentConfig{}, false, nil
	}

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
		return store.AgentConfig{}, false, nil
	}
	if err != nil {
		s.logger.Error("could not read the config an arriving message's channel names",
			"channel", event.ChannelID, "config", declared, "stream_app", origin.app, "error", err)
		return store.AgentConfig{}, false, err
	}
	return config, true, nil
}

// lastCallOn is the last call a channel held in the hook's app, when it held one. err is
// store.ErrAmbiguousHook for calls of more than one customer, or a read of a queued delivery
// that failed. A hook goes on as for a channel that held none when the read fails, as it always
// has; a queued delivery is tried again instead, so it is answered by the agent that actually
// ran there.
func (s *Server) lastCallOn(ctx context.Context, origin hookOrigin, channelID string) (store.Call, bool, error) {
	previous, err := s.store.CallByAgentInApp(ctx, origin.scope(), channelID)
	switch {
	case err == nil:
		return previous, true, nil
	case errors.Is(err, store.ErrNoCall):
		return previous, false, nil
	case errors.Is(err, store.ErrAmbiguousHook):
		return previous, false, err
	case origin.queued:
		s.logger.Error("could not read the last call on a queued message's channel", "channel", channelID, "error", err)
		return previous, false, err
	}
	return previous, false, nil
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

// followUpMessage answers a message in the channel a running session answers for its worker
// (session.Spec.Dispatched), once the reply it may still be writing is done, as a thread's
// session answers one (answerClaimed): with what its voice recording says and its photos, read
// once for every try.
func (s *Server) followUpMessage(origin hookOrigin, found *session.Session, event messageEvent) {
	ctx, cancel := context.WithTimeout(context.Background(), askTimeout)
	defer cancel()
	followUpID := "message_" + store.NewID()
	text, images := s.messagePhotos(ctx, found.SeesImages(), s.heardText(ctx, found.Spec(), origin, event), event)
	for {
		err := found.FollowUp(ctx, followUpID, text, images)
		if errors.Is(err, conversation.ErrResponseRunning) && ctx.Err() == nil {
			time.Sleep(threadTurnPoll)
			continue
		}
		if err != nil {
			s.logger.Error("could not answer a message", "channel", event.ChannelID, "session", found.ID(), "error", err)
		}
		return
	}
}

// answerMessage answers from a session that is already running.
//
// The answer is written rather than spoken. A session on a call has somebody listening to
// it, and saying a reply to something they never said would interrupt them with an answer
// to somebody else's question.
func (s *Server) answerMessage(origin hookOrigin, found *session.Session, event messageEvent) {
	ctx, cancel := context.WithTimeout(context.Background(), askTimeout)
	defer cancel()

	// Ask refuses a speech-to-speech agent, which answers only on the call (Mode), so a
	// recording sent to one is not heard: hearing it is billed.
	heard := event.Message.Text
	if found.Mode() != "native" {
		heard = s.heardText(ctx, found.Spec(), origin, event)
	}
	text, images := s.messagePhotos(ctx, found.SeesImagesInWriting(), heard, event)
	if _, err := found.Ask(ctx, text, images...); err != nil {
		s.logger.Error("could not answer a message",
			"channel", event.ChannelID, "session", found.ID(), "error", err)
		return
	}
	s.logger.Debug("answered a message in writing",
		"channel", event.ChannelID, "session", found.ID())
}
