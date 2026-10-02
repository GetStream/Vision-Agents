// Package chatlog stores what was said in a conversation as Stream Chat messages.
//
// A voice call leaves nothing behind once it ends. Writing each settled transcript and
// each reply into a chat channel gives the conversation a history that outlives the call
// and that any Stream Chat client can already read, without this service having to serve
// a transcript API of its own.
//
// Writing is asynchronous and drops rather than blocks: a conversation must never wait on
// the network to store what was just said.
//
// A reply is shown while it is still being written, and a participant's words while they
// are still being transcribed. The revisions go out as ephemeral updates, which reach
// anyone watching the channel without storing a version per token.
// Responded means the model finished, not that the caller heard it; the durable
// transcript is stored when speech finishes, and an interruption is not stored as a
// fully spoken reply.
package chatlog

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ChannelType is the Stream Chat channel type transcripts are written to.
const ChannelType = streamapp.AgentChannelType

// queueSize bounds how far the writer may fall behind before messages are dropped.
const queueSize = 256

// writeTimeout bounds a single write so a stuck network cannot wedge the writer.
const writeTimeout = 10 * time.Second

// streamInterval is how often a reply still being written is shown. Deltas arrive a token
// at a time, and showing each one would be a request per token.
const streamInterval = 200 * time.Millisecond

// generatingField is the custom field a client reads to know a reply is not finished. The
// Python agent writes the same field, so a client can watch either.
const generatingField = "generating"

// SourceField says where a message came from, so a message the agent has already dealt with
// is not mistaken for one addressed to it. Everything this package writes carries it, and a
// message without it is one a person typed.
const SourceField = "source"

// What SourceField holds.
const (
	// SourceSpeech is a participant's turn, transcribed. The agent answered it as it was
	// said, so it is here as a record rather than as a question.
	SourceSpeech = "speech"
	// SourceAgent is anything the agent wrote.
	SourceAgent = "agent"
)

// interruptedField marks a reply the caller cut off. Clients must not present it as a
// finished spoken utterance.
const interruptedField = "interrupted"

// kind says how a queued message relates to the reply it belongs to.
type kind int

const (
	// whole is a line that was said in full: a participant's turn, or a reply that never
	// streamed.
	whole kind = iota
	// piece is more of a reply that is still being written.
	piece
	// prepared is the model's finished text, which may still be being spoken.
	prepared
	// spoken closes a streamed reply after the caller heard it.
	spoken
	// interrupt closes a reply that was abandoned before it was heard in full.
	interrupt
	// hearing is a participant's words so far, revised as they keep talking.
	hearing
	// heard is what a participant settled on saying.
	heard
	// ignored is speech the agent decided was not meant for it.
	ignored
	// artifact is a card for artifacts a tool stored, apart from anything said.
	artifact
)

// Options configures a Log. Its credentials are given, never read from the environment.
type Options struct {
	// AgentID is required. It is the default channel name for demo calls that have no
	// conversation.
	AgentID string
	// Channel is the Stream Chat channel id (without type) to write into. Empty means
	// AgentID. A bound Athena conversation passes the id from its CID so voice does not
	// open a second channel.
	Channel string
	// Agent is the user the agent's own replies are written as.
	Agent User
	// VisibleTools are the agent config's visible_tools. A shown tool's stored artifacts
	// get a card of their own; empty shows search and web_search.
	VisibleTools []string
	// CustomerID is stamped on a channel this log creates, so reading it back can tell
	// whose it is. A bound conversation's channel already carries the stamp.
	CustomerID string

	// Client writes with a client the caller already holds for the app, which is how the
	// router writes in the Stream app a session is pinned to. With it set, nothing is read
	// from the environment and the key and secret below are not used.
	Client *getstream.Stream
	// APIKey and APISecret build a client when none is given. These are server-side
	// writes, so a secret is required rather than a user token.
	APIKey    string
	APISecret string

	Logger *slog.Logger
}

// User is who a message is written as.
type User struct {
	ID   string
	Name string
}

// Log writes a conversation into one Stream Chat channel.
type Log struct {
	client   *getstream.Stream
	agentID  string
	channel  string
	existing bool
	customer string
	agent    User
	visible  []string
	logger   *slog.Logger

	queue chan message
	done  chan struct{}

	// started reports whether the writer is running, so Close knows whether there is
	// anything to wait for.
	started   atomic.Bool
	closeOnce sync.Once
	dropped   atomic.Int64
}

// message is one line of the conversation waiting to be written.
type message struct {
	author User
	text   string
	// turnID names the reply a piece belongs to. Empty for anything said in full.
	turnID string
	kind   kind
	// source is what the message is written as, one of the SourceField values.
	source string
	// receiptID is the tool call an artifact card is for, which keeps its identity stable.
	receiptID string
	artifacts []conversation.ArtifactAttachment
}

// New validates the options and returns a Log. It writes nothing; Start does that.
func New(options Options) (*Log, error) {
	if options.AgentID == "" {
		return nil, errors.New("chatlog: an agent id is required")
	}
	if options.Agent.ID == "" {
		return nil, errors.New("chatlog: an agent user id is required")
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	client := options.Client
	if client == nil {
		if options.APIKey == "" || options.APISecret == "" {
			return nil, errors.New("chatlog: a client, or an api key and secret, are required")
		}
		var err error
		if client, err = getstream.NewClient(options.APIKey, options.APISecret); err != nil {
			return nil, err
		}
	}

	channel := options.Channel
	if channel == "" {
		channel = options.AgentID
	}

	return &Log{
		client:   client,
		agentID:  options.AgentID,
		channel:  channel,
		existing: options.Channel != "",
		customer: options.CustomerID,
		agent:    options.Agent,
		visible:  options.VisibleTools,
		logger:   options.Logger.With("agent", options.AgentID, "channel", channel),
		queue:    make(chan message, queueSize),
		done:     make(chan struct{}),
	}, nil
}

// Start creates the channel and begins writing. Doing it up front means the first thing
// anyone says is not also paying for the channel being created.
func (l *Log) Start(ctx context.Context) error {
	if l.started.Load() {
		return nil
	}
	if err := l.upsert(ctx, l.agent); err != nil {
		return err
	}
	request := &getstream.GetOrCreateChannelRequest{}
	if l.existing {
		state := true
		request.State = &state
	} else {
		request.Data = &getstream.ChannelInput{CreatedByID: &l.agent.ID}
		if l.customer != "" {
			request.Data.Custom = map[string]any{conversation.CustomerField: l.customer}
		}
	}
	_, err := l.client.Chat().GetOrCreateChannel(ctx, ChannelType, l.channel, request)
	if err != nil {
		return err
	}

	l.started.Store(true)
	go l.run()
	return nil
}

// Record stores the speech in an agent event. Events that are not something somebody
// said are ignored, so a caller can hand it every event without filtering.
func (l *Log) Record(event agent.Event) {
	switch typed := event.(type) {
	case agent.ToolRan:
		if typed.Err != nil || typed.ID == "" || typed.TurnID == "" || !conversation.ToolVisible(l.visible, typed.Tool) {
			return
		}
		if artifacts := conversation.StoredArtifacts(typed.Result); len(artifacts) > 0 {
			l.enqueue(message{author: l.agent, turnID: typed.TurnID, kind: artifact, source: SourceAgent, receiptID: typed.ID, artifacts: artifacts})
		}
	case agent.Hearing:
		if typed.Text == "" {
			return
		}
		l.enqueue(message{author: participantUser(typed.Participant), text: typed.Text, kind: hearing, source: SourceSpeech})
	case agent.Heard:
		if typed.Text == "" {
			return
		}
		l.enqueue(message{author: participantUser(typed.Participant), text: typed.Text, kind: heard, source: SourceSpeech})
	case agent.Decided:
		if typed.Kind == string(agent.ActIgnore) {
			l.enqueue(message{author: participantUser(typed.Participant), kind: ignored, source: SourceSpeech})
		}
	case agent.ResponseDelta:
		l.enqueue(message{author: l.agent, text: typed.Text, turnID: typed.TurnID, kind: piece, source: SourceAgent})
	case agent.Responded:
		if typed.Text == "" {
			return
		}
		if typed.TurnID == "" {
			l.enqueue(message{author: l.agent, text: typed.Text, kind: whole, source: SourceAgent})
			return
		}
		l.enqueue(message{author: l.agent, text: typed.Text, turnID: typed.TurnID, kind: prepared, source: SourceAgent})
	case agent.Spoke:
		if typed.TurnID == "" {
			return
		}
		l.enqueue(message{author: l.agent, turnID: typed.TurnID, kind: spoken, source: SourceAgent})
	case agent.Interrupted:
		// The model may already have finished, but the caller did not hear that reply in
		// full, so it must not be stored as a completed spoken line.
		l.enqueue(message{author: l.agent, turnID: typed.TurnID, kind: interrupt, source: SourceAgent})
	}
}

// Say queues one line of the conversation, dropping it if the writer is too far behind.
func (l *Log) Say(author User, text string) {
	if author.ID == "" || text == "" {
		return
	}
	l.enqueue(message{author: author, text: text, kind: whole, source: SourceSpeech})
}

// Reply queues something the agent wrote rather than said.
//
// It is separate from Record because a written answer is not an agent event: nothing was
// spoken, so no turn was started and no reply streamed. It is marked as the agent's all the
// same, so answering a message in the channel does not read as a new message to answer.
func (l *Log) Reply(text string) {
	if text == "" {
		return
	}
	l.enqueue(message{author: l.agent, text: text, kind: whole, source: SourceAgent})
}

// enqueue hands one message to the writer, dropping it if the writer is too far behind. A
// dropped piece costs a moment of a reply looking behind; the finished reply carries the
// whole of it, so nothing is lost from the transcript itself.
func (l *Log) enqueue(queued message) {
	select {
	case l.queue <- queued:
	default:
		l.dropped.Add(1)
	}
}

// Chat exposes the underlying client so a caller can reach features this does not wrap.
func (l *Log) Chat() *getstream.ChatClient { return l.client.Chat() }

// ChannelID is where this conversation is stored.
func (l *Log) ChannelID() string { return l.channel }

// Close drains the queue and stops the writer.
func (l *Log) Close() {
	l.closeOnce.Do(func() {
		close(l.queue)
		// Without a writer there is nothing draining the queue and nothing to wait for.
		if l.started.Load() {
			<-l.done
		}
		if dropped := l.dropped.Load(); dropped > 0 {
			l.logger.Warn("dropped transcript messages because the writer fell behind", "count", dropped)
		}
	})
}

func (l *Log) run() {
	defer close(l.done)

	writer := newWriter(l)

	ticker := time.NewTicker(streamInterval)
	defer ticker.Stop()

	for {
		select {
		case queued, open := <-l.queue:
			if !open {
				writer.closeOut()
				return
			}
			writer.handle(queued)
		case <-ticker.C:
			writer.show()
		}
	}
}

// reply is a reply being written into the channel a piece at a time.
type reply struct {
	author User
	// messageID is the stored message watchers see updated, once there is one.
	messageID string
	text      string
	// generated is the model's finished text. It is not stored as spoken until the voice
	// has actually finished, because Responded arrives while TTS may still be playing.
	generated string
	// shown is what watchers were last sent, so an unchanged reply is not sent again.
	shown string
}

// writer is the state behind the queue. The writer goroutine is the only one that touches
// it, so it needs no lock of its own.
type writer struct {
	log         *Log
	known       map[string]struct{}
	writing     map[string]*reply
	interrupted map[string]struct{}
	// listening is what each participant is saying, by user id, until it settles.
	listening map[string]*reply
}

func newWriter(l *Log) *writer {
	return &writer{
		log: l,
		// Server-side sends name their author, so a user the app has never seen has to
		// exist before their first message.
		known:       map[string]struct{}{l.agent.ID: {}},
		writing:     map[string]*reply{},
		interrupted: map[string]struct{}{},
		listening:   map[string]*reply{},
	}
}

// handle takes one queued message.
func (w *writer) handle(queued message) {
	switch queued.kind {
	case artifact:
		w.storeArtifacts(queued)
	case piece:
		writing, started := w.writing[queued.turnID]
		if !started {
			writing = &reply{author: queued.author}
			w.writing[queued.turnID] = writing
		}
		writing.text += queued.text
	case prepared:
		writing := w.ensure(queued)
		writing.generated = queued.text
		if _, cut := w.interrupted[queued.turnID]; cut {
			w.abandon(queued.turnID, queued.text)
		}
	case spoken:
		if _, cut := w.interrupted[queued.turnID]; cut {
			return
		}
		writing, started := w.writing[queued.turnID]
		text := queued.text
		if started {
			if writing.generated != "" {
				text = writing.generated
			} else if writing.text != "" {
				text = writing.text
			}
		}
		w.settle(queued.turnID, text)
	case interrupt:
		w.interrupted[queued.turnID] = struct{}{}
		w.abandon(queued.turnID, "")
	case hearing:
		listening, started := w.listening[queued.author.ID]
		if !started {
			listening = &reply{author: queued.author}
			w.listening[queued.author.ID] = listening
		}
		listening.text = queued.text
	case heard:
		listening, started := w.listening[queued.author.ID]
		delete(w.listening, queued.author.ID)
		if !started || listening.messageID == "" {
			w.store(queued.author, queued.text, SourceSpeech, false)
			return
		}
		w.patch(listening, queued.text, false, SourceSpeech)
	case ignored:
		w.retract(queued.author.ID)
	case whole:
		w.store(queued.author, queued.text, queued.source, false)
	}
}

// storeArtifacts stores a card for saved artifacts, which are there even if the reply that
// follows is interrupted. It is apart from the spoken transcript, and its id comes from the
// tool call, so a repeated delivery is the same card rather than a second one.
func (w *writer) storeArtifacts(queued message) {
	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()
	id := fmt.Sprintf("voice-artifact-%x", sha256.Sum256([]byte(w.log.channel+"\x00"+queued.turnID+"\x00"+queued.receiptID)))
	_, err := w.log.client.Chat().SendMessage(ctx, ChannelType, w.log.channel, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{ID: &id, UserID: &queued.author.ID, Attachments: conversation.ChatAttachments(queued.artifacts),
			Custom: map[string]any{generatingField: false, SourceField: SourceAgent}},
	})
	if err != nil {
		w.log.logger.Error("could not store a saved artifact card", "turn", queued.turnID, "error", err)
	}
}

// show sends what has been written or heard since the last tick to anyone watching. The
// first revision is stored, so the channel has a message to update and to keep if it never
// settles; the rest are ephemeral, which reach watchers without a write per token.
func (w *writer) show() {
	for turnID, writing := range w.writing {
		w.showOne(writing, SourceAgent, "turn", turnID)
	}
	for userID, listening := range w.listening {
		w.showOne(listening, SourceSpeech, "user", userID)
	}
}

func (w *writer) showOne(writing *reply, source string, key, id string) {
	if writing.text == writing.shown {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()
	var err error
	if writing.messageID == "" {
		writing.messageID, err = w.send(ctx, writing.author, writing.text, true, false, source)
	} else {
		_, err = w.log.client.Chat().EphemeralMessageUpdate(ctx, writing.messageID,
			&getstream.EphemeralMessageUpdateRequest{
				UserID: &writing.author.ID,
				Set: map[string]any{
					"text": writing.text, generatingField: true, SourceField: source,
				},
			})
	}
	if err != nil {
		w.log.logger.Error("could not show a message being written", key, id, "error", err)
		return
	}
	writing.shown = writing.text
}

// retract removes what a participant was heard saying before it settled, so words the
// agent never took as said are not left in the channel as if they were. Nobody sent it,
// so it is deleted outright rather than left behind as an empty or deleted message.
func (w *writer) retract(userID string) {
	listening, started := w.listening[userID]
	delete(w.listening, userID)
	if !started || listening.messageID == "" {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()
	hard := true
	if _, err := w.log.client.Chat().DeleteMessage(ctx, listening.messageID,
		&getstream.DeleteMessageRequest{Hard: &hard}); err != nil {
		// Left alone it would still show the words, as if still being said. Emptied, it
		// at least stops saying so, and clients do not show empty speech.
		w.log.logger.Error("could not remove speech the agent did not take as said", "user", userID, "error", err)
		w.patch(listening, "", false, SourceSpeech)
	}
}

// settle stores what a reply came to and stops it saying it is still being written. Text
// is what the model ended up with, or empty for a reply nobody finished, which is kept as
// far as it got.
func (w *writer) settle(turnID, text string) {
	writing, streamed := w.writing[turnID]
	if !streamed {
		// A reply that never streamed is just a line of the conversation.
		w.store(w.log.agent, text, SourceAgent, false)
		return
	}
	delete(w.writing, turnID)

	if text == "" {
		text = writing.text
	}
	if writing.messageID == "" {
		// It finished before the first tick, so there is nothing to correct.
		w.store(writing.author, text, SourceAgent, false)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()

	// The pieces were ephemeral, so this is the write that leaves the reply behind.
	_, err := w.log.client.Chat().UpdateMessagePartial(ctx, writing.messageID,
		&getstream.UpdateMessagePartialRequest{
			UserID: &writing.author.ID,
			Set: map[string]any{
				"text": text, generatingField: false, interruptedField: false, SourceField: SourceAgent,
			},
		})
	if err != nil {
		w.log.logger.Error("could not store a finished reply", "turn", turnID, "error", err)
	}
}

// closeOut finishes whatever was still being written. The queue closes when the call is
// over, and a reply left generating would say it was still coming forever. Unplayed model
// text is retracted rather than stored as a finished spoken reply.
func (w *writer) closeOut() {
	for turnID := range w.writing {
		w.abandon(turnID, "")
	}
	for userID := range w.listening {
		w.retract(userID)
	}
}

func (w *writer) ensure(queued message) *reply {
	writing, started := w.writing[queued.turnID]
	if !started {
		writing = &reply{author: queued.author}
		w.writing[queued.turnID] = writing
	}
	return writing
}

// abandon retracts an unplayed reply so interrupted model text is not a finished Chat
// line. spoken is what a native model reported after the cut, which is worth keeping as
// interrupted rather than complete.
func (w *writer) abandon(turnID, spoken string) {
	writing, started := w.writing[turnID]
	delete(w.writing, turnID)
	if spoken != "" {
		if started && writing.messageID != "" {
			w.patch(writing, spoken, true, SourceAgent)
			return
		}
		w.store(w.log.agent, spoken, SourceAgent, true)
		return
	}
	if !started || writing.messageID == "" {
		return
	}
	w.patch(writing, "", true, SourceAgent)
}

func (w *writer) patch(writing *reply, text string, interrupted bool, source string) {
	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()
	_, err := w.log.client.Chat().UpdateMessagePartial(ctx, writing.messageID,
		&getstream.UpdateMessagePartialRequest{
			UserID: &writing.author.ID,
			Set: map[string]any{
				"text": text, generatingField: false, interruptedField: interrupted, SourceField: source,
			},
		})
	if err != nil {
		w.log.logger.Error("could not close an interrupted reply", "error", err)
	}
}

// store writes one whole line of the conversation.
func (w *writer) store(author User, text, source string, interrupted bool) {
	if author.ID == "" || text == "" {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()

	if _, err := w.send(ctx, author, text, false, interrupted, source); err != nil {
		w.log.logger.Error("could not store a message", "user", author.ID, "error", err)
	}
}

// send stores one message and returns its id, creating its author first if the app has
// never seen them.
func (w *writer) send(ctx context.Context, author User, text string, generating, interrupted bool, source string) (string, error) {
	if _, seen := w.known[author.ID]; !seen {
		if err := w.log.upsert(ctx, author); err != nil {
			return "", fmt.Errorf("storing the speaker: %w", err)
		}
		w.known[author.ID] = struct{}{}
	}

	response, err := w.log.client.Chat().SendMessage(ctx, ChannelType, w.log.channel,
		&getstream.SendMessageRequest{
			Message: getstream.MessageRequest{
				Text:   &text,
				UserID: &author.ID,
				Custom: map[string]any{
					generatingField: generating, interruptedField: interrupted, SourceField: source,
				},
			},
		})
	if err != nil {
		return "", err
	}
	return response.Data.Message.ID, nil
}

func (l *Log) upsert(ctx context.Context, user User) error {
	request := getstream.UserRequest{ID: user.ID}
	if user.Name != "" {
		request.Name = &user.Name
	}
	return conversation.CreateMissingUsers(ctx, l.client, map[string]getstream.UserRequest{user.ID: request})
}

// participantUser is who a participant is in chat. Their user id is what identifies them
// across calls; the per-call session id would give them a new identity every time.
func participantUser(participant stt.Participant) User {
	id := participant.UserID
	if id == "" {
		id = participant.ID
	}
	return User{ID: id, Name: participant.Name}
}
