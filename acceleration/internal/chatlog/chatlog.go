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
// A reply is shown while it is still being written. The pieces go out as ephemeral
// updates, which reach anyone watching the channel without storing a version per token,
// and what the reply came to is stored once it is finished.
package chatlog

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"sync/atomic"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const (
	apiKeyEnvVar    = "STREAM_API_KEY"
	apiSecretEnvVar = "STREAM_API_SECRET"
)

// channelType is the Stream Chat channel type transcripts are written to.
const channelType = "messaging"

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

// kind says how a queued message relates to the reply it belongs to.
type kind int

const (
	// whole is a line that was said in full: a participant's turn, or a reply that never
	// streamed.
	whole kind = iota
	// piece is more of a reply that is still being written.
	piece
	// end closes a streamed reply, storing what it came to.
	end
)

// Options configures a Log. The credentials fall back to the environment, the same way
// the Stream edge reads them.
type Options struct {
	// AgentID names the channel: one agent's transcript lives in messaging:{agentID}.
	AgentID string
	// Agent is the user the agent's own replies are written as.
	Agent User

	// APIKey defaults to STREAM_API_KEY.
	APIKey string
	// APISecret defaults to STREAM_API_SECRET. These are server-side writes, so a secret
	// is required rather than a user token.
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
	client  *getstream.Stream
	agentID string
	agent   User
	logger  *slog.Logger

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
}

// New validates the options and returns a Log. It writes nothing; Start does that.
func New(options Options) (*Log, error) {
	if options.AgentID == "" {
		return nil, errors.New("chatlog: an agent id is required")
	}
	if options.Agent.ID == "" {
		return nil, errors.New("chatlog: an agent user id is required")
	}
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APISecret == "" {
		options.APISecret = os.Getenv(apiSecretEnvVar)
	}
	if options.APIKey == "" || options.APISecret == "" {
		return nil, errors.New("chatlog: " + apiKeyEnvVar + " and " + apiSecretEnvVar + " are required")
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	client, err := getstream.NewClient(options.APIKey, options.APISecret)
	if err != nil {
		return nil, err
	}

	return &Log{
		client:  client,
		agentID: options.AgentID,
		agent:   options.Agent,
		logger:  options.Logger.With("agent", options.AgentID),
		queue:   make(chan message, queueSize),
		done:    make(chan struct{}),
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
	_, err := l.client.Chat().GetOrCreateChannel(ctx, channelType, l.agentID,
		&getstream.GetOrCreateChannelRequest{
			Data: &getstream.ChannelInput{CreatedByID: &l.agent.ID},
		})
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
	case agent.Heard:
		l.Say(participantUser(typed.Participant), typed.Text)
	case agent.ResponseDelta:
		l.enqueue(message{author: l.agent, text: typed.Text, turnID: typed.TurnID, kind: piece})
	case agent.Responded:
		if typed.Text == "" {
			return
		}
		l.enqueue(message{author: l.agent, text: typed.Text, turnID: typed.TurnID, kind: end})
	case agent.Interrupted:
		// A reply nobody finished still has to stop saying it is being written, and what
		// the caller heard of it is worth keeping.
		l.enqueue(message{author: l.agent, turnID: typed.TurnID, kind: end})
	}
}

// Say queues one line of the conversation, dropping it if the writer is too far behind.
func (l *Log) Say(author User, text string) {
	if author.ID == "" || text == "" {
		return
	}
	l.enqueue(message{author: author, text: text, kind: whole})
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
func (l *Log) ChannelID() string { return l.agentID }

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
	// shown is what watchers were last sent, so an unchanged reply is not sent again.
	shown string
}

// writer is the state behind the queue. The writer goroutine is the only one that touches
// it, so it needs no lock of its own.
type writer struct {
	log     *Log
	known   map[string]struct{}
	writing map[string]*reply
}

func newWriter(l *Log) *writer {
	return &writer{
		log: l,
		// Server-side sends name their author, so a user the app has never seen has to
		// exist before their first message.
		known:   map[string]struct{}{l.agent.ID: {}},
		writing: map[string]*reply{},
	}
}

// handle takes one queued message.
func (w *writer) handle(queued message) {
	switch queued.kind {
	case piece:
		writing, started := w.writing[queued.turnID]
		if !started {
			writing = &reply{author: queued.author}
			w.writing[queued.turnID] = writing
		}
		writing.text += queued.text
	case end:
		w.settle(queued.turnID, queued.text)
	case whole:
		w.store(queued.author, queued.text)
	}
}

// show sends what has been written since the last tick to anyone watching. The first
// piece is stored, so the channel has a message to update and to keep if the reply is
// never finished; the rest are ephemeral, which reach watchers without a write per token.
func (w *writer) show() {
	for turnID, writing := range w.writing {
		if writing.text == writing.shown {
			continue
		}

		ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
		var err error
		if writing.messageID == "" {
			writing.messageID, err = w.send(ctx, writing.author, writing.text, true)
		} else {
			_, err = w.log.client.Chat().EphemeralMessageUpdate(ctx, writing.messageID,
				&getstream.EphemeralMessageUpdateRequest{
					UserID: &writing.author.ID,
					Set:    map[string]any{"text": writing.text, generatingField: true},
				})
		}
		cancel()

		if err != nil {
			w.log.logger.Error("could not show a reply being written", "turn", turnID, "error", err)
			continue
		}
		writing.shown = writing.text
	}
}

// settle stores what a reply came to and stops it saying it is still being written. Text
// is what the model ended up with, or empty for a reply nobody finished, which is kept as
// far as it got.
func (w *writer) settle(turnID, text string) {
	writing, streamed := w.writing[turnID]
	if !streamed {
		// A reply that never streamed is just a line of the conversation.
		w.store(w.log.agent, text)
		return
	}
	delete(w.writing, turnID)

	if text == "" {
		text = writing.text
	}
	if writing.messageID == "" {
		// It finished before the first tick, so there is nothing to correct.
		w.store(writing.author, text)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()

	// The pieces were ephemeral, so this is the write that leaves the reply behind.
	_, err := w.log.client.Chat().UpdateMessagePartial(ctx, writing.messageID,
		&getstream.UpdateMessagePartialRequest{
			UserID: &writing.author.ID,
			Set:    map[string]any{"text": text, generatingField: false},
		})
	if err != nil {
		w.log.logger.Error("could not store a finished reply", "turn", turnID, "error", err)
	}
}

// closeOut finishes whatever was still being written. The queue closes when the call is
// over, and a reply left generating would say it was still coming forever.
func (w *writer) closeOut() {
	for turnID := range w.writing {
		w.settle(turnID, "")
	}
}

// store writes one whole line of the conversation.
func (w *writer) store(author User, text string) {
	if author.ID == "" || text == "" {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()

	if _, err := w.send(ctx, author, text, false); err != nil {
		w.log.logger.Error("could not store a message", "user", author.ID, "error", err)
	}
}

// send stores one message and returns its id, creating its author first if the app has
// never seen them.
func (w *writer) send(ctx context.Context, author User, text string, generating bool) (string, error) {
	if _, seen := w.known[author.ID]; !seen {
		if err := w.log.upsert(ctx, author); err != nil {
			return "", fmt.Errorf("storing the speaker: %w", err)
		}
		w.known[author.ID] = struct{}{}
	}

	response, err := w.log.client.Chat().SendMessage(ctx, channelType, w.log.agentID,
		&getstream.SendMessageRequest{
			Message: getstream.MessageRequest{
				Text:   &text,
				UserID: &author.ID,
				Custom: map[string]any{generatingField: generating},
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
	_, err := l.client.UpdateUsers(ctx, &getstream.UpdateUsersRequest{
		Users: map[string]getstream.UserRequest{user.ID: request},
	})
	return err
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
