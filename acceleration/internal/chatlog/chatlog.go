// Package chatlog stores what was said in a conversation as Stream Chat messages.
//
// A voice call leaves nothing behind once it ends. Writing each settled transcript and
// each reply into a chat channel gives the conversation a history that outlives the call
// and that any Stream Chat client can already read, without this service having to serve
// a transcript API of its own.
//
// Writing is asynchronous and drops rather than blocks: a conversation must never wait on
// the network to store what was just said.
package chatlog

import (
	"context"
	"errors"
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
	case agent.Responded:
		l.Say(l.agent, typed.Text)
	}
}

// Say queues one line of the conversation, dropping it if the writer is too far behind.
func (l *Log) Say(author User, text string) {
	if author.ID == "" || text == "" {
		return
	}

	select {
	case l.queue <- message{author: author, text: text}:
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

	// Server-side sends name their author, so a user the app has never seen has to exist
	// before their first message. The writer is the only goroutine here, so this needs no
	// lock of its own.
	known := map[string]struct{}{l.agent.ID: {}}

	for queued := range l.queue {
		ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)

		if _, seen := known[queued.author.ID]; !seen {
			if err := l.upsert(ctx, queued.author); err != nil {
				l.logger.Error("could not store a speaker", "user", queued.author.ID, "error", err)
				cancel()
				continue
			}
			known[queued.author.ID] = struct{}{}
		}

		_, err := l.client.Chat().SendMessage(ctx, channelType, l.agentID, &getstream.SendMessageRequest{
			Message: getstream.MessageRequest{Text: &queued.text, UserID: &queued.author.ID},
		})
		if err != nil {
			l.logger.Error("could not store a message", "user", queued.author.ID, "error", err)
		}
		cancel()
	}
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
