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
//
// While the agent works on a reply, the channel carries Stream's AI indicator, as a
// conversation in writing does, so a chat client shows it thinking before any words.
package chatlog

import (
	"cmp"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ChannelType is the Stream Chat channel type transcripts are written to.
const ChannelType = streamapp.AgentChannelType

// queueSize bounds how far the writer may fall behind before messages are dropped.
const queueSize = 256

// writeTimeout bounds a single write so a stuck network cannot wedge the writer.
const writeTimeout = 10 * time.Second

// indicatorTimeout bounds sending an AI indicator, which the writer waits on: a slow one
// holds back the transcript behind it. It is what a conversation in writing allows one.
const indicatorTimeout = 5 * time.Second

// staleIndicator is how long the agent is shown thinking with nothing changing and no tool
// running before the indicator is cleared anyway. Not every way a reply can end reaches the
// transcript (a model switched mid-turn, delegated work that comes to nothing), and a
// watcher would otherwise see it thinking for the rest of the call. A voice reply's first
// words come within seconds, so this is long past any real one.
const staleIndicator = 30 * time.Second

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

// purposeField says why the agent said a line unasked: one of purposes.
const purposeField = "purpose"

// purposes is every reason the agent says a line unasked. A reader passes on no other, since
// anything else that wrote the field is not a reason a client knows.
var purposes = []string{agent.PurposeGreeting, agent.PurposeCheckIn, agent.PurposeUpdate}

// turnField names the turn a reply was written for, so a reader can match the stored reply
// to the one it watched being written. It is the turn_id the session socket's speaking and
// spoke name the reply's utterances by, so a page can reveal the message as it is heard.
// Every message of the agent's that was said carries it: one being written, settled, cut
// off, and a line said as written. A turn's reply is one message, however many syntheses a
// voice taking a sentence at a time makes of it, so a synthesis id is not stamped: a
// message is written before the voice has any of it.
const turnField = "turn_id"

// interruptedTextField keeps what a reply the caller cut off was going to say. Its text is
// emptied, so nothing reads it as a finished spoken line, but a transcript shows it as
// talked over rather than leaving out a reply the caller heard the start of.
const interruptedTextField = "interrupted_text"

// aiGeneratedField is how Stream's AI components tell the assistant's replies from the rest,
// the same field a conversation in writing sets. It goes on the agent's lines only.
const aiGeneratedField = "ai_generated"

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
	// timed is how long the stages of a turn took, to be shown on its reply.
	timed
	// responding is the agent starting on a reply.
	responding
	// unsaid is a reply the model finished with nothing to say.
	unsaid
	// toolStarted is a tool starting, and toolRan it coming back.
	toolStarted
	toolRan
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
	// Opened says the caller has just made or read Channel, so Start does not ask Stream for
	// it again.
	Opened bool
	// Agent is the user the agent's own replies are written as.
	Agent User
	// VisibleTools are the agent config's visible_tools. A shown tool's stored artifacts
	// get a card of their own; empty shows search and web_search.
	VisibleTools []string
	// CustomerID is stamped on a channel this log creates, so reading it back can tell
	// whose it is. A bound conversation's channel already carries the stamp.
	CustomerID string
	// Timings writes how long each stage of a turn took after the reply to it, and on the
	// message as the timings field, for a developer talking to the agent. Off by default:
	// it is not something a caller should read.
	Timings bool

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
	opened   bool
	customer string
	agent    User
	visible  []string
	timings  bool
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
	// lineID is the line a participant's words were heard as, which they are stored under.
	lineID string
	// purpose is why the agent said a line unasked, empty for an answer.
	purpose string
	kind    kind
	// source is what the message is written as, one of the SourceField values.
	source string
	// receiptID is the tool call an artifact card or a tool event is for. It keeps a card's
	// identity stable.
	receiptID string
	// search says a tool that started is a search.
	search bool
	// pending says the model finished a reply with tools or delegated work still to come.
	pending bool
	// continues is the turn whose tools or delegated work a reply being started delivers.
	continues string
	artifacts []conversation.ArtifactAttachment
	// timings is what a timed message carries.
	timings *timings
}

// New validates the options and returns a Log. It writes nothing; Start does that.
func New(options Options) (*Log, error) {
	if options.AgentID == "" {
		return nil, stack.Wrap(errors.New("chatlog: an agent id is required"))
	}
	if options.Agent.ID == "" {
		return nil, stack.Wrap(errors.New("chatlog: an agent user id is required"))
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	client := options.Client
	if client == nil {
		if options.APIKey == "" || options.APISecret == "" {
			return nil, stack.Wrap(errors.New("chatlog: a client, or an api key and secret, are required"))
		}
		var err error
		if client, err = getstream.NewClient(options.APIKey, options.APISecret); err != nil {
			return nil, stack.Wrap(err)
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
		opened:   options.Channel != "" && options.Opened,
		customer: options.CustomerID,
		agent:    options.Agent,
		visible:  options.VisibleTools,
		timings:  options.Timings,
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
	if !l.opened {
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
		if _, err := l.client.Chat().GetOrCreateChannel(ctx, ChannelType, l.channel, request); err != nil {
			return stack.Wrap(err)
		}
	}

	l.started.Store(true)
	go l.run()
	return nil
}

// Record stores the speech in an agent event. Events that are not something somebody
// said are ignored, so a caller can hand it every event without filtering.
func (l *Log) Record(event agent.Event) {
	switch typed := event.(type) {
	case agent.Responding:
		if typed.TurnID != "" {
			l.enqueue(message{turnID: typed.TurnID, continues: typed.Continues, kind: responding})
		}
	case agent.ToolStarted:
		if typed.ID != "" {
			l.enqueue(message{turnID: typed.TurnID, receiptID: typed.ID, kind: toolStarted, search: conversation.SearchTool(typed.Tool)})
		}
	case agent.ToolRan:
		if typed.Err == nil && typed.ID != "" && typed.TurnID != "" && conversation.ToolVisible(l.visible, typed.Tool) {
			if artifacts := conversation.StoredArtifacts(typed.Result); len(artifacts) > 0 {
				l.enqueue(message{author: l.agent, turnID: typed.TurnID, kind: artifact, source: SourceAgent, receiptID: typed.ID, artifacts: artifacts})
			}
		}
		if typed.ID != "" {
			l.enqueue(message{turnID: typed.TurnID, receiptID: typed.ID, kind: toolRan})
		}
	case agent.Hearing:
		if typed.Text == "" {
			return
		}
		l.enqueue(message{author: participantUser(typed.Participant), text: typed.Text, lineID: typed.LineID, kind: hearing, source: SourceSpeech})
	case agent.Heard:
		if typed.Text == "" {
			return
		}
		l.enqueue(message{author: participantUser(typed.Participant), text: typed.Text, lineID: typed.LineID, kind: heard, source: SourceSpeech})
	case agent.Decided:
		if typed.Kind == string(agent.ActIgnore) {
			l.enqueue(message{author: participantUser(typed.Participant), lineID: typed.LineID, kind: ignored, source: SourceSpeech})
		}
	case agent.ResponseDelta:
		l.enqueue(message{author: l.agent, text: typed.Text, turnID: typed.TurnID, kind: piece, source: SourceAgent})
	case agent.Responded:
		if typed.Text == "" {
			// Nothing will be said or spoken, so nothing else ends the reply.
			if typed.TurnID != "" {
				l.enqueue(message{turnID: typed.TurnID, kind: unsaid, pending: typed.PendingWork})
			}
			return
		}
		if typed.TurnID == "" {
			l.enqueue(message{author: l.agent, text: typed.Text, kind: whole, source: SourceAgent})
			return
		}
		l.enqueue(message{author: l.agent, text: typed.Text, turnID: typed.TurnID, purpose: typed.Purpose, kind: prepared, source: SourceAgent, pending: typed.PendingWork})
	case agent.Said:
		if typed.Text == "" || typed.TurnID == "" {
			return
		}
		// Shown while it is said, as a reply is while it is written, and kept like a
		// reply the model finished: stored once the voice has said it, and as far as it
		// got if the caller talks over it.
		l.enqueue(message{author: l.agent, text: typed.Text, turnID: typed.TurnID, purpose: typed.Purpose, kind: piece, source: SourceAgent})
		l.enqueue(message{author: l.agent, text: typed.Text, turnID: typed.TurnID, purpose: typed.Purpose, kind: prepared, source: SourceAgent})
	case agent.Spoke:
		if typed.TurnID == "" {
			return
		}
		l.enqueue(message{author: l.agent, turnID: typed.TurnID, kind: spoken, source: SourceAgent})
	case agent.Interrupted:
		// The model may already have finished, but the caller did not hear that reply in
		// full, so it must not be stored as a completed spoken line.
		l.enqueue(message{author: l.agent, turnID: typed.TurnID, kind: interrupt, source: SourceAgent})
	case agent.Turn:
		if !l.timings || typed.TurnID == "" {
			return
		}
		if described, ok := timingsOf(typed); ok {
			l.enqueue(message{turnID: typed.TurnID, kind: timed, timings: &described})
		}
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
	// named is what the stored message is known by besides its words.
	named identity
	// messageID is the stored message watchers see updated, once there is one.
	messageID string
	text      string
	// generated is the model's finished text. It is not stored as spoken until the voice
	// has actually finished, because Responded arrives while TTS may still be playing.
	generated string
	// shown is what watchers were last sent, so an unchanged reply is not sent again.
	shown string
	// heard is what a line being said again was stored as when it was first heard, so its
	// message goes back to that if the rest is not taken as said.
	heard string
	// spoke says the voice finished some of it before the model finished writing it, which
	// a voice taking a sentence at a time does. It is settled once the model has, so the
	// rest of the reply stays in the same message rather than starting another.
	spoke bool
}

// writer is the state behind the queue. The writer goroutine is the only one that touches
// it, so it needs no lock of its own.
type writer struct {
	log         *Log
	known       map[string]struct{}
	writing     map[string]*reply
	interrupted map[string]struct{}
	// listening is each line still being said, by listenKey, until it settles.
	listening map[string]*reply
	// heardLines is what each recently heard line was stored as, by line id. A line the
	// caller carries on with before it is answered is heard again growing, and is written
	// over rather than stored beside itself. heardOrder bounds it, oldest first.
	heardLines map[string]heardLine
	heardOrder []string
	// recent is what is held of the last few turns until their reply and their timings have
	// met, and turns is the order they were first held in, oldest first.
	recent map[string]*turnRecord
	turns  []string
	// quietUntil is when showing what is being written may be tried again, after a show
	// Stream refused. Every tick would otherwise send it again, and Stream answers a 429
	// by closing the connection, so a limited app spent a new connection on each.
	quietUntil time.Time
	// answering is the turn the agent is working on a reply to, from Responding until it has
	// nothing left to do for it. replied says its reply was heard, cut off or came to nothing,
	// and owed that the model asked for work that has not come back yet.
	answering string
	replied   bool
	owed      bool
	// tools are the tool calls running for answering and the turns it carries on from, by
	// call id, true for a search. A new turn of the caller's lets go of them, so one whose
	// end was never written is not shown for the rest of the call.
	tools map[string]bool
	// unstored is the id each reply the agent started on is stored under, by turn, until it
	// has been; replyOrder bounds it, oldest first, since a reply may never be stored.
	unstored   map[string]string
	replyOrder []string
	// indicated is the AI indicator watchers were last sent, and indicatedAt when it changed.
	indicated   indication
	indicatedAt time.Time
	// now is the writer's clock.
	now func() time.Time
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
		recent:      map[string]*turnRecord{},
		heardLines:  map[string]heardLine{},
		tools:       map[string]bool{},
		unstored:    map[string]string{},
		now:         time.Now,
	}
}

// handle takes one queued message.
func (w *writer) handle(queued message) {
	// After the writes, so the indicator never runs ahead of the messages it is about: a
	// clear goes out once the reply is stored.
	defer w.indicate(false)
	switch queued.kind {
	case artifact:
		w.storeArtifacts(queued)
	case piece:
		writing, started := w.writing[queued.turnID]
		if !started {
			writing = &reply{author: queued.author, named: identity{turn: queued.turnID, purpose: queued.purpose}}
			w.writing[queued.turnID] = writing
		}
		writing.text += queued.text
	case prepared:
		writing := w.ensure(queued)
		writing.generated = queued.text
		if w.answering == queued.turnID {
			w.owed = queued.pending
		}
		if _, cut := w.interrupted[queued.turnID]; cut {
			w.abandon(queued.turnID, queued.text)
			return
		}
		if writing.spoke {
			w.settle(queued.turnID, queued.text)
		}
	case spoken:
		if _, cut := w.interrupted[queued.turnID]; cut {
			return
		}
		writing, started := w.writing[queued.turnID]
		text := queued.text
		if started {
			if writing.generated == "" && writing.text != "" {
				writing.spoke = true
				return
			}
			text = cmp.Or(writing.generated, writing.text)
		}
		w.settle(queued.turnID, text)
	case interrupt:
		w.interrupted[queued.turnID] = struct{}{}
		w.abandon(queued.turnID, "")
	case hearing:
		key := listenKey(queued)
		listening, started := w.listening[key]
		if !started {
			listening = &reply{author: queued.author, named: identity{id: queued.lineID}}
			// A line already heard is being said again, grown: its message is written over.
			kept := w.heardLines[queued.lineID]
			listening.messageID, listening.heard = kept.messageID, kept.text
			w.listening[key] = listening
		}
		listening.text = queued.text
	case heard:
		key := listenKey(queued)
		listening, started := w.listening[key]
		delete(w.listening, key)
		if !started || listening.messageID == "" {
			w.rememberHeard(queued.lineID, queued.text,
				w.store(queued.author, queued.text, SourceSpeech, false, identity{id: queued.lineID}))
			return
		}
		w.patch(listening, queued.text, false, SourceSpeech)
		w.rememberHeard(queued.lineID, queued.text, listening.messageID)
	case ignored:
		w.retract(listenKey(queued))
	case whole:
		w.store(queued.author, queued.text, queued.source, false, identity{})
	case timed:
		w.arrived(queued.turnID, queued.timings)
	case responding:
		w.respond(queued.turnID, queued.continues)
	case unsaid:
		if w.answering == queued.turnID {
			w.owed = queued.pending
		}
		// The model finished with nothing it stands by, so a reply some of which was already
		// spoken is kept as far as it was written.
		if writing, started := w.writing[queued.turnID]; started && writing.spoke {
			w.settle(queued.turnID, "")
		}
		w.done(queued.turnID)
	case toolStarted:
		if w.answering != "" {
			w.tools[queued.receiptID] = queued.search
		}
	case toolRan:
		delete(w.tools, queued.receiptID)
		// What it was waiting on came back. A reply that carries on from it starts a turn of
		// its own, which comes next if there is one.
		if len(w.tools) == 0 {
			w.owed = false
		}
	}
}

// heardKept is how many heard lines the writer remembers the message of. Only the last few
// can be carried on with: one is answered or dropped within a turn or two.
const heardKept = 8

// heardLine is the message a heard line was stored as, and what it said.
type heardLine struct {
	messageID string
	text      string
}

// listenKey is what a line being said is kept under: its id, which the caller's next line
// does not share, or for words with no line, who is saying them.
func listenKey(queued message) string {
	if queued.lineID != "" {
		return queued.lineID
	}
	return "user:" + queued.author.ID
}

// rememberHeard keeps the message a heard line was stored as, so the line said again grown
// is written over.
func (w *writer) rememberHeard(line, text, messageID string) {
	if line == "" || messageID == "" {
		return
	}
	if _, kept := w.heardLines[line]; !kept {
		w.heardOrder = append(w.heardOrder, line)
	}
	w.heardLines[line] = heardLine{messageID: messageID, text: text}
	for len(w.heardOrder) > heardKept {
		delete(w.heardLines, w.heardOrder[0])
		w.heardOrder = w.heardOrder[1:]
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
			Custom: map[string]any{generatingField: false, SourceField: SourceAgent, aiGeneratedField: true}},
	})
	if err != nil {
		w.log.logger.Error("could not store a saved artifact card", "turn", queued.turnID, "error", err)
	}
}

// show sends what has been written or heard since the last tick to anyone watching. The
// first revision is stored, so the channel has a message to update and to keep if it never
// settles; the rest are ephemeral, which reach watchers without a write per token.
func (w *writer) show() {
	// A reply's first show stores its message, and the tick is also what notices an
	// indicator gone stale.
	defer w.indicate(false)
	if w.now().Before(w.quietUntil) {
		return
	}
	for turnID, writing := range w.writing {
		w.showOne(writing, SourceAgent, "turn", turnID)
	}
	for key, listening := range w.listening {
		w.showOne(listening, SourceSpeech, "line", key)
	}
}

func (w *writer) showOne(writing *reply, source string, key, id string) {
	if writing.text == writing.shown || w.now().Before(w.quietUntil) {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()
	var err error
	if writing.messageID == "" {
		writing.messageID, err = w.send(ctx, writing.author, writing.text, true, false, source, writing.named)
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
		w.quietUntil = w.now().Add(quietFor(err))
		w.log.logger.Error("could not show a message being written", key, id, "error", err)
		return
	}
	writing.shown = writing.text
}

// quietFor is how long to stop showing what is being written after Stream refused to: until
// the limit it hit resets, or a moment for any other failure. What is finished is still
// stored in the meantime, so only the live view goes quiet.
func quietFor(err error) time.Duration {
	var refused *getstream.StreamError
	if !errors.As(err, &refused) || !errors.Is(err, getstream.ErrRateLimited) {
		return time.Second
	}
	if refused.RetryAfter > 0 {
		return refused.RetryAfter
	}
	if refused.RateLimit != nil && refused.RateLimit.Reset > 0 {
		return time.Until(time.Unix(refused.RateLimit.Reset, 0))
	}
	return 10 * time.Second
}

// retract removes what a participant was heard saying before it settled, so words the
// agent never took as said are not left in the channel as if they were. Nobody sent it,
// so it is deleted outright rather than left behind as an empty or deleted message. A line
// heard before and being said again goes back to what it was heard as: that much was said.
func (w *writer) retract(key string) {
	listening, started := w.listening[key]
	delete(w.listening, key)
	if !started || listening.messageID == "" {
		return
	}
	if listening.heard != "" {
		w.patch(listening, listening.heard, false, SourceSpeech)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()
	hard := true
	if _, err := w.log.client.Chat().DeleteMessage(ctx, listening.messageID,
		&getstream.DeleteMessageRequest{Hard: &hard}); err != nil {
		// Left alone it would still show the words, as if still being said. Emptied, it
		// at least stops saying so, and clients do not show empty speech.
		w.log.logger.Error("could not remove speech the agent did not take as said", "line", key, "error", err)
		w.patch(listening, "", false, SourceSpeech)
	}
}

// settle stores what a reply came to and stops it saying it is still being written. Text
// is what the model ended up with, or empty for a reply nobody finished, which is kept as
// far as it got.
func (w *writer) settle(turnID, text string) {
	w.done(turnID)
	writing, streamed := w.writing[turnID]
	if !streamed {
		// A reply that never streamed is just a line of the conversation.
		id := w.store(w.log.agent, text, SourceAgent, false, identity{turn: turnID})
		w.keep(turnID, posted{id: id, author: w.log.agent, text: text})
		return
	}
	delete(w.writing, turnID)

	if text == "" {
		text = writing.text
	}
	if writing.messageID == "" {
		// It finished before the first tick, so there is nothing to correct.
		id := w.store(writing.author, text, SourceAgent, false, writing.named)
		w.keep(turnID, posted{id: id, author: writing.author, text: text})
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()

	// The pieces were ephemeral, so this is the write that leaves the reply behind.
	finished := map[string]any{
		"text": text, generatingField: false, interruptedField: false, SourceField: SourceAgent,
	}
	if writing.named.purpose != "" {
		finished[purposeField] = writing.named.purpose
	}
	_, err := w.log.client.Chat().UpdateMessagePartial(ctx, writing.messageID,
		&getstream.UpdateMessagePartialRequest{UserID: &writing.author.ID, Set: finished})
	if err != nil {
		w.log.logger.Error("could not store a finished reply", "turn", turnID, "error", err)
	}
	w.keep(turnID, posted{id: writing.messageID, author: writing.author, text: text})
}

// closeOut finishes whatever was still being written. The queue closes when the call is
// over, and a reply left generating would say it was still coming forever. Unplayed model
// text is retracted rather than stored as a finished spoken reply.
func (w *writer) closeOut() {
	for turnID := range w.writing {
		w.abandon(turnID, "")
	}
	for key := range w.listening {
		w.retract(key)
	}
	// Nothing is being worked on once the call is over, whatever was not reported ending.
	w.forget()
	w.indicate(true)
}

func (w *writer) ensure(queued message) *reply {
	writing, started := w.writing[queued.turnID]
	if !started {
		writing = &reply{author: queued.author, named: identity{turn: queued.turnID, purpose: queued.purpose}}
		w.writing[queued.turnID] = writing
	}
	if queued.purpose != "" {
		writing.named.purpose = queued.purpose
	}
	return writing
}

// abandon retracts an unplayed reply so interrupted model text is not a finished Chat
// line. spoken is what a native model reported after the cut, which is worth keeping as
// interrupted rather than complete.
func (w *writer) abandon(turnID, spoken string) {
	w.done(turnID)
	writing, started := w.writing[turnID]
	delete(w.writing, turnID)
	named := identity{turn: turnID}
	if started {
		named = writing.named
	}
	// What it was for is kept with it, though it reached the writer after the first write.
	more := map[string]any{}
	if named.purpose != "" {
		more[purposeField] = named.purpose
	}
	if spoken != "" {
		if started && writing.messageID != "" {
			w.patchWith(writing, spoken, true, SourceAgent, more)
			w.keep(turnID, posted{id: writing.messageID, author: writing.author, text: spoken, interrupted: true})
			return
		}
		id := w.store(w.log.agent, spoken, SourceAgent, true, named)
		w.keep(turnID, posted{id: id, author: w.log.agent, text: spoken, interrupted: true})
		return
	}
	if !started || writing.messageID == "" {
		return
	}
	// Emptied and told what it was going to say in one write, so a transcript read between
	// the two never finds the reply missing.
	more[interruptedTextField] = cmp.Or(writing.generated, writing.text)
	w.patchWith(writing, "", true, SourceAgent, more)
	w.keep(turnID, posted{id: writing.messageID, author: writing.author, interrupted: true})
}

func (w *writer) patch(writing *reply, text string, interrupted bool, source string) {
	w.patchWith(writing, text, interrupted, source, nil)
}

// patchWith is patch, setting the fields in more as well in the same write.
func (w *writer) patchWith(writing *reply, text string, interrupted bool, source string, more map[string]any) {
	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()
	set := map[string]any{
		"text": text, generatingField: false, interruptedField: interrupted, SourceField: source,
	}
	maps.Copy(set, more)
	_, err := w.log.client.Chat().UpdateMessagePartial(ctx, writing.messageID,
		&getstream.UpdateMessagePartialRequest{UserID: &writing.author.ID, Set: set})
	if err != nil {
		w.log.logger.Error("could not close an interrupted reply", "error", err)
	}
}

// respond takes turnID as the one the agent is working on a reply to. A reply to the
// caller lets go of the tools before it; one delivering their results carries them on.
func (w *writer) respond(turnID, continues string) {
	if continues == "" {
		clear(w.tools)
	}
	w.answering, w.replied, w.owed = turnID, false, false
	if _, given := w.unstored[turnID]; given {
		return
	}
	w.unstored[turnID] = replyID(w.log.channel, turnID)
	w.replyOrder = append(w.replyOrder, turnID)
	for len(w.replyOrder) > recentTurns {
		delete(w.unstored, w.replyOrder[0])
		w.replyOrder = w.replyOrder[1:]
	}
}

// done notes that the reply to turnID was heard, cut off or came to nothing. The agent may
// still be working on it, until its tools come back. A reply to a turn it has started on
// since does not end that one.
func (w *writer) done(turnID string) {
	if w.answering == turnID {
		w.replied = true
	}
}

// forget lets go of the turn the agent was working on, and its tools.
func (w *writer) forget() {
	w.answering, w.replied, w.owed = "", false, false
	clear(w.tools)
}

// replyID is the id a reply the agent started on is stored under, which its AI indicator
// names before the reply has a word in it. It is the same for the same turn, as an
// artifact card's is, so a store that timed out after landing is written over, not repeated.
func replyID(channel, turnID string) string {
	return fmt.Sprintf("voice-reply-%x", sha256.Sum256([]byte(channel+"\x00"+turnID)))
}

// indication is the AI indicator watchers were last sent: the reply's message, and its
// state. Zero is cleared.
type indication struct {
	message string
	state   string
}

// indicate brings Stream's AI indicator in line with what the agent is doing, as a
// conversation in writing does: thinking once it starts on a reply and while its tools run,
// checking external sources while a search runs, generating once the reply shows, and
// cleared once there is nothing left to do for it. It sends only on a change, and nothing
// while Stream has asked the writer to hold off: the next tick after sends what is true by
// then. final sends anyway, since nothing comes after it. The indicator is a best-effort
// signal, the message's own generating field stays authoritative, so a failed send is not
// retried.
func (w *writer) indicate(final bool) {
	now := w.now()
	next := indication{}
	if w.answering != "" {
		next = indication{message: replyID(w.log.channel, w.answering), state: conversation.AIStateThinking}
		writing, written := w.writing[w.answering]
		switch {
		case written && writing.messageID != "":
			next = indication{message: writing.messageID, state: conversation.AIStateGenerating}
		case slices.Contains(slices.Collect(maps.Values(w.tools)), true):
			next.state = conversation.AIStateExternalSources
		}
		// Nothing is left to do for it, or nothing has moved for a long time: no tool is
		// running, no word is being written and the indicator has not changed.
		finished := w.replied && !w.owed
		stale := !written && next == w.indicated && now.Sub(w.indicatedAt) >= staleIndicator
		if len(w.tools) == 0 && (finished || stale) {
			w.forget()
			next = indication{}
		}
	}
	if next == w.indicated || (!final && now.Before(w.quietUntil)) {
		return
	}
	// The last indicator is cleared when nothing replaces it, or when it named a message the
	// next does not, as a conversation in writing clears a reply it moves on from.
	if w.indicated.state != "" && (next.state == "" || w.indicated.message != next.message) {
		w.sendIndicator(w.indicated.message, "")
	}
	if next.state != "" {
		w.sendIndicator(next.message, next.state)
	}
	w.indicated, w.indicatedAt = next, now
}

// sendIndicator sends ai_indicator.update with state, or ai_indicator.clear without one, as
// the agent.
func (w *writer) sendIndicator(messageID, state string) {
	ctx, cancel := context.WithTimeout(context.Background(), indicatorTimeout)
	defer cancel()
	event := conversation.IndicatorEvent(w.log.agent.ID, messageID, state)
	if _, err := w.log.client.Chat().SendEvent(ctx, ChannelType, w.log.channel,
		&getstream.SendEventRequest{Event: event}); err != nil {
		w.log.logger.Warn("could not send the agent's AI indicator", "type", event.Type, "error", err)
	}
}

// recentTurns is how many turns the writer holds a reply or its timings for while it waits for
// the other. A turn is reported when its first sound is out, which is before or after the
// voice has finished the reply, but never by many turns.
const recentTurns = 32

// turnRecord is what is held of one turn until its reply and its timings have met.
type turnRecord struct {
	// timings arrived before the reply was stored.
	timings *timings
	// reply is the message the reply was stored as, once it has been.
	reply *posted
}

// posted is a reply as it was left in the channel, which is what timings are written onto.
type posted struct {
	id          string
	author      User
	text        string
	interrupted bool
}

// keep notes where a turn's reply was left, and writes the turn's timings onto it if they
// are already here. A reply that left no message has nothing to show them on.
func (w *writer) keep(turnID string, post posted) {
	if !w.log.timings || post.id == "" {
		return
	}
	held := w.hold(turnID)
	held.reply = &post
	if held.timings != nil {
		w.annotate(turnID, held)
	}
}

// arrived takes a turn's timings, writing them onto its reply if that has been stored and
// holding them for it if not.
func (w *writer) arrived(turnID string, arrived *timings) {
	held := w.hold(turnID)
	held.timings = arrived
	if held.reply != nil {
		w.annotate(turnID, held)
	}
}

// hold returns what is held of a turn, making room for a turn not held yet by letting go of
// the oldest.
func (w *writer) hold(turnID string) *turnRecord {
	if held, ok := w.recent[turnID]; ok {
		return held
	}
	if len(w.turns) == recentTurns {
		delete(w.recent, w.turns[0])
		w.turns = slices.Delete(w.turns, 0, 1)
	}
	held := &turnRecord{}
	w.recent[turnID] = held
	w.turns = append(w.turns, turnID)
	return held
}

// annotate writes a turn's timings after its reply, which is all there is to keep of it.
func (w *writer) annotate(turnID string, held *turnRecord) {
	delete(w.recent, turnID)
	w.turns = slices.DeleteFunc(w.turns, func(id string) bool { return id == turnID })

	text := held.timings.line
	if held.reply.text != "" {
		text = held.reply.text + conversation.TimingsGap + text
	}
	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()
	_, err := w.log.client.Chat().UpdateMessagePartial(ctx, held.reply.id,
		&getstream.UpdateMessagePartialRequest{
			UserID: &held.reply.author.ID,
			Set:    map[string]any{"text": text, conversation.TimingsField: held.timings.fields},
		})
	if err != nil {
		w.log.logger.Error("could not show a reply's timings", "turn", turnID, "error", err)
	}
}

// store writes one whole line of the conversation and returns its id, empty if it was not
// written.
func (w *writer) store(author User, text, source string, interrupted bool, named identity) string {
	if author.ID == "" || text == "" {
		return ""
	}

	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()

	id, err := w.send(ctx, author, text, false, interrupted, source, named)
	if err != nil {
		w.log.logger.Error("could not store a message", "user", author.ID, "error", err)
	}
	return id
}

// identity is what a stored message is known by besides its words: its id, which is the line
// a participant's words were heard as, and the turn a reply was written for. Either can be
// empty, and the message is then named by Stream and the turn left out. A reply the agent
// started on is stored under the id it was given then (respond).
type identity struct {
	id   string
	turn string
	// purpose is why the agent said it unasked, kept so a reader can say so too.
	purpose string
}

// send stores one message and returns its id, creating its author first if the app has
// never seen them.
func (w *writer) send(ctx context.Context, author User, text string, generating, interrupted bool, source string, named identity) (string, error) {
	if _, seen := w.known[author.ID]; !seen {
		if err := w.log.upsert(ctx, author); err != nil {
			return "", fmt.Errorf("storing the speaker: %w", err)
		}
		w.known[author.ID] = struct{}{}
	}

	custom := map[string]any{
		generatingField: generating, interruptedField: interrupted, SourceField: source,
	}
	if source == SourceAgent {
		custom[aiGeneratedField] = true
	}
	if named.turn != "" {
		custom[turnField] = named.turn
	}
	if named.purpose != "" {
		custom[purposeField] = named.purpose
	}
	request := getstream.MessageRequest{Text: &text, UserID: &author.ID, Custom: custom}
	id, given := named.id, false
	if id == "" && named.turn != "" {
		// Only the reply's first message: a turn spoken in pieces can store another.
		id, given = w.unstored[named.turn]
	}
	if id != "" {
		request.ID = &id
	}
	response, err := w.log.client.Chat().SendMessage(ctx, ChannelType, w.log.channel,
		&getstream.SendMessageRequest{Message: request})
	if err != nil {
		if id == "" || !w.rewrite(ctx, author, id, text, custom) {
			return "", err
		}
	} else {
		id = response.Data.Message.ID
	}
	if given {
		delete(w.unstored, named.turn)
	}
	return id, nil
}

// rewrite writes over a line already stored under its id, which Chat refuses to store
// twice: a store that timed out here and still landed there, or a line forgotten as heard
// and said again. A reply's id is reused only in the first of these. It reports whether there was one to write over.
func (w *writer) rewrite(ctx context.Context, author User, line, text string, custom map[string]any) bool {
	set := maps.Clone(custom)
	set["text"] = text
	_, err := w.log.client.Chat().UpdateMessagePartial(ctx, line,
		&getstream.UpdateMessagePartialRequest{UserID: &author.ID, Set: set})
	return err == nil
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
