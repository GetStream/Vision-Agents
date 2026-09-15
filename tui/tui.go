// Package tui holds a conversation with an agent in a terminal.
//
// The agent is reached through a [Session], which an *agents.Session satisfies, so what
// is shown is whatever the backend reports: the answer as it is written, the tools being
// run to find it and how long each has taken, and whether what was said survived being
// saved. An application says how a session is opened and what the header should call it,
// and everything else is here.
//
//	err := tui.Run(ctx, tui.Options{
//		Open: func(ctx context.Context, id string) (tui.Session, error) {
//			return agent.Chat(ctx, agents.ChatOptions{Persist: true, ConversationID: id})
//		},
//		History:  tui.BackendHistory(stream.Backend{}, "jean"),
//		Branding: tui.Branding{Title: "Jean", Subtitle: "answers questions about weather"},
//	})
package tui

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/stream"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
)

const (
	// pulse is how often the running timers and the spinner move.
	pulse = 100 * time.Millisecond
	// closeTimeout is how long a session is given to end cleanly.
	closeTimeout = 10 * time.Second
	// composerHeight is how many lines of question are visible at once.
	composerHeight = 3
	// maxQuestion is as much as one question may be.
	maxQuestion = 6000
	// pendingID stands in for a question until the backend says what it saved it as.
	pendingID = "pending"
)

// Options is what an application has to say about its own conversation. Only Open is
// required.
type Options struct {
	// Open opens a session on a conversation, and is called again for every /new and
	// /resume. Required.
	Open Opener
	// History reads messages older than the session was given. Nil leaves /older saying
	// that saved history is not available, which is the truth for a conversation that is
	// not persisted.
	History History
	// ConversationID is the conversation to open on start. Empty starts a new one.
	ConversationID string
	// Branding names the agent and what it is for.
	Branding Branding
	// Theme is the palette. Its zero value is [DefaultTheme].
	Theme Theme
	// Header returns extra lines for the header, and is asked again for every frame, so
	// what it says may change as the conversation does.
	Header func(State) []string
	// OnOpen is told about every session as it becomes live, which is the moment to open
	// the same conversation somewhere else. It runs off the event loop and its error is
	// shown on the status line.
	OnOpen func(ctx context.Context, session Session) error
	// Commands are slash commands beyond the built-in ones.
	Commands []Command
	// Logger is where the conversation reports what it could not do quietly, such as
	// failing to close a session on the way out. Nil uses the default.
	Logger *slog.Logger
}

// Model is the conversation as a Bubble Tea model. Use [Run] unless it is being embedded
// in something larger.
type Model struct {
	options  Options
	branding Branding
	styles   styles
	logger   *slog.Logger
	ctx      context.Context

	input textarea.Model
	view  viewport.Model
	cache map[string]string
	// renderers is one markdown renderer per wrap width. It outlives a resize, since a
	// width already built for is the width the terminal may be returning to.
	renderers map[int]*glamour.TermRenderer

	// generation counts the sessions opened, so that what a session said after it was
	// replaced is recognised and dropped.
	generation     int
	session        Session
	conversationID string
	scope          string
	messages       []stream.ConversationMessage
	before         string
	truncated      bool

	width, height int
	// compact is how much of the chrome the terminal has room for. See [Model.relayout].
	compact int
	frame   int

	status       string
	statusFailed bool
	busy         bool
	connecting   bool
	// notice is a block appended to the conversation, such as the command list.
	notice string

	submittedAt      time.Time
	activeQuestionID string
}

// New builds the conversation. ctx bounds everything it does to the backend, including
// the close on the way out.
func New(ctx context.Context, options Options) (*Model, error) {
	if options.Open == nil {
		return nil, errors.New("tui: Options.Open is required")
	}
	if err := validateCommands(options.Commands); err != nil {
		return nil, err
	}
	branding := options.Branding.withDefaults()
	appearance := newStyles(options.Theme)
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}
	m := &Model{
		options:        options,
		branding:       branding,
		styles:         appearance,
		logger:         logger,
		ctx:            ctx,
		input:          composer(branding, appearance),
		view:           viewport.New(80, 15),
		cache:          map[string]string{},
		renderers:      map[int]*glamour.TermRenderer{},
		conversationID: options.ConversationID,
		status:         "Starting…",
		// A size to draw at until the terminal says what it really is.
		width:  80,
		height: 24,
	}
	m.relayout()
	return m, nil
}

// Run holds the conversation until it is left, and closes whatever session is open.
func Run(ctx context.Context, options Options) error {
	model, err := New(ctx, options)
	if err != nil {
		return err
	}
	_, err = tea.NewProgram(model, tea.WithAltScreen(), tea.WithContext(ctx)).Run()
	if model.session != nil {
		model.close(model.session)
	}
	return err
}

// State is what the conversation currently knows about itself.
func (m *Model) State() State {
	state := State{
		ConversationID: m.conversationID,
		Scope:          m.scope,
		Connecting:     m.connecting,
		Busy:           m.busy,
		Truncated:      m.truncated,
	}
	if m.session != nil {
		state.SessionID = m.session.ID()
	}
	return state
}

func (m *Model) Init() tea.Cmd {
	return tea.Batch(textarea.Blink, beat(), m.open(m.options.ConversationID))
}

func (m *Model) View() string {
	if m.width <= 0 || m.height <= 0 {
		return ""
	}
	// The conversation is wrapped to a readable width even when the terminal is narrower
	// than that, so the last word here is the terminal's.
	frame := strings.Join([]string{m.header(), "", m.view.View(), m.footer()}, "\n")
	return fit(truncateBlock(frame, m.width), m.height)
}

func (m *Model) Update(message tea.Msg) (tea.Model, tea.Cmd) {
	var commands []tea.Cmd
	switch msg := message.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		// Every answer was wrapped to the old width.
		m.cache = map[string]string{}

	case opened:
		if msg.generation != m.generation {
			break
		}
		m.connecting = false
		if msg.err != nil {
			m.fail(msg.err)
			break
		}
		m.session = msg.session
		m.conversationID = msg.session.ConversationID()
		m.truncated = msg.session.ContextTruncated()
		m.messages = msg.page.Messages
		m.before = msg.page.Before
		for _, saved := range m.messages {
			m.rescope(saved)
		}
		m.say("Ready")
		commands = append(commands, m.receive())
		if m.options.OnOpen != nil {
			ctx, session, onOpen := m.ctx, m.session, m.options.OnOpen
			commands = append(commands, func() tea.Msg { return acted{err: onOpen(ctx, session)} })
		}

	case received:
		if msg.generation != m.generation {
			break
		}
		if !msg.ok {
			m.busy = false
			m.fail(fmt.Errorf("the session ended; /resume %s reopens the conversation", m.conversationID))
			break
		}
		if updated, ok := msg.event.ConversationMessage(); ok {
			m.upsert(updated)
			m.rescope(updated)
			// An answer to an earlier question must not take over the status line from the
			// one being waited on.
			if updated.Role == "assistant" && (m.activeQuestionID == "" || updated.QuestionID == m.activeQuestionID) {
				m.busy = updated.FinishedAt == nil && !settled(updated.State)
				m.status, m.statusFailed = label(updated.State), updated.State == "failed"
			}
		}
		if msg.event.Kind == "error" {
			m.busy = false
			m.fail(errors.New(msg.event.Error))
		}
		if m.session != nil {
			commands = append(commands, m.receive())
		}

	case older:
		switch {
		case msg.err != nil:
			m.fail(msg.err)
		case len(msg.page.Messages) == 0:
			m.before = ""
			m.say("There is nothing before this")
		default:
			m.messages = append(msg.page.Messages, m.messages...)
			m.before = msg.page.Before
			m.say(fmt.Sprintf("Loaded %d earlier messages", len(msg.page.Messages)))
		}

	case acted:
		switch {
		case msg.err != nil:
			m.busy = false
			m.fail(msg.err)
		case msg.status != "":
			m.say(msg.status)
		}

	case tick:
		m.frame++
		commands = append(commands, beat())

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			return m, tea.Quit
		case "esc":
			if m.session != nil && m.busy {
				m.say("Cancelling…")
				session := m.session
				commands = append(commands, func() tea.Msg { return acted{err: session.Interrupt()} })
			}
		case "pgup", "pgdown", "ctrl+up", "ctrl+down":
			var cmd tea.Cmd
			m.view, cmd = m.view.Update(msg)
			commands = append(commands, cmd)
		case "alt+enter":
			m.input.InsertString("\n")
		case "enter":
			commands = append(commands, m.submit()...)
		default:
			var cmd tea.Cmd
			m.input, cmd = m.input.Update(msg)
			commands = append(commands, cmd)
		}

	default:
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(message)
		commands = append(commands, cmd)
	}
	m.refresh()
	return m, tea.Batch(commands...)
}

// refresh redraws the conversation and resizes it to the chrome that now surrounds it,
// without dragging somebody reading further up back to the bottom.
func (m *Model) refresh() {
	atBottom := m.view.AtBottom()
	m.relayout()
	m.view.SetContent(m.transcript())
	if atBottom {
		m.view.GotoBottom()
	}
}

// submit sends what is in the composer, which is either a question or a command.
func (m *Model) submit() []tea.Cmd {
	question := strings.TrimSpace(m.input.Value())
	if question == "" {
		return nil
	}
	if name, args, ok := parseCommand(question); ok {
		m.input.Reset()
		return m.run(name, args)
	}
	if m.busy || m.connecting || m.session == nil {
		return nil
	}
	m.input.Reset()
	m.notice = ""
	m.busy = true
	m.say("Thinking…")
	asked := time.Now()
	m.submittedAt = asked
	m.activeQuestionID = pendingID
	m.upsert(stream.ConversationMessage{ID: pendingID, Role: "user", Text: question, StartedAt: asked})
	m.view.GotoBottom()
	session := m.session
	return []tea.Cmd{func() tea.Msg { return acted{err: session.Respond(question)} }}
}

// run does a slash command.
func (m *Model) run(name string, args []string) []tea.Cmd {
	switch name {
	case "quit":
		return []tea.Cmd{tea.Quit}
	case "help":
		m.notice = commandHelp(m.styles, m.options.Commands, max(minContent, m.view.Width-2))
		m.view.GotoBottom()
		return nil
	case "new", "resume":
		if m.busy || m.connecting {
			m.say("Finish or cancel this answer first")
			return nil
		}
		if name == "new" {
			return []tea.Cmd{m.open("")}
		}
		if len(args) != 1 {
			m.say("Usage: /resume <id>")
			return nil
		}
		return []tea.Cmd{m.open(args[0])}
	case "older":
		switch {
		case m.options.History == nil:
			m.say("Saved history is not available in this conversation")
		case m.before == "":
			m.say("There is nothing before this")
		default:
			ctx, history, id, before := m.ctx, m.options.History, m.conversationID, m.before
			return []tea.Cmd{func() tea.Msg {
				page, err := history(ctx, id, before)
				return older{page: page, err: err}
			}}
		}
		return nil
	}
	for _, command := range m.options.Commands {
		if command.Name != name {
			continue
		}
		ctx, run := m.ctx, command.Run
		return []tea.Cmd{func() tea.Msg {
			status, err := run(ctx, args)
			return acted{status: status, err: err}
		}}
	}
	m.say("There is no /" + name + "; /help lists the commands")
	return nil
}

// open opens a conversation, closing whatever was open before it. An empty id starts a
// new conversation.
func (m *Model) open(conversationID string) tea.Cmd {
	m.generation++
	m.connecting = true
	m.busy = false
	m.conversationID = conversationID
	m.scope = ""
	m.notice = ""
	m.messages = nil
	m.before = ""
	m.truncated = false
	m.submittedAt = time.Time{}
	m.activeQuestionID = ""
	m.say("Opening the conversation…")
	generation, previous := m.generation, m.session
	m.session = nil
	ctx, open, history := m.ctx, m.options.Open, m.options.History
	return func() tea.Msg {
		if previous != nil {
			m.close(previous)
		}
		session, err := open(ctx, conversationID)
		if err != nil {
			return opened{generation: generation, err: err}
		}
		var page stream.ConversationPage
		if history != nil {
			if page, err = history(ctx, session.ConversationID(), ""); err != nil {
				m.close(session)
				return opened{generation: generation, err: err}
			}
		}
		return opened{generation: generation, session: session, page: page}
	}
}

// receive waits for the next thing the open session has to say.
func (m *Model) receive() tea.Cmd {
	generation, events := m.generation, m.session.Events()
	return func() tea.Msg {
		event, ok := <-events
		return received{generation: generation, event: event, ok: ok}
	}
}

// close gives a session a moment to end cleanly, whatever became of the context the
// conversation was held under.
func (m *Model) close(session Session) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(m.ctx), closeTimeout)
	defer cancel()
	if err := session.Close(ctx); err != nil {
		m.logger.Warn("closing the session failed", "error", err)
	}
}

// upsert files an updated message under the one it replaces, so that a message arriving
// repeatedly as it is written does not arrive repeatedly on screen.
func (m *Model) upsert(message stream.ConversationMessage) {
	for i := range m.messages {
		if m.messages[i].ID == message.ID {
			m.messages[i] = message
			return
		}
	}
	// The question shown the moment it was asked is the one the backend has now saved
	// under an id of its own.
	if last := len(m.messages) - 1; last >= 0 && m.messages[last].ID == pendingID && message.Role == "user" {
		m.messages[last] = message
		m.activeQuestionID = message.ID
		return
	}
	m.messages = append(m.messages, message)
}

// rescope remembers the product and SDK the agent is working in, which is the most
// useful thing the header can say about a conversation in progress.
func (m *Model) rescope(message stream.ConversationMessage) {
	for _, tool := range message.Tools {
		if tool.Product == "" {
			continue
		}
		m.scope = tool.Product
		if tool.SDK != "" {
			m.scope += " / " + tool.SDK
		}
	}
}

func (m *Model) say(status string) { m.status, m.statusFailed = status, false }
func (m *Model) fail(err error)    { m.status, m.statusFailed = err.Error(), true }
func beat() tea.Cmd                { return tea.Tick(pulse, func(t time.Time) tea.Msg { return tick(t) }) }

// tick moves the timers and the spinner.
type tick time.Time

// opened is a session that has been opened, with the history it starts from.
type opened struct {
	generation int
	session    Session
	page       stream.ConversationPage
	err        error
}

// received is one thing a session said, or its closing.
type received struct {
	generation int
	event      stream.Event
	ok         bool
}

// older is a page of messages from before the conversation on screen.
type older struct {
	page stream.ConversationPage
	err  error
}

// acted is the outcome of something done off the event loop.
type acted struct {
	status string
	err    error
}
