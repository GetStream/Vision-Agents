// Package conversation persists text conversations and their visible activity in Stream Chat.
package conversation

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"
	"golang.org/x/sys/unix"
)

type Tool struct {
	Type               string     `json:"type"`
	Product            string     `json:"product,omitempty"`
	SDK                string     `json:"sdk,omitempty"`
	ID                 string     `json:"tool_call_id"`
	Name               string     `json:"name"`
	Title              string     `json:"title"`
	Status             string     `json:"status"`
	Phase              string     `json:"phase"`
	Summary            string     `json:"summary,omitempty"`
	StartedAt          time.Time  `json:"started_at"`
	ExecutionStartedAt *time.Time `json:"execution_started_at,omitempty"`
	FinishedAt         *time.Time `json:"finished_at,omitempty"`
	DurationMS         int64      `json:"duration_ms"`
}
type Source struct {
	ID       string `json:"id"`
	Title    string `json:"title"`
	URL      string `json:"url"`
	Citation string `json:"citation,omitempty"`
}
type Message struct {
	CommandID      string               `json:"command_id,omitempty"`
	TurnID         string               `json:"turn_id,omitempty"`
	ID             string               `json:"id"`
	QuestionID     string               `json:"question_id,omitempty"`
	Role           string               `json:"role"`
	Text           string               `json:"text"`
	State          string               `json:"state"`
	StartedAt      time.Time            `json:"response_started_at"`
	StateStartedAt time.Time            `json:"state_started_at"`
	FinishedAt     *time.Time           `json:"finished_at,omitempty"`
	DurationMS     int64                `json:"duration_ms"`
	Sequence       int                  `json:"sequence"`
	Tools          []Tool               `json:"attachments"`
	Sources        []Source             `json:"sources,omitempty"`
	Artifacts      []ArtifactAttachment `json:"artifacts,omitempty"`
	Saved          bool                 `json:"saved"`
	Error          string               `json:"persistence_error,omitempty"`
}
type Page struct {
	memoryScope memory.Scope
	empty       bool
	Messages    []Message `json:"messages"`
	Before      string    `json:"before,omitempty"`
	Truncated   bool      `json:"context_truncated"`
}
type Updated struct {
	CID     string  `json:"conversation_id"`
	Message Message `json:"message"`
}

// CommandReceipt identifies one durable submission and its two Chat messages.
type CommandReceipt struct {
	CommandID          string `json:"command_id"`
	UserMessageID      string `json:"user_message_id"`
	AssistantMessageID string `json:"assistant_message_id"`
	State              string `json:"state"`
	Duplicate          bool   `json:"duplicate"`
}
type commandRecord struct {
	CommandReceipt
	Digest string
}

var ErrCommandNotFound = errors.New("command not found")

var ErrCommandConflict = errors.New("command ID was already used with different content")
var validCommandID = regexp.MustCompile(`^[A-Za-z0-9_-]{1,128}$`)

type disk struct {
	OutboxVersion int
	CommandLedger bool
	Pending       []operation
	Commands      map[string]commandRecord
	CID           string
	Customer      string
	Agent         string
	Owner         string
	Current       *Message
}
type operation struct {
	Message Message
	Create  bool
}
type Service struct {
	lock   *os.File
	closed bool
	mu     sync.Mutex
	client *getstream.Stream
	root   string
	all    map[string]*Conversation
}
type Conversation struct {
	mu       sync.Mutex
	service  *Service
	data     disk
	active   bool
	stopping bool
	turns    map[string]string
	created  map[string]bool
	emit     func(Updated)
	dirty    bool
	stopped  chan struct{}
	done     chan struct{}
}

var validID = regexp.MustCompile(`^support-[a-f0-9-]{36}$`)

// SessionCommandChannel reserves the persistent conversation namespace for the
// session command path. Webhook delivery cannot opt it into a second trigger path.
func SessionCommandChannel(channelType, id string) bool {
	return channelType == "agent" && validID.MatchString(id)
}

const TriggerField = "support_trigger"
const SessionCommandTrigger = "session_commands"

func New(root string) (*Service, error) {
	client, err := getstream.NewClient(os.Getenv("STREAM_API_KEY"), os.Getenv("STREAM_API_SECRET"))
	if err != nil {
		return nil, err
	}
	if os.Getenv("STREAM_API_KEY") == "" || os.Getenv("STREAM_API_SECRET") == "" {
		return nil, errors.New("Stream Chat credentials are required for persistent conversations")
	}
	return newService(root, client)
}

// NewForChat is New for a caller that already holds a Chat client rather than
// reading one out of the environment.
func NewForChat(root string, client *getstream.Stream) (*Service, error) {
	return newService(root, client)
}
func newService(root string, client *getstream.Stream) (*Service, error) {
	var err error
	if root == "" {
		root = ".local/chat-outbox"
	}
	if err = os.MkdirAll(root, 0700); err != nil {
		return nil, err
	}
	lock, err := os.OpenFile(filepath.Join(root, ".lock"), os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return nil, err
	}
	if err = unix.Flock(int(lock.Fd()), unix.LOCK_EX|unix.LOCK_NB); err != nil {
		lock.Close()
		return nil, errors.New("conversation outbox is already owned by another service")
	}
	s := &Service{client: client, root: root, lock: lock, all: map[string]*Conversation{}}
	ready := false
	defer func() {
		if !ready {
			s.Close()
		}
	}()
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	for _, e := range entries {
		if !e.IsDir() || !validID.MatchString(e.Name()) {
			continue
		}
		d, err := loadDisk(filepath.Join(root, e.Name()))
		if err != nil {
			return nil, err
		}
		c := s.make(d)
		c.mu.Lock()
		c.finish("interrupted")
		c.mu.Unlock()
	}
	ready = true
	return s, nil
}
func (s *Service) make(d disk) *Conversation {
	c := &Conversation{service: s, data: d, turns: map[string]string{}, created: map[string]bool{}, stopped: make(chan struct{}), done: make(chan struct{})}
	s.all[d.CID] = c
	go c.run()
	return c
}
func (s *Service) Close() {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return
	}
	s.closed = true
	all := make([]*Conversation, 0, len(s.all))
	for _, c := range s.all {
		all = append(all, c)
	}
	s.mu.Unlock()
	for _, c := range all {
		c.mu.Lock()
		c.active = false
		c.stopping = true
		c.mu.Unlock()
		close(c.stopped)
	}
	for _, c := range all {
		<-c.done
	}
	_ = unix.Flock(int(s.lock.Fd()), unix.LOCK_UN)
	_ = s.lock.Close()
}
func (s *Service) Open(ctx context.Context, customer, agentID, cid string, scopes ...memory.Scope) (*Conversation, []llm.Message, bool, error) {
	return s.OpenForCaller(ctx, customer, agentID, cid, "", scopes...)
}

// OpenForCaller binds persistent messages to a verified end-user identity. Empty
// caller preserves backend-owned demo channels; it cannot open a private channel.
func (s *Service) OpenForCaller(ctx context.Context, customer, agentID, cid, caller string, scopes ...memory.Scope) (*Conversation, []llm.Message, bool, error) {
	ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, nil, false, errors.New("conversation service is closed")
	}
	var scope memory.Scope
	if len(scopes) > 0 {
		scope = scopes[0]
	}
	fresh := cid == ""
	if fresh {
		cid = "agent:support-" + uuid.NewString()
	}
	id := strings.TrimPrefix(cid, "agent:")
	if cid != "agent:"+id || !validID.MatchString(id) {
		return nil, nil, false, errors.New("invalid conversation channel")
	}
	if c := s.all[cid]; c != nil {
		c.mu.Lock()
		defer c.mu.Unlock()
		if c.data.Customer != customer || c.data.Agent != agentID || c.data.Owner != caller {
			return nil, nil, false, errors.New("conversation belongs to another customer or agent")
		}
		if c.active {
			return nil, nil, false, errors.New("conversation is already open")
		}
		page, err := s.history(ctx, customer, agentID, cid, "", caller)
		if err != nil {
			return nil, nil, false, err
		}
		if !sameMemoryScope(page.memoryScope, scope) {
			return nil, nil, false, errors.New("conversation belongs to another memory scope; reopen with its original organization")
		}
		c.active = true
		h, tr := history(page)
		return c, h, tr, nil
	}
	if fresh {
		userID := caller
		if userID == "" {
			userID = "support-operator"
		}
		_, err := s.client.UpdateUsers(ctx, &getstream.UpdateUsersRequest{Users: map[string]getstream.UserRequest{agentID: {ID: agentID}, userID: {ID: userID}}})
		if err != nil {
			return nil, nil, false, err
		}
		_, err = s.client.Chat().GetOrCreateChannel(ctx, "agent", id, &getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{CreatedByID: &agentID, Members: []getstream.ChannelMemberRequest{{UserID: agentID}, {UserID: userID}}, Custom: map[string]any{"support_customer_id": customer, "support_agent_id": agentID, "support_memory_scope": scope, "support_owner_id": caller, TriggerField: SessionCommandTrigger}}})
		if err != nil {
			return nil, nil, false, err
		}
	}
	page, err := s.history(ctx, customer, agentID, cid, "", caller)
	if err != nil {
		return nil, nil, false, err
	}
	if !sameMemoryScope(page.memoryScope, scope) {
		return nil, nil, false, errors.New("conversation belongs to another memory scope; reopen with its original organization")
	}
	initializeLedger := fresh || caller != "" && page.empty
	c := s.make(disk{CID: cid, Customer: customer, Agent: agentID, Owner: caller, CommandLedger: initializeLedger})
	c.active = true
	if initializeLedger {
		c.mu.Lock()
		err := c.persist()
		c.mu.Unlock()
		if err != nil {
			c.Release()
			return nil, nil, false, err
		}
	}
	h, tr := history(page)
	return c, h, tr, nil
}
func (s *Service) History(ctx context.Context, customer, agentID, cid, before string) (Page, error) {
	return s.HistoryForCaller(ctx, customer, agentID, cid, before, "")
}
func (s *Service) HistoryForCaller(ctx context.Context, customer, agentID, cid, before, caller string) (Page, error) {
	return s.history(ctx, customer, agentID, cid, before, caller)
}

// ContextForCaller returns completed user and assistant turns for a voice
// session that must not open a persistent text conversation. Empty cid is no
// history rather than an error.
func (s *Service) ContextForCaller(ctx context.Context, customer, agentID, cid, caller string) ([]llm.Message, bool, error) {
	if cid == "" {
		return nil, false, nil
	}
	page, err := s.HistoryForCaller(ctx, customer, agentID, cid, "", caller)
	if err != nil {
		return nil, false, err
	}
	messages, truncated := history(page)
	return messages, truncated, nil
}

// ownedBy reads a channel's recorded ownership. A channel with no owner is a
// backend-owned demo channel, which no end user may claim.
func ownedBy(custom map[string]any, customer, agentID, caller string) error {
	if custom["support_customer_id"] != customer || custom["support_agent_id"] != agentID {
		return errors.New("conversation belongs to another customer or agent")
	}
	rawOwner, bound := custom["support_owner_id"]
	owner, valid := rawOwner.(string)
	if bound && !valid {
		return errors.New("conversation has invalid ownership metadata")
	}
	if owner != caller {
		return errors.New("conversation belongs to another user")
	}
	return nil
}

// CommandForCaller reports what one command ended as without opening its conversation,
// starting a session or running anything. It is how a stop whose conversation has no
// session left to reach reconciles the same command id instead of reopening one to ask.
func (s *Service) CommandForCaller(ctx context.Context, customer, agentID, cid, caller, commandID string) (CommandReceipt, error) {
	id := strings.TrimPrefix(cid, "agent:")
	if cid != "agent:"+id || !validID.MatchString(id) || !validCommandID.MatchString(commandID) {
		return CommandReceipt{}, ErrCommandNotFound
	}
	s.mu.Lock()
	closed, open, root := s.closed, s.all[cid], s.root
	s.mu.Unlock()
	if closed {
		return CommandReceipt{}, errors.New("conversation service is closed")
	}

	lookup, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	limit, state := 1, true
	// Query without Data, so asking about a command can neither create a channel nor
	// overwrite the ownership recorded on one.
	r, err := s.client.Chat().GetOrCreateChannel(lookup, "agent", id, &getstream.GetOrCreateChannelRequest{
		State: &state, Messages: &getstream.MessagePaginationParams{Limit: &limit}})
	if err != nil {
		return CommandReceipt{}, err
	}
	if err := ownedBy(r.Data.Channel.Custom, customer, agentID, caller); err != nil {
		return CommandReceipt{}, ErrCommandNotFound
	}
	if caller != "" {
		if r.Data.Channel.Custom[TriggerField] != SessionCommandTrigger {
			return CommandReceipt{}, ErrCommandNotFound
		}
		member := false
		for _, candidate := range r.Data.Members {
			if candidate.UserID != nil && *candidate.UserID == caller {
				member = true
				break
			}
		}
		if !member {
			return CommandReceipt{}, ErrCommandNotFound
		}
	}

	if open != nil {
		return open.receipt(commandID)
	}
	var stored disk
	raw, err := os.ReadFile(filepath.Join(root, id, "state.json"))
	if err != nil {
		return CommandReceipt{}, ErrCommandNotFound
	}
	if err := json.Unmarshal(raw, &stored); err != nil {
		return CommandReceipt{}, errors.New("invalid conversation outbox")
	}
	record, known := stored.Commands[commandID]
	if !stored.CommandLedger || !known {
		return CommandReceipt{}, ErrCommandNotFound
	}
	return record.CommandReceipt, nil
}

func (s *Service) history(ctx context.Context, customer, agentID, cid, before, caller string) (Page, error) {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	id := strings.TrimPrefix(cid, "agent:")
	if cid != "agent:"+id || !validID.MatchString(id) {
		return Page{}, errors.New("invalid conversation channel")
	}
	limit := 100
	state := true
	params := &getstream.MessagePaginationParams{Limit: &limit}
	if before != "" {
		params.IDLt = &before
	}
	// Query without Data: a resume must never create or overwrite a channel's ownership.
	r, err := s.client.Chat().GetOrCreateChannel(ctx, "agent", id, &getstream.GetOrCreateChannelRequest{State: &state, Messages: params})
	if err != nil {
		return Page{}, err
	}
	if err := ownedBy(r.Data.Channel.Custom, customer, agentID, caller); err != nil {
		return Page{}, err
	}
	if caller != "" {
		if r.Data.Channel.Custom[TriggerField] != SessionCommandTrigger {
			return Page{}, errors.New("conversation is not a session-command channel")
		}
		member := false
		for _, candidate := range r.Data.Members {
			if candidate.UserID != nil && *candidate.UserID == caller {
				member = true
				break
			}
		}
		if !member {
			return Page{}, errors.New("conversation caller is not a channel member")
		}
	}
	p := Page{empty: len(r.Data.Messages) == 0, Messages: []Message{}, Truncated: len(r.Data.Messages) == limit}
	if raw, ok := r.Data.Channel.Custom["support_memory_scope"]; ok {
		b, err := json.Marshal(raw)
		if err != nil {
			return Page{}, err
		}
		if err := json.Unmarshal(b, &p.memoryScope); err != nil {
			return Page{}, err
		}
	}
	for _, m := range r.Data.Messages {
		if m.DeletedAt != nil {
			continue
		}
		if msg, err := messageFromWire(m.ID, m.Text, m.Custom); err == nil {
			p.Messages = append(p.Messages, msg)
			continue
		}
		raw, ok := m.Custom["support_message"]
		if !ok {
			continue
		}
		b, _ := json.Marshal(raw)
		var msg Message
		if json.Unmarshal(b, &msg) == nil {
			msg.Saved = true
			p.Messages = append(p.Messages, msg)
		}
	}
	if before == "" {
		// Overlay durable pending snapshots so a reconnect sees unfinished retries truthfully.
		var pending disk
		raw, err := os.ReadFile(filepath.Join(s.root, id, "state.json"))
		if err != nil && !os.IsNotExist(err) {
			return Page{}, err
		}
		if err == nil && json.Unmarshal(raw, &pending) != nil {
			return Page{}, errors.New("invalid conversation outbox")
		}
		for _, op := range pending.Pending {
			op.Message.Saved = false
			op.Message.Error = "Pending Stream Chat save"
			found := false
			for i := range p.Messages {
				if p.Messages[i].ID == op.Message.ID {
					p.Messages[i] = op.Message
					found = true
					break
				}
			}
			if !found {
				p.Messages = append(p.Messages, op.Message)
			}
		}
	}
	sort.SliceStable(p.Messages, func(i, j int) bool { return p.Messages[i].StartedAt.Before(p.Messages[j].StartedAt) })
	if len(p.Messages) > limit {
		p.Messages = p.Messages[len(p.Messages)-limit:]
		p.Truncated = true
	}
	if p.Truncated && len(p.Messages) > 0 {
		p.Before = p.Messages[0].ID
	}
	return p, nil
}
func history(p Page) ([]llm.Message, bool) {
	var out []llm.Message
	size := 0
	tr := p.Truncated
	for i := len(p.Messages) - 1; i >= 0; i-- {
		m := p.Messages[i]
		if m.State != "completed" || m.Text == "" {
			continue
		}
		if m.Role != "user" && m.Role != "assistant" {
			continue
		}
		if size+utf8.RuneCountInString(m.Text) > 60000 || len(out) == 100 {
			tr = true
			break
		}
		size += utf8.RuneCountInString(m.Text)
		role := llm.User
		if m.Role == "assistant" {
			role = llm.Assistant
		}
		out = append(out, llm.Message{Role: role, Content: m.Text})
	}
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out, tr
}
func (c *Conversation) CID() string               { return c.data.CID }
func (c *Conversation) Attach(emit func(Updated)) { c.mu.Lock(); defer c.mu.Unlock(); c.emit = emit }
func (c *Conversation) Release() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.finish("cancelled")
	c.active = false
	c.emit = nil
}
func (c *Conversation) Begin(text string) error {
	_, err := c.beginCommand(uuid.NewString(), text, true)
	return err
}

// Command returns the known receipt without accepting or executing a submission.
// The caller must first authorize access to this conversation. Unknown commands
// are indistinguishable from an unavailable ledger; no new command is created.
func (c *Conversation) Command(id string) (CommandReceipt, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.active || !c.data.CommandLedger || !validCommandID.MatchString(id) {
		return CommandReceipt{}, ErrCommandNotFound
	}
	record, ok := c.data.Commands[id]
	if !ok {
		return CommandReceipt{}, ErrCommandNotFound
	}
	return record.CommandReceipt, nil
}

// BeginCommand atomically records command ownership and both initial Chat writes.
// A recorded command is never automatically executed again, including after a crash.
func (c *Conversation) BeginCommand(id, text string) (CommandReceipt, error) {
	return c.beginCommand(id, text, false)
}
func (c *Conversation) beginCommand(id, text string, legacy bool) (CommandReceipt, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.active {
		return CommandReceipt{}, errors.New("conversation is not open")
	}
	if !legacy && !c.data.CommandLedger {
		return CommandReceipt{}, errors.New("conversation command ledger is unavailable; restore its durable state")
	}
	if !validCommandID.MatchString(id) || text == "" || len(text) > 1024*1024 {
		return CommandReceipt{}, errors.New("invalid command ID or text")
	}
	digest := fmt.Sprintf("%x", sha256.Sum256([]byte(text)))
	if previous, ok := c.data.Commands[id]; ok {
		if previous.Digest != digest {
			return CommandReceipt{}, ErrCommandConflict
		}
		receipt := previous.CommandReceipt
		receipt.Duplicate = true
		return receipt, nil
	}
	if c.data.Current != nil && c.data.Current.FinishedAt == nil {
		return CommandReceipt{}, errors.New("a response is already running")
	}
	now := time.Now().UTC()
	u := Message{ID: uuid.NewString(), CommandID: id, Role: "user", Text: text, State: "completed", StartedAt: now, StateStartedAt: now, FinishedAt: &now, Tools: []Tool{}}
	a := Message{ID: uuid.NewString(), CommandID: id, Role: "assistant", QuestionID: u.ID, State: "thinking", StartedAt: now, StateStartedAt: now, Tools: []Tool{}}
	receipt := CommandReceipt{CommandID: id, UserMessageID: u.ID, AssistantMessageID: a.ID, State: a.State}
	if c.data.Commands == nil {
		c.data.Commands = map[string]commandRecord{}
	}
	c.data.Commands[id] = commandRecord{CommandReceipt: receipt, Digest: digest}
	c.data.Current = &a
	c.data.Pending = append(c.data.Pending, operation{u, true}, operation{a, true})
	if err := c.persist(); err != nil {
		// A rename/fsync failure has an uncertain durable outcome. Retain the IDs,
		// fail the command and never grant another inference attempt for this ID.
		c.finish("failed")
		return CommandReceipt{}, err
	}
	c.publish(u)
	c.publish(a)
	return receipt, nil
}

// receipt reads a recorded command whether or not the conversation is still open, which
// is what reconciling a command whose session has ended needs.
func (c *Conversation) receipt(id string) (CommandReceipt, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	record, known := c.data.Commands[id]
	if !c.data.CommandLedger || !known {
		return CommandReceipt{}, ErrCommandNotFound
	}
	return record.CommandReceipt, nil
}

// BindTurn records which model turn answers a command before that turn's first event
// is observed. A turn abandoned with its command then stays owned by the message it was
// started for, so its late output cannot be appended to whatever command runs next.
func (c *Conversation) BindTurn(commandID, turnID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	m := c.data.Current
	if m == nil || turnID == "" || commandID == "" || m.CommandID != commandID {
		return
	}
	if _, bound := c.turns[turnID]; !bound {
		c.turns[turnID] = m.ID
	}
	m.TurnID = turnID
}

func (c *Conversation) CommandForTurn(turnID string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	messageID, bound := c.turns[turnID]
	if !bound {
		return "", false
	}
	for commandID, record := range c.data.Commands {
		if record.AssistantMessageID == messageID {
			return commandID, true
		}
	}
	return "", false
}

// CancelCommand changes only the named active command. The session must serialize
// execution interruption with command submission; this method only owns the ledger.
func (c *Conversation) CancelCommand(id string) (CommandReceipt, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.active || !c.data.CommandLedger {
		return CommandReceipt{}, ErrCommandNotFound
	}
	record, ok := c.data.Commands[id]
	if !ok {
		return CommandReceipt{}, ErrCommandNotFound
	}
	m := c.data.Current
	if m != nil && m.CommandID == id {
		if m.FinishedAt == nil {
			c.finish("cancelled")
		} else if m.Error != "" {
			c.save()
		}
		if m.Error != "" {
			return CommandReceipt{}, errors.New("command cancellation persistence outcome unknown")
		}
		return c.data.Commands[id].CommandReceipt, nil
	}
	switch record.State {
	case "completed", "cancelled", "interrupted", "failed":
		return record.CommandReceipt, nil
	default:
		return CommandReceipt{}, ErrCommandNotFound
	}
}

func (c *Conversation) Cancel() { c.mu.Lock(); defer c.mu.Unlock(); c.finish("cancelled") }
func (c *Conversation) Observe(event agent.Event) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.stopping {
		return
	}
	m := c.data.Current
	if m == nil || m.FinishedAt != nil {
		return
	}
	persist := false
	switch e := event.(type) {
	case agent.Responding:
		if !c.acceptTurn(e.TurnID, true) {
			return
		}
		c.state("thinking")
		if m.Text != "" && !strings.HasSuffix(m.Text, "\n\n") {
			m.Text += "\n\n"
		}
	case agent.ResponseDelta:
		if !c.acceptTurn(e.TurnID, false) {
			return
		}
		c.state("writing")
		m.Text += e.Text
		m.Saved = false
	case agent.Responded:
		if !c.acceptTurn(e.TurnID, false) {
			return
		}
		if !e.PendingWork {
			c.finish("completed")
			return
		}
	case agent.ToolStarted:
		if !c.acceptTurn(e.TurnID, true) {
			return
		}
		for _, tool := range m.Tools {
			if tool.ID == e.ID {
				return
			}
		}
		m.Tools = append(m.Tools, Tool{Type: "tool_calling", Product: e.Product, SDK: e.SDK, ID: e.ID, Name: e.Tool, Title: title(e.Tool), Status: "running", Phase: "running", StartedAt: e.StartedAt})
		c.state("tools")
		persist = true
	case agent.ToolRan:
		if !c.acceptTurn(e.TurnID, false) {
			return
		}
		changed := false
		for i := range m.Tools {
			t := &m.Tools[i]
			if t.ID != e.ID || t.FinishedAt != nil {
				continue
			}
			now := time.Now().UTC()
			t.FinishedAt = &now
			t.DurationMS = now.Sub(t.StartedAt).Milliseconds()
			t.Status = "completed"
			t.Summary = "Finished"
			if e.Err != nil {
				t.Status = "failed"
				t.Summary = "Tool failed"
			}
			var r struct {
				Status    string `json:"status"`
				Code      string `json:"code"`
				Citations []any  `json:"citations"`
			}
			if json.Unmarshal([]byte(e.Result), &r) == nil {
				switch r.Status {
				case "answered":
					t.Summary = fmt.Sprintf("Verified %d source citations", len(r.Citations))
				case "unavailable", "research_failed", "insufficient_evidence":
					t.Status = "failed"
					t.Summary = strings.TrimSpace(strings.ReplaceAll(r.Status+" "+r.Code, "_", " "))
				}
			}
			t.Phase = t.Status
			if e.Err == nil {
				m.Sources = mergeSources(m.Sources, sourcesOf(e.Tool, e.Result))
				m.Artifacts = mergeArtifacts(m.Artifacts, artifactsOf(e.Result))
			}
			changed = true
		}
		if !changed {
			return
		}
		c.afterTools()
		persist = true
	case agent.Delegated:
		if !c.acceptTurn(e.TurnID, true) {
			return
		}
		for _, tool := range m.Tools {
			if tool.ID == e.TaskID {
				return
			}
		}
		startedAt := e.StartedAt
		if startedAt.IsZero() {
			startedAt = time.Now().UTC()
		}
		m.Tools = append(m.Tools, Tool{Type: "tool_calling", ID: e.TaskID, Name: e.Skill, Title: title(e.Skill), Status: "running", Phase: "running", StartedAt: startedAt})
		c.state("tools")
		persist = true
	case agent.TaskSettled:
		changed := false
		for i := range m.Tools {
			t := &m.Tools[i]
			if t.ID == e.TaskID && t.FinishedAt == nil {
				now := time.Now().UTC()
				t.FinishedAt = &now
				t.DurationMS = now.Sub(t.StartedAt).Milliseconds()
				t.Status = "completed"
				t.Summary = "Skill finished"
				if e.Err != nil {
					t.Status = "failed"
					t.Summary = "Skill failed"
				}
				t.Phase = t.Status
				changed = true
			}
		}
		if !changed {
			return
		}
		c.afterTools()
		persist = true
	case agent.TaskCancelled:
		changed := false
		for i := range m.Tools {
			t := &m.Tools[i]
			if t.ID == e.TaskID && t.FinishedAt == nil {
				now := time.Now().UTC()
				t.FinishedAt = &now
				t.DurationMS = now.Sub(t.StartedAt).Milliseconds()
				t.Status = "cancelled"
				t.Phase = "cancelled"
				t.Summary = "Skill cancelled"
				changed = true
			}
		}
		if !changed {
			return
		}
		c.afterTools()
		persist = true
	case agent.Error:
		c.finish("failed")
		return
	case agent.Interrupted:
		if !c.acceptTurn(e.TurnID, false) {
			return
		}
		c.finish("cancelled")
		return
	default:
		return
	}
	m.Sequence++
	c.dirty = true
	if persist {
		c.save()
	}
	c.publish(*m)
}

// acceptTurn retains the original response binding across internal model/tool rounds.
// In particular, starting the next question may interrupt the previous completed turn.
func (c *Conversation) acceptTurn(id string, start bool) bool {
	if id == "" {
		return true
	}
	if owner, exists := c.turns[id]; exists {
		if owner != c.data.Current.ID {
			return false
		}
		c.data.Current.TurnID = id
		return true
	}
	if !start {
		return false
	}
	c.turns[id] = c.data.Current.ID
	c.data.Current.TurnID = id
	return true
}
func (c *Conversation) Progress(id, phase string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.stopping {
		return
	}
	m := c.data.Current
	if m == nil || m.FinishedAt != nil {
		return
	}
	changed := false
	for i := range m.Tools {
		t := &m.Tools[i]
		if t.ID != id || t.FinishedAt != nil {
			continue
		}
		if t.Phase == phase {
			return
		}
		t.Phase = phase
		if phase == "queued" {
			t.Status = "queued"
			c.state("queued")
		} else {
			t.Status = "running"
			c.state("tools")
			if t.ExecutionStartedAt == nil {
				now := time.Now().UTC()
				t.ExecutionStartedAt = &now
				c.save()
			}
		}
		changed = true
		break
	}
	if !changed {
		return
	}
	m.Sequence++
	c.dirty = true
	c.publish(*m)
}
func (c *Conversation) afterTools() {
	for _, t := range c.data.Current.Tools {
		if t.FinishedAt == nil {
			if t.Status == "queued" {
				c.state("queued")
			} else {
				c.state("tools")
			}
			return
		}
	}
	c.state("thinking")
}
func title(name string) string {
	switch name {
	case "investigate_sdk":
		return "Investigate SDK source"
	case "search_docs":
		return "Search documentation"
	}
	return strings.ReplaceAll(name, "-", " ")
}
func (c *Conversation) state(state string) {
	m := c.data.Current
	if m.State != state {
		m.Saved = false
		m.State = state
		if record, ok := c.data.Commands[m.CommandID]; ok {
			record.State = state
			c.data.Commands[m.CommandID] = record
		}
		m.StateStartedAt = time.Now().UTC()
	}
}
func (c *Conversation) finish(state string) {
	if c.stopping {
		return
	}
	m := c.data.Current
	if m == nil || m.FinishedAt != nil {
		return
	}
	now := time.Now().UTC()
	c.state(state)
	m.FinishedAt = &now
	m.DurationMS = now.Sub(m.StartedAt).Milliseconds()
	for i := range m.Tools {
		t := &m.Tools[i]
		if t.FinishedAt == nil {
			t.FinishedAt = &now
			t.DurationMS = now.Sub(t.StartedAt).Milliseconds()
			t.Status = "cancelled"
			t.Phase = "cancelled"
			t.Summary = "Interrupted"
		}
	}
	m.Sequence++
	c.save()
	c.publish(*m)
}
func (c *Conversation) publish(m Message) {
	if c.emit != nil {
		m.Tools = append([]Tool{}, m.Tools...)
		m.Sources = append([]Source{}, m.Sources...)
		c.emit(Updated{CID: c.data.CID, Message: m})
	}
}
func (c *Conversation) dir() string {
	return filepath.Join(c.service.root, strings.TrimPrefix(c.data.CID, "agent:"))
}
func writeJSON(path string, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path+".tmp", os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err = f.Write(b); err != nil {
		return err
	}
	if err = f.Sync(); err != nil {
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	if err = os.Rename(path+".tmp", path); err != nil {
		return err
	}
	d, err := os.Open(filepath.Dir(path))
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}

// loadDisk imports the earlier per-operation files once. The version marker and
// queue are committed together, so a crash during cleanup cannot replay old files.
func loadDisk(dir string) (disk, error) {
	var d disk
	b, err := os.ReadFile(filepath.Join(dir, "state.json"))
	if err != nil {
		return d, err
	}
	if err = json.Unmarshal(b, &d); err != nil {
		return d, err
	}
	if d.OutboxVersion > 1 {
		return d, errors.New("unsupported conversation outbox version")
	}
	if d.OutboxVersion == 0 {
		files, err := filepath.Glob(filepath.Join(dir, "ops", "*.json"))
		if err != nil {
			return d, err
		}
		sort.Strings(files)
		for _, file := range files {
			b, err := os.ReadFile(file)
			if err != nil {
				return d, err
			}
			var op operation
			if err = json.Unmarshal(b, &op); err != nil {
				return d, err
			}
			d.Pending = append(d.Pending, op)
		}
		d.OutboxVersion = 1
		d.CommandLedger = true
		if err = writeJSON(filepath.Join(dir, "state.json"), d); err != nil {
			return d, err
		}
		for _, file := range files {
			_ = os.Remove(file)
		}
	}
	return d, nil
}

func (c *Conversation) persist() error {
	if err := os.MkdirAll(c.dir(), 0700); err != nil {
		return err
	}
	c.data.OutboxVersion = 1
	return writeJSON(filepath.Join(c.dir(), "state.json"), c.data)
}
func (c *Conversation) enqueue(m Message, create bool) error {
	m.Tools = append([]Tool{}, m.Tools...)
	m.Sources = append([]Source{}, m.Sources...)
	m.Artifacts = append([]ArtifactAttachment{}, m.Artifacts...)
	c.data.Pending = append(c.data.Pending, operation{m, create})
	return c.persist()
}
func (c *Conversation) save() {
	m := c.data.Current
	m.Saved = false
	m.Error = ""
	if err := c.enqueue(*m, false); err != nil {
		m.Error = "Could not save retry record: " + err.Error()
	}
}
func (c *Conversation) send(ctx context.Context, op operation, ephemeral bool) error {
	m := op.Message
	user := c.data.Agent
	if m.Role == "user" {
		user = c.data.Owner
		if user == "" {
			user = "support-operator"
		}
	}
	m.Saved = !ephemeral
	m.Error = ""
	metadata, err := metadataOf(m)
	if err != nil {
		return err
	}
	runtime := runtimeOf(m)
	fields := map[string]any{
		"text": m.Text, "generating": m.FinishedAt == nil, "source": "agent",
		"support_message": metadata, "support_runtime": runtime,
	}
	if attachments := streamAttachments(m.Artifacts); len(attachments) > 0 {
		fields["attachments"] = attachments
	}
	if op.Create {
		_, err := c.service.client.Chat().SendMessage(ctx, "agent", strings.TrimPrefix(c.data.CID, "agent:"), &getstream.SendMessageRequest{Message: getstream.MessageRequest{
			ID: &m.ID, UserID: &user, Text: &m.Text,
			Custom: map[string]any{"source": "agent", "generating": m.FinishedAt == nil,
				"support_message": metadata, "support_runtime": runtime},
		}})
		if err != nil && ctx.Err() == nil {
			existing, readErr := c.service.client.Chat().GetMessage(ctx, m.ID, &getstream.GetMessageRequest{})
			if readErr == nil {
				stored, decodeErr := messageFromWire(existing.Data.Message.ID, existing.Data.Message.Text, existing.Data.Message.Custom)
				if decodeErr == nil && stored.ID == m.ID && stored.Role == m.Role && stored.StartedAt.Equal(m.StartedAt) {
					return nil
				}
			}
		}
		return err
	}
	if ephemeral {
		_, err := c.service.client.Chat().EphemeralMessageUpdate(ctx, m.ID, &getstream.EphemeralMessageUpdateRequest{UserID: &user, Set: fields})
		return err
	}
	_, err = c.service.client.Chat().UpdateMessagePartial(ctx, m.ID, &getstream.UpdateMessagePartialRequest{UserID: &user, Set: fields})
	return err
}
func sameSnapshot(a, b Message) bool {
	a.Saved = false
	b.Saved = false
	a.Error = ""
	b.Error = ""
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return string(x) == string(y)
}
func (c *Conversation) flush() bool {
	for {
		c.mu.Lock()
		if len(c.data.Pending) == 0 {
			c.mu.Unlock()
			return true
		}
		// Never send an operation that only exists in memory after a failed disk write.
		if err := c.persist(); err != nil {
			c.mu.Unlock()
			return false
		}
		op := c.data.Pending[0]
		c.mu.Unlock()
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		err := c.send(ctx, op, false)
		cancel()
		c.mu.Lock()
		if err != nil {
			if c.data.Current != nil {
				c.data.Current.Error = "Stream Chat write failed; retry queued"
				c.publish(*c.data.Current)
			}
			c.mu.Unlock()
			return false
		}
		pending := c.data.Pending
		c.data.Pending = pending[1:]
		if err := c.persist(); err != nil {
			c.data.Pending = pending
			c.mu.Unlock()
			return false
		}
		if op.Create {
			c.created[op.Message.ID] = true
		}
		op.Message.Saved = true
		if c.data.Current != nil && c.data.Current.ID == op.Message.ID {
			if sameSnapshot(*c.data.Current, op.Message) {
				c.data.Current.Saved = true
				c.data.Current.Error = ""
			}
			if op.Message.FinishedAt != nil {
				c.publish(*c.data.Current)
			}
		} else {
			c.publish(op.Message)
		}
		c.mu.Unlock()
	}
}
func (c *Conversation) run() {
	defer close(c.done)
	tick := time.NewTicker(200 * time.Millisecond)
	defer tick.Stop()
	retry := time.Time{}
	for {
		select {
		case <-c.stopped:
			c.flush()
			return
		case <-tick.C:
			if time.Now().Before(retry) {
				continue
			}
			if !c.flush() {
				retry = time.Now().Add(2 * time.Second)
				continue
			}
			c.mu.Lock()
			m := c.data.Current
			dirty := c.dirty && m != nil && c.created[m.ID]
			if m != nil {
				copy := *m
				copy.Tools = append([]Tool{}, m.Tools...)
				copy.Sources = append([]Source{}, m.Sources...)
				m = &copy
			}
			if dirty {
				c.dirty = false
			}
			c.mu.Unlock()
			if dirty && m != nil && m.FinishedAt == nil {
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				err := c.send(ctx, operation{Message: *m}, true)
				cancel()
				if err != nil {
					c.mu.Lock()
					if c.data.Current != nil {
						c.data.Current.Error = "Live Stream Chat update failed; final writes remain queued"
						c.dirty = true
						c.publish(*c.data.Current)
					}
					c.mu.Unlock()
				} else {
					c.mu.Lock()
					if c.data.Current != nil && c.data.Current.ID == m.ID {
						c.data.Current.Error = ""
					}
					c.mu.Unlock()
				}
			}
		}
	}
}

func sameMemoryScope(a, b memory.Scope) bool {
	return a.UserID == b.UserID && a.AppID == b.AppID && maps.Equal(a.Extra, b.Extra)
}
