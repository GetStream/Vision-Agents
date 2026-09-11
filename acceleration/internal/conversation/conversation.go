// Package conversation persists text conversations and their visible activity in Stream Chat.
package conversation

import (
	"context"
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
type Message struct {
	ID             string     `json:"id"`
	QuestionID     string     `json:"question_id,omitempty"`
	Role           string     `json:"role"`
	Text           string     `json:"text"`
	State          string     `json:"state"`
	StartedAt      time.Time  `json:"response_started_at"`
	StateStartedAt time.Time  `json:"state_started_at"`
	FinishedAt     *time.Time `json:"finished_at,omitempty"`
	DurationMS     int64      `json:"duration_ms"`
	Tools          []Tool     `json:"attachments"`
	Saved          bool       `json:"saved"`
	Error          string     `json:"persistence_error,omitempty"`
}
type Page struct {
	memoryScope memory.Scope
	Messages    []Message `json:"messages"`
	Before      string    `json:"before,omitempty"`
	Truncated   bool      `json:"context_truncated"`
}
type Updated struct {
	CID     string  `json:"conversation_id"`
	Message Message `json:"message"`
}
type disk struct {
	CID      string
	Customer string
	Agent    string
	Current  *Message
}
type operation struct {
	Message Message
	Create  bool
}
type Service struct {
	mu     sync.Mutex
	client *getstream.Stream
	root   string
	all    map[string]*Conversation
}
type Conversation struct {
	mu      sync.Mutex
	service *Service
	data    disk
	active  bool
	turns   map[string]string
	created map[string]bool
	emit    func(Updated)
	dirty   bool
	stopped chan struct{}
	done    chan struct{}
}

var validID = regexp.MustCompile(`^support-[a-f0-9-]{36}$`)

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
func newService(root string, client *getstream.Stream) (*Service, error) {
	var err error
	if root == "" {
		root = ".local/chat-outbox"
	}
	if err = os.MkdirAll(root, 0700); err != nil {
		return nil, err
	}
	s := &Service{client: client, root: root, all: map[string]*Conversation{}}
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	for _, e := range entries {
		if !e.IsDir() || !validID.MatchString(e.Name()) {
			continue
		}
		b, err := os.ReadFile(filepath.Join(root, e.Name(), "state.json"))
		if err != nil {
			return nil, err
		}
		var d disk
		if err = json.Unmarshal(b, &d); err != nil {
			return nil, err
		}
		c := s.make(d)
		c.mu.Lock()
		c.finish("interrupted")
		c.mu.Unlock()
	}
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
	all := make([]*Conversation, 0, len(s.all))
	for _, c := range s.all {
		all = append(all, c)
	}
	s.mu.Unlock()
	for _, c := range all {
		close(c.stopped)
	}
	for _, c := range all {
		<-c.done
	}
}
func (s *Service) Open(ctx context.Context, customer, agentID, cid string, scopes ...memory.Scope) (*Conversation, []llm.Message, bool, error) {
	ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	s.mu.Lock()
	defer s.mu.Unlock()
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
		if c.data.Customer != customer || c.data.Agent != agentID {
			return nil, nil, false, errors.New("conversation belongs to another customer or agent")
		}
		if c.active {
			return nil, nil, false, errors.New("conversation is already open")
		}
		page, err := s.history(ctx, customer, agentID, cid, "")
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
		_, err := s.client.UpdateUsers(ctx, &getstream.UpdateUsersRequest{Users: map[string]getstream.UserRequest{agentID: {ID: agentID}, "support-operator": {ID: "support-operator"}}})
		if err != nil {
			return nil, nil, false, err
		}
		_, err = s.client.Chat().GetOrCreateChannel(ctx, "agent", id, &getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{CreatedByID: &agentID, Members: []getstream.ChannelMemberRequest{{UserID: agentID}, {UserID: "support-operator"}}, Custom: map[string]any{"support_customer_id": customer, "support_agent_id": agentID, "support_memory_scope": scope}}})
		if err != nil {
			return nil, nil, false, err
		}
	}
	page, err := s.history(ctx, customer, agentID, cid, "")
	if err != nil {
		return nil, nil, false, err
	}
	if !sameMemoryScope(page.memoryScope, scope) {
		return nil, nil, false, errors.New("conversation belongs to another memory scope; reopen with its original organization")
	}
	c := s.make(disk{CID: cid, Customer: customer, Agent: agentID})
	c.active = true
	h, tr := history(page)
	return c, h, tr, nil
}
func (s *Service) History(ctx context.Context, customer, agentID, cid, before string) (Page, error) {
	return s.history(ctx, customer, agentID, cid, before)
}
func (s *Service) history(ctx context.Context, customer, agentID, cid, before string) (Page, error) {
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
	if r.Data.Channel.Custom["support_customer_id"] != customer || r.Data.Channel.Custom["support_agent_id"] != agentID {
		return Page{}, errors.New("conversation belongs to another customer or agent")
	}
	p := Page{Messages: []Message{}, Truncated: len(r.Data.Messages) == limit}
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
		files, _ := filepath.Glob(filepath.Join(s.root, id, "ops", "*.json"))
		sort.Strings(files)
		for _, file := range files {
			raw, err := os.ReadFile(file)
			if os.IsNotExist(err) {
				continue
			}
			if err != nil {
				return Page{}, err
			}
			var op operation
			if err = json.Unmarshal(raw, &op); err != nil {
				return Page{}, err
			}
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
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.data.Current != nil && c.data.Current.FinishedAt == nil {
		return errors.New("a response is already running")
	}
	now := time.Now().UTC()
	u := Message{ID: uuid.NewString(), Role: "user", Text: text, State: "completed", StartedAt: now, StateStartedAt: now, FinishedAt: &now, Tools: []Tool{}}
	if err := c.enqueue(u, true); err != nil {
		return err
	}
	c.publish(u)
	a := Message{ID: uuid.NewString(), Role: "assistant", QuestionID: u.ID, State: "thinking", StartedAt: now, StateStartedAt: now, Tools: []Tool{}}
	c.data.Current = &a
	if err := c.enqueue(a, true); err != nil {
		c.data.Current = nil
		return err
	}
	c.publish(a)
	return nil
}
func (c *Conversation) Cancel() { c.mu.Lock(); defer c.mu.Unlock(); c.finish("cancelled") }
func (c *Conversation) Observe(event agent.Event) {
	c.mu.Lock()
	defer c.mu.Unlock()
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
			}
		}
		c.afterTools()
		persist = true
	case agent.TaskCancelled:
		for i := range m.Tools {
			t := &m.Tools[i]
			if t.ID == e.TaskID && t.FinishedAt == nil {
				now := time.Now().UTC()
				t.FinishedAt = &now
				t.DurationMS = now.Sub(t.StartedAt).Milliseconds()
				t.Status = "cancelled"
				t.Phase = "cancelled"
				t.Summary = "Skill cancelled"
			}
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
		return owner == c.data.Current.ID
	}
	if !start {
		return false
	}
	c.turns[id] = c.data.Current.ID
	return true
}
func (c *Conversation) Progress(id, phase string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	m := c.data.Current
	if m == nil || m.FinishedAt != nil {
		return
	}
	for i := range m.Tools {
		t := &m.Tools[i]
		if t.ID != id || t.FinishedAt != nil {
			continue
		}
		t.Phase = phase
		if phase == "queued" {
			t.Status = "queued"
			c.state("queued")
		} else {
			t.Status = "running"
			c.state("tools")
			if t.ExecutionStartedAt == nil && phase != "recovering_workspace" && phase != "starting_worker" {
				now := time.Now().UTC()
				t.ExecutionStartedAt = &now
				c.save()
			}
		}
	}
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
		m.StateStartedAt = time.Now().UTC()
	}
}
func (c *Conversation) finish(state string) {
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
	c.save()
	c.publish(*m)
}
func (c *Conversation) publish(m Message) {
	if c.emit != nil {
		m.Tools = append([]Tool{}, m.Tools...)
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
func (c *Conversation) enqueue(m Message, create bool) error {
	dir := c.dir()
	if err := os.MkdirAll(filepath.Join(dir, "ops"), 0700); err != nil {
		return err
	}
	if err := writeJSON(filepath.Join(dir, "state.json"), c.data); err != nil {
		return err
	}
	return writeJSON(filepath.Join(dir, "ops", fmt.Sprintf("%020d-%s.json", time.Now().UnixNano(), uuid.NewString())), operation{m, create})
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
		user = "support-operator"
	}
	m.Saved = !ephemeral
	m.Error = ""
	fields := map[string]any{"text": m.Text, "generating": m.FinishedAt == nil, "source": "agent", "support_message": m, "attachments": m.Tools}
	if op.Create {
		raw, _ := json.Marshal(m.Tools)
		var attachments []getstream.Attachment
		_ = json.Unmarshal(raw, &attachments)
		_, err := c.service.client.Chat().SendMessage(ctx, "agent", strings.TrimPrefix(c.data.CID, "agent:"), &getstream.SendMessageRequest{Message: getstream.MessageRequest{ID: &m.ID, UserID: &user, Text: &m.Text, Attachments: attachments, Custom: map[string]any{"source": "agent", "generating": m.FinishedAt == nil, "support_message": m}}})
		if err != nil && ctx.Err() == nil {
			existing, readErr := c.service.client.Chat().GetMessage(ctx, m.ID, &getstream.GetMessageRequest{})
			if readErr == nil {
				raw, _ := json.Marshal(existing.Data.Message.Custom["support_message"])
				var stored Message
				if json.Unmarshal(raw, &stored) == nil && stored.ID == m.ID && stored.Role == m.Role && stored.StartedAt.Equal(m.StartedAt) {
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
	_, err := c.service.client.Chat().UpdateMessagePartial(ctx, m.ID, &getstream.UpdateMessagePartialRequest{UserID: &user, Set: fields})
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
	files, _ := filepath.Glob(filepath.Join(c.dir(), "ops", "*.json"))
	sort.Strings(files)
	for _, file := range files {
		b, err := os.ReadFile(file)
		if err != nil {
			return false
		}
		var op operation
		if json.Unmarshal(b, &op) != nil {
			return false
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		err = c.send(ctx, op, false)
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
		_ = os.Remove(file)
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
	return true
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
