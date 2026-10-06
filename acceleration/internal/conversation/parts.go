package conversation

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// A reply's steps, in order, as the Stream attachments Stream's AI components render:
// each round of the model's thinking (ai_reasoning) and each tool call people may see
// (ai_tool_call). The attachments' order is the timeline and the answer stays in the
// message text. The runtime is their only writer: they ride on the live updates while a
// reply works and are stored with the settled reply.
//
// A reasoning step stores its first sentence as its summary and its opening as its
// preview. The whole of the thinking is only ever shown live, through the reasoning
// windows of ephemeral updates, and is never stored.
const (
	partReasoning = "ai_reasoning"
	partToolCall  = "ai_tool_call"
	partVersion   = 1

	// Stream allows 30 attachments on a message, at most 5 KB together. Artifacts count
	// towards both, and the parts give way to them.
	maxMessageAttachments = 30
	maxAttachmentBytes    = 4800

	// A settled reasoning step keeps its opening as its preview; a streaming one carries
	// its latest thoughts, for clients that do not follow the windows.
	maxStoredPreview = 500
	maxLivePreview   = 200
	maxSummary       = 140
	// maxHead is how much of a round's start is kept to make its summary and preview.
	maxHead = 2048
	// A client tool's arguments are shown to every channel member, so they stay small.
	maxClientArguments = 512
	maxToolSummary     = 120
	// An approval's question is shown to every channel member too.
	maxApprovalTitle   = 80
	maxApprovalMessage = 240
	maxApprovalReason  = 160
	maxApprovalButton  = 40
)

// Part is one step of a reply. It is kept in the local ledger with the reply, so a restart
// recovers the steps shown so far.
type Part struct {
	Type   string `json:"type"`
	V      int    `json:"v"`
	ID     string `json:"id"`
	Status string `json:"status"`

	Summary    string `json:"summary,omitempty"`
	Preview    string `json:"preview,omitempty"`
	DurationMS int64  `json:"duration_ms,omitempty"`

	Name           string          `json:"name,omitempty"`
	DisplayTitle   string          `json:"display_title,omitempty"`
	Executor       string          `json:"executor,omitempty"`
	TargetUserID   string          `json:"target_user_id,omitempty"`
	TargetClientID string          `json:"target_client_id,omitempty"`
	Arguments      json.RawMessage `json:"arguments,omitempty"`
	// Approval is the question a call that waits for a person asks them, and their answer.
	Approval   *Approval  `json:"approval,omitempty"`
	StartedAt  *time.Time `json:"started_at,omitempty"`
	FinishedAt *time.Time `json:"finished_at,omitempty"`
}

// Approval is what a call asks the person it waits for, and how they answered.
type Approval struct {
	Title   string `json:"title"`
	Message string `json:"message,omitempty"`
	// Reason is the model's own words for why it wants the call.
	Reason       string `json:"reason,omitempty"`
	AllowTitle   string `json:"allow_title,omitempty"`
	DeclineTitle string `json:"decline_title,omitempty"`
	// Decision is "allowed" or "declined" once they answered.
	Decision string `json:"decision,omitempty"`
}

// ToolDisplay is what the people in a conversation may know about one of the caller's
// tools: what its calls are doing, whether a person's device runs it, and what a person
// is asked before each call runs.
type ToolDisplay struct {
	Title    string
	Client   bool
	Approval *ToolApproval
}

// ToolApproval is a tool's question to the person its calls wait for.
type ToolApproval struct {
	Title   string
	Message string
	// ReasonArgument names the string argument in which the model says why.
	ReasonArgument string
	AllowTitle     string
	DeclineTitle   string
}

// DescribeTools records the caller's tools for this session.
func (c *Conversation) DescribeTools(tools map[string]ToolDisplay) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.tools = tools
}

// thought records a piece of thinking, opening a reasoning step when none is streaming.
// It reports whether a step was opened, which changes the reply.
func (c *Conversation) thought(m *Message, text string, now time.Time) bool {
	opened := false
	if last := len(m.Parts) - 1; last < 0 || m.Parts[last].Type != partReasoning || m.Parts[last].Status != "streaming" {
		rounds := 0
		for _, part := range m.Parts {
			if part.Type == partReasoning {
				rounds++
			}
		}
		id := "r" + strconv.Itoa(rounds+1)
		started := now.UTC()
		m.Parts = append(m.Parts, Part{Type: partReasoning, V: partVersion, ID: id, Status: "streaming", StartedAt: &started})
		c.reasoning = liveReasoning{id: id}
		opened = true
	}
	c.reasoning.add(text, now)
	return opened
}

// settleThinking completes the streaming reasoning step, if there is one, with its summary
// and stored preview. It reports whether the reply changed.
func (c *Conversation) settleThinking(m *Message, now time.Time) bool {
	last := len(m.Parts) - 1
	if last < 0 || m.Parts[last].Type != partReasoning || m.Parts[last].Status != "streaming" {
		return false
	}
	part := &m.Parts[last]
	part.Status = "completed"
	if c.reasoning.id == part.ID && c.reasoning.total > 0 {
		part.Summary = summaryOf(c.reasoning.head)
		part.Preview = openingOf(c.reasoning.head, c.reasoning.total)
		part.DurationMS = c.reasoning.last.Sub(c.reasoning.first).Milliseconds()
	}
	finished := now.UTC()
	part.FinishedAt = &finished
	return true
}

// called opens a tool step for a call people may see: one the agent config's visible_tools
// names, or one a person's device runs.
func (c *Conversation) called(m *Message, e toolCall) {
	display := c.tools[e.name]
	// A call a person's device runs or a person must allow is always shown, whatever the
	// agent config shows: somebody is waiting on it.
	if !display.Client && display.Approval == nil && !ToolVisible(c.data.VisibleTools, e.name) {
		return
	}
	started := e.startedAt.UTC()
	part := Part{Type: partToolCall, V: partVersion, ID: e.id, Status: "running", Name: e.name,
		DisplayTitle: display.Title, Executor: "server", StartedAt: &started}
	// The runtime's own search has no caller to title it. Other untitled calls are left to
	// the clients, which name them from the tool.
	if part.DisplayTitle == "" && (e.name == "search" || e.name == "web_search") {
		part.DisplayTitle = "Searching the web"
	}
	if display.Client {
		part.Executor = "client"
		record := c.data.Commands[m.CommandID]
		part.TargetUserID = record.Initiator
		part.TargetClientID = record.ClientID
		if arguments := json.RawMessage(e.arguments); len(arguments) <= maxClientArguments && json.Valid(arguments) {
			part.Arguments = arguments
		}
		// Without an install to address, nobody is waiting on it: the caller answers it.
		if part.TargetUserID != "" && part.TargetClientID != "" {
			part.Status = "awaiting_client"
		}
	}
	// A call a person must allow first waits for the one whose command it answers, with
	// the question their client asks. A client tool with no install to address runs
	// nowhere, so nobody is asked.
	if approval := display.Approval; approval != nil && (!display.Client || part.Status == "awaiting_client") {
		if initiator := c.data.Commands[m.CommandID].Initiator; initiator != "" {
			part.TargetUserID = initiator
			part.Status = "awaiting_approval"
			part.Approval = &Approval{
				Title:        capRunes(strings.TrimSpace(approval.Title), maxApprovalTitle),
				Message:      capRunes(strings.TrimSpace(approval.Message), maxApprovalMessage),
				Reason:       reasonOf(e.arguments, approval.ReasonArgument),
				AllowTitle:   capRunes(strings.TrimSpace(approval.AllowTitle), maxApprovalButton),
				DeclineTitle: capRunes(strings.TrimSpace(approval.DeclineTitle), maxApprovalButton),
			}
		}
	}
	m.Parts = append(m.Parts, part)
}

// reasonOf is the model's reason for a call: the named string argument, on one line.
func reasonOf(arguments, name string) string {
	if name == "" {
		return ""
	}
	var values map[string]any
	if json.Unmarshal([]byte(arguments), &values) != nil {
		return ""
	}
	text, _ := values[name].(string)
	return capRunes(strings.Join(strings.Fields(text), " "), maxApprovalReason)
}

// decided records a person's answer to a call awaiting their approval. Allowed, the call
// goes on as it would have; declined, it is cancelled with their summary. It reports
// whether the call was waiting for an answer.
func decided(m *Message, id string, allowed bool, summary string, now time.Time) bool {
	for i := range m.Parts {
		part := &m.Parts[i]
		if part.Type != partToolCall || part.ID != id || part.Status != "awaiting_approval" || part.Approval == nil {
			continue
		}
		if allowed {
			part.Approval.Decision = "allowed"
			part.Status = "running"
			if part.Executor == "client" {
				part.Status = "awaiting_client"
			}
			return true
		}
		finished := now.UTC()
		part.Approval.Decision = "declined"
		part.Status, part.FinishedAt = "cancelled", &finished
		if part.StartedAt != nil {
			part.DurationMS = finished.Sub(*part.StartedAt).Milliseconds()
		}
		part.Summary = capRunes(strings.TrimSpace(summary), maxToolSummary)
		return true
	}
	return false
}

type toolCall struct {
	id, name, arguments string
	startedAt           time.Time
}

// ran settles a tool step. A client tool's result may carry a summary people can see, and
// when the device could not run it, the caller's reason is the summary.
func ran(m *Message, id, status, result, failure string, now time.Time) {
	for i := range m.Parts {
		part := &m.Parts[i]
		if part.Type != partToolCall || part.ID != id || part.FinishedAt != nil {
			continue
		}
		finished := now.UTC()
		part.Status, part.FinishedAt = status, &finished
		if part.StartedAt != nil {
			part.DurationMS = finished.Sub(*part.StartedAt).Milliseconds()
		}
		if part.Executor == "client" {
			var outcome struct {
				Summary string `json:"summary"`
			}
			if failure != "" {
				part.Summary = capRunes(strings.TrimSpace(failure), maxToolSummary)
			} else if json.Unmarshal([]byte(result), &outcome) == nil {
				part.Summary = capRunes(strings.TrimSpace(outcome.Summary), maxToolSummary)
			}
		} else if part.Approval != nil && part.Approval.Decision == "" && failure != "" {
			// A question nobody answered says why the call ended, as a device does.
			part.Summary = capRunes(strings.TrimSpace(failure), maxToolSummary)
		}
	}
}

// stopParts ends every step still in progress when a reply finishes.
func (c *Conversation) stopParts(m *Message, now time.Time) {
	c.settleThinking(m, now)
	finished := now.UTC()
	for i := range m.Parts {
		part := &m.Parts[i]
		if part.Type == partToolCall && part.FinishedAt == nil {
			part.Status, part.FinishedAt = "cancelled", &finished
		}
	}
}

// liveParts is a reply's steps as a watcher sees them now: the streaming step carries its
// latest thoughts and how long it has been thinking.
func liveParts(parts []Part, live liveSnapshot) []Part {
	out := append([]Part{}, parts...)
	if last := len(out) - 1; last >= 0 && out[last].Status == "streaming" && out[last].ID == live.id {
		out[last].Preview = live.preview
		out[last].DurationMS = live.durationMS
	}
	return out
}

// liveSnapshot is what an update needs of the streaming thinking, taken under the lock.
type liveSnapshot struct {
	id         string
	preview    string
	durationMS int64
}

func (r *liveReasoning) snapshot() liveSnapshot {
	return liveSnapshot{id: r.id, preview: tailRunes(r.buf, maxLivePreview), durationMS: r.last.Sub(r.first).Milliseconds()}
}

// messageAttachments renders a reply's steps and artifacts within Stream's limits. The
// artifacts are what the reply delivered, so they are kept whole; the steps give way,
// oldest first: a stored preview shrinks, then goes, then summaries go, then the oldest
// finished steps.
func messageAttachments(parts []Part, artifacts []map[string]any) []map[string]any {
	parts = append([]Part{}, parts...)
	for len(parts)+len(artifacts) > maxMessageAttachments && dropFinished(&parts) {
	}
	fits := func() bool { return attachmentBytes(parts, artifacts) <= maxAttachmentBytes }
	for _, shrink := range []func(*Part) bool{
		func(p *Part) bool { return trimPreview(p, 160) },
		func(p *Part) bool { return trimPreview(p, 0) },
		func(p *Part) bool {
			// A question that was answered keeps its title and the answer.
			if p.Approval == nil || inProgress(p.Status) || p.Approval.Message == "" && p.Approval.Reason == "" {
				return false
			}
			approval := *p.Approval
			approval.Message, approval.Reason = "", ""
			p.Approval = &approval
			return true
		},
		func(p *Part) bool {
			if p.Summary == "" || p.Status == "streaming" || p.Status == "awaiting_client" || p.Status == "awaiting_approval" {
				return false
			}
			p.Summary = ""
			return true
		},
	} {
		for i := 0; i < len(parts) && !fits(); i++ {
			shrink(&parts[i])
		}
	}
	for !fits() && dropFinished(&parts) {
	}
	out := make([]map[string]any, 0, len(parts)+len(artifacts))
	for _, part := range parts {
		out = append(out, part.attachment())
	}
	return append(out, artifacts...)
}

func (p Part) attachment() map[string]any {
	raw, _ := json.Marshal(p)
	var out map[string]any
	_ = json.Unmarshal(raw, &out)
	return out
}

func attachmentBytes(parts []Part, artifacts []map[string]any) int {
	size := 2
	for _, part := range parts {
		raw, _ := json.Marshal(part)
		size += len(raw) + 1
	}
	for _, artifact := range artifacts {
		raw, _ := json.Marshal(artifact)
		size += len(raw) + 1
	}
	return size
}

func trimPreview(p *Part, limit int) bool {
	if p.Type != partReasoning || p.Status == "streaming" || utf8.RuneCountInString(p.Preview) <= limit {
		return false
	}
	if limit == 0 {
		p.Preview = ""
	} else {
		p.Preview = capRunes(p.Preview, limit)
	}
	return true
}

// dropFinished removes the oldest step that is over, keeping anything still in progress.
func dropFinished(parts *[]Part) bool {
	for i, part := range *parts {
		if !inProgress(part.Status) {
			*parts = append((*parts)[:i], (*parts)[i+1:]...)
			return true
		}
	}
	return false
}

// inProgress reports whether a step is still going: thinking, running, or waiting for a
// person or their device.
func inProgress(status string) bool {
	switch status {
	case "streaming", "running", "awaiting_client", "awaiting_approval":
		return true
	}
	return false
}

// summaryOf is a round's first sentence.
func summaryOf(head string) string {
	text := strings.TrimSpace(head)
	end := len(text)
	for i, r := range text {
		if r == '\n' {
			end = i
			break
		}
		if (r == '.' || r == '?' || r == '!') && (i+1 == len(text) || text[i+1] == ' ' || text[i+1] == '\n') {
			end = i + 1
			break
		}
	}
	return capRunes(strings.TrimSpace(text[:end]), maxSummary)
}

// openingOf is the stored preview: a round's opening, marked when there was more.
func openingOf(head string, total int) string {
	text := strings.TrimSpace(head)
	if utf8.RuneCountInString(text) > maxStoredPreview {
		return capRunes(text, maxStoredPreview)
	}
	if total > utf8.RuneCountInString(head) {
		return text + "…"
	}
	return text
}

// capRunes cuts text to limit characters, ending with an ellipsis when it cut.
func capRunes(text string, limit int) string {
	if utf8.RuneCountInString(text) <= limit {
		return text
	}
	runes := []rune(text)[:limit-1]
	return strings.TrimRight(string(runes), " \n") + "…"
}

// tailRunes is the last limit characters of text.
func tailRunes(text string, limit int) string {
	if utf8.RuneCountInString(text) <= limit {
		return text
	}
	runes := []rune(text)
	return string(runes[len(runes)-limit:])
}
