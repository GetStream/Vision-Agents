package conversation

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// MarshalJSON writes a message's connector authorizations as connector_authorization
// attachments after its tool_calling ones, the same attachment a Chat channel's copy of the
// message carries.
func (m Message) MarshalJSON() ([]byte, error) {
	type plain Message
	attachments := make([]any, 0, len(m.Tools)+len(m.ConnectorAuthorizations))
	for _, tool := range m.Tools {
		attachments = append(attachments, tool)
	}
	for _, authorization := range m.ConnectorAuthorizations {
		attachments = append(attachments, authorization)
	}
	return json.Marshal(struct {
		plain
		Attachments []any `json:"attachments"`
	}{plain(m), attachments})
}

// BeginFollowUp records a reply nobody wrote a message for, such as the one that carries on
// once the login a reply asked for is made. text is what the model is told instead, which
// the conversation never shows. The id is the caller's own, chosen so that a follow-up
// delivered twice (a worker that died mid-delivery and was replaced) is answered Duplicate
// and started once: the ledger holds it in this process, and after a restart commandsIn
// rebuilds it from the reply Stream holds. writer is the Stream user whose message it answers
// in a channel of a type an agent config answers in (agent.WriterOf), empty otherwise.
func (c *Conversation) BeginFollowUp(id, text, writer string) (CommandReceipt, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.active {
		return CommandReceipt{}, stack.Wrap(errors.New("conversation is not open"))
	}
	if !validCommandID.MatchString(id) {
		return CommandReceipt{}, stack.Wrap(errors.New("invalid follow-up ID"))
	}
	if previous, ok := c.data.Commands[id]; ok {
		receipt := previous.CommandReceipt
		receipt.Duplicate = true
		return receipt, nil
	}
	if c.data.Current != nil && c.data.Current.FinishedAt == nil {
		return CommandReceipt{}, stack.Wrap(ErrResponseRunning)
	}
	now := time.Now().UTC()
	question := ""
	if c.data.Current != nil {
		question = c.data.Current.QuestionID
	}
	a := Message{ID: uuid.NewString(), CommandID: id, Role: "assistant", TextLayout: 1, QuestionID: question, State: "thinking", StartedAt: now, StateStartedAt: now, Tools: []Tool{}}
	receipt := CommandReceipt{CommandID: id, AssistantMessageID: a.ID, State: a.State}
	if c.data.Commands == nil {
		c.data.Commands = map[string]commandRecord{}
	}
	c.data.Commands[id] = commandRecord{CommandReceipt: receipt, Digest: fmt.Sprintf("%x", sha256.Sum256([]byte(text))), Initiator: c.data.Owner, Writer: writer}
	c.data.Current = &a
	c.reasoning = liveReasoning{}
	c.data.Pending = append(c.data.Pending, operation{Message: a, Create: true})
	c.publish(a)
	return receipt, nil
}

// AbandonFollowUp cancels a follow-up whose turn never started and forgets its id, so the
// same follow-up delivered again starts rather than being answered Duplicate with nothing
// run for it.
func (c *Conversation) AbandonFollowUp(id string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if m := c.data.Current; m != nil && m.CommandID == id {
		c.finish("cancelled")
	}
	delete(c.data.Commands, id)
}

// FollowUpReply is how the reply to the follow-up under id stands: its state and, once it
// has finished, what it said. known is false for an id the conversation does not hold.
func (c *Conversation) FollowUpReply(id string) (state, text string, known bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	record, known := c.data.Commands[id]
	return record.State, record.Reply, known
}

// FollowUpFinished is the reply to the follow-up under id as OnFinishedReply tells it, with
// its files, for a follow-up delivered again to send rather than run twice. One the
// conversation does not hold has no message id, text or files.
func (c *Conversation) FollowUpFinished(id string) FinishedReply {
	c.mu.Lock()
	defer c.mu.Unlock()
	record := c.data.Commands[id]
	return FinishedReply{Customer: c.data.Customer, CID: c.data.CID, MessageID: record.AssistantMessageID, CommandID: id,
		Text: record.Reply, Files: append([]sandbox.Attachment(nil), record.Files...)}
}

// rememberAsked keeps a finished reply that asked for a login, to mark it once the login is
// made. Only the latest few are kept: an older button is one nobody is waiting on.
func (c *Conversation) rememberAsked(m Message) {
	if len(m.ConnectorAuthorizations) == 0 {
		return
	}
	m.ConnectorAuthorizations = append([]ConnectorAuthorization{}, m.ConnectorAuthorizations...)
	c.asked = append(c.asked, m)
	if len(c.asked) > maxAsked {
		c.asked = c.asked[len(c.asked)-maxAsked:]
	}
}

const maxAsked = 10
