package conversation

import (
	"encoding/json"
	"errors"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// BranchField is the custom key a message's place among the conversation's branches is kept
// under. It is a key of its own, not part of support_runtime, which a router decodes strictly:
// a field added there would make a router one version older refuse every message written
// since, during a deploy or after a rollback.
const BranchField = "support_branch"

// What a Branch's Index and Count count.
const (
	// BranchOfReply counts the answers to one question.
	BranchOfReply = store.SiblingsOfReply
	// BranchOfQuestion counts the questions asked at one point of the conversation.
	BranchOfQuestion = store.SiblingsOfQuestion
)

// Branch is a message's place among the conversation's branches (BranchField): the response
// it belongs to, and where it stands among its alternatives. A message without one, or with no
// Of, is 1 of 1.
type Branch struct {
	ResponseID         string `json:"response_id"`
	ParentResponseID   string `json:"parent_response_id,omitempty"`
	QuestionResponseID string `json:"question_response_id,omitempty"`
	Of                 string `json:"of,omitempty"`
	Index              int    `json:"index,omitempty"`
	Count              int    `json:"count,omitempty"`
}

// maxBranches is the most alternatives a mark is read back with: far more than the 100
// rewound ranges Chat keeps for a channel can make.
const maxBranches = 10000

// branchFrom reads a stored mark back. Whatever it cannot read, as a router of another
// version may have written, is no mark at all rather than a message that cannot be read.
func branchFrom(raw any) *Branch {
	if raw == nil {
		return nil
	}
	encoded, err := json.Marshal(raw)
	if err != nil || len(encoded) > 4<<10 {
		return nil
	}
	var b Branch
	if json.Unmarshal(encoded, &b) != nil || !validCommandID.MatchString(b.ResponseID) ||
		len(b.ParentResponseID) > 128 || len(b.QuestionResponseID) > 128 {
		return nil
	}
	switch b.Of {
	case "":
		b.Index, b.Count = 0, 0
	case BranchOfReply, BranchOfQuestion:
		if b.Count < 1 || b.Count > maxBranches || b.Index < 1 || b.Index > b.Count {
			b.Of, b.Index, b.Count = "", 0, 0
		}
	default:
		b.Of, b.Index, b.Count = "", 0, 0
	}
	return &b
}

// Place is where a message stands among its alternatives, 1-based. The zero Place is 1 of 1,
// which is not written.
type Place struct {
	Index, Count int
}

// Turn is the response a command begins, with the id its caller allocated, so both of the
// command's messages carry it from the moment they are written, and where its question and
// its reply stand among their alternatives. The zero Turn marks nothing.
type Turn struct {
	ResponseID         string
	ParentResponseID   string
	QuestionResponseID string
	Question, Reply    Place
}

func (t Turn) valid() bool {
	if t.ResponseID == "" {
		return t == Turn{}
	}
	for _, id := range []string{t.ResponseID, t.ParentResponseID, t.QuestionResponseID} {
		if id != "" && !validCommandID.MatchString(id) {
			return false
		}
	}
	for _, p := range []Place{t.Question, t.Reply} {
		if p != (Place{}) && (p.Count < 1 || p.Count > maxBranches || p.Index < 1 || p.Index > p.Count) {
			return false
		}
	}
	return true
}

// mark is the mark of the turn's message of, nil for the zero Turn.
func (t Turn) mark(of string, at Place) *Branch {
	if t.ResponseID == "" {
		return nil
	}
	b := &Branch{ResponseID: t.ResponseID, ParentResponseID: t.ParentResponseID, QuestionResponseID: t.QuestionResponseID}
	if at != (Place{}) {
		b.Of, b.Index, b.Count = of, at.Index, at.Count
	}
	return b
}

// regenerateDigest is what a regenerate under one command id is matched on: the question it
// answers again, which its reply carries (question_id), so commandsIn rebuilds it.
func regenerateDigest(questionID string) string {
	return commandDigest("\x00regenerate\x00"+questionID, nil, nil)
}

// BeginCommandWithTurn is BeginCommandWithAction for a command that begins turn, whose two
// messages carry its marks (BranchField).
func (c *Conversation) BeginCommandWithTurn(turn Turn, id, text, clientID string, action map[string]any, photos ...Photo) (CommandReceipt, error) {
	if !turn.valid() {
		return CommandReceipt{}, stack.Wrap(errors.New("invalid turn"))
	}
	return c.beginCommand(turn, id, text, clientID, action, photos...)
}

// BeginRegenerate answers the person's message questionID again under the command id: it
// writes only a new reply, which carries turn's marks and points at the question. A retry
// with the same id and question is the same command; the same id for another question is a
// conflict. The caller has rewound the old reply away first (Rewind).
func (c *Conversation) BeginRegenerate(turn Turn, id, questionID string) (CommandReceipt, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.active {
		return CommandReceipt{}, stack.Wrap(errors.New("conversation is not open"))
	}
	if !validCommandID.MatchString(id) || !validCommandID.MatchString(questionID) || turn.ResponseID == "" || !turn.valid() {
		return CommandReceipt{}, stack.Wrap(errInvalidCommand)
	}
	digest := regenerateDigest(questionID)
	if previous, ok := c.data.Commands[id]; ok {
		if previous.Initiator != c.data.Owner {
			return CommandReceipt{}, stack.Wrap(ErrCommandNotFound)
		}
		if previous.Digest != digest {
			return CommandReceipt{}, stack.Wrap(ErrCommandConflict)
		}
		receipt := previous.CommandReceipt
		receipt.Duplicate = true
		return receipt, nil
	}
	// A question older than the page read at open is not in the ledger, and is the owner's
	// all the same: the conversation is.
	if asked, ok := c.data.Commands[questionID]; ok && asked.Initiator != c.data.Owner {
		return CommandReceipt{}, stack.Wrap(ErrCommandNotFound)
	}
	if c.data.Current != nil && c.data.Current.FinishedAt == nil {
		return CommandReceipt{}, stack.Wrap(ErrResponseRunning)
	}
	now := time.Now().UTC()
	a := Message{ID: uuid.NewString(), CommandID: id, Role: "assistant", TextLayout: 1, QuestionID: questionID,
		State: "thinking", StartedAt: now, StateStartedAt: now, Tools: []Tool{}, Branch: turn.mark(BranchOfReply, turn.Reply)}
	receipt := CommandReceipt{CommandID: id, UserMessageID: questionID, AssistantMessageID: a.ID, State: a.State}
	if c.data.Commands == nil {
		c.data.Commands = map[string]commandRecord{}
	}
	c.data.Commands[id] = commandRecord{CommandReceipt: receipt, Digest: digest, Initiator: c.data.Owner}
	c.data.Current = &a
	c.reasoning = liveReasoning{}
	c.data.Pending = append(c.data.Pending, operation{Message: a, Create: true})
	c.publish(a)
	return receipt, nil
}
