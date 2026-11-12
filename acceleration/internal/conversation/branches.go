package conversation

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"

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

// Branches keeps the copies of the messages branch switches hide and the journal of the
// switches, so a router that stops partway is finished by the next open. *store.Store is one.
type Branches interface {
	SaveBranchMessages(ctx context.Context, messages []store.AgentBranchMessage) error
	BranchMessages(ctx context.Context, customerID, sessionID string, responseIDs []string) ([]store.AgentBranchMessage, error)
	StartBranchSwitch(ctx context.Context, branchSwitch *store.AgentBranchSwitch) error
	AdvanceBranchSwitch(ctx context.Context, customerID, id, state string) error
	AbandonBranchSwitch(ctx context.Context, customerID, sessionID, id string) error
	CountBranchResent(ctx context.Context, customerID, id string) (int, error)
	UnfinishedBranchSwitches(ctx context.Context, customerID, sessionID string) ([]store.AgentBranchSwitch, error)
}

// SwitchHooks are the session's part of finishing a switch an earlier router left unfinished,
// which open runs.
type SwitchHooks struct {
	// Chosen is what an activation sends again, oldest first: the copies of the branch it
	// brings back.
	Chosen func(ctx context.Context, sw store.AgentBranchSwitch) ([]store.AgentBranchMessage, error)
	// Switched is told once Chat shows the switch's branch, before the switch is done: the
	// session moves which of its responses are on the active path.
	Switched func(ctx context.Context, sw store.AgentBranchSwitch) error
}

// SetBranches says where conversations keep their branch switches. Without it a conversation
// cannot switch branches.
func (s *Service) SetBranches(branches Branches) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.branches = branches
}

// SetSwitchHooks says how a switch an earlier router left unfinished is finished.
func (s *Service) SetSwitchHooks(hooks SwitchHooks) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.hooks = hooks
}

type sessionKey struct{}

// WithSession is ctx carrying the session a conversation opened under it keeps its branches
// under (Branches): the switches an earlier router left unfinished are finished as it opens,
// and its Rewind and Activate journal theirs there.
func WithSession(ctx context.Context, sessionID string) context.Context {
	return context.WithValue(ctx, sessionKey{}, sessionID)
}

func sessionOf(ctx context.Context) string {
	id, _ := ctx.Value(sessionKey{}).(string)
	return id
}

// Why a branch switch is refused. Each is matched with errors.Is; Chat's refusal, when there
// was one, stays in the chain.
var (
	// ErrBranchesUnsupported is a conversation that cannot have branches. errors.As with a
	// *BranchesUnsupportedError says why.
	ErrBranchesUnsupported = errors.New("conversation: this conversation cannot have branches")
	// ErrBranchLimit is a conversation with as many edit and regenerate points as Chat keeps
	// for a channel. errors.As with a *TooManyRewindsError gives the limit.
	ErrBranchLimit = errors.New("conversation: this conversation has as many edit and regenerate points as it can have; continue in a new chat")
	// ErrBranchConflict is a conversation that changed while it was being switched: another
	// rewind landed, the message rewound to is gone, or Chat could not tell it apart.
	ErrBranchConflict = errors.New("conversation: the conversation changed while it was being switched to another branch")
	// ErrBranchUnavailable is an app without Chat's rewind, or with it frozen.
	ErrBranchUnavailable = errors.New("conversation: branches are not enabled for this conversation's app")
	// ErrBranchTooLong is an activation that would send more than MaxResent messages again.
	ErrBranchTooLong = errors.New("conversation: this branch is too long to bring back; continue in a new chat")
	// ErrConversationUnwritable is a conversation whose queued writes cannot be written now,
	// such as one whose app is out of reach: it cannot be rewound until they are. Nothing was
	// changed, and the switch can be tried again later.
	ErrConversationUnwritable = errors.New("conversation: the conversation's queued writes cannot be written now")
)

// Why a conversation cannot have branches (BranchesUnsupportedError).
const (
	// UnsupportedThread is a thread channel the channel bridge writes.
	UnsupportedThread = "thread"
	// UnsupportedDispatched is a channel a worker was handed.
	UnsupportedDispatched = "dispatched"
	// UnsupportedChannel is a channel that is not a session's own.
	UnsupportedChannel = "channel"
	// UnsupportedShared is a conversation open to every member of its channel.
	UnsupportedShared = "shared"
	// UnsupportedNotKept is a session whose conversation is not kept in Chat, which the
	// session refuses with this error itself.
	UnsupportedNotKept = "not_kept"
	// UnsupportedCall is a session on a call, whose spoken turns a rewind would cut under the
	// caller. The session refuses it.
	UnsupportedCall = "call"
	// UnsupportedNative is a session on a speech-to-speech model, which keeps its own context
	// out of reach. The session refuses it.
	UnsupportedNative = "native"
	// UnsupportedUnmarked is a response whose messages carry no response id, as one written
	// before branches were kept: there is nothing to find them by.
	UnsupportedUnmarked = "unmarked"
)

// BranchesUnsupportedError is ErrBranchesUnsupported with the reason.
type BranchesUnsupportedError struct {
	Reason string
}

func (e *BranchesUnsupportedError) Error() string {
	return ErrBranchesUnsupported.Error() + " (" + e.Reason + ")"
}
func (e *BranchesUnsupportedError) Is(target error) bool { return target == ErrBranchesUnsupported }

// MaxResent is the most messages an activation sends again.
const MaxResent = 100

// RewindTarget is where a switch rewinds the conversation's channel to.
type RewindTarget struct {
	// SwitchID names the switch in the journal. A switch tried again under the same id is
	// finished from where it got to.
	SwitchID string
	// Kind is store.SwitchRegenerate, store.SwitchEdit or store.SwitchActivate.
	Kind string
	// TargetResponseID is the response the switch replaces or brings back.
	TargetResponseID string
	// MessageID is the message the channel is rewound to, hidden too when Inclusive.
	MessageID string
	Inclusive bool
	// TurnResponses are the responses of replies written before messages carried their
	// response id (BranchField), by turn id, for their copies to be found by.
	TurnResponses map[string]string
}

// RewindOutcome is what a switch did, and what the session rebuilds its model's history from.
type RewindOutcome struct {
	// Rewound is whether Chat hid anything: not when the message rewound to was the newest,
	// as a reply that never landed leaves it.
	Rewound bool
	// Hidden are the messages the switch hid, oldest first.
	Hidden []string
	// Resent are the messages an activation sent again, as Chat holds them now.
	Resent []Message
	// Page is the conversation as it is after the switch: what was read before the rewind, up
	// to its point, then what was sent again. Chat is not read again after a rewind, whose
	// replicas show what it hid for a moment.
	Page Page
	// History and Truncated are Page as a model is handed it, as Open hands it back.
	History   []llm.Message
	Truncated bool
	// Switch is the switch's journal row, zero when nothing was hidden. Once the session has
	// moved its responses' active path it calls FinishSwitch.
	Switch store.AgentBranchSwitch
}

// Rewind quiesces the conversation and rewinds its channel to target, for a regenerate or an
// edit. The caller holds the session's command lock, so no command begins meanwhile, and has
// stopped the agent's reply: the reply being written is finished as interrupted, every queued
// write is written, the messages the rewind hides are copied and the switch journaled before
// Chat is asked, and only once Chat has hidden them are their queued writes dropped.
func (c *Conversation) Rewind(ctx context.Context, target RewindTarget) (RewindOutcome, error) {
	if target.Kind != store.SwitchRegenerate && target.Kind != store.SwitchEdit {
		return RewindOutcome{}, stack.Wrap(fmt.Errorf("conversation: rewind: kind %q: want %s or %s", target.Kind, store.SwitchRegenerate, store.SwitchEdit))
	}
	return c.switchTo(ctx, target, nil)
}

// Activate brings back the branch whose copies chosen are, oldest first: as Rewind, it
// rewinds the channel to target, copying the branch active until then, and then sends chosen
// again under new ids, each derived from the switch and the message it copies so a switch
// finished by a later open sends none twice. A person's message is sent under the request id
// its reply names, the reply's question_id follows it, and turn_id, response_id and source
// are kept; each carries custom.resent_from and is sent with skip_push. More than MaxResent
// messages are refused before anything is written.
func (c *Conversation) Activate(ctx context.Context, target RewindTarget, chosen []store.AgentBranchMessage) (RewindOutcome, error) {
	if len(chosen) > MaxResent {
		return RewindOutcome{}, stack.Wrap(ErrBranchTooLong)
	}
	target.Kind = store.SwitchActivate
	return c.switchTo(ctx, target, chosen)
}

// FinishSwitch marks a switch done, once the session has moved its responses' active path.
func (c *Conversation) FinishSwitch(ctx context.Context, switchID string) error {
	c.service.mu.Lock()
	branches := c.service.branches
	c.service.mu.Unlock()
	if branches == nil {
		return stack.Wrap(errors.New("conversation: branch switches are not kept anywhere"))
	}
	return branches.AdvanceBranchSwitch(ctx, c.data.Customer, switchID, store.SwitchDone)
}

// hide drops the queued writes of ids and keeps them out of every read from now on.
func (c *Conversation) hide(ids map[string]bool) {
	c.sendMu.Lock()
	defer c.sendMu.Unlock()
	c.mu.Lock()
	defer c.mu.Unlock()
	maps.Copy(c.hidden, ids)
	c.dropLocked(ids)
}

// switcherFor is what switches the conversation's branches, or why it cannot.
func (c *Conversation) switcherFor(ctx context.Context, target RewindTarget) (switcher, error) {
	c.service.mu.Lock()
	branches, hooks, logger := c.service.branches, c.service.hooks, c.service.logger
	c.service.mu.Unlock()
	c.mu.Lock()
	active, thread, dispatched, own, shared := c.active, c.thread, c.dispatched, c.sessionCommand, c.shared
	session, customer, agentID, cid := c.session, c.data.Customer, c.data.Agent, c.data.CID
	c.mu.Unlock()
	switch {
	case !active:
		return switcher{}, stack.Wrap(errors.New("conversation is not open"))
	case thread:
		return switcher{}, stack.Wrap(&BranchesUnsupportedError{Reason: UnsupportedThread})
	case dispatched:
		return switcher{}, stack.Wrap(&BranchesUnsupportedError{Reason: UnsupportedDispatched})
	case !own:
		return switcher{}, stack.Wrap(&BranchesUnsupportedError{Reason: UnsupportedChannel})
	case shared:
		return switcher{}, stack.Wrap(&BranchesUnsupportedError{Reason: UnsupportedShared})
	case branches == nil:
		return switcher{}, stack.Wrap(errors.New("conversation: branch switches are not kept anywhere"))
	case session == "":
		return switcher{}, stack.Wrap(errors.New("conversation: opened without a session to keep its branches under"))
	case !validCommandID.MatchString(target.SwitchID) || target.MessageID == "" || len(target.MessageID) > 128:
		return switcher{}, stack.Wrap(errors.New("conversation: invalid rewind target"))
	}
	client, err := c.chat(ctx)
	if err != nil {
		return switcher{}, err
	}
	return switcher{branches: branches, hooks: hooks, logger: logger, rewinder: NewRewinder(client), client: client,
		customer: customer, session: session, cid: cid, agent: agentID}, nil
}

func (c *Conversation) switchTo(ctx context.Context, target RewindTarget, chosen []store.AgentBranchMessage) (RewindOutcome, error) {
	sw, err := c.switcherFor(ctx, target)
	if err != nil {
		return RewindOutcome{}, err
	}
	// Nothing is written to the channel from here until the switch is over, so the newest
	// message read is still the newest when Chat is asked to rewind.
	c.sendMu.Lock()
	defer c.sendMu.Unlock()
	c.mu.Lock()
	c.finish("interrupted")
	parkedUntil := c.parkedUntil
	c.mu.Unlock()
	if time.Now().Before(parkedUntil) {
		return RewindOutcome{}, stack.Wrap(ErrConversationUnwritable)
	}
	// A conversation that switches branches is never on a thread channel, so none of these
	// writes is a finished reply to hand on (OnFinishedReply).
	for {
		more, _, err := c.flushOne()
		if err != nil {
			return RewindOutcome{}, stack.Wrap(&rewindRefusal{sentinel: ErrConversationUnwritable, cause: err})
		}
		if !more {
			break
		}
	}

	unfinished, err := sw.branches.UnfinishedBranchSwitches(ctx, sw.customer, sw.session)
	if err != nil {
		return RewindOutcome{}, err
	}
	for _, row := range unfinished {
		if row.ID != target.SwitchID {
			continue
		}
		outcome, carried, err := c.finishTried(ctx, sw, row, chosen)
		if refusal := branchRefusal(err); refusal != nil {
			return RewindOutcome{}, stack.Wrap(refusal)
		}
		if err != nil || carried {
			return outcome, err
		}
		// The earlier try never rewound the channel and was forgotten: this one starts afresh.
	}

	r, err := sw.rewind(ctx, target, func(before string, tail map[string]bool) (Page, error) {
		return c.readUpTo(ctx, sw.client, before, tail)
	})
	if err != nil {
		return RewindOutcome{}, err
	}
	hidden := map[string]bool{}
	for _, id := range r.hidden {
		hidden[id] = true
	}
	c.mu.Lock()
	maps.Copy(c.hidden, hidden)
	c.dropLocked(hidden)
	c.mu.Unlock()
	outcome := RewindOutcome{Rewound: r.rewound, Hidden: r.hidden, Page: r.page, Switch: r.row}
	if target.Kind == store.SwitchActivate {
		outcome.Switch.State = store.SwitchResending
		resent, err := sw.resend(ctx, r.row, chosen)
		if err != nil {
			return RewindOutcome{}, err
		}
		c.resent(resent)
		outcome.Resent = resent
		outcome.Page.Messages = append(outcome.Page.Messages, resent...)
	}
	outcome.History, outcome.Truncated = restored(ctx, c.data.CID, outcome.Page)
	return outcome, nil
}

// SettleSwitch finishes the switch target names after Rewind or Activate failed with an error
// that leaves it unknown whether Chat rewound: Chat is read again, a switch whose rewind
// landed is carried on as a retry of it would be, and one whose rewind never landed is
// forgotten. carried says which; nothing journaled under the id is not carried either. The
// outcome is the conversation as Chat shows it now, as Rewind's.
func (c *Conversation) SettleSwitch(ctx context.Context, target RewindTarget, chosen []store.AgentBranchMessage) (RewindOutcome, bool, error) {
	sw, err := c.switcherFor(ctx, target)
	if err != nil {
		return RewindOutcome{}, false, err
	}
	c.sendMu.Lock()
	defer c.sendMu.Unlock()
	unfinished, err := sw.branches.UnfinishedBranchSwitches(ctx, sw.customer, sw.session)
	if err != nil {
		return RewindOutcome{}, false, err
	}
	for _, row := range unfinished {
		if row.ID != target.SwitchID {
			continue
		}
		outcome, carried, err := c.finishTried(ctx, sw, row, chosen)
		if refusal := branchRefusal(err); refusal != nil {
			return RewindOutcome{}, false, stack.Wrap(refusal)
		}
		return outcome, carried, err
	}
	return RewindOutcome{}, false, nil
}

// Refused is whether err is a switch Chat or the conversation refused, which changed
// nothing: there is no switch left to settle.
func Refused(err error) bool {
	if _, ok := errors.AsType[*BranchesUnsupportedError](err); ok {
		return true
	}
	for _, refusal := range []error{ErrBranchLimit, ErrBranchConflict, ErrBranchUnavailable, ErrBranchTooLong, ErrConversationUnwritable} {
		if errors.Is(err, refusal) {
			return true
		}
	}
	return false
}

// finishTried finishes a switch tried before under the same id, from where it got to, and
// says whether it carried it on: not when its rewind never landed, which forgets it. The page
// is read as Chat shows it now, without what the switch hid.
func (c *Conversation) finishTried(ctx context.Context, sw switcher, row store.AgentBranchSwitch, chosen []store.AgentBranchMessage) (RewindOutcome, bool, error) {
	if row.Kind == store.SwitchActivate && chosen == nil {
		chosen = []store.AgentBranchMessage{}
	}
	finished, err := sw.resume(ctx, row, chosen, false)
	if err != nil {
		return RewindOutcome{}, false, err
	}
	if finished.state == "" {
		return RewindOutcome{}, false, nil
	}
	hidden := map[string]bool{}
	for _, id := range finished.hidden {
		hidden[id] = true
	}
	c.mu.Lock()
	maps.Copy(c.hidden, hidden)
	c.dropLocked(hidden)
	all := maps.Clone(c.hidden)
	c.mu.Unlock()
	c.resent(finished.resent)
	page, err := c.readUpTo(ctx, sw.client, "", all)
	if err != nil {
		return RewindOutcome{}, false, err
	}
	row.State = finished.state
	outcome := RewindOutcome{Rewound: len(finished.hidden) > 0, Hidden: finished.hidden, Resent: finished.resent, Page: page, Switch: row}
	outcome.History, outcome.Truncated = restored(ctx, c.data.CID, page)
	return outcome, true, nil
}

// readUpTo reads the conversation before the message before, or its newest page for none,
// leaving out what it hid and the messages in tail, which a read that does not stop at before
// may still show.
func (c *Conversation) readUpTo(ctx context.Context, client *getstream.Stream, before string, tail map[string]bool) (Page, error) {
	c.mu.Lock()
	hidden := maps.Clone(c.hidden)
	customer, agentID, cid, owner, voiceAgent := c.data.Customer, c.data.Agent, c.data.CID, c.data.Owner, c.voiceAgent
	c.mu.Unlock()
	maps.Copy(hidden, tail)
	page, err := c.service.readPage(ctx, client, customer, agentID, cid, before, owner, nil, hidden, nil, voiceAgent)
	if err != nil {
		return Page{}, stack.Wrap(err)
	}
	return page, nil
}

// resent takes the commands of messages sent again into the ledger, so a retry of one is
// known as it is after a reopen.
func (c *Conversation) resent(messages []Message) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.data.Commands == nil {
		c.data.Commands = map[string]commandRecord{}
	}
	for id, record := range commandsIn(messages) {
		if _, known := c.data.Commands[id]; !known {
			c.data.Commands[id] = record
		}
	}
}

// resumeSwitches finishes the switches of session an earlier router left unfinished, and
// returns every message the session's switches hid. A switch it cannot finish now is left for
// the next open, and the conversation opens as Chat shows it.
func (s *Service) resumeSwitches(ctx context.Context, client *getstream.Stream, customer, agentID, cid, session string) map[string]bool {
	s.mu.Lock()
	branches, hooks, logger := s.branches, s.hooks, s.logger
	s.mu.Unlock()
	if branches == nil || session == "" {
		return nil
	}
	sw := switcher{branches: branches, hooks: hooks, logger: logger, rewinder: NewRewinder(client), client: client,
		customer: customer, session: session, cid: cid, agent: agentID}
	hidden, err := sw.resumeAll(ctx)
	if err != nil {
		logger.Warn("could not finish a branch switch an earlier router left unfinished; the conversation opens as Chat shows it",
			"customer", customer, "cid", cid, "session", session, "error", err)
	}
	return hidden
}

// switcher switches one session's conversation between branches in Chat.
type switcher struct {
	branches Branches
	hooks    SwitchHooks
	logger   *slog.Logger
	rewinder Rewinder
	client   *getstream.Stream
	// The conversation's customer, session, channel and agent.
	customer, session, cid, agent string
}

// rewound is what a rewind did.
type rewound struct {
	row     store.AgentBranchSwitch
	hidden  []string
	page    Page
	rewound bool
}

// rewind copies what a rewind to target hides, journals the switch and has Chat make it,
// reading the conversation up to the point with read first. A rewind Chat refuses forgets
// the switch and its copies, which are of messages still visible.
func (sw switcher) rewind(ctx context.Context, target RewindTarget, read func(before string, tail map[string]bool) (Page, error)) (rewound, error) {
	channelType, id := channelOf(sw.cid)
	tail, hidden, err := sw.tailFrom(ctx, target)
	if err != nil {
		return rewound{}, err
	}
	before := target.MessageID
	if !target.Inclusive && len(hidden) > 0 {
		before = hidden[0].ID
	} else if !target.Inclusive {
		before = ""
	}
	page, err := read(before, idsOf(hidden))
	if err != nil {
		return rewound{}, err
	}
	if len(hidden) == 0 && target.Kind != store.SwitchActivate {
		// Nothing after the question: a reply that never landed is asked for again.
		return rewound{page: page}, nil
	}

	row := store.AgentBranchSwitch{ID: target.SwitchID, SessionID: sw.session, CustomerID: sw.customer, Kind: target.Kind,
		TargetResponseID: target.TargetResponseID, ForkMessageID: target.MessageID, Inclusive: target.Inclusive,
		HiddenFrom: createdAt(tail.Messages[0]), State: store.SwitchSnapshotted}
	if err := sw.branches.StartBranchSwitch(ctx, &row); err != nil {
		return rewound{}, err
	}
	if err := sw.branches.SaveBranchMessages(ctx, copiesOf(row, hidden, target.TurnResponses)); err != nil {
		return rewound{}, err
	}
	if len(hidden) > 0 {
		for attempt := 0; ; attempt++ {
			_, err = sw.rewinder.Rewind(ctx, channelType, id, target.MessageID, target.Inclusive, tail.LastID, sw.agent)
			var conflict *RewindConflictError
			if !errors.As(err, &conflict) {
				break
			}
			// A rewind that already landed, ours retried or another to the same point.
			if !target.Inclusive && conflict.LastMessageID == target.MessageID {
				err = nil
				break
			}
			if attempt > 0 {
				break
			}
			// A message was written since the read, by another device or another router: copy
			// the tail again and try once more.
			if tail, hidden, err = sw.tailFrom(ctx, target); err != nil {
				break
			}
			if err = sw.branches.SaveBranchMessages(ctx, copiesOf(row, hidden, target.TurnResponses)); err != nil {
				return rewound{}, err
			}
		}
		if errors.Is(err, ErrNothingToRewind) {
			err = nil
			hidden = nil
		}
		if refusal := branchRefusal(err); refusal != nil {
			if abandonErr := sw.branches.AbandonBranchSwitch(ctx, sw.customer, sw.session, row.ID); abandonErr != nil {
				// The next open finds the rewind never landed and forgets the switch then.
				return rewound{}, errors.Join(stack.Wrap(refusal), abandonErr)
			}
			return rewound{}, stack.Wrap(refusal)
		}
		if err != nil {
			// Whether Chat rewound is not known: the next open finds out from the channel.
			return rewound{}, err
		}
	}
	if err := sw.branches.AdvanceBranchSwitch(ctx, sw.customer, row.ID, store.SwitchRewound); err != nil {
		return rewound{}, err
	}
	row.State = store.SwitchRewound
	ids := make([]string, 0, len(hidden))
	for _, m := range hidden {
		ids = append(ids, m.ID)
	}
	return rewound{row: row, hidden: ids, page: page, rewound: len(ids) > 0}, nil
}

// tailFrom reads the channel from target's message to the end, and what a rewind to it
// hides. A message no longer there is a conversation that changed under the switch.
func (sw switcher) tailFrom(ctx context.Context, target RewindTarget) (Tail, []getstream.MessageResponse, error) {
	channelType, id := channelOf(sw.cid)
	tail, err := sw.rewinder.ReadFrom(ctx, channelType, id, target.MessageID, true)
	if refused, ok := errors.AsType[*getstream.StreamError](err); ok && refused.StatusCode == http.StatusBadRequest {
		// Chat refuses a read from a message it does not have, or that a rewind hid, with an
		// input error.
		return Tail{}, nil, stack.Wrap(&rewindRefusal{sentinel: ErrBranchConflict, cause: err})
	}
	if err != nil {
		return Tail{}, nil, err
	}
	if len(tail.Messages) == 0 || tail.Messages[0].ID != target.MessageID {
		return Tail{}, nil, stack.Wrap(&rewindRefusal{sentinel: ErrBranchConflict, cause: errors.New("the message to rewind to is no longer in the conversation")})
	}
	if target.Inclusive {
		return tail, tail.Messages, nil
	}
	return tail, tail.Messages[1:], nil
}

// branchRefusal is the switch refusal err is, nil for an error that is none: one that leaves
// it unknown whether Chat rewound.
func branchRefusal(err error) error {
	switch {
	case err == nil:
		return nil
	case errors.Is(err, ErrTooManyRewinds):
		return &rewindRefusal{sentinel: ErrBranchLimit, cause: err}
	case errors.Is(err, ErrRewindUnavailable):
		return &rewindRefusal{sentinel: ErrBranchUnavailable, cause: err}
	case errors.Is(err, ErrRewindConflict), errors.Is(err, ErrRewindAmbiguous), errors.Is(err, ErrForkInProgress), errors.Is(err, ErrBranchConflict):
		return &rewindRefusal{sentinel: ErrBranchConflict, cause: err}
	}
	// Chat answers a message to rewind to that another rewind hid with a plain input error.
	if refused, ok := errors.AsType[*getstream.StreamError](err); ok && refused.StatusCode == http.StatusBadRequest {
		return &rewindRefusal{sentinel: ErrBranchConflict, cause: err}
	}
	return nil
}

// finished is how far resume took a switch: its state, "" for a switch forgotten because its
// rewind never landed, the messages it hid and those it sent again.
type finished struct {
	state  string
	hidden []string
	resent []Message
}

// resumeAll finishes the session's unfinished switches, oldest first, and returns every
// message its switches hid, but those of a switch whose rewind is not known to have landed.
func (sw switcher) resumeAll(ctx context.Context) (map[string]bool, error) {
	unfinished, err := sw.branches.UnfinishedBranchSwitches(ctx, sw.customer, sw.session)
	if err != nil {
		return nil, err
	}
	var errs []error
	unknown := map[string]bool{}
	for _, row := range unfinished {
		got, err := sw.resume(ctx, row, nil, true)
		if err != nil {
			errs = append(errs, err)
		}
		if got.state == store.SwitchSnapshotted {
			unknown[row.ID] = true
		}
	}
	copies, err := sw.branches.BranchMessages(ctx, sw.customer, sw.session, nil)
	if err != nil {
		return nil, errors.Join(append(errs, err)...)
	}
	hidden := map[string]bool{}
	for _, m := range copies {
		if !unknown[m.SwitchID] {
			hidden[m.OriginalMessageID] = true
		}
	}
	return hidden, errors.Join(errs...)
}

// resume finishes a switch from where it got to. A switch still snapshotted is either rewound
// or not, which the channel says: its rewind landed when none of the messages it copied is
// visible, and is then carried on; otherwise it never landed, and the switch is forgotten.
// chosen, when nil, is asked of the hooks. With done, the session is told the switch is over
// (SwitchHooks.Switched) and it is marked done.
func (sw switcher) resume(ctx context.Context, row store.AgentBranchSwitch, chosen []store.AgentBranchMessage, done bool) (finished, error) {
	got := finished{state: row.State}
	copies, err := sw.branches.BranchMessages(ctx, sw.customer, sw.session, nil)
	if err != nil {
		return got, err
	}
	for _, m := range copies {
		if m.SwitchID == row.ID {
			got.hidden = append(got.hidden, m.OriginalMessageID)
		}
	}
	if row.State == store.SwitchSnapshotted {
		landed, err := sw.landed(ctx, row, got.hidden)
		if err != nil {
			return got, err
		}
		if !landed {
			if err := sw.branches.AbandonBranchSwitch(ctx, sw.customer, sw.session, row.ID); err != nil {
				return got, err
			}
			return finished{}, nil
		}
		if err := sw.branches.AdvanceBranchSwitch(ctx, sw.customer, row.ID, store.SwitchRewound); err != nil {
			return got, err
		}
		row.State, got.state = store.SwitchRewound, store.SwitchRewound
	}
	if row.Kind == store.SwitchActivate {
		if chosen == nil {
			if sw.hooks.Chosen == nil {
				return got, stack.Wrap(errors.New("conversation: no way to find the branch an activation sends again"))
			}
			if chosen, err = sw.hooks.Chosen(ctx, row); err != nil {
				return got, err
			}
		}
		if got.resent, err = sw.resend(ctx, row, chosen); err != nil {
			return got, err
		}
		got.state = store.SwitchResending
	}
	if !done {
		return got, nil
	}
	if sw.hooks.Switched != nil {
		if err := sw.hooks.Switched(ctx, row); err != nil {
			return got, err
		}
	}
	if err := sw.branches.AdvanceBranchSwitch(ctx, sw.customer, row.ID, store.SwitchDone); err != nil {
		return got, err
	}
	got.state = store.SwitchDone
	return got, nil
}

// landed is whether a journaled switch's rewind landed: none of the messages it copied is
// visible. One with no copies never asked Chat. The channel is read from its newest message
// back, not from the message rewound to: an inclusive rewind hides that one too, and Chat
// refuses a read from a message a rewind hid.
func (sw switcher) landed(ctx context.Context, row store.AgentBranchSwitch, copied []string) (bool, error) {
	if len(copied) == 0 {
		return row.Kind == store.SwitchActivate && row.Resent > 0, nil
	}
	wanted := make(map[string]bool, len(copied))
	for _, copiedID := range copied {
		wanted[copiedID] = true
	}
	shown := false
	channelType, id := channelOf(sw.cid)
	err := scanChannel(ctx, chatRewinder{client: sw.client}, channelType, id, func(m getstream.MessageResponse) bool {
		// Every message the switch copied was created at or after the one rewound to.
		if m.CreatedAt.Time != nil && createdAt(m).Before(row.HiddenFrom) {
			return true
		}
		if wanted[m.ID] && m.DeletedAt == nil && m.Type != "deleted" {
			shown = true
			return true
		}
		return false
	})
	if err != nil {
		return false, err
	}
	return !shown, nil
}

// resend sends chosen again, from the first the switch has not counted as sent.
func (sw switcher) resend(ctx context.Context, row store.AgentBranchSwitch, chosen []store.AgentBranchMessage) ([]Message, error) {
	if len(chosen) > MaxResent {
		return nil, stack.Wrap(ErrBranchTooLong)
	}
	if err := sw.branches.AdvanceBranchSwitch(ctx, sw.customer, row.ID, store.SwitchResending); err != nil {
		return nil, err
	}
	remap := make(map[string]string, len(chosen))
	for _, m := range chosen {
		remap[m.OriginalMessageID] = ResentID(row.ID, m.OriginalMessageID)
	}
	channelType, id := channelOf(sw.cid)
	skip := true
	var out []Message
	for i, m := range chosen {
		req := resentMessage(m, remap)
		if i >= row.Resent {
			_, err := sw.client.Chat().SendMessage(ctx, channelType, id, &getstream.SendMessageRequest{Message: req, SkipPush: &skip})
			if err != nil && !sw.alreadyResent(ctx, *req.ID, m.OriginalMessageID) {
				return out, stack.Wrap(err)
			}
			if _, err := sw.branches.CountBranchResent(ctx, sw.customer, row.ID); err != nil {
				return out, err
			}
		}
		if sent, ok := resentRead(req); ok {
			out = append(out, sent)
		}
	}
	return out, nil
}

// alreadyResent is whether the message id is in Chat as the copy of original: a switch that
// stopped between sending it and counting it, finished again.
func (sw switcher) alreadyResent(ctx context.Context, id, original string) bool {
	existing, err := sw.client.Chat().GetMessage(ctx, id, &getstream.GetMessageRequest{})
	return err == nil && existing.Data.Message.Custom[ResentFromField] == original
}

// ResentFromField names the message a message sent again copies, so a customer's backend can
// tell the copy from a message somebody wrote.
const ResentFromField = "resent_from"

// ResentID is the id the message original is sent again under by the switch switchID. It is
// the same every time, so a switch finished again finds what it sent.
func ResentID(switchID, original string) string {
	sum := sha256.Sum256([]byte(switchID + "\x00" + original))
	return "rs" + hex.EncodeToString(sum[:])[:30]
}

// heldMessage is what a copy keeps of a Chat message (store.AgentBranchMessage.Message).
type heldMessage struct {
	Text        string                 `json:"text"`
	Attachments []getstream.Attachment `json:"attachments,omitempty"`
	Custom      map[string]any         `json:"custom"`
}

// copiesOf are the copies of the messages a switch hides. A message is listed under its
// response (BranchField); one written before messages carried it, under the response of its
// reply's turn, and a person's message under that of the reply that answers it.
func copiesOf(row store.AgentBranchSwitch, messages []getstream.MessageResponse, turnResponses map[string]string) []store.AgentBranchMessage {
	responses := map[string]string{}
	answering := map[string]string{}
	for _, m := range messages {
		runtime, _ := m.Custom["support_runtime"].(map[string]any)
		if b := branchFrom(m.Custom[BranchField]); b != nil {
			responses[m.ID] = b.ResponseID
		} else if turn, _ := runtime["turn_id"].(string); turn != "" && turnResponses[turn] != "" {
			responses[m.ID] = turnResponses[turn]
		}
		if question, _ := runtime["question_id"].(string); question != "" && runtime["role"] == "assistant" && responses[m.ID] != "" {
			answering[question] = responses[m.ID]
		}
	}
	copies := make([]store.AgentBranchMessage, 0, len(messages))
	for _, m := range messages {
		response := responses[m.ID]
		if response == "" {
			response = answering[m.ID]
		}
		held, _ := toMap(heldMessage{Text: m.Text, Attachments: m.Attachments, Custom: m.Custom})
		copies = append(copies, store.AgentBranchMessage{
			SessionID: row.SessionID, OriginalMessageID: m.ID, CustomerID: row.CustomerID, ResponseID: response,
			SwitchID: row.ID, ChatCreatedAt: createdAt(m), AuthorID: m.User.ID, Message: held,
		})
	}
	return copies
}

// resentMessage is the copy m sent again: under its new id, with the request and question ids
// of the branch's messages following theirs, and custom.resent_from naming it.
func resentMessage(m store.AgentBranchMessage, remap map[string]string) getstream.MessageRequest {
	var held heldMessage
	// Through JSON, so the copy the caller holds is not written under it.
	encoded, _ := json.Marshal(m.Message)
	_ = json.Unmarshal(encoded, &held)
	if held.Custom == nil {
		held.Custom = map[string]any{}
	}
	for _, key := range []string{"support_runtime", "support_message"} {
		fields, ok := held.Custom[key].(map[string]any)
		if !ok {
			continue
		}
		for _, field := range []string{"command_id", "question_id"} {
			if was, _ := fields[field].(string); remap[was] != "" {
				fields[field] = remap[was]
			}
		}
	}
	held.Custom[ResentFromField] = m.OriginalMessageID
	id, user := remap[m.OriginalMessageID], m.AuthorID
	return getstream.MessageRequest{ID: &id, UserID: &user, Text: &held.Text, Custom: held.Custom, Attachments: held.Attachments}
}

// resentRead is a message sent again as a read gives it back, for one the router wrote.
func resentRead(req getstream.MessageRequest) (Message, bool) {
	m, err := messageFromWire(*req.ID, *req.Text, req.Custom)
	if err != nil {
		return Message{}, false
	}
	m.Artifacts = artifactsFromAttachments(req.Attachments)
	m.Files = filesFromAttachments(req.Attachments)
	m.ConnectorAuthorizations = connectorAuthorizationsFromAttachments(req.Attachments)
	m.Refused = req.Custom[RefusedField] == true
	m.authorID = *req.UserID
	return m, true
}

func idsOf(messages []getstream.MessageResponse) map[string]bool {
	ids := make(map[string]bool, len(messages))
	for _, m := range messages {
		ids[m.ID] = true
	}
	return ids
}

func createdAt(m getstream.MessageResponse) time.Time {
	if m.CreatedAt.Time == nil {
		return time.Time{}
	}
	return m.CreatedAt.Time.UTC()
}

func toMap(v any) (map[string]any, error) {
	encoded, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var out map[string]any
	return out, json.Unmarshal(encoded, &out)
}

// ResponseMessages are where a response's messages are in the channel as it shows now.
type ResponseMessages struct {
	// Question is the person's message the response answers, Reply the response's own. Either
	// is empty when the channel does not show it: a reply that never landed has none.
	Question, Reply string
}

// maxScanPages bounds how far back a response's messages are looked for: Chat keeps at most
// 10,000 messages a reopen or a rewind reads anyway.
const maxScanPages = 100

// ResponseMessages finds the messages of responseID in the channel, newest first, by the
// response id they carry (BranchField). A regenerated reply answers a question another
// response asked, questionResponseID, whose message carries that one's id.
func (c *Conversation) ResponseMessages(ctx context.Context, responseID, questionResponseID string) (ResponseMessages, error) {
	var found ResponseMessages
	asking := ""
	err := c.scanBack(ctx, func(m getstream.MessageResponse) bool {
		b := branchFrom(m.Custom[BranchField])
		runtime, _ := m.Custom["support_runtime"].(map[string]any)
		role, _ := runtime["role"].(string)
		switch {
		case found.Reply == "" && role == "assistant" && b != nil && b.ResponseID == responseID:
			found.Reply = m.ID
			asking, _ = runtime["question_id"].(string)
		case role == "user" && (asking != "" && m.ID == asking ||
			asking == "" && b != nil && (b.ResponseID == responseID || b.ResponseID == questionResponseID)):
			found.Question = m.ID
			// A reply is after its question, so nothing older is the response's.
			return true
		}
		return false
	})
	return found, err
}

// AfterResponse is what the channel shows after a response, as it is now.
type AfterResponse struct {
	// Question is the person's message asked after the response, and QuestionResponseID the
	// response that asked it (BranchField). Both are empty when none is shown.
	Question, QuestionResponseID string
	// Newest is the newest message the channel shows, empty for an empty channel.
	Newest string
}

// QuestionAfter finds the newest person's message the channel shows that was asked after
// parentResponseID, empty for the conversation's first, by the marks its messages carry.
func (c *Conversation) QuestionAfter(ctx context.Context, parentResponseID string) (AfterResponse, error) {
	var found AfterResponse
	err := c.scanBack(ctx, func(m getstream.MessageResponse) bool {
		if found.Newest == "" {
			found.Newest = m.ID
		}
		b := branchFrom(m.Custom[BranchField])
		runtime, _ := m.Custom["support_runtime"].(map[string]any)
		if b == nil || runtime["role"] != "user" || b.ParentResponseID != parentResponseID {
			return false
		}
		found.Question, found.QuestionResponseID = m.ID, b.ResponseID
		return true
	})
	return found, err
}

// scanBack hands the messages the channel shows to visit, newest first, until it answers
// true or the channel's oldest page is read.
func (c *Conversation) scanBack(ctx context.Context, visit func(getstream.MessageResponse) bool) error {
	client, err := c.chat(ctx)
	if err != nil {
		return err
	}
	c.mu.Lock()
	hidden := maps.Clone(c.hidden)
	cid := c.data.CID
	c.mu.Unlock()
	channelType, id := channelOf(cid)
	return scanChannel(ctx, chatRewinder{client: client}, channelType, id, func(m getstream.MessageResponse) bool {
		if hidden[m.ID] || m.DeletedAt != nil || m.Type == "deleted" {
			return false
		}
		return visit(m)
	})
}

// scanChannel hands every message a read of the channel returns to visit, newest first, until
// it answers true or the channel's oldest page is read. Each page is read before the oldest
// message of the one before, which Chat showed, so no read names a message a rewind hid.
func scanChannel(ctx context.Context, reader chatRewinder, channelType, id string, visit func(getstream.MessageResponse) bool) error {
	limit := tailPage
	params := &getstream.MessagePaginationParams{Limit: &limit}
	for range maxScanPages {
		page, err := reader.page(ctx, channelType, id, params)
		if err != nil {
			return err
		}
		for i := len(page) - 1; i >= 0; i-- {
			if visit(page[i]) {
				return nil
			}
		}
		if len(page) < tailPage {
			return nil
		}
		oldest := page[0].ID
		if params.IDLt != nil && *params.IDLt == oldest {
			return nil
		}
		params = &getstream.MessagePaginationParams{Limit: &limit, IDLt: &oldest}
	}
	return nil
}

// HistoryOf is p as a model is handed it, the way Open hands the conversation back: tool
// rounds come back when the ctx it was opened with asks for them (WithRestore).
func (c *Conversation) HistoryOf(ctx context.Context, p Page) ([]llm.Message, bool) {
	return restored(ctx, c.data.CID, p)
}

// Remarked is chosen with the marks of the messages named in marks, by original message id,
// replaced: a branch sent again shows where each of its messages stands among their
// alternatives now, not where they stood when they were hidden. A mark with no Of is 1 of 1.
func Remarked(chosen []store.AgentBranchMessage, marks map[string]Branch) []store.AgentBranchMessage {
	out := make([]store.AgentBranchMessage, 0, len(chosen))
	for _, m := range chosen {
		b, ok := marks[m.OriginalMessageID]
		if !ok {
			out = append(out, m)
			continue
		}
		var held heldMessage
		// Through JSON, so the copy the caller holds is not written under it.
		encoded, _ := json.Marshal(m.Message)
		_ = json.Unmarshal(encoded, &held)
		if held.Custom == nil {
			held.Custom = map[string]any{}
		}
		mark, _ := toMap(b)
		held.Custom[BranchField] = mark
		m.Message, _ = toMap(held)
		out = append(out, m)
	}
	return out
}
