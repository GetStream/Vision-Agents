package session

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ErrBranchGone is a branch whose copies are no longer kept: its messages cannot be sent
// again, so it cannot be brought back.
var ErrBranchGone = errors.New("session: this branch is no longer kept")

// rangesWarned is how many hidden ranges a conversation may reach before each switch logs a
// warning: Chat refuses a rewind past 100.
const rangesWarned = 80

// mergedRanges is how many hidden ranges Chat keeps for a conversation's channel after each
// switch, which Chat caps at 100. kind is the switch's: regenerate, edit or activate.
var mergedRanges = promauto.NewHistogramVec(prometheus.HistogramOpts{
	Namespace: "router",
	Name:      "branch_merged_ranges",
	Help:      "Hidden ranges Stream Chat keeps for a conversation after a branch switch, merged as Chat merges them (store.MergedRangeCount), by the switch's kind; Chat refuses a rewind past 100",
	Buckets:   []float64{1, 2, 5, 10, 20, 40, 60, 80, 90, 100},
}, []string{"kind"})

// RewoundTo is where a switch rewound the conversation's channel to: the message, hidden too
// when Inclusive.
type RewoundTo struct {
	MessageID string
	Inclusive bool
}

// Branched is what a regenerate or an edit began.
type Branched struct {
	persistent.CommandReceipt
	// ResponseID is the response answering, empty for a retry of a request already taken,
	// which is answered with its receipt, and for a model that names no turn.
	ResponseID string
	// ParentResponseID is the response it is answered after, the replaced one's parent, and
	// QuestionResponseID the response whose question it answers: the replaced one's for a
	// regenerate, its own for an edit.
	ParentResponseID, QuestionResponseID string
	// Of is what Index and Count count, store.SiblingsOfReply for a regenerate and
	// store.SiblingsOfQuestion for an edit: where the new response stands among them, newest.
	Of           string
	Index, Count int
	// RewoundTo is where the channel was rewound to, and Rewound whether Chat hid anything:
	// not when the reply regenerated never landed.
	RewoundTo RewoundTo
	Rewound   bool
}

// Activated is what an activation did.
type Activated struct {
	// LeafResponseID is the newest response of the branch brought back, which the next
	// question is answered after.
	LeafResponseID string
	// AlreadyActive is a branch already on the conversation, which nothing changed.
	AlreadyActive bool
	// Resent is how many messages were sent again.
	Resent    int
	RewoundTo RewoundTo
}

// Regenerate answers the question of response responseID again, as a new response beside it,
// under the request id: the reply and everything after it are hidden in Chat and kept as a
// branch. The replaced response must be on the conversation. A retry of a request already
// taken is answered with its receipt. The files sent with the question are kept under the new
// response as well.
func (s *Session) Regenerate(ctx context.Context, id, responseID string, asked options.LLM) (Branched, error) {
	return s.branch(ctx, id, responseID, "", "", false, asked)
}

// Edit asks text in place of the question of response responseID, as a new question beside
// it, under the request id: the question, its reply and everything after them are hidden in
// Chat and kept as a branch. clientID names the install it came from, as RespondCommand's.
// The files sent with the replaced question are not carried to the new one: they stay with
// the branch it leaves, and come back when that branch is activated.
func (s *Session) Edit(ctx context.Context, id, responseID, text, clientID string, asked options.LLM) (Branched, error) {
	return s.branch(ctx, id, responseID, text, clientID, true, asked)
}

// Activate brings back the branch of response responseID: the branch on the conversation
// from where they part is hidden and kept, and the one ending at the newest response under
// responseID is sent again. The model is answered from it from then on. A branch already on
// the conversation is left as it is.
func (s *Session) Activate(ctx context.Context, responseID string) (Activated, error) {
	s.commandMu.Lock()
	defer s.commandMu.Unlock()
	if err := s.refuseBranches(ctx); err != nil {
		return Activated{}, err
	}
	customer := s.spec.CustomerID
	target, err := s.tree.SessionResponse(ctx, customer, s.id, responseID)
	if err != nil {
		return Activated{}, err
	}
	if target.QuestionResponseID == "" {
		return Activated{}, stack.Wrap(&persistent.BranchesUnsupportedError{Reason: persistent.UnsupportedUnmarked})
	}
	if target.RewoundAt == nil {
		s.turnsMu.Lock()
		leaf := s.leaf
		s.turnsMu.Unlock()
		return Activated{LeafResponseID: leaf, AlreadyActive: true}, nil
	}
	top, tail, err := branchOf(ctx, s.tree, target)
	if err != nil {
		return Activated{}, err
	}

	s.current().Interrupt()
	if err := s.records.Flush(ctx); err != nil {
		return Activated{}, stack.Wrap(err)
	}
	after, err := s.persisted.QuestionAfter(ctx, top.ParentResponseID)
	if err != nil {
		return Activated{}, err
	}
	rewind := persistent.RewindTarget{TargetResponseID: target.ID}
	kept := false
	switch {
	case after.Question != "" && after.QuestionResponseID == top.QuestionResponseID:
		// Another answer to the same question is shown: it goes, the question stays.
		rewind.MessageID, kept = after.Question, true
	case after.Question != "":
		// Another question is shown in its place: it goes with everything after it.
		rewind.MessageID, rewind.Inclusive = after.Question, true
	case after.Newest != "":
		// Nothing is shown after the response's parent, as a question asked again whose reply
		// never started leaves it: nothing is hidden, and the branch is sent after the end.
		rewind.MessageID = after.Newest
	default:
		return Activated{}, stack.Wrap(persistent.ErrBranchConflict)
	}
	rewind.SwitchID, err = s.activationID(ctx, target.ID, kept)
	if err != nil {
		return Activated{}, err
	}
	chosen, err := chosenFor(ctx, s.tree, top, tail, kept)
	if err != nil {
		return Activated{}, err
	}
	reading, err := s.restoring(ctx)
	if err != nil {
		return Activated{}, err
	}
	outcome, err := s.persisted.Activate(reading, rewind, chosen)
	if err != nil {
		return Activated{}, err
	}
	leaf := tail[len(tail)-1].ID
	if err := s.tree.MoveActivePath(ctx, customer, s.id, leaf, time.Now().UTC()); err != nil {
		return Activated{}, err
	}
	if err := s.persisted.FinishSwitch(ctx, rewind.SwitchID); err != nil {
		return Activated{}, err
	}
	s.turnsMu.Lock()
	s.leaf = leaf
	s.turnsMu.Unlock()
	history, _ := s.persisted.HistoryOf(reading, outcome.Page)
	s.carryOnFrom(history)
	s.pinActivatedDocuments(ctx)
	s.countRanges(ctx, store.SwitchActivate)
	return Activated{LeafResponseID: leaf, Resent: len(outcome.Resent),
		RewoundTo: RewoundTo{MessageID: rewind.MessageID, Inclusive: rewind.Inclusive}}, nil
}

// BranchShown is what the branch of response responseID would show, read from tree without a
// running session: the copies of its messages from where it leaves the conversation down to
// the newest response under responseID, oldest first, each marked with its place among its
// alternatives now, as an activation would send them again. A response on the conversation is
// what the channel shows, so active is set and nothing is returned.
func BranchShown(ctx context.Context, tree *store.Store, customerID, sessionID, responseID string) (active bool, shown []store.AgentBranchMessage, err error) {
	target, err := tree.SessionResponse(ctx, customerID, sessionID, responseID)
	if err != nil {
		return false, nil, err
	}
	if target.RewoundAt == nil {
		return true, nil, nil
	}
	if target.QuestionResponseID == "" {
		return false, nil, stack.Wrap(&persistent.BranchesUnsupportedError{Reason: persistent.UnsupportedUnmarked})
	}
	top, tail, err := branchOf(ctx, tree, target)
	if err != nil {
		return false, nil, err
	}
	shown, err = chosenFor(ctx, tree, top, tail, false)
	return false, shown, err
}

// branch is Regenerate, or Edit when edit is set.
func (s *Session) branch(ctx context.Context, id, responseID, text, clientID string, edit bool, asked options.LLM) (Branched, error) {
	s.commandMu.Lock()
	defer s.commandMu.Unlock()
	if err := s.refuseBranches(ctx); err != nil {
		return Branched{}, err
	}
	if err := s.current().CheckAsked(asked); err != nil {
		return Branched{}, stack.Wrap(err)
	}
	if taken, err := s.persisted.Command(id); err == nil {
		// A retry of a request already taken is answered as one: rewinding again would hide
		// the reply it began.
		if edit {
			receipt, err := s.persisted.BeginCommandWithTurn(persistent.Turn{}, id, text, clientID, nil)
			return Branched{CommandReceipt: receipt}, stack.Wrap(err)
		}
		// A regenerate answers a question another request asked; one that asked its own is
		// another request under the same id.
		if taken.UserMessageID == id {
			return Branched{}, stack.Wrap(persistent.ErrCommandConflict)
		}
		taken.Duplicate = true
		return Branched{CommandReceipt: taken}, nil
	}
	customer := s.spec.CustomerID
	replaced, err := s.tree.SessionResponse(ctx, customer, s.id, responseID)
	if err != nil {
		return Branched{}, err
	}
	if replaced.QuestionResponseID == "" {
		return Branched{}, stack.Wrap(&persistent.BranchesUnsupportedError{Reason: persistent.UnsupportedUnmarked})
	}
	if replaced.RewoundAt != nil {
		return Branched{}, stack.Wrap(fmt.Errorf("%w: response %s is not on the conversation; activate its branch first", persistent.ErrBranchConflict, responseID))
	}

	s.current().Interrupt()
	if err := s.records.Flush(ctx); err != nil {
		return Branched{}, stack.Wrap(err)
	}
	found, err := s.persisted.ResponseMessages(ctx, replaced.ID, replaced.QuestionResponseID)
	if err != nil {
		return Branched{}, err
	}
	if found.Question == "" {
		return Branched{}, stack.Wrap(&persistent.BranchesUnsupportedError{Reason: persistent.UnsupportedUnmarked})
	}
	turn, err := s.replacing(ctx, replaced, edit)
	if err != nil {
		return Branched{}, err
	}
	documents, err := s.branchDocuments(ctx, replaced, edit)
	if err != nil {
		return Branched{}, err
	}
	rewind := persistent.RewindTarget{SwitchID: switchID(s.id, id), Kind: store.SwitchRegenerate, TargetResponseID: replaced.ID,
		MessageID: found.Question, Inclusive: edit}
	if edit {
		rewind.Kind = store.SwitchEdit
	}
	reading, err := s.restoring(ctx)
	if err != nil {
		return Branched{}, err
	}
	outcome, err := s.persisted.Rewind(reading, rewind)
	if err != nil {
		return Branched{}, err
	}
	// The conversation ends at the replaced response's parent until the new one is written:
	// a router stopping before then opens it there (switchHooks).
	if err := s.tree.MoveActivePath(ctx, customer, s.id, replaced.ParentResponseID, time.Now().UTC()); err != nil {
		return Branched{}, err
	}
	if outcome.Switch.ID != "" {
		if err := s.persisted.FinishSwitch(ctx, outcome.Switch.ID); err != nil {
			return Branched{}, err
		}
	}
	s.turnsMu.Lock()
	s.leaf = replaced.ParentResponseID
	s.turnsMu.Unlock()
	s.countRanges(ctx, rewind.Kind)

	var receipt persistent.CommandReceipt
	if edit {
		receipt, err = s.persisted.BeginCommandWithTurn(turn, id, text, clientID, nil)
	} else {
		receipt, err = s.persisted.BeginRegenerate(turn, id, found.Question)
	}
	if err != nil {
		return Branched{}, stack.Wrap(err)
	}
	s.broadcast(receipt)

	history, _ := s.persisted.HistoryOf(reading, outcome.Page)
	prompt := text
	if !edit {
		// The question is asked of the model again, so it is not in the history it is handed.
		prompt = replaced.Said
		if n := len(history); n > 0 && history[n-1].Role == llm.User {
			prompt = history[n-1].Content
			history = history[:n-1]
		}
	}
	s.carryOnFrom(history)
	s.current().ReplaceDocuments(pinnable(documents)...)
	branched := Branched{CommandReceipt: receipt, ParentResponseID: turn.ParentResponseID, QuestionResponseID: turn.QuestionResponseID,
		RewoundTo: RewoundTo{MessageID: rewind.MessageID, Inclusive: rewind.Inclusive}, Rewound: outcome.Rewound}
	place := turn.Reply
	branched.Of = store.SiblingsOfReply
	if edit {
		place, branched.Of = turn.Question, store.SiblingsOfQuestion
	}
	branched.Index, branched.Count = max(place.Index, 1), max(place.Count, 1)

	s.claim(turn)
	turnID, err := s.current().RespondTo(ctx, prompt, nil, asked)
	if err != nil {
		s.unclaim()
		s.persisted.Cancel()
		return branched, stack.Wrap(err)
	}
	s.persisted.BindTurn(receipt.CommandID, turnID)
	if turnID == "" {
		s.unclaim()
		return branched, nil
	}
	branched.ResponseID = s.openAskedTurn(turnID, prompt)
	if !edit && branched.ResponseID != "" {
		s.carryQuestionDocuments(ctx, replaced.ID, branched.ResponseID)
	}
	return branched, nil
}

// documentScope is the session's conversation whose documents are read, for the person it is
// with, up to the turn until: all of them when it is empty.
func (s *Session) documentScope(until string) store.DocumentScope {
	return store.DocumentScope{ConversationID: s.spec.ConversationID, SessionID: s.id, Owner: s.spec.Caller.UserID, UntilResponse: until}
}

// branchDocuments are the documents the conversation holds once replaced is answered again or
// asked differently. They are read before anything is rewound, so a read that fails leaves the
// session as it was. A regenerate keeps the question, and the files sent with it. An edit
// replaces the question, so the files sent with it stay with the branch it leaves, and come
// back only when that branch is activated.
func (s *Session) branchDocuments(ctx context.Context, replaced store.AgentResponse, edit bool) ([]store.ConversationDocument, error) {
	until := replaced.ID
	if edit {
		until = replaced.ParentResponseID
	}
	documents, err := s.tree.ConversationDocuments(ctx, s.spec.CustomerID, s.documentScope(until))
	if err != nil || until != "" {
		return documents, err
	}
	// An edit of the session's first question: no file came with a turn of this session before
	// it. An empty UntilResponse reads them all.
	return slices.DeleteFunc(documents, func(document store.ConversationDocument) bool {
		return document.SessionID == s.id && document.ResponseID != ""
	}), nil
}

// carryQuestionDocuments keeps the files sent with the question a regenerate answers again
// under the new response as well, so a reopen, which reads only the conversation's turns,
// still finds them: the replaced response holding them is off it now. A copy rather than a
// move, so activating the replaced response's branch finds its own. One that cannot be kept
// leaves the turn answered and says so in the log, as keepDocument does.
func (s *Session) carryQuestionDocuments(ctx context.Context, replacedID, responseID string) {
	// The turn is running whether or not the caller is still there to hear it started.
	err := s.tree.CopyResponseDocuments(context.WithoutCancel(ctx), s.spec.CustomerID, s.id, replacedID, responseID)
	if err != nil {
		s.logger.Error("could not keep the files of a regenerated question", "session", s.id, "response", responseID, "error", err)
	}
}

// pinActivatedDocuments pins the documents of the branch an activation brought back in place of
// the ones pinned before. They can only be read once the branch is on the conversation, which
// is after Chat was changed, so a read that fails leaves the session without the conversation's
// files rather than with the abandoned branch's, and says so in the log, as an opening session
// does.
func (s *Session) pinActivatedDocuments(ctx context.Context) {
	documents, err := s.tree.ConversationDocuments(ctx, s.spec.CustomerID, s.documentScope(""))
	if err != nil {
		s.logger.Error("could not read the documents of an activated branch", "session", s.id, "error", err)
	}
	s.current().ReplaceDocuments(pinnable(documents)...)
}

// refuseBranches refuses what cannot switch branches, before anything is stopped: a
// conversation not kept in Chat, a call, a speech-to-speech model, a channel that is not the
// session's own, a deployment that keeps no responses, and a caller the conversation is not
// theirs.
func (s *Session) refuseBranches(ctx context.Context) error {
	unsupported := func(reason string) error {
		return stack.Wrap(&persistent.BranchesUnsupportedError{Reason: reason})
	}
	switch {
	case s.persisted == nil:
		return unsupported(persistent.UnsupportedNotKept)
	case s.Spec().Native():
		return unsupported(persistent.UnsupportedNative)
	case s.Voicing():
		return unsupported(persistent.UnsupportedCall)
	case s.spec.Thread:
		return unsupported(persistent.UnsupportedThread)
	case s.spec.Dispatched:
		return unsupported(persistent.UnsupportedDispatched)
	case s.records == nil || s.tree == nil || s.restoring == nil:
		return stack.Wrap(fmt.Errorf("%w: this deployment keeps no responses", persistent.ErrBranchUnavailable))
	}
	return s.persisted.CheckCaller(ctx, s.spec.Caller.UserID)
}

// replacing is the response replacing replaced, and where it stands among its alternatives:
// the newest answer to the same question for a regenerate, the newest question at the same
// point for an edit.
func (s *Session) replacing(ctx context.Context, replaced store.AgentResponse, edit bool) (persistent.Turn, error) {
	customer := s.spec.CustomerID
	questions, err := s.tree.Siblings(ctx, customer, s.id, replaced.ID, store.SiblingsOfQuestion)
	if err != nil {
		return persistent.Turn{}, err
	}
	id := newID()
	turn := persistent.Turn{ResponseID: id, ParentResponseID: replaced.ParentResponseID}
	if edit {
		turn.QuestionResponseID = id
		turn.Question = placeOf(len(questions)+1, len(questions)+1)
		return turn, nil
	}
	replies, err := s.tree.Siblings(ctx, customer, s.id, replaced.ID, store.SiblingsOfReply)
	if err != nil {
		return persistent.Turn{}, err
	}
	turn.QuestionResponseID = replaced.QuestionResponseID
	turn.Question = placeOf(store.SiblingPosition(questions, replaced, store.SiblingsOfQuestion))
	turn.Reply = placeOf(len(replies)+1, len(replies)+1)
	return turn, nil
}

// placeOf is index of count as a mark holds it: 1 of 1 is no place at all.
func placeOf(index, count int) persistent.Place {
	if count <= 1 {
		return persistent.Place{}
	}
	return persistent.Place{Index: index, Count: count}
}

// carryOnFrom hands the model the conversation's active path, after the person's other
// episodes as Create did, and has the review and the titler read it from there: a switch
// does not rename the conversation, but its next naming is of the branch on it.
func (s *Session) carryOnFrom(history []llm.Message) {
	s.current().RestoreHistory(append(slices.Clone(s.cards), history...))
	said := spokenOf(history)
	s.saidMu.Lock()
	defer s.saidMu.Unlock()
	s.said = said[:min(len(said), reviewLimit)]
	if s.naming != nil {
		s.naming.earlier = nil
	}
}

// countRanges records how many hidden ranges Chat keeps for the conversation now, and warns
// as it nears the cap. A count that cannot be read costs the metric, not the switch.
func (s *Session) countRanges(ctx context.Context, kind string) {
	count, err := s.tree.MergedRangeCount(ctx, s.spec.CustomerID, s.id)
	if err != nil {
		s.logger.Warn("could not count the conversation's hidden ranges", "session", s.id, "error", err)
		return
	}
	mergedRanges.WithLabelValues(kind).Observe(float64(count))
	if count > rangesWarned {
		s.logger.Warn("a conversation nears the most edit and regenerate points Chat keeps",
			"session", s.id, "conversation", s.spec.ConversationID, "ranges", count, "limit", 100)
	}
}

// switchID is the journal id of the switch request id begins in session: the same for a
// retry, so it finishes its own switch, and apart from any other session's.
func switchID(session, request string) string {
	sum := sha256.Sum256([]byte(session + "\x00" + request))
	return "sw" + hex.EncodeToString(sum[:])[:30]
}

// keptQuestionPrefix starts the id of an activation that kept the person's message its branch
// answers in the channel, rewinding only after it: a later open that finishes the switch
// sends again what it would have sent, without a second copy of that message.
const keptQuestionPrefix = "aq"

// activationID is the journal id of an activation of target: the one an earlier try left
// unfinished, so it is finished rather than sent again beside it, else a new one.
func (s *Session) activationID(ctx context.Context, target string, kept bool) (string, error) {
	unfinished, err := s.tree.UnfinishedBranchSwitches(ctx, s.spec.CustomerID, s.id)
	if err != nil {
		return "", err
	}
	for _, sw := range unfinished {
		if sw.Kind == store.SwitchActivate && sw.TargetResponseID == target && keptQuestion(sw.ID) == kept {
			return sw.ID, nil
		}
	}
	prefix := "as"
	if kept {
		prefix = keptQuestionPrefix
	}
	return prefix + newID()[:30], nil
}

func keptQuestion(switchID string) bool { return strings.HasPrefix(switchID, keptQuestionPrefix) }

// branchOf is the branch an activation of target brings back: top, the first of its responses
// off the conversation, the one after the newest the conversation still has, and tail, the
// path from top through target down to the newest response under target.
func branchOf(ctx context.Context, tree *store.Store, target store.AgentResponse) (store.AgentResponse, []store.AgentResponse, error) {
	customer, session := target.CustomerID, target.SessionID
	var above []store.AgentResponse
	top := target
	// Each step is one response nearer the session's first, so a session's length bounds it;
	// MaxResent bounds what could be sent again anyway.
	for range persistent.MaxResent {
		if top.ParentResponseID == "" {
			break
		}
		parent, err := tree.SessionResponse(ctx, customer, session, top.ParentResponseID)
		if err != nil {
			return store.AgentResponse{}, nil, err
		}
		if parent.RewoundAt == nil {
			break
		}
		above = append(above, parent)
		top = parent
	}
	below, err := tree.BranchTail(ctx, customer, session, target.ID)
	if err != nil {
		return store.AgentResponse{}, nil, err
	}
	slices.Reverse(above)
	return top, append(above, below...), nil
}

// chosenFor is what an activation of target sends again, oldest first: the copies of the
// messages of every response on tail, the path from target down to its newest response, and
// the question each one answers that another response asked. kept leaves out the question
// target answers, which the channel still shows. Each carries where it stands among its
// alternatives now, not where it stood when it was hidden.
func chosenFor(ctx context.Context, tree *store.Store, target store.AgentResponse, tail []store.AgentResponse, kept bool) ([]store.AgentBranchMessage, error) {
	customer, session := target.CustomerID, target.SessionID
	rows := map[string]store.AgentResponse{}
	for _, response := range tail {
		rows[response.ID] = response
	}
	// The questions the path's regenerated replies answer, whose messages are copied under the
	// response that asked them, with that response's reply.
	questionsOnly := map[string]bool{}
	ids := make([]string, 0, len(tail))
	for _, response := range tail {
		ids = append(ids, response.ID)
		if q := response.QuestionResponseID; q != response.ID && rows[q].ID == "" && !questionsOnly[q] {
			questionsOnly[q] = true
			ids = append(ids, q)
		}
	}
	copies, err := tree.BranchMessages(ctx, customer, session, ids)
	if err != nil {
		return nil, err
	}

	// A message sent again and hidden once more is copied twice, as the original and as the
	// copy sent again (resent_from): only the newest of each is sent.
	resentFrom := map[string]string{}
	for _, m := range copies {
		if from := resentFromOf(m); from != "" {
			resentFrom[m.OriginalMessageID] = from
		}
	}
	origin := func(id string) string {
		for range len(resentFrom) {
			from, ok := resentFrom[id]
			if !ok {
				break
			}
			id = from
		}
		return id
	}
	newest := map[string]store.AgentBranchMessage{}
	ownCopied := false
	for _, m := range copies {
		user := roleOf(m) == "user"
		switch {
		case kept && user && m.ResponseID == target.QuestionResponseID:
			continue
		case questionsOnly[m.ResponseID] && !user:
			continue
		}
		ownCopied = ownCopied || m.ResponseID == target.ID
		root := origin(m.OriginalMessageID)
		if held, ok := newest[root]; !ok || m.ChatCreatedAt.After(held.ChatCreatedAt) {
			newest[root] = m
		}
	}
	if !ownCopied {
		return nil, stack.Wrap(ErrBranchGone)
	}
	chosen := make([]store.AgentBranchMessage, 0, len(newest))
	for _, m := range newest {
		chosen = append(chosen, m)
	}
	slices.SortFunc(chosen, func(a, b store.AgentBranchMessage) int {
		return cmp.Or(a.ChatCreatedAt.Compare(b.ChatCreatedAt), strings.Compare(a.OriginalMessageID, b.OriginalMessageID))
	})

	marks, err := marksOf(ctx, tree, customer, session, rows, chosen)
	if err != nil {
		return nil, err
	}
	return persistent.Remarked(chosen, marks), nil
}

// marksOf is where each of chosen's messages stands among its alternatives, read from the
// tree now, by original message id. rows are the responses already read.
func marksOf(ctx context.Context, tree *store.Store, customer, session string, rows map[string]store.AgentResponse, chosen []store.AgentBranchMessage) (map[string]persistent.Branch, error) {
	type group struct{ response, of string }
	places := map[group]persistent.Place{}
	marks := map[string]persistent.Branch{}
	for _, m := range chosen {
		if m.ResponseID == "" {
			continue
		}
		row, ok := rows[m.ResponseID]
		if !ok {
			read, err := tree.SessionResponse(ctx, customer, session, m.ResponseID)
			if err != nil {
				return nil, err
			}
			row, rows[m.ResponseID] = read, read
		}
		of := store.SiblingsOfReply
		if roleOf(m) == "user" {
			of = store.SiblingsOfQuestion
		}
		at, ok := places[group{row.ID, of}]
		if !ok {
			siblings, err := tree.Siblings(ctx, customer, session, row.ID, of)
			if err != nil {
				return nil, err
			}
			at = placeOf(store.SiblingPosition(siblings, row, of))
			places[group{row.ID, of}] = at
		}
		mark := persistent.Branch{ResponseID: row.ID, ParentResponseID: row.ParentResponseID, QuestionResponseID: row.QuestionResponseID}
		if at != (persistent.Place{}) {
			mark.Of, mark.Index, mark.Count = of, at.Index, at.Count
		}
		marks[m.OriginalMessageID] = mark
	}
	return marks, nil
}

// roleOf is who wrote a copied message: user or assistant, as its support_runtime says.
func roleOf(m store.AgentBranchMessage) string {
	custom, _ := m.Message["custom"].(map[string]any)
	runtime, _ := custom["support_runtime"].(map[string]any)
	role, _ := runtime["role"].(string)
	return role
}

// resentFromOf is the message a copied message was itself a copy of, sent again by an
// earlier activation, empty for one somebody wrote.
func resentFromOf(m store.AgentBranchMessage) string {
	custom, _ := m.Message["custom"].(map[string]any)
	from, _ := custom[persistent.ResentFromField].(string)
	return from
}

// switchHooks finish a switch an earlier router left unfinished, when the session's
// conversation is next opened: the branch an activation sends again is chosen as it was, and
// the session's responses are moved to the switch's branch. A regenerate or an edit leaves the
// conversation at the replaced response's parent, as one whose new response never started.
func switchHooks(tree *store.Store) persistent.SwitchHooks {
	return persistent.SwitchHooks{
		Chosen: func(ctx context.Context, sw store.AgentBranchSwitch) ([]store.AgentBranchMessage, error) {
			target, err := tree.SessionResponse(ctx, sw.CustomerID, sw.SessionID, sw.TargetResponseID)
			if err != nil {
				return nil, err
			}
			// The responses are where they were when the switch began: they are moved once
			// the branch is sent again (Switched).
			top, tail, err := branchOf(ctx, tree, target)
			if err != nil {
				return nil, err
			}
			return chosenFor(ctx, tree, top, tail, keptQuestion(sw.ID))
		},
		Switched: func(ctx context.Context, sw store.AgentBranchSwitch) error {
			leaf := ""
			if sw.Kind == store.SwitchActivate {
				tail, err := tree.BranchTail(ctx, sw.CustomerID, sw.SessionID, sw.TargetResponseID)
				if err != nil {
					return err
				}
				leaf = tail[len(tail)-1].ID
			} else {
				target, err := tree.SessionResponse(ctx, sw.CustomerID, sw.SessionID, sw.TargetResponseID)
				if err != nil {
					return err
				}
				leaf = target.ParentResponseID
			}
			return tree.MoveActivePath(ctx, sw.CustomerID, sw.SessionID, leaf, time.Now().UTC())
		},
	}
}
