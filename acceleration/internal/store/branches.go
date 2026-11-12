package store

import (
	"cmp"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/pgdialect"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// What a response's alternatives are counted among.
const (
	// SiblingsOfReply are the answers to one question: the response and its regenerations.
	SiblingsOfReply = "reply"
	// SiblingsOfQuestion are the questions asked at one point of the conversation: the
	// original and its edits, one per question however often each was answered.
	SiblingsOfQuestion = "question"
)

// Siblings returns the alternatives to one of a session's responses, oldest first, the
// response's own included. Of a reply, every response answering its question; of a question,
// the first response of each question asked after the same parent. A response from before
// branches were kept has none but itself.
func (s *Store) Siblings(ctx context.Context, customerID, sessionID, responseID, of string) ([]AgentResponse, error) {
	if customerID == "" || sessionID == "" || responseID == "" {
		return nil, stack.Wrap(errors.New("store: a customer, a session and a response id are required"))
	}
	if of != SiblingsOfReply && of != SiblingsOfQuestion {
		return nil, stack.Wrap(fmt.Errorf("store: siblings of %q: want %s or %s", of, SiblingsOfReply, SiblingsOfQuestion))
	}

	response, err := s.sessionResponse(ctx, customerID, sessionID, responseID)
	if err != nil {
		return nil, err
	}
	if response.QuestionResponseID == "" {
		return []AgentResponse{response}, nil
	}
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]AgentResponse, error) {
		query := db.NewSelect().Model((*AgentResponse)(nil)).
			Where("customer_id = ?", customerID).
			Where("session_id = ?", sessionID).
			Where("question_response_id IS NOT NULL").
			Order("created_at ASC", "id ASC")
		if response.ParentResponseID == "" {
			query = query.Where("parent_response_id IS NULL")
		} else {
			query = query.Where("parent_response_id = ?", response.ParentResponseID)
		}
		if of == SiblingsOfReply {
			query = query.Where("question_response_id = ?", response.QuestionResponseID)
		}
		var siblings []AgentResponse
		err := query.Scan(ctx, &siblings)
		return siblings, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: siblings: %w", err))
	}
	siblings := merged(lists, oldestResponseFirst, 0)
	if of == SiblingsOfQuestion {
		siblings = distinct(siblings, func(r AgentResponse) string { return r.QuestionResponseID })
	}
	return siblings, nil
}

// SiblingPosition is where response stands among siblings as Siblings returned them for of:
// its 1-based index and how many there are. A response not among them is 1 of 1.
func SiblingPosition(siblings []AgentResponse, response AgentResponse, of string) (index, count int) {
	at := slices.IndexFunc(siblings, func(sibling AgentResponse) bool {
		if of == SiblingsOfQuestion {
			return sibling.QuestionResponseID != "" && sibling.QuestionResponseID == response.QuestionResponseID
		}
		return sibling.ID == response.ID
	})
	if at < 0 {
		return 1, 1
	}
	return at + 1, len(siblings)
}

// BranchTail returns the path from one of a session's responses down to the newest response
// under it, that response first and the newest last. A response with nothing under it is a
// path of itself.
func (s *Store) BranchTail(ctx context.Context, customerID, sessionID, fromResponseID string) ([]AgentResponse, error) {
	if customerID == "" || sessionID == "" || fromResponseID == "" {
		return nil, stack.Wrap(errors.New("store: a customer, a session and a response id are required"))
	}
	tree, err := s.responseTree(ctx, customerID, sessionID)
	if err != nil {
		return nil, err
	}
	from, ok := tree.byID[fromResponseID]
	if !ok {
		return nil, stack.Wrap(ErrUnknownResponse)
	}

	// The newest descendant, wherever it is under from, rather than the newest child's newest
	// child: an older branch can have been answered after a newer one was started.
	newest := from
	under := []string{from.ID}
	for len(under) > 0 {
		id := under[0]
		under = under[1:]
		for _, child := range tree.children[id] {
			if oldestResponseFirst(child, newest) > 0 {
				newest = child
			}
			under = append(under, child.ID)
		}
	}
	path := tree.pathTo(newest.ID, from.ID)
	slices.Reverse(path)
	return path, nil
}

// MoveActivePath makes the path ending at leafResponseID the session's conversation. The
// responses after the point where it leaves the current conversation are rewound at at, and
// the ones on the new path after that point are brought back. Running it again changes
// nothing.
func (s *Store) MoveActivePath(ctx context.Context, customerID, sessionID, leafResponseID string, at time.Time) error {
	if customerID == "" || sessionID == "" || leafResponseID == "" {
		return stack.Wrap(errors.New("store: a customer, a session and a response id are required"))
	}
	if at.IsZero() {
		at = time.Now().UTC()
	}
	tree, err := s.responseTree(ctx, customerID, sessionID)
	if err != nil {
		return err
	}
	if _, ok := tree.byID[leafResponseID]; !ok {
		return stack.Wrap(ErrUnknownResponse)
	}

	// The fork is the newest response the new path shares with the conversation; everything
	// on the new path above it is to be brought back.
	var fork *AgentResponse
	var back []string
	for _, response := range tree.pathTo(leafResponseID, "") {
		if response.RewoundAt == nil {
			fork = &response
			break
		}
		back = append(back, response.ID)
	}

	// The old branch is rewound before the new one is brought back, so a router stopping in
	// between leaves the conversation cut short at the fork rather than two branches in it.
	_, err = onEveryShard(ctx, s, func(ctx context.Context, db bun.IDB) (sql.Result, error) {
		query := db.NewUpdate().Model((*AgentResponse)(nil)).
			Set("rewound_at = ?", at).
			Where("customer_id = ?", customerID).
			Where("session_id = ?", sessionID).
			Where("rewound_at IS NULL")
		if fork != nil {
			query = query.Where("(created_at, id) > (?, ?)", fork.CreatedAt, fork.ID)
		}
		return query.Exec(ctx)
	})
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: move active path: rewind: %w", err))
	}
	if len(back) == 0 {
		return nil
	}
	_, err = onEveryShard(ctx, s, func(ctx context.Context, db bun.IDB) (sql.Result, error) {
		return db.NewUpdate().Model((*AgentResponse)(nil)).
			Set("rewound_at = NULL").
			Where("customer_id = ?", customerID).
			Where("session_id = ?", sessionID).
			Where("id IN (?)", bun.In(back)).
			Exec(ctx)
	})
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: move active path: bring back: %w", err))
	}
	return nil
}

// sessionResponse is one of a session's responses, from whichever shard holds it.
func (s *Store) sessionResponse(ctx context.Context, customerID, sessionID, responseID string) (AgentResponse, error) {
	response, err := whicheverShard(ctx, s, func(ctx context.Context, db bun.IDB) (AgentResponse, error) {
		var response AgentResponse
		err := db.NewSelect().Model(&response).
			Where("id = ?", responseID).
			Where("customer_id = ?", customerID).
			Where("session_id = ?", sessionID).
			Scan(ctx)
		return response, err
	})
	if errors.Is(err, sql.ErrNoRows) {
		return AgentResponse{}, stack.Wrap(ErrUnknownResponse)
	}
	if err != nil {
		return AgentResponse{}, stack.Wrap(fmt.Errorf("store: session response: %w", err))
	}
	return response, nil
}

// responseTree is a session's responses by id and by parent.
type responseTree struct {
	byID     map[string]AgentResponse
	children map[string][]AgentResponse
}

// responseTree reads every response of a session in every shard, rewound or not.
func (s *Store) responseTree(ctx context.Context, customerID, sessionID string) (responseTree, error) {
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]AgentResponse, error) {
		var responses []AgentResponse
		err := db.NewSelect().Model((*AgentResponse)(nil)).
			Column("id", "session_id", "customer_id", "status", "created_at", "rewound_at", "parent_response_id", "question_response_id").
			Where("customer_id = ?", customerID).
			Where("session_id = ?", sessionID).
			Order("created_at ASC", "id ASC").
			Scan(ctx, &responses)
		return responses, err
	})
	if err != nil {
		return responseTree{}, stack.Wrap(fmt.Errorf("store: response tree: %w", err))
	}
	tree := responseTree{byID: map[string]AgentResponse{}, children: map[string][]AgentResponse{}}
	for _, response := range merged(lists, oldestResponseFirst, 0) {
		tree.byID[response.ID] = response
		if response.ParentResponseID != "" {
			tree.children[response.ParentResponseID] = append(tree.children[response.ParentResponseID], response)
		}
	}
	return tree, nil
}

// pathTo walks up from id through its parents and returns the responses it passed, id first,
// stopping after stop or at a response whose parent is not in the session. The walk is at
// most as long as the session, so parents that loop end it rather than hang it.
func (t responseTree) pathTo(id, stop string) []AgentResponse {
	var path []AgentResponse
	for range len(t.byID) {
		response, ok := t.byID[id]
		if !ok {
			break
		}
		path = append(path, response)
		if response.ID == stop || response.ParentResponseID == "" {
			break
		}
		id = response.ParentResponseID
	}
	return path
}

// SaveBranchMessages keeps copies of messages a rewind is about to hide. Saving one again
// replaces it with what it says now: a switch retried after the channel moved on copies the
// messages again, and the later copy is the one that was hidden.
func (s *Store) SaveBranchMessages(ctx context.Context, messages []AgentBranchMessage) error {
	if len(messages) == 0 {
		return nil
	}
	for i := range messages {
		if messages[i].SessionID == "" || messages[i].OriginalMessageID == "" || messages[i].CustomerID == "" {
			return stack.Wrap(errors.New("store: a session, a message and a customer id are required"))
		}
		if messages[i].Message == nil {
			messages[i].Message = map[string]any{}
		}
	}
	_, err := s.shard.NewInsert().Model(&messages).
		On("CONFLICT (session_id, original_message_id) DO UPDATE").
		Set("response_id = EXCLUDED.response_id").
		Set("switch_id = EXCLUDED.switch_id").
		Set("chat_created_at = EXCLUDED.chat_created_at").
		Set("author_id = EXCLUDED.author_id").
		Set("message = EXCLUDED.message").
		Where("abm.customer_id = EXCLUDED.customer_id").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: save branch messages: %w", err))
	}
	return nil
}

// BranchMessages returns the copies of a session's hidden messages that belong to responseIDs,
// or all of them when it names none, oldest first.
func (s *Store) BranchMessages(ctx context.Context, customerID, sessionID string, responseIDs []string) ([]AgentBranchMessage, error) {
	if customerID == "" || sessionID == "" {
		return nil, stack.Wrap(errors.New("store: a customer and a session id are required"))
	}
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]AgentBranchMessage, error) {
		query := db.NewSelect().Model((*AgentBranchMessage)(nil)).
			Where("customer_id = ?", customerID).
			Where("session_id = ?", sessionID).
			Order("chat_created_at ASC", "original_message_id ASC")
		if len(responseIDs) > 0 {
			query = query.Where("response_id IN (?)", bun.In(responseIDs))
		}
		var messages []AgentBranchMessage
		err := query.Scan(ctx, &messages)
		return messages, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: branch messages: %w", err))
	}
	// A switch finished by a router of another shard copies the messages again into its own.
	all := merged(lists, func(a, b AgentBranchMessage) int {
		return cmp.Or(a.ChatCreatedAt.Compare(b.ChatCreatedAt), strings.Compare(a.OriginalMessageID, b.OriginalMessageID))
	}, 0)
	return distinct(all, func(m AgentBranchMessage) string { return m.OriginalMessageID }), nil
}

// DeleteBranchMessages deletes every copy of a session's hidden messages, and answers how
// many it deleted.
func (s *Store) DeleteBranchMessages(ctx context.Context, customerID, sessionID string) (int64, error) {
	if customerID == "" || sessionID == "" {
		return 0, stack.Wrap(errors.New("store: a customer and a session id are required"))
	}
	deleted, err := onEveryShard(ctx, s, func(ctx context.Context, db bun.IDB) (sql.Result, error) {
		return db.NewDelete().Model((*AgentBranchMessage)(nil)).
			Where("customer_id = ?", customerID).
			Where("session_id = ?", sessionID).
			Exec(ctx)
	})
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: delete branch messages: %w", err))
	}
	return deleted, nil
}

// DeleteUserBranchMessages deletes every copy of a hidden message in the sessions of an app's
// end user, on every shard, and answers how many it deleted.
func (s *Store) DeleteUserBranchMessages(ctx context.Context, appID, userID string) (int64, error) {
	if appID == "" || userID == "" {
		return 0, stack.Wrap(errors.New("store: an app and a user id are required"))
	}
	deleted, err := onEveryShard(ctx, s, func(ctx context.Context, db bun.IDB) (sql.Result, error) {
		return db.NewDelete().Model((*AgentBranchMessage)(nil)).
			Where("customer_id = ?", appID).
			Where("session_id IN (SELECT id FROM agent_sessions WHERE customer_id = ? AND user_id = ?)", appID, userID).
			Exec(ctx)
	})
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: delete user branch messages: %w", err))
	}
	return deleted, nil
}

// switchStates are a switch's states in the order it passes them.
var switchStates = []string{SwitchSnapshotted, SwitchRewound, SwitchResending, SwitchDone}

// StartBranchSwitch records a switch about to begin. Starting the same switch again is not an
// error: a request retried after the first attempt wrote its row finds it there.
func (s *Store) StartBranchSwitch(ctx context.Context, branchSwitch *AgentBranchSwitch) error {
	if branchSwitch.ID == "" || branchSwitch.SessionID == "" || branchSwitch.CustomerID == "" {
		return stack.Wrap(errors.New("store: a switch, a session and a customer id are required"))
	}
	now := time.Now().UTC()
	if branchSwitch.CreatedAt.IsZero() {
		branchSwitch.CreatedAt = now
	}
	branchSwitch.UpdatedAt = now
	if branchSwitch.State == "" {
		branchSwitch.State = SwitchSnapshotted
	}
	_, err := s.shard.NewInsert().Model(branchSwitch).
		On("CONFLICT (id) DO NOTHING").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: start branch switch: %w", err))
	}
	return nil
}

// AdvanceBranchSwitch moves a switch on to state. A switch already there or past it stays
// where it is, so a router finishing a switch another router already moved on never moves it
// back.
func (s *Store) AdvanceBranchSwitch(ctx context.Context, customerID, id, state string) error {
	if customerID == "" || id == "" {
		return stack.Wrap(errors.New("store: a customer and a switch id are required"))
	}
	if !slices.Contains(switchStates, state) {
		return stack.Wrap(fmt.Errorf("store: advance branch switch: unknown state %q", state))
	}
	_, err := onEveryShard(ctx, s, func(ctx context.Context, db bun.IDB) (sql.Result, error) {
		return db.NewUpdate().Model((*AgentBranchSwitch)(nil)).
			Set("state = ?", state).
			Set("updated_at = ?", time.Now().UTC()).
			Where("customer_id = ?", customerID).
			Where("id = ?", id).
			Where("array_position(?::text[], state) < array_position(?::text[], ?)",
				pgdialect.Array(switchStates), pgdialect.Array(switchStates), state).
			Exec(ctx)
	})
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: advance branch switch: %w", err))
	}
	return nil
}

// CountBranchResent counts one more message of a switch's branch sent again, and answers how
// many have been.
func (s *Store) CountBranchResent(ctx context.Context, customerID, id string) (int, error) {
	if customerID == "" || id == "" {
		return 0, stack.Wrap(errors.New("store: a customer and a switch id are required"))
	}
	resent, err := whicheverShard(ctx, s, func(ctx context.Context, db bun.IDB) (int, error) {
		var resent int
		err := db.NewUpdate().Model((*AgentBranchSwitch)(nil)).
			Set("resent = resent + 1").
			Set("updated_at = ?", time.Now().UTC()).
			Where("customer_id = ?", customerID).
			Where("id = ?", id).
			Returning("resent").
			Scan(ctx, &resent)
		return resent, err
	})
	if errors.Is(err, sql.ErrNoRows) {
		return 0, stack.Wrap(fmt.Errorf("store: count branch resent: no switch %s", id))
	}
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: count branch resent: %w", err))
	}
	return resent, nil
}

// UnfinishedBranchSwitches returns a session's switches that are not done, oldest first.
func (s *Store) UnfinishedBranchSwitches(ctx context.Context, customerID, sessionID string) ([]AgentBranchSwitch, error) {
	if customerID == "" || sessionID == "" {
		return nil, stack.Wrap(errors.New("store: a customer and a session id are required"))
	}
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]AgentBranchSwitch, error) {
		var switches []AgentBranchSwitch
		err := db.NewSelect().Model((*AgentBranchSwitch)(nil)).
			Where("customer_id = ?", customerID).
			Where("session_id = ?", sessionID).
			Where("state <> ?", SwitchDone).
			Order("created_at ASC", "id ASC").
			Scan(ctx, &switches)
		return switches, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: unfinished branch switches: %w", err))
	}
	return merged(lists, oldestSwitchFirst, 0), nil
}

func oldestSwitchFirst(a, b AgentBranchSwitch) int {
	return cmp.Or(a.CreatedAt.Compare(b.CreatedAt), strings.Compare(a.ID, b.ID))
}

// MergedRangeCount is how many hidden ranges Stream Chat keeps for a session's channel after
// the switches that rewound it, merged the way Chat merges them, which is what Chat caps. A
// switch counts once the channel was rewound, from the rewound state on.
func (s *Store) MergedRangeCount(ctx context.Context, customerID, sessionID string) (int, error) {
	if customerID == "" || sessionID == "" {
		return 0, stack.Wrap(errors.New("store: a customer and a session id are required"))
	}
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]AgentBranchSwitch, error) {
		var switches []AgentBranchSwitch
		err := db.NewSelect().Model((*AgentBranchSwitch)(nil)).
			Column("id", "hidden_from", "inclusive", "created_at").
			Where("customer_id = ?", customerID).
			Where("session_id = ?", sessionID).
			Where("state <> ?", SwitchSnapshotted).
			Scan(ctx, &switches)
		return switches, err
	})
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: merged range count: %w", err))
	}
	switches := distinct(merged(lists, oldestSwitchFirst, 0), func(b AgentBranchSwitch) string { return b.ID })
	ranges := make([]HiddenRange, 0, len(switches))
	for _, b := range switches {
		ranges = append(ranges, HiddenRange{From: b.HiddenFrom, To: b.CreatedAt, FromInclusive: b.Inclusive})
	}
	return len(MergeHiddenRanges(ranges)), nil
}

// HiddenRange is the created_at range of the messages one rewind hid. To is inclusive; From
// is inclusive only when the message rewound to was hidden too.
type HiddenRange struct {
	From          time.Time
	To            time.Time
	FromInclusive bool
}

// MergeHiddenRanges sorts ranges by From and merges the ones that overlap or touch, as
// Stream Chat's rewindstate.Merge does: To is inclusive, so a range starting where another
// ends leaves no gap whatever its own lower bound.
func MergeHiddenRanges(ranges []HiddenRange) []HiddenRange {
	sorted := slices.Clone(ranges)
	slices.SortStableFunc(sorted, func(a, b HiddenRange) int {
		if c := a.From.Compare(b.From); c != 0 {
			return c
		}
		// Inclusive first, so a merged range keeps the inclusive lower bound.
		switch {
		case a.FromInclusive == b.FromInclusive:
			return 0
		case a.FromInclusive:
			return -1
		default:
			return 1
		}
	})
	out := make([]HiddenRange, 0, len(sorted))
	for _, r := range sorted {
		if n := len(out); n > 0 && !r.From.After(out[n-1].To) {
			last := &out[n-1]
			if r.From.Equal(last.From) {
				last.FromInclusive = last.FromInclusive || r.FromInclusive
			}
			if r.To.After(last.To) {
				last.To = r.To
			}
			continue
		}
		out = append(out, r)
	}
	return out
}
