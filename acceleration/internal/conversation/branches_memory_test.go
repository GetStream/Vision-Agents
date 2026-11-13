package conversation_test

import (
	"cmp"
	"context"
	"errors"
	"maps"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The store keeps a conversation's branches in production.
var _ conversation.Branches = (*store.Store)(nil)

// memoryBranches keeps branch switches the way the store does, in memory, for one router.
// Advancing to failAdvanceTo, or counting the failCountAt-th message sent again, fails once:
// a router that stopped there.
type memoryBranches struct {
	mu            sync.Mutex
	copies        map[string]store.AgentBranchMessage
	switches      map[string]store.AgentBranchSwitch
	failAdvanceTo string
	failCountAt   int
	counted       int
}

var errStopped = errors.New("the router stopped here")

func newMemoryBranches() *memoryBranches {
	return &memoryBranches{copies: map[string]store.AgentBranchMessage{}, switches: map[string]store.AgentBranchSwitch{}}
}

func (m *memoryBranches) SaveBranchMessages(_ context.Context, messages []store.AgentBranchMessage) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, message := range messages {
		m.copies[message.SessionID+"/"+message.OriginalMessageID] = message
	}
	return nil
}

func (m *memoryBranches) BranchMessages(_ context.Context, customerID, sessionID string, responseIDs []string) ([]store.AgentBranchMessage, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var found []store.AgentBranchMessage
	for _, message := range m.copies {
		if message.CustomerID == customerID && message.SessionID == sessionID &&
			(len(responseIDs) == 0 || slices.Contains(responseIDs, message.ResponseID)) {
			found = append(found, message)
		}
	}
	slices.SortFunc(found, func(a, b store.AgentBranchMessage) int {
		return cmp.Or(a.ChatCreatedAt.Compare(b.ChatCreatedAt), strings.Compare(a.OriginalMessageID, b.OriginalMessageID))
	})
	return found, nil
}

func (m *memoryBranches) StartBranchSwitch(_ context.Context, row *store.AgentBranchSwitch) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.switches[row.ID]; ok {
		return nil
	}
	if row.CreatedAt.IsZero() {
		row.CreatedAt = time.Now().UTC()
	}
	if row.State == "" {
		row.State = store.SwitchSnapshotted
	}
	m.switches[row.ID] = *row
	return nil
}

var switchOrder = []string{store.SwitchSnapshotted, store.SwitchRewound, store.SwitchResending, store.SwitchDone}

func (m *memoryBranches) AdvanceBranchSwitch(_ context.Context, customerID, id, state string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.failAdvanceTo == state {
		m.failAdvanceTo = ""
		return errStopped
	}
	row, ok := m.switches[id]
	if ok && row.CustomerID == customerID && slices.Index(switchOrder, row.State) < slices.Index(switchOrder, state) {
		row.State = state
		m.switches[id] = row
	}
	return nil
}

func (m *memoryBranches) AbandonBranchSwitch(_ context.Context, customerID, sessionID, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	maps.DeleteFunc(m.copies, func(_ string, message store.AgentBranchMessage) bool {
		return message.CustomerID == customerID && message.SessionID == sessionID && message.SwitchID == id
	})
	delete(m.switches, id)
	return nil
}

func (m *memoryBranches) CountBranchResent(_ context.Context, _, id string) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.counted++
	if m.counted == m.failCountAt {
		return 0, errStopped
	}
	row := m.switches[id]
	row.Resent++
	m.switches[id] = row
	return row.Resent, nil
}

func (m *memoryBranches) UnfinishedBranchSwitches(_ context.Context, customerID, sessionID string) ([]store.AgentBranchSwitch, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var found []store.AgentBranchSwitch
	for _, row := range m.switches {
		if row.CustomerID == customerID && row.SessionID == sessionID && row.State != store.SwitchDone {
			found = append(found, row)
		}
	}
	slices.SortFunc(found, func(a, b store.AgentBranchSwitch) int { return a.CreatedAt.Compare(b.CreatedAt) })
	return found, nil
}

// row is the switch id as kept now, and whether it is kept at all.
func (m *memoryBranches) row(id string) (store.AgentBranchSwitch, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	row, ok := m.switches[id]
	return row, ok
}

// copied are the ids of the messages kept as copies, oldest first.
func (m *memoryBranches) copied(session string) []string {
	found, _ := m.BranchMessages(context.Background(), "customer", session, nil)
	var ids []string
	for _, message := range found {
		ids = append(ids, message.OriginalMessageID)
	}
	return ids
}
