package metered

import (
	"context"
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// The models a memory row is recorded as, each priced on its own.
const (
	// MemorySearch is a recall.
	MemorySearch = "search"
	// MemoryAdd is an exchange handed over to be learned from.
	MemoryAdd = "add"
	// MemoryForget is a session's or a user's memories deleted.
	MemoryForget = "forget"
)

// memoryFailed is the error code of a memory call that failed.
const memoryFailed = "memory_failed"

// Memory holds the memory backend. A Store comes only from For, which needs an owner, so
// nothing is recalled or remembered that nobody is charged for.
type Memory struct {
	backend  memory.Backend
	recorder *routing.Recorder
}

// NewMemory meters a backend, recording through recorder as memory rows. A nil recorder
// records nothing.
func NewMemory(backend memory.Backend, recorder *routing.Recorder) *Memory {
	if recorder != nil {
		recorder = recorder.For(routing.Memory)
	}
	return &Memory{backend: backend, recorder: recorder}
}

// For is the memory store as owner uses it: every call is a row of owner's.
func (m *Memory) For(owner routing.Owner) OwnedMemory {
	return OwnedMemory{memory: m, owner: owner}
}

// Close closes the backend.
func (m *Memory) Close() error { return m.backend.Close() }

func (m *Memory) meter(owner routing.Owner, model string) routing.Meter {
	return routing.Meter{
		Recorder: m.recorder,
		Config:   routing.ProviderConfig{Provider: m.backend.Provider(), Model: model},
		Owner:    owner,
		Code:     func(error) string { return memoryFailed },
	}
}

// OwnedMemory is a memory.Store for one owner. Each call is a routing.Call: one row, a search
// for a recall, an add for an exchange remembered, a forget for a deletion, with the requests
// the provider was asked.
type OwnedMemory struct {
	memory *Memory
	owner  routing.Owner
}

var _ memory.Store = OwnedMemory{}

// Recall asks what is known.
func (o OwnedMemory) Recall(ctx context.Context, query memory.Query) ([]memory.Memory, error) {
	recalled, err := routing.Call(ctx, o.memory.meter(o.owner, MemorySearch), func(ctx context.Context) (memory.Recalled, error) {
		return o.memory.backend.Recall(ctx, query)
	})
	return recalled.Memories, err
}

// Remember hands an exchange over. One with nothing said in it asks the provider nothing.
func (o OwnedMemory) Remember(ctx context.Context, scope memory.Scope, messages []llm.Message) error {
	if !slices.ContainsFunc(messages, func(said llm.Message) bool { return said.Content != "" }) {
		return nil
	}
	_, err := routing.Call(ctx, o.memory.meter(o.owner, MemoryAdd), func(ctx context.Context) (memory.Billed, error) {
		return o.memory.backend.Remember(ctx, scope, messages)
	})
	return err
}

// Truncate deletes everything known about one user of one app.
func (o OwnedMemory) Truncate(ctx context.Context, appID, userID string) error {
	_, err := routing.Call(ctx, o.memory.meter(o.owner, MemoryForget), func(ctx context.Context) (memory.Billed, error) {
		return o.memory.backend.Truncate(ctx, appID, userID)
	})
	return err
}

// ForgetRun deletes what one session of one app learned.
func (o OwnedMemory) ForgetRun(ctx context.Context, appID, runID string) error {
	_, err := routing.Call(ctx, o.memory.meter(o.owner, MemoryForget), func(ctx context.Context) (memory.Billed, error) {
		return o.memory.backend.ForgetRun(ctx, appID, runID)
	})
	return err
}

// Provider is the backend's name.
func (o OwnedMemory) Provider() string { return o.memory.backend.Provider() }

// Close releases nothing: the backend is the process's, and closes with it.
func (o OwnedMemory) Close() error { return nil }
