// Package memory defines what an agent remembers between conversations.
//
// A call ends and its history goes with it. A memory store keeps the facts worth carrying
// forward, so the next call starts knowing what the last one established rather than
// asking again.
//
// The contract is deliberately two methods. Everything a memory product differs on, which
// is how facts are extracted, deduplicated, ranked and expired, is the provider's own
// business; what a caller needs is to hand over a conversation and later ask what is
// known. Anything a provider offers beyond that is reached through its own client.
package memory

import (
	"context"
	"errors"
	"fmt"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// Scope is who a memory belongs to.
//
// Nothing is stored without one. Memories are personal, so writing them against a missing
// or shared identity would mean one caller reading another's, which is why an empty scope
// is an error rather than a default.
type Scope struct {
	// AppID is the API key or application using this service. It separates two
	// deployments sharing one memory account.
	AppID string
	// UserID is who the memories are about, which here is the customer.
	UserID string
}

// Validate reports whether the scope identifies someone.
func (s Scope) Validate() error {
	if s.UserID == "" {
		return errors.New("memory: a user id is required, memories are never shared")
	}
	return nil
}

// Query asks what is known that bears on the conversation about to happen.
type Query struct {
	Scope Scope
	// Text is what to recall against, usually what the participant just said. Empty
	// returns whatever the provider considers most relevant to the user overall.
	Text string
	// Limit caps how many memories come back. Zero leaves the provider's default.
	Limit int
}

// Memory is one remembered fact.
type Memory struct {
	// ID is the provider's identifier, so a caller can correct or forget it later.
	ID string
	// Text is the fact as the provider stored it, which is usually a rewrite of what was
	// said rather than the words themselves.
	Text string
	// Score is how relevant the provider judged it, where the provider reports one.
	Score float64
}

// Store is a memory provider.
//
// Remember is fire-and-forget by design: extraction happens on the provider's side and
// takes longer than a turn, so a conversation neither waits for it nor sees its result.
type Store interface {
	// Recall returns what is known that bears on the query, most relevant first.
	Recall(ctx context.Context, query Query) ([]Memory, error)
	// Remember hands a conversation over to be learned from.
	Remember(ctx context.Context, scope Scope, messages []llm.Message) error
	// Provider is the stable provider name used in stats, e.g. "mem0".
	Provider() string
	Close() error
}

// Prompt renders memories as a system message to prepend to an agent's instructions.
// It returns an empty string when there is nothing to say, so a caller can add it
// unconditionally.
func Prompt(memories []Memory) string {
	if len(memories) == 0 {
		return ""
	}

	prompt := "What you already know about this person:"
	for _, remembered := range memories {
		prompt += fmt.Sprintf("\n- %s", remembered.Text)
	}
	return prompt
}
