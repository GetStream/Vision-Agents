// Package knowledge defines what an agent may look up rather than know.
//
// Memory is what the agent learned about a person; knowledge is what the business already
// wrote down. The two are separate because they are read at different moments: memory is
// recalled once on joining, while knowledge is looked up mid-sentence, when the caller
// asks something the instructions do not answer.
//
// The contract is deliberately one method. How a provider chunks, indexes and ranks is its
// own business; what a caller needs is a question answered with passages it can put in
// front of the model. Anything a provider offers beyond that is reached through its own
// client.
package knowledge

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// Query is a question to answer out of what the business wrote down.
type Query struct {
	// Namespace is which body of knowledge to read. Nothing is searched without one:
	// searching everything would answer one customer's caller out of another's handbook.
	Namespace string
	// Text is the question, in the caller's own words.
	Text string
	// Limit caps how many passages come back. Zero leaves the provider's default.
	Limit int
}

// Validate reports whether the query names something to search.
func (q Query) Validate() error {
	if strings.TrimSpace(q.Namespace) == "" {
		return errors.New("knowledge: a namespace is required, knowledge is never shared")
	}
	if strings.TrimSpace(q.Text) == "" {
		return errors.New("knowledge: there is nothing to look up")
	}
	return nil
}

// Document is one passage that bears on the question.
type Document struct {
	// ID is the provider's identifier for the passage.
	ID string
	// Text is the passage itself, which is what the model reads.
	Text string
	// Source is where it came from, so the agent can say where it read something.
	Source string
	// Score is how relevant the provider judged it.
	Score float64
}

// Store is a knowledge provider.
type Store interface {
	// Search returns the passages that bear on the question, most relevant first.
	Search(ctx context.Context, query Query) ([]Document, error)
	// Provider is the stable provider name used in stats, e.g. "turbopuffer".
	Provider() string
	Close() error
}

// Writer fills a knowledge base, which Store only ever reads.
//
// It is separate because almost nothing writes: a conversation looks things up and never
// puts anything there, so the reading half is what a session is given.
type Writer interface {
	// Upsert writes passages, replacing whatever is already stored under their ids.
	Upsert(ctx context.Context, namespace string, documents []Document) error
	// Delete removes passages by id. Ids that are not there are not an error: what the
	// caller asked for is that they are gone.
	Delete(ctx context.Context, namespace string, ids []string) error
}

// Prompt renders passages as the answer to a lookup, which is what the model is handed.
// It says so plainly when there is nothing, because a model given an empty answer invents
// one.
func Prompt(documents []Document) string {
	if len(documents) == 0 {
		return "Nothing in the knowledge base covers that. Say so rather than guessing."
	}

	var prompt strings.Builder
	prompt.WriteString("What the knowledge base says:")
	for _, found := range documents {
		if found.Source == "" {
			fmt.Fprintf(&prompt, "\n\n%s", found.Text)
			continue
		}
		fmt.Fprintf(&prompt, "\n\nFrom %s:\n%s", found.Source, found.Text)
	}
	return prompt.String()
}
