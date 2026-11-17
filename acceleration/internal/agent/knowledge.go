package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// lookupArguments is what the model fills in to look something up.
type lookupArguments struct {
	Query string `json:"query"`
}

// knowledgeReader answers the model's lookups out of what the business wrote down.
//
// It is on the live path rather than behind a queue, unlike the memory writer: the caller
// is waiting on the answer, which is the whole reason the model asked. What bounds it is
// the store's own timeout. The store is the agent's owner's, so every lookup is recorded
// under the same customer and labels as the models the conversation used.
type knowledgeReader struct {
	store     knowledge.Store
	namespace string
	limit     int
}

func newKnowledgeReader(store knowledge.Store, namespace string, limit int) *knowledgeReader {
	return &knowledgeReader{store: store, namespace: namespace, limit: limit}
}

// Search answers one question out of the knowledge base.
func (r *knowledgeReader) Search(ctx context.Context, question string) ([]knowledge.Document, error) {
	return r.store.Search(ctx, knowledge.Query{
		Namespace: r.namespace,
		Text:      question,
		Limit:     r.limit,
	})
}

// lookup answers a question out of the knowledge base and hands the passages to the model.
//
// A question nothing covers is answered in words rather than an error: the model is told
// the handbook says nothing, which is what stops it inventing an answer, and the caller
// hears "I don't have that" instead of an apology for a broken tool.
func (a *Agent) lookup(ctx context.Context, call llm.ToolCall) (string, bool, error) {
	var arguments lookupArguments
	if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil {
		return "", false, fmt.Errorf("agent: could not read the lookup arguments: %w", err)
	}
	said, err := a.knowledgeAnswer(ctx, call.ID, arguments.Query)
	return said, false, err
}

// subagentLookup is how delegated work reads the knowledge base, nil when there is none.
func (a *Agent) subagentLookup() func(ctx context.Context, turnID, query string) (string, error) {
	if a.knowledge == nil {
		return nil
	}
	return a.knowledgeAnswer
}

func (a *Agent) knowledgeAnswer(ctx context.Context, turnID, query string) (string, error) {
	question := strings.TrimSpace(query)
	if question == "" {
		return "", errors.New("agent: a lookup needs something to look for")
	}
	found, err := a.knowledge.Search(ctx, question)
	if err != nil {
		return "", fmt.Errorf("agent: look up %q: %w", question, err)
	}
	a.emitter.Send(LookedUp{TurnID: turnID, Query: question, Documents: len(found)})
	return knowledge.Prompt(found), nil
}
