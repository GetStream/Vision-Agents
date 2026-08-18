package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// knowledgeModel names the model row a lookup is recorded as, so what looking things up
// costs is reported next to what the models cost.
const knowledgeModel = "search"

// lookupArguments is what the model fills in to look something up.
type lookupArguments struct {
	Query string `json:"query"`
}

// knowledgeReader answers the model's lookups out of what the business wrote down.
//
// It is on the live path rather than behind a queue, unlike the memory writer: the caller
// is waiting on the answer, which is the whole reason the model asked. What bounds it is
// the store's own timeout.
type knowledgeReader struct {
	store     knowledge.Store
	namespace string
	limit     int
	owner     routing.Owner
	recorder  *routing.Recorder
	logger    *slog.Logger
}

func newKnowledgeReader(
	store knowledge.Store,
	namespace string,
	limit int,
	owner routing.Owner,
	recorder *routing.Recorder,
	logger *slog.Logger,
) *knowledgeReader {
	return &knowledgeReader{
		store:     store,
		namespace: namespace,
		limit:     limit,
		owner:     owner,
		recorder:  recorder,
		logger:    logger,
	}
}

// Search answers one question out of the knowledge base.
func (r *knowledgeReader) Search(ctx context.Context, question string) ([]knowledge.Document, error) {
	started := time.Now()
	found, err := r.store.Search(ctx, knowledge.Query{
		Namespace: r.namespace,
		Text:      question,
		Limit:     r.limit,
	})
	r.record(started, err)
	if err != nil {
		return nil, err
	}
	return found, nil
}

// Close releases the recorder, which is what flushes what the lookups cost.
func (r *knowledgeReader) Close() {
	if r.recorder != nil {
		r.recorder.Close()
	}
}

// record files the lookup as a request row, under the same customer and labels as the
// models the conversation used.
func (r *knowledgeReader) record(started time.Time, err error) {
	if r.recorder == nil {
		return
	}

	stat := routing.Stat{
		Owner:     r.owner,
		StartedAt: started.UTC(),
		LatencyMs: routing.MsSince(started),
		Success:   err == nil,
	}
	if err != nil {
		stat.ErrorCode = "knowledge_failed"
	}
	r.recorder.Record(routing.ProviderConfig{
		Provider: r.store.Provider(),
		Model:    knowledgeModel,
	}, stat)
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
	question := strings.TrimSpace(arguments.Query)
	if question == "" {
		return "", false, errors.New("agent: a lookup needs something to look for")
	}

	found, err := a.knowledge.Search(ctx, question)
	if err != nil {
		return "", false, fmt.Errorf("agent: look up %q: %w", question, err)
	}
	a.emitter.Send(LookedUp{TurnID: call.ID, Query: question, Documents: len(found)})
	return knowledge.Prompt(found), false, nil
}
