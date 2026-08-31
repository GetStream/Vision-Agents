package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
)

// searchArguments is what the model fills in to find something out.
type searchArguments struct {
	Query string `json:"query"`
}

// search finds something out and hands what was found to the model.
//
// Like a lookup and unlike the memory writer it is on the live path: the caller is waiting
// on the answer, which is the whole reason the model asked. What bounds it is the
// provider's own timeout, and what records it is the session, which knows which provider
// routing chose.
//
// A question nothing covers is answered in words rather than an error, for the same reason
// a lookup is: the model is told the search found nothing, which is what stops it inventing
// an answer or apologising for a tool the caller never knew about.
func (a *Agent) search(ctx context.Context, call llm.ToolCall) (string, bool, error) {
	var arguments searchArguments
	if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil {
		return "", false, fmt.Errorf("agent: could not read the search arguments: %w", err)
	}
	question := strings.TrimSpace(arguments.Query)
	if question == "" {
		return "", false, errors.New("agent: a search needs something to look for")
	}

	found, err := a.searcher.Search(ctx, search.Query{Text: question})
	if err != nil {
		return "", false, fmt.Errorf("agent: search %q: %w", question, err)
	}
	a.emitter.Send(Searched{TurnID: call.ID, Query: question, Results: len(found.Documents)})
	return search.Prompt(found), false, nil
}

// startSearching opens the search session, so the provider routing chose is settled before
// the model is offered the tool rather than while somebody waits on the answer.
//
// Nothing starting is not a reason to abandon the call, unlike the three modalities a
// conversation cannot happen without: the tool is simply not offered, and the agent answers
// out of what it knows the way it did before any of this existed.
func (a *Agent) startSearching(ctx context.Context) {
	if a.options.Search == nil || a.options.SearchTarget == "" {
		return
	}

	session, err := a.options.Search.Start(ctx, searchrouter.Request{
		CustomerID:    a.options.CustomerID,
		AgentID:       a.options.AgentID,
		CallID:        a.options.CallID,
		Tags:          a.options.Tags,
		Target:        a.options.SearchTarget,
		LanguageHints: a.options.LanguageHints,
	})
	if err != nil {
		a.logger.Warn("this agent cannot find out what is true today",
			"target", a.options.SearchTarget, "error", err)
		return
	}
	a.searcher = session
}
