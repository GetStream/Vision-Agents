package simulation

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
)

// expandTimeout bounds the rewrite, and expandTokens bounds it. Ten ways of asking one
// thing is a paragraph, not an essay.
const (
	expandTimeout = 60 * time.Second
	expandTokens  = 1500
)

// expandInstructions is what the rewriter is told to be.
//
// The trap is helpfulness: asked for ten ways of asking the same thing, a model will
// happily give ten different things to ask. Every fact in the scenario has to survive, or
// the assertion is being checked against a conversation it was not written for.
const expandInstructions = `You are given a scenario somebody wrote to test an automated phone agent. Rewrite it as different ways the same call could go.

Keep every step, in the same order, and every specific fact exactly as written: the same items, the same times, the same names, the same numbers. If the scenario says 8pm, every rewrite says 8pm.

Vary only how it is asked: the wording, how blunt or chatty the caller is, how much they say per turn, whether they volunteer details or wait to be asked, and how sure of themselves they sound. Some callers should be awkward. None should be impossible.

Answer with JSON only: {"variations": ["...", "..."]}`

// expand asks a model for other ways of asking the same thing.
//
// The scenario as written is always the first way of asking and is not produced here, so
// this is asked for one fewer than the run wants. Coming back with fewer than that is used
// as it is: ten is what was hoped for, not a contract, and nine conversations answer the
// question nearly as well as ten.
func expand(
	ctx context.Context,
	router *llmrouter.Router,
	request llmrouter.Request,
	id, scenario string,
	count int,
) ([]string, error) {
	if count <= 0 {
		return nil, nil
	}
	if request.Target == "" {
		request.Target = defaultCallerTarget
	}

	ctx, cancel := context.WithTimeout(ctx, expandTimeout)
	defer cancel()

	session, err := router.Start(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("simulation: open the rewriter: %w", err)
	}
	defer session.Close()

	asked := fmt.Sprintf("Give %d rewrites of this scenario.\n\nThe scenario:\n\n%s", count, scenario)
	err = session.Respond(llm.Request{
		ID:           id,
		Instructions: expandInstructions,
		Messages:     []llm.Message{{Role: llm.User, Content: asked}},
		MaxTokens:    expandTokens,
		JSON:         true,
	})
	if err != nil {
		return nil, err
	}

	answer, err := llmrouter.Await(ctx, session, id)
	if err != nil {
		return nil, err
	}
	return parseVariations(answer, count)
}

func parseVariations(answer string, count int) ([]string, error) {
	var rewritten struct {
		Variations []string `json:"variations"`
	}
	if err := json.Unmarshal([]byte(llm.Unfence(answer)), &rewritten); err != nil {
		return nil, fmt.Errorf("simulation: decode the rewrites: %w", err)
	}

	kept := make([]string, 0, count)
	for _, variation := range rewritten.Variations {
		trimmed := strings.TrimSpace(variation)
		if trimmed == "" {
			continue
		}
		kept = append(kept, trimmed)
		if len(kept) == count {
			break
		}
	}
	return kept, nil
}
