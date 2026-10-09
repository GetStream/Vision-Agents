package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// composePrefix names a request for a line the agent says between replies. No audio is
// gated on it: the line, if it is said at all, is said under a turn of its own.
const composePrefix = "compose-"

// composeTimeout is how long a composed line may take. One said late is worse than none:
// by then the moment it was written for has usually passed.
const composeTimeout = 3 * time.Second

// composeNote asks for the line and nothing else. It ends the conversation for that one
// request and is never kept.
const composeNote = "Write only the one short sentence you would say to the caller right now, " +
	"in your own words, fitted to this conversation and unlike anything you have already " +
	"said, with no markup. %s"

// What a composed line is for.
const (
	holdPurpose = "You have just started on what they asked and have said nothing yet: " +
		"tell them what you are doing, so they are not left in silence."
	updatePurpose = "They have heard nothing for a while and what they asked for is still " +
		"being worked on: tell them it is still going, in terms of what they asked for."
	idlePurpose = "Nobody has said anything for a while: check whether they are still " +
		"there or want anything else."
)

// compose asks the reply model for one line to say at a moment the agent chose rather
// than the caller, such as telling someone kept waiting that their answer is on its way.
// The model reads the conversation so far but is given no tools, and nothing of the
// request is kept: the line joins the history only once it has been said.
//
// A native agent has no text model to ask, so it is never given a line.
func (a *Agent) compose(ctx context.Context, purpose string) (string, error) {
	if a.native() {
		return "", nil
	}

	a.mu.Lock()
	if a.closed || a.llm == nil {
		a.mu.Unlock()
		return "", errors.New("agent: not joined")
	}
	history := append(append(withOpenCallsAnswered(a.replayLocked()), a.lateResults...),
		llm.Message{Role: llm.User, Content: fmt.Sprintf(composeNote, purpose)})
	instructions := a.instructions()
	model, overwrites := a.llm, a.options.Overwrites
	a.mu.Unlock()

	ctx, cancel := context.WithTimeout(ctx, composeTimeout)
	defer cancel()
	stream, err := model.Create(ctx, llm.ResponseParams{
		ID:              composePrefix + turnStamp(),
		Purpose:         "compose",
		Instructions:    instructions,
		Input:           history,
		MaxOutputTokens: a.options.MaxTokens,
		PromptCacheKey:  a.options.ConfigID,
	}.Overwrite(overwrites))
	if err != nil {
		return "", err
	}
	response, err := llm.Collect(stream)
	if err != nil {
		return "", err
	}

	var directions tts.Directions
	line := strings.TrimSpace(directions.Add(response.OutputText) + directions.Flush())
	// The line is spoken as it is, past the filter that acts on a reply's skill tags.
	if strings.Contains(line, "<") {
		return "", fmt.Errorf("agent: a composed line carried markup: %q", line)
	}
	return line, nil
}

// withOpenCallsAnswered answers the calls history ends on that are still running, for a
// request sent before they finish: a provider refuses a conversation that replays a call
// without its result.
func withOpenCallsAnswered(history []llm.Message) []llm.Message {
	for _, call := range openCalls(history) {
		history = append(history, llm.Message{Role: llm.ToolResult, ToolCallID: call.ID, Content: stillRunning})
	}
	return history
}
