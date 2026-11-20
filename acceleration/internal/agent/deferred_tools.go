package agent

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// defers reports whether the model finds and loads some of its tools as it needs them
// (harness.Tools.Deferred). A speech-to-speech model is offered its tools once, when it
// opens, so it is offered the deferred ones whole instead (offeredDeferred).
func (a *Agent) defers() bool {
	return len(a.tools().Deferred) > 0 && !a.native()
}

// offeredDeferred is what the model is offered for its deferred tools: search_tool,
// load_tool and the tools it has loaded, in the order it loaded them, so a load adds to the
// end of what it is offered and leaves the rest in place. A speech-to-speech model gets them
// all.
func (a *Agent) offeredDeferred() []harness.Tool {
	deferred := a.tools().Deferred
	if len(deferred) == 0 {
		return nil
	}
	if a.native() {
		return deferred
	}
	offered := []harness.Tool{harness.SearchTool(), harness.LoadTool()}
	for _, name := range a.loaded() {
		if at := slices.IndexFunc(deferred, func(tool harness.Tool) bool { return tool.Name == name }); at >= 0 {
			offered = append(offered, deferred[at])
		}
	}
	return offered
}

// runDeferred answers search_tool and load_tool.
func (a *Agent) runDeferred(call llm.ToolCall) (string, error) {
	deferred := a.tools().Deferred
	if call.Name == harness.SearchToolName {
		var arguments struct {
			Query string `json:"query"`
		}
		if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil {
			return "", fmt.Errorf("agent: could not read the search_tool arguments: %w", err)
		}
		return harness.SearchDeferred(deferred, arguments.Query), nil
	}
	result, found := harness.LoadDeferred(deferred, harness.LoadedNames(call.Arguments))
	if err := a.load(found); err != nil {
		return "", err
	}
	return result, nil
}

// load adds names to the tools the model has loaded and offers them from its next request,
// the next step of the turn that loaded them included: the harness reads its tools on every
// request, so they are set on it directly rather than through SetTools, which waits for the
// turn to finish and so for the very call that asked.
func (a *Agent) load(names []string) error {
	a.toolsMu.Lock()
	added := false
	for _, name := range names {
		if !slices.Contains(a.loadedTools, name) {
			a.loadedTools = append(a.loadedTools, name)
			added = true
		}
	}
	a.toolsMu.Unlock()
	if !added {
		return nil
	}
	a.mu.Lock()
	current := a.harness
	a.mu.Unlock()
	if current == nil {
		return nil
	}
	return current.SetTools(a.availableTools())
}

// offerLoaded is load for a conversation carried on, which nothing answers a failure to:
// the tools stay as they were, and the model can load them again.
func (a *Agent) offerLoaded(names []string) {
	if err := a.load(names); err != nil {
		a.logger.Warn("could not offer the tools the conversation had loaded", "tools", names, "error", err)
	}
}

// loaded are the deferred tools the model has loaded, in the order it loaded them.
func (a *Agent) loaded() []string {
	a.toolsMu.RLock()
	defer a.toolsMu.RUnlock()
	return slices.Clone(a.loadedTools)
}

// loadedIn are the tools the load_tool calls in history asked for, in the order asked, so a
// conversation carried on is offered what it had loaded. A name that is no deferred tool is
// dropped where it is offered (offeredDeferred).
func loadedIn(history []llm.Message) []string {
	var names []string
	for _, message := range history {
		for _, call := range message.ToolCalls {
			if call.Name != harness.LoadToolName {
				continue
			}
			for _, name := range harness.LoadedNames(call.Arguments) {
				if !slices.Contains(names, name) {
					names = append(names, name)
				}
			}
		}
	}
	return names
}
