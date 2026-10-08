package agents

import (
	"errors"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// Sandbox is where code the agent writes gets run.
//
// Code execution never happens on the live speech path: a sandbox is offered to the slower
// model doing delegated work, not to the one holding the conversation.
type Sandbox struct {
	// Provider is the sandbox provider's name, as the backend knows it.
	Provider string
}

// Daytona is a Daytona sandbox. The backend needs DAYTONA_API_KEY for it to do anything.
func Daytona() *Sandbox {
	return &Sandbox{Provider: "daytona"}
}

// Skill is a kind of work worth handing to the slower model.
//
// There is nothing behind a skill but a better model and more time. What it declares is the
// description the fast model chooses by, and the instructions the slow one answers under.
type Skill struct {
	CaptureVideo bool
	// Name is how the fast model asks for it.
	Name string
	// Description is the one line the fast model sees.
	Description string
	// Instructions is the full prompt, which only the subagent sees.
	Instructions string
	// Deadline is how long the work may run before it is abandoned. Zero leaves the
	// backend's default.
	Deadline time.Duration
}

// Harness is what stands between what a caller said and the model that answers them.
//
// The loop runs in the backend and is part of the agent's stored config, never of a
// session: Sync writes it, and every session created from the config runs it. That backend
// hands work to the subagent, loads a skill's instructions when the skill is used, compacts
// the conversation as it nears the model's context window and starts the sandbox the first
// time delegated work needs one.
type Harness struct {
	// Name is which harness the backend runs. Empty is "default", the only one there is.
	Name string
	// Subagents are model targets for the work handed over, keyed by name. The entry under
	// "default", or the only entry, is the model that runs skills. Empty means the fast
	// model answers everything itself.
	Subagents map[string]string
	// VM is where delegated code runs.
	VM *Sandbox
	// Skills of your own, stored and named by the config in place of the built-in set.
	Skills []Skill
}

// DefaultHarness is the harness most agents want: the built-in skills and nothing else
// changed.
func DefaultHarness() *Harness {
	return &Harness{}
}

// Subagent is the model that runs delegated work, or the empty string when nothing is
// delegated.
func (h *Harness) Subagent() string {
	if h == nil || len(h.Subagents) == 0 {
		return ""
	}
	if named, ok := h.Subagents["default"]; ok {
		return named
	}
	// Go randomises map iteration, so the single-entry shorthand is only well defined for
	// one entry. More than one without a default is a configuration mistake, caught by
	// Validate before it can pick differently on two runs.
	for _, target := range h.Subagents {
		return target
	}
	return ""
}

// Validate refuses a harness that would mean something different on every run.
func (h *Harness) Validate() error {
	if h == nil {
		return nil
	}
	if h.Name != "" && !acceleration.Harness(h.Name).Valid() {
		return errors.New("agents: there is no harness called " + h.Name)
	}
	if len(h.Subagents) > 1 {
		if _, ok := h.Subagents["default"]; !ok {
			return errors.New(`agents: several subagents and no "default", so which one runs skills is undecided`)
		}
	}
	for _, skill := range h.Skills {
		if skill.Name == "" {
			return errors.New("agents: a skill needs a name")
		}
		if skill.Description == "" {
			return errors.New("agents: " + skill.Name + " needs a description, since it is all the fast model sees")
		}
		if skill.Instructions == "" {
			return errors.New("agents: " + skill.Name + " needs instructions, since they are what the subagent answers under")
		}
	}
	if h.VM != nil && h.VM.Provider == "" {
		return errors.New("agents: a sandbox needs a provider")
	}
	return nil
}

// stored is what the harness sets on the agent's config: its name, subagent and sandbox.
// Empty fields are left out, so the router keeps whatever is already stored for them.
func (h *Harness) stored() (name *acceleration.Harness, subagent *string, sandbox *acceleration.Sandbox) {
	if h == nil {
		return nil, nil, nil
	}
	if h.Name != "" {
		named := acceleration.Harness(h.Name)
		name = &named
	}
	setString(&subagent, h.Subagent())
	if h.VM != nil {
		provider := acceleration.Sandbox(h.VM.Provider)
		sandbox = &provider
	}
	return name, subagent, sandbox
}
