package agents

import (
	"errors"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// Sandbox is where code the agent writes gets run.
//
// Code execution never happens on the live speech path: a sandbox is offered to the slower
// model doing delegated work, not to the one holding the conversation.
type Sandbox struct {
	// Provider is the sandbox provider's name, as the backend knows it.
	Provider string
	// Profile selects a backend-managed research workspace instead of Python execution.
	Profile string
}

// Daytona is a Daytona sandbox. The backend needs DAYTONA_API_KEY for it to do anything.
func ManagedSandbox(name string) *Sandbox { return &Sandbox{Profile: name} }

func Daytona() *Sandbox {
	return &Sandbox{Provider: "daytona"}
}

// Skill is a kind of work worth handing to the slower model.
//
// There is nothing behind a skill but a better model and more time. What it declares is the
// description the fast model chooses by, and the instructions the slow one answers under.
type Skill struct {
	Subagent     string
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
// The loop itself runs in the backend, so this is configuration rather than behaviour: it
// is serialized into the session and the decisions are taken there. Reimplementing the loop
// here would mean two of them to keep in step.
type Harness struct {
	// UseSkills offers the backend's built-in skills. Setting Skills replaces them.
	UseSkills bool
	// Subagents are model targets for the work handed over, keyed by name. The entry under
	// "default" runs unbound skills; other entries run skills bound to that name.
	Subagents map[string]string
	// VM is where delegated code runs.
	VM *Sandbox
	// Skills of your own, replacing the built-in set.
	Skills []Skill
	// Tasks is how much delegated work may run at once. Zero leaves the backend's default.
	Tasks int
}

// DefaultHarness is the harness most agents want: the built-in skills and nothing else
// changed.
func DefaultHarness() *Harness {
	return &Harness{UseSkills: true}
}

// Subagent is the default worker's model, or the empty string when none is configured.
func (h *Harness) Subagent() string {
	if h == nil || len(h.Subagents) == 0 {
		return ""
	}
	return h.Subagents["default"]
}

// Validate refuses a harness that would mean something different on every run.
func (h *Harness) Validate() error {
	if h == nil {
		return nil
	}
	if h.Tasks < 0 {
		return errors.New("agents: tasks cannot be negative")
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
	if h.VM != nil && h.VM.Provider != "" && h.VM.Profile != "" {
		return errors.New("agents: choose Python sandbox or managed research profile")
	}
	if h.VM != nil && h.VM.Provider == "" && h.VM.Profile == "" {
		return errors.New("agents: a sandbox needs a provider")
	}
	return nil
}

// ReplacesSkills reports whether the harness turns the built-in set off, either by naming
// skills of its own or by asking for none. An absent list and an empty one mean different
// things: one leaves the defaults alone, the other turns delegation off.
func (h *Harness) ReplacesSkills() bool {
	return h != nil && (len(h.Skills) > 0 || !h.UseSkills)
}

// apply folds the harness into the call it is configuring.
func (h *Harness) apply(call *stream.Call) {
	if h == nil {
		return
	}

	call.Subagents = h.Subagents
	call.Tasks = h.Tasks
	if h.VM != nil {
		call.Sandbox = h.VM.Provider
		call.SandboxProfile = h.VM.Profile
	}
	if h.ReplacesSkills() {
		replacements := make([]acceleration.SessionSkill, 0, len(h.Skills))
		for _, skill := range h.Skills {
			replacements = append(replacements, skill.session())
		}
		call.Skills = &replacements
	}
}

// session renders a skill as the session spec understands it.
func (s Skill) session() acceleration.SessionSkill {
	rendered := acceleration.SessionSkill{
		Name:     s.Name,
		Subagent: &s.Subagent, CaptureVideo: &s.CaptureVideo,
		Description:  s.Description,
		Instructions: s.Instructions,
	}
	if s.Deadline > 0 {
		milliseconds := s.Deadline.Milliseconds()
		rendered.DeadlineMs = &milliseconds
	}
	return rendered
}
