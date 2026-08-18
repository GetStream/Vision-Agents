package harness

import (
	"embed"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// defaultSkillsFS carries the built-in skill set so the harness works without an external
// file.
//
//go:embed skills.yaml
var defaultSkillsFS embed.FS

// defaultDeadline bounds a skill that declares no deadline of its own. Work the caller is
// waiting through small talk for has stopped being worth having long before this.
const defaultDeadline = 30 * time.Second

// Skill is one kind of work the fast model may hand to the subagent.
//
// It is deliberately not a tool: there is nothing behind a skill but a better model. What
// it declares is the sort of question worth paying that model's latency for, and the
// instructions it answers under.
type Skill struct {
	// Name is how the fast model asks for it.
	Name string `yaml:"name"`
	// Description is what the fast model is told the skill is for, so it knows when to
	// ask. It is the only part of a skill the fast model ever sees.
	Description string `yaml:"description"`
	// Instructions are the subagent's system prompt when it runs the skill.
	Instructions string `yaml:"instructions"`
	// Deadline bounds how long the answer stays worth having.
	Deadline time.Duration `yaml:"deadline"`
}

// Skills is the set a harness was configured with.
type Skills struct {
	Skills []Skill `yaml:"skills"`
}

// Lookup returns a skill by name.
func (s Skills) Lookup(name string) (Skill, bool) {
	for _, skill := range s.Skills {
		if skill.Name == name {
			return skill, true
		}
	}
	return Skill{}, false
}

// Prompt is what the fast model is told about delegating: which skills exist and how to
// ask for one. It is empty when there are no skills, so a harness without them adds
// nothing to the system prompt.
func (s Skills) Prompt() string {
	if len(s.Skills) == 0 {
		return ""
	}

	var prompt strings.Builder
	prompt.WriteString("You have a slower, more capable colleague listening in. " +
		"When something needs more thought than a phone conversation leaves room for, " +
		"hand it over rather than working it out yourself, and keep talking to the " +
		"caller while they do it.\n\n" +
		"To hand something over, write <ask skill=\"name\">what you need</ask> in your " +
		"reply. It is never spoken aloud, so the caller hears only the rest of the " +
		"sentence: say something that fills the pause, like \"let me check that\", and " +
		"put the request beside it. Write <drop skill=\"name\"/> if the caller has " +
		"moved on and the answer no longer matters.\n\n" +
		"What they can take on:\n")
	for _, skill := range s.Skills {
		fmt.Fprintf(&prompt, "- %s: %s\n", skill.Name, skill.Description)
	}
	return strings.TrimRight(prompt.String(), "\n")
}

// Normalize fills in what a skill left out. A skill with no deadline gets one: nothing the
// caller is waiting through small talk for may run forever, and a zero deadline would
// abandon the work before it started.
func (s *Skills) Normalize() {
	for index, skill := range s.Skills {
		if skill.Deadline <= 0 {
			s.Skills[index].Deadline = defaultDeadline
		}
	}
}

// Validate reports the first skill the harness could not use.
func (s Skills) Validate() error {
	seen := map[string]struct{}{}
	for _, skill := range s.Skills {
		if skill.Name == "" {
			return errors.New("harness: every skill needs a name")
		}
		if skill.Description == "" {
			return fmt.Errorf("harness: skill %s has no description, so the model "+
				"would never know when to ask for it", skill.Name)
		}
		if skill.Instructions == "" {
			return fmt.Errorf("harness: skill %s has no instructions", skill.Name)
		}
		if _, duplicate := seen[skill.Name]; duplicate {
			return fmt.Errorf("harness: skill %s is declared twice", skill.Name)
		}
		seen[skill.Name] = struct{}{}
	}
	return nil
}

// DefaultSkills returns the built-in skill set.
func DefaultSkills() (Skills, error) {
	raw, err := defaultSkillsFS.ReadFile("skills.yaml")
	if err != nil {
		return Skills{}, fmt.Errorf("harness: read default skills: %w", err)
	}
	return parseSkills(raw)
}

// LoadSkills reads a skill set, or the built-in default when path is empty.
func LoadSkills(path string) (Skills, error) {
	if path == "" {
		return DefaultSkills()
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		return Skills{}, fmt.Errorf("harness: read skills %s: %w", path, err)
	}
	return parseSkills(raw)
}

func parseSkills(raw []byte) (Skills, error) {
	var skills Skills
	if err := yaml.Unmarshal(raw, &skills); err != nil {
		return Skills{}, fmt.Errorf("harness: parse skills: %w", err)
	}
	if err := skills.Validate(); err != nil {
		return Skills{}, err
	}
	skills.Normalize()
	return skills, nil
}
