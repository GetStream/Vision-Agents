package harness

import (
	"context"
	"embed"
	"errors"
	"fmt"
	"maps"
	"os"
	"path"
	"slices"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
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
	VideoSource  string `yaml:"-"`
	VideoFrames  int    `yaml:"-"`
	Revision     int64  `yaml:"-"`
	CaptureVideo bool   `yaml:"capture_video"`
	// Name is how the fast model asks for it.
	Name string `yaml:"name"`
	// Description is what the fast model is told the skill is for, so it knows when to
	// ask. It is the only part of a skill the fast model ever sees.
	Description string `yaml:"description"`
	// Instructions are the subagent's system prompt when it runs the skill.
	Instructions string `yaml:"instructions"`
	// Deadline bounds how long the answer stays worth having.
	Deadline time.Duration `yaml:"deadline"`
	// Files are the paths of the rest of the skill's folder, which the model reads one at a
	// time when it needs them, and runs in the sandbox when they are scripts.
	Files []string `yaml:"-"`
}

// Skills is the set a harness was configured with.
type Skills struct {
	Skills []Skill `yaml:"skills"`
	// Load reads a skill at the moment the subagent runs it, for a skill held by its name
	// and description alone. The fast model only ever sees the index, so nothing more is
	// read until a skill is used, and a skill edited mid-conversation is used as it now is.
	// Nil means every skill carries its own instructions and has no files.
	Load func(ctx context.Context, name string) (Loaded, error) `yaml:"-"`
}

// Loaded is a skill as read at the moment it is used.
type Loaded struct {
	Instructions string
	// Files are the rest of the skill's folder, by path relative to it.
	Files map[string]string
}

// documents are the extensions a skill's file is read by the model rather than run.
var documents = map[string]bool{".md": true, ".mdx": true, ".txt": true}

// Scripts are the files of a skill that are run rather than read, in path order.
func Scripts(paths []string) []string {
	var scripts []string
	for _, file := range paths {
		if !documents[strings.ToLower(path.Ext(file))] {
			scripts = append(scripts, file)
		}
	}
	slices.Sort(scripts)
	return scripts
}

// NeedsSandbox refuses a skill that brings scripts to an agent with nowhere to run them.
func NeedsSandbox(name string, paths []string, sandboxed bool) error {
	scripts := Scripts(paths)
	if sandboxed || len(scripts) == 0 {
		return nil
	}
	return fmt.Errorf("skill %q runs scripts (%s), so the agent needs a sandbox: set sandbox: daytona in agent.yaml",
		name, strings.Join(scripts, ", "))
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
		"Hand work over only when a value looks truncated, mashed, or you cannot " +
		"tell what they said. If the caller already gave complete identifiers, " +
		"answer yourself and call any tool you need; do not wait for a colleague " +
		"on a complete thought.\n\n" +
		"To hand something over, write <ask skill=\"name\">what you need</ask> in your " +
		"reply. It is never spoken aloud, so the caller hears only the rest of the " +
		"sentence: say in your own words what you are getting for them, and put the " +
		"request beside it. Write <drop skill=\"name\"/> if the caller has " +
		"moved on and the answer no longer matters.\n\n" +
		"What they can take on:\n")
	for _, skill := range s.Skills {
		fmt.Fprintf(&prompt, "- %s: %s\n", skill.Name, skill.Description)
		if skill.CaptureVideo {
			prompt.WriteString("  Optional capture attributes: source=\"participant/track\" and frames=\"2\". Request 2–8 recent frames for comparisons over time; omit for the session default. Use only source names reported by capture or the video state tool.\n")
		}
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
			return stack.Wrap(errors.New("harness: every skill needs a name"))
		}
		if skill.Description == "" {
			return stack.Wrap(fmt.Errorf("harness: skill %s has no description, so the model "+
				"would never know when to ask for it", skill.Name))
		}
		if skill.Instructions == "" && s.Load == nil {
			return stack.Wrap(fmt.Errorf("harness: skill %s has no instructions", skill.Name))
		}
		if _, duplicate := seen[skill.Name]; duplicate {
			return stack.Wrap(fmt.Errorf("harness: skill %s is declared twice", skill.Name))
		}
		seen[skill.Name] = struct{}{}
	}
	return nil
}

// DefaultSkills returns the built-in skill set.
func DefaultSkills() (Skills, error) {
	raw, err := defaultSkillsFS.ReadFile("skills.yaml")
	if err != nil {
		return Skills{}, stack.Wrap(fmt.Errorf("harness: read default skills: %w", err))
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
		return Skills{}, stack.Wrap(fmt.Errorf("harness: parse skills: %w", err))
	}
	if err := skills.Validate(); err != nil {
		return Skills{}, err
	}
	skills.Normalize()
	return skills, nil
}

// LoadSkillTool is the tool a written conversation loads a skill with. A text session runs
// on one model with every tool, so a skill there is instructions that model reads and then
// acts on, rather than work handed to a colleague who can call nothing.
const LoadSkillTool = "load_skill"

// Instructions returns what a skill says to do, reading a stored skill at the moment it is
// asked for, so one edited mid-conversation is used as it now is.
func (s Skills) Instructions(ctx context.Context, name string) (string, error) {
	loaded, err := s.Read(ctx, name)
	return loaded.Instructions, err
}

// Read returns a skill's instructions and files, reading a stored skill at the moment it is
// asked for.
func (s Skills) Read(ctx context.Context, name string) (Loaded, error) {
	skill, found := s.Lookup(name)
	if !found {
		names := make([]string, 0, len(s.Skills))
		for _, known := range s.Skills {
			names = append(names, known.Name)
		}
		return Loaded{}, fmt.Errorf("harness: there is no skill called %q; use one of: %s", name, strings.Join(names, ", "))
	}
	if skill.Instructions != "" || s.Load == nil {
		return Loaded{Instructions: skill.Instructions}, nil
	}
	return s.Load(ctx, name)
}

// File returns one of the files a skill brought with it.
func (s Skills) File(ctx context.Context, name, file string) (string, error) {
	loaded, err := s.Read(ctx, name)
	if err != nil {
		return "", err
	}
	return bundled(loaded.Files, file)
}

// ReadSkillFileTool is the tool the subagent reads a skill's own files with, one at a time
// and only when it decides it needs one.
const ReadSkillFileTool = "read_skill_file"

// BundlePath is where a skill's file is put in the sandbox, relative to where code runs.
func BundlePath(skill, file string) string {
	return path.Join("skills", skill, file)
}

// Listing tells the model which files a skill brought, so it can read the ones it needs
// rather than being handed them all.
func Listing(skill string, paths []string, sandboxed bool) string {
	var listing strings.Builder
	listing.WriteString("This skill comes with these files. Read one only when you need it.")
	if sandboxed && len(Scripts(paths)) > 0 {
		fmt.Fprintf(&listing, " In the sandbox they are under %s/, so run the scripts from there.", BundlePath(skill, ""))
	}
	for _, file := range paths {
		listing.WriteString("\n- " + file)
	}
	return listing.String()
}

// bundled is one file out of a skill's folder, or an error naming the ones there are.
func bundled(files map[string]string, file string) (string, error) {
	if content, ok := files[strings.TrimPrefix(path.Clean(file), "./")]; ok {
		return content, nil
	}
	if len(files) == 0 {
		return "", fmt.Errorf("harness: this skill has no files")
	}
	return "", fmt.Errorf("harness: the skill has no file %q; it has: %s", file,
		strings.Join(slices.Sorted(maps.Keys(files)), ", "))
}

// readFileTool is read_skill_file as the subagent is offered it, naming the files there are.
func readFileTool(paths []string) llm.Tool {
	return llm.Tool{
		Name:        ReadSkillFileTool,
		Description: "Read one of the files the skill came with, such as a reference it points you to.",
		Parameters: map[string]any{
			"type":     "object",
			"required": []string{"path"},
			"properties": map[string]any{
				"path": map[string]any{"type": "string", "enum": paths},
			},
		},
	}
}

// LoadTool is load_skill as the model is offered it: each skill by name and by the
// description that is the only thing it can choose one by.
func (s Skills) LoadTool() Tool {
	names := make([]string, 0, len(s.Skills))
	var description strings.Builder
	description.WriteString("Load a skill's instructions and follow them yourself. When a request " +
		"matches a skill's description, load it before any other tool: it says how to answer " +
		"and which sources to use. When no description matches, load none. A skill that lists " +
		"files of its own: load it again with path to read one, only when you need it. Skills:")
	for _, skill := range s.Skills {
		names = append(names, skill.Name)
		fmt.Fprintf(&description, "\n- %s: %s", skill.Name, skill.Description)
	}
	return Tool{
		Name:        LoadSkillTool,
		Description: description.String(),
		Parameters: map[string]any{
			"type":     "object",
			"required": []string{"name"},
			"properties": map[string]any{
				"name": map[string]any{"type": "string", "enum": names},
				"path": map[string]any{"type": "string", "description": "One of the files the skill listed, to read it instead of the instructions."},
			},
		},
	}
}

// TextPrompt offers native skills for matching written requests, including complete questions.
func (s Skills) TextPrompt() string {
	if len(s.Skills) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("Use the appropriate skill when a written request matches its description or explicitly names it. To run a skill, write <ask skill=\"name\">the task and relevant context</ask>. The harness runs this directive privately and returns the result; do not merely claim to have used the skill. Give a brief visible progress update, then wait for the result before answering. Do not re-delegate a result you just received. Use <drop skill=\"name\"/> to cancel work no longer needed. Available skills:\n")
	for _, skill := range s.Skills {
		fmt.Fprintf(&b, "- %s: %s\n", skill.Name, skill.Description)
	}
	return strings.TrimSpace(b.String())
}
