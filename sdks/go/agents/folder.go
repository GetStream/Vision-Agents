package agents

import (
	"bytes"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// AgentFile is what makes a directory an agent: it names it and says what it runs on.
const AgentFile = "agent.yaml"

// AgentStamp is where a directory records the fingerprint it was last synced under.
const AgentStamp = ".agent_sync"

// InstructionsFile is what an agent directory calls its system prompt.
const InstructionsFile = "instructions.md"

// GuardrailFile is what it calls the policy screening what may be asked of it.
const GuardrailFile = "guardrail.md"

// SkillsDir, KnowledgeDir and SimulationsDir are what it calls the rest.
const (
	SkillsDir      = "skills"
	KnowledgeDir   = "knowledge"
	SimulationsDir = "simulations"
)

// KnowledgeURLsFile is what a knowledge directory calls the pages it is kept filled from,
// as opposed to the files it is filled from directly.
const KnowledgeURLsFile = "urls.yaml"

// readable are the extensions a knowledge directory is read from. Anything else in there is
// left alone: a model looks things up in prose, not in a binary.
var readable = map[string]bool{
	".md": true, ".mdx": true, ".txt": true, ".rst": true, ".yaml": true, ".yml": true,
}

// Document is one file from an agent's knowledge directory, as it will be ingested.
type Document struct {
	// Source is the path relative to the knowledge directory, which is what a passage is
	// keyed and cited by.
	Source string
	Text   string
}

// KnowledgeURL is one page from knowledge/urls.yaml. A page is a subscription rather than
// a copy: what a crawler makes of it is what ends up in the knowledge base.
type KnowledgeURL struct {
	URL string
	// Title and Description are what the declaration says the page is, for a reader of the
	// subscription. Both are optional; a page that says nothing is described by what it
	// called itself when it was last read.
	Title       string
	Description string
	// RefreshHours is how often the backend reads the page again on its own. Zero is never:
	// it is read when it is synced, and when somebody asks.
	RefreshHours int
}

// UnmarshalYAML reads a page written either way: the url on its own, or a mapping naming it
// alongside what it is.
//
//	urls.yaml:
//	    - https://example.com/pricing
//	    - url: https://example.com/plans
//	      title: Plans
//	      description: What each plan includes.
//	      refresh_hours: 24
//
// Unknown keys are refused, so a misspelt one is reported rather than dropped into a
// subscription nobody described.
func (k *KnowledgeURL) UnmarshalYAML(node *yaml.Node) error {
	if node.Kind == yaml.ScalarNode {
		return node.Decode(&k.URL)
	}
	if node.Kind != yaml.MappingNode {
		return errors.New("a page is a url, or a mapping naming one")
	}

	for index := 0; index+1 < len(node.Content); index += 2 {
		key, value := node.Content[index], node.Content[index+1]
		if key.Value == "refresh_hours" {
			if err := value.Decode(&k.RefreshHours); err != nil {
				return err
			}
			if k.RefreshHours < 1 {
				return errors.New("refresh_hours is how many hours between reads, so it is at least 1; leave it out for never")
			}
			continue
		}
		field := map[string]*string{
			"url": &k.URL, "title": &k.Title, "description": &k.Description,
		}[key.Value]
		if field == nil {
			return fmt.Errorf("%q is not something a page says; url, title, description and refresh_hours are", key.Value)
		}
		if err := value.Decode(field); err != nil {
			return err
		}
	}
	return nil
}

// Simulation is one conversation a simulations/*.yaml file declares, run against the agent
// the directory is. A file holds a list of them, so related ones can share a file.
//
//	simulations/lunch.yaml:
//	    - name: lunch order with a change
//	      scenario: Order a turkey club, then swap it for a veggie wrap.
//	      assertion: The final order is one veggie wrap.
//	      variations: 3
type Simulation struct {
	Name         string            `yaml:"name" json:"name"`
	Scenario     string            `yaml:"scenario" json:"scenario"`
	Assertion    string            `yaml:"assertion" json:"assertion"`
	Mode         string            `yaml:"mode" json:"mode"`
	Variations   int               `yaml:"variations" json:"variations"`
	MaxTurns     int               `yaml:"max_turns" json:"max_turns"`
	CallerTarget string            `yaml:"caller_target" json:"caller_target"`
	JudgeTarget  string            `yaml:"judge_target" json:"judge_target"`
	CallerSTT    string            `yaml:"caller_stt" json:"caller_stt"`
	CallerTTS    string            `yaml:"caller_tts" json:"caller_tts"`
	CallerVoice  string            `yaml:"caller_voice" json:"caller_voice"`
	Tags         map[string]string `yaml:"tags" json:"tags"`
}

// Settings is what agent.yaml declares.
//
// The rest of the directory is what the agent is told; this is what it is run with. A
// field left out leaves whatever the config already has stored, so a model chosen in the
// dashboard survives a sync that says nothing about it.
type Settings struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
	Mode        string `yaml:"mode"`
	STT         string `yaml:"stt"`
	TTS         string `yaml:"tts"`
	// STS is nil when the declaration says nothing, and empty when it turns it off.
	STS     *string `yaml:"sts"`
	Voice   string  `yaml:"voice"`
	Speed   float64 `yaml:"speed"`
	LLM     string  `yaml:"llm"`
	Harness string  `yaml:"harness"`
	// ThinkingLLM is the model a voice agent hands its skills to. A text agent runs on its
	// llm alone, and the router refuses one that names it.
	ThinkingLLM string `yaml:"thinking_llm"`
	Search      string `yaml:"search"`
	Greeting    string `yaml:"greeting"`
	Sandbox     string `yaml:"sandbox"`
	// SandboxOptions is how the sandbox is built and how long code may run in it.
	SandboxOptions *SandboxSettings `yaml:"sandbox_options"`
	Plugins        []string         `yaml:"plugins"`
	// UserPlugins are the plugins each end user connects with their own account, in the
	// conversation, rather than the app once for everybody.
	UserPlugins []string `yaml:"user_plugins"`
	// PluginEvents are the MCP events the agent subscribes to on its plugins, each opening
	// a text conversation when it arrives.
	PluginEvents []PluginEventSettings `yaml:"plugin_events"`
	// MCPServers are MCP servers outside the plugin catalog, which the router opens by
	// their URL with no login.
	MCPServers []MCPServerSettings `yaml:"mcp_servers"`
	Keyterms   []string            `yaml:"keyterms"`
	Tags       map[string]string   `yaml:"tags"`
	Video      *VideoSettings      `yaml:"video"`
	Dispatch   *DispatchSettings   `yaml:"dispatch"`
	// App is the application's own section of the declaration, which this SDK never reads
	// and the backend is never sent. It is the one place an unknown key is not refused.
	App map[string]any `yaml:"app"`
}

// PluginEventSettings is one MCP event the agent subscribes to on a plugin it names.
type PluginEventSettings struct {
	Plugin string `yaml:"plugin"`
	Event  string `yaml:"event"`
	// Arguments are the event's filters, as its inputSchema describes them.
	Arguments map[string]any `yaml:"arguments"`
	// Instructions say what the agent does with the event when it arrives.
	Instructions string `yaml:"instructions"`
}

// MCPServerSettings is an MCP server outside the plugin catalog. Name prefixes its tools,
// as <name>__<tool>.
type MCPServerSettings struct {
	Name string `yaml:"name"`
	URL  string `yaml:"url"`
}

// VideoSettings is which video a skill that captures it sees.
type VideoSettings struct {
	Source string `yaml:"source"`
	// MaxFrames is how many recent frames are captured, from 1 to 8. Zero reads as one.
	MaxFrames int `yaml:"max_frames"`
}

// SandboxSettings is how the agent's sandbox is built and how long code may run in it. Any
// of them builds an image on top of Image, which the provider keeps.
type SandboxSettings struct {
	// Image is the container image to start from, which must have Python. Empty is a slim
	// Python image.
	Image string `yaml:"image"`
	// Setup are shell commands run once on top of the image when it is built.
	Setup []string `yaml:"setup"`
	// Timeout is how long one run of code may take, such as 5m, at most 30m. Empty is 30s.
	Timeout string `yaml:"timeout"`
	// CPU, MemoryGB and DiskGB size the sandbox. Zero leaves the provider's default.
	CPU      int `yaml:"cpu"`
	MemoryGB int `yaml:"memory_gb"`
	DiskGB   int `yaml:"disk_gb"`

	timeout time.Duration
}

// DispatchSettings is what the agent leaves to this application's own dispatch worker,
// each "enabled" or "disabled". With Text enabled the model does not answer what end users
// write: the worker is handed it, and answers by creating a response on the session.
type DispatchSettings struct {
	IncomingCall string `yaml:"incoming_call"`
	Text         string `yaml:"text"`
}

// Folder is an agent written down as a directory.
//
//	agents/jean/
//	  agent.yaml
//	  instructions.md
//	  guardrail.md
//	  skills/think.md
//	  knowledge/pricing.md
//	  knowledge/urls.yaml
//	  simulations/lunch.yaml
//
// A skill is a markdown file with YAML-ish frontmatter naming what the fast model sees; the
// body is the prompt only the subagent sees.
type Folder struct {
	// Path is the directory this was read from.
	Path string
	// Name is what agent.yaml calls the agent, or the directory's own name if it does not.
	Name string
	// Declaration is agent.yaml as written, which is what its fingerprint is taken over.
	Declaration string
	// Settings is what agent.yaml declares.
	Settings Settings
	// Instructions is instructions.md, or empty if there is none.
	Instructions string
	// Guardrail is guardrail.md, whole and unparsed, or empty if there is none. The
	// backend parses it, so a policy this SDK has never heard of still reaches it.
	Guardrail string
	// Skills are the files in skills/, in name order.
	Skills []Skill
	// Knowledge are the readable files under knowledge/, in path order.
	Knowledge []Document
	// KnowledgeURLs are the pages knowledge/urls.yaml declares, in the order it lists them.
	KnowledgeURLs []KnowledgeURL
	// Simulations are what simulations/*.yaml declare, by file name and then as listed. Nil
	// when there is no simulations/, which leaves the stored ones alone; empty when it has
	// none, which deletes them.
	Simulations []Simulation
}

// Load reads an agent directory.
//
// agent.yaml is what makes a directory an agent, so it is required. Everything else is
// optional: a directory with only instructions.md beside it is a valid agent, and so is one
// with only skills.
func Load(path string) (*Folder, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("agents: reading %s: %w", path, err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("agents: %s is not an agent directory", path)
	}

	folder := &Folder{Path: path, Name: filepath.Base(filepath.Clean(path))}

	declaration, err := os.ReadFile(filepath.Join(path, AgentFile))
	if errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("agents: %s has no %s, so it is not an agent directory", path, AgentFile)
	}
	if err != nil {
		return nil, fmt.Errorf("agents: reading %s: %w", AgentFile, err)
	}
	folder.Declaration = strings.TrimSpace(string(declaration))
	if folder.Settings, err = declare(declaration); err != nil {
		return nil, fmt.Errorf("agents: %s: %w", filepath.Join(path, AgentFile), err)
	}
	if folder.Settings.Name != "" {
		folder.Name = folder.Settings.Name
	}

	instructions, err := os.ReadFile(filepath.Join(path, InstructionsFile))
	switch {
	case err == nil:
		folder.Instructions = strings.TrimSpace(string(instructions))
	case !errors.Is(err, fs.ErrNotExist):
		return nil, fmt.Errorf("agents: reading %s: %w", InstructionsFile, err)
	}

	policy, err := os.ReadFile(filepath.Join(path, GuardrailFile))
	switch {
	case err == nil:
		folder.Guardrail = strings.TrimSpace(string(policy))
	case !errors.Is(err, fs.ErrNotExist):
		return nil, fmt.Errorf("agents: reading %s: %w", GuardrailFile, err)
	}

	if folder.Skills, err = loadSkills(filepath.Join(path, SkillsDir)); err != nil {
		return nil, err
	}
	if folder.Knowledge, err = loadKnowledge(filepath.Join(path, KnowledgeDir)); err != nil {
		return nil, err
	}
	if folder.KnowledgeURLs, err = loadKnowledgeURLs(filepath.Join(path, KnowledgeDir, KnowledgeURLsFile)); err != nil {
		return nil, err
	}
	if folder.Simulations, err = loadSimulations(filepath.Join(path, SimulationsDir)); err != nil {
		return nil, err
	}
	return folder, nil
}

// fill puts what the directory says into whatever the options left empty. What is written
// in code wins, so a directory is a starting point rather than an override.
func (f *Folder) fill(options *Options) {
	if options.Name == "" {
		options.Name = f.Name
	}
	if options.Instructions == "" {
		options.Instructions = f.Instructions
	}
	if options.Guardrail == "" {
		options.Guardrail = f.Guardrail
	}

	if len(f.Skills) == 0 {
		return
	}
	if options.Harness == nil {
		options.Harness = &Harness{Skills: f.Skills}
		return
	}
	if len(options.Harness.Skills) == 0 {
		// Copied rather than written through, since the caller's harness is a pointer they
		// may be using for another agent too.
		harness := *options.Harness
		harness.Skills = f.Skills
		options.Harness = &harness
	}
}

// KnowledgeNamespace is where the directory's knowledge is looked up, which is the agent's
// own name so two agents never read each other's.
func (f *Folder) KnowledgeNamespace() string {
	if len(f.Knowledge) == 0 && len(f.KnowledgeURLs) == 0 {
		return ""
	}
	return f.Name
}

// Hash is a fingerprint of the directory. The same files produce the same hash, and the
// Python SDK takes it the same way, so a stamp either one wrote is understood by both.
func (f *Folder) Hash() string {
	return fingerprint(f.Declaration, f.Instructions, f.Guardrail, f.Skills, f.Knowledge, f.KnowledgeURLs, f.Simulations)
}

func fingerprint(
	declaration, instructions, guardrail string,
	skills []Skill,
	knowledge []Document,
	pages []KnowledgeURL,
	simulations []Simulation,
) string {
	hasher := md5.New()
	io.WriteString(hasher, declaration+"\n"+instructions+"\n"+guardrail)

	sorted := slices.SortedFunc(slices.Values(skills), func(a, b Skill) int {
		return strings.Compare(a.Name, b.Name)
	})
	for _, skill := range sorted {
		// Written the way Python prints a bool and a float, which is what keeps the two
		// SDKs' fingerprints of one directory the same.
		captured := "False"
		if skill.CaptureVideo {
			captured = "True"
		}
		io.WriteString(hasher, "\nskill:"+skill.Name+"\n"+skill.Description+"\n"+skill.Instructions+captured+"\n")
		if skill.Deadline > 0 {
			seconds := strconv.FormatFloat(skill.Deadline.Seconds(), 'f', -1, 64)
			if !strings.Contains(seconds, ".") {
				seconds += ".0"
			}
			io.WriteString(hasher, seconds)
		}
	}

	documents := slices.SortedFunc(slices.Values(knowledge), func(a, b Document) int {
		return strings.Compare(a.Source, b.Source)
	})
	for _, document := range documents {
		io.WriteString(hasher, "\nknowledge:"+document.Source+"\n"+document.Text)
	}
	for _, page := range pages {
		io.WriteString(hasher, "\nurl:"+page.URL+"\n"+page.Title+"\n"+page.Description)
		// Written only when there is one, so a page without keeps the fingerprint the
		// Python SDK takes of it.
		if page.RefreshHours > 0 {
			io.WriteString(hasher, "\nrefresh_hours:"+strconv.Itoa(page.RefreshHours))
		}
	}
	// Written only when there is a simulations/, so a directory without one keeps the
	// fingerprint the Python SDK takes of it.
	if simulations != nil {
		io.WriteString(hasher, "\nsimulations:")
		for _, simulation := range simulations {
			encoded, _ := json.Marshal(simulation)
			hasher.Write(encoded)
		}
	}
	return hex.EncodeToString(hasher.Sum(nil))
}

// syncStamp is what .agent_sync holds.
type syncStamp struct {
	Hash     string `json:"hash"`
	SyncedAt string `json:"synced_at"`
}

// ReadStamp is the fingerprint a directory was last synced under, or empty when it never
// was or the stamp cannot be read.
func ReadStamp(path string) string {
	raw, err := os.ReadFile(filepath.Join(path, AgentStamp))
	if err != nil {
		return ""
	}
	var recorded syncStamp
	if json.Unmarshal(raw, &recorded) != nil {
		return ""
	}
	return recorded.Hash
}

// WriteStamp records what was synced and when, so a second sync can do nothing.
func WriteStamp(path, hash string) error {
	recorded, err := json.Marshal(syncStamp{
		Hash:     hash,
		SyncedAt: time.Now().UTC().Truncate(time.Second).Format("2006-01-02T15:04:05+00:00"),
	})
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(path, AgentStamp), append(recorded, '\n'), 0o644); err != nil {
		return fmt.Errorf("agents: writing %s: %w", AgentStamp, err)
	}
	return nil
}

// declare reads agent.yaml. A key nobody knows is refused rather than dropped, since a
// misspelled llm that goes quietly is a config running on a model the file does not name.
func declare(raw []byte) (Settings, error) {
	var settings Settings
	decoder := yaml.NewDecoder(bytes.NewReader(raw))
	decoder.KnownFields(true)
	if err := decoder.Decode(&settings); err != nil && !errors.Is(err, io.EOF) {
		return Settings{}, err
	}
	if settings.Video != nil {
		if settings.Video.MaxFrames == 0 {
			settings.Video.MaxFrames = 1
		}
		if settings.Video.MaxFrames < 1 || settings.Video.MaxFrames > 8 {
			return Settings{}, errors.New("video.max_frames must be an integer from 1 to 8")
		}
	}
	if options := settings.SandboxOptions; options != nil && options.Timeout != "" {
		timeout, err := time.ParseDuration(options.Timeout)
		if err != nil || timeout <= 0 || timeout > 30*time.Minute {
			return Settings{}, fmt.Errorf("sandbox_options.timeout must be a duration up to 30m, not %q", options.Timeout)
		}
		options.timeout = timeout
	}
	return settings, nil
}

func loadSkills(path string) ([]Skill, error) {
	entries, err := os.ReadDir(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("agents: reading %s: %w", path, err)
	}

	var skills []Skill
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".md" {
			continue
		}

		file := filepath.Join(path, entry.Name())
		content, err := os.ReadFile(file)
		if err != nil {
			return nil, fmt.Errorf("agents: reading %s: %w", file, err)
		}

		skill, err := parseSkill(strings.TrimSuffix(entry.Name(), ".md"), string(content))
		if err != nil {
			return nil, fmt.Errorf("agents: %s: %w", file, err)
		}
		skills = append(skills, skill)
	}
	return skills, nil
}

// parseSkill reads a skill file: frontmatter between --- lines, then the instructions.
//
// The recognised keys are name, description and deadline. A deadline is a Go duration, so
// "30s" and "2m" both read the way they look.
func parseSkill(name, content string) (Skill, error) {
	skill := Skill{Name: name}

	frontmatter, body, found := cutFrontmatter(content)
	if found {
		for line := range strings.SplitSeq(frontmatter, "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			key, value, ok := strings.Cut(line, ":")
			if !ok {
				return skill, fmt.Errorf("%q is not a key and a value", line)
			}
			value = strings.Trim(strings.TrimSpace(value), `"'`)

			switch strings.TrimSpace(key) {
			case "name":
				skill.Name = value
			case "description":
				skill.Description = value
			case "capture_video":
				if value != "true" && value != "false" {
					return Skill{}, fmt.Errorf("capture_video must be true or false")
				}
				skill.CaptureVideo = value == "true"
			case "deadline":
				deadline, err := parseDeadline(value)
				if err != nil {
					return skill, err
				}
				skill.Deadline = deadline
			}
		}
	}

	skill.Instructions = strings.TrimSpace(body)
	if skill.Description == "" {
		return skill, errors.New("a skill needs a description, since it is all the fast model sees")
	}
	if skill.Instructions == "" {
		return skill, errors.New("a skill needs instructions, since they are what the subagent answers under")
	}
	return skill, nil
}

// parseDeadline takes a Go duration, and a bare number as seconds.
func parseDeadline(value string) (time.Duration, error) {
	if seconds, err := strconv.ParseFloat(value, 64); err == nil {
		return time.Duration(seconds * float64(time.Second)), nil
	}
	deadline, err := time.ParseDuration(value)
	if err != nil {
		return 0, fmt.Errorf("%q is not a deadline", value)
	}
	return deadline, nil
}

// cutFrontmatter separates a leading --- block from the body.
func cutFrontmatter(content string) (frontmatter, body string, found bool) {
	trimmed := strings.TrimLeft(content, "\ufeff \t\r\n")
	if !strings.HasPrefix(trimmed, "---") {
		return "", content, false
	}

	rest := strings.TrimPrefix(trimmed, "---")
	rest = strings.TrimLeft(rest, "\r\n")
	frontmatter, body, found = strings.Cut(rest, "\n---")
	if !found {
		return "", content, false
	}
	return frontmatter, strings.TrimLeft(body, "-\r\n"), true
}

func loadKnowledge(path string) ([]Document, error) {
	root, err := os.Stat(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("agents: reading %s: %w", path, err)
	}
	if !root.IsDir() {
		return nil, fmt.Errorf("agents: %s is not a directory", path)
	}

	var documents []Document
	err = filepath.WalkDir(path, func(file string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() || !readable[strings.ToLower(filepath.Ext(entry.Name()))] {
			return nil
		}
		// The declaration of what pages to read is not itself something to look things up
		// in. Only the one at the root is the declaration; deeper, urls.yaml is a document
		// like any other.
		if file == filepath.Join(path, KnowledgeURLsFile) {
			return nil
		}

		content, err := os.ReadFile(file)
		if err != nil {
			return err
		}
		if strings.TrimSpace(string(content)) == "" {
			return nil
		}

		source, err := filepath.Rel(path, file)
		if err != nil {
			return err
		}
		documents = append(documents, Document{
			Source: filepath.ToSlash(source),
			Text:   string(content),
		})
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("agents: reading %s: %w", path, err)
	}
	return documents, nil
}

// loadKnowledgeURLs reads the pages a knowledge base is kept filled from.
//
// A bad url is refused here rather than when it is subscribed, since a directory that
// cannot be turned into a knowledge base is worth hearing about before anything is written.
func loadKnowledgeURLs(path string) ([]KnowledgeURL, error) {
	raw, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("agents: reading %s: %w", path, err)
	}

	var pages []KnowledgeURL
	if err := yaml.Unmarshal(raw, &pages); err != nil {
		return nil, fmt.Errorf("agents: %s: %w", path, err)
	}
	for _, page := range pages {
		if !strings.HasPrefix(page.URL, "http://") && !strings.HasPrefix(page.URL, "https://") {
			return nil, fmt.Errorf("agents: %s: %q is not an http or https url", path, page.URL)
		}
	}
	return pages, nil
}

// loadSimulations reads every .yaml and .yml file in simulations/, each a list of
// simulations. A key nobody knows is refused, as in agent.yaml, and so is a name two
// simulations share, since a sync finds a simulation by its name.
func loadSimulations(path string) ([]Simulation, error) {
	entries, err := os.ReadDir(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("agents: reading %s: %w", path, err)
	}

	simulations := []Simulation{}
	named := map[string]string{}
	for _, entry := range entries {
		extension := strings.ToLower(filepath.Ext(entry.Name()))
		if entry.IsDir() || (extension != ".yaml" && extension != ".yml") {
			continue
		}

		file := filepath.Join(path, entry.Name())
		raw, err := os.ReadFile(file)
		if err != nil {
			return nil, fmt.Errorf("agents: reading %s: %w", file, err)
		}
		var listed []Simulation
		decoder := yaml.NewDecoder(bytes.NewReader(raw))
		decoder.KnownFields(true)
		if err := decoder.Decode(&listed); err != nil && !errors.Is(err, io.EOF) {
			return nil, fmt.Errorf("agents: %s: %w", file, err)
		}

		for _, simulation := range listed {
			switch {
			case simulation.Name == "":
				return nil, fmt.Errorf("agents: %s: a simulation needs a name", file)
			case simulation.Scenario == "":
				return nil, fmt.Errorf("agents: %s: simulation %q needs a scenario", file, simulation.Name)
			case simulation.Assertion == "":
				return nil, fmt.Errorf("agents: %s: simulation %q needs an assertion", file, simulation.Name)
			case simulation.Mode != "" && simulation.Mode != "text" && simulation.Mode != "audio":
				return nil, fmt.Errorf("agents: %s: simulation %q is text or audio, not %q", file, simulation.Name, simulation.Mode)
			}
			if first, taken := named[simulation.Name]; taken {
				return nil, fmt.Errorf("agents: %s: simulation %q is also declared in %s", file, simulation.Name, first)
			}
			named[simulation.Name] = entry.Name()
			simulations = append(simulations, simulation)
		}
	}
	return simulations, nil
}
