package agents

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// InstructionsFile is what an agent directory calls its system prompt.
const InstructionsFile = "instructions.md"

// SkillsDir and KnowledgeDir are what it calls the rest.
const (
	SkillsDir    = "skills"
	KnowledgeDir = "knowledge"
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
}

// UnmarshalYAML reads a page written either way: the url on its own, or a mapping naming it
// alongside what it is.
//
//	urls.yaml:
//	    - https://example.com/pricing
//	    - url: https://example.com/plans
//	      title: Plans
//	      description: What each plan includes.
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
		field := map[string]*string{
			"url": &k.URL, "title": &k.Title, "description": &k.Description,
		}[key.Value]
		if field == nil {
			return fmt.Errorf("%q is not something a page says; url, title and description are", key.Value)
		}
		if err := value.Decode(field); err != nil {
			return err
		}
	}
	return nil
}

// Folder is an agent written down as a directory.
//
//	agents/jean/
//	  instructions.md
//	  skills/think.md
//	  knowledge/pricing.md
//	  knowledge/urls.yaml
//
// A skill is a markdown file with YAML-ish frontmatter naming what the fast model sees; the
// body is the prompt only the subagent sees.
type Folder struct {
	// Path is the directory this was read from.
	Path string
	// Name is the directory's own name, which is what the agent is called.
	Name string
	// Instructions is instructions.md, or empty if there is none.
	Instructions string
	// Skills are the files in skills/, in name order.
	Skills []Skill
	// Knowledge are the readable files under knowledge/, in path order.
	Knowledge []Document
	// KnowledgeURLs are the pages knowledge/urls.yaml declares, in the order it lists them.
	KnowledgeURLs []KnowledgeURL
}

// Load reads an agent directory.
//
// Everything in it is optional: a directory with only instructions.md is a valid agent, and
// so is one with only skills.
func Load(path string) (*Folder, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("agents: reading %s: %w", path, err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("agents: %s is not an agent directory", path)
	}

	folder := &Folder{Path: path, Name: filepath.Base(filepath.Clean(path))}

	instructions, err := os.ReadFile(filepath.Join(path, InstructionsFile))
	switch {
	case err == nil:
		folder.Instructions = strings.TrimSpace(string(instructions))
	case !errors.Is(err, fs.ErrNotExist):
		return nil, fmt.Errorf("agents: reading %s: %w", InstructionsFile, err)
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

	if len(f.Skills) == 0 {
		return
	}
	if options.Harness == nil {
		options.Harness = &Harness{UseSkills: true, Skills: f.Skills}
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
			case "subagent":
				skill.Subagent = value
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
