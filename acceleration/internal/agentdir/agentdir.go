// Package agentdir is the agent directory format: what agent.yaml and the files beside it
// say, read from raw files and written back as them.
//
//	agent.yaml
//	instructions.md
//	guardrail.md
//	skills/think.md
//	skills/pdf/SKILL.md
//	skills/pdf/scripts/fill.py
//	knowledge/pricing.md
//	knowledge/urls.yaml
//	simulations/lunch.yaml
//	ui/components.json
//	ui/blueprints.yaml
//
// It is the one reader and the one writer of the format: every client sends the files as
// they are, and Parse decides what they mean.
package agentdir

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"maps"
	"path"
	"slices"
	"strings"
	"time"
	"unicode/utf8"
)

// What an agent directory calls its files.
const (
	AgentFile         = "agent.yaml"
	InstructionsFile  = "instructions.md"
	GuardrailFile     = "guardrail.md"
	SkillFile         = "SKILL.md"
	SkillsDir         = "skills"
	KnowledgeDir      = "knowledge"
	KnowledgeURLsFile = KnowledgeDir + "/urls.yaml"
	SimulationsDir    = "simulations"
	UIComponentsFile  = "ui/components.json"
	UIBlueprintsFile  = "ui/blueprints.yaml"
)

// What one directory may hold.
const (
	// MaxBytes is all its files together: what one knowledge ingest takes (api.MaxKnowledgeBody),
	// since the knowledge is most of a directory.
	MaxBytes = 8 << 20
	// MaxFileBytes is one file: what all of one skill's files may hold (api.maxSkillBytes).
	MaxFileBytes = 1 << 20
	// MaxFiles is how many files a client may send: sixteen skills at the 64 files a skill may
	// bring. The API holds a request to it.
	MaxFiles = 1024
	// maxUIFileBytes is a UI catalog or its blueprints, which the model reads on every turn.
	maxUIFileBytes = 512 << 10
)

// File is one file of an agent directory.
type File struct {
	// Path is relative to the directory, with forward slashes.
	Path    string
	Content string
}

// Problem is one thing wrong with a directory, at its place in a file.
type Problem struct {
	File string
	// Line and Column count from 1. Both are zero when the problem is the file as a whole.
	Line    int
	Column  int
	Message string
}

func (p Problem) String() string {
	if p.Line == 0 {
		return p.File + ": " + p.Message
	}
	return fmt.Sprintf("%s:%d:%d: %s", p.File, p.Line, p.Column, p.Message)
}

// Problems is everything wrong with a directory, in the order it was found.
type Problems []Problem

func (p Problems) Error() string {
	texts := make([]string, len(p))
	for i, problem := range p {
		texts[i] = problem.String()
	}
	return strings.Join(texts, "; ")
}

// Directory is what an agent directory says.
type Directory struct {
	Settings     Settings
	Instructions string
	// Guardrail is guardrail.md, whole: the router's guardrail package reads it.
	Guardrail string
	// UI is agent.yaml's ui key with the files it names, or nil without one.
	UI     *UI
	Skills []Skill
	// Knowledge are the readable files under knowledge/, in path order.
	Knowledge     []Document
	KnowledgeURLs []KnowledgeURL
	// Simulations is nil when the directory has no simulations/*.yaml, which leaves the stored
	// ones alone, and empty when its files declare none, which deletes them.
	Simulations []Simulation
	// Files are the files the directory was read from, in path order: what its hash is of.
	Files []File
	// Ignored are the files sent that are no part of the format, by path.
	Ignored []string

	declaration *reader
}

// Skill is a skill file, or a skill folder's SKILL.md with the files beside it.
type Skill struct {
	Name         string
	Description  string
	Instructions string
	Deadline     time.Duration
	CaptureVideo bool
	// Files are the rest of a skill folder, by path relative to it.
	Files map[string]string
}

// Document is one file of knowledge/, by its path relative to it.
type Document struct {
	Source string
	Text   string
}

// KnowledgeURL is one page of knowledge/urls.yaml.
type KnowledgeURL struct {
	URL          string
	Title        string
	Description  string
	RefreshHours int
}

// Simulation is one conversation a simulations/*.yaml file declares.
type Simulation struct {
	Name         string
	Scenario     string
	Assertion    string
	Mode         string
	Variations   int
	MaxTurns     int
	CallerTarget string
	JudgeTarget  string
	CallerSTT    string
	CallerTTS    string
	CallerVoice  string
	Tags         map[string]string
}

// knowledgeExtensions are what a knowledge directory is read from: a model looks things up in
// prose, not in a binary.
var knowledgeExtensions = []string{".md", ".mdx", ".txt", ".rst", ".yaml", ".yml"}

// UIVersion is the version of what the ui: compiler writes. It is part of the hash of a
// directory that has a ui key, so a directory is synced again, and its UI instructions
// compiled again, by a router whose compiler writes something else: add one to it with any
// change to what compileUIPrompt or the A2UI catalog produces.
const UIVersion = "1"

// Hash is the fingerprint of files and of whatever is synced beside them: the SHA-256, in
// hex, of every part in turn as its length in bytes in decimal, a colon and the part. The
// parts are each file's path and then its content, in path order by byte, and then each of
// beside. The length is what keeps one part from running into the next.
func Hash(files []File, beside ...string) string {
	sorted := slices.SortedFunc(slices.Values(files), func(a, b File) int { return strings.Compare(a.Path, b.Path) })
	hasher := sha256.New()
	part := func(text string) { fmt.Fprintf(hasher, "%d:%s", len(text), text) }
	for _, file := range sorted {
		part(file.Path)
		part(file.Content)
	}
	for _, more := range beside {
		part(more)
	}
	return hex.EncodeToString(hasher.Sum(nil))
}

// Parse reads a directory from the files a client sent. Everything wrong with it is reported
// at once, as Problems, and the directory is returned beside them whenever there is an
// agent.yaml to read, so a caller can go on to say what else is wrong with it.
//
// A client sends what it finds and Parse decides what is the directory: a hidden file is
// skipped, a file that is no part of the format is left out and listed in Ignored, and the
// limits and the hash are of the files that are left.
func Parse(files []File) (*Directory, error) {
	var problems Problems
	fail := func(file, format string, args ...any) {
		problems = append(problems, Problem{File: file, Message: fmt.Sprintf(format, args...)})
	}

	held := map[string]string{}
	for _, file := range files {
		switch _, twice := held[file.Path]; {
		case !ValidPath(file.Path):
			fail(file.Path, "is not a path inside the directory: use forward slashes, and no .. or leading /")
		case hidden(file.Path):
		case twice:
			fail(file.Path, "is sent twice")
		default:
			held[file.Path] = file.Content
		}
	}
	declaration, declared := held[AgentFile]
	if !declared {
		return nil, append(problems, Problem{File: AgentFile, Message: "is missing, and it is what makes a directory an agent"})
	}

	used := map[string]bool{}
	text := func(file string) (string, bool) {
		content, ok := held[file]
		if !ok {
			return "", false
		}
		if !used[file] && len(content) > MaxFileBytes {
			fail(file, "holds %d bytes, and a file holds at most %d", len(content), MaxFileBytes)
		}
		used[file] = true
		if !utf8.ValidString(content) {
			fail(file, "is not text")
			return "", false
		}
		return content, true
	}
	paths := slices.Sorted(maps.Keys(held))

	directory := &Directory{declaration: &reader{file: AgentFile, problems: &problems}}
	text(AgentFile)
	directory.Settings = directory.declaration.settings(declaration)
	instructions, _ := text(InstructionsFile)
	guardrail, _ := text(GuardrailFile)
	directory.Instructions, directory.Guardrail = strings.TrimSpace(instructions), strings.TrimSpace(guardrail)
	directory.Skills = readSkills(paths, text, &problems)
	directory.Knowledge = readKnowledge(paths, text)
	if pages, ok := text(KnowledgeURLsFile); ok {
		directory.KnowledgeURLs = (&reader{file: KnowledgeURLsFile, problems: &problems}).knowledgeURLs(pages)
	}
	directory.Simulations = readSimulations(paths, text, &problems)
	if directory.Settings.UI != nil {
		directory.UI = readUI(*directory.Settings.UI, directory.declaration, text, &problems)
	}

	total := 0
	for _, file := range paths {
		if !used[file] {
			directory.Ignored = append(directory.Ignored, file)
			continue
		}
		directory.Files = append(directory.Files, File{Path: file, Content: held[file]})
		total += len(held[file])
	}
	if total > MaxBytes {
		fail(AgentFile, "the directory holds %d bytes, and one holds at most %d", total, MaxBytes)
	}
	if len(problems) > 0 {
		return directory, problems
	}
	return directory, nil
}

// Locate is where agent.yaml says the setting a dotted path names, such as greeting.mode or
// chat_channels[1].answer, or the nearest setting above it the file does say.
func (d *Directory) Locate(setting string) (line, column int) {
	return d.declaration.locate(setting)
}

// ValidPath reports whether a slash path names a file inside the folder it is relative to.
func ValidPath(file string) bool {
	return file != "" && !path.IsAbs(file) && path.Clean(file) == file && file != ".." &&
		!strings.HasPrefix(file, "../") && !strings.Contains(file, `\`)
}

// hidden reports whether a file is left behind wherever it is: dot files such as .DS_Store
// and .agent_sync, the __pycache__ that running a skill's scripts locally leaves, and
// node_modules.
func hidden(file string) bool {
	for segment := range strings.SplitSeq(file, "/") {
		if strings.HasPrefix(segment, ".") || segment == "__pycache__" || segment == "node_modules" {
			return true
		}
	}
	return false
}
