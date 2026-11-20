package agentdir

import (
	"errors"
	"fmt"
	"path"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"gopkg.in/yaml.v3"
)

// readSkills reads skills/: a skill is either a markdown file, or a folder holding a SKILL.md
// and whatever files it comes with, nested as deep as they are.
func readSkills(paths []string, text func(string) (string, bool), problems *Problems) []Skill {
	var skills []Skill
	declared, named := map[string]string{}, map[string]string{}
	folders := map[string]bool{}
	for _, file := range paths {
		rest, inside := strings.CutPrefix(file, SkillsDir+"/")
		if !inside {
			continue
		}
		name, _, folder := strings.Cut(rest, "/")
		switch {
		case folder && folders[name]:
			continue
		case folder:
			folders[name] = true
			file = SkillsDir + "/" + name + "/" + SkillFile
		case path.Ext(name) == ".md":
			name = strings.TrimSuffix(name, ".md")
		default:
			continue
		}

		content, ok := text(file)
		if !ok {
			if folder {
				*problems = append(*problems, Problem{File: file, Message: "is missing, so " + SkillsDir + "/" + name + " is not a skill"})
			}
			continue
		}
		if first, taken := declared[name]; taken {
			*problems = append(*problems, Problem{File: file, Message: fmt.Sprintf("skill %q is also %s", name, first)})
			continue
		}
		declared[name] = file

		skill := parseSkill(name, file, content, problems)
		// A sync finds a skill by its name, which the frontmatter may give in place of the file's.
		if first, taken := named[skill.Name]; taken {
			*problems = append(*problems, Problem{File: file, Line: 1, Column: 1,
				Message: fmt.Sprintf("skill %q is also declared in %s", skill.Name, first)})
		}
		named[skill.Name] = file
		if folder {
			prefix := SkillsDir + "/" + name + "/"
			for _, beside := range paths {
				relative, within := strings.CutPrefix(beside, prefix)
				if !within || relative == SkillFile {
					continue
				}
				if content, ok := text(beside); ok {
					if skill.Files == nil {
						skill.Files = map[string]string{}
					}
					skill.Files[relative] = content
				}
			}
		}
		skills = append(skills, skill)
	}
	return skills
}

// parseSkill reads a skill file: frontmatter between --- lines, then the instructions.
//
// The frontmatter is read a line at a time, as a key and its value, and not as YAML. The keys
// read are name, description, capture_video and deadline; any other is left alone, since a
// skill written for another harness brings keys of its own.
func parseSkill(name, file, content string, problems *Problems) Skill {
	skill := Skill{Name: name}
	fail := func(line int, format string, args ...any) {
		*problems = append(*problems, Problem{File: file, Line: line, Column: 1, Message: fmt.Sprintf(format, args...)})
	}

	frontmatter, body, first := cutFrontmatter(content)
	for index, line := range strings.Split(frontmatter, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			fail(first+index, "%q is not a key and a value", line)
			continue
		}
		value = unquoted(strings.TrimSpace(value))
		switch strings.TrimSpace(key) {
		case "name":
			skill.Name = value
		case "description":
			skill.Description = value
		case "capture_video":
			if value != "true" && value != "false" {
				fail(first+index, "capture_video must be true or false")
			}
			skill.CaptureVideo = value == "true"
		case "deadline":
			deadline, err := parseDeadline(value)
			if err != nil {
				fail(first+index, "%q is not a deadline: write a duration such as 30s or 2m", value)
			}
			skill.Deadline = deadline
		}
	}

	skill.Instructions = strings.TrimSpace(body)
	if skill.Description == "" {
		fail(1, "a skill needs a description, since it is all the fast model sees")
	}
	if skill.Instructions == "" {
		fail(1, "a skill needs instructions, since they are what the subagent answers under")
	}
	return skill
}

// parseDeadline takes a Go duration, and a bare number as seconds.
func parseDeadline(value string) (time.Duration, error) {
	deadline, err := time.ParseDuration(value)
	if seconds, bare := strconv.ParseFloat(value, 64); bare == nil {
		deadline, err = time.Duration(seconds*float64(time.Second)), nil
	}
	if err == nil && deadline < 0 {
		err = errors.New("a deadline is not negative")
	}
	return deadline, err
}

// unquoted is a frontmatter value without the pair of quotes around it, if it has one.
func unquoted(value string) string {
	if len(value) >= 2 && (value[0] == '"' || value[0] == '\'') && value[len(value)-1] == value[0] {
		return value[1 : len(value)-1]
	}
	return value
}

// cutFrontmatter separates a leading --- block from the body, and says which line of the file
// the block's first line is. The block ends at the next line that is --- alone, and the body
// is everything after that line, as written.
func cutFrontmatter(content string) (frontmatter, body string, line int) {
	trimmed := strings.TrimLeft(content, "\ufeff \t\r\n")
	opening, rest, found := strings.Cut(trimmed, "\n")
	if !found || strings.TrimRight(opening, " \t\r") != "---" {
		return "", content, 0
	}
	line = 2 + strings.Count(content[:len(content)-len(trimmed)], "\n")
	for at := 0; at < len(rest); {
		end := strings.IndexByte(rest[at:], '\n')
		next := len(rest)
		if end >= 0 {
			next = at + end + 1
		}
		if strings.TrimRight(rest[at:next], " \t\r\n") == "---" {
			return rest[:at], rest[next:], line
		}
		at = next
	}
	return "", content, 0
}

// readKnowledge reads the readable files under knowledge/ that hold anything. The urls.yaml
// at its root declares pages and is not itself looked things up in; deeper, a urls.yaml is a
// document like any other.
func readKnowledge(paths []string, text func(string) (string, bool)) []Document {
	var documents []Document
	for _, file := range paths {
		source, inside := strings.CutPrefix(file, KnowledgeDir+"/")
		if !inside || !knowledgePath(source) {
			continue
		}
		if content, ok := text(file); ok && strings.TrimSpace(content) != "" {
			documents = append(documents, Document{Source: source, Text: content})
		}
	}
	return documents
}

// KnowledgeFile reports whether knowledge/ holds a document under source: a path inside it
// with an extension that is read, holding text that says something.
func KnowledgeFile(source, content string) bool {
	return knowledgePath(source) && utf8.ValidString(content) && strings.TrimSpace(content) != ""
}

func knowledgePath(source string) bool {
	return ValidPath(source) && !hidden(source) && KnowledgeDir+"/"+source != KnowledgeURLsFile &&
		slices.Contains(knowledgeExtensions, strings.ToLower(path.Ext(source)))
}

// knowledgeURLs reads knowledge/urls.yaml: a list of pages, each the url on its own or a
// mapping naming it alongside what it is.
func (r *reader) knowledgeURLs(content string) []KnowledgeURL {
	root := r.document(content)
	if root == nil {
		return nil
	}
	var pages []KnowledgeURL
	r.each(root, func(item *yaml.Node) {
		if null(item) {
			return
		}
		var page KnowledgeURL
		where := item
		if item = resolved(item); item.Kind == yaml.ScalarNode {
			page.URL = item.Value
		} else if !r.mapping(item, "a page",
			field{"url", func(value *yaml.Node) { where = value; r.text(&page.URL)(value) }},
			field{"title", r.text(&page.Title)},
			field{"description", r.text(&page.Description)},
			field{"refresh_hours", func(value *yaml.Node) {
				r.integer(&page.RefreshHours)(value)
				if page.RefreshHours < 1 {
					r.fail(value, "is how many hours between reads, so it is at least 1; leave it out for never")
				}
			}}) {
			return
		}
		if !strings.HasPrefix(page.URL, "http://") && !strings.HasPrefix(page.URL, "https://") {
			r.fail(where, "%q is not an http or https url", page.URL)
		}
		pages = append(pages, page)
	})
	return pages
}

// readSimulations reads every .yaml and .yml file directly in simulations/, each a list of
// simulations, in file order. A name two simulations share is refused, since a sync finds a
// simulation by its name.
func readSimulations(paths []string, text func(string) (string, bool), problems *Problems) []Simulation {
	var simulations []Simulation
	named := map[string]string{}
	for _, file := range paths {
		name, inside := strings.CutPrefix(file, SimulationsDir+"/")
		extension := strings.ToLower(path.Ext(file))
		if !inside || strings.Contains(name, "/") || (extension != ".yaml" && extension != ".yml") {
			continue
		}
		content, ok := text(file)
		if !ok {
			continue
		}
		if simulations == nil {
			simulations = []Simulation{}
		}
		r := &reader{file: file, problems: problems}
		root := r.document(content)
		if root == nil {
			continue
		}
		r.each(root, func(item *yaml.Node) {
			if null(item) {
				return
			}
			var simulation Simulation
			at := map[string]*yaml.Node{}
			keep := func(key string, read func(*yaml.Node)) field {
				return field{key, func(value *yaml.Node) { at[key] = value; read(value) }}
			}
			if !r.mapping(item, "a simulation",
				keep("name", r.text(&simulation.Name)),
				keep("scenario", r.text(&simulation.Scenario)),
				keep("assertion", r.text(&simulation.Assertion)),
				keep("mode", r.text(&simulation.Mode)),
				field{"variations", r.integer(&simulation.Variations)},
				field{"max_turns", r.integer(&simulation.MaxTurns)},
				field{"caller_target", r.text(&simulation.CallerTarget)},
				field{"judge_target", r.text(&simulation.JudgeTarget)},
				field{"caller_stt", r.text(&simulation.CallerSTT)},
				field{"caller_tts", r.text(&simulation.CallerTTS)},
				field{"caller_voice", r.text(&simulation.CallerVoice)},
				field{"tags", r.labels(&simulation.Tags)}) {
				return
			}
			where := func(key string) *yaml.Node {
				if node := at[key]; node != nil {
					return node
				}
				return item
			}
			switch {
			case simulation.Name == "":
				r.fail(where("name"), "a simulation needs a name")
			case simulation.Scenario == "":
				r.fail(where("scenario"), "simulation %q needs a scenario", simulation.Name)
			case simulation.Assertion == "":
				r.fail(where("assertion"), "simulation %q needs an assertion", simulation.Name)
			case simulation.Mode != "" && simulation.Mode != "text" && simulation.Mode != "audio":
				r.fail(where("mode"), "simulation %q is text or audio, not %q", simulation.Name, simulation.Mode)
			}
			if first, taken := named[simulation.Name]; taken && simulation.Name != "" {
				r.fail(where("name"), "simulation %q is also declared in %s", simulation.Name, first)
			}
			named[simulation.Name] = file
			simulations = append(simulations, simulation)
		})
	}
	return simulations
}
