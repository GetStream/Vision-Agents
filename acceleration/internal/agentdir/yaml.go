package agentdir

import (
	"fmt"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// reader reads one YAML file, reporting each thing wrong with it at its line and column
// rather than stopping at the first.
type reader struct {
	file     string
	problems *Problems
	root     *yaml.Node
}

func (r *reader) fail(node *yaml.Node, format string, args ...any) {
	*r.problems = append(*r.problems, Problem{
		File: r.file, Line: node.Line, Column: node.Column, Message: fmt.Sprintf(format, args...),
	})
}

// document parses the file and returns what it holds, or nil when it holds nothing or does
// not parse.
func (r *reader) document(content string) *yaml.Node {
	var document yaml.Node
	if err := yaml.Unmarshal([]byte(content), &document); err != nil {
		line, message := 1, strings.TrimPrefix(err.Error(), "yaml: ")
		if rest, found := strings.CutPrefix(message, "line "); found {
			if number, text, cut := strings.Cut(rest, ": "); cut {
				if parsed, err := strconv.Atoi(number); err == nil {
					line, message = parsed, text
				}
			}
		}
		*r.problems = append(*r.problems, Problem{File: r.file, Line: line, Column: 1, Message: message})
		return nil
	}
	if len(document.Content) == 0 || null(document.Content[0]) {
		return nil
	}
	r.root = document.Content[0]
	return r.root
}

// field is one key a mapping may hold, and what reads its value.
type field struct {
	name string
	read func(value *yaml.Node)
}

// mapping reads a mapping's keys, each by its field. A key no field names is refused, and so
// is one written twice. A key whose value is null reads as one left out.
func (r *reader) mapping(node *yaml.Node, what string, fields ...field) bool {
	node = resolved(node)
	if node.Kind != yaml.MappingNode {
		r.fail(node, "must be a mapping")
		return false
	}
	seen := map[string]bool{}
	for i := 0; i+1 < len(node.Content); i += 2 {
		key, value := node.Content[i], node.Content[i+1]
		if seen[key.Value] {
			r.fail(key, "%s is written twice", key.Value)
			continue
		}
		seen[key.Value] = true
		known := false
		for _, field := range fields {
			if field.name == key.Value {
				known = true
				if !null(value) {
					before := len(*r.problems)
					field.read(value)
					for i := before; i < len(*r.problems); i++ {
						(*r.problems)[i].Message = key.Value + ": " + (*r.problems)[i].Message
					}
				}
			}
		}
		if !known {
			// A retired key is read only to be refused, so it is not one the mapping says.
			var names []string
			for _, field := range fields {
				if _, gone := retired[field.name]; !gone {
					names = append(names, field.name)
				}
			}
			r.fail(key, "%q is not something %s says; %s are", key.Value, what, strings.Join(names, ", "))
		}
	}
	return true
}

// each reads the entries of a list.
func (r *reader) each(node *yaml.Node, read func(item *yaml.Node)) bool {
	node = resolved(node)
	if node.Kind != yaml.SequenceNode {
		r.fail(node, "must be a list")
		return false
	}
	for _, item := range node.Content {
		read(item)
	}
	return true
}

func (r *reader) text(into *string) func(*yaml.Node) {
	return func(node *yaml.Node) {
		node = resolved(node)
		if node.Kind != yaml.ScalarNode {
			r.fail(node, "must be text, not a %s", kind(node))
			return
		}
		*into = node.Value
	}
}

func (r *reader) optionalText(into **string) func(*yaml.Node) {
	return func(node *yaml.Node) {
		var value string
		before := len(*r.problems)
		r.text(&value)(node)
		if len(*r.problems) == before {
			*into = &value
		}
	}
}

func (r *reader) integer(into *int) func(*yaml.Node) {
	return func(node *yaml.Node) {
		node = resolved(node)
		value, err := strconv.Atoi(node.Value)
		if node.Kind != yaml.ScalarNode || node.ShortTag() != "!!int" || err != nil {
			r.fail(node, "must be a whole number")
			return
		}
		*into = value
	}
}

func (r *reader) optionalInteger(into **int) func(*yaml.Node) {
	return func(node *yaml.Node) {
		var value int
		before := len(*r.problems)
		r.integer(&value)(node)
		if len(*r.problems) == before {
			*into = &value
		}
	}
}

func (r *reader) boolean(into *bool) func(*yaml.Node) {
	return func(node *yaml.Node) {
		node = resolved(node)
		if node.Kind != yaml.ScalarNode || node.ShortTag() != "!!bool" {
			r.fail(node, "must be true or false")
			return
		}
		*into = node.Value == "true"
	}
}

func (r *reader) optionalBoolean(into **bool) func(*yaml.Node) {
	return func(node *yaml.Node) {
		var value bool
		before := len(*r.problems)
		r.boolean(&value)(node)
		if len(*r.problems) == before {
			*into = &value
		}
	}
}

// list reads a list of text, which is empty rather than nil when the file lists none.
func (r *reader) list(into *[]string) func(*yaml.Node) {
	return func(node *yaml.Node) {
		values := []string{}
		if r.each(node, func(item *yaml.Node) {
			var value string
			r.text(&value)(item)
			values = append(values, value)
		}) {
			*into = values
		}
	}
}

func (r *reader) optionalList(into **[]string) func(*yaml.Node) {
	return func(node *yaml.Node) {
		var values []string
		r.list(&values)(node)
		if values != nil {
			*into = &values
		}
	}
}

// labels reads a mapping of text to text, such as tags.
func (r *reader) labels(into *map[string]string) func(*yaml.Node) {
	return func(node *yaml.Node) {
		node = resolved(node)
		if node.Kind != yaml.MappingNode {
			r.fail(node, "must be a mapping of names to values")
			return
		}
		values := map[string]string{}
		for i := 0; i+1 < len(node.Content); i += 2 {
			var value string
			r.text(&value)(node.Content[i+1])
			values[node.Content[i].Value] = value
		}
		*into = values
	}
}

// locate is where the file says the value a dotted path names, or the nearest value above it.
func (r *reader) locate(setting string) (line, column int) {
	node := r.root
	if node == nil {
		return 1, 1
	}
	for _, step := range strings.FieldsFunc(setting, func(char rune) bool { return char == '.' || char == '[' || char == ']' }) {
		next := child(resolved(node), step)
		if next == nil {
			break
		}
		node = next
	}
	return node.Line, node.Column
}

// child is the value of a mapping's key called step, or the entry of a list at index step.
// A mapping or a list under a key is found at the key, which is where a reader looks for it.
func child(node *yaml.Node, step string) *yaml.Node {
	if index, err := strconv.Atoi(step); err == nil {
		if node.Kind == yaml.SequenceNode && index < len(node.Content) {
			return node.Content[index]
		}
		return nil
	}
	if node.Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i+1 < len(node.Content); i += 2 {
		if node.Content[i].Value == step {
			if value := node.Content[i+1]; value.Kind == yaml.ScalarNode {
				return value
			}
			return positioned(node.Content[i+1], node.Content[i])
		}
	}
	return nil
}

// positioned is a value with the position of its key, which is where a mapping or a list
// under a key is named.
func positioned(value, key *yaml.Node) *yaml.Node {
	copied := *value
	copied.Line, copied.Column = key.Line, key.Column
	return &copied
}

func resolved(node *yaml.Node) *yaml.Node {
	if node.Kind == yaml.AliasNode && node.Alias != nil {
		return node.Alias
	}
	return node
}

func null(node *yaml.Node) bool {
	return node.Kind == yaml.ScalarNode && node.ShortTag() == "!!null"
}

func kind(node *yaml.Node) string {
	switch node.Kind {
	case yaml.MappingNode:
		return "mapping"
	case yaml.SequenceNode:
		return "list"
	}
	return "value"
}
