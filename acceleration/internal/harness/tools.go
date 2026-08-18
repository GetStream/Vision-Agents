package harness

import (
	"embed"
	"errors"
	"fmt"
	"os"

	"gopkg.in/yaml.v3"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// defaultToolsFS carries the built-in tool set so an agent with telephony works without an
// external file.
//
//go:embed tools.yaml
var defaultToolsFS embed.FS

// Tool is one thing the fast model may do rather than say.
//
// It is deliberately not a skill: a skill asks a better model a question and folds the
// answer into the conversation, while a tool reaches outside the conversation and changes
// something that cannot be changed back. That difference is why a tool is run by the agent,
// which knows what call it is on, rather than here.
type Tool struct {
	// Name is how the model asks for it.
	Name string `yaml:"name"`
	// Description is what the model is told the tool does, and is the whole of how it
	// decides when to reach for one.
	Description string `yaml:"description"`
	// Parameters is a JSON Schema object describing the arguments. It is untyped because
	// a schema is untyped: the shape is whatever the tool accepts.
	Parameters map[string]any `yaml:"parameters"`
}

// Tools is the set a harness was configured with.
type Tools struct {
	Tools []Tool `yaml:"tools"`
}

// Lookup returns a tool by name.
func (t Tools) Lookup(name string) (Tool, bool) {
	for _, tool := range t.Tools {
		if tool.Name == name {
			return tool, true
		}
	}
	return Tool{}, false
}

// Requests renders the set for a completion request. It returns nil when there are none,
// so a request carrying no tools is not merely one carrying an empty list: a model offered
// an empty toolbox still answers as though it had one.
func (t Tools) Requests() []llm.Tool {
	if len(t.Tools) == 0 {
		return nil
	}

	rendered := make([]llm.Tool, 0, len(t.Tools))
	for _, tool := range t.Tools {
		rendered = append(rendered, llm.Tool{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  tool.Parameters,
		})
	}
	return rendered
}

// Validate reports the first tool the harness could not use.
func (t Tools) Validate() error {
	seen := map[string]struct{}{}
	for _, tool := range t.Tools {
		if tool.Name == "" {
			return errors.New("harness: every tool needs a name")
		}
		if tool.Description == "" {
			return fmt.Errorf("harness: tool %s has no description, so the model would "+
				"never know when to use it", tool.Name)
		}
		if _, duplicate := seen[tool.Name]; duplicate {
			return fmt.Errorf("harness: tool %s is declared twice", tool.Name)
		}
		seen[tool.Name] = struct{}{}
	}
	return nil
}

// DefaultTools returns the built-in tool set.
func DefaultTools() (Tools, error) {
	raw, err := defaultToolsFS.ReadFile("tools.yaml")
	if err != nil {
		return Tools{}, fmt.Errorf("harness: read default tools: %w", err)
	}
	return parseTools(raw)
}

// LoadTools reads a tool set, or the built-in default when path is empty.
func LoadTools(path string) (Tools, error) {
	if path == "" {
		return DefaultTools()
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		return Tools{}, fmt.Errorf("harness: read tools %s: %w", path, err)
	}
	return parseTools(raw)
}

func parseTools(raw []byte) (Tools, error) {
	var tools Tools
	if err := yaml.Unmarshal(raw, &tools); err != nil {
		return Tools{}, fmt.Errorf("harness: parse tools: %w", err)
	}
	if err := tools.Validate(); err != nil {
		return Tools{}, err
	}
	return tools, nil
}
