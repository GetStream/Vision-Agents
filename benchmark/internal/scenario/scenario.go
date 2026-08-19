package scenario

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// Category is one of the four call types plus checklist variants.
type Category string

const (
	Golden      Category = "golden"
	Interrupt   Category = "interrupt"
	EntityDense Category = "entity_dense"
	Adversarial Category = "adversarial"
	Checklist   Category = "checklist"
)

// Scenario is one scripted phone call.
type Scenario struct {
	ID            string            `yaml:"id"`
	Pack          string            `yaml:"pack"`
	Category      Category          `yaml:"category"`
	Name          string            `yaml:"name"`
	Persona       string            `yaml:"persona"`
	Instructions  string            `yaml:"instructions"`
	MaxDurationS  int               `yaml:"max_duration_s"`
	Noise         string            `yaml:"noise"`
	SNRDB         float64           `yaml:"snr_db"`
	ToolDelayMS   map[string]int    `yaml:"tool_delay_ms"`
	Turns         []Turn            `yaml:"turns"`
	Seed          map[string]any    `yaml:"seed"`
	EndState      []Assertion       `yaml:"end_state"`
	ExpectedTools []ExpectedTool    `yaml:"expected_tools"`
	ToolOrder     []OrderConstraint `yaml:"tool_order"`
	Entities      []Entity          `yaml:"entities"`
	Policy        []string          `yaml:"policy"`
	Judge         JudgeSpec         `yaml:"judge"`
	AgentReplies  []string          `yaml:"agent_replies"`
}

// Turn is one caller utterance or overlap sound.
type Turn struct {
	ID           string  `yaml:"id"`
	Text         string  `yaml:"text"`
	OverlapSound string  `yaml:"overlap_sound"`
	Trigger      Trigger `yaml:"trigger"`
}

// Trigger decides when the turn is played.
type Trigger struct {
	Kind    string `yaml:"kind"`
	DelayMS int    `yaml:"delay_ms"`
	AfterMS int    `yaml:"after_ms"`
}

const (
	TriggerAfterAgent  = "after_agent_turn"
	TriggerBargeIn     = "barge_in_after_ms"
	TriggerDuringAgent = "during_agent_after_ms"
	TriggerImmediate   = "immediate"
)

// Assertion checks a path in the world end-state.
type Assertion struct {
	Path   string `yaml:"path"`
	Eq     any    `yaml:"eq"`
	Exists bool   `yaml:"exists"`
	Neq    any    `yaml:"neq"`
}

// ExpectedTool is a tool the agent should call.
type ExpectedTool struct {
	Name string         `yaml:"name"`
	Args map[string]any `yaml:"args"`
}

// OrderConstraint requires one tool to run before another.
type OrderConstraint struct {
	Before string `yaml:"before"`
	After  string `yaml:"after"`
}

// Entity must appear in speech, tools, or both.
type Entity struct {
	Name     string `yaml:"name"`
	Value    string `yaml:"value"`
	InSpeech bool   `yaml:"in_speech"`
	InTools  bool   `yaml:"in_tools"`
}

// JudgeSpec is extra prompt material for the LLM judge.
type JudgeSpec struct {
	Task            string   `yaml:"task"`
	MustRefuse      []string `yaml:"must_refuse"`
	MustNotDisclose []string `yaml:"must_not_disclose"`
	Coherence       bool     `yaml:"coherence"`
}

// LoadFile reads one scenario YAML file.
func LoadFile(path string) (Scenario, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Scenario{}, err
	}
	var s Scenario
	if err := yaml.Unmarshal(raw, &s); err != nil {
		return Scenario{}, fmt.Errorf("scenario: parse %s: %w", path, err)
	}
	if s.MaxDurationS <= 0 {
		s.MaxDurationS = 180
	}
	if err := s.Validate(); err != nil {
		return Scenario{}, fmt.Errorf("scenario: %s: %w", path, err)
	}
	return s, nil
}

// LoadPack loads every YAML file in a directory.
func LoadPack(dir string) ([]Scenario, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []Scenario
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".yaml") {
			continue
		}
		s, err := LoadFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("scenario: no yaml files in %s", dir)
	}
	return out, nil
}

// Validate checks required fields.
func (s Scenario) Validate() error {
	if s.ID == "" {
		return fmt.Errorf("id is required")
	}
	if s.Pack == "" {
		return fmt.Errorf("pack is required")
	}
	if s.Category == "" {
		return fmt.Errorf("category is required")
	}
	if len(s.Turns) == 0 {
		return fmt.Errorf("at least one turn is required")
	}
	for i, turn := range s.Turns {
		if turn.Text == "" && turn.OverlapSound == "" {
			return fmt.Errorf("turn %d needs text or overlap_sound", i)
		}
		kind := turn.Trigger.Kind
		if kind == "" {
			kind = TriggerAfterAgent
		}
		switch kind {
		case TriggerAfterAgent, TriggerBargeIn, TriggerDuringAgent, TriggerImmediate:
		default:
			return fmt.Errorf("turn %d: unknown trigger %q", i, kind)
		}
	}
	return nil
}

// SpeechTexts returns caller lines that need TTS.
func (s Scenario) SpeechTexts() []string {
	var out []string
	for _, turn := range s.Turns {
		if turn.Text != "" {
			out = append(out, turn.Text)
		}
	}
	return out
}

// Packs is the known vertical set.
func Packs() []string {
	return []string{"restaurant", "healthcare", "telecom"}
}
