// Command llmgroups feeds the llm_groups skill: it dumps every LLM shortcut with the models
// it routes over, and measures those models on a fixed task suite through the same registry
// the router builds them with.
//
//	go run ./cmd/llmgroups snapshot > snapshot.json
//	go run ./cmd/llmgroups bench -group llm-fast -inventory ../.claude/skills/llm_groups/inventory.json
//	go run ./cmd/llmgroups bench -models gemini/gemini-4-hydrogen -efforts low,medium,high,xhigh -inventory ...
//
// Bench reads each vendor's key from the environment, as the router does.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: llmgroups snapshot | bench [flags]")
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "snapshot":
		err = snapshot(os.Args[2:])
	case "bench":
		err = bench(os.Args[2:])
	default:
		err = fmt.Errorf("llmgroups: unknown command %q", os.Args[1])
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func llmConfig() (routing.ModalityConfig, error) {
	config, err := routing.DefaultConfig()
	if err != nil {
		return routing.ModalityConfig{}, err
	}
	return config[routing.LLM], nil
}

// model is what the report needs to know about one routed model from router.yaml.
type model struct {
	Name            string               `json:"name"`
	Description     string               `json:"description"`
	Tier            string               `json:"tier"`
	Realtime        bool                 `json:"realtime"`
	Languages       int                  `json:"languages"`
	InputModalities []string             `json:"input_modalities"`
	IgnoresTools    bool                 `json:"ignores_tools"`
	ContextWindow   int64                `json:"context_window"`
	ReasoningEffort string               `json:"reasoning_effort"`
	Thinking        bool                 `json:"thinking"`
	Local           bool                 `json:"local"`
	DataPolicy      options.DataHandling `json:"data_policy"`
	Benchmark       benchmark            `json:"benchmark"`
}

// benchmark is the Artificial Analysis numbers router.yaml carries for a text model.
type benchmark struct {
	IntelligenceIndex     int     `json:"intelligence_index,omitempty"`
	OutputTokensPerSecond float64 `json:"output_tokens_per_second,omitempty"`
}

type group struct {
	Name        string   `json:"name"`
	Title       string   `json:"title"`
	Description string   `json:"description"`
	Prefer      string   `json:"prefer"`
	Requires    requires `json:"requires"`
	Members     []string `json:"members"`
}

// requires is what a shortcut asks of its members, as router.yaml writes it.
type requires struct {
	Tier            string   `json:"tier,omitempty"`
	Realtime        bool     `json:"realtime,omitempty"`
	Tools           bool     `json:"tools,omitempty"`
	Languages       []string `json:"languages,omitempty"`
	Multilingual    bool     `json:"multilingual,omitempty"`
	InputModalities []string `json:"input_modalities,omitempty"`
	Only            []string `json:"only,omitempty"`
}

func snapshot(args []string) error {
	flags := flag.NewFlagSet("snapshot", flag.ExitOnError)
	if err := flags.Parse(args); err != nil {
		return err
	}
	config, err := llmConfig()
	if err != nil {
		return err
	}

	var models []model
	seen := map[string]bool{}
	for _, p := range config.Providers {
		// A model declared once per region is one row: the report compares models, not hosts.
		if seen[p.Name()] {
			continue
		}
		seen[p.Name()] = true
		tier := string(p.Tier)
		if tier == "" {
			tier = string(routing.LowLatency)
		}
		models = append(models, model{
			Name: p.Name(), Description: p.Description, Tier: tier, Realtime: p.Realtime,
			Languages: len(p.Languages), InputModalities: p.InputModalities, IgnoresTools: p.IgnoresTools,
			ContextWindow: p.ContextWindow, ReasoningEffort: p.ReasoningEffort, Thinking: p.Thinking,
			Local: p.Local, DataPolicy: p.DataPolicy,
			Benchmark: benchmark{
				IntelligenceIndex:     p.Benchmark.IntelligenceIndex,
				OutputTokensPerSecond: p.Benchmark.OutputTokensPerSecond,
			},
		})
	}

	var groups []group
	for _, name := range aliasNames(config) {
		alias := config.Aliases[name]
		var members []string
		for _, p := range config.Members(name) {
			if !slices.Contains(members, p.Name()) {
				members = append(members, p.Name())
			}
		}
		groups = append(groups, group{
			Name: name, Title: alias.Title, Description: alias.Description,
			Prefer: alias.Prefer, Members: members,
			Requires: requires{
				Tier: string(alias.Tier), Realtime: alias.RequireRealtime, Tools: alias.RequireTools,
				Languages: alias.Languages, Multilingual: alias.Multilingual,
				InputModalities: alias.RequireInputModalities, Only: alias.Only,
			},
		})
	}

	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(map[string]any{"models": models, "groups": groups})
}

// aliasNames are the shortcuts offered first, in the config's order, then the rest by name.
func aliasNames(config routing.ModalityConfig) []string {
	names := config.Offered()
	var rest []string
	for name := range config.Aliases {
		if !slices.Contains(names, name) {
			rest = append(rest, name)
		}
	}
	slices.Sort(rest)
	return append(names, rest...)
}

func bench(args []string) error {
	flags := flag.NewFlagSet("bench", flag.ExitOnError)
	models := flags.String("models", "", "comma-separated provider/model names")
	groupName := flags.String("group", "", "measure every model this shortcut routes over")
	all := flags.Bool("all", false, "measure every routed LLM")
	missing := flags.Bool("missing", false, "skip models the inventory already measured at the same effort with this suite")
	efforts := flags.String("efforts", "", "comma-separated reasoning efforts to sweep; empty sends what router.yaml sends")
	reps := flags.Int("reps", 3, "runs of each task")
	concurrency := flags.Int("concurrency", 8, "requests in flight across all models")
	inventoryPath := flags.String("inventory", "", "inventory.json to merge the results into; empty prints them")
	if err := flags.Parse(args); err != nil {
		return err
	}

	config, err := llmConfig()
	if err != nil {
		return err
	}
	targets, err := selectModels(config, *models, *groupName, *all)
	if err != nil {
		return err
	}
	inventory, err := readInventory(*inventoryPath)
	if err != nil {
		return err
	}

	sweep := []string{""}
	if *efforts != "" {
		sweep = strings.Split(*efforts, ",")
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))
	registry := llmrouter.DefaultRegistry()

	var subjects []subject
	for _, p := range targets {
		for _, effort := range sweep {
			s, err := build(registry, p, effort, logger)
			if err != nil {
				fmt.Fprintf(os.Stderr, "skip %s@%s: %v\n", p.Name(), effortOr(effort), err)
				continue
			}
			if *missing && inventory.measured(p.Name(), s.effort) {
				s.llm.Close()
				continue
			}
			subjects = append(subjects, s)
		}
	}
	if len(subjects) == 0 {
		return errors.New("llmgroups: nothing to measure")
	}

	results := run(subjects, *reps, *concurrency)
	for _, s := range subjects {
		s.llm.Close()
	}
	if *inventoryPath == "" {
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		return encoder.Encode(results)
	}
	for _, r := range results {
		inventory.record(r)
	}
	return inventory.write(*inventoryPath)
}

func effortOr(effort string) string {
	if effort == "" {
		return "default"
	}
	return effort
}

func selectModels(config routing.ModalityConfig, models, group string, all bool) ([]routing.ProviderConfig, error) {
	var names []string
	switch {
	case all:
		for _, p := range config.Providers {
			names = append(names, p.Name())
		}
	case group != "":
		members := config.Members(group)
		if members == nil {
			return nil, fmt.Errorf("llmgroups: %q is not an LLM shortcut", group)
		}
		for _, p := range members {
			names = append(names, p.Name())
		}
	case models != "":
		names = strings.Split(models, ",")
	default:
		return nil, errors.New("llmgroups: name -models, -group or -all")
	}

	var targets []routing.ProviderConfig
	for _, name := range names {
		if slices.ContainsFunc(targets, func(p routing.ProviderConfig) bool { return p.Name() == name }) {
			continue
		}
		p, ok := config.Provider(strings.TrimSpace(name))
		if !ok {
			return nil, fmt.Errorf("llmgroups: %q is not a routed LLM", name)
		}
		// The load benchmark's stand-in answers instantly and would top every speed column.
		if p.Provider == "mock" {
			continue
		}
		targets = append(targets, p)
	}
	return targets, nil
}

// subject is one model at one reasoning effort, built the way the router builds it.
type subject struct {
	config routing.ProviderConfig
	effort string
	// asked is the effort each request names, empty to leave the model at what router.yaml
	// builds it with. A sweep asks per request rather than per build because not every
	// factory passes Spec.ReasoningEffort on, while every provider validates the request's.
	asked string
	llm   llm.LLM
}

func build(registry *llmrouter.Registry, p routing.ProviderConfig, effort string, logger *slog.Logger) (subject, error) {
	spec := routing.Spec{
		Model:           p.Model,
		Thinking:        p.Thinking,
		ReasoningEffort: p.ReasoningEffort,
		BaseURL:         p.Endpoint,
		Region:          p.Region,
		Logger:          logger,
	}
	if p.APIKeyEnv != "" {
		spec.APIKey = os.Getenv(p.APIKeyEnv)
	}
	built, err := registry.Build(p.Provider, spec)
	if err != nil {
		return subject{}, err
	}
	if effort == "" {
		return subject{config: p, effort: effortOr(built.Capabilities().DefaultEffort), llm: built}, nil
	}
	if err := built.Capabilities().Validate(llm.ResponseParams{Reasoning: llm.ReasoningParams{Effort: effort}}); err != nil {
		built.Close()
		return subject{}, err
	}
	return subject{config: p, effort: effort, asked: effort, llm: built}, nil
}
