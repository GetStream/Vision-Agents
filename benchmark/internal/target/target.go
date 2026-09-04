package target

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
)

const (
	PythonName       = "python"
	AccelerationName = "acceleration"
	AcceleratedName  = "accelerated"
	LiveKitName      = "livekit"
)

// Target is a system Voicebench can start for one benchmark call.
type Target interface {
	Prepare(ctx context.Context) (func(), error)
	StartCall(ctx context.Context, callID string, callType string) (func(), error)
}

// Config contains the common inputs used to construct a target.
type Config struct {
	Root              string
	Pack              string
	URL               string
	Bin               string
	WorldURL          string
	Spawn             bool
	LiveKitAgentName  string
	LiveKitDeployment string
	Logger            *slog.Logger
}

// Definition identifies a supported target and its media transport.
type Definition struct {
	Transport string
	System    string
	build     func(Config) Target
}

var definitions = map[string]Definition{
	PythonName: {
		Transport: "stream",
		System:    "vision-agents",
		build: func(cfg Config) Target {
			return &Python{
				Root:     cfg.Root,
				Pack:     cfg.Pack,
				URL:      cfg.URL,
				Spawn:    cfg.Spawn,
				WorldURL: cfg.WorldURL,
				Logger:   cfg.Logger,
			}
		},
	},
	AccelerationName: {
		Transport: "stream",
		System:    "acceleration",
		build: func(cfg Config) Target {
			return &Acceleration{
				Root:     cfg.Root,
				Pack:     cfg.Pack,
				URL:      cfg.URL,
				Spawn:    cfg.Spawn,
				Bin:      cfg.Bin,
				WorldURL: cfg.WorldURL,
				Logger:   cfg.Logger,
			}
		},
	},
	AcceleratedName: {
		Transport: "stream",
		System:    "accelerated",
		build: func(cfg Config) Target {
			return &Accelerated{
				Python: Python{
					Root:     cfg.Root,
					Pack:     cfg.Pack,
					URL:      cfg.URL,
					Spawn:    cfg.Spawn,
					WorldURL: cfg.WorldURL,
					Pipeline: "accelerated",
					Logger:   cfg.Logger,
				},
				Bin: cfg.Bin,
			}
		},
	},
	LiveKitName: {
		Transport: "livekit",
		System:    "livekit",
		build: func(cfg Config) Target {
			return &LiveKit{
				Root:       cfg.Root,
				Pack:       cfg.Pack,
				URL:        cfg.URL,
				Spawn:      cfg.Spawn,
				AgentName:  cfg.LiveKitAgentName,
				Deployment: cfg.LiveKitDeployment,
				WorldURL:   cfg.WorldURL,
				Logger:     cfg.Logger,
			}
		},
	},
}

// Lookup returns the definition for a named target.
func Lookup(name string) (Definition, bool) {
	definition, ok := definitions[name]
	return definition, ok
}

// Names returns the supported target names.
func Names() []string {
	names := make([]string, 0, len(definitions))
	for name := range definitions {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Build constructs a named target.
func Build(name string, cfg Config) (Target, error) {
	definition, ok := Lookup(name)
	if !ok {
		return nil, fmt.Errorf("unknown target %q (want one of %v)", name, Names())
	}
	return definition.build(cfg), nil
}

// Noop joins no agent. It is used when --call-id points at an agent already in the call.
type Noop struct{}

func (Noop) Prepare(context.Context) (func(), error) { return func() {}, nil }

func (Noop) StartCall(context.Context, string, string) (func(), error) { return func() {}, nil }
