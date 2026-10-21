// Command phonecheck places test phone calls through a router and logs where each hop ran.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck/person"
)

type Config struct {
	Direction     person.Direction
	Provider      string
	Router        string
	Customer      string
	AgentNumber   string
	PersonNumber  string
	Talk          time.Duration
	Runs          int
	SIPLookupFrom string
	Region        string // free label, logged as region_label
	LogFormat     string // "text" | "json"
}

// provider makes a provider's person adapter.
type provider func() person.Person

const usage = "usage: phonecheck outbound|inbound -provider NAME -agent-number E164 -person-number E164 [flags]"

func main() {
	if err := runCommand(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runCommand(args []string) error {
	if len(args) == 0 {
		return errors.New(usage)
	}
	var dir person.Direction
	switch args[0] {
	case "outbound":
		dir = person.Outbound
	case "inbound":
		dir = person.Inbound
	default:
		return fmt.Errorf("unknown command %q; %s", args[0], usage)
	}
	providers := map[string]provider{}
	_, _, err := parseRun(dir, args[1:], providers)
	return err
}

// parseRun reads the flags after the subcommand. The provider is picked first, so its own
// flags can be parsed together with the shared ones.
func parseRun(dir person.Direction, args []string, providers map[string]provider) (Config, person.Person, error) {
	name := providerName(args)
	if name == "" {
		return Config{}, nil, errors.New("-provider is required")
	}
	makePerson, ok := providers[name]
	if !ok {
		return Config{}, nil, fmt.Errorf("unknown provider %q", name)
	}
	p := makePerson()

	cfg := Config{Direction: dir}
	fs := flag.NewFlagSet(dir.String(), flag.ContinueOnError)
	fs.StringVar(&cfg.Provider, "provider", "", "the person side's vendor")
	fs.StringVar(&cfg.Router, "router", "http://localhost:8080", "the router's address")
	fs.StringVar(&cfg.Customer, "customer", "", "the customer id; empty uses STREAM_ACCELERATION_CUSTOMER_ID")
	fs.StringVar(&cfg.AgentNumber, "agent-number", "", "the router's number the agent is on")
	fs.StringVar(&cfg.PersonNumber, "person-number", "", "the number the person side is on")
	fs.DurationVar(&cfg.Talk, "talk", 20*time.Second, "how long the person stays on")
	fs.IntVar(&cfg.Runs, "runs", 1, "how many calls to place, one after another")
	fs.StringVar(&cfg.SIPLookupFrom, "sip-lookup-from", "us", "where to resolve the Stream SIP host from")
	fs.StringVar(&cfg.Region, "region", "", "a free label, logged as region_label; it changes nothing")
	fs.StringVar(&cfg.LogFormat, "log-format", "text", "text or json")
	p.Flags(fs)
	if err := fs.Parse(args); err != nil {
		return Config{}, nil, err
	}
	if err := cfg.validate(); err != nil {
		return Config{}, nil, err
	}
	return cfg, p, nil
}

// providerName finds -provider's value in args before they are parsed.
func providerName(args []string) string {
	for i, arg := range args {
		flagName, value, hasValue := strings.Cut(strings.TrimLeft(arg, "-"), "=")
		if !strings.HasPrefix(arg, "-") || flagName != "provider" {
			continue
		}
		if hasValue {
			return value
		}
		if i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

func (c Config) validate() error {
	switch {
	case c.AgentNumber == "":
		return errors.New("-agent-number is required")
	case c.PersonNumber == "":
		return errors.New("-person-number is required")
	case c.Runs < 1:
		return errors.New("-runs must be at least 1")
	case c.LogFormat != "text" && c.LogFormat != "json":
		return errors.New("-log-format must be text or json")
	}
	return nil
}

func newLogger(format string, w io.Writer) *slog.Logger {
	if format == "json" {
		return slog.New(slog.NewJSONHandler(w, nil))
	}
	return slog.New(slog.NewTextHandler(w, nil))
}
