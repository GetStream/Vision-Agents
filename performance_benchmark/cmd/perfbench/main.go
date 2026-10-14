// Command perfbench finds how many cascaded voice agents a router holds at once.
//
// Every agent is a real conversation: the Go SDK opens a session, the backend joins the
// call, and a caller here joins the same call over WebRTC and talks to it. What is not
// real is the three models, which are the router's own mock/* providers: they reach no
// vendor and bill nothing, so the only thing a run measures is the router.
//
//	perfbench run -target staging -start 1 -max 128 -hold 90s -out out/before
//	perfbench compare out/before out/after
//
// A run needs STREAM_API_KEY and STREAM_API_SECRET, a router built with
// ROUTER_MOCK_PROVIDERS=true, and the webrtc build tag:
//
//	CGO_ENABLED=1 go run -tags webrtc ./cmd/perfbench run ...
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/GetStream/Vision-Agents/performance_benchmark/internal/bench"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// instructions are what the agent is, which the mock model never reads. They are here so
// the stored config is a real one rather than an empty one.
const instructions = "You are a load test. Answer in one short sentence."

// hosted is a router this is meant to be run against, and where its pods are, so a run can
// say what they were doing. The router carries no component label, and its release name is
// what tells it from the gateway's.
type hosted struct {
	url, namespace, selector string
}

// targets are the hosted routers -target names. benchmark is the release deploys are
// rehearsed on, so a run there disturbs nobody; staging is the one people use.
var targets = map[string]hosted{
	"staging": {
		url:       "https://accelerate.gcp.stream-io-api.com",
		namespace: "accelerate-staging",
		selector:  "app.kubernetes.io/name=accelerate,app.kubernetes.io/instance=chat-accelerate-staging",
	},
	"benchmark": {
		url:       "https://accelerate-benchmark.gcp.stream-io-api.com",
		namespace: "accelerate-benchmark",
		selector:  "app.kubernetes.io/name=accelerate,app.kubernetes.io/instance=chat-accelerate-benchmark",
	},
}

// hostedContext is the kubectl context both hosted routers are in.
const hostedContext = "gke_stream-gcp-2026-v2_us-east1_us-east1-acceleration"

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "perfbench:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return errors.New("say what to do: run, or compare <before> <after>")
	}
	switch args[0] {
	case "run":
		return ramp(args[1:])
	case "compare":
		return compare(args[1:])
	default:
		return fmt.Errorf("unknown command %q: run, or compare <before> <after>", args[0])
	}
}

func ramp(args []string) error {
	flags := flag.NewFlagSet("run", flag.ExitOnError)
	target := flags.String("target", "staging", "staging, benchmark, or a router URL")
	label := flags.String("label", "", "what this run is of, for the comparison")
	agent := flags.String("agent", "perfbench", "the stored config the conversations run")
	start := flags.Int("start", 1, "how many agents the first step runs")
	factor := flags.Int("factor", 2, "what each step multiplies the last by")
	maximum := flags.Int("max", 64, "where to stop, whatever is still holding")
	hold := flags.Duration("hold", 90*time.Second, "how long each step talks for")
	namespace := flags.String("namespace", "", "the router's namespace, for pod readings; empty with a hosted -target uses its own")
	kubeContext := flags.String("kube-context", "", "the kubectl context the namespace is in; empty is the current one")
	selector := flags.String("selector", "", "which pods are the router; empty with a hosted -target uses its release's")
	out := flags.String("out", "", "where to write the report; empty is out/<time>")
	stt := flags.String("stt", "mock/mock-stt", "the transcriber")
	tts := flags.String("tts", "mock/mock-tts", "the voice")
	model := flags.String("llm", "mock/mock-llm", "the model that answers")
	if err := flags.Parse(args); err != nil {
		return err
	}

	url := *target
	if known, ok := targets[url]; ok {
		url = known.url
		// A hosted router is behind the authenticating proxy, which refuses the customer
		// header a local .env sets.
		if err := os.Unsetenv(stream.CustomerEnv); err != nil {
			return err
		}
		if *namespace == "" {
			*namespace = known.namespace
			if *kubeContext == "" {
				*kubeContext = hostedContext
			}
			if *selector == "" {
				*selector = known.selector
			}
		}
	}
	if err := os.Setenv(stream.URLEnv, url); err != nil {
		return err
	}

	directory := *out
	if directory == "" {
		directory = filepath.Join("out", time.Now().Format("20060102-150405"))
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	report, err := bench.Run(ctx, bench.Options{
		Label:        *label,
		Agent:        *agent,
		Instructions: instructions,
		Models:       bench.Models{STT: *stt, TTS: *tts, LLM: *model},
		Start:        *start,
		Factor:       *factor,
		Max:          *maximum,
		Hold:         *hold,
		Target:       url,
		Cluster:      bench.Cluster{Namespace: *namespace, Selector: *selector, Context: *kubeContext},
		Logger:       logger,
	})
	if err != nil {
		return err
	}
	if err := report.Save(directory); err != nil {
		return err
	}

	fmt.Println(report.Markdown())
	fmt.Fprintln(os.Stderr, "written to", directory)
	return nil
}

func compare(args []string) error {
	if len(args) != 2 {
		return errors.New("compare takes the run directory before the change and the one after")
	}
	before, err := bench.Load(args[0])
	if err != nil {
		return err
	}
	after, err := bench.Load(args[1])
	if err != nil {
		return err
	}
	fmt.Println(bench.Compare(before, after))
	return nil
}
