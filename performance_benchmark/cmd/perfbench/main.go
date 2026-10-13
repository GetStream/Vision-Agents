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

// staging is the hosted router this is meant to be run against.
const staging = "https://accelerate.gcp.stream-io-api.com"

// Where the staging router's pods are, so a run can say what they were doing. The router
// carries no component label, and its release name is what tells it from the gateway's.
const (
	stagingNamespace = "accelerate-staging"
	stagingContext   = "gke_stream-gcp-2026-v2_us-east1_us-east1-acceleration"
	stagingSelector  = "app.kubernetes.io/name=accelerate,app.kubernetes.io/instance=chat-accelerate-staging"
)

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
	target := flags.String("target", "staging", "staging, or a router URL")
	label := flags.String("label", "", "what this run is of, for the comparison")
	agent := flags.String("agent", "perfbench", "the stored config the conversations run")
	start := flags.Int("start", 1, "how many agents the first step runs")
	factor := flags.Int("factor", 2, "what each step multiplies the last by")
	maximum := flags.Int("max", 64, "where to stop, whatever is still holding")
	hold := flags.Duration("hold", 90*time.Second, "how long each step talks for")
	namespace := flags.String("namespace", "", "the router's namespace, for pod readings; empty with -target staging uses "+stagingNamespace)
	kubeContext := flags.String("kube-context", "", "the kubectl context the namespace is in; empty is the current one")
	selector := flags.String("selector", "", "which pods are the router; empty with -target staging uses the staging release's")
	out := flags.String("out", "", "where to write the report; empty is out/<time>")
	stt := flags.String("stt", "mock/mock-stt", "the transcriber")
	tts := flags.String("tts", "mock/mock-tts", "the voice")
	model := flags.String("llm", "mock/mock-llm", "the model that answers")
	if err := flags.Parse(args); err != nil {
		return err
	}

	url := *target
	if url == "staging" {
		url = staging
		if *namespace == "" {
			*namespace = stagingNamespace
			if *kubeContext == "" {
				*kubeContext = stagingContext
			}
			if *selector == "" {
				*selector = stagingSelector
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
