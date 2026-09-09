// Command chat talks to a model through the LLM router: type a line, read the answer.
//
// The answer is printed as it streams rather than after the model has finished, so the
// latency the router optimises for is visible rather than merely reported. Each turn ends
// with a line saying which model answered, how long the first token took, how many tokens
// it spent and what that cost.
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	postgresEnvVar = "ROUTER_POSTGRES_DSN"
	redisEnvVar    = "ROUTER_REDIS_ADDR"
	configEnvVar   = "ROUTER_CONFIG"
)

// completionTimeout bounds how long one turn may take before the prompt comes back.
const completionTimeout = 2 * time.Minute

func main() {
	target := flag.String("target", "llm-fast", "provider/model or capability shortcut")
	customerID := flag.String("customer", "demo", "customer the usage is billed to")
	language := flag.String("language", "", "language hint, e.g. es")
	instructions := flag.String("system", "You are a helpful assistant. Keep answers short.",
		"system prompt")
	text := flag.String("text", "", "ask this and exit, instead of reading stdin")
	maxTokens := flag.Int("max-tokens", 0, "cap the answer, or 0 for the model's default")
	thinking := flag.Bool("thinking", false, "show the model's reasoning as it arrives")
	verbose := flag.Bool("verbose", false, "log lifecycle events")
	var tags routing.TagsFlag
	flag.Var(&tags, "tag", "cost label as key=value, repeat for several")
	flag.Parse()

	level := slog.LevelWarn
	if *verbose {
		level = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))
	slog.SetDefault(logger)

	options := options{
		target:       *target,
		customerID:   *customerID,
		language:     *language,
		instructions: *instructions,
		text:         *text,
		maxTokens:    *maxTokens,
		thinking:     *thinking,
		tags:         tags.Tags,
	}
	if err := run(options, logger); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

type options struct {
	target       string
	customerID   string
	language     string
	instructions string
	text         string
	maxTokens    int
	thinking     bool
	tags         routing.Tags
}

func (o options) languages() []string {
	if trimmed := strings.TrimSpace(o.language); trimmed != "" {
		return []string{trimmed}
	}
	return nil
}

func run(options options, logger *slog.Logger) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	router, cleanup, err := buildRouter(ctx, logger)
	if err != nil {
		return err
	}
	defer cleanup()

	session, err := router.Start(ctx, llmrouter.Request{
		CustomerID:    options.customerID,
		Tags:          options.tags,
		Target:        options.target,
		LanguageHints: options.languages(),
	})
	if err != nil {
		return err
	}
	defer session.Close()

	talker := newTalker(session, options)

	fmt.Printf("model: %s/%s\n", session.Provider(), session.Model())

	if options.text != "" {
		return talker.ask(ctx, options.text)
	}
	return repl(ctx, talker)
}

// repl answers each line as it is typed. A blank line is ignored rather than sent.
func repl(ctx context.Context, talker *talker) error {
	fmt.Println("type a line to ask it, or Ctrl-D to stop")

	lines := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !lines.Scan() {
			fmt.Println()
			return lines.Err()
		}

		line := strings.TrimSpace(lines.Text())
		if line == "" {
			continue
		}
		if line == "quit" || line == "exit" {
			return nil
		}

		if err := talker.ask(ctx, line); err != nil {
			return err
		}
	}
}

// buildRouter wires the router, using Postgres and Redis when they are configured. The demo
// is useful without them: it just stops recording usage.
func buildRouter(ctx context.Context, logger *slog.Logger) (*llmrouter.Router, func(), error) {
	config, err := routing.LoadConfig(os.Getenv(configEnvVar))
	if err != nil {
		return nil, nil, err
	}
	section, ok := config[routing.LLM]
	if !ok {
		return nil, nil, errors.New("the config declares no llm providers")
	}

	var closers []func()
	cleanup := func() {
		for i := len(closers) - 1; i >= 0; i-- {
			closers[i]()
		}
	}

	var pgStore *store.Store
	if dsn := os.Getenv(postgresEnvVar); dsn != "" {
		pgStore, err = store.Open(dsn)
		if err != nil {
			cleanup()
			return nil, nil, err
		}
		closers = append(closers, func() { pgStore.Close() })

		if err := pgStore.Migrate(ctx); err != nil {
			cleanup()
			return nil, nil, err
		}
	}

	var liveClient *live.Client
	if address := os.Getenv(redisEnvVar); address != "" {
		liveClient, err = live.New(live.Options{Address: address})
		if err != nil {
			cleanup()
			return nil, nil, err
		}
		closers = append(closers, liveClient.Close)
	}

	chatRouter, err := llmrouter.New(llmrouter.Options{
		Config:   section,
		Registry: llmrouter.DefaultRegistry(),
		Store:    pgStore,
		Live:     liveClient,
		Logger:   logger,
	})
	if err != nil {
		cleanup()
		return nil, nil, err
	}
	closers = append(closers, chatRouter.Close)

	return chatRouter, cleanup, nil
}

// talker asks one thing at a time, so what is printed stays next to what was asked. It keeps
// the conversation so far, which is also what a failover would need to carry across.
type talker struct {
	session *llmrouter.Session
	options options
	price   routing.Price

	// history is the conversation so far, sent in full on every turn.
	history []llm.Message
}

func newTalker(session *llmrouter.Session, options options) *talker {
	return &talker{session: session, options: options, price: session.Price()}
}

// ask sends one turn, prints the answer as it arrives, and reports what the turn cost.
func (t *talker) ask(ctx context.Context, question string) error {
	t.history = append(t.history, llm.Message{Role: llm.User, Content: question})

	turn, cancel := context.WithTimeout(ctx, completionTimeout)
	defer cancel()

	stream, err := t.session.Create(turn, llm.ResponseParams{
		ID:              fmt.Sprintf("chat-%d", time.Now().UnixNano()),
		Instructions:    t.options.instructions,
		Input:           t.history,
		MaxOutputTokens: t.options.maxTokens,
	})
	if err != nil {
		return err
	}
	// Ctrl-C during a turn is barge-in, which is the thing this is for.
	stop := context.AfterFunc(ctx, func() { stream.Close() })
	defer stop()

	for stream.Next() {
		switch typed := stream.Current().(type) {
		case llm.OutputTextDelta:
			fmt.Print(typed.Delta)
		case llm.ReasoningTextDelta:
			// Thinking is not the answer, so it is only shown when asked for.
			if t.options.thinking {
				fmt.Fprint(os.Stderr, typed.Delta)
			}
		case llm.ResponseFailed:
			fmt.Fprintf(os.Stderr, "provider error (%s): %v\n", typed.Provider, typed.Err)
		}
	}

	response := stream.Response()
	// The answer joins the history so the next turn has the context, exactly as an agent
	// would keep it.
	t.history = append(t.history, llm.Message{Role: llm.Assistant, Content: response.OutputText})
	fmt.Printf("\n%s\n", t.summarise(response))
	return stream.Err()
}

// summarise is the line printed after each turn: what answered, how long the reader waited,
// how many tokens it spent and what it cost.
func (t *talker) summarise(response llm.Response) string {
	usage := response.Usage
	costMicros := t.price.CostMicros(routing.Usage{
		InputTokens:       usage.InputTokens,
		CachedInputTokens: usage.InputTokensDetails.CachedTokens,
		OutputTokens:      usage.OutputTokens,
	})

	summary := fmt.Sprintf("%s/%s  first token %.0fms  %d in  %d out  $%.6f",
		response.Provider, response.Model,
		response.TimeToFirstTokenMs,
		usage.InputTokens, usage.OutputTokens,
		float64(costMicros)/1_000_000)

	if usage.InputTokensDetails.CachedTokens > 0 {
		summary += fmt.Sprintf("  (%d cached)", usage.InputTokensDetails.CachedTokens)
	}
	if usage.OutputTokensDetails.ReasoningTokens > 0 {
		summary += fmt.Sprintf("  (%d thinking)", usage.OutputTokensDetails.ReasoningTokens)
	}
	if response.Status != llm.StatusCompleted {
		summary += fmt.Sprintf("  (%s)", response.Status)
	}
	return summary
}
