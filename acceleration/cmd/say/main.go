// Command say turns text into speech through the TTS router: type a line, hear it.
//
// Audio is played as it arrives rather than after the sentence is finished, so the latency
// the router optimises for is audible rather than merely reported. Playback is handed to
// ffplay, which keeps this buildable anywhere; -out writes a WAV file instead.
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
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

const (
	postgresEnvVar = "ROUTER_POSTGRES_DSN"
	redisEnvVar    = "ROUTER_REDIS_ADDR"
	configEnvVar   = "ROUTER_CONFIG"
)

// synthesisTimeout bounds how long one utterance may take before the prompt comes back.
const synthesisTimeout = 2 * time.Minute

func main() {
	target := flag.String("target", "en-low-latency", "provider/model or capability shortcut")
	voice := flag.String("voice", "", "provider-specific voice id")
	customerID := flag.String("customer", "demo", "customer the usage is billed to")
	language := flag.String("language", "", "language hint, e.g. es")
	text := flag.String("text", "", "say this and exit, instead of reading stdin")
	out := flag.String("out", "", "write a WAV file instead of playing the audio")
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
		target:     *target,
		voice:      *voice,
		customerID: *customerID,
		language:   *language,
		text:       *text,
		out:        *out,
		tags:       tags.Tags,
	}
	if err := run(options, logger); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

type options struct {
	target     string
	voice      string
	customerID string
	language   string
	text       string
	out        string
	tags       routing.Tags
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

	audioSink, err := newSink(options.out)
	if err != nil {
		return err
	}
	// The sink is closed last, so the session has stopped producing audio by the time the
	// file is finalised or playback is allowed to finish.
	defer audioSink.Close()

	session, err := router.Start(ctx, ttsrouter.Request{
		CustomerID:    options.customerID,
		Tags:          options.tags,
		Target:        options.target,
		LanguageHints: options.languages(),
		Voice:         options.voice,
	})
	if err != nil {
		return err
	}
	defer session.Close()

	speaker := newSpeaker(session, audioSink)
	go speaker.consume()

	fmt.Printf("voice: %s/%s\n", session.Provider(), session.Model())

	if options.text != "" {
		return speaker.say(ctx, options.text)
	}
	return repl(ctx, speaker)
}

// repl says each line as it is typed. A blank line is ignored rather than synthesised.
func repl(ctx context.Context, speaker *speaker) error {
	fmt.Println("type a line to hear it, or Ctrl-D to stop")

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

		if err := speaker.say(ctx, line); err != nil {
			return err
		}
	}
}

// buildRouter wires the router, using Postgres and Redis when they are configured. The
// demo is useful without them: it just stops recording usage.
func buildRouter(ctx context.Context, logger *slog.Logger) (*ttsrouter.Router, func(), error) {
	config, err := routing.LoadConfig(os.Getenv(configEnvVar))
	if err != nil {
		return nil, nil, err
	}
	section, ok := config[routing.TTS]
	if !ok {
		return nil, nil, errors.New("the config declares no tts providers")
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

	voiceRouter, err := ttsrouter.New(ttsrouter.Options{
		Config:   section,
		Registry: ttsrouter.DefaultRegistry(),
		Store:    pgStore,
		Live:     liveClient,
		Logger:   logger,
	})
	if err != nil {
		cleanup()
		return nil, nil, err
	}
	closers = append(closers, voiceRouter.Close)

	return voiceRouter, cleanup, nil
}

// speaker says one thing at a time, so what is printed stays next to what was heard.
type speaker struct {
	session *ttsrouter.Session
	sink    sink
	price   routing.Price

	// settled carries the summary of each finished utterance.
	settled chan tts.SynthesisComplete
}

func newSpeaker(session *ttsrouter.Session, audioSink sink) *speaker {
	return &speaker{
		session: session,
		sink:    audioSink,
		price:   session.Price(),
		settled: make(chan tts.SynthesisComplete, 4),
	}
}

// say synthesises one utterance and waits for that utterance to finish. It waits on the id
// it sent, so the summary of an utterance that timed out earlier cannot be mistaken for
// this one's.
func (s *speaker) say(ctx context.Context, text string) error {
	request := tts.Request{
		ID:    fmt.Sprintf("say-%d", time.Now().UnixNano()),
		Text:  text,
		Final: true,
	}
	if err := s.session.Synthesize(request); err != nil {
		return err
	}

	deadline := time.After(synthesisTimeout)
	for {
		select {
		case complete := <-s.settled:
			if complete.SynthesisID != request.ID {
				continue
			}
			fmt.Println(s.summarise(complete))
			return nil
		case <-ctx.Done():
			// Ctrl-C during an utterance is barge-in, which is the thing this is for.
			return s.session.Interrupt()
		case <-deadline:
			return fmt.Errorf("gave up waiting for %q after %s", text, synthesisTimeout)
		}
	}
}

// consume plays audio as it arrives and reports each utterance once it settles.
func (s *speaker) consume() {
	for event := range s.session.Events() {
		switch typed := event.(type) {
		case tts.AudioChunk:
			if err := s.sink.Write(typed.Audio); err != nil {
				fmt.Fprintf(os.Stderr, "playback failed: %v\n", err)
			}
		case tts.SynthesisComplete:
			select {
			case s.settled <- typed:
			default:
			}
		case tts.Error:
			fmt.Fprintf(os.Stderr, "provider error (%s): %v\n", typed.Provider, typed.Err)
		}
	}
}

// summarise is the line printed after each utterance: what said it, how long the listener
// waited, how much speech came back and what it cost.
func (s *speaker) summarise(complete tts.SynthesisComplete) string {
	costMicros := s.price.CostMicros(routing.Usage{
		Characters: complete.Characters,
		AudioMs:    int64(complete.AudioDurationMs),
	})

	summary := fmt.Sprintf("%s/%s  first audio %.0fms  audio %.1fs  %d chars  $%.6f",
		complete.Provider, complete.Model,
		complete.TimeToFirstByteMs, complete.AudioDurationMs/1000,
		complete.Characters, float64(costMicros)/1_000_000)

	if complete.Interrupted {
		summary += "  (interrupted)"
	}
	return summary
}
