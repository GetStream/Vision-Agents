// Command chat holds a conversation with an agent in the terminal.
//
// Nothing is transcribed and nothing is spoken, so no call is joined and no Stream
// credentials are needed. The conversation is persisted, so leaving it and resuming it
// picks up where it was left.
//
//	STREAM_ACCELERATION_URL=http://localhost:8080 \
//	STREAM_ACCELERATION_CUSTOMER_ID=acme \
//	go run ./cmd/chat -name jean
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"os"
	"os/signal"

	"github.com/GetStream/Vision-Agents/sdks/go/agents"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
	"github.com/GetStream/Vision-Agents/tui"
)

const banner = `  ◢◣  VISION
 ◥◤   AGENTS`

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := run(ctx); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context) error {
	name := flag.String("name", "jean", "the agent to talk to, by the name its config is stored under")
	model := flag.String("llm", "", "the model that answers; empty leaves the backend's default")
	dir := flag.String("dir", "", "an agent directory to read instructions, skills and knowledge from")
	resume := flag.String("conversation", "", "a saved conversation to resume")
	logs := flag.String("log", "", "a file to write logs to; empty throws them away")
	flag.Parse()

	// The conversation owns the screen, so nothing else may write to it.
	logger, err := logging(*logs)
	if err != nil {
		return err
	}
	slog.SetDefault(logger)

	llm := stream.Accelerated(stream.Config{Agent: *name, LLM: *model, Logger: logger})
	err = agents.RegisterFunction(llm, "get_weather",
		"Get the current weather for a location",
		func(_ context.Context, in struct {
			Location string `json:"location" schema:"the city and state, e.g. Boulder, CO"`
		}) (any, error) {
			return fmt.Sprintf("It is 20 degrees and sunny in %s.", in.Location), nil
		})
	if err != nil {
		return err
	}

	agent, err := agents.New(agents.Options{
		Name:    *name,
		Dir:     *dir,
		UserID:  *name,
		LLM:     llm,
		Harness: agents.DefaultHarness(),
		Logger:  logger,
	})
	if err != nil {
		return err
	}

	return tui.Run(ctx, tui.Options{
		Open: func(ctx context.Context, conversationID string) (tui.Session, error) {
			return agent.Chat(ctx, agents.ChatOptions{Persist: true, ConversationID: conversationID})
		},
		History:        tui.BackendHistory(stream.Backend{}, *name),
		ConversationID: *resume,
		Branding: tui.Branding{
			Banner:      banner,
			Title:       agent.Name(),
			Subtitle:    "ask about the weather, or about anything this agent knows",
			Agent:       "AGENT",
			Placeholder: "Ask " + agent.Name() + " something…",
		},
		Logger: logger,
	})
}

// logging keeps the log off the screen: in a file if one was named, and nowhere if not.
func logging(path string) (*slog.Logger, error) {
	if path == "" {
		return slog.New(slog.DiscardHandler), nil
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, err
	}
	return slog.New(slog.NewTextHandler(file, nil)), nil
}
