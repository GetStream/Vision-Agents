// Command text holds a conversation in writing.
//
// Nothing is transcribed and nothing is spoken, so no call is joined and no Stream
// credentials are needed. Everything between hearing a question and answering it is
// unchanged: the same instructions, the same skills and the same functions a call
// would have had.
//
//	STREAM_ACCELERATION_URL=http://localhost:8080 \
//	STREAM_ACCELERATION_CUSTOMER_ID=acme \
//	go run ./examples/text
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"

	"github.com/GetStream/Vision-Agents/sdks/go/agents"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := run(ctx); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context) error {
	llm := stream.Accelerated(stream.Config{LLM: "llm-fast"})

	agent, err := agents.New(agents.Options{
		Name:         "jean",
		Instructions: "You are Jean, a friendly assistant. Keep answers to a sentence or two.",
		LLM:          llm,
		Harness:      agents.DefaultHarness(),
		CostTracking: map[string]string{"customer_id": "123"},
		MemoryFilter: map[string]string{"user_id": "123"},
	})
	if err != nil {
		return err
	}
	if err := agent.Tools().Add(GetWeather{}); err != nil {
		return err
	}

	session, err := agent.Chat(ctx)
	if err != nil {
		return err
	}
	defer session.Close(context.WithoutCancel(ctx))

	go printReplies(session)

	fmt.Println("Ask Jean something. Ctrl-D to leave.")
	lines := bufio.NewScanner(os.Stdin)
	for lines.Scan() {
		question := strings.TrimSpace(lines.Text())
		if question == "" {
			continue
		}
		if _, err := session.Responses.Create(ctx, question); err != nil {
			return err
		}
	}
	return lines.Err()
}

// GetWeather is the one tool Jean has. Its field is what the model fills in.
type GetWeather struct {
	Location string `json:"location" schema:"the city and state, e.g. Boulder, CO"`
}

func (GetWeather) Name() string        { return "get_weather" }
func (GetWeather) Description() string { return "Get the current weather for a location" }
func (w GetWeather) Run(context.Context) (any, error) {
	return fmt.Sprintf("It is 20 degrees and sunny in %s.", w.Location), nil
}

// printReplies writes what the agent said as it says it.
func printReplies(session *agents.Session) {
	for event := range session.Events() {
		switch event.Kind {
		case "response_delta":
			fmt.Print(event.Text)
		case "responded":
			fmt.Println()
		case "error":
			fmt.Fprintln(os.Stderr, "error:", event.Error)
		}
	}
}
