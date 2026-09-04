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

	"github.com/GetStream/Vision-Agents/agents-core-go/agents"
	"github.com/GetStream/Vision-Agents/agents-core-go/stream"
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

	if err := agents.RegisterFunction(llm, "get_weather",
		"Get the current weather for a location",
		func(_ context.Context, in struct {
			Location string `json:"location" schema:"the city and state, e.g. Boulder, CO"`
		}) (any, error) {
			return fmt.Sprintf("It is 20 degrees and sunny in %s.", in.Location), nil
		}); err != nil {
		return err
	}

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
		if err := session.Respond(question); err != nil {
			return err
		}
	}
	return lines.Err()
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
