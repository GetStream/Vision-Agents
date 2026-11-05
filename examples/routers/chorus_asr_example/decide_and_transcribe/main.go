// Command decide_and_transcribe holds a call on which chorus-asr writes what the caller says
// and decides when they have finished, and the agent's own model and voice answer.
//
// Pick this when one model should hear the caller for both the words and the turn: no other
// transcriber, and turn-taking tuned in agent.yaml's end_of_turn.
//
// The router runs with ROUTER_CHORUS_ASR_URL and ROUTER_CHORUS_ASR_MODE=act (../README.md).
//
//	go run .
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"

	"github.com/joho/godotenv"

	"github.com/GetStream/Vision-Agents/sdks/go/agents"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	_ = godotenv.Load("../../../../.env")

	// stt: stream/chorus-asr and end_of_turn: {max_wait_ms: 1100}, beside its llm and tts.
	agent, err := agents.New(agents.Options{Dir: "../agents/chorus-concierge"})
	if err != nil {
		log.Fatal(err)
	}
	if _, err := agent.Sync(ctx); err != nil {
		log.Fatal(err)
	}
	if err := talk(ctx, agent); err != nil {
		log.Fatal(err)
	}
}

// talk joins a Stream call, prints a link a person can open to speak on it, and prints the
// conversation until the call ends.
func talk(ctx context.Context, agent *agents.Agent) error {
	session, err := agent.Join(ctx)
	if err != nil {
		return err
	}
	defer session.Close(context.WithoutCancel(ctx))
	if link, err := session.MonitorURL(); err == nil {
		fmt.Println("Open this to talk to the agent:", link)
	}
	for event := range session.Events() {
		switch event.Kind {
		case "heard":
			fmt.Println("caller:", event.Text)
		case "responded":
			fmt.Println("agent: ", event.Text)
		case "error":
			fmt.Fprintln(os.Stderr, "error:", event.Error)
		}
	}
	return nil
}
