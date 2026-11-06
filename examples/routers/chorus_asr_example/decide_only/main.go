// Command decide_only holds a call on which another transcriber writes what the caller says
// and chorus-asr, hearing their audio, decides when they have finished.
//
// Pick this to keep the transcriber you have, for its languages or its words, and take the
// turn-taking from the audio rather than from the words.
//
// The router runs with ROUTER_CHORUS_ASR_URL, and the agent opts in with floor_control in its
// agent.yaml (../README.md).
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

	// stt: deepgram/flux-general-en, and floor_control: stream/chorus-asr:
	// chorus-asr decides the turns by the server's default end_of_turn.
	agent, err := agents.New(agents.Options{Dir: "../agents/clinic-front-desk"})
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
