// Command configure stores the assistant this directory describes on the router.
//
// It is the server-side half of the assistant: writing an agent config is not marked
// x-client-accessible, so the router refuses it from a phone, and the app only ever names
// the agent this leaves behind. agent.yaml says what runs it and instructions.md what it is
// told; the Go SDK reads both and syncs them in one request, to Stream's hosted router under
// the app STREAM_API_KEY names, which is the one the iOS app is built with.
//
// Run it once, and again whenever the directory changes:
//
//	STREAM_API_KEY=<your Stream app's key> \
//	STREAM_API_SECRET=<its secret> \
//	go run ./configure
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/agents"
)

func main() {
	dir := flag.String("dir", ".", "the agent directory to read")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := run(ctx, *dir); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, dir string) error {
	agent, err := agents.New(agents.Options{Dir: dir})
	if err != nil {
		return err
	}
	config, err := agent.Sync(ctx)
	if err != nil {
		return err
	}
	fmt.Printf("agent      %s (config %s)\n", config.Name, config.Id)

	// agent.yaml cannot say which tools' steps a reply shows, or which skills the agent has,
	// so both are patched on after the sync, which leaves a field it does not mention as it
	// is. Naming any tools replaces the default of search and web_search, so those are named
	// again beside the phone's own; naming skills replaces the built-in think, recall and
	// explain, so those stay beside vision, which looks through the phone's camera on a call.
	client, err := agent.Client()
	if err != nil {
		return err
	}
	visible := []string{"search", "web_search", "get_location", "list_events", "add_reminder", "send_email", "gmail__*", "linear__*"}
	skills := []string{"think", "recall", "explain", "vision"}
	patched, err := client.PatchAgentConfigWithResponse(ctx, config.Id,
		acceleration.PatchAgentConfigJSONRequestBody{VisibleTools: &visible, Skills: &skills})
	if err != nil {
		return err
	}
	if patched.JSON200 == nil {
		return fmt.Errorf("the router refused the visible tools: %s", patched.Status())
	}
	fmt.Printf("visible    %v\n", visible)
	fmt.Printf("skills     %v\n", skills)
	fmt.Printf("\nnext       STREAM_API_KEY=<the same key> STREAM_API_SECRET=<its secret> go run ./backend\n")
	fmt.Printf("           open examples/voice_agents/swift_assistant/app/SwiftAssistant.xcodeproj and run it\n")
	return nil
}
