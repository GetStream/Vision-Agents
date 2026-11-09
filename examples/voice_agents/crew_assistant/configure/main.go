// Command configure stores the crew assistant this directory describes on the router.
//
// Writing an agent config is server-side only, so the app never does it: it only names the
// agent this leaves behind. agent.yaml says what runs it and instructions.md what it is told;
// the Go SDK syncs both, then this adds what teaches it to answer with interactive UI in A2UI,
// the tools whose steps a reply shows and the skills it has.
//
// Run it once, and again whenever the directory changes:
//
//	STREAM_ACCELERATION_URL=http://localhost:8080 \
//	STREAM_ACCELERATION_CUSTOMER_ID=examples \
//	GOWORK=off go run ./configure
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/agents"
)

// visibleTools are the tools whose steps the person sees on a reply. search_handbook's result
// also adds the handbook passages it found to the reply's sources.
var visibleTools = []string{
	"search", "web_search",
	"get_my_schedule", "check_callout", "find_coverage", "request_shift_swap", "notify_shift_leader",
	"search_handbook", "get_updates_since", "get_equipment_guide", "create_support_ticket",
	"get_team", "book_check_in", "get_menu", "save_shift_reflection", "calcom__*",
}

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

	told, err := os.ReadFile(filepath.Join(dir, "instructions.md"))
	if err != nil {
		return err
	}
	ui, err := os.ReadFile(filepath.Join(dir, "a2ui", "instructions.md"))
	if err != nil {
		return err
	}
	instructions := string(told) + "\n\n## Interactive UI in A2UI\n\n" + string(ui)

	client, err := agent.Client()
	if err != nil {
		return err
	}
	skills := []string{"think", "recall", "explain", "vision"}
	patched, err := client.PatchAgentConfigWithResponse(ctx, config.Id,
		acceleration.PatchAgentConfigJSONRequestBody{
			Instructions: &instructions,
			Skills:       &skills,
			VisibleTools: &visibleTools,
		})
	if err != nil {
		return err
	}
	if patched.JSON200 == nil {
		return fmt.Errorf("the router refused the config: %s %s", patched.Status(), patched.Body)
	}
	fmt.Printf("ui         A2UI\n")
	fmt.Printf("visible    %v\n", visibleTools)
	fmt.Printf("skills     %v\n", skills)
	fmt.Printf("\nnext       STREAM_API_SECRET=<the router's> GOWORK=off go run ./backend\n")
	return nil
}
