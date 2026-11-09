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
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

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
	if err := uploadDocuments(ctx, client, config.Name, filepath.Join(dir, "knowledge")); err != nil {
		return err
	}
	fmt.Printf("ui         A2UI\n")
	fmt.Printf("visible    %v\n", visibleTools)
	fmt.Printf("skills     %v\n", skills)
	fmt.Printf("\nnext       STREAM_API_SECRET=<the router's> GOWORK=off go run ./backend\n")
	return nil
}

// documentTypes are the files of knowledge/ the sync leaves out, which the router reads itself:
// it turns each into Markdown that keeps its headings, lists and tables, page by page.
var documentTypes = map[string]bool{
	".pdf": true, ".docx": true, ".doc": true, ".xlsx": true, ".rtf": true, ".odt": true,
}

// uploadDocuments uploads the documents in dir into the agent's knowledge base, by file name, so
// uploading one again replaces it, and waits for the router to have read each.
func uploadDocuments(ctx context.Context, client *acceleration.ClientWithResponses, namespace, dir string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	for _, entry := range entries {
		if entry.IsDir() || !documentTypes[strings.ToLower(filepath.Ext(entry.Name()))] {
			continue
		}
		id, err := uploadDocument(ctx, client, namespace, filepath.Join(dir, entry.Name()))
		if err != nil {
			return err
		}
		state, err := waitForDocument(ctx, client, id)
		if err != nil {
			return err
		}
		fmt.Printf("knowledge  %s: %s\n", entry.Name(), state)
	}
	return nil
}

func uploadDocument(ctx context.Context, client *acceleration.ClientWithResponses, namespace, path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	part, err := form.CreateFormFile("file", filepath.Base(path))
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(part, file); err != nil {
		return "", err
	}
	if err := form.Close(); err != nil {
		return "", err
	}
	source := filepath.Base(path)
	uploaded, err := client.UploadKnowledgeFileWithBodyWithResponse(ctx,
		&acceleration.UploadKnowledgeFileParams{Namespace: namespace, Source: &source},
		form.FormDataContentType(), &body)
	if err != nil {
		return "", err
	}
	if uploaded.JSON202 == nil {
		return "", fmt.Errorf("the router refused %s: %s %s", source, uploaded.Status(), uploaded.Body)
	}
	return uploaded.JSON202.Id, nil
}

// waitForDocument waits for the router to have read a document, for at most two minutes.
func waitForDocument(ctx context.Context, client *acceleration.ClientWithResponses, id string) (string, error) {
	for range 60 {
		read, err := client.GetKnowledgeDocumentWithResponse(ctx, id)
		if err != nil {
			return "", err
		}
		if read.JSON200 == nil {
			return "", fmt.Errorf("reading document %s: %s", id, read.Status())
		}
		switch read.JSON200.State {
		case acceleration.KnowledgeDocumentStateIndexed:
			return fmt.Sprintf("read into %d passages", read.JSON200.Passages), nil
		case acceleration.KnowledgeDocumentStateFailed:
			reason := "no reason given"
			if read.JSON200.Error != nil {
				reason = *read.JSON200.Error
			}
			return "", fmt.Errorf("the router could not read document %s: %s", id, reason)
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(2 * time.Second):
		}
	}
	return "still being read; check again later", nil
}
