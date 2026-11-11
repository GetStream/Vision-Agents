// Command configure finishes what `getstream agents sync` leaves out of the crew assistant.
//
// The CLI syncs the folder: agent.yaml, instructions.md, the Markdown in knowledge/ and the
// simulations. This uploads the documents in knowledge/ it skips, such as PDFs, which the
// router reads into Markdown itself, and sets the tools whose steps a reply shows and the skills
// the agent has, which no file in the folder holds. Run it after the first sync, and again
// whenever a document changes, from configure/:
//
//	getstream agents sync ..
//	GOWORK=off go run . -dir ..
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
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
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
	name := flag.String("agent", "crew_assistant", "the agent the CLI synced")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := run(ctx, *dir, *name); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, dir, name string) error {
	client, err := stream.Backend{}.Client()
	if err != nil {
		return err
	}
	configs, err := client.ListAgentConfigsWithResponse(ctx, &acceleration.ListAgentConfigsParams{Name: &name})
	if err != nil {
		return err
	}
	if configs.JSON200 == nil {
		return fmt.Errorf("finding %s: %s %s", name, configs.Status(), configs.Body)
	}
	if len(*configs.JSON200) == 0 {
		return fmt.Errorf("there is no agent %s yet: run getstream agents sync first", name)
	}
	config := (*configs.JSON200)[0]
	fmt.Printf("agent      %s (config %s)\n", config.Name, config.Id)

	skills := []string{"think", "recall", "explain", "vision"}
	patched, err := client.PatchAgentConfigWithResponse(ctx, config.Id,
		acceleration.PatchAgentConfigJSONRequestBody{Skills: &skills, VisibleTools: &visibleTools})
	if err != nil {
		return err
	}
	if patched.JSON200 == nil {
		return fmt.Errorf("the router refused the config: %s %s", patched.Status(), patched.Body)
	}
	if err := uploadDocuments(ctx, client, config.Name, filepath.Join(dir, "knowledge")); err != nil {
		return err
	}
	fmt.Printf("visible    %v\n", visibleTools)
	fmt.Printf("skills     %v\n", skills)
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
