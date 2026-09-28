// Command configure stores the agent this directory describes on the router.
//
// This is the server-side half of the Swift demo, and it is deliberately a separate program.
// Writing an agent config, defining its skills and filling its knowledge base are all marked
// x-server-side-only in the OpenAPI spec, and the router refuses them from a phone. The app
// only ever names the agent this leaves behind.
//
// Two calls, with distinct jobs:
//
//   - DefineAgent says what runs the agent: which models transcribe, answer, speak and think.
//   - SyncAgent pushes what the agent knows: instructions.md, skills/ and knowledge/, read the
//     same way the Python SDK reads them, so the same directory can be pushed from either.
//
// Run it from the agent directory once, and again whenever that directory changes:
//
//	STREAM_ACCELERATION_URL=http://localhost:8080 \
//	STREAM_ACCELERATION_CUSTOMER_ID=examples \
//	go run ./configure
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/agents"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

const agentName = "swift_demo"

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
	// Resolved first, because the agent is named after the directory and the base of "."
	// is ".".
	path, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	folder, err := agents.Load(path)
	if err != nil {
		return err
	}
	if folder.Name != agentName {
		return fmt.Errorf("expected the %s directory, got %s", agentName, folder.Name)
	}

	client, err := stream.Backend{}.Client()
	if err != nil {
		return err
	}

	// The models first, because storing a config replaces it: doing this after the sync would
	// wipe the instructions and skills the sync had just written. A subagent is what makes a
	// skill mean anything -- without one the fast model answers everything itself.
	config, err := agents.DefineAgent(ctx, client, acceleration.AgentConfigRequest{
		Name: folder.Name,
		Stt:  text("deepgram/flux-general-en"),
		Tts:  text("cartesia/sonic-preview"),
		// The shortcut rather than one model by name. A named model is the one thing this
		// config cannot route around: it has broken this demo twice, once when the model was
		// retired from router.yaml and once when the Google project hit its spending cap, and
		// both times the agent answered nothing at all. `llm-fast` prefers the same model and
		// falls to the rest of the tier while it is unavailable.
		Llm:      text("llm-fast"),
		Subagent: text("openai/gpt-5.6-sol"),
		Greeting: text("Larkspur support, how can I help?"),
		Keyterms: &[]string{"Larkspur", "store credit", "refund"},
	})
	if err != nil {
		return err
	}
	fmt.Printf("agent      %s (config %s)\n", config.Name, config.Id)

	skills := make([]acceleration.SkillRequest, 0, len(folder.Skills))
	for _, skill := range folder.Skills {
		wanted := acceleration.SkillRequest{
			Name:         skill.Name,
			Description:  skill.Description,
			Instructions: skill.Instructions,
		}
		if skill.Deadline > 0 {
			deadline := skill.Deadline.Milliseconds()
			wanted.DeadlineMs = &deadline
		}
		skills = append(skills, wanted)
		fmt.Printf("skill      %s (%s)\n", skill.Name, skill.Deadline)
	}

	documents := make([]acceleration.KnowledgeDocument, 0, len(folder.Knowledge))
	for _, document := range folder.Knowledge {
		documents = append(documents, acceleration.KnowledgeDocument{
			Source: document.Source,
			Text:   document.Text,
		})
		fmt.Printf("knowledge  %s (%d characters)\n", document.Source, len(document.Text))
	}

	wanted := acceleration.SyncAgentRequest{
		Name:         folder.Name,
		Hash:         fingerprint(folder),
		Instructions: &folder.Instructions,
		Skills:       &skills,
		Knowledge:    &documents,
	}

	result, err := sync(ctx, client, wanted)
	if err != nil {
		return err
	}
	if result.Unchanged {
		fmt.Println("\nthe directory has not changed since the last sync")
	}
	fmt.Printf("\nopen       examples/agents/swift_demo/app/SwiftDemo.xcodeproj and run it\n")
	return nil
}

// sync pushes the directory, folding the knowledge base into the instructions if the router
// has nowhere to put it.
//
// Knowledge needs an embeddings provider, which a deployment can be run without -- and this
// demo is meant to run on a router somebody started five minutes ago. Dropping the policy
// left an agent that could not answer the one question the demo is about: the refund skill
// asked the caller what the returns policy said, and went round again on whatever they
// answered. So the documents are written into the prompt instead. It is not what a knowledge
// base is for -- a real one is looked up a passage at a time rather than read in full every
// turn, and it holds more than fits in a prompt -- but a shop's returns policy is four
// paragraphs, and an agent that knows them is the point.
func sync(
	ctx context.Context,
	client *acceleration.ClientWithResponses,
	wanted acceleration.SyncAgentRequest,
) (*acceleration.SyncAgentResult, error) {
	synced, err := client.SyncAgentWithResponse(ctx, wanted)
	if err != nil {
		return nil, err
	}
	if synced.JSON200 != nil {
		return synced.JSON200, nil
	}

	refused := complaint(synced)
	if wanted.Knowledge == nil || !strings.Contains(refused, "knowledge is not available") {
		return nil, fmt.Errorf("the router refused the sync: %s", refused)
	}

	fmt.Printf("\nknowledge  %s\n           so it goes in the prompts instead\n", refused)
	documents := *wanted.Knowledge
	inlined := told(*wanted.Instructions, documents)
	wanted.Instructions = &inlined
	// The skills as well, and not only the agent: a skill runs on the subagent under its own
	// instructions and the conversation so far, and the agent's prompt is not part of either.
	// Without this the refund skill asks the caller what the returns policy says.
	if wanted.Skills != nil {
		skills := append([]acceleration.SkillRequest(nil), *wanted.Skills...)
		for i := range skills {
			skills[i].Instructions = told(skills[i].Instructions, documents)
		}
		wanted.Skills = &skills
	}
	wanted.Knowledge = nil
	retried, err := client.SyncAgentWithResponse(ctx, wanted)
	if err != nil {
		return nil, err
	}
	if retried.JSON200 == nil {
		return nil, fmt.Errorf("the router refused the sync: %s", complaint(retried))
	}
	return retried.JSON200, nil
}

// told renders the knowledge base as part of a prompt, for the router that cannot hold one.
//
// It says the policy is here rather than somewhere to look it up, because both prompts say
// to look things up and the tool that does is not offered on a deployment with nowhere to
// look.
func told(prompt string, documents []acceleration.KnowledgeDocument) string {
	written := &strings.Builder{}
	written.WriteString(prompt)
	written.WriteString("\n\nWhat the shop has written down is below, rather than " +
		"somewhere to look it up. Answer out of it, and never guess past it.\n")
	for _, document := range documents {
		fmt.Fprintf(written, "\n## %s\n\n%s\n", document.Source, document.Text)
	}
	return written.String()
}

func complaint(response *acceleration.SyncAgentResponse) string {
	for _, failure := range []*acceleration.Error{response.JSON400, response.JSON401} {
		if failure != nil {
			return failure.Error
		}
	}
	return response.Status()
}

// fingerprint is what the router compares against the last sync to decide whether there is
// anything to do. It only ever compares it to one it stored, so any stable summary of the
// directory will do.
func fingerprint(folder *agents.Folder) string {
	hasher := sha256.New()
	fmt.Fprint(hasher, folder.Instructions)
	for _, skill := range folder.Skills {
		fmt.Fprintf(hasher, "\nskill:%s\n%s\n%s\n%s",
			skill.Name, skill.Description, skill.Instructions, skill.Deadline)
	}
	for _, document := range folder.Knowledge {
		fmt.Fprintf(hasher, "\nknowledge:%s\n%s", document.Source, document.Text)
	}
	return hex.EncodeToString(hasher.Sum(nil))
}

func text(value string) *string { return &value }
