package main

import (
	"context"
	"errors"
	"regexp"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// handbookFreshFor is how long the passages read from the router are used before they are read
// again, so a document synced or uploaded since shows up within it.
const handbookFreshFor = time.Minute

// citationLength is the most a citation may hold for the router to show it as a source.
const citationLength = 500

// handbookPassage is one passage of a handbook document, as the router cut it.
type handbookPassage struct {
	ID       string
	Document string
	Heading  string
	Text     string
}

// handbook is the restaurant's handbook as the agent's knowledge base on the router holds it:
// the documents in knowledge/, which configure syncs, and any file uploaded to it, such as a PDF,
// which the router reads into Markdown.
type handbook struct {
	mu   sync.Mutex
	read time.Time
	held []handbookPassage
}

// passages are the handbook's passages, read again once they are older than handbookFreshFor.
func (h *handbook) passages(ctx context.Context) ([]handbookPassage, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.held != nil && time.Since(h.read) < handbookFreshFor {
		return h.held, nil
	}
	read, err := readHandbook(ctx)
	if err != nil {
		if h.held != nil {
			return h.held, nil
		}
		return nil, err
	}
	h.held, h.read = read, time.Now()
	return read, nil
}

// readHandbook reads every passage of the agent's knowledge base that the router has read.
func readHandbook(ctx context.Context) ([]handbookPassage, error) {
	client, err := stream.Backend{}.Client()
	if err != nil {
		return nil, err
	}
	namespace := agentName
	listed, err := client.ListKnowledgeDocumentsWithResponse(ctx, &acceleration.ListKnowledgeDocumentsParams{Namespace: &namespace})
	if err != nil {
		return nil, err
	}
	if listed.JSON200 == nil {
		return nil, errors.New(refused("listing the handbook's documents", listed.StatusCode(), listed.Body))
	}
	passages := []handbookPassage{}
	for _, document := range *listed.JSON200 {
		if document.Passages == 0 {
			continue
		}
		cut, err := client.ListKnowledgeDocumentPassagesWithResponse(ctx, document.Id)
		if err != nil {
			return nil, err
		}
		if cut.JSON200 == nil {
			return nil, errors.New(refused("reading "+document.Source, cut.StatusCode(), cut.Body))
		}
		for _, passage := range *cut.JSON200 {
			passages = append(passages, handbookPassage{
				ID:       passage.Id,
				Document: document.Source,
				Heading:  headingOf(passage.Source),
				Text:     bodyOf(passage.Text),
			})
		}
	}
	return passages, nil
}

// headingOf is the last heading a passage sits under, from the path the router names it by.
func headingOf(source string) string {
	parts := strings.Split(source, ">")
	return strings.TrimSpace(strings.TrimLeft(strings.TrimSpace(parts[len(parts)-1]), "#"))
}

// bodyOf is a passage's text without the headings it repeats.
func bodyOf(text string) string {
	lines := []string{}
	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}
		lines = append(lines, line)
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

var notInID = regexp.MustCompile(`[^A-Za-z0-9_.-]+`)

// citation is the passage in the shape the router shows as a reply's source.
func (p handbookPassage) citation() map[string]string {
	document := strings.TrimSuffix(p.Document, ".md")
	text := p.Text
	if utf8.RuneCountInString(text) > citationLength {
		text = string([]rune(text)[:citationLength-1]) + "…"
	}
	return map[string]string{
		"id":       "handbook-" + strings.Trim(notInID.ReplaceAllString(p.ID, "-"), "-"),
		"title":    "Handbook: " + p.Heading,
		"url":      handbookURL + notInID.ReplaceAllString(document, "-") + "#" + notInID.ReplaceAllString(strings.ToLower(p.Heading), "-"),
		"citation": text,
	}
}
