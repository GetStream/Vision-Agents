// Package search defines what an agent may find out rather than know or look up.
//
// Knowledge is what the business wrote down and memory is what the agent learned about a
// person. Neither can answer what is true right now: traffic, weather, a score, a price
// that moved this morning. A caller asking about any of those is asking for something no
// model holds and no handbook records, and an agent without this can only apologise or,
// worse, answer from what it remembers being true.
//
// The contract is one method, for the same reason knowledge's is: how a provider crawls,
// ranks or summarises is its own business. What a caller needs is a question turned into
// something worth reading aloud.
package search

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// Query is a question to answer out of what is true now.
type Query struct {
	// Text is the question, in the caller's own words.
	Text string
	// Limit caps how many results come back. Zero leaves the provider's default.
	Limit int
}

// Validate reports whether the query names something to find out.
func (q Query) Validate() error {
	if strings.TrimSpace(q.Text) == "" {
		return errors.New("search: there is nothing to look for")
	}
	return nil
}

// Result is what a provider found.
type Result struct {
	// Answer is the provider's own summary of the results, where it offers one. It is what
	// a voice agent wants: a sentence to say rather than a page to read.
	Answer string
	// Documents are the sources behind it, most relevant first.
	Documents []Document
}

// Document is one source that bears on the question.
type Document struct {
	// Title is what the page called itself.
	Title string
	// URL is where it came from, so the agent can say where it read something.
	URL string
	// Text is the relevant extract, which is what the model reads.
	Text string
	// Score is how relevant the provider judged it.
	Score float64
}

// Provider is a search provider.
//
// It satisfies routing.Provider, so searches are routed, ranked and billed the way the
// three model modalities are. Start and Close are what that contract asks for rather than
// anything a search needs: an HTTP client has nothing to open, and a provider missing its
// key fails when it is built, which is where the router picks the next candidate.
type Provider interface {
	// Search answers the question out of what is true now.
	Search(ctx context.Context, query Query) (Result, error)
	Start(ctx context.Context) error
	Close() error
	// Provider is the stable provider name used in stats, e.g. "tavily".
	Provider() string
	// Model is which of a provider's modes is in use, e.g. "sonar-pro". Providers differ
	// more between their own modes than they do from each other, so this is what a
	// routing target actually chooses between.
	Model() string
}

// Page is one document fetched whole, as markdown.
type Page struct {
	URL   string
	Title string
	// Text is the page as markdown, with the navigation and the advertising taken out.
	Text string
}

// Reader turns a URL into something a knowledge base can hold.
//
// It is separate from Provider because not every search provider will fetch a page you
// name: searching is finding out which pages exist, and this is reading one you already
// decided on.
type Reader interface {
	Read(ctx context.Context, url string) (Page, error)
}

// Prompt renders what was found for the model to read.
//
// A question nothing covers comes back as words rather than an empty string, because the
// model is about to speak either way: told plainly that nothing was found, it says so,
// where an empty result reads as a tool that broke and gets apologised for.
func Prompt(found Result) string {
	answer := strings.TrimSpace(found.Answer)
	if answer == "" && len(found.Documents) == 0 {
		return "The search found nothing about this. Tell the caller you could not find out."
	}

	var built strings.Builder
	built.WriteString("Here is what the search found. It is current, so prefer it over " +
		"what you remember, and say the one thing that answers the question rather than " +
		"reading the sources out.\n")
	if answer != "" {
		built.WriteString("\nSummary: " + answer + "\n")
	}
	for i, document := range found.Documents {
		text := strings.TrimSpace(document.Text)
		if text == "" {
			continue
		}
		built.WriteString(fmt.Sprintf("\n[%d] %s\n%s\n", i+1, sourceOf(document), text))
	}
	return built.String()
}

// sourceOf is what to call a document when quoting it back.
func sourceOf(document Document) string {
	if title := strings.TrimSpace(document.Title); title != "" {
		return title
	}
	if url := strings.TrimSpace(document.URL); url != "" {
		return url
	}
	return "untitled source"
}
