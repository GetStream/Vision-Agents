//go:build integration

package searchsuite

import "github.com/GetStream/Vision-Agents/acceleration/internal/search"

// The fixtures are what the tests ask. Each has had the same answer for a long time and
// is written about everywhere, so a test that fails is the provider failing rather than
// the web having changed its mind. Tests read them and never change them.

// austen is a question the whole web agrees on.
var austen = search.Query{Text: "Who wrote the novel Pride and Prejudice?"}

// austenAnswer is the word any honest answer to austen contains.
const austenAnswer = "austen"

// wikipedia is a domain that has an article on austen, so narrowing to it or away from it
// still leaves something to find.
const wikipedia = "wikipedia.org"

// examplePage is a page that has said the same thing for decades, and examplePageSays is
// what it says.
const (
	examplePage     = "https://example.com"
	examplePageSays = "example domain"
)
