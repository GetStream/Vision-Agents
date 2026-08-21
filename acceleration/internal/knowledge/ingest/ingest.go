// Package ingest cuts documents into the passages a knowledge base is filled with.
//
// It is the writing half of internal/knowledge, which only ever reads. Both the command
// that reads a docs tree off disk and the endpoint an SDK posts one to go through here, so
// the same file is cut the same way whichever way it arrives.
package ingest

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
)

// DefaultChunk is how much of a document goes in one passage. It is small enough that
// several can be put in front of a model without crowding out the conversation, and large
// enough that a passage still answers the question on its own.
const DefaultChunk = 1200

// IDSeparator divides the file a passage came from from its place in that file. Keeping
// the path in the id is what makes ingesting the same documentation twice an update
// rather than a second copy of it.
const IDSeparator = "#"

// readable are the files worth putting in front of a model. Anything else in a docs tree
// is an image, a lockfile or a build artefact, none of which answer a question.
var readable = map[string]struct{}{
	".md":   {},
	".mdx":  {},
	".txt":  {},
	".rst":  {},
	".yaml": {},
	".yml":  {},
}

// Read turns the named files and directories into passages, ready to be written.
func Read(paths []string, size int) ([]knowledge.Document, error) {
	var documents []knowledge.Document
	for _, path := range paths {
		info, err := os.Stat(path)
		if err != nil {
			return nil, err
		}
		if !info.IsDir() {
			found, err := readFile(path, filepath.Base(path), size)
			if err != nil {
				return nil, err
			}
			documents = append(documents, found...)
			continue
		}

		err = filepath.WalkDir(path, func(name string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			// A dot directory is version control, tooling or a build, none of which is
			// documentation somebody wrote to be read.
			if entry.IsDir() {
				if name != path && strings.HasPrefix(entry.Name(), ".") {
					return fs.SkipDir
				}
				return nil
			}
			if _, ok := readable[strings.ToLower(filepath.Ext(name))]; !ok {
				return nil
			}

			relative, err := filepath.Rel(path, name)
			if err != nil {
				relative = filepath.Base(name)
			}
			found, err := readFile(name, filepath.ToSlash(relative), size)
			if err != nil {
				return err
			}
			documents = append(documents, found...)
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return documents, nil
}

// readFile cuts one file into passages. The name is what a passage says it came from, so
// it is the path as a reader would recognise it rather than wherever the command was run.
func readFile(path, name string, size int) ([]knowledge.Document, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Split(name, string(content), size), nil
}

// Split cuts one document into the passages it will be stored as.
//
// The name is what a passage says it came from and what its id is keyed by, so it is the
// path as a reader would recognise it rather than wherever the file happened to be.
func Split(name, content string, size int) []knowledge.Document {
	pieces := split(name, content, size)
	documents := make([]knowledge.Document, 0, len(pieces))
	for index, piece := range pieces {
		documents = append(documents, knowledge.Document{
			ID:     fmt.Sprintf("%s%s%d", name, IDSeparator, index),
			Text:   piece.text,
			Source: piece.source,
		})
	}
	return documents
}

// Files is how many documents a set of passages came from.
func Files(documents []knowledge.Document) int {
	seen := map[string]struct{}{}
	for _, document := range documents {
		file, _, _ := strings.Cut(document.ID, IDSeparator)
		seen[file] = struct{}{}
	}
	return len(seen)
}

// passage is one piece of a document: what it says, and where a reader would find it.
type passage struct {
	source string
	text   string
}

// split cuts a document at its headings, and cuts long sections again at paragraph
// boundaries so no passage is longer than a model should be handed at once.
//
// The heading is repeated at the top of every piece of a section it was cut from. Without
// it the second half of a long section is prose with nothing saying what it is about,
// which is exactly what full-text search needs to match on.
func split(name, content string, size int) []passage {
	var passages []passage
	var heading string
	var current strings.Builder

	flush := func() {
		text := strings.TrimSpace(current.String())
		current.Reset()
		if !prose(text) {
			return
		}
		source := name
		if heading != "" {
			source = name + " > " + heading
		}
		passages = append(passages, passage{source: source, text: text})
	}

	for _, block := range blocks(content) {
		if title, ok := headingOf(block); ok {
			flush()
			heading = title
			current.WriteString(block + "\n\n")
			continue
		}
		for _, piece := range rows(block, size) {
			if current.Len() > 0 && current.Len()+len(piece) > size {
				flush()
				if heading != "" {
					current.WriteString("## " + heading + "\n\n")
				}
			}
			current.WriteString(piece + "\n\n")
		}
	}
	flush()
	return passages
}

// rows cuts a paragraph too long to be a passage on its own at its line breaks. A table
// or a long list has no blank line to cut at, so without this a section written as one
// runs back whole however long it is. Fenced code is left alone, because half a code
// block answers nothing.
func rows(block string, size int) []string {
	if len(block) <= size || strings.HasPrefix(strings.TrimSpace(block), "```") {
		return []string{block}
	}

	var pieces []string
	var current []string
	length := 0
	for _, line := range strings.Split(block, "\n") {
		if length > 0 && length+len(line) > size {
			pieces = append(pieces, strings.Join(current, "\n"))
			current, length = nil, 0
		}
		current = append(current, line)
		length += len(line) + 1
	}
	if len(current) > 0 {
		pieces = append(pieces, strings.Join(current, "\n"))
	}
	return pieces
}

// blocks cuts a document into paragraphs. Fenced code is kept whole: a blank line inside
// a code block is part of the code rather than a break in the prose.
func blocks(content string) []string {
	var found []string
	var current []string
	fenced := false

	flush := func() {
		if text := strings.TrimSpace(strings.Join(current, "\n")); text != "" {
			found = append(found, text)
		}
		current = nil
	}

	for _, line := range strings.Split(content, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "```") {
			fenced = !fenced
			current = append(current, line)
			continue
		}
		if !fenced && strings.TrimSpace(line) == "" {
			flush()
			continue
		}
		current = append(current, line)
	}
	flush()
	return found
}

// prose reports whether a passage says anything beyond the headings it opens with. A
// heading with nothing under it is a section that is only a name, and handing one back as
// an answer gives the model a title where it needed something to read.
func prose(text string) bool {
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" && !strings.HasPrefix(trimmed, "#") {
			return true
		}
	}
	return false
}

// headingOf reports the title a block opens with, when it opens with one.
func headingOf(block string) (string, bool) {
	first, _, _ := strings.Cut(block, "\n")
	trimmed := strings.TrimSpace(first)
	if !strings.HasPrefix(trimmed, "#") {
		return "", false
	}
	return strings.TrimSpace(strings.TrimLeft(trimmed, "#")), true
}
