package harness

import (
	"strings"
)

// Directive kinds. A model that wants help writes one of these into its reply, and the
// scanner takes it back out before the reply reaches the voice.
const (
	kindAsk  = "ask"
	kindDrop = "drop"
)

// tagLimit bounds how much text may be held back waiting for an opening tag to finish.
// A model that writes a lone angle bracket and then talks for a paragraph must not
// silence the agent while the scanner waits for a "greater than" that is never coming.
const tagLimit = 256

// directive is a request the model made of the harness rather than of the caller.
type directive struct {
	// kind is ask or drop.
	kind string
	// skill names which skill the request is for.
	skill string
	// body is what was asked for. A drop has none.
	body string
}

// scanner splits a streaming reply into the part that is spoken and the directives that
// are not.
//
// It exists because a model emits a few characters at a time: a tag arrives in pieces,
// so the text before one cannot be spoken until enough of the next piece has arrived to
// know whether a tag is starting. Everything that cannot yet be a tag is released
// immediately, because the caller is listening to the gap.
type scanner struct {
	// pending is text held back because it might be the start of a tag, or the body of
	// one that has not closed.
	pending strings.Builder
	// open is the directive being read, when the scanner is inside one.
	open *directive
}

// Add takes a delta of the reply and returns the part of it that may be spoken, along
// with every directive that finished within it.
func (s *scanner) Add(text string) (string, []directive) {
	s.pending.WriteString(text)

	var speech strings.Builder
	var directives []directive

	for {
		buffered := s.pending.String()
		if buffered == "" {
			break
		}

		if s.open != nil {
			body, rest, closed := splitClosing(buffered, s.open.kind)
			if !closed {
				break
			}
			s.open.body = strings.TrimSpace(body)
			directives = append(directives, *s.open)
			s.open = nil
			s.reset(rest)
			continue
		}

		before, rest, found := strings.Cut(buffered, "<")
		speech.WriteString(before)
		if !found {
			s.reset("")
			break
		}

		tag := "<" + rest
		opened, remainder, state := openTag(tag)

		// Nothing can be decided until the rest of the tag arrives. Waiting for one that
		// never comes would silence the agent, so a bracket that has gone on too long to
		// be a tag stops being treated as one.
		if state == tagIncomplete && len(tag) <= tagLimit {
			s.reset(tag)
			return speech.String(), directives
		}

		switch state {
		case tagIncomplete, tagAbsent:
			// A bracket that cannot start a tag is just something the model said.
			speech.WriteString("<")
			s.reset(rest)
		case tagSelfClosed:
			directives = append(directives, opened)
			s.reset(remainder)
		case tagOpened:
			s.open = &opened
			s.reset(remainder)
		}
	}

	return speech.String(), directives
}

// Flush releases what is left at the end of a reply.
//
// Text held back for a tag that never arrived was only ever text, so it is spoken. A
// directive that never closed is dropped: half a request is not a request, and its body
// was never meant to be heard.
func (s *scanner) Flush() string {
	defer s.Reset()
	if s.open != nil {
		return ""
	}
	return s.pending.String()
}

// Reset forgets a reply that has ended, whether it finished or was interrupted.
func (s *scanner) Reset() {
	s.pending.Reset()
	s.open = nil
}

func (s *scanner) reset(text string) {
	s.pending.Reset()
	s.pending.WriteString(text)
}

// tagState is what could be made of the text starting at an angle bracket.
type tagState int

const (
	// tagIncomplete means it may yet become a tag, once more text arrives.
	tagIncomplete tagState = iota
	// tagAbsent means it cannot become one.
	tagAbsent
	// tagOpened means a tag was read and its body follows.
	tagOpened
	// tagSelfClosed means a tag was read that has no body.
	tagSelfClosed
)

// openTag reads an opening tag from text that begins with an angle bracket, returning the
// directive, the text after the tag, and what could be made of it.
func openTag(text string) (directive, string, tagState) {
	kind, ok := openingKind(text)
	if !ok {
		if couldOpen(text) {
			return directive{}, "", tagIncomplete
		}
		return directive{}, "", tagAbsent
	}

	head, rest, closed := strings.Cut(text, ">")
	if !closed {
		return directive{}, "", tagIncomplete
	}

	found := directive{kind: kind, skill: attribute(head, "skill")}
	if strings.HasSuffix(head, "/") {
		return found, rest, tagSelfClosed
	}
	return found, rest, tagOpened
}

// openingKind reports which directive an opening tag starts, if any. The name must be
// followed by something that could not be part of a longer word, so a model saying
// "<asking>" is not read as a request for help.
func openingKind(text string) (string, bool) {
	for _, kind := range []string{kindAsk, kindDrop} {
		prefix := "<" + kind
		if !strings.HasPrefix(text, prefix) {
			continue
		}
		if rest := text[len(prefix):]; rest == "" || rest[0] == ' ' || rest[0] == '>' || rest[0] == '/' {
			return kind, true
		}
	}
	return "", false
}

// couldOpen reports whether more text might still turn this into an opening tag.
func couldOpen(text string) bool {
	for _, kind := range []string{kindAsk, kindDrop} {
		if strings.HasPrefix("<"+kind, text) {
			return true
		}
	}
	return false
}

// splitClosing separates a directive's body from what follows its closing tag.
func splitClosing(text, kind string) (string, string, bool) {
	body, rest, found := strings.Cut(text, "</"+kind+">")
	return body, rest, found
}

// attribute reads a quoted attribute out of an opening tag.
func attribute(tag, name string) string {
	_, rest, found := strings.Cut(tag, name+"=\"")
	if !found {
		return ""
	}
	value, _, closed := strings.Cut(rest, "\"")
	if !closed {
		return ""
	}
	return value
}
