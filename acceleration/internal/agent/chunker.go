package agent

import (
	"strings"
	"unicode"
)

// minChunkRunes stops an abbreviation or a stray initial from being sent on its own. "Dr."
// is not a sentence, and synthesising it alone would put a pause in the middle of a name.
const minChunkRunes = 12

// maxClauseRunes is how much a streaming voice will hold before releasing on a space, so
// the model writing slower than playback does not starve the utterance.
const maxClauseRunes = 48

// chunker turns a stream of model deltas into sentences, or clause-sized pieces when the
// voice can take deltas.
//
// A model emits text a few characters at a time, but a voice wants whole clauses: handing a
// provider two words at a time produces speech that pauses in the wrong places, and waiting
// for the whole reply throws away the streaming the rest of the design is for. A sentence is
// the unit that satisfies both for a voice that needs a final request per piece. A streaming
// voice can take commas, so one reply stays one utterance instead of draining between
// sentences.
type chunker struct {
	pending strings.Builder
	clauses bool
}

// Add takes a delta and returns whatever complete pieces it finished, in order. Usually
// that is nothing, and occasionally more than one.
func (c *chunker) Add(text string) []string {
	var chunks []string

	for _, r := range text {
		c.pending.WriteRune(r)

		if !c.shouldRelease(r) {
			continue
		}
		chunks = append(chunks, c.take())
	}
	return chunks
}

// shouldRelease reports whether the pending text is ready to send.
func (c *chunker) shouldRelease(r rune) bool {
	n := c.pendingRunes()
	if n < minChunkRunes {
		return false
	}
	if isSentenceEnd(r) {
		return true
	}
	if !c.clauses {
		return false
	}
	if isClauseEnd(r) {
		return true
	}
	return n >= maxClauseRunes && unicode.IsSpace(r)
}

// Flush returns whatever is left, for the end of a reply that did not end in punctuation.
func (c *chunker) Flush() string {
	if strings.TrimSpace(c.pending.String()) == "" {
		c.pending.Reset()
		return ""
	}
	return c.take()
}

// Reset throws away the text in hand, for a reply that was interrupted.
func (c *chunker) Reset() { c.pending.Reset() }

// take returns the pending text and clears it.
func (c *chunker) take() string {
	chunk := strings.TrimSpace(c.pending.String())
	c.pending.Reset()
	return chunk
}

// pendingRunes counts characters rather than bytes, so a multi-byte language is not treated
// as though it had written more than it has.
func (c *chunker) pendingRunes() int {
	return len([]rune(c.pending.String()))
}

// isSentenceEnd reports whether a rune closes a sentence. The non-ASCII marks are included
// because the models are multilingual and those languages do not use the ASCII ones.
func isSentenceEnd(r rune) bool {
	switch r {
	case '.', '!', '?', '\n', '。', '！', '？', '…', '؟', '۔':
		return true
	}
	return false
}

// isClauseEnd reports whether a rune closes a clause worth sending to a streaming voice.
func isClauseEnd(r rune) bool {
	switch r {
	case ',', ';', ':', '—', '–':
		return true
	}
	return false
}
