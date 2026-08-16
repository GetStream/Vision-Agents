package agent

import (
	"strings"
)

// minChunkRunes stops an abbreviation or a stray initial from being sent on its own. "Dr."
// is not a sentence, and synthesising it alone would put a pause in the middle of a name.
const minChunkRunes = 12

// chunker turns a stream of model deltas into sentences.
//
// A model emits text a few characters at a time, but a voice wants whole clauses: handing a
// provider two words at a time produces speech that pauses in the wrong places, and waiting
// for the whole reply throws away the streaming the rest of the design is for. A sentence is
// the unit that satisfies both.
type chunker struct {
	pending strings.Builder
}

// Add takes a delta and returns whatever complete sentences it finished, in order. Usually
// that is nothing, and occasionally more than one.
func (c *chunker) Add(text string) []string {
	var chunks []string

	for _, r := range text {
		c.pending.WriteRune(r)

		if !isSentenceEnd(r) {
			continue
		}
		if c.pendingRunes() < minChunkRunes {
			continue
		}
		chunks = append(chunks, c.take())
	}
	return chunks
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
