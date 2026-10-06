package agent

import (
	"strings"
	"unicode"
)

// minChunkRunes stops an abbreviation or a stray initial from being sent on its own. "Dr."
// is not a sentence, and synthesising it alone would put a pause in the middle of a name.
const minChunkRunes = 12

// minFirstClauseRunes is how much of a reply's opening clause is enough to say on its own.
// The first thing a caller hears is what they wait for, so it goes to the voice at the first
// clause rather than the first sentence; below this a lead-in like "Sure," would be said
// alone and sound clipped.
const minFirstClauseRunes = 20

// chunker turns a stream of model deltas into sentences.
//
// A model emits text a few characters at a time, but a voice wants whole clauses: handing a
// provider two words at a time produces speech that pauses in the wrong places, and waiting
// for the whole reply throws away the streaming the rest of the design is for. A sentence is
// the unit that satisfies both.
type chunker struct {
	pending strings.Builder
	// started says a chunk of this reply has already gone to the voice. Only the first one
	// may end at a clause: after it the voice is busy speaking, and whole sentences sound
	// better than a reply broken at every comma.
	started bool
	// last is the rune before the one being read, so a comma between digits is not taken
	// for the end of a clause.
	last rune
}

// Add takes a delta and returns whatever complete sentences it finished, in order. Usually
// that is nothing, and occasionally more than one.
func (c *chunker) Add(text string) []string {
	var chunks []string

	for _, r := range text {
		c.pending.WriteRune(r)
		after := c.last
		c.last = r

		if !c.started && isClauseEnd(r) && unicode.IsLetter(after) && c.pendingRunes() >= minFirstClauseRunes {
			chunks = append(chunks, c.take())
			continue
		}
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
	defer c.Reset()
	if strings.TrimSpace(c.pending.String()) == "" {
		return ""
	}
	return strings.TrimSpace(c.pending.String())
}

// Reset throws away the text in hand, for a reply that was interrupted, and makes the next
// reply's opening clause eligible again.
func (c *chunker) Reset() {
	c.pending.Reset()
	c.started = false
	c.last = 0
}

// take returns the pending text and clears it.
func (c *chunker) take() string {
	chunk := strings.TrimSpace(c.pending.String())
	c.pending.Reset()
	c.started = true
	return chunk
}

// pendingRunes counts characters rather than bytes, so a multi-byte language is not treated
// as though it had written more than it has.
func (c *chunker) pendingRunes() int {
	return len([]rune(c.pending.String()))
}

// isClauseEnd reports whether a rune closes a clause that can be said on its own.
func isClauseEnd(r rune) bool {
	switch r {
	case ',', ';', ':', '，', '；', '：', '、':
		return true
	}
	return false
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
