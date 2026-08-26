package tts

import "strings"

// directionLimit bounds how much text may be held back waiting for a direction to close.
// A model that writes a lone opening bracket and then talks for a paragraph must not
// leave the reader watching a blank screen while the stripper waits for a closing bracket
// that is never coming.
const directionLimit = 64

// Directions takes bracketed performance directions such as [laughs] back out of a
// streaming reply.
//
// A voice that can act them wants them, but a transcript never does, and a voice that
// cannot act them would read them out. Text arrives a few characters at a time, so a
// direction can be split across deltas: everything that cannot yet be one is released
// immediately, because whoever is reading is waiting for it.
type Directions struct {
	// pending is text held back because it might be a direction that has not closed.
	pending strings.Builder
	// written reports whether anything has been released yet in this reply.
	written bool
	// spaced reports whether the last character released ended a word.
	spaced bool
	// swallow drops the space that followed a direction lifted from between two words,
	// which would otherwise leave two where there was one. It outlives the delta the
	// direction closed in, because the space after it may not have arrived yet.
	swallow bool
}

// Add takes a delta of a reply and returns it without the directions that closed in it.
func (d *Directions) Add(text string) string {
	d.pending.WriteString(text)

	var kept strings.Builder
	for {
		buffered := d.pending.String()
		if buffered == "" {
			break
		}

		before, rest, found := strings.Cut(buffered, "[")
		d.release(&kept, before)
		if !found {
			d.reset("")
			break
		}

		direction := "[" + rest
		_, after, closed := strings.Cut(direction, "]")
		if !closed {
			// Nothing can be decided until the rest of it arrives. A bracket that has
			// gone on too long to be a direction is just something the model said.
			if len(direction) <= directionLimit {
				d.reset(direction)
				return kept.String()
			}
			d.release(&kept, "[")
			d.reset(rest)
			continue
		}

		if !d.written || d.spaced {
			d.swallow = true
		}
		d.reset(after)
	}

	return kept.String()
}

// Flush releases what is left at the end of a reply. Text held back for a direction that
// never closed was only ever text.
func (d *Directions) Flush() string {
	held := d.pending.String()
	d.Reset()
	return held
}

// Reset forgets a reply, whether it finished or was abandoned part-way through.
func (d *Directions) Reset() {
	d.pending.Reset()
	d.written = false
	d.spaced = false
	d.swallow = false
}

func (d *Directions) release(kept *strings.Builder, text string) {
	if d.swallow {
		text = strings.TrimLeft(text, " ")
	}
	if text == "" {
		return
	}
	d.swallow = false
	kept.WriteString(text)
	d.written = true
	d.spaced = strings.HasSuffix(text, " ")
}

func (d *Directions) reset(text string) {
	d.pending.Reset()
	d.pending.WriteString(text)
}
