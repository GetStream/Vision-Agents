package openaicompat

import (
	"strings"
	"unicode"
)

// Markers Gemma and Harmony-style models leak into content instead of the reasoning field.
const (
	channelThoughtOpen  = "<|channel>thought"
	channelThoughtOpen2 = "<|channel|>thought"
	channelThoughtClose = "<channel|>"
	channelSwitch       = "<|channel|>"
	thinkOpen           = "<think>"
	thinkClose          = "</think>"
)

// tagHold bounds how much text may be held back waiting for a thought marker to finish.
const tagHold = 256

type thoughtKind int

const (
	thoughtNone thoughtKind = iota
	thoughtChannel
	thoughtThink
)

// thoughtStripper takes thought-channel markup out of a streaming reply so it is never
// spoken. Models that think in-band write <|channel>thought…<channel|>, <think>…</think>,
// or a leading "thought" / "Thought." instead of using the reasoning field.
type thoughtStripper struct {
	pending strings.Builder
	inside  thoughtKind
	spoken  bool
}

// Add takes a content delta and returns the part that may be spoken.
func (s *thoughtStripper) Add(text string) string {
	if text == "" {
		return ""
	}
	s.pending.WriteString(text)
	return s.drain(false)
}

// Flush releases what is left at the end of a reply. Text held back for a marker that
// never arrived was only ever text. A thought block that never closed is dropped.
func (s *thoughtStripper) Flush() string {
	spoken := s.drain(true)
	s.pending.Reset()
	s.inside = thoughtNone
	s.spoken = false
	return spoken
}

func (s *thoughtStripper) drain(flush bool) string {
	var spoken strings.Builder
	for {
		buffered := s.pending.String()
		if buffered == "" {
			break
		}

		if s.inside != thoughtNone {
			closeAt, closeLen := s.findClose(buffered)
			if closeAt < 0 {
				if flush || len(buffered) > tagHold {
					s.pending.Reset()
				}
				break
			}
			s.inside = thoughtNone
			s.reset(buffered[closeAt+closeLen:])
			continue
		}

		if !s.spoken {
			rest, held, stripped := stripLeadingThought(buffered, flush)
			if stripped {
				s.reset(rest)
				continue
			}
			if held {
				break
			}
		}

		before, rest, kind, incomplete := splitThoughtOpen(buffered)
		if incomplete {
			// Hold the whole buffer, including text before a possible marker, until
			// the next delta completes it or Flush proves it was only ever text.
			if flush {
				spoken.WriteString(buffered)
				s.spoken = s.spoken || buffered != ""
				s.pending.Reset()
			}
			break
		}
		if before != "" {
			spoken.WriteString(before)
			s.spoken = true
		}
		if kind == thoughtNone {
			s.reset("")
			break
		}
		s.inside = kind
		s.reset(rest)
	}
	return spoken.String()
}

func (s *thoughtStripper) reset(text string) {
	s.pending.Reset()
	s.pending.WriteString(text)
}

func (s *thoughtStripper) findClose(text string) (int, int) {
	switch s.inside {
	case thoughtThink:
		if i := strings.Index(text, thinkClose); i >= 0 {
			return i, len(thinkClose)
		}
	case thoughtChannel:
		closeAt, closeLen := -1, 0
		if i := strings.Index(text, channelThoughtClose); i >= 0 {
			closeAt, closeLen = i, len(channelThoughtClose)
		}
		if i := strings.Index(text, channelSwitch); i >= 0 && (closeAt < 0 || i < closeAt) {
			closeAt, closeLen = i, len(channelSwitch)
		}
		return closeAt, closeLen
	}
	return -1, 0
}

func splitThoughtOpen(text string) (before, rest string, kind thoughtKind, incomplete bool) {
	earliest := -1
	found := thoughtNone
	opens := []struct {
		marker string
		kind   thoughtKind
	}{
		{channelThoughtOpen, thoughtChannel},
		{channelThoughtOpen2, thoughtChannel},
		{thinkOpen, thoughtThink},
	}
	for _, open := range opens {
		i := strings.Index(text, open.marker)
		if i >= 0 && (earliest < 0 || i < earliest) {
			earliest = i
			found = open.kind
		}
	}
	if earliest >= 0 {
		marker := thinkOpen
		if found == thoughtChannel {
			if strings.HasPrefix(text[earliest:], channelThoughtOpen2) {
				marker = channelThoughtOpen2
			} else {
				marker = channelThoughtOpen
			}
		}
		return text[:earliest], text[earliest+len(marker):], found, false
	}

	hold := longestOpenPrefix(text)
	if hold > 0 {
		return text[:len(text)-hold], text[len(text)-hold:], thoughtNone, true
	}
	return text, "", thoughtNone, false
}

func longestOpenPrefix(text string) int {
	markers := []string{channelThoughtOpen2, channelThoughtOpen, thinkOpen}
	hold := 0
	for n := 1; n <= len(text) && n < len(channelThoughtOpen2); n++ {
		suffix := text[len(text)-n:]
		for _, marker := range markers {
			if strings.HasPrefix(marker, suffix) {
				hold = n
				break
			}
		}
	}
	return hold
}

// stripLeadingThought removes a leaked "thought" / "Thought." / "Channel thought" at the
// start of a reply. held is true when more text is needed to know.
func stripLeadingThought(text string, flush bool) (rest string, held, stripped bool) {
	trim := 0
	for trim < len(text) {
		r, size := decodeLeadingSpace(text[trim:])
		if r == 0 {
			break
		}
		trim += size
	}
	body := text[trim:]
	if body == "" {
		if flush {
			return text, false, false
		}
		return text, true, false
	}

	lower := strings.ToLower(body)
	for _, prefix := range []string{"channel thought", "thought"} {
		if !strings.HasPrefix(lower, prefix) {
			continue
		}
		after := body[len(prefix):]
		if after == "" && !flush {
			return text, true, false
		}
		if after == "" || after[0] == '.' || unicode.IsSpace(rune(after[0])) {
			if strings.HasPrefix(after, ".") {
				after = after[1:]
			}
			return strings.TrimLeft(after, " \t\n\r"), false, true
		}
	}

	if !flush {
		for _, prefix := range []string{"channel thought", "thought"} {
			if strings.HasPrefix(strings.ToLower(prefix), strings.ToLower(body)) {
				return text, true, false
			}
		}
	}
	return text, false, false
}

func decodeLeadingSpace(text string) (rune, int) {
	if text == "" {
		return 0, 0
	}
	r := rune(text[0])
	if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
		return r, 1
	}
	return 0, 0
}

// looksLikeThinking reports whether text still looks like a thought channel that would
// be spoken, which is a leak the stripper missed.
func looksLikeThinking(text string) bool {
	t := strings.TrimSpace(text)
	if t == "" {
		return false
	}
	lower := strings.ToLower(t)
	if strings.Contains(lower, "<|channel>thought") ||
		strings.Contains(lower, "<|channel|>thought") ||
		strings.Contains(lower, "<channel|>") ||
		strings.Contains(lower, "<think>") ||
		strings.Contains(lower, "channel thought") {
		return true
	}
	if strings.HasPrefix(lower, "thought.") || lower == "thought" {
		return true
	}
	if strings.HasPrefix(lower, "thought ") || strings.HasPrefix(lower, "thought\n") {
		return true
	}
	return false
}
