package openaicompat

import (
	"strings"
	"unicode"
)

const (
	channelThoughtOpen = "<|channel>thought"
	channelClose       = "<channel|>"
	thinkOpen          = "<think>"
	thinkClose         = "</think>"
)

// thoughtStripper takes reasoning markers out of streamed content so they are never spoken.
//
// Gemma 4 opens every reply with an empty thought channel when thinking is off. The
// special tokens are often skipped by the decoder, which leaves the word "thought" in
// the answer. The same stripper also catches the tagged form and <think> blocks.
type thoughtStripper struct {
	started   bool
	inThought bool
	closeTag  string
	buf       string
}

func (s *thoughtStripper) Add(delta string) (speech, thinking string) {
	if delta == "" {
		return "", ""
	}
	s.buf += delta
	return s.drain(false)
}

func (s *thoughtStripper) Flush() (speech, thinking string) {
	return s.drain(true)
}

func (s *thoughtStripper) drain(flush bool) (speech, thinking string) {
	var spoken, thought strings.Builder
	for s.buf != "" {
		if s.inThought {
			if i := strings.Index(s.buf, s.closeTag); i >= 0 {
				thought.WriteString(s.buf[:i])
				s.buf = s.buf[i+len(s.closeTag):]
				s.inThought = false
				s.closeTag = ""
				s.started = true
				continue
			}
			if flush {
				thought.WriteString(s.buf)
				s.buf = ""
			}
			break
		}

		if !s.started {
			stripped, rest, known := splitLeadingThought(s.buf, flush)
			if stripped != "" {
				thought.WriteString(stripped)
				s.buf = rest
				s.started = true
				continue
			}
			if !known {
				break
			}
		}

		open, closeTag, at := nextThoughtOpen(s.buf)
		if at >= 0 {
			spoken.WriteString(s.buf[:at])
			s.buf = s.buf[at+len(open):]
			if strings.HasPrefix(s.buf, "\n") {
				s.buf = s.buf[1:]
			}
			s.inThought = true
			s.closeTag = closeTag
			s.started = true
			continue
		}
		if !flush && hasIncompleteTag(s.buf) {
			keep := strings.LastIndex(s.buf, "<")
			if keep > 0 {
				spoken.WriteString(s.buf[:keep])
				s.buf = s.buf[keep:]
				s.started = true
			}
			break
		}
		spoken.WriteString(s.buf)
		s.buf = ""
		s.started = true
	}
	return spoken.String(), thought.String()
}

func nextThoughtOpen(text string) (open, closeTag string, at int) {
	channelAt := strings.Index(text, channelThoughtOpen)
	thinkAt := strings.Index(text, thinkOpen)
	switch {
	case channelAt >= 0 && (thinkAt < 0 || channelAt <= thinkAt):
		return channelThoughtOpen, channelClose, channelAt
	case thinkAt >= 0:
		return thinkOpen, thinkClose, thinkAt
	default:
		return "", "", -1
	}
}

// splitLeadingThought strips a decoder-leaked "thought" token at the start of a reply.
// known is false when more bytes are needed to decide.
func splitLeadingThought(text string, flush bool) (stripped, rest string, known bool) {
	i := 0
	for i < len(text) && unicode.IsSpace(rune(text[i])) {
		i++
	}
	if i == len(text) {
		return "", text, flush
	}
	lower := strings.ToLower(text[i:])
	if strings.HasPrefix("thought", lower) && len(lower) < len("thought") {
		return "", text, flush
	}
	if !strings.HasPrefix(lower, "thought") {
		return "", text, true
	}
	after := text[i+len("thought"):]
	if after == "" {
		if flush {
			return text, "", true
		}
		return "", text, false
	}
	switch after[0] {
	case '\n':
		return text[:i+len("thought")+1], after[1:], true
	case '.', '!', ':':
		if len(after) == 1 {
			if flush {
				return text, "", true
			}
			return "", text, false
		}
		if after[1] == '\n' || after[1] == ' ' || after[1] == '\t' {
			return text[:i+len("thought")+2], after[2:], true
		}
		return "", text, true
	default:
		return "", text, true
	}
}

func hasIncompleteTag(text string) bool {
	at := strings.LastIndex(text, "<")
	if at < 0 {
		return false
	}
	tail := text[at:]
	if strings.Contains(tail, ">") {
		return false
	}
	for _, tag := range []string{channelThoughtOpen, thinkOpen, channelClose} {
		if strings.HasPrefix(tag, tail) {
			return true
		}
	}
	return false
}
