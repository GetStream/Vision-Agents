package openaicompat

import "strings"

// Gemma 4 writes its thinking into the answer as a channel, `<|channel>thought\n…<channel|>`,
// and still writes an empty one with thinking off, mostly just before it calls a tool.
// vLLM's reasoning parser takes it out at the start of a reply but not after the model has
// already said something, and when it does take the tags it can leave the bare channel name
// behind. Either way a voice agent reads "thought" to the caller.
const (
	channelOpen  = "<|channel>"
	channelClose = "<channel|>"
	channelName  = "thought\n"
)

// channelSplitter moves Gemma's thought channel out of the answer as it streams. A marker
// can arrive split over several deltas, so the end of what it has seen is held back while
// it could still be the start of one.
type channelSplitter struct {
	pending   string
	inChannel bool
	// started is set once the reply is past where a bare channel name could appear.
	started bool
}

// Write takes the next piece of the answer and returns what of it is answer and what is
// thinking.
func (c *channelSplitter) Write(delta string) (text, thought string) {
	c.pending += delta
	var answer, thinking strings.Builder
	for {
		if !c.started {
			if strings.HasPrefix(channelName, c.pending) {
				return answer.String(), thinking.String()
			}
			c.pending = strings.TrimPrefix(c.pending, channelName)
			c.started = true
		}
		if c.inChannel {
			end := strings.Index(c.pending, channelClose)
			if end < 0 {
				keep := partialSuffix(c.pending, channelClose)
				thinking.WriteString(c.pending[:len(c.pending)-keep])
				c.pending = c.pending[len(c.pending)-keep:]
				return answer.String(), thinking.String()
			}
			thinking.WriteString(c.pending[:end])
			c.pending = c.pending[end+len(channelClose):]
			c.inChannel = false
			continue
		}
		start := strings.Index(c.pending, channelOpen)
		if start < 0 {
			keep := partialSuffix(c.pending, channelOpen)
			answer.WriteString(c.pending[:len(c.pending)-keep])
			c.pending = c.pending[len(c.pending)-keep:]
			return answer.String(), thinking.String()
		}
		answer.WriteString(c.pending[:start])
		c.pending = strings.TrimPrefix(c.pending[start+len(channelOpen):], channelName)
		c.inChannel = true
	}
}

// Flush returns what was held back once the reply has ended: a marker that never finished
// is ordinary text, and a channel that never closed is still thinking.
func (c *channelSplitter) Flush() (text, thought string) {
	rest := c.pending
	c.pending = ""
	if !c.started {
		rest = strings.TrimPrefix(rest, channelName)
	}
	if c.inChannel {
		return "", rest
	}
	return rest, ""
}

// partialSuffix is the length of the longest end of s that is a proper start of marker.
func partialSuffix(s, marker string) int {
	for n := min(len(s), len(marker)-1); n > 0; n-- {
		if strings.HasSuffix(s, marker[:n]) {
			return n
		}
	}
	return 0
}
