package conversation

import (
	"time"
	"unicode/utf8"
)

// The whole of the model's thinking is shown to people watching a reply live, and only
// through ephemeral updates. A round of it can run to tens of kilobytes, so rather than
// repeat all of it on every update, each update carries a window of the streaming
// reasoning step: its thinking from where the last delivered window ended. Positions count
// Unicode scalars (as answer_start does), so a client never counts or splits what it
// already has.
const (
	// maxReasoningBuffer bounds the thinking kept to send. A backlog longer than this
	// (Stream unreachable for a while) skips ahead, and watchers see a gap.
	maxReasoningBuffer = 32 << 10
	// maxReasoningWindow bounds one window, leaving room within Stream's custom data
	// limit for the reply's own metadata. A longer backlog goes out over several ticks.
	maxReasoningWindow = 2000
	// reasoningKeyframe is how much recent thinking a window repeats now and then, so a
	// watcher who opened the conversation midway, or reconnected, catches up.
	reasoningKeyframe      = 1500
	reasoningKeyframeEvery = 3 * time.Second
	// reasoningEvery is how often thinking alone is worth an update. The answer keeps
	// the live tick's cadence, and any update it sends carries new thinking too.
	reasoningEvery = 200 * time.Millisecond
	// liveTick is how often live updates may go out: Stream's guidance for streamed
	// replies is at most every 50 to 100 ms.
	liveTick = 100 * time.Millisecond
)

// reasoningWindow is the "reasoning" field of an ephemeral update: text holds the
// thinking of reasoning step id from offset up to length. A client appends what it does
// not have yet.
type reasoningWindow struct {
	ID     string `json:"id"`
	Offset int    `json:"offset"`
	Text   string `json:"text"`
	Length int    `json:"length"`
	// key says the window repeats recent thinking for watchers who missed it.
	key bool
}

// liveReasoning is the thinking of the reasoning step being streamed. Only its opening
// (head), as the step's summary and preview, is ever stored.
type liveReasoning struct {
	id string
	// buf is the most recent thinking, starting at scalar position start.
	buf   string
	start int
	head  string
	// total is how long the whole thinking is, and sent how much of it was delivered.
	total       int
	sent        int
	first, last time.Time
	keyed       time.Time
}

// add records a piece of thinking.
func (r *liveReasoning) add(text string, now time.Time) {
	if text == "" {
		return
	}
	if r.total == 0 {
		r.first = now
	}
	if len(r.head) < maxHead {
		r.head += text[:boundaryBefore(text, min(len(text), maxHead-len(r.head)))]
	}
	r.buf += text
	r.total += utf8.RuneCountInString(text)
	r.last = now
	if len(r.buf) > maxReasoningBuffer {
		cut := boundary(r.buf, len(r.buf)-maxReasoningBuffer)
		r.start += utf8.RuneCountInString(r.buf[:cut])
		r.buf = r.buf[cut:]
	}
}

// pending reports thinking watchers have not been sent.
func (r *liveReasoning) pending() bool { return r.total > r.sent }

// window returns what the next update carries, if anything: the undelivered thinking,
// reaching back to repeat recent thinking when a keyframe is due.
func (r *liveReasoning) window(now time.Time) (reasoningWindow, bool) {
	key := now.Sub(r.keyed) >= reasoningKeyframeEvery
	if r.total == 0 || !r.pending() && !key {
		return reasoningWindow{}, false
	}
	from := max(r.sent, r.start)
	i := r.byteAt(from)
	if key {
		if k := boundary(r.buf, len(r.buf)-reasoningKeyframe); k < i {
			i, from = k, r.total-utf8.RuneCountInString(r.buf[k:])
		}
	}
	j := len(r.buf)
	if j-i > maxReasoningWindow {
		j = boundaryBefore(r.buf, i+maxReasoningWindow)
	}
	text := r.buf[i:j]
	return reasoningWindow{ID: r.id, Offset: from, Text: text, Length: from + utf8.RuneCountInString(text), key: key}, true
}

// delivered records a window Stream accepted.
func (r *liveReasoning) delivered(w reasoningWindow, now time.Time) {
	if w.ID != r.id {
		return
	}
	r.sent = max(r.sent, w.Length)
	if w.key {
		r.keyed = now
	}
}

// byteAt is where scalar position pos (start <= pos <= total) sits in buf. It walks
// back from the end, since what is sent next is nearly always recent.
func (r *liveReasoning) byteAt(pos int) int {
	i := len(r.buf)
	for n := r.total; n > pos && i > 0; n-- {
		_, size := utf8.DecodeLastRuneInString(r.buf[:i])
		i -= size
	}
	return i
}

// boundary is the first character boundary at or after byte i (0 when i is negative).
func boundary(s string, i int) int {
	if i <= 0 {
		return 0
	}
	for i < len(s) && !utf8.RuneStart(s[i]) {
		i++
	}
	return i
}

// boundaryBefore is the last character boundary at or before byte i.
func boundaryBefore(s string, i int) int {
	for i > 0 && i < len(s) && !utf8.RuneStart(s[i]) {
		i--
	}
	return i
}
