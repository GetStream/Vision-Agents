package conversation

import (
	"cmp"
	"context"
	"slices"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// Restore is how a conversation is handed back to a model when a session opens on it. The
// zero value is how it always was: completed turns as text, the newest MaxHistoryMessages
// of them within MaxHistoryRunes.
type Restore struct {
	// MaxMessages and MaxRunes replace MaxHistoryMessages and MaxHistoryRunes when above
	// zero. MaxMessages counts turns, never the tool calls replayed with them.
	MaxMessages, MaxRunes int
	// Recorded, when set, hands each reply back with the tool calls it made and what they
	// returned. Over MaxRunes, the oldest results are cut to ResultTrimmed before any turn
	// is left out, and the conversation's first message is kept.
	Recorded Recorded
}

// Recorded is what a conversation's sessions recorded about the replies in it.
type Recorded interface {
	// Rounds are the tool calls the replies with these turn ids made, by turn id: an
	// assistant message asking for calls, then a result for each of them, oldest first. A
	// reply with nothing recorded, or with records since removed, is left out and handed
	// back as text.
	Rounds(ctx context.Context, cid string, turns []string) map[string][]llm.Message
	// Opening is the first thing said in the conversation, empty when nothing recorded it.
	Opening(ctx context.Context, cid string) string
}

// ResultTrimmed stands in for a tool result cut to keep a restored history within its size.
const ResultTrimmed = "[result trimmed]"

type restoreKey struct{}

// WithRestore has every history read under ctx handed back as r says.
func WithRestore(ctx context.Context, r Restore) context.Context {
	return context.WithValue(ctx, restoreKey{}, r)
}

// restored is the history of a page as ctx asks for it.
func restored(ctx context.Context, cid string, p Page) ([]llm.Message, bool) {
	r, _ := ctx.Value(restoreKey{}).(Restore)
	if r.Recorded == nil {
		return r.history(p, nil, "")
	}
	var turns []string
	for _, m := range p.Messages {
		if m.Role == "assistant" && m.TurnID != "" {
			turns = append(turns, m.TurnID)
		}
	}
	var rounds map[string][]llm.Message
	if len(turns) > 0 {
		rounds = r.Recorded.Rounds(ctx, cid, turns)
	}
	opening := ""
	// A page that reached the start of the conversation holds its first message already.
	if p.Truncated {
		opening = r.Recorded.Opening(ctx, cid)
	}
	return r.history(p, rounds, opening)
}

// openedIn reports whether the page holds the conversation's first message: all of the
// conversation was read, or what was recorded as said first is in it.
func openedIn(p Page, opening string) bool {
	return !p.Truncated || opening != "" && slices.ContainsFunc(p.Messages, func(m Message) bool {
		return m.Role == "user" && m.Text == opening
	})
}

func (r Restore) history(p Page, rounds map[string][]llm.Message, opening string) ([]llm.Message, bool) {
	limit := cmp.Or(r.MaxMessages, MaxHistoryMessages)
	budget := cmp.Or(r.MaxRunes, MaxHistoryRunes)
	size := 0
	if p.shared {
		size = utf8.RuneCountInString(sharedHistoryAttribution)
		limit--
	}
	all := entries(p)
	var out []llm.Message
	tr := p.Truncated
	if r.Recorded == nil {
		for _, e := range all {
			if e.message.Content == "" {
				continue
			}
			n := utf8.RuneCountInString(e.message.Content)
			if size+n > budget || len(out) == limit {
				tr = true
				break
			}
			size += n
			out = append(out, e.message)
		}
		slices.Reverse(out)
	} else {
		out, tr = withTools(p, all, rounds, opening, limit, budget-size, tr)
	}
	if p.shared && len(out) > 0 {
		out = append([]llm.Message{{Role: llm.System, Content: sharedHistoryAttribution}}, out...)
	}
	return out, tr
}

// told is one turn handed back with the tool rounds before it. A pinned turn is the
// conversation's first message, left out only when nothing else is left to cut.
type told struct {
	rounds  []llm.Message
	message llm.Message
	pinned  bool
}

func (t told) size() int {
	n := utf8.RuneCountInString(t.message.Content)
	for _, m := range t.rounds {
		n += utf8.RuneCountInString(m.Content)
		for _, call := range m.ToolCalls {
			n += utf8.RuneCountInString(call.Name) + utf8.RuneCountInString(call.Arguments)
		}
	}
	return n
}

// withTools is history (all, newest first) with each reply's tool rounds before it, as the
// model was given them while the session ran.
//
// A call and its result are kept or left out together, so every result answers a call
// before it, as Anthropic and OpenAI both require. The first message is pinned: it is
// usually the task, which a long agent run would otherwise lose first.
func withTools(p Page, all []entry, rounds map[string][]llm.Message, opening string, limit, budget int, tr bool) ([]llm.Message, bool) {
	// opened is where the conversation's first message is in all, when all reaches it.
	opened := -1
	if openedIn(p, opening) {
		opening = ""
		for k := len(all) - 1; k >= 0 && opened < 0; k-- {
			if all[k].message.Role == llm.User {
				opened = k
			}
		}
	}
	var picked []told // newest first
	var first *told   // the first message, when it is older than every turn picked
	for k, e := range all {
		calls := cloneRounds(rounds[e.turn])
		if e.message.Content == "" && len(calls) == 0 {
			continue
		}
		if len(picked) == limit {
			tr = true
			if opened >= k {
				first = &told{message: all[opened].message}
			}
			break
		}
		picked = append(picked, told{rounds: calls, message: e.message, pinned: k == opened})
	}
	if opening != "" {
		first = &told{message: entryOf(p, Message{Role: "user", Text: opening, State: "completed"}).message}
	}
	if first != nil && len(picked) >= limit {
		picked = picked[:max(limit-1, 0)]
		tr = true
	}

	size := 0
	for _, t := range picked {
		size += t.size()
	}
	if first != nil {
		size += first.size()
	}
	for size > budget {
		tr = true
		if cut := trimOldestResult(picked); cut > 0 {
			size -= cut
			continue
		}
		oldest := len(picked) - 1
		for oldest > 0 && picked[oldest].pinned {
			oldest--
		}
		if oldest >= 0 {
			size -= picked[oldest].size()
			picked = slices.Delete(picked, oldest, oldest+1)
			continue
		}
		if first == nil {
			break
		}
		size -= first.size()
		first = nil
	}

	var out []llm.Message
	if first != nil {
		out = append(out, first.message)
	}
	for i := len(picked) - 1; i >= 0; i-- {
		out = append(out, picked[i].rounds...)
		if picked[i].message.Content != "" {
			out = append(out, picked[i].message)
		}
	}
	// Gemini refuses a history that opens on a tool call or a result, and with the turn
	// before it left out the round answers nothing the model can see.
	for len(out) > 0 && (out[0].Role == llm.ToolResult || len(out[0].ToolCalls) > 0) {
		out = out[1:]
	}
	return out, tr
}

// trimOldestResult cuts the oldest tool result not yet cut, reporting how many runes that
// saved, or 0 when none is left to cut.
func trimOldestResult(picked []told) int {
	marker := utf8.RuneCountInString(ResultTrimmed)
	for i := len(picked) - 1; i >= 0; i-- {
		for j, m := range picked[i].rounds {
			if m.Role != llm.ToolResult || m.Content == ResultTrimmed {
				continue
			}
			if saved := utf8.RuneCountInString(m.Content) - marker; saved > 0 {
				picked[i].rounds[j].Content = ResultTrimmed
				return saved
			}
		}
	}
	return 0
}

func cloneRounds(rounds []llm.Message) []llm.Message {
	out := slices.Clone(rounds)
	for i := range out {
		out[i].ToolCalls = slices.Clone(out[i].ToolCalls)
	}
	return out
}
