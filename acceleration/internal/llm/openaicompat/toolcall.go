package openaicompat

import (
	"encoding/json"
	"strconv"
	"strings"
	"unicode"
)

// Gemma 4 writes a tool call as `<|tool_call>call:NAME{key:value,...}<tool_call|>`, with strings
// between `<|"|>` marks. vLLM's tool parser turns that into a real call, but not when the
// opening marker goes missing: in Voicebench it went with an empty thought channel, and the
// rest, `call:check_availability{party_size:4,...}<tool_call|>`, came back as text and was read
// to the caller. A call written as text is still a call, so it is made, not spoken.
//
// Without the marker it does not always keep its own syntax either. Voicebench heard
// `call_tool: check_outage{}`, `call tool: create_ticket{address: "14 Cedar Lane"}` and
// `call:check_availability(party_size=4, time="7:30")` read out, so those are calls too.
const (
	toolCallOpen  = "<|tool_call>"
	toolCallClose = "<tool_call|>"
	gemmaQuote    = `<|"|>`
)

// toolCallPrefixes are what a call written as text starts with, ahead of the tool's name.
var toolCallPrefixes = []string{"call:", "call_tool:", "call tool:"}

// toolCallSplitter moves tool calls written as text out of the answer as it streams. Like the
// channel splitter, it holds back the end of what it has seen while that could still be the
// start of a marker or of a call.
type toolCallSplitter struct {
	// tools are the names offered on this request. A bare `call:` starts a call only when the
	// name after it is one of them, so "call: me back" is still spoken.
	tools   map[string]bool
	pending string
	// opened is set after `<|tool_call>`, where a call is expected whatever its name.
	opened bool
	// inCall is set while a call's text is being read, body holding it from its name on.
	inCall bool
	body   strings.Builder
	depth  int
	// quote is the mark that opened the string being read, `<|"|>` or `"`, or empty outside one.
	quote string
	// closing is set after a call, while its closing marker may still follow.
	closing bool
}

func newToolCallSplitter(tools []string) *toolCallSplitter {
	names := make(map[string]bool, len(tools))
	for _, tool := range tools {
		names[tool] = true
	}
	return &toolCallSplitter{tools: names}
}

// Write takes the next piece of the answer and returns what of it is answer, and the text of
// each call it finished, from the tool's name to its closing brace.
func (c *toolCallSplitter) Write(delta string) (text string, calls []string) {
	c.pending += delta
	var answer strings.Builder
	for {
		switch {
		case c.closing:
			if strings.HasPrefix(c.pending, toolCallClose) {
				c.pending = c.pending[len(toolCallClose):]
				c.closing = false
				continue
			}
			if strings.HasPrefix(toolCallClose, c.pending) {
				return answer.String(), calls
			}
			c.closing = false
		case c.inCall:
			if !c.readCall() {
				return answer.String(), calls
			}
			calls = append(calls, c.body.String())
			c.body.Reset()
			c.inCall, c.opened, c.closing = false, false, true
		default:
			start, skip, partial := c.nextCall()
			if start < 0 {
				keep := len(c.pending) - partial
				answer.WriteString(c.pending[:keep])
				c.pending = c.pending[keep:]
				return answer.String(), calls
			}
			answer.WriteString(c.pending[:start])
			c.pending = c.pending[start+skip:]
			if skip == len(toolCallOpen) {
				c.opened = true
				continue
			}
			c.inCall, c.depth, c.quote = true, 0, ""
		}
	}
}

// Flush returns what was held back once the reply has ended: an opening that never became a
// call is text, and a call that never closed is returned unfinished, for the caller to drop.
func (c *toolCallSplitter) Flush() (text, unfinished string) {
	rest := c.pending
	c.pending = ""
	if c.inCall {
		unfinished = c.body.String() + rest
		c.body.Reset()
		c.inCall = false
		return "", unfinished
	}
	if c.closing && strings.HasPrefix(toolCallClose, rest) {
		return "", ""
	}
	return rest, ""
}

// nextCall finds where the next call starts in pending: the index, how much to skip there (the
// opening marker, or a prefix and the spaces after it), and otherwise how much of the end to
// hold back because it could still become one.
func (c *toolCallSplitter) nextCall() (start, skip, partial int) {
	open := strings.Index(c.pending, toolCallOpen)
	for from := 0; ; {
		at, prefix := nextPrefix(c.pending, from)
		if at < 0 {
			break
		}
		if open >= 0 && open < at {
			break
		}
		rest := c.pending[at+prefix:]
		name := strings.TrimLeft(rest, " ")
		switch c.callName(name) {
		case nameFound:
			return at, prefix + len(rest) - len(name), 0
		case nameMaybe:
			if open >= 0 {
				return open, len(toolCallOpen), 0
			}
			return -1, 0, len(c.pending) - at
		}
		from = at + prefix
	}
	if open >= 0 {
		return open, len(toolCallOpen), 0
	}
	partial = partialSuffix(c.pending, toolCallOpen)
	for _, prefix := range toolCallPrefixes {
		partial = max(partial, partialSuffix(c.pending, prefix))
	}
	return -1, 0, partial
}

// nextPrefix finds the first call prefix in s at or after from, and its length.
func nextPrefix(s string, from int) (at, length int) {
	at = -1
	for _, prefix := range toolCallPrefixes {
		i := strings.Index(s[from:], prefix)
		if i >= 0 && (at < 0 || from+i < at) {
			at, length = from+i, len(prefix)
		}
	}
	return at, length
}

type nameMatch int

const (
	nameNone nameMatch = iota
	nameMaybe
	nameFound
)

// callName says whether rest starts with a callable name and its opening brace or
// parenthesis, could still grow into one, or cannot.
func (c *toolCallSplitter) callName(rest string) nameMatch {
	brace := strings.IndexAny(rest, "{(")
	if brace < 0 {
		if !isName(rest) {
			return nameNone
		}
		if c.opened {
			return nameMaybe
		}
		for tool := range c.tools {
			if strings.HasPrefix(tool, rest) {
				return nameMaybe
			}
		}
		return nameNone
	}
	name := rest[:brace]
	if name != "" && isName(name) && (c.opened || c.tools[name]) {
		return nameFound
	}
	return nameNone
}

// readCall moves the call's text from pending into body up to its closing brace, and says
// whether it got there. A string mark or an escape split over two deltas is held back.
func (c *toolCallSplitter) readCall() bool {
	i := 0
	for i < len(c.pending) {
		rest := c.pending[i:]
		if c.quote != `"` {
			if strings.HasPrefix(rest, gemmaQuote) {
				c.quote = toggle(c.quote, gemmaQuote)
				c.body.WriteString(gemmaQuote)
				i += len(gemmaQuote)
				continue
			}
			if strings.HasPrefix(gemmaQuote, rest) {
				break
			}
		}
		ch := c.pending[i]
		if c.quote == `"` && ch == '\\' {
			if i+1 == len(c.pending) {
				break
			}
			c.body.WriteString(c.pending[i : i+2])
			i += 2
			continue
		}
		c.body.WriteByte(ch)
		i++
		if ch == '"' && c.quote != gemmaQuote {
			c.quote = toggle(c.quote, `"`)
			continue
		}
		if c.quote != "" {
			continue
		}
		switch ch {
		case '{', '[', '(':
			c.depth++
		case '}', ']', ')':
			c.depth--
			if c.depth == 0 {
				c.pending = c.pending[i:]
				return true
			}
		}
	}
	c.pending = c.pending[i:]
	return false
}

// toggle is the quote after mark: mark opens a string outside one and closes it inside.
func toggle(quote, mark string) string {
	if quote == "" {
		return mark
	}
	return ""
}

func isName(s string) bool {
	for _, r := range s {
		if r != '_' && r != '-' && r != '.' && !unicode.IsLetter(r) && !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

// parseGemmaCall reads `NAME{key:value,...}`, or `NAME(key=value,...)`, into the tool's name
// and its arguments as JSON. It says false for text it cannot read whole.
func parseGemmaCall(text string) (name, arguments string, ok bool) {
	brace := strings.IndexAny(text, "{(")
	if brace <= 0 || !isName(text[:brace]) {
		return "", "", false
	}
	p := gemmaParser{text: text[brace:], ends: ",}]"}
	var value any
	if text[brace] == '(' {
		p.pos++
		p.ends += ")"
		value, ok = p.fields(')', '=')
	} else {
		value, ok = p.value()
	}
	if !ok {
		return "", "", false
	}
	p.space()
	if p.pos != len(p.text) {
		return "", "", false
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return "", "", false
	}
	return text[:brace], string(raw), true
}

// gemmaParser reads the argument syntax Gemma 4 writes: bare keys, strings between `<|"|>`
// marks or double quotes, numbers, true, false and null, and nested objects and lists.
type gemmaParser struct {
	text string
	pos  int
	// ends are what end a bare value: a parenthesis as well, in `NAME(key=value)` arguments.
	ends string
}

func (p *gemmaParser) space() {
	for p.pos < len(p.text) && unicode.IsSpace(rune(p.text[p.pos])) {
		p.pos++
	}
}

func (p *gemmaParser) value() (any, bool) {
	p.space()
	if p.pos >= len(p.text) {
		return nil, false
	}
	switch {
	case strings.HasPrefix(p.text[p.pos:], gemmaQuote):
		return p.quoted()
	case p.text[p.pos] == '"':
		return p.jsonString()
	case p.text[p.pos] == '{':
		p.pos++
		return p.fields('}', ':')
	case p.text[p.pos] == '[':
		return p.list()
	}
	start := p.pos
	for p.pos < len(p.text) && !strings.ContainsRune(p.ends, rune(p.text[p.pos])) {
		p.pos++
	}
	token := strings.TrimSpace(p.text[start:p.pos])
	switch token {
	case "":
		return nil, false
	case "true":
		return true, true
	case "false":
		return false, true
	case "null":
		return nil, true
	}
	if _, err := strconv.ParseFloat(token, 64); err == nil {
		return json.Number(token), true
	}
	return token, true
}

func (p *gemmaParser) quoted() (string, bool) {
	p.pos += len(gemmaQuote)
	end := strings.Index(p.text[p.pos:], gemmaQuote)
	if end < 0 {
		return "", false
	}
	value := p.text[p.pos : p.pos+end]
	p.pos += end + len(gemmaQuote)
	return value, true
}

// jsonString reads a string between double quotes, with JSON's escapes.
func (p *gemmaParser) jsonString() (string, bool) {
	end := p.pos + 1
	for end < len(p.text) && p.text[end] != '"' {
		if p.text[end] == '\\' {
			end++
		}
		end++
	}
	if end >= len(p.text) {
		return "", false
	}
	var value string
	if err := json.Unmarshal([]byte(p.text[p.pos:end+1]), &value); err != nil {
		return "", false
	}
	p.pos = end + 1
	return value, true
}

// fields reads keys and values up to closing, each key followed by assign, once the opening
// bracket has been read.
func (p *gemmaParser) fields(closing, assign byte) (any, bool) {
	out := map[string]any{}
	for {
		p.space()
		if p.pos < len(p.text) && p.text[p.pos] == closing {
			p.pos++
			return out, true
		}
		key, ok := p.key(assign)
		if !ok {
			return nil, false
		}
		value, ok := p.value()
		if !ok {
			return nil, false
		}
		out[key] = value
		if !p.separator(closing) {
			return nil, false
		}
	}
}

func (p *gemmaParser) list() (any, bool) {
	p.pos++
	out := []any{}
	for {
		p.space()
		if p.pos < len(p.text) && p.text[p.pos] == ']' {
			p.pos++
			return out, true
		}
		value, ok := p.value()
		if !ok {
			return nil, false
		}
		out = append(out, value)
		if !p.separator(']') {
			return nil, false
		}
	}
}

// key reads an object key and the assign mark after it.
func (p *gemmaParser) key(assign byte) (string, bool) {
	var quoted func() (string, bool)
	switch {
	case strings.HasPrefix(p.text[p.pos:], gemmaQuote):
		quoted = p.quoted
	case strings.HasPrefix(p.text[p.pos:], `"`):
		quoted = p.jsonString
	}
	if quoted != nil {
		value, ok := quoted()
		p.space()
		if !ok || p.pos >= len(p.text) || p.text[p.pos] != assign {
			return "", false
		}
		p.pos++
		return value, true
	}
	end := strings.IndexByte(p.text[p.pos:], assign)
	if end < 0 {
		return "", false
	}
	key := strings.TrimSpace(p.text[p.pos : p.pos+end])
	if key == "" || !isName(key) {
		return "", false
	}
	p.pos += end + 1
	return key, true
}

// separator reads the comma after a value, or the closing bracket, leaving the bracket for the
// caller's loop to take.
func (p *gemmaParser) separator(closing byte) bool {
	p.space()
	if p.pos >= len(p.text) {
		return false
	}
	switch p.text[p.pos] {
	case ',':
		p.pos++
		return true
	case closing:
		return true
	}
	return false
}
