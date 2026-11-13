package conversation

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strconv"
	"strings"

	getstream "github.com/GetStream/getstream-go/v5"
)

// A reply's generated UI: a fenced block of the reply's text in a format a client renders, which
// the reply carries as custom.ai_ui while its text keeps the words around it. The text is what
// people read wherever the UI cannot show, such as in a push notification, an external thread or
// a client without a renderer for the format, and the client that can show it shows it in place
// of the text. Only the agent writes UI: a block in a person's message is part of their words.
//
// The router knows no format. A block is UI when its info string is ui:<format>, optionally
// ui:<format>@<version>, or the fence a format's own prompts ask models for, such as OpenUI's
// openui-lang. The first such block is the reply's UI, and any later one is dropped from the text.
//
// A person's message can carry what they did in such a UI as custom.ai_ui_action, which the
// model is told with their words.
const (
	uiVersion = 1
	// maxMessageCustom is the most custom data a write may set on a message, as JSON. Stream
	// refuses a message whose custom data is over 50 KB, and a write keeps the fields it does
	// not set, such as client_id and ai_generated.
	maxMessageCustom = 48 << 10
	// MaxUIAction is the most a person's action may hold, written as JSON.
	MaxUIAction = 8 << 10
	// maxActedOnRunes is how much of a reply's words the model is shown to tell it which of
	// its replies a person's action was in.
	maxActedOnRunes = 200
)

// UIOverflowStore stores generated UI programs that exceed Stream Chat's 50 KB custom-field
// budget, returning an artifact attachment the message points to instead of dropping the UI.
type UIOverflowStore interface {
	StoreUI(ctx context.Context, customer, cid, messageID, format, content string) (ArtifactAttachment, error)
}

// uiFences are the fences formats ask models for, by the format they hold.
var uiFences = map[string]string{
	"openui-lang":  "openui",
	"openui-patch": "openui",
}

var openUIIdent = regexp.MustCompile(`^\$?[A-Za-z_][A-Za-z0-9_]*$`)

// uiFormatModeOf reads a fence's info string as a UI format, optional version, and mode
// ("replace" or "patch").
func uiFormatModeOf(info string) (format, version, mode string, ok bool) {
	info = strings.TrimSpace(info)
	if info == "openui-patch" {
		return "openui", "", "patch", true
	}
	if format, known := uiFences[info]; known {
		return format, "", "replace", true
	}
	named, found := strings.CutPrefix(info, "ui:")
	if !found || named == "" {
		return "", "", "", false
	}
	mode = "replace"
	if base, patched := strings.CutSuffix(named, "+patch"); patched {
		mode = "patch"
		named = base
	}
	format, version, _ = strings.Cut(named, "@")
	return format, version, mode, displayID.MatchString(format)
}

// uiFormatOf reads a fence's info string as a UI format, and its version when it names one.
func uiFormatOf(info string) (format, version string, ok bool) {
	format, version, _, ok = uiFormatModeOf(info)
	return format, version, ok
}

// splitUIWithBase separates a reply's text into the words people read and its UI, merging
// statement patches against baseContent when the fence requests patch mode.
func splitUIWithBase(text string, finished bool, baseFormat, baseContent string) (string, map[string]any) {
	var prose, content []string
	var ui map[string]any
	mode := "replace"
	// capturing is whether the block being read is the reply's UI rather than a later one.
	inUI, inOther, capturing, closed := false, false, false, false
	lines := strings.Split(text, "\n")
	for index, line := range lines {
		fence := strings.HasPrefix(strings.TrimLeft(line, " "), "```")
		last := index == len(lines)-1
		switch {
		case inUI:
			if fence && strings.TrimSpace(line) == "```" {
				closed = closed || capturing
				inUI, capturing = false, false
				continue
			}
			if capturing {
				content = append(content, line)
			}
		case inOther:
			prose = append(prose, line)
			if fence && strings.TrimSpace(line) == "```" {
				inOther = false
			}
		case fence:
			if last && !finished {
				// The info string may still be on its way.
				continue
			}
			format, version, parsedMode, ok := uiFormatModeOf(strings.TrimLeft(line, " ")[3:])
			if !ok {
				prose = append(prose, line)
				inOther = true
				continue
			}
			inUI = true
			if ui == nil {
				capturing = true
				mode = parsedMode
				ui = map[string]any{"v": uiVersion, "format": format}
				if version != "" {
					ui["version"] = version
				}
			}
		default:
			prose = append(prose, line)
		}
	}
	if ui == nil {
		return strings.Join(prose, "\n"), nil
	}
	body := strings.Join(content, "\n")
	format, _ := ui["format"].(string)
	if mode == "patch" && format == "openui" && baseFormat == "openui" && baseContent != "" {
		body = MergeOpenUIStatements(baseContent, body)
	}
	ui["content"] = body
	ui["status"] = "streaming"
	if closed || finished {
		ui["status"] = "complete"
	}
	return strings.TrimSpace(strings.Join(prose, "\n")), ui
}

// splitUI separates a reply's text into the words people read and its UI. finished says the
// reply is over: until then a block not closed yet is still being written, and a last line that
// may be opening one is held back rather than shown for a moment as text.
func splitUI(text string, finished bool) (string, map[string]any) {
	return splitUIWithBase(text, finished, "", "")
}

// wireUI is a message's text and UI as Stream keeps them. Only an agent's reply has UI.
func wireUI(m Message) (string, map[string]any) {
	return wireUIWithBase(m, "", "")
}

func wireUIWithBase(m Message, baseFormat, baseContent string) (string, map[string]any) {
	if m.Role != "assistant" {
		return m.Text, nil
	}
	text, ui := splitUIWithBase(m.Text, m.FinishedAt != nil, baseFormat, baseContent)
	if ui == nil && m.UI != nil {
		ui = maps.Clone(m.UI)
		if m.FinishedAt != nil {
			ui["status"] = "complete"
		}
		text = strings.TrimSpace(text)
	}
	return text, ui
}

// VoiceUIMap builds the custom.ai_ui object for a voice transcript reply.
func VoiceUIMap(format, version, status, content string) map[string]any {
	if format == "" || content == "" {
		return nil
	}
	ui := map[string]any{
		"v":       uiVersion,
		"format":  format,
		"status":  status,
		"content": content,
	}
	if version != "" {
		ui["version"] = version
	}
	return ui
}

// FitVoiceUI ensures a voice reply's partial or initial write never exceeds Stream Chat's
// custom-data limit, returning the fields to unset when ai_ui had to be omitted.
func FitVoiceUI(set map[string]any, own ...string) []string {
	if fitUI(set, own...) {
		return []string{"ai_ui"}
	}
	return nil
}

// fitUI leaves a reply's UI out of what a write sets when it would take the message's custom
// data over maxMessageCustom, measured as Stream receives it: as JSON, escaping included, beside
// everything else the write sets but own, the message's own fields. Stream refuses such a write,
// and a refused write stays at the front of the queue, holding back every write after it, so a
// UI too large to carry leaves the reply its words. It reports whether it left the UI out.
func fitUI(set map[string]any, own ...string) bool {
	if _, ok := set["ai_ui"]; !ok {
		return false
	}
	if fitsCustomBudget(set, own...) {
		return false
	}
	delete(set, "ai_ui")
	return true
}

func fitsCustomBudget(set map[string]any, own ...string) bool {
	custom := make(map[string]any, len(set))
	for key, value := range set {
		if !slices.Contains(own, key) {
			custom[key] = value
		}
	}
	encoded, err := json.Marshal(custom)
	return err == nil && len(encoded) <= maxMessageCustom
}

// uiUnset fits an update's UI (fitUI), and unsets the UI an earlier update set when this one
// leaves it out, so the reply does not keep a UI cut off midway.
func uiUnset(set map[string]any) []string {
	if fitUI(set, "text", "attachments") {
		return []string{"ai_ui"}
	}
	return nil
}

// spillOversizedUI moves an oversized settled ai_ui payload into overflow storage when
// configured, replacing custom.ai_ui with a compact artifact pointer that fits under 48 KiB.
func spillOversizedUI(
	ctx context.Context,
	store UIOverflowStore,
	customer, cid, messageID string,
	ui map[string]any,
	probe map[string]any,
	own ...string,
) (map[string]any, *ArtifactAttachment) {
	if store == nil || ui == nil {
		return ui, nil
	}
	probe["ai_ui"] = ui
	if fitsCustomBudget(probe, own...) {
		return ui, nil
	}
	format, _ := ui["format"].(string)
	content, _ := ui["content"].(string)
	status, _ := ui["status"].(string)
	if format == "" || content == "" {
		return ui, nil
	}
	stored, err := store.StoreUI(ctx, customer, cid, messageID, format, content)
	if err != nil || !validArtifact(stored) {
		return ui, nil
	}
	pointer := map[string]any{
		"v":           uiVersion,
		"format":      format,
		"status":      status,
		"overflow":    true,
		"artifact_id": stored.ArtifactID,
		"revision":    stored.Revision,
	}
	if version, ok := ui["version"].(string); ok && version != "" {
		pointer["version"] = version
	}
	return pointer, &stored
}

// MergeOpenUIStatements merges an OpenUI statement patch onto an earlier OpenUI program,
// updating matching identifiers in place, removing component identifiers assigned null, and
// appending newly introduced identifiers at the end.
func MergeOpenUIStatements(base, patch string) string {
	if strings.TrimSpace(base) == "" {
		return patch
	}
	if strings.TrimSpace(patch) == "" {
		return base
	}
	baseStmts := splitOpenUIStatements(base)
	patchStmts := splitOpenUIStatements(patch)

	var order []string
	byIdent := make(map[string]string, len(baseStmts)+len(patchStmts))
	for _, raw := range baseStmts {
		ident, _, ok := parseOpenUIAssignment(raw)
		if !ok {
			continue
		}
		if _, exists := byIdent[ident]; !exists {
			order = append(order, ident)
		}
		byIdent[ident] = raw
	}

	for _, raw := range patchStmts {
		ident, rhs, ok := parseOpenUIAssignment(raw)
		if !ok {
			continue
		}
		if rhs == "null" && !strings.HasPrefix(ident, "$") {
			delete(byIdent, ident)
			order = slices.DeleteFunc(order, func(k string) bool { return k == ident })
			continue
		}
		if _, exists := byIdent[ident]; !exists {
			order = append(order, ident)
		}
		byIdent[ident] = ident + " = " + rhs
	}

	out := make([]string, 0, len(order))
	for _, ident := range order {
		if stmt, ok := byIdent[ident]; ok {
			out = append(out, stmt)
		}
	}
	return strings.Join(out, "\n")
}

func splitOpenUIStatements(program string) []string {
	var stmts []string
	var current strings.Builder
	var quote byte
	escaped := false
	depth := 0
	lines := strings.Split(program, "\n")

	for _, line := range lines {
		if quote == 0 && depth == 0 {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" || strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "#") {
				continue
			}
		}
		if current.Len() > 0 {
			current.WriteByte('\n')
		}
		current.WriteString(line)

		for i := 0; i < len(line); i++ {
			ch := line[i]
			if escaped {
				escaped = false
				continue
			}
			if quote != 0 {
				if ch == '\\' {
					escaped = true
				} else if ch == quote {
					quote = 0
				}
				continue
			}
			if ch == '/' && i+1 < len(line) && line[i+1] == '/' {
				break
			}
			switch ch {
			case '"', '\'':
				quote = ch
			case '(', '[', '{':
				depth++
			case ')', ']', '}':
				if depth > 0 {
					depth--
				}
			}
		}

		if quote == 0 && depth == 0 {
			if stmt := strings.TrimSpace(current.String()); stmt != "" {
				stmts = append(stmts, stmt)
			}
			current.Reset()
		}
	}
	if tail := strings.TrimSpace(current.String()); tail != "" {
		stmts = append(stmts, tail)
	}
	return stmts
}

func parseOpenUIAssignment(stmt string) (ident, rhs string, ok bool) {
	for i := 0; i < len(stmt); i++ {
		if stmt[i] != '=' {
			continue
		}
		if i > 0 && (stmt[i-1] == '!' || stmt[i-1] == '<' || stmt[i-1] == '>' || stmt[i-1] == '=') {
			continue
		}
		if i+1 < len(stmt) && stmt[i+1] == '=' {
			i++
			continue
		}
		lhs := strings.TrimSpace(stmt[:i])
		if !openUIIdent.MatchString(lhs) {
			return "", "", false
		}
		return lhs, strings.TrimSpace(stmt[i+1:]), true
	}
	return "", "", false
}

// uiWords are the words of the reply a person's action names, among messages, or nothing when
// it is not one of them.
func uiWords(messages []Message, action map[string]any) string {
	id, _ := action["message_id"].(string)
	for _, m := range messages {
		if m.ID == id && m.Role == "assistant" {
			return m.Text
		}
	}
	return ""
}

// ActedOn are the words of this conversation's reply that a person's action names, read from
// Stream, which tell the model which of its replies the person was in. They are empty when the
// reply cannot be read, or is not this conversation's agent's.
func (c *Conversation) ActedOn(ctx context.Context, action map[string]any) string {
	id, _ := action["message_id"].(string)
	if id == "" {
		return ""
	}
	client, err := c.chat(ctx)
	if err != nil {
		return ""
	}
	found, err := client.Chat().GetMessage(ctx, id, &getstream.GetMessageRequest{})
	if err != nil {
		return ""
	}
	c.mu.Lock()
	cid, agentID := c.data.CID, c.data.Agent
	c.mu.Unlock()
	message := found.Data.Message
	if message.Cid != cid || !writtenByAgent(cid, agentID, message.User.ID) {
		return ""
	}
	stored, err := messageFromWire(message.ID, message.Text, message.Custom)
	if err != nil {
		return ""
	}
	return uiWords([]Message{stored}, action)
}

// uiBlock writes a reply's UI back as the block the model wrote, for the model's history.
func uiBlock(ui map[string]any) string {
	format, _ := ui["format"].(string)
	content, _ := ui["content"].(string)
	if format == "" || content == "" {
		return ""
	}
	info := "ui:" + format
	if format == "openui" {
		info = "openui-lang"
	} else if version, ok := ui["version"].(string); ok && version != "" {
		info += "@" + version
	}
	return "```" + info + "\n" + content + "\n```"
}

// ValidUIAction reports whether what a person did in a reply's UI may be recorded: an object
// naming the reply, the format and the action, of at most MaxUIAction bytes.
func ValidUIAction(action map[string]any) bool {
	if action == nil {
		return true
	}
	encoded, err := json.Marshal(action)
	if err != nil || len(encoded) > MaxUIAction {
		return false
	}
	for _, key := range []string{"message_id", "format", "name"} {
		if value, ok := action[key].(string); !ok || value == "" {
			return false
		}
	}
	return true
}

// DescribeUIAction is how the model is told what a person did in a reply's UI, after their
// words. reply is that reply's own words (ActedOn), which tell the model which of its replies
// the person was in when it wrote the same UI more than once, such as a plan for each of two
// trips. The action and the words are quoted as data: what a UI sends is not an instruction.
func DescribeUIAction(action map[string]any, reply string) string {
	if len(action) == 0 {
		return ""
	}
	format, _ := action["format"].(string)
	name, _ := action["name"].(string)
	payload, _ := json.Marshal(action["payload"])
	which := "your earlier reply"
	if words := strings.Join(strings.Fields(reply), " "); words != "" {
		if runes := []rune(words); len(runes) > maxActedOnRunes {
			words = string(runes[:maxActedOnRunes]) + "…"
		}
		which += " " + strconv.Quote(words)
	}
	return fmt.Sprintf("(They did this in the %s UI of %s: %s, with %s)", format, which, name, payload)
}
