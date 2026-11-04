package conversation

import (
	"context"
	"encoding/json"
	"fmt"
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

// uiFences are the fences formats ask models for, by the format they hold.
var uiFences = map[string]string{"openui-lang": "openui"}

// uiFormatOf reads a fence's info string as a UI format, and its version when it names one.
func uiFormatOf(info string) (format, version string, ok bool) {
	info = strings.TrimSpace(info)
	if format, known := uiFences[info]; known {
		return format, "", true
	}
	named, found := strings.CutPrefix(info, "ui:")
	if !found || named == "" {
		return "", "", false
	}
	format, version, _ = strings.Cut(named, "@")
	return format, version, displayID.MatchString(format)
}

// splitUI separates a reply's text into the words people read and its UI. finished says the
// reply is over: until then a block not closed yet is still being written, and a last line that
// may be opening one is held back rather than shown for a moment as text.
func splitUI(text string, finished bool) (string, map[string]any) {
	var prose, content []string
	var ui map[string]any
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
			format, version, ok := uiFormatOf(strings.TrimLeft(line, " ")[3:])
			if !ok {
				prose = append(prose, line)
				inOther = true
				continue
			}
			inUI = true
			if ui == nil {
				capturing = true
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
	ui["content"] = strings.Join(content, "\n")
	ui["status"] = "streaming"
	if closed || finished {
		ui["status"] = "complete"
	}
	return strings.TrimSpace(strings.Join(prose, "\n")), ui
}

// wireUI is a message's text and UI as Stream keeps them. Only an agent's reply has UI.
func wireUI(m Message) (string, map[string]any) {
	if m.Role != "assistant" {
		return m.Text, nil
	}
	text, ui := splitUI(m.Text, m.FinishedAt != nil)
	if ui == nil {
		ui = m.UI
	}
	return text, ui
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
	custom := make(map[string]any, len(set))
	for key, value := range set {
		if !slices.Contains(own, key) {
			custom[key] = value
		}
	}
	if encoded, err := json.Marshal(custom); err == nil && len(encoded) <= maxMessageCustom {
		return false
	}
	delete(set, "ai_ui")
	return true
}

// uiUnset fits an update's UI (fitUI), and unsets the UI an earlier update set when this one
// leaves it out, so the reply does not keep a UI cut off midway.
func uiUnset(set map[string]any) []string {
	if fitUI(set, "text", "attachments") {
		return []string{"ai_ui"}
	}
	return nil
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
	cid := c.data.CID
	c.mu.Unlock()
	message := found.Data.Message
	if message.Cid != cid {
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
	for fence, named := range uiFences {
		if named == format {
			info = fence
		}
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
