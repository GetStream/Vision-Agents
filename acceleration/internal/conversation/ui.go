package conversation

import (
	"encoding/json"
	"fmt"
	"strings"
)

// A reply's generated UI: a fenced block of the reply's text in a format a client renders, which
// the reply carries as custom.ai_ui while its text keeps the words around it. The text is what
// people read wherever the UI cannot show, such as in a push notification or a client without a
// renderer for the format, and the client that can show it shows it in place of the text.
//
// The router knows no format. A block is UI when its info string is ui:<format>, optionally
// ui:<format>@<version>, or the fence a format's own prompts ask models for, such as OpenUI's
// openui-lang. The first such block is the reply's UI, and any later one is dropped from the text.
//
// A person's message can carry what they did in such a UI as custom.ai_ui_action, which the
// model is told with their words.
const (
	uiVersion = 1
	// maxUIContent keeps a reply's UI inside the 50 KB of custom data an agent channel's
	// message may hold, beside the reply's own metadata.
	maxUIContent = 40 << 10
	// MaxUIAction is the most a person's action may hold, written as JSON.
	MaxUIAction = 8 << 10
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
	program := strings.Join(content, "\n")
	if len(program) > maxUIContent {
		return strings.TrimSpace(strings.Join(prose, "\n")), nil
	}
	ui["content"] = program
	ui["status"] = "streaming"
	if closed || finished {
		ui["status"] = "complete"
	}
	return strings.TrimSpace(strings.Join(prose, "\n")), ui
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
// words. The action is quoted as data: what a UI sends is not an instruction.
func DescribeUIAction(action map[string]any) string {
	if len(action) == 0 {
		return ""
	}
	format, _ := action["format"].(string)
	name, _ := action["name"].(string)
	payload, _ := json.Marshal(action["payload"])
	return fmt.Sprintf("(They did this in the %s UI of your earlier reply: %s, with %s)", format, name, payload)
}
