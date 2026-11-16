package agent

import (
	"regexp"
	"strings"
)

const (
	uiModeReplace = "replace"
	uiModePatch   = "patch"
	uiHeaderLimit = 256
)

type uiScanState uint8

const (
	uiOutside uiScanState = iota
	uiInOtherFence
	uiInFence
)

var (
	uiFormatPattern  = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,128}$`)
	uiVersionPattern = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,64}$`)
)

// uiStream separates fenced UI blocks from the spoken or fallback prose of a reply as tokens
// arrive, releasing both sides immediately except at the start of a line, where a fence may
// open or close a block. A block opens only on a fence at the start of a line: this runs on
// every agent's reply, and prose that only mentions UI must stay prose.
type uiStream struct {
	state       uiScanState
	midLine     bool
	firstUILine bool
	captured    bool
	capturing   bool
	format      string
	version     string
	mode        string
	pending     strings.Builder
	content     strings.Builder
}

// Reset clears all per-turn state for the next reply.
func (u *uiStream) Reset() {
	*u = uiStream{}
}

// Add consumes a chunk of model text after harness directive filtering, returning the
// prose meant for speech/text and any UI events completed or streamed within the chunk.
func (u *uiStream) Add(turnID, text string) (string, []Event) {
	if text == "" {
		return "", nil
	}
	u.pending.WriteString(text)

	var prose strings.Builder
	var events []Event

	for {
		buf := u.pending.String()
		if buf == "" {
			break
		}

		switch u.state {
		case uiOutside:
			if !u.stepOutside(turnID, buf, &prose, &events) {
				return prose.String(), events
			}
		case uiInOtherFence:
			if !u.stepOtherFence(buf, &prose) {
				return prose.String(), events
			}
		case uiInFence:
			if !u.stepUIFence(turnID, buf, &events) {
				return prose.String(), events
			}
		}
	}

	return prose.String(), events
}

// Flush releases any trailing buffered bytes when the model's reply ends.
func (u *uiStream) Flush(turnID string) (string, []Event) {
	buf := u.pending.String()
	u.pending.Reset()

	var prose strings.Builder
	var events []Event

	switch u.state {
	case uiOutside, uiInOtherFence:
		prose.WriteString(buf)
		u.state = uiOutside
		u.midLine = false

	case uiInFence:
		if !u.midLine && strings.TrimSpace(buf) == "```" {
			u.closeUI(&events, turnID)
		} else {
			if buf != "" {
				prefix := ""
				if !u.midLine && !u.firstUILine {
					prefix = "\n"
				}
				u.emitUI(&events, turnID, prefix+buf)
			}
			u.closeUI(&events, turnID)
		}
		u.state = uiOutside
		u.midLine = false
	}

	return prose.String(), events
}

// HistoryBlock formats the captured UI block back as a canonical fence so the model's
// own conversation history remembers the UI it rendered on earlier turns.
func (u *uiStream) HistoryBlock() string {
	if !u.captured || u.format == "" || u.content.Len() == 0 {
		return ""
	}
	info := "ui:" + u.format
	if u.version != "" {
		info += "@" + u.version
	}
	if u.mode == uiModePatch {
		if u.format == "openui" && u.version == "" {
			info = "openui-patch"
		} else {
			info += "+patch"
		}
	} else if u.format == "openui" && u.version == "" {
		info = "openui-lang"
	}
	return "```" + info + "\n" + u.content.String() + "\n```"
}

func (u *uiStream) stepOutside(turnID, buf string, prose *strings.Builder, events *[]Event) bool {
	if !u.midLine {
		trimmed := strings.TrimLeft(buf, " \t")
		switch {
		case trimmed == "":
			if len(buf) <= uiHeaderLimit {
				return false
			}
			prose.WriteString(buf)
			u.midLine = true
			u.pending.Reset()
			return false

		case trimmed == "`" || trimmed == "``":
			return false

		case strings.HasPrefix(trimmed, "```"):
			nl := strings.IndexByte(buf, '\n')
			if nl < 0 {
				if len(buf) <= uiHeaderLimit {
					return false
				}
				prose.WriteString(buf)
				u.midLine = true
				u.pending.Reset()
				return false
			}
			line := strings.TrimRight(buf[:nl], "\r")
			info := strings.TrimLeft(line, " \t")[3:]
			rest := buf[nl+1:]
			if format, version, mode, ok := parseUIFenceInfo(info); ok {
				u.openUI(format, version, mode)
				u.state = uiInFence
				u.midLine = false
				u.firstUILine = true
				u.setPending(rest)
				return true
			}
			prose.WriteString(buf[:nl+1])
			u.state = uiInOtherFence
			u.midLine = false
			u.setPending(rest)
			return true

		default:
			lead := len(buf) - len(trimmed)
			if lead > 0 {
				prose.WriteString(buf[:lead])
				u.setPending(trimmed)
			}
			u.midLine = true
			return true
		}
	}

	nl := strings.IndexByte(buf, '\n')
	if nl < 0 {
		prose.WriteString(buf)
		u.pending.Reset()
		return false
	}
	prose.WriteString(buf[:nl+1])
	u.midLine = false
	u.setPending(buf[nl+1:])
	return true
}

func (u *uiStream) stepOtherFence(buf string, prose *strings.Builder) bool {
	if !u.midLine {
		nl := strings.IndexByte(buf, '\n')
		if nl >= 0 {
			line := strings.TrimRight(buf[:nl], "\r")
			prose.WriteString(buf[:nl+1])
			if strings.TrimSpace(line) == "```" {
				u.state = uiOutside
			}
			u.midLine = false
			u.setPending(buf[nl+1:])
			return true
		}
		if isPossibleClosingFence(buf) && len(buf) <= uiHeaderLimit {
			return false
		}
		prose.WriteString(buf)
		u.midLine = true
		u.pending.Reset()
		return false
	}

	nl := strings.IndexByte(buf, '\n')
	if nl < 0 {
		prose.WriteString(buf)
		u.pending.Reset()
		return false
	}
	prose.WriteString(buf[:nl+1])
	u.midLine = false
	u.setPending(buf[nl+1:])
	return true
}

func (u *uiStream) stepUIFence(turnID, buf string, events *[]Event) bool {
	if !u.midLine {
		nl := strings.IndexByte(buf, '\n')
		if nl >= 0 {
			line := strings.TrimRight(buf[:nl], "\r")
			rest := buf[nl+1:]
			if strings.TrimSpace(line) == "```" {
				u.closeUI(events, turnID)
				u.state = uiOutside
				u.midLine = false
				u.setPending(rest)
				return true
			}
			prefix := ""
			if !u.firstUILine {
				prefix = "\n"
			}
			u.firstUILine = false
			u.emitUI(events, turnID, prefix+line)
			u.midLine = false
			u.setPending(rest)
			return true
		}
		if isPossibleClosingFence(buf) && len(buf) <= uiHeaderLimit {
			return false
		}
		prefix := ""
		if !u.firstUILine {
			prefix = "\n"
		}
		u.firstUILine = false
		u.emitUI(events, turnID, prefix+buf)
		u.midLine = true
		u.pending.Reset()
		return false
	}

	nl := strings.IndexByte(buf, '\n')
	if nl < 0 {
		u.emitUI(events, turnID, buf)
		u.pending.Reset()
		return false
	}
	if nl > 0 {
		u.emitUI(events, turnID, buf[:nl])
	}
	u.midLine = false
	u.setPending(buf[nl+1:])
	return true
}

func (u *uiStream) openUI(format, version, mode string) {
	if !u.captured {
		u.captured = true
		u.capturing = true
		u.format = format
		u.version = version
		u.mode = mode
		return
	}
	u.capturing = false
}

func (u *uiStream) emitUI(events *[]Event, turnID, delta string) {
	if !u.capturing || delta == "" {
		return
	}
	u.content.WriteString(delta)
	*events = append(*events, UIDelta{
		TurnID:  turnID,
		Format:  u.format,
		Version: u.version,
		Mode:    u.mode,
		Delta:   delta,
	})
}

func (u *uiStream) closeUI(events *[]Event, turnID string) {
	if !u.capturing {
		return
	}
	u.capturing = false
	*events = append(*events, UICompleted{
		TurnID:  turnID,
		Format:  u.format,
		Version: u.version,
		Mode:    u.mode,
		Content: u.content.String(),
	})
}

func (u *uiStream) setPending(rest string) {
	u.pending.Reset()
	if rest != "" {
		u.pending.WriteString(rest)
	}
}

func isPossibleClosingFence(line string) bool {
	trimmed := strings.TrimLeft(line, " \t")
	if trimmed == "" {
		return true
	}
	if len(trimmed) < 3 {
		return strings.HasPrefix("```", trimmed)
	}
	if !strings.HasPrefix(trimmed, "```") {
		return false
	}
	return strings.Trim(trimmed[3:], " \t\r") == ""
}

func parseUIFenceInfo(info string) (format, version, mode string, ok bool) {
	info = strings.TrimSpace(info)
	switch info {
	case "openui-lang":
		return "openui", "", uiModeReplace, true
	case "openui-patch":
		return "openui", "", uiModePatch, true
	}
	named, found := strings.CutPrefix(info, "ui:")
	if !found || named == "" {
		return "", "", "", false
	}
	mode = uiModeReplace
	if base, patched := strings.CutSuffix(named, "+patch"); patched {
		mode = uiModePatch
		named = base
	}
	format, version, _ = strings.Cut(named, "@")
	if !uiFormatPattern.MatchString(format) {
		return "", "", "", false
	}
	if version != "" && !uiVersionPattern.MatchString(version) {
		return "", "", "", false
	}
	return format, version, mode, true
}
