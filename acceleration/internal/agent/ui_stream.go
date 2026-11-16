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
	uiInTag
)

var (
	uiFormatPattern  = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,128}$`)
	uiVersionPattern = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,64}$`)
)

// uiStream separates interactive UI blocks from the spoken or fallback prose of a reply
// as tokens arrive, releasing both sides immediately except at line or tag boundaries that
// may open or close a block.
type uiStream struct {
	state       uiScanState
	midLine     bool
	firstUILine bool
	tagTrimLead bool
	tagHeldNL   string
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
		case uiInTag:
			if !u.stepUITag(turnID, buf, &events) {
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

	case uiInTag:
		if strings.TrimSpace(buf) == "</ui>" {
			u.tagHeldNL = ""
			u.closeUI(&events, turnID)
		} else {
			tail := ""
			if buf != "" {
				tail = u.tagHeldNL + buf
			}
			u.tagHeldNL = ""
			if u.tagTrimLead {
				tail = trimSingleNewline(tail)
				u.tagTrimLead = false
			}
			if tail != "" {
				u.emitUI(&events, turnID, tail)
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
	lt := strings.IndexByte(buf, '<')

	if nl < 0 && lt < 0 {
		prose.WriteString(buf)
		u.pending.Reset()
		return false
	}

	if nl >= 0 && (lt < 0 || nl < lt) {
		prose.WriteString(buf[:nl+1])
		u.midLine = false
		u.setPending(buf[nl+1:])
		return true
	}

	if lt > 0 {
		prose.WriteString(buf[:lt])
	}
	tag := buf[lt:]
	format, version, mode, rest, status := parseOpeningUITag(tag)
	switch status {
	case tagNeedMore:
		u.setPending(tag)
		return false
	case tagNotUI:
		prose.WriteByte('<')
		u.setPending(tag[1:])
		return true
	default:
		u.openUI(format, version, mode)
		u.state = uiInTag
		u.tagTrimLead = true
		u.tagHeldNL = ""
		u.setPending(rest)
		return true
	}
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

func (u *uiStream) stepUITag(turnID, buf string, events *[]Event) bool {
	if u.tagTrimLead {
		if buf == "\r" {
			return false
		}
		buf = trimSingleNewline(buf)
		u.tagTrimLead = false
		u.setPending(buf)
		if buf == "" {
			return false
		}
	}

	lt := strings.IndexByte(buf, '<')
	if lt >= 0 {
		if lt > 0 {
			before := u.tagHeldNL + buf[:lt]
			u.tagHeldNL = ""
			body, held := splitTrailingNewline(before)
			if body != "" {
				u.emitUI(events, turnID, body)
			}
			u.tagHeldNL = held
		}
		tag := buf[lt:]
		if strings.HasPrefix(tag, "</ui>") {
			u.tagHeldNL = ""
			u.closeUI(events, turnID)
			u.state = uiOutside
			u.midLine = true
			u.setPending(tag[len("</ui>"):])
			return true
		}
		if len(tag) < len("</ui>") && strings.HasPrefix("</ui>", tag) {
			u.setPending(tag)
			return false
		}
		u.emitUI(events, turnID, u.tagHeldNL+"<")
		u.tagHeldNL = ""
		u.setPending(tag[1:])
		return true
	}

	combined := u.tagHeldNL + buf
	body, held := splitTrailingNewline(combined)
	if body != "" {
		u.emitUI(events, turnID, body)
	}
	u.tagHeldNL = held
	u.pending.Reset()
	return false
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

type uiTagStatus uint8

const (
	tagNotUI uiTagStatus = iota
	tagNeedMore
	tagMatched
)

func parseOpeningUITag(tag string) (format, version, mode, rest string, status uiTagStatus) {
	if len(tag) < 3 {
		if strings.HasPrefix("<ui", tag) {
			return "", "", "", "", tagNeedMore
		}
		return "", "", "", "", tagNotUI
	}
	if !strings.HasPrefix(tag, "<ui") {
		return "", "", "", "", tagNotUI
	}
	if len(tag) == 3 {
		return "", "", "", "", tagNeedMore
	}
	next := tag[3]
	if next != '>' && next != ' ' && next != '\t' && next != '\r' && next != '\n' {
		return "", "", "", "", tagNotUI
	}
	gt := strings.IndexByte(tag, '>')
	if gt < 0 {
		if len(tag) <= uiHeaderLimit {
			return "", "", "", "", tagNeedMore
		}
		return "", "", "", "", tagNotUI
	}
	if tag[gt-1] == '/' {
		return "", "", "", "", tagNotUI
	}
	attrs, ok := parseUITagAttrs(tag[3:gt])
	if !ok {
		return "", "", "", "", tagNotUI
	}
	format = attrs["format"]
	if format == "" {
		format = "openui"
	}
	if !uiFormatPattern.MatchString(format) {
		return "", "", "", "", tagNotUI
	}
	version = attrs["version"]
	if version != "" && !uiVersionPattern.MatchString(version) {
		return "", "", "", "", tagNotUI
	}
	mode = attrs["mode"]
	if mode == "" {
		mode = uiModeReplace
	}
	if mode != uiModeReplace && mode != uiModePatch {
		return "", "", "", "", tagNotUI
	}
	return format, version, mode, tag[gt+1:], tagMatched
}

func parseUITagAttrs(raw string) (map[string]string, bool) {
	attrs := map[string]string{}
	s := strings.TrimSpace(raw)
	for s != "" {
		eq := strings.IndexByte(s, '=')
		if eq <= 0 {
			return nil, false
		}
		key := strings.TrimSpace(s[:eq])
		switch key {
		case "format", "version", "mode", "surface":
		default:
			return nil, false
		}
		s = strings.TrimLeft(s[eq+1:], " \t\r\n")
		if s == "" || (s[0] != '"' && s[0] != '\'') {
			return nil, false
		}
		quote := s[0]
		end := strings.IndexByte(s[1:], quote)
		if end < 0 {
			return nil, false
		}
		attrs[key] = s[1 : 1+end]
		s = strings.TrimSpace(s[2+end:])
	}
	return attrs, true
}

func trimSingleNewline(s string) string {
	if rest, ok := strings.CutPrefix(s, "\r\n"); ok {
		return rest
	}
	return strings.TrimPrefix(s, "\n")
}

func splitTrailingNewline(s string) (body, newline string) {
	if strings.HasSuffix(s, "\r\n") {
		return s[:len(s)-2], "\r\n"
	}
	if strings.HasSuffix(s, "\n") {
		return s[:len(s)-1], "\n"
	}
	if strings.HasSuffix(s, "\r") {
		return s[:len(s)-1], "\r"
	}
	return s, ""
}
