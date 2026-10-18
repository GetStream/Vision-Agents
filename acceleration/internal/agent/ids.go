package agent

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"unicode"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// unfoundID reports an ID argument whose value no tool has returned, under that key or as an
// "id", with the IDs the tools have returned so far. PhoneLLM Mini passed the patient's "p1"
// as an appointment_id, "Thursday 2pm with doctor Chen" as one, and the gateway's error code
// as a ticket_id before any ticket was opened.
//
// Only an ID one of the tools hands out is checked: "ticket_id" when there is a create_ticket,
// "appointment_id" when there is a lookup_appointment. Any other ID, such as a member_id, is
// the caller's to give, and so is one the caller said along with what it is ("my ticket is
// 1842"). The error code "eight forty", said where no ticket was mentioned, is not a ticket.
func unfoundID(history []llm.Message, call llm.ToolCall, tools []string) (key, value string, known []string, found bool) {
	var arguments map[string]any
	if json.Unmarshal([]byte(call.Arguments), &arguments) != nil {
		return "", "", nil, false
	}
	var said strings.Builder
	var results []any
	for _, message := range history {
		switch message.Role {
		case llm.User:
			said.WriteString(" " + message.Content + " " + digitsFor(message.Content))
		case llm.ToolResult:
			var result any
			if json.Unmarshal([]byte(message.Content), &result) == nil {
				results = append(results, result)
			}
		}
	}
	heard := alphanumeric(said.String())
	keys := make([]string, 0, len(arguments))
	for name := range arguments {
		keys = append(keys, name)
	}
	slices.Sort(keys)
	for _, name := range keys {
		text, ok := arguments[name].(string)
		wanted := alphanumeric(text)
		if !ok || !handedOut(name, tools) {
			continue
		}
		if wanted == "" {
			return name, text, collectIDs(results), true
		}
		if strings.Contains(heard, wanted) && strings.Contains(strings.ToLower(said.String()), entityOf(name)) {
			continue
		}
		returned := false
		var ids []string
		for _, result := range results {
			walkIDs(result, "", func(at, id string) {
				if (at == name || at == "id") && alphanumeric(id) == wanted {
					returned = true
				}
				if entry := fmt.Sprintf("%s (%s)", id, at); !slices.Contains(ids, entry) {
					ids = append(ids, entry)
				}
			})
		}
		if !returned {
			return name, text, ids, true
		}
	}
	return "", "", nil, false
}

// placeholderName reports a name argument filled with a stand-in for one the caller never
// gave: PhoneLLM Mini booked tables for "Caller", Qwen for "John Doe".
func placeholderName(call llm.ToolCall) (string, bool) {
	var arguments map[string]any
	if json.Unmarshal([]byte(call.Arguments), &arguments) != nil {
		return "", false
	}
	for key, value := range arguments {
		text, ok := value.(string)
		if ok && (key == "name" || strings.HasSuffix(key, "_name")) && slices.Contains(placeholders, strings.ToLower(strings.TrimSpace(text))) {
			return text, true
		}
	}
	return "", false
}

var placeholders = []string{"caller", "the caller", "customer", "guest", "user", "john doe", "jane doe", "unknown", "n/a"}

// placeholderResult is what a call with a stand-in name is answered with.
func placeholderResult(name string) string {
	return fmt.Sprintf(didNotWork+"%q is not a name the caller gave. Ask them for their name, and call "+
		"again with it.", name)
}

func toolNames(tools harness.Tools) []string {
	names := make([]string, 0, len(tools.Tools))
	for _, tool := range tools.Tools {
		names = append(names, tool.Name)
	}
	return names
}

func collectIDs(results []any) []string {
	var ids []string
	for _, result := range results {
		walkIDs(result, "", func(at, id string) {
			if entry := fmt.Sprintf("%s (%s)", id, at); !slices.Contains(ids, entry) {
				ids = append(ids, entry)
			}
		})
	}
	return ids
}

// entityOf is what an ID argument names: "appointment" for "appointment_id".
func entityOf(key string) string {
	return strings.ReplaceAll(strings.TrimSuffix(key, "_id"), "_", " ")
}

// handedOut reports whether key is the ID of something one of the tools makes or looks up.
func handedOut(key string, tools []string) bool {
	if !strings.HasSuffix(key, "_id") {
		return false
	}
	entity := strings.TrimSuffix(key, "_id")
	return slices.ContainsFunc(tools, func(tool string) bool { return tool != "" && strings.Contains(tool, "_"+entity) })
}

// walkIDs calls found with every string or number held under "id" or a key ending in "_id".
func walkIDs(value any, at string, found func(at, id string)) {
	switch typed := value.(type) {
	case map[string]any:
		for name, inner := range typed {
			walkIDs(inner, name, found)
		}
	case []any:
		for _, inner := range typed {
			walkIDs(inner, at, found)
		}
	case string:
		if at == "id" || strings.HasSuffix(at, "_id") {
			found(at, typed)
		}
	case float64:
		if at == "id" || strings.HasSuffix(at, "_id") {
			found(at, fmt.Sprint(typed))
		}
	}
}

// alphanumeric lowercases text and drops everything but letters and digits, so "T-1842"
// and "t1842" match.
func alphanumeric(text string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return unicode.ToLower(r)
		}
		return -1
	}, text)
}

// unfoundIDResult is what a call with an ID nothing gave the model is answered with. It opens
// with didNotWork so that it backs no claim.
func unfoundIDResult(key, value string, known []string) string {
	result := fmt.Sprintf(didNotWork+"no tool has given you %q as a %s, and the caller did not say it. ", value, key)
	if strings.TrimSpace(value) == "" {
		result = fmt.Sprintf(didNotWork+"%s is empty. ", key)
	}
	if len(known) > 0 {
		result += "The IDs your tools have returned are: " + strings.Join(known, ", ") + ". "
	}
	return result + "Call the tool that gives you the right one first, or use one of those."
}
