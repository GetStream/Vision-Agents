package agent

import (
	"encoding/json"
	"fmt"
	"regexp"
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
// gave: PhoneLLM Mini booked tables for "Caller", Qwen for "John Doe". A name of more than
// four words is not one either: PhoneLLM Max booked a table under the note that resumes a
// turn ("Carry on with the caller from what is in your instructions...").
func placeholderName(call llm.ToolCall) (string, bool) {
	var arguments map[string]any
	if json.Unmarshal([]byte(call.Arguments), &arguments) != nil {
		return "", false
	}
	for key, value := range arguments {
		text, ok := value.(string)
		if !ok || (key != "name" && !strings.HasSuffix(key, "_name")) {
			continue
		}
		if slices.Contains(placeholders, strings.ToLower(strings.TrimSpace(text))) || len(strings.Fields(text)) > 4 {
			return text, true
		}
	}
	return "", false
}

// misspelledName reports a name argument that is not the name the caller spelled out
// letter by letter. The caller said "The name is Okafor, o k a f o r" and every model booked
// the transcriber's "Alkafor" or "Alcophor".
func misspelledName(history []llm.Message, call llm.ToolCall) (name, spelled string, misspelled bool) {
	var arguments map[string]any
	if json.Unmarshal([]byte(call.Arguments), &arguments) != nil {
		return "", "", false
	}
	var spellings []string
	for _, message := range history {
		if message.Role != llm.User {
			continue
		}
		for _, match := range spelledOut.FindAllString(message.Content, -1) {
			spellings = append(spellings, strings.Join(strings.FieldsFunc(match, func(r rune) bool { return !unicode.IsLetter(r) }), ""))
		}
	}
	if len(spellings) == 0 {
		return "", "", false
	}
	for key, value := range arguments {
		text, ok := value.(string)
		if !ok || (key != "name" && !strings.HasSuffix(key, "_name")) {
			continue
		}
		for _, word := range strings.Fields(text) {
			word = alphanumeric(word)
			for _, spelling := range spellings {
				spelling = strings.ToLower(spelling)
				if word != spelling && len(word) >= 3 && closeTo(word, spelling) {
					return text, spelling, true
				}
			}
		}
	}
	return "", "", false
}

// spelledOut matches three or more single letters read out one at a time: "o k a f o r",
// "P-A-T-E-L".
var spelledOut = regexp.MustCompile(`(?i)\b[a-z](?:[\s,.-]+[a-z]\b){2,}`)

// closeTo reports whether two words are near enough to be one name heard two ways: the same
// first or last two letters, and lengths within three.
func closeTo(a, b string) bool {
	if d := len(a) - len(b); d > 3 || d < -3 || len(a) < 2 || len(b) < 2 {
		return false
	}
	return a[len(a)-2:] == b[len(b)-2:] || a[:2] == b[:2]
}

// misspelledResult is what a call with a misspelled name is answered with.
func misspelledResult(name, spelled string) string {
	return fmt.Sprintf(didNotWork+"the caller spelled the name out as %s, so it is %s, not %q. Call "+
		"again with it.", strings.Join(strings.Split(spelled, ""), " "),
		strings.ToUpper(spelled[:1])+spelled[1:], name)
}

var placeholders = []string{"caller", "the caller", "customer", "guest", "user", "john doe", "jane doe", "unknown", "n/a"}

// placeholderResult is what a call with a stand-in name is answered with.
func placeholderResult(name string) string {
	return fmt.Sprintf(didNotWork+"%q is not a name the caller gave. Call again with the name they "+
		"gave, or ask them for it.", name)
}

// misshapenLastFour reports a "last four" argument that is not four digits. The transcriber
// ran "nine eight two one, fourteen Cedar Lane" together as "982114 Cedar Lane", and
// PhoneLLM Mini sent verify_account a last4 of 98214 and of 129821.
func misshapenLastFour(call llm.ToolCall) (key, value string, misshapen bool) {
	var arguments map[string]any
	if json.Unmarshal([]byte(call.Arguments), &arguments) != nil {
		return "", "", false
	}
	for name, raw := range arguments {
		text, ok := raw.(string)
		if !ok || text == "" || !slices.Contains(lastFourKeys, strings.ToLower(name)) {
			continue
		}
		digits := strings.Map(func(r rune) rune {
			if unicode.IsDigit(r) {
				return r
			}
			return -1
		}, text)
		if len(digits) != 4 || len(digits) != len(strings.TrimSpace(text)) {
			return name, text, true
		}
		// Given the split, PhoneLLM Mini sent last4 9821 with the address "98214 Cedar Lane".
		if address, ok := arguments["address"].(string); ok {
			if number, _, _ := strings.Cut(strings.TrimSpace(address), " "); len(number) > 4 && strings.HasPrefix(number, digits) {
				return "address", address, true
			}
		}
	}
	return "", "", false
}

var lastFourKeys = []string{"last4", "last_4", "last_four", "lastfour"}

// misshapenResult is what a call with a misshapen last four is answered with. When the
// caller's words hold a number run into the street after it, it spells out the split:
// told only the rule, PhoneLLM Mini asked the caller for four digits they had already given.
func misshapenResult(key, value string, history []llm.Message) string {
	result := fmt.Sprintf(didNotWork+"%s takes exactly four digits, and %q is not four digits. ", key, value)
	if !slices.Contains(lastFourKeys, strings.ToLower(key)) {
		result = fmt.Sprintf(didNotWork+"the address %q still opens with the last four. ", value)
	}
	if number, street, ok := runTogether(history); ok {
		return result + fmt.Sprintf("The caller's %q is the last four %s followed by the address %s %s: "+
			"call again with those.", number+" "+street, number[:4], number[4:], street)
	}
	return result + "When what you heard runs into the next number (\"553017 Oak Road\"), the first " +
		"four are the last four and the rest begins the address (17 Oak Road). Otherwise read back " +
		"what you heard and ask for the four digits."
}

var runOn = regexp.MustCompile(`\b(\d{5,8})\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)`)

// runTogether finds, in what the caller said last, a number of five to eight digits
// followed by a capitalised street.
func runTogether(history []llm.Message) (number, street string, ok bool) {
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role != llm.User {
			continue
		}
		if match := runOn.FindStringSubmatch(history[i].Content); match != nil {
			return match[1], match[2], true
		}
	}
	return "", "", false
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
