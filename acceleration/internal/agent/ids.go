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
	// texts are results that are not JSON objects or lists: τ-bench's find_user_id_by_email
	// answers with the bare ID, noah_ito_3850, under no key to look it up by.
	var texts []string
	for _, message := range history {
		switch message.Role {
		case llm.User:
			said.WriteString(" " + message.Content + " " + withSpokenSeparators(digitsFor(message.Content)))
		case llm.ToolResult:
			if failedResult(message.Content) {
				// A refusal quotes the value it refused, which does not make it returned.
				continue
			}
			var result any
			if json.Unmarshal([]byte(message.Content), &result) != nil {
				texts = append(texts, message.Content)
			} else if text, bare := result.(string); bare {
				texts = append(texts, text)
			} else {
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
		if !returned && !slices.ContainsFunc(texts, func(result string) bool { return holdsValue(result, wanted) }) {
			return name, text, ids, true
		}
	}
	return "", "", nil, false
}

// missingID reports a handed-out ID the tool takes but the call left out, once a tool that
// makes or finds that thing has returned one. PhoneLLM Max dispatched a technician with no
// ticket_id right after create_ticket returned T-1842, and the tool ran with none.
func missingID(history []llm.Message, call llm.ToolCall, tools harness.Tools) (key string, known []string, missing bool) {
	var arguments map[string]any
	if json.Unmarshal([]byte(call.Arguments), &arguments) != nil {
		return "", nil, false
	}
	names := toolNames(tools)
	var properties map[string]any
	for _, tool := range tools.Tools {
		if tool.Name == call.Name {
			properties, _ = tool.Parameters["properties"].(map[string]any)
		}
	}
	keys := make([]string, 0, len(properties))
	for name := range properties {
		keys = append(keys, name)
	}
	slices.Sort(keys)
	called := map[string]string{}
	for _, message := range history {
		for _, made := range message.ToolCalls {
			called[made.ID] = made.Name
		}
	}
	for _, name := range keys {
		if _, given := arguments[name]; given || !handedOut(name, names) {
			continue
		}
		entity := "_" + strings.TrimSuffix(name, "_id")
		var ids []string
		for _, message := range history {
			if message.Role != llm.ToolResult || !strings.Contains(called[message.ToolCallID], entity) {
				continue
			}
			var result any
			if json.Unmarshal([]byte(message.Content), &result) != nil {
				continue
			}
			walkIDs(result, "", func(at, id string) {
				if entry := fmt.Sprintf("%s (%s)", id, at); (at == "id" || at == name) && !slices.Contains(ids, entry) {
					ids = append(ids, entry)
				}
			})
		}
		if len(ids) > 0 {
			return name, ids, true
		}
	}
	return "", nil, false
}

// emptyNamed reports a string argument sent empty that the tool's description asks for.
// MiniMax verified "with PIN, last four of the card or account, and service address" with
// the address empty, before the caller had finished reading it out. One the description
// offers as an alternative ("member ID or phone") may be left empty.
func emptyNamed(call llm.ToolCall, tools harness.Tools) (string, bool) {
	var arguments map[string]any
	if json.Unmarshal([]byte(call.Arguments), &arguments) != nil {
		return "", false
	}
	var description string
	var properties map[string]any
	for _, tool := range tools.Tools {
		if tool.Name == call.Name {
			description = " " + strings.ToLower(strings.Join(strings.FieldsFunc(tool.Description, func(r rune) bool {
				return !unicode.IsLetter(r) && !unicode.IsDigit(r)
			}), " ")) + " "
			properties, _ = tool.Parameters["properties"].(map[string]any)
		}
	}
	// A string property left out is as empty as one sent as "": MiniMax left the address out.
	keys := make([]string, 0, len(arguments)+len(properties))
	for key := range arguments {
		keys = append(keys, key)
	}
	for key, schema := range properties {
		if kind, _ := schema.(map[string]any)["type"].(string); kind == "string" && !slices.Contains(keys, key) {
			keys = append(keys, key)
		}
	}
	slices.Sort(keys)
	for _, key := range keys {
		value, given := arguments[key]
		if text, ok := value.(string); given && (!ok || strings.TrimSpace(text) != "") {
			continue
		}
		before, after, named := strings.Cut(description, " "+strings.ReplaceAll(key, "_", " ")+" ")
		if !named {
			continue
		}
		words := strings.Fields(before)
		if len(words) > 0 && words[len(words)-1] == "or" || strings.HasPrefix(after, "or ") {
			continue
		}
		// One left out counts only where the description lists it among what the tool
		// takes ("..., and service address"), not where it only names the thing ("Look up
		// an order").
		if !given && !slices.Contains(words[max(0, len(words)-3):], "and") {
			continue
		}
		return key, true
	}
	return "", false
}

// emptyResult is what a call with a named argument left empty is answered with.
func emptyResult(tool, key string) string {
	return fmt.Sprintf(didNotWork+"%s needs %s, and it was empty. If the caller has not given it yet, ask "+
		"them for it; if they have, call again with what they said.", tool, strings.ReplaceAll(key, "_", " "))
}

// placeholderName reports a name argument filled with a stand-in for one the caller never
// gave: PhoneLLM Mini booked tables for "Caller", Qwen for "John Doe". More than four words
// is suspicious unless a caller or tool supplied them: PhoneLLM Max booked a table under
// the note that resumes a turn ("Carry on with the caller from what is in your instructions...").
func placeholderName(history []llm.Message, call llm.ToolCall) (string, bool) {
	var arguments map[string]any
	if json.Unmarshal([]byte(call.Arguments), &arguments) != nil {
		return "", false
	}
	for key, value := range arguments {
		text, ok := value.(string)
		if !ok || (key != "name" && !strings.HasSuffix(key, "_name")) {
			continue
		}
		if (slices.Contains(placeholders, strings.ToLower(strings.TrimSpace(text))) || len(strings.Fields(text)) > 4 ||
			notAName(text)) && !nameWasGiven(history, text) {
			return text, true
		}
	}
	return "", false
}

// A name field can identify a place or business, not just a person. The lexical
// heuristics must not reject a name the caller or a successful tool already gave.
func nameWasGiven(history []llm.Message, name string) bool {
	words := func(text string) string {
		return " " + strings.ToLower(strings.Join(strings.FieldsFunc(text, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r)
		}), " ")) + " "
	}
	wanted := words(name)
	if wanted == "  " {
		return false
	}
	var contains func(any) bool
	contains = func(value any) bool {
		switch value := value.(type) {
		case string:
			return strings.Contains(words(value), wanted)
		case []any:
			return slices.ContainsFunc(value, contains)
		case map[string]any:
			for _, child := range value {
				if contains(child) {
					return true
				}
			}
		}
		return false
	}
	for _, message := range history {
		switch message.Role {
		case llm.User:
			if contains(message.Content) {
				return true
			}
		case llm.ToolResult:
			if failedResult(message.Content) {
				continue
			}
			var result any
			if json.Unmarshal([]byte(message.Content), &result) != nil {
				result = message.Content
			}
			if contains(result) {
				return true
			}
		}
	}
	return false
}

// unheardNumber reports an argument of four or more digits that neither the caller nor a
// tool ever gave. PhoneLLM Max verified with a last four of 1982 for "nine eight two one",
// and booked a phone of 512550142 for "five one two, five five five, oh one four two".
func unheardNumber(history []llm.Message, call llm.ToolCall) (key, value string, heard []string, unheard bool) {
	var arguments map[string]any
	if json.Unmarshal([]byte(call.Arguments), &arguments) != nil {
		return "", "", nil, false
	}
	var sources []string
	for _, message := range history {
		switch message.Role {
		case llm.User:
			sources = append(sources, message.Content, withSpokenSeparators(digitsFor(message.Content)))
			for _, number := range digitRun.FindAllString(digitsFor(message.Content), -1) {
				if len(number) >= 3 && !slices.Contains(heard, number) {
					heard = append(heard, number)
				}
			}
		case llm.ToolResult:
			// A refusal quotes the value refused: "19821" refused let 1982 through.
			if !failedResult(message.Content) {
				sources = append(sources, message.Content)
			}
		}
	}
	var digits []string
	for _, source := range sources {
		for _, group := range numberGroup.FindAllString(source, -1) {
			digits = append(digits, strings.Map(func(r rune) rune {
				if unicode.IsDigit(r) {
					return r
				}
				return -1
			}, group))
		}
	}
	keys := make([]string, 0, len(arguments))
	for name := range arguments {
		keys = append(keys, name)
	}
	slices.Sort(keys)
	for _, name := range keys {
		text, ok := arguments[name].(string)
		if !ok || len(text) < 4 || !allDigits(text) {
			continue
		}
		if !slices.ContainsFunc(digits, func(said string) bool { return strings.Contains(said, text) }) {
			return name, text, heard, true
		}
	}
	return "", "", nil, false
}

var digitRun = regexp.MustCompile(`\d+`)

// numberGroup is one number as written, its dashes and brackets included: "512-555-0142".
// Numbers apart are kept apart, so "4471, last four 9821" holds no 1982.
var numberGroup = regexp.MustCompile(`\d(?:[\d().-]*\d)?`)

func allDigits(text string) bool {
	return strings.IndexFunc(text, func(r rune) bool { return !unicode.IsDigit(r) }) < 0
}

// unheardResult is what a call with a number nobody gave is answered with.
func unheardResult(key, value string, heard []string) string {
	result := fmt.Sprintf(didNotWork+"the caller never said %q, and no tool returned it, so it cannot be "+
		"the %s. ", value, key)
	if len(heard) > 0 {
		result += "The numbers the caller said are: " + strings.Join(heard, ", ") + ". "
	}
	return result + "Call again with the one they gave, digit for digit, or read back what you heard and ask."
}

// misspelledName reports a name argument that is not the name the caller spelled out
// letter by letter. The caller said "The name is Okafor, o k a f o r" and every model booked
// the transcriber's "Alkafor" or "Alcophor".
//
// A name the caller spelled exactly at some point in the call is let through even where
// another spelling is close to it: the transcriber hears the same letters differently from
// one spelling to the next ("n o a h", then "n o a a"), and holding the name to whichever
// close spelling came first refused Noah and Ivan as the callers had spelled them.
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
		spellings = append(spellings, spelledNames(message.Content)...)
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
			if len(word) < 3 || slices.Contains(spellings, word) {
				continue
			}
			for _, spelling := range spellings {
				if closeTo(word, spelling) {
					return text, spelling, true
				}
			}
		}
	}
	return "", "", false
}

// spelledNames are the names a caller spelled out in one message, lowercase: three or more
// single letters read out one at a time, "o k a f o r", "P-A-T-E-L", "J. O. H. N.".
//
// A spelling's letters are joined by one kind of separator, spaces, commas, hyphens or
// periods, and a change of separator ends it: in "f i s c h e r. I need" the period ends
// the sentence, and read as one more letter the "I" made Fischer "fischeri". A letter only
// counts when it stands alone: in "that's a l v a r e z" the s ends a word, and read as a
// spelled letter it made Alvarez "Salvarez".
//
// A caller who says a name and then spells it with the next one in a single breath ("Lay
// Wilson. That's l e i w i l s o n") gives the transcriber no pause to write. A spelling
// that opens or closes with a name said in the same message is also read as the two
// names, so Lei and Wilson are each spelled as well as leiwilson.
func spelledNames(text string) []string {
	runes := []rune(strings.ToLower(text))
	alone := func(at int) bool {
		return at < len(runes) && unicode.IsLetter(runes[at]) &&
			(at == 0 || !unicode.IsLetter(runes[at-1]) && runes[at-1] != '\'' && runes[at-1] != '’') &&
			(at+1 == len(runes) || !unicode.IsLetter(runes[at+1]))
	}
	var runs []string
	for at := 0; at < len(runes); at++ {
		if !alone(at) {
			continue
		}
		letters := []rune{runes[at]}
		kind := rune(0)
		for {
			next := at + 1
			for next < len(runes) && strings.ContainsRune(" \t\n,.-", runes[next]) {
				next++
			}
			if next == at+1 || !alone(next) {
				break
			}
			separator := separatorKind(runes[at+1 : next])
			if kind == 0 {
				kind = separator
			} else if separator != kind {
				break
			}
			letters = append(letters, runes[next])
			at = next
		}
		if len(letters) >= 3 {
			runs = append(runs, string(letters))
		}
	}
	if len(runs) == 0 {
		return nil
	}
	said := strings.FieldsFunc(string(runes), func(r rune) bool { return !unicode.IsLetter(r) })
	spellings := slices.Clone(runs)
	for _, run := range runs {
		for _, word := range said {
			if len(word) < 3 || len(run)-len(word) < 2 {
				continue
			}
			if rest, opens := strings.CutPrefix(run, word); opens {
				spellings = append(spellings, word, rest)
			}
			if rest, closes := strings.CutSuffix(run, word); closes {
				spellings = append(spellings, rest, word)
			}
		}
	}
	return spellings
}

// separatorKind names what joins two spelled letters: a comma, a period or a hyphen when
// there is one, and a space when there is only space.
func separatorKind(between []rune) rune {
	for _, kind := range []rune{',', '.', '-'} {
		if slices.Contains(between, kind) {
			return kind
		}
	}
	return ' '
}

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

// notNames are words no name holds. Flux heard "Last name is Alvarez" as "Let Last name is
// Alvarez", and MiniMax booked "Let Alvarez"; it heard a barge-in's "wait, wait" as "Wing
// Wait" and booked that before the caller gave a name. Words that are also names (Will,
// May, Rose) are left out.
var notNames = []string{"let", "let's", "lets", "is", "my", "the", "name", "last", "first", "yes",
	"okay", "ok", "um", "uh", "hi", "hello", "please", "thanks", "it's", "its", "under", "this", "that",
	"wait", "sorry", "actually", "hold", "hang"}

// notAName reports a name with a word in it that no name holds.
func notAName(text string) bool {
	for _, word := range strings.Fields(strings.ToLower(text)) {
		if slices.Contains(notNames, strings.Trim(word, ".,!?\"")) {
			return true
		}
	}
	return false
}

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

// listedID is the ID key a list under key holds: "order_id" for "orders" or "order_ids",
// and "" for a key that is not plural.
func listedID(key string) string {
	if entity, ok := strings.CutSuffix(key, "_ids"); ok && entity != "" {
		return entity + "_id"
	}
	if entity, ok := strings.CutSuffix(key, "s"); ok && entity != "" && !strings.HasSuffix(entity, "s") && !strings.HasSuffix(entity, "_id") {
		return entity + "_id"
	}
	return ""
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

// walkIDs calls found with every string or number held under "id" or a key ending in "_id",
// and with every one listed under a plural key as the ID of what that key names: τ-bench's
// get_user_details lists a user's orders as "orders": ["#W5272531"], each an order_id.
func walkIDs(value any, at string, found func(at, id string)) {
	switch typed := value.(type) {
	case map[string]any:
		for name, inner := range typed {
			walkIDs(inner, name, found)
		}
	case []any:
		if listed := listedID(at); listed != "" {
			at = listed
		}
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

// holdsValue reports whether a result written as text gives wanted as one of its words, as
// the bare noah_ito_3850 does. wanted is already alphanumeric.
func holdsValue(result, wanted string) bool {
	return slices.ContainsFunc(strings.FieldsFunc(result, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && !strings.ContainsRune("_-#", r)
	}), func(word string) bool { return alphanumeric(word) == wanted })
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
