package harness

import (
	"encoding/json"
	"slices"
	"strconv"
	"strings"
	"unicode"
)

// SearchToolName and LoadToolName are how a model finds and loads the tools it was not
// offered at the start (Tools.Deferred). No other tool may take either name.
const (
	SearchToolName = "search_tool"
	LoadToolName   = "load_tool"
)

// DeferredToolNames are the names the router keeps for loading deferred tools.
var DeferredToolNames = []string{SearchToolName, LoadToolName}

// searchLimit is the most tools one search_tool call returns, enough to choose between
// without spending the context deferring them saved.
const searchLimit = 10

// summaryLimit caps the one line a tool is summarised by, in runes.
const summaryLimit = 200

// SearchTool is search_tool as the model is offered it.
func SearchTool() Tool {
	return Tool{
		Name: SearchToolName,
		Description: "Find tools you were not offered yet by what they do. Returns up to 10 tools, each " +
			"with its name and a line on what it does; load the ones you need with load_tool before " +
			"calling them. Search whenever none of your tools does what the conversation needs.",
		Parameters: map[string]any{
			"type":     "object",
			"required": []string{"query"},
			"properties": map[string]any{
				"query": map[string]any{"type": "string", "description": "Words for what the tool should do, such as look up a booking."},
			},
		},
	}
}

// LoadTool is load_tool as the model is offered it.
func LoadTool() Tool {
	return Tool{
		Name: LoadToolName,
		Description: "Load tools search_tool found, by name. Returns each one's full description and " +
			"input schema, and you can call them from your next step on.",
		Parameters: map[string]any{
			"type":     "object",
			"required": []string{"names"},
			"properties": map[string]any{
				"names": map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "minItems": 1},
			},
		},
	}
}

// SearchDeferred is what search_tool answers query with: the deferred tools whose name or
// description shares a word with it, those sharing the most first and a word in the name
// counting for more, in the order they were declared when they tie.
func SearchDeferred(deferred []Tool, query string) string {
	words := wordsOf(query)
	type match struct {
		tool  Tool
		score int
	}
	var matches []match
	for _, tool := range deferred {
		named, described := wordsOf(tool.Name), wordsOf(tool.Description)
		score := 0
		for _, word := range words {
			if slices.Contains(named, word) {
				score += 3
			}
			if slices.Contains(described, word) {
				score++
			}
		}
		if score > 0 {
			matches = append(matches, match{tool: tool, score: score})
		}
	}
	slices.SortStableFunc(matches, func(a, b match) int { return b.score - a.score })
	if len(matches) == 0 {
		return "No tool matches " + strconv.Quote(query) + ". Try other words for what it should do."
	}
	found := make([]map[string]string, 0, min(len(matches), searchLimit))
	for _, match := range matches[:min(len(matches), searchLimit)] {
		found = append(found, map[string]string{"name": match.tool.Name, "description": Summary(match.tool.Description)})
	}
	encoded, _ := json.Marshal(map[string]any{"tools": found})
	return string(encoded)
}

// LoadDeferred is what load_tool answers with for the deferred tools called names: each one's
// full description and input schema, and the names that are no deferred tool. found are the
// names it loaded, in the order asked.
func LoadDeferred(deferred []Tool, names []string) (result string, found []string) {
	var loaded []map[string]any
	var unknown []string
	for _, name := range names {
		name = strings.TrimSpace(name)
		at := slices.IndexFunc(deferred, func(tool Tool) bool { return tool.Name == name })
		if at < 0 {
			unknown = append(unknown, name)
			continue
		}
		if slices.Contains(found, name) {
			continue
		}
		found = append(found, name)
		loaded = append(loaded, map[string]any{
			"name": name, "description": deferred[at].Description, "input_schema": deferred[at].Parameters,
		})
	}
	answer := map[string]any{"loaded": loaded}
	if len(unknown) > 0 {
		answer["unknown"] = unknown
		answer["note"] = "Find a tool's name with search_tool before loading it."
	}
	encoded, _ := json.Marshal(answer)
	return string(encoded), found
}

// LoadedNames are the names a load_tool call's arguments ask for, or none when they cannot
// be read.
func LoadedNames(arguments string) []string {
	var asked struct {
		Names []string `json:"names"`
	}
	if err := json.Unmarshal([]byte(arguments), &asked); err != nil {
		return nil
	}
	return asked.Names
}

// Summary is the first line of a description, cut at a word to 200 runes.
func Summary(description string) string {
	var first string
	for line := range strings.Lines(description) {
		if first = strings.TrimSpace(line); first != "" {
			break
		}
	}
	runes := []rune(first)
	if len(runes) <= summaryLimit {
		return first
	}
	cut := string(runes[:summaryLimit])
	if space := strings.LastIndex(cut, " "); space > 0 {
		cut = cut[:space]
	}
	return cut + "…"
}

// wordsOf is text as lowercase words, split at anything but a letter or a digit, so the
// tool lookup_booking is found by "booking".
func wordsOf(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}
