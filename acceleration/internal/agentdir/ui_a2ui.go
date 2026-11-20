package agentdir

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strings"
)

// The A2UI basic catalog and protocol as stream-chat-swift's renderer has them, which
// a2ui/generate.sh writes, so the agent is told what the apps draw. writing.md is the renderer's
// own section on how a reply carries its UI (A2UIInstructions), which isn't public to generate.
var (
	//go:embed a2ui/basic_catalog.json
	a2uiBasicCatalogJSON []byte
	//go:embed a2ui/protocol.md
	a2uiProtocol string
	//go:embed a2ui/writing.md
	a2uiWriting string
)

const (
	a2uiFence = "ui:a2ui@1.0"
	// a2uiCatalogID is the id the apps' A2UI renderers give the basic catalog with the app's
	// components beside its own (A2UIRenderer.defaultCatalogId in stream-chat-swift).
	a2uiCatalogID = "getstream.io:app-catalog"
	// a2uiMaxNesting is how deep a renderer draws components (A2UIProtocol.maxNestingDepth).
	a2uiMaxNesting = 50
)

var a2uiIdentifier = regexp.MustCompile(`^[\p{L}_][\p{L}\p{N}_]*$`)

var a2uiMessageKinds = []string{
	"createSurface", "updateComponents", "updateDataModel", "deleteSurface",
	"callRendererFunction", "agentFunctionResponse",
}

type a2uiCatalog struct {
	Schema       string                     `json:"$schema"`
	Title        string                     `json:"title"`
	Description  string                     `json:"description"`
	Instructions string                     `json:"instructions"`
	Components   map[string]json.RawMessage `json:"components"`
	Functions    map[string]json.RawMessage `json:"functions"`
}

var a2uiBasic = func() a2uiCatalog {
	var catalog a2uiCatalog
	if err := json.Unmarshal(a2uiBasicCatalogJSON, &catalog); err != nil {
		panic(fmt.Sprintf("agentdir: a2ui/basic_catalog.json: %v", err))
	}
	return catalog
}()

// compileA2UIPrompt writes what the model is told to write UI in A2UI with: the renderer's
// protocol and guidelines, then the catalog it draws with, the basic catalog's components and
// functions that keep names beside the app's own, as the renderer serializes it.
func compileA2UIPrompt(settings UISettings, components []UIPortableComponent, blueprints []UIBlueprint) (string, error) {
	custom := make(map[string]json.RawMessage, len(components))
	for _, comp := range components {
		schema, err := a2uiComponentSchema(comp)
		if err != nil {
			return "", err
		}
		custom[uiPascalName(comp.Name)] = schema
	}

	catalogComponents := make(map[string]json.RawMessage)
	catalogFunctions := make(map[string]json.RawMessage)
	if len(settings.Keep) == 0 {
		maps.Copy(catalogComponents, a2uiBasic.Components)
		maps.Copy(catalogFunctions, a2uiBasic.Functions)
	}
	for _, name := range settings.Keep {
		if schema, ok := a2uiBasic.Components[name]; ok {
			catalogComponents[name] = schema
		} else if schema, ok := a2uiBasic.Functions[name]; ok {
			catalogFunctions[name] = schema
		} else if _, ok := custom[name]; !ok {
			return "", fmt.Errorf("ui.keep names %q, which is neither a component nor a function of the A2UI basic catalog", name)
		}
	}
	maps.Copy(catalogComponents, custom)

	for _, bp := range blueprints {
		if bp.Example == "" {
			continue
		}
		if problems := checkA2UIProgram(extractUIFence(bp.Example, a2uiFence), catalogComponents); len(problems) > 0 {
			return "", fmt.Errorf("blueprint %q example is invalid: %s", bp.Name, strings.Join(problems, "; "))
		}
	}

	document, err := a2uiCatalogDocument(catalogComponents, catalogFunctions)
	if err != nil {
		return "", err
	}

	var b strings.Builder
	writeUIOpening(&b, a2uiFence, settings.VoiceUI)
	b.WriteString(strings.TrimSpace(a2uiWriting))
	b.WriteString("\n\n")
	b.WriteString(a2uiSubsection(a2uiProtocol))
	b.WriteString("\n\n")
	b.WriteString(a2uiSubsection(a2uiBasic.Instructions))
	b.WriteString("\n\n### The catalog `" + a2uiCatalogID + "`\n\n```json\n")
	b.Write(document)
	b.WriteString("\n```\n")

	if len(blueprints) > 0 {
		b.WriteString(uiBlueprintsHeading)
		for _, bp := range blueprints {
			fmt.Fprintf(&b, "- **%s** (when %s): a Column of `%s`", bp.Name, strings.TrimSpace(bp.When), strings.Join(bp.Order, " -> "))
			if len(bp.Rules) > 0 {
				b.WriteString(" — ")
				b.WriteString(strings.Join(bp.Rules, "; "))
			}
			b.WriteByte('\n')
			if bp.Example != "" {
				b.WriteString("\n```" + a2uiFence + "\n")
				b.WriteString(strings.TrimSpace(extractUIFence(bp.Example, a2uiFence)))
				b.WriteString("\n```\n")
			}
		}
	}

	writeUIRules(&b, settings.Rules)

	return strings.TrimSpace(b.String()), nil
}

// a2uiSubsection nests the renderer's text under the section's own heading.
func a2uiSubsection(text string) string {
	lines := strings.Split(strings.TrimSpace(text), "\n")
	for i, line := range lines {
		if rest, ok := strings.CutPrefix(line, "## "); ok {
			lines[i] = "### " + rest
		} else if rest, ok := strings.CutPrefix(line, "# "); ok {
			lines[i] = "### " + rest
		}
	}
	return strings.Join(lines, "\n")
}

// a2uiComponentSchema is an app component as the renderer has it in its catalog
// (PortableComponent in stream-chat-swift): its parameters are its properties, and what its
// events do joins its description, as the component reports them itself.
func a2uiComponentSchema(comp UIPortableComponent) (json.RawMessage, error) {
	properties := map[string]any{"component": map[string]string{"const": uiPascalName(comp.Name)}}
	for key, prop := range comp.Parameters.Properties {
		schema := map[string]any{}
		if prop.Type != "" {
			schema["type"] = prop.Type
		}
		if prop.Format != "" {
			schema["format"] = prop.Format
		}
		if prop.Description != "" {
			schema["description"] = prop.Description
		}
		if len(prop.Enum) > 0 {
			schema["enum"] = prop.Enum
		}
		if len(prop.Items) > 0 {
			// Decoded with its numbers as written, and written again with its keys sorted as the
			// rest of the catalog is.
			var items any
			decoder := json.NewDecoder(bytes.NewReader(prop.Items))
			decoder.UseNumber()
			if err := decoder.Decode(&items); err != nil {
				return nil, fmt.Errorf("component %q property %q items: %w", comp.Name, key, err)
			}
			schema["items"] = items
		}
		properties[key] = schema
	}
	required := append([]string{"component"}, comp.Parameters.Required...)
	slices.Sort(required)

	var notes []string
	for _, text := range append([]string{comp.Description}, eventDescriptions(comp.Events)...) {
		if sentence := a2uiSentence(text); sentence != "" {
			notes = append(notes, sentence)
		}
	}
	return marshalA2UI(map[string]any{
		"description": strings.Join(notes, " "),
		"properties":  properties,
		"required":    required,
		"type":        "object",
	})
}

func eventDescriptions(events []UIPortableEvent) []string {
	described := make([]string, 0, len(events))
	for _, event := range events {
		described = append(described, event.Description)
	}
	return described
}

func a2uiSentence(text string) string {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" || strings.ContainsAny(trimmed[len(trimmed)-1:], ".!?") {
		return trimmed
	}
	return trimmed + "."
}

// a2uiCatalogDocument is the catalog as the renderer serializes it, keys sorted, on one line.
func a2uiCatalogDocument(components, functions map[string]json.RawMessage) ([]byte, error) {
	componentRefs := make([]map[string]string, 0, len(components))
	for _, name := range slices.Sorted(maps.Keys(components)) {
		componentRefs = append(componentRefs, map[string]string{"$ref": "#/components/" + name})
	}
	functionRefs := make([]map[string]string, 0, len(functions))
	for _, name := range slices.Sorted(maps.Keys(functions)) {
		functionRefs = append(functionRefs, map[string]string{"$ref": "#/functions/" + name})
	}
	document := map[string]any{
		"$defs": map[string]any{
			"anyComponent": map[string]any{
				"discriminator": map[string]string{"propertyName": "component"},
				"oneOf":         componentRefs,
			},
			"anyFunction": map[string]any{"oneOf": functionRefs},
		},
		"$id":             a2uiCatalogID,
		"$schema":         a2uiBasic.Schema,
		"catalogId":       a2uiCatalogID,
		"description":     a2uiBasic.Description,
		"protocolVersion": "1.0",
		"title":           a2uiBasic.Title,
	}
	if len(components) > 0 {
		document["components"] = components
	}
	if len(functions) > 0 {
		document["functions"] = functions
	}
	return marshalA2UI(document)
}

// marshalA2UI writes JSON as the renderer does, without escaping <, > and &.
func marshalA2UI(value any) ([]byte, error) {
	var out bytes.Buffer
	encoder := json.NewEncoder(&out)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, fmt.Errorf("writing the A2UI catalog: %w", err)
	}
	return bytes.TrimSuffix(out.Bytes(), []byte("\n")), nil
}

// checkA2UIProgram is what is wrong with the A2UI messages of an example for a renderer with
// catalog, as stream-chat-swift's A2UIProgram finds it: messages it can't read, components it
// doesn't have, and references that lead nowhere. An example teaches the model, so it has to
// render as written.
func checkA2UIProgram(text string, catalog map[string]json.RawMessage) []string {
	var problems []string
	surfaces := map[string]map[string]map[string]any{}
	var order []string
	for _, raw := range a2uiMessages(text) {
		var message map[string]any
		decoder := json.NewDecoder(strings.NewReader(raw))
		decoder.UseNumber()
		if err := decoder.Decode(&message); err != nil {
			problems = append(problems, "this isn't JSON: "+truncate(raw, 80))
			continue
		}
		kind, body, problem := a2uiMessageOf(message)
		if problem != "" {
			problems = append(problems, problem)
			continue
		}
		surfaceID, _ := body["surfaceId"].(string)
		switch kind {
		case "createSurface":
			components, more := a2uiComponentsOf(body, surfaceID)
			problems = append(problems, more...)
			if catalogID, ok := body["catalogId"].(string); ok && catalogID != a2uiCatalogID {
				problems = append(problems, fmt.Sprintf("surface %s names the catalog %s, not %s", surfaceID, catalogID, a2uiCatalogID))
			}
			surfaces[surfaceID] = components
			order = append(order, surfaceID)
		case "updateComponents":
			existing, created := surfaces[surfaceID]
			if !created {
				problems = append(problems, fmt.Sprintf("surface %s gets components before it is created", surfaceID))
				continue
			}
			components, more := a2uiComponentsOf(body, surfaceID)
			problems = append(problems, more...)
			maps.Copy(existing, components)
		case "updateDataModel":
			if _, created := surfaces[surfaceID]; !created {
				problems = append(problems, fmt.Sprintf("surface %s gets data before it is created", surfaceID))
			}
		case "deleteSurface":
		default:
			problems = append(problems, "a reply's UI calls no functions")
		}
	}
	if len(order) == 0 {
		problems = append(problems, "nothing renders: there is no createSurface")
	}
	for _, surfaceID := range order {
		problems = append(problems, a2uiSurfaceProblems(surfaces[surfaceID], surfaceID, catalog)...)
	}
	return problems
}

// a2uiMessages splits text into the JSON objects in it, as the renderer reads a reply's block.
func a2uiMessages(text string) []string {
	var messages []string
	depth, start := 0, 0
	inString, escaped := false, false
	for i := 0; i < len(text); i++ {
		c := text[i]
		if inString {
			switch {
			case escaped:
				escaped = false
			case c == '\\':
				escaped = true
			case c == '"':
				inString = false
			}
			continue
		}
		switch c {
		case '"':
			inString = depth > 0
		case '{':
			if depth == 0 {
				start = i
			}
			depth++
		case '}':
			if depth > 0 {
				depth--
				if depth == 0 {
					messages = append(messages, text[start:i+1])
				}
			}
		}
	}
	return messages
}

func a2uiMessageOf(message map[string]any) (kind string, body map[string]any, problem string) {
	if version, _ := message["version"].(string); version != "1.0" {
		return "", nil, `a message needs "version": "1.0"`
	}
	for key := range message {
		if key == "version" {
			continue
		}
		if !slices.Contains(a2uiMessageKinds, key) || kind != "" {
			return "", nil, fmt.Sprintf("a message has exactly one of %s", strings.Join(a2uiMessageKinds, ", "))
		}
		kind = key
	}
	if kind == "" {
		return "", nil, fmt.Sprintf("a message has exactly one of %s", strings.Join(a2uiMessageKinds, ", "))
	}
	body, ok := message[kind].(map[string]any)
	if !ok {
		return "", nil, fmt.Sprintf("%s is an object", kind)
	}
	if surfaceID, _ := body["surfaceId"].(string); surfaceID == "" && kind != "callRendererFunction" && kind != "agentFunctionResponse" {
		return "", nil, fmt.Sprintf("%s needs a surfaceId", kind)
	}
	return kind, body, ""
}

// a2uiComponentsOf is a message's components by id, and the problems with them.
func a2uiComponentsOf(body map[string]any, surfaceID string) (map[string]map[string]any, []string) {
	listed, _ := body["components"].([]any)
	components := make(map[string]map[string]any, len(listed))
	var problems []string
	var repeated []string
	for _, item := range listed {
		component, _ := item.(map[string]any)
		id, _ := component["id"].(string)
		if _, typed := component["component"].(string); id == "" || !typed {
			problems = append(problems, fmt.Sprintf("surface %s has a component without an id and a component type", surfaceID))
			continue
		}
		if _, seen := components[id]; seen && !slices.Contains(repeated, id) {
			repeated = append(repeated, id)
		}
		components[id] = component
	}
	for _, id := range repeated {
		problems = append(problems, fmt.Sprintf("surface %s defines %s more than once in one message", surfaceID, id))
	}
	return components, problems
}

func a2uiSurfaceProblems(components map[string]map[string]any, surfaceID string, catalog map[string]json.RawMessage) []string {
	var problems []string
	references := map[string][]string{}
	for _, id := range slices.Sorted(maps.Keys(components)) {
		component := components[id]
		kind, _ := component["component"].(string)
		if !a2uiIdentifier.MatchString(id) {
			problems = append(problems, fmt.Sprintf("%s isn't an identifier: use letters, digits and _, not starting with a digit", id))
		}
		raw, known := catalog[kind]
		if !known {
			problems = append(problems, fmt.Sprintf("%s is a %s, which the catalog doesn't have", id, kind))
			continue
		}
		var schema struct {
			Properties map[string]map[string]any `json:"properties"`
			Required   []string                  `json:"required"`
		}
		if err := json.Unmarshal(raw, &schema); err != nil {
			problems = append(problems, fmt.Sprintf("the catalog's %s can't be read: %v", kind, err))
			continue
		}
		for _, name := range schema.Required {
			if _, given := component[name]; !given && name != "component" {
				problems = append(problems, fmt.Sprintf("%s is a %s without its %s", id, kind, name))
			}
		}
		for _, key := range slices.Sorted(maps.Keys(component)) {
			switch key {
			case "id", "component", "catalogId", "checks", "weight":
				continue
			}
			if _, has := schema.Properties[key]; !has {
				problems = append(problems, fmt.Sprintf("%s is a %s, which has no %s", id, kind, key))
			}
		}
		var held []string
		for _, key := range slices.Sorted(maps.Keys(schema.Properties)) {
			held = append(held, a2uiReferences(component[key], schema.Properties[key])...)
		}
		references[id] = held
	}
	for _, id := range slices.Sorted(maps.Keys(references)) {
		for _, child := range references[id] {
			if _, defined := components[child]; !defined {
				problems = append(problems, fmt.Sprintf("%s holds %s, which is not defined", id, child))
			}
		}
	}
	if _, rooted := components["root"]; !rooted {
		return append(problems, fmt.Sprintf("surface %s has no component with the id root", surfaceID))
	}
	reached := map[string]bool{}
	next := []string{"root"}
	for len(next) > 0 {
		id := next[len(next)-1]
		next = next[:len(next)-1]
		if reached[id] {
			continue
		}
		reached[id] = true
		next = append(next, references[id]...)
	}
	for _, id := range slices.Sorted(maps.Keys(components)) {
		if !reached[id] {
			problems = append(problems, fmt.Sprintf("%s is defined but nothing holds it", id))
		}
	}
	return append(problems, a2uiNestingProblems(references, surfaceID)...)
}

// a2uiReferences is the ids of the components value holds, by what schema says it is: a
// ComponentId, a ChildList of ids or a template, or an array or object of them.
func a2uiReferences(value any, schema map[string]any) []string {
	if ref, ok := schema["$ref"].(string); ok {
		if strings.HasSuffix(ref, "/ComponentId") {
			if id, ok := value.(string); ok {
				return []string{id}
			}
			return nil
		}
		if strings.HasSuffix(ref, "/ChildList") {
			switch list := value.(type) {
			case []any:
				ids := make([]string, 0, len(list))
				for _, item := range list {
					id, ok := item.(string)
					if !ok {
						return nil
					}
					ids = append(ids, id)
				}
				return ids
			case map[string]any:
				if id, ok := list["componentId"].(string); ok {
					return []string{id}
				}
			}
			return nil
		}
	}
	if items, ok := schema["items"].(map[string]any); ok {
		if array, ok := value.([]any); ok {
			var ids []string
			for _, item := range array {
				ids = append(ids, a2uiReferences(item, items)...)
			}
			return ids
		}
	}
	if properties, ok := schema["properties"].(map[string]any); ok {
		if object, ok := value.(map[string]any); ok {
			var ids []string
			for _, key := range slices.Sorted(maps.Keys(properties)) {
				if property, ok := properties[key].(map[string]any); ok {
					ids = append(ids, a2uiReferences(object[key], property)...)
				}
			}
			return ids
		}
	}
	return nil
}

// a2uiNestingProblems finds a component that holds itself, which never finishes rendering, and
// a tree deeper than a renderer draws.
func a2uiNestingProblems(references map[string][]string, surfaceID string) []string {
	var problems []string
	depths := map[string]int{}
	visiting := map[string]bool{}
	reportedCycle := false
	var depth func(id string) int
	depth = func(id string) int {
		if known, ok := depths[id]; ok {
			return known
		}
		if visiting[id] {
			if !reportedCycle {
				problems = append(problems, fmt.Sprintf("%s holds itself through its children, so it never finishes rendering", id))
				reportedCycle = true
			}
			return 0
		}
		visiting[id] = true
		deepest := 0
		for _, child := range references[id] {
			if _, holds := references[child]; holds {
				deepest = max(deepest, depth(child))
			}
		}
		delete(visiting, id)
		depths[id] = deepest + 1
		return deepest + 1
	}
	if _, rooted := references["root"]; rooted && depth("root") > a2uiMaxNesting {
		problems = append(problems, fmt.Sprintf("surface %s nests components deeper than %d levels", surfaceID, a2uiMaxNesting))
	}
	for _, id := range slices.Sorted(maps.Keys(references)) {
		depth(id)
	}
	return problems
}

func truncate(text string, limit int) string {
	runes := []rune(text)
	if len(runes) <= limit {
		return text
	}
	return string(runes[:limit])
}
