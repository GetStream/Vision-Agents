package agentdir

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

var (
	uiPortableName = regexp.MustCompile(`^[a-z][a-z0-9_]{0,63}$`)
	uiIdentPattern = regexp.MustCompile(`^\$?[A-Za-z_][A-Za-z0-9_]*$`)
	uiRefPattern   = regexp.MustCompile(`\$?[A-Za-z_][A-Za-z0-9_]*`)
)

// UI is agent.yaml's ui key with the files it names.
type UI struct {
	Settings UISettings
	// Components and Blueprints are the two files as written, empty without one.
	Components string
	Blueprints string
	// Instructions is what the three compile to: what teaches the model to answer with UI.
	Instructions string
}

// UISettings is agent.yaml's ui key: the catalog an agent draws with, its own components and
// its layout recipes.
type UISettings struct {
	// Format is openui, the default, or a2ui.
	Format string
	// Keep restricts the built-in catalog to the named components. In OpenUI their required
	// child types come along; in A2UI it names the basic catalog's components and functions,
	// and nothing else is offered.
	Keep []string
	// Components is the path of a portable UIComponents.json. Empty is ui/components.json
	// when the directory has one.
	Components string
	// Blueprints is the path of a YAML list of layout recipes. Empty is ui/blueprints.yaml
	// when the directory has one.
	Blueprints string
	// PatchMode has a turn that refines the last UI send only the statements that changed.
	PatchMode bool
	// VoiceUI streams UI beside a spoken answer on a call.
	VoiceUI bool
	// Rules are brand or interaction rules appended to what the model is told.
	Rules []string
}

func (r *reader) uiSettings(node *yaml.Node) *UISettings {
	settings := &UISettings{}
	var format, patch *yaml.Node
	r.mapping(node, "ui",
		field{"format", func(value *yaml.Node) { format = value; r.text(&settings.Format)(value) }},
		field{"keep", r.list(&settings.Keep)},
		field{"components", r.text(&settings.Components)},
		field{"blueprints", r.text(&settings.Blueprints)},
		field{"patch_mode", func(value *yaml.Node) { patch = value; r.boolean(&settings.PatchMode)(value) }},
		field{"voice_ui", r.boolean(&settings.VoiceUI)},
		field{"rules", r.list(&settings.Rules)})
	if settings.Format != "" && settings.Format != "openui" && settings.Format != "a2ui" {
		r.fail(format, "ui.format must be openui or a2ui, not %q", settings.Format)
	}
	// The router and the apps merge only OpenUI patches; an A2UI reply writes a new surface.
	if settings.Format == "a2ui" && settings.PatchMode {
		r.fail(patch, "ui.patch_mode is for openui: an A2UI reply writes its UI on a new surface")
	}
	return settings
}

// UIPortableComponent is one custom component in ui/components.json.
type UIPortableComponent struct {
	Name        string               `json:"name"`
	Description string               `json:"description"`
	Parameters  UIPortableParameters `json:"parameters"`
	Events      []UIPortableEvent    `json:"events"`
	Slots       []string             `json:"slots,omitempty"`
}

type UIPortableParameters struct {
	Type       string                        `json:"type"`
	Properties map[string]UIPortableProperty `json:"properties"`
	Required   []string                      `json:"required"`
	Order      []string                      `json:"x-order"`
}

type UIPortableProperty struct {
	Type        string   `json:"type"`
	Format      string   `json:"format,omitempty"`
	Description string   `json:"description,omitempty"`
	Enum        []string `json:"enum,omitempty"`
	// Items is an array's item schema, which the apps read and the prompt leaves out: the
	// signature says only `array`.
	Items json.RawMessage `json:"items,omitempty"`
}

type UIPortableEvent struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// UIBlueprint is a canonical component layout recipe in ui/blueprints.yaml.
type UIBlueprint struct {
	Name    string
	When    string
	Order   []string
	Rules   []string
	Example string
}

type uiBuiltin struct {
	Name      string
	Signature string
	MaxArgs   int
	Deps      []string
}

var uiBuiltins = []uiBuiltin{
	{Name: "Card", Signature: "Card(children: Component[], sources?: {title: string, sourceName: string, url?: string}[]) — Root vertical card container.", MaxArgs: 2},
	{Name: "CardHeader", Signature: "CardHeader(title?: string, subtitle?: string) — Primary card heading and subtitle.", MaxArgs: 2},
	{Name: "InlineHeader", Signature: "InlineHeader(heading: string, description?: string) — Compact section label.", MaxArgs: 2},
	{Name: "TextContent", Signature: `TextContent(text: string, size?: "small" | "default" | "large" | "small-heavy" | "large-heavy") — Markdown-capable text block.`, MaxArgs: 2},
	{Name: "Callout", Signature: `Callout(variant: "info" | "warning" | "error" | "success" | "neutral", title: string, description: string) — Highlighted status callout.`, MaxArgs: 4},
	{Name: "Separator", Signature: `Separator(orientation?: "horizontal" | "vertical") — Hairline divider.`, MaxArgs: 2},
	{Name: "TagBlock", Signature: `TagBlock(tags: string[], size?: "sm" | "md" | "lg") — Compact pill row (at most 3 tags).`, MaxArgs: 2},
	{Name: "EntityList", Signature: `EntityList(rows: {left: string, right: string}[], size?: "small" | "default") — Key/value summary rows.`, MaxArgs: 4},
	{Name: "Table", Signature: "Table(columns: Col[]) — Column-oriented read-only data table.", MaxArgs: 1, Deps: []string{"Col"}},
	{Name: "Col", Signature: `Col(label: string, data: (string | number)[], type?: "string" | "number") — Table column.`, MaxArgs: 3},
	{Name: "BarChart", Signature: `BarChart(labels: string[], series: Series[], variant?: "grouped" | "stacked", xLabel?: string, yLabel?: string) — Categorical bar chart.`, MaxArgs: 6, Deps: []string{"Series"}},
	{Name: "Series", Signature: "Series(category: string, values: number[]) — Named chart data series.", MaxArgs: 2},
	{Name: "PieChart", Signature: `PieChart(labels: string[], values: number[], variant?: "pie" | "donut", appearance?: "circular" | "semiCircular") — Proportional or semicircular gauge chart.`, MaxArgs: 4},
	{Name: "Tabs", Signature: "Tabs(items: TabItem[]) — Multi-view tab container.", MaxArgs: 1, Deps: []string{"TabItem"}},
	{Name: "TabItem", Signature: "TabItem(value: string, trigger: string, content: Component[]) — One tab panel.", MaxArgs: 3},
	{Name: "Steps", Signature: "Steps(items: StepsItem[]) — Sequential timeline steps.", MaxArgs: 1, Deps: []string{"StepsItem"}},
	{Name: "StepsItem", Signature: "StepsItem(title: string, details: string) — Single step entry.", MaxArgs: 2},
	{Name: "ListBlock", Signature: `ListBlock(items: ListItem[], variant?: "number" | "image", size?: "default" | "small") — Structured list.`, MaxArgs: 3, Deps: []string{"ListItem"}},
	{Name: "ListItem", Signature: "ListItem(title: string, subtitle?: string, image?: {src: string, alt: string}, actionLabel?: string, action?: ActionExpression) — Clickable list row.", MaxArgs: 5},
	{Name: "Form", Signature: "Form(name: string, buttons: Buttons, fields?: FormControl[]) — Interactive form with explicit Buttons.", MaxArgs: 3, Deps: []string{"FormControl", "Buttons", "Button"}},
	{Name: "FormControl", Signature: "FormControl(label: string, input: Component, hint?: string) — Labeled form field.", MaxArgs: 3},
	{Name: "Select", Signature: "Select(name: string, items: SelectItem[], placeholder?: string, rules?: object, value?: $binding) — Dropdown selector.", MaxArgs: 6, Deps: []string{"SelectItem"}},
	{Name: "SelectItem", Signature: "SelectItem(value: string, label: string) — Option inside Select.", MaxArgs: 2},
	{Name: "Chips", Signature: `Chips(name: string, type?: "single" | "multiple", items?: ChipItem[], rules?: object, defaultValue?: string | string[]) — Selectable pill group.`, MaxArgs: 5, Deps: []string{"ChipItem"}},
	{Name: "ChipItem", Signature: "ChipItem(value: string, label: string) — Single chip option.", MaxArgs: 4},
	{Name: "OptionCards", Signature: `OptionCards(name: string, type?: "single" | "multiple", items?: OptionCard[], rules?: object, defaultValue?: string | string[]) — Selectable card grid.`, MaxArgs: 5, Deps: []string{"OptionCard"}},
	{Name: "OptionCard", Signature: "OptionCard(value: string, title: string, subtitle?: string) — Single option card.", MaxArgs: 5},
	{Name: "DatePicker", Signature: `DatePicker(name: string, mode?: "single" | "range", rules?: object, value?: $binding) — Date selector.`, MaxArgs: 4},
	{Name: "Slider", Signature: `Slider(name: string, variant: "continuous" | "discrete", min: number, max: number, step?: number, defaultValue?: number[]) — Numeric range slider.`, MaxArgs: 9},
	{Name: "Buttons", Signature: `Buttons(buttons: Button[], direction?: "row" | "column") — Action button group.`, MaxArgs: 2, Deps: []string{"Button"}},
	{Name: "Button", Signature: `Button(label: string, action?: ActionExpression, variant?: "primary" | "secondary" | "tertiary") — Interactive trigger button.`, MaxArgs: 5},
	{Name: "FollowUpBlock", Signature: "FollowUpBlock(items: FollowUpItem[]) — Suggested next turns at the bottom of a card.", MaxArgs: 1, Deps: []string{"FollowUpItem"}},
	{Name: "FollowUpItem", Signature: "FollowUpItem(text: string) — Single tappable follow-up prompt.", MaxArgs: 1},
}

// readUI reads the files agent.yaml's ui key names and compiles the three into what the model
// is told. A problem with what they add up to, such as an example that uses a component the
// catalog lacks, is reported at the ui key.
func readUI(settings UISettings, declaration *reader, text func(string) (string, bool), problems *Problems) *UI {
	ui := &UI{Settings: settings}
	read := func(named, fallback string) (file, content string) {
		file = named
		if file == "" {
			file = fallback
		}
		content, found := text(file)
		switch {
		case !found && named == "":
			return "", ""
		case !found:
			line, column := declaration.locate("ui")
			*problems = append(*problems, Problem{File: AgentFile, Line: line, Column: column,
				Message: fmt.Sprintf("ui names %s, which is not a text file in the directory", file)})
			return "", ""
		case len(content) > maxUIFileBytes:
			*problems = append(*problems, Problem{File: file,
				Message: fmt.Sprintf("holds %d bytes, and a UI file holds at most %d", len(content), maxUIFileBytes)})
			return "", ""
		}
		return file, content
	}
	before := len(*problems)

	var components []UIPortableComponent
	if file, content := read(settings.Components, UIComponentsFile); file != "" {
		ui.Components = content
		decoder := json.NewDecoder(strings.NewReader(content))
		decoder.DisallowUnknownFields()
		err := decoder.Decode(&components)
		if err == nil {
			err = validateUIComponents(components, settings.Format)
		}
		if err != nil {
			line, column := jsonPosition(content, err)
			*problems = append(*problems, Problem{File: file, Line: line, Column: column, Message: err.Error()})
		}
	}
	var blueprints []UIBlueprint
	if file, content := read(settings.Blueprints, UIBlueprintsFile); file != "" {
		ui.Blueprints = content
		blueprints = (&reader{file: file, problems: problems}).blueprints(content)
	}
	if len(*problems) > before {
		return ui
	}

	instructions, err := compileUIPrompt(settings, components, blueprints)
	if err != nil {
		line, column := declaration.locate("ui")
		*problems = append(*problems, Problem{File: AgentFile, Line: line, Column: column, Message: err.Error()})
	}
	ui.Instructions = instructions
	return ui
}

// jsonPosition is where a JSON file stopped reading, or its first character for a problem
// that is not about its syntax.
func jsonPosition(content string, err error) (line, column int) {
	var syntax *json.SyntaxError
	var mistyped *json.UnmarshalTypeError
	offset := 0
	switch {
	case errors.As(err, &syntax):
		offset = int(syntax.Offset)
	case errors.As(err, &mistyped):
		offset = int(mistyped.Offset)
	}
	read := content[:min(offset, len(content))]
	return 1 + strings.Count(read, "\n"), 1 + len(read) - (strings.LastIndex(read, "\n") + 1)
}

// blueprints reads ui/blueprints.yaml: a list of layout recipes.
func (r *reader) blueprints(content string) []UIBlueprint {
	root := r.document(content)
	if root == nil {
		return nil
	}
	var blueprints []UIBlueprint
	seen := map[string]int{}
	r.each(root, func(item *yaml.Node) {
		if null(item) {
			return
		}
		var blueprint UIBlueprint
		name := item
		if !r.mapping(item, "a blueprint",
			field{"name", func(value *yaml.Node) { name = value; r.text(&blueprint.Name)(value) }},
			field{"when", r.text(&blueprint.When)},
			field{"order", r.list(&blueprint.Order)},
			field{"rules", r.list(&blueprint.Rules)},
			field{"example", r.text(&blueprint.Example)}) {
			return
		}
		switch {
		case strings.TrimSpace(blueprint.Name) == "":
			r.fail(item, "a blueprint needs a name")
		case strings.TrimSpace(blueprint.When) == "":
			r.fail(item, "blueprint %q needs a when condition", blueprint.Name)
		case len(blueprint.Order) == 0:
			r.fail(item, "blueprint %q needs an order list", blueprint.Name)
		}
		if line, declared := seen[blueprint.Name]; declared {
			r.fail(name, "blueprint %q was already declared on line %d", blueprint.Name, line)
		}
		seen[blueprint.Name] = name.Line
		blueprints = append(blueprints, blueprint)
	})
	return blueprints
}

func validateUIComponents(components []UIPortableComponent, format string) error {
	builtins := make(map[string]struct{}, len(uiBuiltins))
	if format == "a2ui" {
		for name := range a2uiBasic.Components {
			builtins[name] = struct{}{}
		}
		// A2UI reserves Surface for the surface itself.
		builtins["Surface"] = struct{}{}
	} else {
		for _, b := range uiBuiltins {
			builtins[b.Name] = struct{}{}
		}
	}
	seen := make(map[string]struct{}, len(components))
	for _, comp := range components {
		if !uiPortableName.MatchString(comp.Name) {
			return fmt.Errorf("component %q must be snake_case", comp.Name)
		}
		pascal := uiPascalName(comp.Name)
		if _, exists := builtins[pascal]; exists {
			return fmt.Errorf("component %q (%s) collides with a built-in component", comp.Name, pascal)
		}
		if _, exists := seen[pascal]; exists {
			return fmt.Errorf("duplicate component %q", comp.Name)
		}
		seen[pascal] = struct{}{}
		if strings.TrimSpace(comp.Description) == "" {
			return fmt.Errorf("component %q needs a description", comp.Name)
		}
		if len(comp.Parameters.Properties) > 0 && len(comp.Parameters.Order) != len(comp.Parameters.Properties) {
			return fmt.Errorf("component %q parameters.x-order must list all %d properties",
				comp.Name, len(comp.Parameters.Properties))
		}
		for _, key := range comp.Parameters.Order {
			if _, ok := comp.Parameters.Properties[key]; !ok {
				return fmt.Errorf("component %q x-order names unknown property %q", comp.Name, key)
			}
		}
		// An A2UI property is a key of the component's JSON object, which the renderers refuse
		// unless it is an identifier.
		if format == "a2ui" {
			for _, key := range comp.Parameters.Order {
				if !a2uiIdentifier.MatchString(key) {
					return fmt.Errorf("component %q property %q must be an identifier: letters, digits and _, not starting with a digit", comp.Name, key)
				}
			}
		}
	}
	return nil
}

func compileUIPrompt(settings UISettings, components []UIPortableComponent, blueprints []UIBlueprint) (string, error) {
	if settings.Format == "a2ui" {
		return compileA2UIPrompt(settings, components, blueprints)
	}
	byName := make(map[string]uiBuiltin, len(uiBuiltins))
	for _, b := range uiBuiltins {
		byName[b.Name] = b
	}
	customArity := make(map[string]int, len(components))
	for _, c := range components {
		customArity[uiPascalName(c.Name)] = len(c.Parameters.Order) + len(c.Slots)
	}

	included := map[string]bool{"Card": true}
	if len(settings.Keep) == 0 {
		for _, b := range uiBuiltins {
			included[b.Name] = true
		}
	} else {
		var queue []string
		for _, name := range settings.Keep {
			if _, isCustom := customArity[name]; isCustom {
				continue
			}
			if _, ok := byName[name]; !ok {
				return "", fmt.Errorf("ui.keep names unknown component %q", name)
			}
			if !included[name] {
				included[name] = true
				queue = append(queue, name)
			}
		}
		for len(queue) > 0 {
			curr := queue[0]
			queue = queue[1:]
			for _, dep := range byName[curr].Deps {
				if !included[dep] {
					included[dep] = true
					queue = append(queue, dep)
				}
			}
		}
	}

	arity := make(map[string]int, len(included)+len(customArity))
	var selected []uiBuiltin
	for _, b := range uiBuiltins {
		if included[b.Name] {
			selected = append(selected, b)
			arity[b.Name] = b.MaxArgs
		}
	}
	for k, v := range customArity {
		arity[k] = v
	}

	for _, bp := range blueprints {
		if bp.Example != "" {
			if problems := checkOpenUIProgram(extractUIFence(bp.Example, "openui-lang"), arity); len(problems) > 0 {
				return "", fmt.Errorf("blueprint %q example is invalid: %s", bp.Name, strings.Join(problems, "; "))
			}
		}
	}

	var b strings.Builder
	writeUIOpening(&b, "openui-lang", settings.VoiceUI)

	b.WriteString("### Syntax & Streaming Invariants\n")
	b.WriteString("1. Put `root = Card([...])` on the FIRST line so the card shell streams immediately.\n")
	b.WriteString("2. Every statement is `identifier = Expression` on its own line; arguments are strictly POSITIONAL (`TypeName(arg1, arg2)` — never `key: value`).\n")
	b.WriteString("3. Every defined identifier other than `root` and `$state` variables must be referenced from `root`.\n")
	b.WriteString("4. Never fabricate image URLs; use custom components or icons when no verified URL is in context.\n")
	if settings.PatchMode {
		b.WriteString("5. When refining an existing UI from the immediately preceding turn (e.g. updating a time slot, party size, or filter), emit only the changed statements inside a ```openui-patch fenced block (`$party = 4`, or `oldSection = null` to remove).\n")
	}

	b.WriteString("\n### Allowed Component Signatures\n")
	for _, def := range selected {
		b.WriteString("- ")
		b.WriteString(def.Signature)
		b.WriteByte('\n')
	}
	for _, comp := range components {
		b.WriteString("- ")
		b.WriteString(formatPortableSignature(comp))
		b.WriteByte('\n')
	}

	if len(blueprints) > 0 {
		b.WriteString(uiBlueprintsHeading)
		for _, bp := range blueprints {
			b.WriteString(fmt.Sprintf("- **%s** (when %s): `Card([%s])`", bp.Name, strings.TrimSpace(bp.When), strings.Join(bp.Order, " -> ")))
			if len(bp.Rules) > 0 {
				b.WriteString(" — ")
				b.WriteString(strings.Join(bp.Rules, "; "))
			}
			b.WriteByte('\n')
			if bp.Example != "" {
				b.WriteString("\n```openui-lang\n")
				b.WriteString(strings.TrimSpace(extractUIFence(bp.Example, "openui-lang")))
				b.WriteString("\n```\n")
			}
		}
	}

	writeUIRules(&b, settings.Rules)

	return strings.TrimSpace(b.String()), nil
}

const uiBlueprintsHeading = "\n### Canonical Layout Blueprints (Predictability Rules)\n" +
	"Match similar prompts to the exact component sequence below so repeated asks produce a consistent layout:\n"

// writeUIOpening starts the section with when to answer with UI, in a block fenced with fence.
func writeUIOpening(b *strings.Builder, fence string, voiceUI bool) {
	b.WriteString("## Interactive UI Generation\n\n")
	if voiceUI {
		b.WriteString("When an answer benefits from interactive UI, first speak or write one concise sentence summarizing the key takeaway, then emit the UI in a single ```" + fence + " fenced block. Never read component names or UI syntax aloud.\n\n")
	} else {
		b.WriteString("When an answer benefits from interactive UI, first write one short sentence summarizing what the card shows, then write the whole UI in one ```" + fence + " fenced block. On a voice call without a screen, speak concisely with no UI block.\n\n")
	}
}

func writeUIRules(b *strings.Builder, rules []string) {
	if len(rules) == 0 {
		return
	}
	b.WriteString("\n### Brand & Design Rules\n")
	for _, rule := range rules {
		b.WriteString("- ")
		b.WriteString(strings.TrimSpace(rule))
		b.WriteByte('\n')
	}
}

func uiPascalName(snake string) string {
	var out strings.Builder
	for part := range strings.SplitSeq(snake, "_") {
		if part != "" {
			out.WriteString(strings.ToUpper(part[:1]) + part[1:])
		}
	}
	return out.String()
}

func formatPortableSignature(comp UIPortableComponent) string {
	required := make(map[string]bool, len(comp.Parameters.Required))
	for _, r := range comp.Parameters.Required {
		required[r] = true
	}
	var params []string
	for _, key := range comp.Parameters.Order {
		prop := comp.Parameters.Properties[key]
		opt := "?"
		if required[key] {
			opt = ""
		}
		typ := prop.Type
		if len(prop.Enum) > 0 {
			quoted := make([]string, len(prop.Enum))
			for i, e := range prop.Enum {
				quoted[i] = strconv.Quote(e)
			}
			typ = strings.Join(quoted, " | ")
		}
		params = append(params, fmt.Sprintf("%s%s: %s", key, opt, typ))
	}
	for _, slot := range comp.Slots {
		params = append(params, fmt.Sprintf("%s?: Component[]", slot))
	}
	details := []string{strings.TrimSpace(comp.Description)}
	for _, ev := range comp.Events {
		if strings.TrimSpace(ev.Description) != "" {
			details = append(details, strings.TrimSpace(ev.Description))
		}
	}
	return fmt.Sprintf("%s(%s) — %s", uiPascalName(comp.Name), strings.Join(params, ", "), strings.Join(details, " "))
}

// extractUIFence is the program in an example's block fenced with fence, or the example itself
// when it has none.
func extractUIFence(example, fence string) string {
	if _, after, ok := strings.Cut(example, "```"+fence+"\n"); ok {
		if body, _, closed := strings.Cut(after, "\n```"); closed {
			return body
		}
	}
	return example
}

func checkOpenUIProgram(program string, knownArity map[string]int) []string {
	trimmed := strings.TrimSpace(program)
	if trimmed == "" {
		return []string{"program is empty"}
	}
	var stmts []string
	var current strings.Builder
	var quote byte
	escaped := false
	depth := 0
	for line := range strings.SplitSeq(trimmed, "\n") {
		if quote == 0 && depth == 0 {
			clean := strings.TrimSpace(line)
			if clean == "" || strings.HasPrefix(clean, "//") || strings.HasPrefix(clean, "#") {
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
				depth--
				if depth < 0 {
					return []string{"unmatched closing bracket"}
				}
			}
		}
		if quote == 0 && depth == 0 {
			if s := strings.TrimSpace(current.String()); s != "" {
				stmts = append(stmts, s)
			}
			current.Reset()
		}
	}
	if quote != 0 || depth != 0 {
		return []string{"unclosed string or bracket"}
	}

	defined := map[string]string{}
	var order []string
	for _, stmt := range stmts {
		lhs, rhs, ok := splitUIAssign(stmt)
		if !ok {
			return []string{fmt.Sprintf("invalid statement %q", stmt)}
		}
		defined[lhs] = rhs
		order = append(order, lhs)
	}
	if _, ok := defined["root"]; !ok {
		return []string{"root is not defined"}
	}
	var problems []string
	used := map[string]bool{"root": true}
	for _, lhs := range order {
		stripped := stripUIStrings(defined[lhs])
		for _, span := range uiRefPattern.FindAllStringIndex(stripped, -1) {
			tok := stripped[span[0]:span[1]]
			if span[0] > 0 && (stripped[span[0]-1] == '@' || stripped[span[0]-1] == '.') {
				continue
			}
			if strings.HasPrefix(strings.TrimLeft(stripped[span[1]:], " \t"), ":") {
				continue
			}
			if slices.Contains([]string{"true", "false", "null", "Action", "Query", "Mutation"}, tok) {
				continue
			}
			if span[1] < len(stripped) && stripped[span[1]] == '(' {
				if maxArgs, ok := knownArity[tok]; ok {
					if count := countUIArgs(stripped[span[1]:]); count > maxArgs {
						problems = append(problems, fmt.Sprintf("%s passed %d arguments, max is %d", tok, count, maxArgs))
					}
				} else {
					problems = append(problems, fmt.Sprintf("unknown component %s", tok))
				}
				continue
			}
			if _, ok := defined[tok]; ok {
				used[tok] = true
			} else {
				problems = append(problems, fmt.Sprintf("%s is referenced but not defined", tok))
			}
		}
	}
	for _, lhs := range order {
		if lhs != "root" && !strings.HasPrefix(lhs, "$") && !used[lhs] {
			problems = append(problems, fmt.Sprintf("%s is defined but never referenced", lhs))
		}
	}
	return problems
}

func splitUIAssign(stmt string) (string, string, bool) {
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
		if !uiIdentPattern.MatchString(lhs) {
			return "", "", false
		}
		return lhs, strings.TrimSpace(stmt[i+1:]), true
	}
	return "", "", false
}

func stripUIStrings(s string) string {
	var out strings.Builder
	var quote byte
	escaped := false
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if escaped {
			escaped = false
			continue
		}
		if quote != 0 {
			if ch == '\\' {
				escaped = true
			} else if ch == quote {
				quote = 0
				out.WriteByte('"')
			}
			continue
		}
		if ch == '"' || ch == '\'' {
			quote = ch
			out.WriteByte('"')
			continue
		}
		out.WriteByte(ch)
	}
	return out.String()
}

func countUIArgs(callSuffix string) int {
	if callSuffix == "" || callSuffix[0] != '(' {
		return 0
	}
	depth := 0
	commas := 0
	hasContent := false
	for i := 0; i < len(callSuffix); i++ {
		ch := callSuffix[i]
		switch ch {
		case '(', '[', '{':
			if depth == 1 {
				hasContent = true
			}
			depth++
		case ')', ']', '}':
			depth--
			if depth == 0 {
				if !hasContent {
					return 0
				}
				return commas + 1
			}
		case ',':
			if depth == 1 {
				commas++
			}
		default:
			if depth == 1 && ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n' {
				hasContent = true
			}
		}
	}
	return 0
}
