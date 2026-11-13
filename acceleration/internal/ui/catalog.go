// Package ui compiles deterministic, prefix-cache-stable generative UI prompts from a
// pruned component allowlist, portable custom components, and canonical layout blueprints,
// and validates OpenUI Lang programs before they teach or reach a client.
package ui

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

const (
	FormatOpenUI = "openui"
	FormatA2UI   = "a2ui"
)

var (
	portableNamePattern = regexp.MustCompile(`^[a-z][a-z0-9_]{0,63}$`)
	identPattern        = regexp.MustCompile(`^\$?[A-Za-z_][A-Za-z0-9_]*$`)
	refCallPattern      = regexp.MustCompile(`\$?[A-Za-z_][A-Za-z0-9_]*`)
)

// Spec describes an agent's generative UI design system, custom components, and domain
// layout blueprints.
type Spec struct {
	Format     string
	Keep       []string
	Components []PortableComponent
	Blueprints []Blueprint
	Rules      []string
	PatchMode  bool
	VoiceUI    bool
}

// PortableComponent is the cross-SDK JSON definition shared across Web, Flutter, iOS, and
// Android (UIComponents.json).
type PortableComponent struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Parameters  PortableParameters `json:"parameters"`
	Events      []PortableEvent    `json:"events"`
	Slots       []string           `json:"slots,omitempty"`
}

type PortableParameters struct {
	Type       string                      `json:"type"`
	Properties map[string]PortableProperty `json:"properties"`
	Required   []string                    `json:"required"`
	Order      []string                    `json:"x-order"`
}

type PortableProperty struct {
	Type        string   `json:"type"`
	Format      string   `json:"format,omitempty"`
	Description string   `json:"description,omitempty"`
	Enum        []string `json:"enum,omitempty"`
}

type PortableEvent struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// Blueprint is a canonical layout recipe so similar user intents yield a consistent
// component hierarchy across turns and models.
type Blueprint struct {
	Name    string   `json:"name" yaml:"name"`
	When    string   `json:"when" yaml:"when"`
	Order   []string `json:"order" yaml:"order"`
	Rules   []string `json:"rules,omitempty" yaml:"rules"`
	Example string   `json:"example,omitempty" yaml:"example"`
}

type builtinDef struct {
	Name      string
	Group     string
	Signature string
	MaxArgs   int
	Deps      []string
}

var builtinCatalog = []builtinDef{
	{Name: "Card", Group: "Layout", Signature: "Card(children: Component[], sources?: {title: string, sourceName: string, url?: string}[]) — Root vertical card container.", MaxArgs: 2},
	{Name: "CardHeader", Group: "Content", Signature: "CardHeader(title?: string, subtitle?: string) — Primary card heading and subtitle.", MaxArgs: 2},
	{Name: "InlineHeader", Group: "Content", Signature: "InlineHeader(heading: string, description?: string) — Compact section label.", MaxArgs: 2},
	{Name: "TextContent", Group: "Content", Signature: `TextContent(text: string, size?: "small" | "default" | "large" | "small-heavy" | "large-heavy") — Markdown-capable text block.`, MaxArgs: 2},
	{Name: "Callout", Group: "Content", Signature: `Callout(variant: "info" | "warning" | "error" | "success" | "neutral", title: string, description: string) — Highlighted status callout.`, MaxArgs: 4},
	{Name: "Separator", Group: "Content", Signature: `Separator(orientation?: "horizontal" | "vertical") — Hairline divider.`, MaxArgs: 2},
	{Name: "TagBlock", Group: "Data", Signature: `TagBlock(tags: string[], size?: "sm" | "md" | "lg") — Compact pill row (at most 3 tags).`, MaxArgs: 2},
	{Name: "EntityList", Group: "Data", Signature: `EntityList(rows: {left: string, right: string}[], size?: "small" | "default") — Key/value summary rows.`, MaxArgs: 4},
	{Name: "Table", Group: "Data", Signature: "Table(columns: Col[]) — Column-oriented read-only data table.", MaxArgs: 1, Deps: []string{"Col"}},
	{Name: "Col", Group: "Data", Signature: `Col(label: string, data: (string | number)[], type?: "string" | "number") — Table column.`, MaxArgs: 3},
	{Name: "BarChart", Group: "Charts", Signature: `BarChart(labels: string[], series: Series[], variant?: "grouped" | "stacked", xLabel?: string, yLabel?: string) — Categorical bar chart.`, MaxArgs: 6, Deps: []string{"Series"}},
	{Name: "Series", Group: "Charts", Signature: "Series(category: string, values: number[]) — Named chart data series.", MaxArgs: 2},
	{Name: "PieChart", Group: "Charts", Signature: `PieChart(labels: string[], values: number[], variant?: "pie" | "donut", appearance?: "circular" | "semiCircular") — Proportional or semicircular gauge chart.`, MaxArgs: 4},
	{Name: "Tabs", Group: "Layout", Signature: "Tabs(items: TabItem[]) — Multi-view tab container.", MaxArgs: 1, Deps: []string{"TabItem"}},
	{Name: "TabItem", Group: "Layout", Signature: "TabItem(value: string, trigger: string, content: Component[]) — One tab panel.", MaxArgs: 3},
	{Name: "Steps", Group: "Layout", Signature: "Steps(items: StepsItem[]) — Sequential timeline steps.", MaxArgs: 1, Deps: []string{"StepsItem"}},
	{Name: "StepsItem", Group: "Layout", Signature: "StepsItem(title: string, details: string) — Single step entry.", MaxArgs: 2},
	{Name: "ListBlock", Group: "Lists", Signature: `ListBlock(items: ListItem[], variant?: "number" | "image", size?: "default" | "small") — Structured list.`, MaxArgs: 3, Deps: []string{"ListItem"}},
	{Name: "ListItem", Group: "Lists", Signature: "ListItem(title: string, subtitle?: string, image?: {src: string, alt: string}, actionLabel?: string, action?: ActionExpression) — Clickable list row.", MaxArgs: 5},
	{Name: "Form", Group: "Forms", Signature: "Form(name: string, buttons: Buttons, fields?: FormControl[]) — Interactive form with explicit Buttons.", MaxArgs: 3, Deps: []string{"FormControl", "Buttons", "Button"}},
	{Name: "FormControl", Group: "Forms", Signature: "FormControl(label: string, input: Component, hint?: string) — Labeled form field.", MaxArgs: 3},
	{Name: "Select", Group: "Forms", Signature: "Select(name: string, items: SelectItem[], placeholder?: string, rules?: object, value?: $binding) — Dropdown selector.", MaxArgs: 6, Deps: []string{"SelectItem"}},
	{Name: "SelectItem", Group: "Forms", Signature: "SelectItem(value: string, label: string) — Option inside Select.", MaxArgs: 2},
	{Name: "Chips", Group: "Forms", Signature: `Chips(name: string, type?: "single" | "multiple", items?: ChipItem[], rules?: object, defaultValue?: string | string[]) — Selectable pill group.`, MaxArgs: 5, Deps: []string{"ChipItem"}},
	{Name: "ChipItem", Group: "Forms", Signature: "ChipItem(value: string, label: string) — Single chip option.", MaxArgs: 4},
	{Name: "OptionCards", Group: "Forms", Signature: `OptionCards(name: string, type?: "single" | "multiple", items?: OptionCard[], rules?: object, defaultValue?: string | string[]) — Selectable card grid.`, MaxArgs: 5, Deps: []string{"OptionCard"}},
	{Name: "OptionCard", Group: "Forms", Signature: "OptionCard(value: string, title: string, subtitle?: string) — Single option card.", MaxArgs: 5},
	{Name: "DatePicker", Group: "Forms", Signature: `DatePicker(name: string, mode?: "single" | "range", rules?: object, value?: $binding) — Date selector.`, MaxArgs: 4},
	{Name: "Slider", Group: "Forms", Signature: `Slider(name: string, variant: "continuous" | "discrete", min: number, max: number, step?: number, defaultValue?: number[]) — Numeric range slider.`, MaxArgs: 9},
	{Name: "Buttons", Group: "Actions", Signature: `Buttons(buttons: Button[], direction?: "row" | "column") — Action button group.`, MaxArgs: 2, Deps: []string{"Button"}},
	{Name: "Button", Group: "Actions", Signature: `Button(label: string, action?: ActionExpression, variant?: "primary" | "secondary" | "tertiary") — Interactive trigger button.`, MaxArgs: 5},
	{Name: "FollowUpBlock", Group: "Actions", Signature: "FollowUpBlock(items: FollowUpItem[]) — Suggested next turns at the bottom of a card.", MaxArgs: 1, Deps: []string{"FollowUpItem"}},
	{Name: "FollowUpItem", Group: "Actions", Signature: "FollowUpItem(text: string) — Single tappable follow-up prompt.", MaxArgs: 1},
}

// PascalName converts a snake_case portable component name (e.g. restaurant_card) to its
// OpenUI / A2UI constructor identifier (RestaurantCard).
func PascalName(snake string) string {
	parts := strings.Split(snake, "_")
	var out strings.Builder
	for _, part := range parts {
		if part == "" {
			continue
		}
		out.WriteString(strings.ToUpper(part[:1]) + part[1:])
	}
	return out.String()
}

// ParsePortableComponents decodes and validates a UIComponents.json byte slice.
func ParsePortableComponents(raw []byte) ([]PortableComponent, error) {
	var list []PortableComponent
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&list); err != nil {
		return nil, fmt.Errorf("ui: invalid components json: %w", err)
	}
	if err := ValidatePortableComponents(list); err != nil {
		return nil, err
	}
	return list, nil
}

// ValidatePortableComponents checks naming,builtin collisions, and positional parameter
// ordering on custom components.
func ValidatePortableComponents(components []PortableComponent) error {
	builtins := make(map[string]struct{}, len(builtinCatalog))
	for _, b := range builtinCatalog {
		builtins[b.Name] = struct{}{}
	}
	seen := make(map[string]struct{}, len(components))
	for _, comp := range components {
		if !portableNamePattern.MatchString(comp.Name) {
			return fmt.Errorf("ui: component %q must be snake_case (a-z, 0-9, _)", comp.Name)
		}
		pascal := PascalName(comp.Name)
		if _, exists := builtins[pascal]; exists {
			return fmt.Errorf("ui: component %q (%s) collides with a built-in component", comp.Name, pascal)
		}
		if _, exists := seen[pascal]; exists {
			return fmt.Errorf("ui: duplicate component %q", comp.Name)
		}
		seen[pascal] = struct{}{}
		if strings.TrimSpace(comp.Description) == "" {
			return fmt.Errorf("ui: component %q needs a description", comp.Name)
		}
		if len(comp.Parameters.Properties) > 0 && len(comp.Parameters.Order) != len(comp.Parameters.Properties) {
			return fmt.Errorf("ui: component %q parameters.x-order must list all %d properties in positional order",
				comp.Name, len(comp.Parameters.Properties))
		}
		for _, key := range comp.Parameters.Order {
			if _, ok := comp.Parameters.Properties[key]; !ok {
				return fmt.Errorf("ui: component %q x-order names unknown property %q", comp.Name, key)
			}
		}
	}
	return nil
}

// CompileInstructions builds a deterministic, byte-identical system prompt section from
// Spec so LLM prefix caching stays warm across turns.
func CompileInstructions(spec Spec) (string, error) {
	format := spec.Format
	if format == "" {
		format = FormatOpenUI
	}
	if format != FormatOpenUI && format != FormatA2UI {
		return "", fmt.Errorf("ui: unsupported format %q (expected openui or a2ui)", format)
	}
	if err := ValidatePortableComponents(spec.Components); err != nil {
		return "", err
	}

	selected, knownArity, err := selectComponents(spec.Keep, spec.Components)
	if err != nil {
		return "", err
	}

	for _, bp := range spec.Blueprints {
		if strings.TrimSpace(bp.Name) == "" {
			return "", errors.New("ui: every blueprint needs a name")
		}
		if strings.TrimSpace(bp.When) == "" {
			return "", fmt.Errorf("ui: blueprint %q needs a when condition", bp.Name)
		}
		if len(bp.Order) == 0 {
			return "", fmt.Errorf("ui: blueprint %q needs a non-empty component order", bp.Name)
		}
		if bp.Example != "" && format == FormatOpenUI {
			if problems := ValidateOpenUI(extractFenceOrRaw(bp.Example), knownArity); len(problems) > 0 {
				return "", fmt.Errorf("ui: blueprint %q example is invalid: %s", bp.Name, strings.Join(problems, "; "))
			}
		}
	}

	var b strings.Builder
	b.WriteString("## Interactive UI Generation\n\n")
	if spec.VoiceUI {
		b.WriteString("When an answer benefits from interactive UI, first speak or write one concise sentence summarizing the key takeaway, then emit the UI in a single ```openui-lang fenced block. Never read component names or UI syntax aloud.\n\n")
	} else {
		b.WriteString("When an answer benefits from interactive UI, first write one short sentence summarizing what the card shows, then write the whole UI in one ```openui-lang fenced block. On a voice call without a screen, speak concisely with no UI block.\n\n")
	}

	b.WriteString("### Syntax & Streaming Invariants\n")
	b.WriteString("1. Put `root = Card([...])` on the FIRST line so the card shell streams immediately.\n")
	b.WriteString("2. Every statement is `identifier = Expression` on its own line; arguments are strictly POSITIONAL (`TypeName(arg1, arg2)` — never `key: value`).\n")
	b.WriteString("3. Every defined identifier other than `root` and `$state` variables must be referenced from `root`.\n")
	b.WriteString("4. Never fabricate image URLs; use custom components or icons when no verified URL is in context.\n")
	if spec.PatchMode {
		b.WriteString("5. When refining an existing UI from the immediately preceding turn (e.g. updating a time slot, party size, or filter), emit only the changed statements inside a ```openui-patch fenced block (`$party = 4`, or `oldSection = null` to remove).\n")
	}

	b.WriteString("\n### Allowed Component Signatures\n")
	for _, def := range selected {
		b.WriteString("- ")
		b.WriteString(def.Signature)
		b.WriteByte('\n')
	}
	for _, comp := range spec.Components {
		b.WriteString("- ")
		b.WriteString(portableSignature(comp))
		b.WriteByte('\n')
	}

	if len(spec.Blueprints) > 0 {
		b.WriteString("\n### Canonical Layout Blueprints (Predictability Rules)\n")
		b.WriteString("Match similar prompts to the exact component sequence below so repeated asks produce a consistent layout:\n")
		for _, bp := range spec.Blueprints {
			b.WriteString(fmt.Sprintf("- **%s** (when %s): `Card([%s])`", bp.Name, bp.When, strings.Join(bp.Order, " -> ")))
			if len(bp.Rules) > 0 {
				b.WriteString(" — ")
				b.WriteString(strings.Join(bp.Rules, "; "))
			}
			b.WriteByte('\n')
			if bp.Example != "" {
				b.WriteString("\n```openui-lang\n")
				b.WriteString(strings.TrimSpace(extractFenceOrRaw(bp.Example)))
				b.WriteString("\n```\n")
			}
		}
	}

	if len(spec.Rules) > 0 {
		b.WriteString("\n### Brand & Design Rules\n")
		for _, rule := range spec.Rules {
			b.WriteString("- ")
			b.WriteString(strings.TrimSpace(rule))
			b.WriteByte('\n')
		}
	}

	return strings.TrimSpace(b.String()), nil
}

// ValidateOpenUI performs fast structural validation on an OpenUI Lang program: checking
// balanced delimiters, a valid root assignment, known component arities, unresolved
// references, and orphaned statements.
func ValidateOpenUI(program string, knownArity map[string]int) []string {
	trimmed := strings.TrimSpace(program)
	if trimmed == "" {
		return []string{"program is empty"}
	}
	stmts, balanceErr := splitStatementsChecked(trimmed)
	if balanceErr != "" {
		return []string{balanceErr}
	}

	defined := map[string]string{}
	var order []string
	for _, stmt := range stmts {
		lhs, rhs, ok := splitAssignment(stmt)
		if !ok {
			return []string{fmt.Sprintf("invalid statement %q", stmt)}
		}
		defined[lhs] = rhs
		order = append(order, lhs)
	}

	if _, hasRoot := defined["root"]; !hasRoot {
		return []string{"root = Card(...) is not defined"}
	}

	var problems []string
	used := map[string]bool{"root": true}
	for _, lhs := range order {
		rhs := defined[lhs]
		problems = append(problems, checkCallsAndRefs(rhs, defined, used, knownArity)...)
	}

	for _, lhs := range order {
		if lhs == "root" || strings.HasPrefix(lhs, "$") {
			continue
		}
		if !used[lhs] {
			problems = append(problems, fmt.Sprintf("%s is defined but never referenced", lhs))
		}
	}
	return problems
}

func selectComponents(keep []string, custom []PortableComponent) ([]builtinDef, map[string]int, error) {
	byName := make(map[string]builtinDef, len(builtinCatalog))
	for _, b := range builtinCatalog {
		byName[b.Name] = b
	}
	customArity := make(map[string]int, len(custom))
	for _, c := range custom {
		maxArgs := len(c.Parameters.Order) + len(c.Slots)
		customArity[PascalName(c.Name)] = maxArgs
	}

	included := map[string]bool{"Card": true}
	if len(keep) == 0 {
		for _, b := range builtinCatalog {
			included[b.Name] = true
		}
	} else {
		var queue []string
		for _, name := range keep {
			if _, isCustom := customArity[name]; isCustom {
				continue
			}
			if _, ok := byName[name]; !ok {
				return nil, nil, fmt.Errorf("ui: unknown component %q in keep allowlist", name)
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

	var selected []builtinDef
	arity := make(map[string]int, len(included)+len(customArity))
	for _, b := range builtinCatalog {
		if included[b.Name] {
			selected = append(selected, b)
			arity[b.Name] = b.MaxArgs
		}
	}
	for k, v := range customArity {
		arity[k] = v
	}
	return selected, arity, nil
}

func portableSignature(comp PortableComponent) string {
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
	var details []string
	details = append(details, strings.TrimSpace(comp.Description))
	for _, ev := range comp.Events {
		if strings.TrimSpace(ev.Description) != "" {
			details = append(details, strings.TrimSpace(ev.Description))
		}
	}
	return fmt.Sprintf("%s(%s) — %s", PascalName(comp.Name), strings.Join(params, ", "), strings.Join(details, " "))
}

func extractFenceOrRaw(example string) string {
	if _, after, ok := strings.Cut(example, "```openui-lang\n"); ok {
		if body, _, closed := strings.Cut(after, "\n```"); closed {
			return body
		}
	}
	return example
}

func splitStatementsChecked(program string) ([]string, string) {
	var stmts []string
	var current strings.Builder
	var quote byte
	escaped := false
	depth := 0

	for _, line := range strings.Split(program, "\n") {
		if quote == 0 && depth == 0 {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" || strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "#") {
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
					return nil, "unmatched closing bracket"
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
	if quote != 0 {
		return nil, "unterminated string literal"
	}
	if depth != 0 {
		return nil, "unclosed bracket or parenthesis"
	}
	return stmts, ""
}

func splitAssignment(stmt string) (string, string, bool) {
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
		if !identPattern.MatchString(lhs) {
			return "", "", false
		}
		return lhs, strings.TrimSpace(stmt[i+1:]), true
	}
	return "", "", false
}

func checkCallsAndRefs(rhs string, defined map[string]string, used map[string]bool, knownArity map[string]int) []string {
	var problems []string
	stripped := stripStrings(rhs)
	matches := refCallPattern.FindAllStringIndex(stripped, -1)
	for _, span := range matches {
		token := stripped[span[0]:span[1]]
		if span[0] > 0 && (stripped[span[0]-1] == '@' || stripped[span[0]-1] == '.') {
			continue
		}
		after := strings.TrimLeft(stripped[span[1]:], " \t")
		if strings.HasPrefix(after, ":") {
			continue
		}
		if slices.Contains([]string{"true", "false", "null", "Action", "Query", "Mutation"}, token) {
			continue
		}
		isCall := span[1] < len(stripped) && stripped[span[1]] == '('
		if isCall {
			if len(knownArity) > 0 {
				maxArgs, known := knownArity[token]
				if !known {
					problems = append(problems, fmt.Sprintf("unknown component %s", token))
					continue
				}
				if count := countTopArgs(stripped[span[1]:]); count > maxArgs {
					problems = append(problems, fmt.Sprintf("%s passed %d arguments, max is %d", token, count, maxArgs))
				}
			}
			continue
		}
		if _, ok := defined[token]; ok {
			used[token] = true
		} else {
			problems = append(problems, fmt.Sprintf("%s is referenced but not defined", token))
		}
	}
	return problems
}

func stripStrings(s string) string {
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

func countTopArgs(callSuffix string) int {
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
