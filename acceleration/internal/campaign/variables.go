package campaign

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
)

// VariableName is the shape of a contact column that a placeholder may name.
var VariableName = regexp.MustCompile(`^[a-z][a-z0-9_]{0,63}$`)

var placeholder = regexp.MustCompile(`\{\{ *([a-z][a-z0-9_]{0,63}) *\}\}`)

// Render fills each {{ name }} with the contact's value. A name the contact lacks becomes "".
func Render(text string, variables map[string]string) string {
	return placeholder.ReplaceAllStringFunc(text, func(match string) string {
		return variables[placeholder.FindStringSubmatch(match)[1]]
	})
}

// Brief lists the variables the texts never mention, so the agent still knows them.
// It is empty when every variable is mentioned.
func Brief(texts []string, variables map[string]string) string {
	mentioned := map[string]bool{}
	for _, text := range texts {
		for _, m := range placeholder.FindAllStringSubmatch(text, -1) {
			mentioned[m[1]] = true
		}
	}

	var names []string
	for name := range variables {
		if !mentioned[name] {
			names = append(names, name)
		}
	}
	if len(names) == 0 {
		return ""
	}
	slices.Sort(names)

	lines := []string{"About the person you are calling:"}
	for _, name := range names {
		lines = append(lines, fmt.Sprintf("- %s: %s", name, variables[name]))
	}
	return strings.Join(lines, "\n")
}
