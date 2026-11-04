package session

import (
	"fmt"
	"strings"
)

// serverInstructionsLimit caps what one server of a custom connector adds to the agent's
// instructions, since nobody vetted how much it says.
const serverInstructionsLimit = 4000

// serverSaid is what one server said about using its tools: name is what the config calls
// it, tools the pattern of the names its tools are offered under.
type serverSaid struct {
	name, tools, text string
}

// serverInstructions are what the config's custom connectors' servers said about using their
// tools, for the agent's instructions. Built-in connectors are left out: their tools are
// described well enough to use without.
func serverInstructions(connectors []serverSaid) string {
	var said []string
	for _, server := range connectors {
		text := server.text
		if text == "" {
			continue
		}
		if len(text) > serverInstructionsLimit {
			text = strings.ToValidUTF8(text[:serverInstructionsLimit], "")
		}
		said = append(said, fmt.Sprintf("The %s tools (%s) come with these instructions from their server:\n\n%s",
			server.name, server.tools, text))
	}
	return strings.Join(said, "\n\n")
}
