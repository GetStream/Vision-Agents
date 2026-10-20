// Package moderator holds what an agent_moderator config's runs are held to, apart from the
// instructions the app may edit (acceleration/templates/agent_moderator).
package moderator

import _ "embed"

// Contract is the answer format the chat review worker parses. A run of an agent_moderator
// config adds it after every other instruction, so no edit to the agent can change the shape.
//
//go:embed contract.md
var Contract string
