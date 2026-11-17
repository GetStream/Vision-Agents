package agentscorego

import (
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// The README's first program is the quickstart example, word for word, so the code a reader
// copies is code that compiles: the example is built with the module.
func TestTheReadmesQuickstartIsTheQuickstartExample(t *testing.T) {
	readme, err := os.ReadFile("README.md")
	require.NoError(t, err)
	example, err := os.ReadFile("examples/quickstart/main.go")
	require.NoError(t, err)

	_, block, found := strings.Cut(string(readme), "```go\n")
	require.True(t, found, "the README has no Go block")
	block, _, found = strings.Cut(block, "```")
	require.True(t, found, "the README's first Go block never ends")
	program := string(example)
	program = program[strings.Index(program, "package main"):]

	require.Equal(t, strings.TrimSpace(program), strings.TrimSpace(block),
		"the README's quickstart and examples/quickstart/main.go have drifted apart")
}
