package tui

import (
	"context"
	"fmt"
	"strings"

	"github.com/charmbracelet/x/ansi"
)

// Command is a slash command an application adds to the built-in ones. It is reached by
// typing its name after a slash, so "ticket" is typed "/ticket 4821".
type Command struct {
	// Name is the word after the slash, without the slash.
	Name string
	// Args is a usage hint for the help, such as "<id>".
	Args string
	// Help is one line saying what the command does.
	Help string
	// Run does it. What it returns is shown on the status line and so is an error. It
	// runs off the event loop, so it may take its time, and the conversation stays
	// readable while it does.
	Run func(ctx context.Context, args []string) (string, error)
}

// builtins are the commands every conversation has. Run is nil because the model
// handles them itself; they are listed here so the help is written once.
var builtins = []Command{
	{Name: "new", Help: "start a new conversation"},
	{Name: "resume", Args: "<id>", Help: "open a saved conversation"},
	{Name: "older", Help: "load the page of messages before this one"},
	{Name: "help", Help: "list the commands"},
	{Name: "quit", Help: "leave, keeping the conversation"},
}

func isBuiltin(name string) bool {
	for _, c := range builtins {
		if c.Name == name {
			return true
		}
	}
	return false
}

// validateCommands refuses a command that could not be reached or that would shadow one
// of the built-ins, rather than letting it be silently unavailable.
func validateCommands(commands []Command) error {
	seen := map[string]bool{}
	for _, c := range commands {
		switch {
		case c.Name == "":
			return fmt.Errorf("tui: a command needs a name")
		case strings.ContainsAny(c.Name, " \t/"):
			return fmt.Errorf("tui: command %q may not contain a space or a slash", c.Name)
		case c.Run == nil:
			return fmt.Errorf("tui: command %q needs something to run", c.Name)
		case isBuiltin(c.Name):
			return fmt.Errorf("tui: command %q is built in", c.Name)
		case seen[c.Name]:
			return fmt.Errorf("tui: command %q is declared twice", c.Name)
		}
		seen[c.Name] = true
	}
	return nil
}

// parseCommand splits a submitted line that starts with a slash. It returns the name
// without the slash and whatever followed it.
func parseCommand(line string) (name string, args []string, ok bool) {
	if !strings.HasPrefix(line, "/") {
		return "", nil, false
	}
	fields := strings.Fields(strings.TrimPrefix(line, "/"))
	if len(fields) == 0 {
		return "", nil, false
	}
	return fields[0], fields[1:], true
}

// commandHelp lists the commands as a block for the transcript, the names in one column
// and what they do in another.
func commandHelp(s styles, extra []Command, width int) string {
	all := append(append([]Command{}, builtins...), extra...)
	names := make([]string, len(all))
	widest := 0
	for i, c := range all {
		names[i] = "/" + c.Name
		if c.Args != "" {
			names[i] += " " + c.Args
		}
		widest = max(widest, ansi.StringWidth(names[i]))
	}
	var b strings.Builder
	b.WriteString(s.muted.Render("Commands") + "\n")
	for i, c := range all {
		name := s.accent.Render(names[i]) + strings.Repeat(" ", widest-ansi.StringWidth(names[i]))
		line := "  " + name + "  " + s.muted.Render(c.Help)
		b.WriteString(ansi.Truncate(line, max(1, width), "…") + "\n")
	}
	return strings.TrimRight(b.String(), "\n")
}
