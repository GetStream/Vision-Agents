package main

import (
	"bytes"
	"context"
	"log/slog"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pluginmigrate"
)

// PluginsCommandSuite is router plugins as a command line, without a database: what it
// refuses before it opens anything.
type PluginsCommandSuite struct {
	suite.Suite
}

func TestPluginsCommandSuite(t *testing.T) {
	suite.Run(t, new(PluginsCommandSuite))
}

func (s *PluginsCommandSuite) TestNoSubcommandPrintsTheUsage() {
	err := dispatchCommand("plugins", nil, config.Config{}, slog.Default())

	s.ErrorContains(err, "usage: router plugins migrate [--apply]")
}

func (s *PluginsCommandSuite) TestAnUnknownSubcommandIsRefused() {
	err := dispatchCommand("plugins", []string{"drop"}, config.Config{}, slog.Default())

	s.ErrorContains(err, `unknown plugins command "drop"`)
}

// TestApplyIsRefusedWithConnectorsOff: a binding wins over its plugin entry in a session, so
// bindings written where connectors are off would leave the agent with neither. The refusal
// comes before the database is opened: these settings name none.
func (s *PluginsCommandSuite) TestApplyIsRefusedWithConnectorsOff() {
	var out bytes.Buffer

	err := migratePlugins(context.Background(), config.Config{}, slog.Default(), pluginmigrate.Options{}, true, &out)

	s.ErrorContains(err, "needs connectors.enabled")
	s.Empty(out.String())
}

// TestADryRunNeedsADatabase: without --apply the command still reads, so it needs postgres.dsn
// like every command that reads.
func (s *PluginsCommandSuite) TestADryRunNeedsADatabase() {
	var out bytes.Buffer

	err := migratePlugins(context.Background(), config.Config{}, slog.Default(), pluginmigrate.Options{}, false, &out)

	s.ErrorContains(err, "set postgres.dsn")
}
