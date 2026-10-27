package main

import (
	"log/slog"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/ingest"
)

type CommandSuite struct {
	suite.Suite
}

func TestCommandSuite(t *testing.T) {
	suite.Run(t, new(CommandSuite))
}

func (s *CommandSuite) TestANegativeLimitIsRefused() {
	options := options{customer: "acme", namespace: "docs", size: ingest.DefaultChunk,
		limits: ingest.Limits{MinChunk: ingest.DefaultLimits.MinChunk, MaxHeading: -1}, dryRun: true}

	err := run(options, []string{s.T().TempDir()}, slog.New(slog.DiscardHandler))

	s.ErrorContains(err, "cannot be negative")
}
