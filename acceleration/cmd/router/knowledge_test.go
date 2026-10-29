package main

import (
	"bytes"
	"log/slog"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
)

// KnowledgeLimitsSuite covers what the router warns of at startup about how much a
// document's passages can grow.
type KnowledgeLimitsSuite struct {
	suite.Suite
	settings config.Config
	logs     bytes.Buffer
}

func TestKnowledgeLimitsSuite(t *testing.T) {
	suite.Run(t, new(KnowledgeLimitsSuite))
}

func (s *KnowledgeLimitsSuite) SetupTest() {
	s.settings = config.Defaults()
	s.logs.Reset()
}

func (s *KnowledgeLimitsSuite) TestLimitsThatGrowADocumentFourTimesOrLessAreNotWarnedOf() {
	s.settings.Knowledge.MinChunk, s.settings.Knowledge.MaxHeading = 500, 235

	warnKnowledgeLimits(s.settings, slog.New(slog.NewTextHandler(&s.logs, nil)))

	s.Empty(s.logs.String())
}

func (s *KnowledgeLimitsSuite) TestLimitsThatGrowADocumentMoreThanFourTimesAreWarnedOf() {
	s.settings.Knowledge.MinChunk, s.settings.Knowledge.MaxHeading = 500, 300

	warnKnowledgeLimits(s.settings, slog.New(slog.NewTextHandler(&s.logs, nil)))

	s.Contains(s.logs.String(), "level=WARN")
	s.Contains(s.logs.String(), "5.6")
	s.Contains(s.logs.String(), "ROUTER_KNOWLEDGE_MAX_HEADING")
	s.Contains(s.logs.String(), "ROUTER_KNOWLEDGE_MIN_CHUNK")
}

func (s *KnowledgeLimitsSuite) TestNoMinimumChunkIsWarnedOf() {
	s.settings.Knowledge.MinChunk, s.settings.Knowledge.MaxHeading = 0, 0

	warnKnowledgeLimits(s.settings, slog.New(slog.NewTextHandler(&s.logs, nil)))

	s.Contains(s.logs.String(), "level=WARN")
	s.Contains(s.logs.String(), "ROUTER_KNOWLEDGE_MIN_CHUNK")
}
