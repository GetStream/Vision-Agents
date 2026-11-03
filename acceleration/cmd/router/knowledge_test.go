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

func (s *KnowledgeLimitsSuite) TestTwoRegionsKeptInOneTurbopufferRegionAreRefused() {
	s.T().Setenv("TURBOPUFFER_API_KEY", "tpuf-test")
	s.settings.Knowledge.Regions = map[string]string{"us": "aws-eu-central-1", "eu": "aws-eu-central-1"}

	_, err := knowledgeBackends(s.settings.Knowledge, 0, slog.New(slog.DiscardHandler))
	s.ErrorContains(err, "keeps eu and us in one turbopuffer region, aws-eu-central-1",
		"its knowledge bases would be listed under both, and their storage billed twice")
}

func (s *KnowledgeLimitsSuite) TestARegionKeptWhereTheDefaultRegionIsIsRefused() {
	s.T().Setenv("TURBOPUFFER_API_KEY", "tpuf-test")
	s.T().Setenv("TURBOPUFFER_REGION", "gcp-us-east4")
	s.settings.Knowledge.Regions = map[string]string{"eu": "gcp-us-east4"}

	_, err := knowledgeBackends(s.settings.Knowledge, 0, slog.New(slog.DiscardHandler))
	s.ErrorContains(err, "keeps eu and us in one turbopuffer region, gcp-us-east4")

	s.settings.Knowledge.Regions = map[string]string{"eu": "aws-eu-central-1"}
	backends, err := knowledgeBackends(s.settings.Knowledge, 0, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	s.Len(backends, 2)
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
