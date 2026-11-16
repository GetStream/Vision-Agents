package api

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

type LogHandlerSuite struct {
	suite.Suite

	out    *bytes.Buffer
	logger *slog.Logger
}

func TestLogHandlerSuite(t *testing.T) {
	suite.Run(t, new(LogHandlerSuite))
}

func (s *LogHandlerSuite) SetupTest() {
	s.out = &bytes.Buffer{}
	s.logger = slog.New(NewLogHandler(slog.NewTextHandler(s.out, nil)))
}

func (s *LogHandlerSuite) TestALineLoggedWithTheRequestsContextNamesItsAppAndKey() {
	ctx := withLoggedApp(context.Background(), "1732479", "pubkey123")

	s.logger.InfoContext(ctx, "probe", "x", 1)

	s.Contains(s.out.String(), " x=1 app_id=1732479 api_key=pubkey123\n")
}

func (s *LogHandlerSuite) TestALineLoggedWithoutTheContextNamesNeither() {
	s.logger.Info("probe")
	s.logger.InfoContext(context.Background(), "probe")

	s.NotContains(s.out.String(), "app_id")
	s.NotContains(s.out.String(), "api_key")
}

func (s *LogHandlerSuite) TestAKeyNobodyNamedIsNotLoggedEmpty() {
	s.logger.InfoContext(withLoggedApp(context.Background(), "1732479", ""), "probe")

	s.Contains(s.out.String(), "app_id=1732479")
	s.NotContains(s.out.String(), "api_key")
}

func (s *LogHandlerSuite) TestALineThatNamesTheAppItselfIsNotGivenASecondOne() {
	ctx := withLoggedApp(context.Background(), "1732479", "pubkey123")

	s.logger.InfoContext(ctx, "probe", "app_id", "other")
	s.logger.InfoContext(ctx, "probe", "api_key", "other")

	lines := strings.Split(strings.TrimSpace(s.out.String()), "\n")
	s.Require().Len(lines, 2)
	s.Equal(1, strings.Count(lines[0], "app_id="))
	s.Contains(lines[0], "app_id=other")
	s.Contains(lines[0], "api_key=pubkey123")
	s.Equal(1, strings.Count(lines[1], "api_key="))
	s.Contains(lines[1], "api_key=other")
	s.Contains(lines[1], "app_id=1732479")
}

func (s *LogHandlerSuite) TestALoggerThatAlreadyNamesTheAppIsNotGivenASecondOne() {
	ctx := withLoggedApp(context.Background(), "1732479", "pubkey123")

	s.logger.With("app_id", "other").InfoContext(ctx, "probe")
	s.logger.With("api_key", "other").With("x", 1).InfoContext(ctx, "probe")

	lines := strings.Split(strings.TrimSpace(s.out.String()), "\n")
	s.Require().Len(lines, 2)
	s.Equal(1, strings.Count(lines[0], "app_id="))
	s.Contains(lines[0], "api_key=pubkey123")
	s.Equal(1, strings.Count(lines[1], "api_key="))
	s.Contains(lines[1], "app_id=1732479")
}

func (s *LogHandlerSuite) TestALoggerWithAFieldDoesNotHideTheAppFromItsSiblings() {
	ctx := withLoggedApp(context.Background(), "1732479", "pubkey123")
	named := s.logger.With("app_id", "other")

	s.logger.InfoContext(ctx, "probe")
	named.InfoContext(ctx, "probe")
	s.logger.With("x", 1).InfoContext(ctx, "probe")

	lines := strings.Split(strings.TrimSpace(s.out.String()), "\n")
	s.Require().Len(lines, 3)
	s.Contains(lines[0], "app_id=1732479")
	s.Equal(1, strings.Count(lines[1], "app_id="))
	s.Contains(lines[2], "app_id=1732479")
}
