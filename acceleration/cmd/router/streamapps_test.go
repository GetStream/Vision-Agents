package main

import (
	"bytes"
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// StreamAppsCommandSuite covers the stream-apps commands' own reading and printing.
type StreamAppsCommandSuite struct {
	suite.Suite
}

func TestStreamAppsCommandSuite(t *testing.T) {
	suite.Run(t, new(StreamAppsCommandSuite))
}

func (s *StreamAppsCommandSuite) TestAWindowIsReadInDaysOrAsADuration() {
	days, err := lookBack("14d")
	s.Require().NoError(err)
	s.Equal(14*24*time.Hour, days)

	hours, err := lookBack("36h")
	s.Require().NoError(err)
	s.Equal(36*time.Hour, hours)

	for _, bad := range []string{"0d", "-3d", "soon", "-1h"} {
		_, err := lookBack(bad)
		s.ErrorContains(err, "--since", bad)
	}
}

func (s *StreamAppsCommandSuite) TestFallbacksNameNobodyUnlessAsked() {
	at := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	uses := []store.StreamFallbackUse{
		{CustomerID: "first-customer", Uses: 3, FirstAt: at, LastAt: at},
		{CustomerID: "second-customer", Uses: 4, FirstAt: at, LastAt: at.Add(time.Hour)},
	}

	var counted bytes.Buffer
	s.Require().NoError(printFallbacks(&counted, uses, "14d", false))
	s.Equal("2 customers wrote into the deployment's app 7 times in the last 14d\n", counted.String())

	var listed bytes.Buffer
	s.Require().NoError(printFallbacks(&listed, uses, "14d", true))
	s.Contains(listed.String(), "first-customer")
	s.Less(bytes.Index(listed.Bytes(), []byte("second-customer")), bytes.Index(listed.Bytes(), []byte("first-customer")),
		"the most recent first")
}

func (s *StreamAppsCommandSuite) TestBackfillIsRefusedInDeploymentMode() {
	err := runBackfillPins(context.Background(), config.Defaults(), slog.New(slog.DiscardHandler), &bytes.Buffer{})

	s.ErrorContains(err, "stream.tenancy=app")
}

func (s *StreamAppsCommandSuite) TestAnUnknownCommandSaysWhatThereIs() {
	err := runStreamApps([]string{"frobnicate"}, config.Defaults(), slog.New(slog.DiscardHandler))

	s.ErrorContains(err, "backfill-pins")
}

func (s *StreamAppsCommandSuite) TestLegacyCountsNameNobodyUnlessAsked() {
	counted := []store.LegacyCount{
		{Kind: "session", CustomerID: "first-customer", Rows: 3},
		{Kind: "session", CustomerID: "second-customer", Rows: 2},
		{Kind: "call", CustomerID: "first-customer", Rows: 1},
	}

	var totals bytes.Buffer
	s.Require().NoError(printLegacy(&totals, counted, false))
	s.Contains(totals.String(), "session: 5 rows of 2 customers")
	s.Contains(totals.String(), "call: 1 rows of 1 customers")
	s.NotContains(totals.String(), "first-customer")

	var listed bytes.Buffer
	s.Require().NoError(printLegacy(&listed, counted, true))
	s.Contains(listed.String(), "second-customer")
}
