//go:build integration

package usage_test

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

// ClientSuite calls the real service, on a database of its own, through the client producers
// use.
type ClientSuite struct {
	suite.Suite
	ctx     context.Context
	dsn     string
	service *usagetest.Service
}

func TestClientSuite(t *testing.T) {
	suite.Run(t, new(ClientSuite))
}

func (s *ClientSuite) SetupSuite() {
	s.ctx = context.Background()
	s.dsn = usagetest.Database(s.T(), "usage_client")
}

func (s *ClientSuite) SetupTest() {
	s.service = usagetest.Start(s.T(), usagetest.Options{DSN: s.dsn})
	_, err := s.service.Ledger.DB().ExecContext(s.ctx, "TRUNCATE usage_events, usage_windows")
	s.Require().NoError(err)
}

func (s *ClientSuite) TestTheClientShipsAndReads() {
	batch := []usage.Event{
		usagetest.Event("o1", func(e *usage.Event) { e.AppID, e.CostMicros = "a1", 400 }),
		usagetest.Event("o1", func(e *usage.Event) { e.AppID, e.CostMicros = "a1", 600 }),
	}

	ack, err := s.service.Client.Ship(s.ctx, batch)
	s.Require().NoError(err)
	s.Equal(usage.Ack{Accepted: 2, Duplicates: 0, Rejected: []usage.Rejection{}}, ack)

	windows, err := s.service.Client.Windows(s.ctx, []usage.SubjectRef{{OrganizationID: "o1", Subject: usage.SubjectApp, AppID: "a1"}})
	s.Require().NoError(err)
	s.Require().Len(windows, 4)
	for _, window := range windows {
		s.Equal(int64(1000), window.SpentMicros, window.Interval)
		s.Equal(int64(2), window.Events, window.Interval)
	}

	again, err := s.service.Client.Ship(s.ctx, batch)
	s.Require().NoError(err)
	s.Equal(usage.Ack{Accepted: 0, Duplicates: 2, Rejected: []usage.Rejection{}}, again)
}

func (s *ClientSuite) TestTheClientSendsMoreThanABatchInSeveralCalls() {
	events := make([]usage.Event, usage.MaxEvents+100)
	for i := range events {
		events[i] = usagetest.Event("o1")
	}
	events[usage.MaxEvents+5].APIKey = ""

	ack, err := s.service.Client.Ship(s.ctx, events)

	s.Require().NoError(err)
	s.Equal(usage.MaxEvents+99, ack.Accepted)
	s.Require().Len(ack.Rejected, 1)
	s.Equal(events[usage.MaxEvents+5].UsageID, ack.Rejected[0].UsageID)
	s.Equal(usage.MaxEvents+5, ack.Rejected[0].Index, "the index is into the events shipped, not the batch")
}

func (s *ClientSuite) TestAnEventNoBodyCanHoldIsRejectedAndTheRestShipped() {
	huge := usagetest.Event("o1", func(e *usage.Event) {
		e.Attributes = map[string]string{"padding": strings.Repeat("x", usage.MaxBodyBytes)}
	})
	events := []usage.Event{usagetest.Event("o1"), huge, usagetest.Event("o1", func(e *usage.Event) { e.APIKey = "" }), usagetest.Event("o1")}

	ack, err := s.service.Client.Ship(s.ctx, events)

	s.Require().NoError(err)
	s.Equal(2, ack.Accepted)
	s.Require().Len(ack.Rejected, 2)
	s.Equal(usage.Rejection{Index: 1, UsageID: huge.UsageID, Reason: usage.ReasonTooLarge, Message: ack.Rejected[0].Message}, ack.Rejected[0])
	s.Contains(ack.Rejected[0].Message, "not sent")
	s.Equal(2, ack.Rejected[1].Index, "the service's rejection, at its place among the events shipped")
	s.Equal(usage.ReasonAPIKey, ack.Rejected[1].Reason)
}

func (s *ClientSuite) TestTheClientSplitsEventsTooBigForOneBody() {
	// 500 events of 8 KiB each are 4 MiB, twice what one body may hold.
	events := make([]usage.Event, usage.MaxEvents)
	for i := range events {
		events[i] = usagetest.Event("o1", func(e *usage.Event) {
			e.Attributes = map[string]string{"padding": strings.Repeat("x", 8<<10)}
		})
	}

	ack, err := s.service.Client.Ship(s.ctx, events)

	s.Require().NoError(err)
	s.Equal(usage.MaxEvents, ack.Accepted)
	s.Empty(ack.Rejected)
}

func (s *ClientSuite) TestAWrongTokenIsAStatusErrorTheCallerCanRead() {
	client, err := usage.NewClient(usage.ClientOptions{URL: s.service.URL, Token: "wrong"})
	s.Require().NoError(err)

	_, err = client.Ship(s.ctx, []usage.Event{usagetest.Event("o1")})

	var failed *usage.StatusError
	s.Require().ErrorAs(err, &failed)
	s.Equal(http.StatusUnauthorized, failed.Status)
	s.Contains(failed.Body, "ingest token")
}

func (s *ClientSuite) TestShippingNothingCallsNobody() {
	client, err := usage.NewClient(usage.ClientOptions{URL: "http://127.0.0.1:1", Token: "t"})
	s.Require().NoError(err)

	ack, err := client.Ship(s.ctx, nil)

	s.Require().NoError(err)
	s.Zero(ack.Accepted)
}
