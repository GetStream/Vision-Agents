//go:build integration

package usage_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
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
	usagetest.Empty(s.T(), s.service.Ledger)
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

func (s *ClientSuite) TestTheClientReadsAnOrganizationsStanding() {
	_, _, err := s.service.Ledger.AddCredit(s.ctx, ledger.CreditEntry{
		OrganizationID: "o1", Kind: usage.CreditTopUp, AmountMicros: 5000, IdempotencyKey: "payment-1", Actor: "proxy",
	})
	s.Require().NoError(err)
	_, err = s.service.Client.Ship(s.ctx, []usage.Event{usagetest.Event("o1", func(e *usage.Event) { e.CostMicros = 1200 })})
	s.Require().NoError(err)

	standings, err := s.service.Client.Standings(s.ctx, usage.StandingQuery{Organizations: []usage.StandingOrganization{
		{OrganizationID: "o1", OrganizationTier: "tier-a"},
	}})

	s.Require().NoError(err)
	s.Require().Len(standings, 1)
	s.Equal(int64(3800), standings[0].BalanceMicros)
	s.Equal(int64(1200), standings[0].SpendWindows.Daily.SpentMicros)
	s.Equal(usage.BuiltInDefaultTier(), standings[0].Tier)
	s.True(standings[0].FallbackTier)
}

// The client narrows a read by origin alone: api_key, its old name, is no longer sent
// (AI-2851).
func (s *ClientSuite) TestTheClientReadsUsageByOrigin() {
	hook := usagetest.Event("o1", func(e *usage.Event) { e.APIKey = "hook:chat" })
	_, err := s.service.Client.Ship(s.ctx, []usage.Event{hook, usagetest.Event("o1")})
	s.Require().NoError(err)
	query := usage.UsageQuery{Origin: "hook:chat"}

	page, err := s.service.Client.Usage(s.ctx, "o1", query)

	s.Require().NoError(err)
	s.Require().Len(page.Items, 1)
	s.Equal(hook.UsageID, page.Items[0].UsageID)
	s.Equal("hook:chat", page.Items[0].Origin)
	s.Equal(url.Values{"origin": {"hook:chat"}}, query.Values())
}

// While pods roll, a router reads a usage pod of the release before (v0.6.280 to v0.6.284). Such
// a pod reads origin and answers it beside api_key, so origin alone still narrows the read and
// names who started the work (AI-2851). The fake answers as that pod does.
func (s *ClientSuite) TestTheClientReadsAnOlderUsageServiceByOrigin() {
	var asked url.Values
	older := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asked = r.URL.Query()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"items":[{"usage_id":"0190f3a4-7b1c-7000-8000-000000000001","occurred_at":"2026-11-01T00:00:00Z",` +
			`"app_id":"a1","origin":"hook:chat","api_key":"hook:chat","usage_categories":[],"meter":"llm",` +
			`"quantities":{},"cost_micros":1,"billable":true,"success":true}],"has_more":false}`))
	}))
	s.T().Cleanup(older.Close)
	client, err := usage.NewClient(usage.ClientOptions{URL: older.URL, Token: "token"})
	s.Require().NoError(err)

	page, err := client.Usage(s.ctx, "o1", usage.UsageQuery{Origin: "hook:chat"})

	s.Require().NoError(err)
	s.Equal(url.Values{"origin": {"hook:chat"}}, asked)
	s.Require().Len(page.Items, 1)
	s.Equal("hook:chat", page.Items[0].Origin)
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
