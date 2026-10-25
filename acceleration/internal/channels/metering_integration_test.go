//go:build integration

package channels

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// TextMeteringSuite is what a text costs as the service records it, written to real Postgres
// through a recorder priced from the vendor list, with Telnyx stood in for by a local server.
type TextMeteringSuite struct {
	suite.Suite
	ctx      context.Context
	store    *store.Store
	service  *Service
	customer string
}

func TestTextMeteringSuite(t *testing.T) { suite.Run(t, new(TextMeteringSuite)) }

func (s *TextMeteringSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "channels_metering"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
}

func (s *TextMeteringSuite) SetupTest() {
	list, err := pricing.NewList("vendor", prices.Vendor().Entries, 0, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	recorder := routing.NewRecorder(routing.Phone, s.store, nil, list, slog.New(slog.DiscardHandler))
	s.T().Cleanup(recorder.Close)
	s.service = &Service{store: s.store, recorder: recorder, logger: slog.New(slog.DiscardHandler)}
	s.customer = "customer-" + uuid.NewString()
}

func (s *TextMeteringSuite) owner() routing.Owner {
	return routing.Owner{CustomerID: s.customer, Attribution: routing.Attribution{APIKey: routing.OriginPhoneInbound}}
}

func (s *TextMeteringSuite) TestAReplyThatFailedPartwayIsBilledForWhatWentOut() {
	failing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if _, media := body["media_urls"]; media {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer failing.Close()
	provider := &telnyxProvider{client: failing.Client(), baseURL: failing.URL}
	line := Account{Kind: SMS, E164: "+12187021098", Token: "telnyx-key"}

	err := s.service.deliver(s.ctx, line, provider, "+13479018418", "+13479018418",
		Reply{Text: "Here it is.", Files: []File{{URL: "https://files/render.png"}}}, s.owner())

	s.Require().Error(err, "the reply did not all go out")
	row := s.row()
	s.True(row.Success)
	s.True(row.Billable)
	s.Equal(pricing.Quantities{pricing.Messages: 1, pricing.MessageSegments: 1}, row.Meters, "the words, which went out")
	s.Equal(int64(7_800), row.CostMicros)
}

func (s *TextMeteringSuite) TestAnMMSThatArrivedIsBilledAsOne() {
	account := store.ChannelAccount{CustomerID: s.customer, Kind: string(SMS), E164: "+12187021098"}

	s.service.received(s.ctx, account, Message{Kind: SMS, ID: "mms-" + uuid.NewString(), From: "+13479018418", Media: 2})

	row := s.row()
	s.Equal(pricing.Quantities{pricing.Messages: 1, pricing.MediaMessages: 1}, row.Meters)
	s.Equal(int64(5_000), row.CostMicros, "Telnyx's $0.005 an MMS received")
}

// row waits for the customer's one text row.
func (s *TextMeteringSuite) row() store.Request {
	var rows []store.Request
	s.Require().Eventually(func() bool {
		rows = nil
		s.Require().NoError(s.store.ShardDB().NewSelect().Model(&rows).
			Where("customer_id = ?", s.customer).Where("modality = ?", routing.Phone).Scan(s.ctx))
		return len(rows) > 0
	}, 5*time.Second, 25*time.Millisecond, "no text row")
	s.Require().Len(rows, 1)
	return rows[0]
}
