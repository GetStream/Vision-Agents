//go:build integration

package channelbridge

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/eventforward"
	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// chatKey is the api key the suite's Stream clients call chattest with.
const chatKey = "test"

// InboundSuite is the worker that writes queued inbound messages, against Postgres and the
// suite's Stream Chat (chattest): what it writes once, what it writes again, what it gives up,
// and how it starts and stops.
type InboundSuite struct {
	suite.Suite
	ctx        context.Context
	store      *store.Store
	sealer     *auth.Sealer
	transports *core.Transports
	resolver   *resolver.Resolver
	chat       *chattest.Server
	stream     *streamapp.Clients
	logs       *syncBuffer
	customer   string
}

func TestInboundSuite(t *testing.T) {
	suite.Run(t, new(InboundSuite))
}

func (s *InboundSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	// A database of this suite's own, as the store suite has, since it drops the schema.
	db, err := store.Open(testenv.Database(dsn, "bridge"))
	s.Require().NoError(err)
	s.store = db
	var database string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&database))
	s.Require().True(strings.HasSuffix(database, "_test"), "refusing to drop the schema of %s, which is not a test database", database)
	_, err = db.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.sealer, err = auth.NewSealerWithKeyring(1, map[int]string{1: "test key one"})
	s.Require().NoError(err)
	credentials, err := pgsealed.New(db, s.sealer)
	s.Require().NoError(err)
	s.resolver, err = resolver.New(resolver.Config{Store: db, Credentials: credentials})
	s.Require().NoError(err)
	s.transports, err = core.NewTransports(core.TransportsConfig{Resolver: s.resolver})
	s.Require().NoError(err)
}

func (s *InboundSuite) TearDownSuite() {
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
}

func (s *InboundSuite) SetupTest() {
	s.chat = chattest.NewServer(s.T())
	s.stream = streamapp.NewClients(streamapp.NewDeployment(streamapp.DeploymentOptions{
		APIKey: chatKey, Secret: "secret", BaseURL: s.chat.URL,
	}), streamapp.ClientsOptions{})
	s.logs = &syncBuffer{}
	s.customer = "customer-" + uuid.NewString()
}

func (s *InboundSuite) TestAQueuedMessageIsWrittenOnceUnderItsOwnID() {
	thread := s.thread("fake_line")
	s.queue(thread, "m-1", "hello", nil)

	s.bridge(nil)

	s.Require().Eventually(func() bool { return s.row(thread, "m-1").State == store.ChannelInboundDone }, 5*time.Second, 10*time.Millisecond)
	stored := s.chat.Stored(thread.ChannelID)
	s.Require().Len(stored, 1)
	s.Equal(thread.ChannelID+"-m-1", stored[0]["id"])
	s.Equal("hello", stored[0]["text"])
	s.Equal(1, s.writes(thread.ChannelID))
}

// An attempt whose answer was lost, or whose lease ran out while Stream Chat stored it, left
// the message there: the next attempt is refused with code 4, and that is the message written.
func (s *InboundSuite) TestAMessageStreamChatAlreadyHoldsCountsAsWritten() {
	thread := s.thread("fake_line")
	s.writeAhead(thread, inboundMessageID(thread.ChannelID, "m-1"), "hello")
	s.queue(thread, "m-1", "hello", nil)

	s.bridge(nil)

	s.Require().Eventually(func() bool { return s.row(thread, "m-1").State == store.ChannelInboundDone }, 5*time.Second, 10*time.Millisecond)
	s.Equal(0, s.row(thread, "m-1").Attempts)
	s.Len(s.chat.Stored(thread.ChannelID), 1)
}

// The first attempt opens the episode, and its write fails before the card is written; the
// retry finds the episode open and writes its card all the same.
func (s *InboundSuite) TestARetryWritesTheCardOfTheEpisodeAnEarlierAttemptOpened() {
	thread := s.thread("slack_bot")
	s.chat.RateLimit("/" + thread.ChannelID + "/message")
	s.queue(thread, "1793125611.816139", "hello", nil)
	s.bridge(&Options{InboundRetries: []time.Duration{50 * time.Millisecond, time.Hour}})
	s.Require().Eventually(func() bool { return s.row(thread, "1793125611.816139").Attempts == 1 }, 5*time.Second, 10*time.Millisecond)

	s.chat.LiftRateLimit("/" + thread.ChannelID + "/message")

	s.Require().Eventually(func() bool { return s.row(thread, "1793125611.816139").State == store.ChannelInboundDone }, 5*time.Second, 10*time.Millisecond)
	omni, card := s.card(thread)
	s.Equal([]string{card}, messageIDs(s.chat.Stored(omni)))
}

// A router whose clock is behind Postgres's opens the episode, and its write fails before the
// card is written: the retry writes the card all the same, since the episode starts at the
// time Postgres queued the message, whatever the router's clock says.
func (s *InboundSuite) TestARetryWritesTheCardWhenTheRoutersClockIsBehindPostgres() {
	thread := s.thread("slack_bot")
	s.chat.RateLimit("/" + thread.ChannelID + "/message")
	s.queue(thread, "1793125611.816139", "hello", nil)
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE channel_thread_inbound SET created_at = created_at + interval '1 hour' WHERE channel_id = ?", thread.ChannelID)
	s.Require().NoError(err)
	s.bridge(&Options{InboundRetries: []time.Duration{50 * time.Millisecond, time.Hour}})
	s.Require().Eventually(func() bool { return s.row(thread, "1793125611.816139").Attempts == 1 }, 5*time.Second, 10*time.Millisecond)

	s.chat.LiftRateLimit("/" + thread.ChannelID + "/message")

	s.Require().Eventually(func() bool { return s.row(thread, "1793125611.816139").State == store.ChannelInboundDone }, 5*time.Second, 10*time.Millisecond)
	omni, card := s.card(thread)
	s.Equal([]string{card}, messageIDs(s.chat.Stored(omni)))
}

// A router opened the episode and stopped before its card was written, as its lease ran out:
// the attempt that writes the message after it writes the card all the same.
func (s *InboundSuite) TestACardAStoppedRouterLeftIsWrittenByTheNextAttempt() {
	thread := s.thread("slack_bot")
	s.queue(thread, "1793125611.816139", "hello", nil)
	s.openEpisode(thread, s.row(thread, "1793125611.816139"))

	s.bridge(nil)

	s.Require().Eventually(func() bool { return s.row(thread, "1793125611.816139").State == store.ChannelInboundDone }, 5*time.Second, 10*time.Millisecond)
	s.Equal(0, s.row(thread, "1793125611.816139").Attempts)
	omni, card := s.card(thread)
	s.Equal([]string{card}, messageIDs(s.chat.Stored(omni)))
}

// A message whose card is in the omni-channel already, written by an attempt whose answer was
// lost or whose lease ran out, is written; the card stays one.
func (s *InboundSuite) TestACardStreamChatAlreadyHoldsCountsAsWritten() {
	thread := s.thread("slack_bot")
	s.queue(thread, "1793125611.816139", "hello", nil)
	opened := s.openEpisode(thread, s.row(thread, "1793125611.816139"))
	opened.New = true
	s.Require().NoError(s.newBridge(nil).cards.Write(s.ctx, opened))

	s.bridge(nil)

	s.Require().Eventually(func() bool { return s.row(thread, "1793125611.816139").State == store.ChannelInboundDone }, 5*time.Second, 10*time.Millisecond)
	s.Equal(0, s.row(thread, "1793125611.816139").Attempts, "written on its first attempt")
	omni, card := s.card(thread)
	s.Equal([]string{card}, messageIDs(s.chat.Stored(omni)))
}

// A later message of a thread finds its episode open with its card written, and writes nothing
// into the omni-channel, on its first attempt as on a retry.
func (s *InboundSuite) TestALaterMessageOfAnOpenEpisodeWritesNoCard() {
	thread := s.thread("slack_bot")
	s.queue(thread, "1793125611.000001", "first", nil)
	bridge := s.bridge(nil)
	s.Require().Eventually(func() bool { return s.row(thread, "1793125611.000001").State == store.ChannelInboundDone }, 5*time.Second, 10*time.Millisecond)
	omni, card := s.card(thread)
	asked := s.asked(omni)

	s.queue(thread, "1793125611.000002", "second", nil)
	bridge.wake()

	s.Require().Eventually(func() bool { return s.row(thread, "1793125611.000002").State == store.ChannelInboundDone }, 5*time.Second, 10*time.Millisecond)
	s.Equal(0, s.row(thread, "1793125611.000002").Attempts)
	s.Equal(asked, s.asked(omni), "no request to the omni-channel")
	s.Equal([]string{card}, messageIDs(s.chat.Stored(omni)))
	s.Equal([]string{"first", "second"}, s.chat.Messages(thread.ChannelID))
}

// A message whose episode another message opened after it was queued, as Deliver does at the
// router's time, finds it open with its card written, and writes nothing into the omni-channel.
func (s *InboundSuite) TestAMessageOfAnEpisodeOpenedAfterItWasQueuedWritesNoCard() {
	thread := s.thread("slack_bot")
	s.queue(thread, "1793125611.000002", "second", nil)
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE channel_thread_inbound SET created_at = created_at - interval '1 hour' WHERE channel_id = ?", thread.ChannelID)
	s.Require().NoError(err)
	message := s.row(thread, "1793125611.000002")
	bridge := s.newBridge(nil)
	opened, err := bridge.episode(s.ctx, thread, store.AgentConfig{ID: message.ConfigID, Name: message.ConfigName}, message.Message(), time.Time{})
	s.Require().NoError(err)
	s.Require().NoError(bridge.cards.Write(s.ctx, opened))
	omni, card := s.card(thread)
	asked := s.asked(omni)

	s.bridge(nil)

	s.Require().Eventually(func() bool { return s.row(thread, "1793125611.000002").State == store.ChannelInboundDone }, 5*time.Second, 10*time.Millisecond)
	s.Equal(asked, s.asked(omni), "no request to the omni-channel")
	s.Equal([]string{card}, messageIDs(s.chat.Stored(omni)))
}

// Each failed write is due again after the next wait of the schedule.
func (s *InboundSuite) TestAFailedWriteIsDueAgainAfterEachWait() {
	thread := s.thread("fake_line")
	s.chat.RateLimit("/" + thread.ChannelID + "/message")
	s.queue(thread, "m-1", "hello", nil)

	s.bridge(&Options{InboundRetries: []time.Duration{50 * time.Millisecond, time.Hour, time.Hour}})

	s.Require().Eventually(func() bool { return s.row(thread, "m-1").Attempts == 2 }, 5*time.Second, 10*time.Millisecond)
	row := s.row(thread, "m-1")
	s.Equal(store.ChannelInboundQueued, row.State)
	s.WithinDuration(time.Now().Add(time.Hour), row.NextAttemptAt, time.Minute)
	s.Empty(s.chat.Stored(thread.ChannelID))
}

// A bridge given no retries waits the first of defaultInboundRetries.
func (s *InboundSuite) TestAFailedWriteIsDueAgainAfterTwoSecondsByDefault() {
	thread := s.thread("fake_line")
	s.chat.RateLimit("/" + thread.ChannelID + "/message")
	s.queue(thread, "m-1", "hello", nil)

	s.bridge(nil)

	s.Require().Eventually(func() bool { return s.row(thread, "m-1").Attempts == 1 }, 5*time.Second, 10*time.Millisecond)
	row := s.row(thread, "m-1")
	s.Equal(store.ChannelInboundQueued, row.State)
	s.WithinDuration(time.Now().Add(2*time.Second), row.NextAttemptAt, time.Second)
}

func (s *InboundSuite) TestAMessageWrittenOnARetryIsWrittenOnce() {
	thread := s.thread("fake_line")
	s.chat.RateLimit("/" + thread.ChannelID + "/message")
	s.queue(thread, "m-1", "hello", nil)
	s.bridge(&Options{InboundRetries: []time.Duration{50 * time.Millisecond, 50 * time.Millisecond, time.Hour}})
	s.Require().Eventually(func() bool { return s.row(thread, "m-1").Attempts >= 1 }, 5*time.Second, 10*time.Millisecond)

	s.chat.LiftRateLimit("/" + thread.ChannelID + "/message")

	s.Require().Eventually(func() bool { return s.row(thread, "m-1").State == store.ChannelInboundDone }, 5*time.Second, 10*time.Millisecond)
	s.Equal([]string{"hello"}, s.chat.Messages(thread.ChannelID))
	s.Equal(s.writes(thread.ChannelID), s.writes(""), "a connector with no card writes none on a retry")
}

// After the first attempt and one for each wait the message is given up: failed, a warning,
// and its delivery goes to the customer's destinations of unhandled events, once for the two
// messages of one delivery.
func (s *InboundSuite) TestAMessageThatNeverGetsThroughIsGivenUpAndItsDeliveryForwardedOnce() {
	forwarder, err := eventforward.New(eventforward.Options{Store: s.store, Secrets: s.sealer})
	s.Require().NoError(err)
	destination := s.destination(forwarder)
	thread := s.thread("fake_line")
	s.chat.RateLimit("/" + thread.ChannelID + "/message")
	until := time.Now().Add(5 * time.Minute).UTC().Truncate(time.Second)
	forward := &store.ChannelInboundForward{ID: "Ev0001", Headers: map[string]string{"Content-Type": "application/json"}, HeadersUntil: &until}
	s.queue(thread, "m-1", "hello", forward)
	s.queue(thread, "m-2", "again", forward)

	s.bridge(&Options{InboundRetries: []time.Duration{time.Millisecond, time.Millisecond}, Unanswered: forwarder.ForwardUnanswered})

	s.Require().Eventually(func() bool { return s.row(thread, "m-2").State == store.ChannelInboundFailed }, 5*time.Second, 10*time.Millisecond)
	s.Equal(store.ChannelInboundFailed, s.row(thread, "m-1").State)
	s.Equal(3, s.row(thread, "m-1").Attempts, "the first attempt and the two waits")
	s.Equal(2, strings.Count(s.logs.String(), "level=WARN msg=\"gave up an inbound message"))
	var deliveries []store.EventDelivery
	s.Require().NoError(s.store.DB().NewSelect().Model(&deliveries).Where("destination_id = ?", destination).Scan(s.ctx))
	s.Require().Len(deliveries, 1)
	s.Equal([]byte(`{"text":"hello"}`), deliveries[0].Body)
	s.Equal(forward.Headers, deliveries[0].Headers)
	s.Require().NotNil(deliveries[0].ProviderHeadersUntil)
	s.True(until.Equal(*deliveries[0].ProviderHeadersUntil))
}

// A delivery whose forward fails when its first message is given up, as when the router stops
// before it, is forwarded by its next message given up, once.
func (s *InboundSuite) TestADeliveryWhoseForwardFailedIsForwardedByItsNextMessageGivenUp() {
	forwarder, err := eventforward.New(eventforward.Options{Store: s.store, Secrets: s.sealer})
	s.Require().NoError(err)
	destination := s.destination(forwarder)
	thread := s.thread("fake_line")
	s.chat.RateLimit("/" + thread.ChannelID + "/message")
	forward := &store.ChannelInboundForward{ID: "Ev0001"}
	s.queue(thread, "m-1", "hello", forward)
	s.queue(thread, "m-2", "again", forward)
	s.queue(thread, "m-3", "and again", forward)
	var calls atomic.Int32
	unanswered := func(ctx context.Context, event eventforward.Event) error {
		if calls.Add(1) == 1 {
			return errors.New("synthetic forward failure")
		}
		return forwarder.ForwardUnanswered(ctx, event)
	}

	s.bridge(&Options{InboundRetries: []time.Duration{time.Millisecond}, Unanswered: unanswered})

	s.Require().Eventually(func() bool { return s.row(thread, "m-3").State == store.ChannelInboundFailed }, 5*time.Second, 10*time.Millisecond)
	s.Equal(store.ChannelInboundFailed, s.row(thread, "m-1").State, "given up although its forward failed")
	s.Contains(s.logs.String(), "synthetic forward failure")
	var deliveries []store.EventDelivery
	s.Require().NoError(s.store.DB().NewSelect().Model(&deliveries).Where("destination_id = ?", destination).Scan(s.ctx))
	s.Require().Len(deliveries, 1)
	s.Equal([]byte(`{"text":"again"}`), deliveries[0].Body, "the second message forwards it, the third does not")
	s.Equal(int32(2), calls.Load())
}

// Stream Chat refusing a message with a 4xx other than 429 refuses it again: it is given up on
// its first attempt, with its warning and its forward, and does not hold its thread.
func (s *InboundSuite) TestAMessageStreamChatRefusesIsGivenUpAtOnce() {
	forwarder, err := eventforward.New(eventforward.Options{Store: s.store, Secrets: s.sealer})
	s.Require().NoError(err)
	destination := s.destination(forwarder)
	thread := s.thread("fake_line")
	s.answerWith(thread.ChannelID+"/message", http.StatusBadRequest, `{"code":4,"message":"SendMessage failed with error: \"text is too long\"","StatusCode":400}`)
	s.queue(thread, "m-1", "hello", &store.ChannelInboundForward{ID: "Ev0001"})

	s.bridge(&Options{InboundRetries: []time.Duration{time.Hour}, Unanswered: forwarder.ForwardUnanswered})

	s.Require().Eventually(func() bool { return s.row(thread, "m-1").State == store.ChannelInboundFailed }, 5*time.Second, 10*time.Millisecond)
	s.Equal(1, s.row(thread, "m-1").Attempts)
	s.Equal(1, strings.Count(s.logs.String(), "level=WARN msg=\"gave up an inbound message"))
	count, err := s.store.DB().NewSelect().Model((*store.EventDelivery)(nil)).Where("destination_id = ?", destination).Count(s.ctx)
	s.Require().NoError(err)
	s.Equal(1, count)
}

// Stream Chat answering 5xx may take the message later: it is due again after the first wait.
func (s *InboundSuite) TestAMessageStreamChatFailsWith5xxIsDueAgain() {
	thread := s.thread("fake_line")
	s.answerWith(thread.ChannelID+"/message", http.StatusInternalServerError, `{"code":-1,"message":"internal error","StatusCode":500}`)
	s.queue(thread, "m-1", "hello", nil)

	s.bridge(&Options{InboundRetries: []time.Duration{time.Hour}})

	s.Require().Eventually(func() bool { return s.row(thread, "m-1").Attempts == 1 }, 5*time.Second, 10*time.Millisecond)
	row := s.row(thread, "m-1")
	s.Equal(store.ChannelInboundQueued, row.State)
	s.WithinDuration(time.Now().Add(time.Hour), row.NextAttemptAt, time.Minute)
}

// A worker whose lease ran out while it wrote a message on its last attempt records nothing:
// another worker took the message and may write it, so this one neither gives it up nor
// forwards its delivery.
func (s *InboundSuite) TestAWorkerThatLostItsLeaseGivesNothingUp() {
	forwarder, err := eventforward.New(eventforward.Options{Store: s.store, Secrets: s.sealer})
	s.Require().NoError(err)
	destination := s.destination(forwarder)
	thread := s.thread("fake_line")
	s.chat.RateLimit("/" + thread.ChannelID + "/message")
	s.queue(thread, "m-1", "hello", &store.ChannelInboundForward{ID: "Ev0001"})
	_, err = s.store.DB().NewUpdate().Model((*store.ChannelInbound)(nil)).Set("attempts = 1").
		Where("channel_id = ? AND provider_message_id = ?", thread.ChannelID, "m-1").Exec(s.ctx)
	s.Require().NoError(err)
	stale, err := s.store.ClaimChannelInbound(s.ctx, time.Now(), 1, time.Now().Add(time.Millisecond))
	s.Require().NoError(err)
	s.Require().Len(stale, 1)
	time.Sleep(5 * time.Millisecond)
	taken, err := s.store.ClaimChannelInbound(s.ctx, time.Now(), 1, time.Now().Add(time.Hour))
	s.Require().NoError(err)
	s.Require().Len(taken, 1, "another worker took it once the lease ran out")

	s.newBridge(&Options{InboundRetries: []time.Duration{time.Millisecond}, Unanswered: forwarder.ForwardUnanswered}).attempt(stale[0])

	row := s.row(thread, "m-1")
	s.Equal(store.ChannelInboundQueued, row.State)
	s.Equal(1, row.Attempts)
	s.NotContains(s.logs.String(), "level=WARN")
	count, err := s.store.DB().NewSelect().Model((*store.EventDelivery)(nil)).Where("destination_id = ?", destination).Count(s.ctx)
	s.Require().NoError(err)
	s.Zero(count)
}

// A STOP whose opt-out the store keeps failing to record is never given up: past the waits it
// is due again after the last one. A trigger refuses the test customer's opt-outs.
func (s *InboundSuite) TestAStopPastItsWaitsIsDueAgainAfterTheLastWait() {
	_, err := s.store.DB().ExecContext(s.ctx, `BEGIN;
		CREATE FUNCTION inbound_suite_refuse_opt_out() RETURNS trigger AS $$
			BEGIN RAISE EXCEPTION 'refused by the test'; END $$ LANGUAGE plpgsql;
		CREATE TRIGGER inbound_suite_refuse_opt_out BEFORE INSERT ON opt_outs FOR EACH ROW
			WHEN (NEW.customer_id = '`+s.customer+`') EXECUTE FUNCTION inbound_suite_refuse_opt_out();
		COMMIT`)
	s.Require().NoError(err)
	defer func() {
		_, err := s.store.DB().ExecContext(s.ctx, `BEGIN;
			DROP TRIGGER inbound_suite_refuse_opt_out ON opt_outs;
			DROP FUNCTION inbound_suite_refuse_opt_out();
			COMMIT`)
		s.Require().NoError(err)
	}()
	s.emptyQueue()
	thread := s.thread("telnyx")
	s.queue(thread, "m-1", "STOP", nil)
	_, err = s.store.DB().NewUpdate().Model((*store.ChannelInbound)(nil)).Set("attempts = 2").
		Where("channel_id = ? AND provider_message_id = ?", thread.ChannelID, "m-1").Exec(s.ctx)
	s.Require().NoError(err)
	claimed, err := s.store.ClaimChannelInbound(s.ctx, time.Now(), 1, time.Now().Add(time.Minute))
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)

	s.newBridge(&Options{InboundRetries: []time.Duration{time.Millisecond, time.Hour}}).attempt(claimed[0])

	row := s.row(thread, "m-1")
	s.Equal(store.ChannelInboundQueued, row.State, "the STOP is not given up")
	s.Equal(3, row.Attempts)
	s.WithinDuration(time.Now().Add(time.Hour), row.NextAttemptAt, time.Minute, "it is due again after the last wait")
	s.Equal(1, strings.Count(s.logs.String(), "level=WARN msg=\"an opt-out is still not recorded"), "past its waits it logs at WARN")
}

// The WARN for a STOP that is still not recorded starts past the waits: attempt 1 and the attempt
// that uses the last wait are retried at INFO only.
func (s *InboundSuite) TestAStopWithinItsWaitsIsRetriedWithoutAWarning() {
	_, err := s.store.DB().ExecContext(s.ctx, `BEGIN;
		CREATE FUNCTION inbound_suite_refuse_opt_out() RETURNS trigger AS $$
			BEGIN RAISE EXCEPTION 'refused by the test'; END $$ LANGUAGE plpgsql;
		CREATE TRIGGER inbound_suite_refuse_opt_out BEFORE INSERT ON opt_outs FOR EACH ROW
			WHEN (NEW.customer_id = '`+s.customer+`') EXECUTE FUNCTION inbound_suite_refuse_opt_out();
		COMMIT`)
	s.Require().NoError(err)
	defer func() {
		_, err := s.store.DB().ExecContext(s.ctx, `BEGIN;
			DROP TRIGGER inbound_suite_refuse_opt_out ON opt_outs;
			DROP FUNCTION inbound_suite_refuse_opt_out();
			COMMIT`)
		s.Require().NoError(err)
	}()
	s.emptyQueue()
	thread := s.thread("telnyx")
	bridge := s.newBridge(&Options{InboundRetries: []time.Duration{time.Millisecond, time.Hour}})
	warnings := func() int { return strings.Count(s.logs.String(), "level=WARN") }

	for _, tried := range []int{0, 1, 2} {
		id := fmt.Sprintf("m-%d", tried)
		s.emptyQueue()
		s.queue(thread, id, "STOP", nil)
		_, err = s.store.DB().NewUpdate().Model((*store.ChannelInbound)(nil)).Set("attempts = ?", tried).
			Where("channel_id = ? AND provider_message_id = ?", thread.ChannelID, id).Exec(s.ctx)
		s.Require().NoError(err)
		claimed, err := s.store.ClaimChannelInbound(s.ctx, time.Now(), 1, time.Now().Add(time.Minute))
		s.Require().NoError(err)
		s.Require().Len(claimed, 1)
		before := warnings()

		bridge.attempt(claimed[0])

		s.Equal(store.ChannelInboundQueued, s.row(thread, id).State)
		if tried+1 > 2 {
			s.Equal(before+1, warnings(), "attempt %d is past the waits: one WARN", tried+1)
		} else {
			s.Equal(before, warnings(), "attempt %d is within the waits: no WARN", tried+1)
		}
	}
}

// A STOP is held for its opt-out channel (optsOut); a "stop" on a connector with none is not,
// so it is given up after its waits like any other message.
func (s *InboundSuite) TestAStopOnAConnectorWithNoOptOutChannelIsGivenUp() {
	thread := s.thread("slack_bot")
	s.chat.RateLimit("/" + thread.ChannelID + "/message")
	s.queue(thread, "m-1", "stop", nil)

	s.bridge(&Options{InboundRetries: []time.Duration{time.Millisecond, time.Millisecond}})

	s.Require().Eventually(func() bool { return s.row(thread, "m-1").State == store.ChannelInboundFailed }, 5*time.Second, 10*time.Millisecond)
	s.Equal(3, s.row(thread, "m-1").Attempts)
}

// A message whose delivery no destination would take, such as a reply that waited, is given up
// with no forward.
func (s *InboundSuite) TestAGivenUpMessageWithNoForwardForwardsNothing() {
	forwarder, err := eventforward.New(eventforward.Options{Store: s.store, Secrets: s.sealer})
	s.Require().NoError(err)
	destination := s.destination(forwarder)
	thread := s.thread("fake_line")
	s.chat.RateLimit("/" + thread.ChannelID + "/message")
	s.queue(thread, "m-1", "hello", nil)

	s.bridge(&Options{InboundRetries: []time.Duration{time.Millisecond}, Unanswered: forwarder.ForwardUnanswered})

	s.Require().Eventually(func() bool { return s.row(thread, "m-1").State == store.ChannelInboundFailed }, 5*time.Second, 10*time.Millisecond)
	count, err := s.store.DB().NewSelect().Model((*store.EventDelivery)(nil)).Where("destination_id = ?", destination).Count(s.ctx)
	s.Require().NoError(err)
	s.Zero(count)
}

// A router stops while a message waits for its retry; the router started after it writes it.
func (s *InboundSuite) TestAMessageAStoppedWorkerLeftIsWrittenByTheNextOne() {
	thread := s.thread("fake_line")
	s.chat.RateLimit("/" + thread.ChannelID + "/message")
	s.queue(thread, "m-1", "hello", nil)
	first := s.bridge(&Options{InboundRetries: []time.Duration{50 * time.Millisecond, 50 * time.Millisecond, time.Hour}})
	s.Require().Eventually(func() bool { return s.row(thread, "m-1").Attempts >= 1 }, 5*time.Second, 10*time.Millisecond)
	first.Close()
	s.chat.LiftRateLimit("/" + thread.ChannelID + "/message")

	s.bridge(nil)

	s.Require().Eventually(func() bool { return s.row(thread, "m-1").State == store.ChannelInboundDone }, 5*time.Second, 10*time.Millisecond)
	s.Equal([]string{"hello"}, s.chat.Messages(thread.ChannelID))
}

// A router took a message and stopped before it wrote it: an idle worker writes it once the
// lease runs out, not at its next look a lease later.
func (s *InboundSuite) TestAMessageALeaseHeldIsWrittenWhenTheLeaseRunsOut() {
	thread := s.thread("fake_line")
	s.queue(thread, "m-1", "hello", nil)
	claimed, err := s.store.ClaimChannelInbound(s.ctx, time.Now(), 1, time.Now().Add(300*time.Millisecond))
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)

	s.bridge(nil)

	s.Require().Eventually(func() bool { return s.row(thread, "m-1").State == store.ChannelInboundDone }, 5*time.Second, 10*time.Millisecond)
	s.Equal([]string{"hello"}, s.chat.Messages(thread.ChannelID))
}

// An idle worker waits a lease between looks; a message queued meanwhile is written when the
// worker is woken, not then.
func (s *InboundSuite) TestAWakeWritesAMessageQueuedIntoAnIdleQueue() {
	bridge := s.bridge(nil)
	time.Sleep(50 * time.Millisecond) // the worker's first look found nothing and it waits a lease
	thread := s.thread("fake_line")
	s.queue(thread, "m-1", "hello", nil)

	bridge.wake()

	s.Require().Eventually(func() bool { return s.row(thread, "m-1").State == store.ChannelInboundDone }, 5*time.Second, 10*time.Millisecond)
}

// Two routers write every message once, and each thread's messages in the order they were
// queued. 36 messages are more than the two routers' 32 slots, so each slot is used again.
func (s *InboundSuite) TestTwoWorkersWriteEachMessageOnceAndEachThreadInOrder() {
	threads := []store.ChannelThread{s.thread("fake_line"), s.thread("fake_line"), s.thread("fake_line")}
	var want []string
	for i := range 12 {
		want = append(want, fmt.Sprintf("message %d", i))
		for _, thread := range threads {
			s.queue(thread, fmt.Sprintf("m-%d", i), want[i], nil)
		}
	}

	s.bridge(nil)
	s.bridge(nil)

	for _, thread := range threads {
		s.Require().Eventually(func() bool { return s.row(thread, "m-11").State == store.ChannelInboundDone }, 10*time.Second, 10*time.Millisecond)
		s.Equal(want, s.chat.Messages(thread.ChannelID))
		s.Equal(12, s.writes(thread.ChannelID), "no message was sent twice")
	}
}

// Close lets a write in flight finish and record itself, and the worker takes nothing after.
func (s *InboundSuite) TestCloseWaitsForAWriteInFlightAndStopsTheWorker() {
	thread := s.thread("fake_line")
	waiting, release := s.chat.Hold("/" + thread.ChannelID + "/message")
	s.queue(thread, "m-1", "hello", nil)
	bridge := s.bridge(nil)
	<-waiting
	closed := make(chan struct{})
	go func() {
		bridge.Close()
		close(closed)
	}()
	s.Never(func() bool { return isClosed(closed) }, 200*time.Millisecond, 10*time.Millisecond, "Close did not wait for the write")

	release()

	s.Require().Eventually(func() bool { return isClosed(closed) }, 5*time.Second, 10*time.Millisecond)
	s.Equal(store.ChannelInboundDone, s.row(thread, "m-1").State)
	s.queue(thread, "m-2", "after", nil)
	bridge.wake()
	s.Never(func() bool { return s.row(thread, "m-2").State != store.ChannelInboundQueued }, 300*time.Millisecond, 20*time.Millisecond)
	s.Equal([]string{"hello"}, s.chat.Messages(thread.ChannelID))
}

func (s *InboundSuite) TestAWorkerWithNothingQueuedCallsNoStreamChat() {
	bridge := s.bridge(nil)
	for range 5 {
		bridge.wake()
		time.Sleep(20 * time.Millisecond)
	}

	s.Empty(s.chat.Requests(chatKey))
}

// With nothing queued a worker looks again a lease later, not at the poll.
func (s *InboundSuite) TestAWorkerWithNothingQueuedLooksAgainALeaseLater() {
	s.emptyQueue()

	wait, again := s.newBridge(nil).writeDue()

	s.True(again)
	s.Equal(inboundLease, wait)
}

// A worker deletes the rows past their keep at its first look, and not again within the hour.
func (s *InboundSuite) TestTheWorkerDeletesRowsPastTheirKeepOnceAnHour() {
	s.emptyQueue()
	bridge := s.newBridge(nil)
	bridge.writeDue()
	thread := s.thread("fake_line")
	s.queue(thread, "m-1", "hello", nil)
	_, err := s.store.DB().NewUpdate().Model((*store.ChannelInbound)(nil)).
		Set("state = 'done'").Set("finished_at = ?", time.Now().Add(-25*time.Hour)).
		Where("channel_id = ? AND provider_message_id = 'm-1'", thread.ChannelID).Exec(s.ctx)
	s.Require().NoError(err)

	bridge.writeDue()

	s.Equal(1, s.rows(thread), "the next prune is an hour after the first")
}

// A written message's row is kept a day, so a provider's retry queues nothing; after that the
// worker deletes it.
func (s *InboundSuite) TestTheWorkerDeletesAWrittenMessagesRowAfterADay() {
	thread := s.thread("fake_line")
	s.queue(thread, "m-1", "hello", nil)
	s.queue(thread, "m-2", "hello", nil)
	_, err := s.store.DB().NewUpdate().Model((*store.ChannelInbound)(nil)).
		Set("state = 'done'").Set("finished_at = ?", time.Now().Add(-25*time.Hour)).
		Where("channel_id = ? AND provider_message_id = 'm-1'", thread.ChannelID).Exec(s.ctx)
	s.Require().NoError(err)

	s.bridge(nil)

	s.Require().Eventually(func() bool { return s.row(thread, "m-2").State == store.ChannelInboundDone }, 5*time.Second, 10*time.Millisecond)
	s.Equal(1, s.rows(thread), "the day-old row is gone, the fresh one kept")
}

// bridge is a started Bridge over the suite's store and Stream Chat, closed with the test.
// options, when not nil, set the inbound retries and the forward of what it gives up.
func (s *InboundSuite) bridge(options *Options) *Bridge {
	bridge := s.newBridge(options)
	bridge.Start(nil)
	return bridge
}

// newBridge is a Bridge as bridge makes it, whose worker is not started.
func (s *InboundSuite) newBridge(options *Options) *Bridge {
	all := Options{
		Store: s.store, Stream: s.stream, Transports: s.transports, Resolver: s.resolver,
		Logger: slog.New(slog.NewTextHandler(s.logs, nil)), InboundPoll: 10 * time.Millisecond,
	}
	if options != nil {
		all.InboundRetries, all.Unanswered = options.InboundRetries, options.Unanswered
	}
	bridge, err := New(all)
	s.Require().NoError(err)
	s.T().Cleanup(bridge.Close)
	return bridge
}

// answerWith answers every request to the suite's Stream Chat whose path ends in suffix with
// status and body, for the bridges made after it; chattest answers the rest.
func (s *InboundSuite) answerWith(suffix string, status int, body string) {
	target, err := url.Parse(s.chat.URL)
	s.Require().NoError(err)
	proxy := httputil.NewSingleHostReverseProxy(target)
	front := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, suffix) {
			proxy.ServeHTTP(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	s.T().Cleanup(front.Close)
	s.stream = streamapp.NewClients(streamapp.NewDeployment(streamapp.DeploymentOptions{
		APIKey: chatKey, Secret: "secret", BaseURL: front.URL,
	}), streamapp.ClientsOptions{})
}

// openEpisode opens the episode of a queued message as a worker does before it writes the
// message, and returns what Open returned.
func (s *InboundSuite) openEpisode(thread store.ChannelThread, message store.ChannelInbound) omnichannel.Opened {
	opened, err := s.newBridge(nil).episode(s.ctx, thread, store.AgentConfig{ID: message.ConfigID, Name: message.ConfigName}, message.Message(), message.CreatedAt)
	s.Require().NoError(err)
	s.Require().True(opened.New)
	return opened
}

// emptyQueue deletes every row of the suite's own database's queue, the ones earlier tests
// left too.
func (s *InboundSuite) emptyQueue() {
	_, err := s.store.DB().NewDelete().Model((*store.ChannelInbound)(nil)).Where("TRUE").Exec(s.ctx)
	s.Require().NoError(err)
}

// asked is how many requests the suite's Stream Chat had about a channel.
func (s *InboundSuite) asked(channelID string) int {
	count := 0
	for _, request := range s.chat.Requests(chatKey) {
		if strings.Contains(request.Path, "/"+channelID) {
			count++
		}
	}
	return count
}

// thread links a thread channel of the test's customer on connector.
func (s *InboundSuite) thread(connector string) store.ChannelThread {
	thread := store.ChannelThread{
		ChannelID: threadChannelPrefix + uuid.NewString(), CustomerID: s.customer, ConnectorID: connector,
		ProviderUnitID: "T0000TEAM", ThreadKey: uuid.NewString(), ConnectionID: "connection-1",
	}
	_, err := s.store.LinkChannelThread(s.ctx, &thread)
	s.Require().NoError(err)
	return thread
}

// queue queues a message of U0000ALICE in thread, due now.
func (s *InboundSuite) queue(thread store.ChannelThread, providerMessageID, text string, forward *store.ChannelInboundForward) {
	s.queueAt(thread, providerMessageID, text, forward, time.Now())
}

// queueAt queues a message of U0000ALICE in thread, due at due.
func (s *InboundSuite) queueAt(thread store.ChannelThread, providerMessageID, text string, forward *store.ChannelInboundForward, due time.Time) {
	queued, err := s.store.AcceptChannelInbound(s.ctx, store.ChannelInbound{
		ChannelID: thread.ChannelID, ProviderMessageID: providerMessageID, CustomerID: thread.CustomerID,
		ConnectorID: thread.ConnectorID, ProviderUnitID: thread.ProviderUnitID, ThreadKey: thread.ThreadKey,
		ConfigID: "config-1", ConfigName: "Support", AuthorID: "U0000ALICE", Text: text,
		Raw: []byte(`{"text":"` + text + `"}`), Forward: forward, NextAttemptAt: due,
	}, nil)
	s.Require().NoError(err)
	s.Require().True(queued)
}

// row is a queued message's row as it is now.
func (s *InboundSuite) row(thread store.ChannelThread, providerMessageID string) store.ChannelInbound {
	var row store.ChannelInbound
	s.Require().NoError(s.store.DB().NewSelect().Model(&row).
		Where("channel_id = ? AND provider_message_id = ?", thread.ChannelID, providerMessageID).Scan(s.ctx))
	return row
}

// rows is how many rows thread has.
func (s *InboundSuite) rows(thread store.ChannelThread) int {
	count, err := s.store.DB().NewSelect().Model((*store.ChannelInbound)(nil)).Where("channel_id = ?", thread.ChannelID).Count(s.ctx)
	s.Require().NoError(err)
	return count
}

// writes is how many messages were sent to a channel, refused ones included; to any channel
// when channelID is empty.
func (s *InboundSuite) writes(channelID string) int {
	count := 0
	for _, request := range s.chat.Requests(chatKey) {
		if channelID == "" && strings.HasSuffix(request.Path, "/message") || channelID != "" && strings.HasSuffix(request.Path, "/"+channelID+"/message") {
			count++
		}
	}
	return count
}

// writeAhead stores a message in thread's channel under id, as an earlier attempt did.
func (s *InboundSuite) writeAhead(thread store.ChannelThread, id, text string) {
	author := AuthorUserID(thread.CustomerID, thread.ConnectorID, thread.ProviderUnitID, "U0000ALICE")
	_, err := s.chat.Client.UpdateUsers(s.ctx, &getstream.UpdateUsersRequest{Users: map[string]getstream.UserRequest{author: {ID: author}}})
	s.Require().NoError(err)
	_, err = s.chat.Client.Chat().GetOrCreateChannel(s.ctx, chatlog.ChannelType, thread.ChannelID, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{CreatedByID: &author},
	})
	s.Require().NoError(err)
	_, err = s.chat.Client.Chat().SendMessage(s.ctx, chatlog.ChannelType, thread.ChannelID, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{ID: &id, Text: &text, UserID: &author},
	})
	s.Require().NoError(err)
}

// card is the omni-channel id and the card message id of thread's episode.
func (s *InboundSuite) card(thread store.ChannelThread) (omni, card string) {
	var conversation string
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, `SELECT cm.conversation_id, ep.card_message_id FROM episodes ep
JOIN contact_map cm ON cm.id = ep.contact_id WHERE ep.thread_channel = ?`, chatlog.ChannelType+":"+thread.ChannelID).Scan(&conversation, &card))
	return strings.TrimPrefix(conversation, chatlog.ChannelType+":"), card
}

// destination stores a destination of the test's customer for fake_line's unhandled events.
func (s *InboundSuite) destination(forwarder *eventforward.Forwarder) string {
	destination := store.EventDestination{
		ID: uuid.NewString(), CustomerID: s.customer, ConnectorID: "fake_line", URL: "https://example.com/hooks", Forward: store.ForwardUnhandled,
	}
	secret, err := forwarder.NewSecret(destination.CustomerID, destination.ConnectorID, destination.ID)
	s.Require().NoError(err)
	destination.SecretSealed, destination.KEKVersion = secret.Sealed, secret.Version
	s.Require().NoError(s.store.CreateEventDestination(s.ctx, &destination))
	return destination.ID
}

func messageIDs(messages []map[string]any) []string {
	var ids []string
	for _, message := range messages {
		id, _ := message["id"].(string)
		ids = append(ids, id)
	}
	return ids
}

func isClosed(done <-chan struct{}) bool {
	select {
	case <-done:
		return true
	default:
		return false
	}
}

// syncBuffer is a log the worker's goroutines write to while a test reads it.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// The person a Linq chat is with is decided by the message admitted alone: a group message
// deletes the person the thread arrives with, as store.ChannelThread returns it for a queued row.
func (s *InboundSuite) TestAdmitDecidesTheChatPersonFromTheMessageAlone() {
	for name, test := range map[string]struct {
		isGroup bool
		want    string
	}{
		"a group message deletes the stored person": {isGroup: true, want: ""},
		"a one-to-one message names its author":     {isGroup: false, want: "U0000ALICE"},
	} {
		s.Run(name, func() {
			thread := s.thread("linq")
			s.Require().NoError(s.store.SetChannelThreadChatPerson(s.ctx, thread.ChannelID, "U0000BOB"))
			s.queueLinq(thread, "m1", test.isGroup)
			stored, err := s.store.ChannelThread(s.ctx, thread.ChannelID)
			s.Require().NoError(err)
			s.Require().Equal("U0000BOB", stored.ChatPerson, "the thread arrives with a stored person")

			admitted, err := s.newBridge(nil).admit(s.ctx, stored, s.row(thread, "m1"))
			s.Require().NoError(err)
			s.True(admitted)

			after, err := s.store.ChannelThread(s.ctx, thread.ChannelID)
			s.Require().NoError(err)
			s.Equal(test.want, after.ChatPerson)
		})
	}
}

// queueLinq queues a Linq message of U0000ALICE in thread, whose raw event says whether the chat is a group.
func (s *InboundSuite) queueLinq(thread store.ChannelThread, providerMessageID string, isGroup bool) {
	queued, err := s.store.AcceptChannelInbound(s.ctx, store.ChannelInbound{
		ChannelID: thread.ChannelID, ProviderMessageID: providerMessageID, CustomerID: thread.CustomerID,
		ConnectorID: thread.ConnectorID, ProviderUnitID: thread.ProviderUnitID, ThreadKey: thread.ThreadKey,
		ConfigID: "config-1", ConfigName: "Support", AuthorID: "U0000ALICE", Text: "hi",
		Raw: []byte(fmt.Sprintf(`{"data":{"chat":{"is_group":%t}}}`, isGroup)), NextAttemptAt: time.Now(),
	}, nil)
	s.Require().NoError(err)
	s.Require().True(queued)
}
