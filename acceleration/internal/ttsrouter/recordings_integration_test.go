//go:build integration

package ttsrouter

import (
	"context"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/inworld"
)

// RecordingStatsSuite reads back the stats a recording leaves, which is what a model's
// health is made of.
type RecordingStatsSuite struct {
	suite.Suite
	ctx   context.Context
	store *store.Store
	live  *live.Client
}

func TestRecordingStatsSuite(t *testing.T) {
	suite.Run(t, new(RecordingStatsSuite))
}

func (s *RecordingStatsSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if dsn == "" || address == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN and ROUTER_REDIS_ADDR must be set")
	}
	s.ctx = context.Background()

	pgStore, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(pgStore.Migrate(s.ctx))
	s.store = pgStore

	liveClient, err := live.New(live.Options{Address: address})
	s.Require().NoError(err)
	s.live = liveClient
}

func (s *RecordingStatsSuite) TearDownSuite() {
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
	if s.live != nil {
		s.live.Close()
	}
}

// recordInworld records the text pinned to inworld/inworld-tts-2 for a customer of its own
// and returns the rows the customer left, once the stat writer has drained, and the error.
func (s *RecordingStatsSuite) recordInworld(text string, failing bool) ([]store.Request, error) {
	customer := "customer-" + time.Now().Format("150405.000000000")
	s.T().Setenv("INWORLD_API_KEY", "test-key")
	real := http.DefaultTransport
	http.DefaultTransport = &inworldStub{failing: failing}
	s.T().Cleanup(func() { http.DefaultTransport = real })
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	recordings, err := NewRecordings(Options{
		Config: config[routing.TTS], Recorders: DefaultRecorderRegistry(), Store: s.store, Live: s.live,
	})
	s.Require().NoError(err)

	_, _, recordErr := recordings.Record(s.ctx, Recording{
		CustomerID: customer,
		Options:    options.TTS{Target: "inworld/inworld-tts-2"},
		Text:       text,
	})
	// Closing the router drains the stat writer, so the rows are all in by now.
	recordings.Close()

	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", customer).Scan(s.ctx))
	return rows, recordErr
}

// recordOn records the text for a customer of its own with every vendor's own recorder,
// speaking to a vendorStub that answers statuses, and returns the rows the customer left
// once the stat writer has drained, and the error.
func (s *RecordingStatsSuite) recordOn(asked options.TTS, text string, statuses map[string]int) ([]store.Request, error) {
	customer := "customer-" + time.Now().Format("150405.000000000")
	for _, key := range []string{"ELEVENLABS_API_KEY", "CARTESIA_API_KEY", "INWORLD_API_KEY"} {
		s.T().Setenv(key, "test-key")
	}
	real := http.DefaultTransport
	http.DefaultTransport = &vendorStub{statuses: statuses, body: `{"detail":"down"}`}
	s.T().Cleanup(func() { http.DefaultTransport = real })
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	// No live health: the failures here would demote real models for every suite and
	// router sharing the Redis, and reorder the candidates of the next test.
	recordings, err := NewRecordings(Options{Config: config[routing.TTS], Recorders: DefaultRecorderRegistry(), Store: s.store})
	s.Require().NoError(err)

	_, _, recordErr := recordings.Record(s.ctx, Recording{CustomerID: customer, Options: asked, Text: text})
	recordings.Close()

	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", customer).Scan(s.ctx))
	return rows, recordErr
}

// outcomes is each row's model and error code, empty for a success.
func outcomes(rows []store.Request) []string {
	var said []string
	for _, row := range rows {
		said = append(said, row.Model+" "+row.ErrorCode)
	}
	return said
}

func (s *RecordingStatsSuite) TestACandidateThatFailsTheRecordingIsAFailedStatOnItsModelAndTheNextServes() {
	rows, err := s.recordOn(options.TTS{Target: "en-recorded", Providers: []string{"elevenlabs", "cartesia"}}, "hello",
		map[string]int{elevenLabsHost: http.StatusInternalServerError})

	s.Require().NoError(err)
	s.ElementsMatch([]string{"eleven_v3 record_failed", "sonic-3.6 "}, outcomes(rows))
}

func (s *RecordingStatsSuite) TestEveryCandidateThatFailsTheRecordingIsAFailedStatOnItsModel() {
	rows, err := s.recordOn(options.TTS{Target: "en-recorded", Providers: []string{"elevenlabs", "cartesia"}}, "hello",
		map[string]int{elevenLabsHost: http.StatusPaymentRequired, cartesiaHost: http.StatusServiceUnavailable})

	s.Require().Error(err)
	s.ElementsMatch([]string{"eleven_v3 record_failed", "sonic-3.6 record_failed"}, outcomes(rows))
}

func (s *RecordingStatsSuite) TestACandidateThatRefusesTheTextLeavesNoStatAndTheNextServes() {
	rows, err := s.recordOn(options.TTS{Target: "en-recorded", Providers: []string{"inworld", "cartesia"}},
		strings.Repeat("a", inworld.MaxRecordingText+1), nil)

	s.Require().NoError(err)
	s.Equal([]string{"sonic-3.6 "}, outcomes(rows), "Inworld's limit is not counted against its health")
}

func (s *RecordingStatsSuite) TestARecordingWithNothingToSayLeavesNoStat() {
	// Probed on 1c5fa58db: one row, eleven_v3 record_failed.
	rows, err := s.recordOn(options.TTS{Target: "en-recorded", Providers: []string{"elevenlabs", "cartesia"}}, "  ", nil)

	s.Require().ErrorIs(err, routing.ErrRequest)
	s.Empty(rows)
}

func (s *RecordingStatsSuite) TestATextOverInworldsLimitLeavesNoFailedStatOnTheModel() {
	rows, err := s.recordInworld(strings.Repeat("a", inworld.MaxRecordingText+1), false)

	s.Require().ErrorIs(err, routing.ErrRequest)
	s.Empty(rows, "the caller's mistake is not counted against a model every customer shares")
}

func (s *RecordingStatsSuite) TestAnInworldOutageIsStillAFailedStatOnTheModel() {
	rows, err := s.recordInworld("hello", true)

	s.Require().Error(err)
	s.NotErrorIs(err, routing.ErrRequest)
	s.Require().Len(rows, 1)
	s.Equal("record_failed", rows[0].ErrorCode)
	s.Equal("inworld-tts-2", rows[0].Model)
}
