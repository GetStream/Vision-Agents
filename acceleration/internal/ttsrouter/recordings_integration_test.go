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
