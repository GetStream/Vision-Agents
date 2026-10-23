//go:build integration

package ttsrouter

import (
	"context"
	"errors"
	"net/http"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
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

// keyedRecorder is a recorder that keeps its key in asked when it is asked to record, and
// fails it when told to.
type keyedRecorder struct {
	key   string
	fails bool
	asked *[]string
}

func (r keyedRecorder) Record(context.Context, tts.Recording) (tts.Recorded, error) {
	*r.asked = append(*r.asked, r.key)
	if r.fails {
		return tts.Recorded{}, errors.New(r.key + ": record: 503")
	}
	return tts.Recorded{Audio: []byte("a")}, nil
}
func (keyedRecorder) Start(context.Context) error { return nil }
func (keyedRecorder) Close() error                { return nil }
func (r keyedRecorder) Provider() string          { return r.key }
func (keyedRecorder) Model() string               { return "m" }

// demoting is a policy that, the first time it admits, has live health mark its providers
// as failing every caller: health that changes while a recording is routed.
type demoting struct {
	live      *live.Client
	providers []string
	done      *bool
}

func (d demoting) Admit(ctx context.Context, owner routing.Owner) (routing.Admission, error) {
	customerID := owner.CustomerID
	if !*d.done {
		*d.done = true
		for _, provider := range d.providers {
			if err := d.live.RecordRequest(ctx, live.Usage{
				Modality: string(routing.TTS), CustomerID: customerID, Provider: provider, Model: "m", Fatal: true,
			}); err != nil {
				return routing.Admission{}, err
			}
		}
	}
	return routing.Admission{}, nil
}

// recordAcross records over a shortcut of three providers a, b and c of the test's own
// (their names unique, so their live health is theirs alone), demoting the ones in demote
// while the recording is routed. It returns the keys asked to record, how often each was
// built, who served and the error.
func (s *RecordingStatsSuite) recordAcross(
	fails, unbuilt map[string]bool, demote ...string,
) ([]string, map[string]int, routing.ProviderConfig, error) {
	id := uuid.NewString()[:8]
	var asked []string
	builds := map[string]int{}
	registry := NewRecorderRegistry()
	config := routing.ModalityConfig{Aliases: map[string]routing.Alias{"three": {Languages: []string{"en"}}}}
	var demoted []string
	for _, key := range []string{"a", "b", "c"} {
		name := key + id
		config.Providers = append(config.Providers, routing.ProviderConfig{Provider: name, Model: "m", Languages: []string{"en"}})
		if slices.Contains(demote, key) {
			demoted = append(demoted, name)
		}
		registry.Register(name, func(routing.Spec) (tts.Recorder, error) {
			builds[key]++
			if unbuilt[key] {
				return nil, errors.New(key + ": build failed")
			}
			return keyedRecorder{key: key, fails: fails[key], asked: &asked}, nil
		})
	}
	done := false
	recordings, err := NewRecordings(Options{Config: config, Recorders: registry, Live: s.live,
		Gate: demoting{live: s.live, providers: demoted, done: &done}})
	s.Require().NoError(err)
	defer recordings.Close()

	_, served, err := recordings.Record(s.ctx, Recording{CustomerID: "customer-" + id, Text: "hello",
		Options: options.TTS{Target: "three"}})
	return asked, builds, served, err
}

func (s *RecordingStatsSuite) TestARecordingWalksTheOrderItWasRoutedInEvenWhenHealthChangesMeanwhile() {
	// a falls to the back while the recording is routed; b and c fail it, and a, which
	// records, is still asked.
	asked, builds, served, err := s.recordAcross(map[string]bool{"b": true, "c": true}, nil, "a")

	s.Require().NoError(err)
	s.Equal([]string{"b", "c", "a"}, asked)
	s.Equal("a", served.Provider[:1])
	s.Equal(map[string]int{"a": 1, "b": 1, "c": 1}, builds)
}

func (s *RecordingStatsSuite) TestARecordingBuildsEachCandidateOnceEvenWhenHealthChangesMeanwhile() {
	// a and b fall to the back while the recording is routed, so c is first; it cannot be
	// built, and a and b fail it.
	asked, builds, _, err := s.recordAcross(map[string]bool{"a": true, "b": true}, map[string]bool{"c": true}, "a", "b")

	s.Require().Error(err)
	s.ElementsMatch([]string{"a", "b"}, asked)
	s.Equal(map[string]int{"a": 1, "b": 1, "c": 1}, builds)
}
