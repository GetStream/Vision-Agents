//go:build integration

package session

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// runtimeSlice is how long a slice of runtime runs in these tests, so a session lives
// through several in a fraction of a second.
const runtimeSlice = 50 * time.Millisecond

// refusingEdge is a call the agent cannot join.
type refusingEdge struct{ *quietEdge }

func (refusingEdge) Join(context.Context) error { return errors.New("the call refused the agent") }

// RuntimeSuite opens sessions on a manager over Postgres and reads back what their time cost
// and what they wrote down.
type RuntimeSuite struct {
	suite.Suite
	ctx    context.Context
	store  *store.Store
	prices *pricing.List

	manager  *Manager
	customer string
	// refuses makes every call the manager opens refuse the agent.
	refuses bool
}

func TestRuntimeSuite(t *testing.T) {
	suite.Run(t, new(RuntimeSuite))
}

func (s *RuntimeSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "session_runtime"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
	perMinute := map[pricing.Meter]pricing.Rate{pricing.Seconds: {USD: 0.06, Per: 60}}
	s.prices, err = pricing.NewList("runtime", []pricing.Entry{
		{Modality: string(routing.Runtime), Provider: runtimeProvider, Model: store.ModalityText, Free: true},
		{Modality: string(routing.Runtime), Provider: runtimeProvider, Model: store.ModalityVoice, Rates: perMinute},
		{Modality: string(routing.Runtime), Provider: runtimeProvider, Model: store.ModalityVideo, Rates: perMinute},
	}, 0, slog.New(slog.NewTextHandler(io.Discard, nil)))
	s.Require().NoError(err)
}

func (s *RuntimeSuite) SetupTest() {
	s.customer = "runtime-" + uuid.NewString()
	s.refuses = false
	s.manager = nil
}

// manages builds a manager over stub providers that keeps its sessions in Postgres.
func (s *RuntimeSuite) manages() {
	logger := slog.New(slog.DiscardHandler)

	transcription := sttrouter.NewRegistry()
	transcription.Register("stub", func(routing.Spec) (stt.STT, error) { return &stubSTT{emitter: stt.NewEmitter(64)}, nil })
	transcriber, err := sttrouter.New(sttrouter.Options{Config: stubConfig(), Registry: transcription, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(transcriber.Close)

	reasoning := llmrouter.NewRegistry()
	reasoning.Register("stub", func(routing.Spec) (llmrouter.Provider, error) { return &stubLLM{reply: "Hello."}, nil })
	reasoner, err := llmrouter.New(llmrouter.Options{Config: stubConfig(), Registry: reasoning, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(reasoner.Close)

	speech := ttsrouter.NewRegistry()
	speech.Register("stub", func(routing.Spec) (tts.TTS, error) { return &stubTTS{emitter: tts.NewEmitter(64)}, nil })
	speaker, err := ttsrouter.New(ttsrouter.Options{Config: stubConfig(), Registry: speech, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(speaker.Close)

	manager, err := NewManager(ManagerOptions{
		LLM:    reasoner,
		STT:    transcriber,
		TTS:    speaker,
		Store:  s.store,
		Pricer: s.prices,
		Logger: logger,
		Edge: func(context.Context, Spec, streamapp.Bound, *slog.Logger) (agent.Edge, error) {
			edge := newQuietEdge()
			if s.refuses {
				return refusingEdge{edge}, nil
			}
			return attendedEdge{edge}, nil
		},
	})
	s.Require().NoError(err)
	manager.sliceEvery = runtimeSlice
	s.manager = manager
	s.T().Cleanup(func() { manager.Shutdown() })
}

// opens creates a session of the test's customer, defaulting what the test did not name.
func (s *RuntimeSuite) opens(spec Spec) (*Session, error) {
	spec.CustomerID = s.customer
	spec.Attribution = routing.Attribution{APIKey: "key-runtime"}
	spec.LLMTarget = "en-low-latency"
	spec.NoReview = true
	if !spec.Text {
		spec.STTTarget, spec.TTSTarget = "en-low-latency", "en-low-latency"
		if spec.CallID == "" {
			spec.CallID = "call-" + uuid.NewString()
		}
	}
	return s.manager.Create(s.ctx, spec)
}

// runtime shuts the manager down, which writes what its sessions' time is still queued as,
// and returns the session's runtime rows, oldest first.
func (s *RuntimeSuite) runtime(sessionID string) []store.Request {
	s.Require().NoError(s.manager.Shutdown())
	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).
		Where("customer_id = ?", s.customer).
		Where("session_id = ?", sessionID).
		Where("modality = ?", string(routing.Runtime)).
		Order("started_at").Scan(s.ctx))
	return rows
}

// modes is the models the rows were billed as, each run of the same one once.
func modes(rows []store.Request) []string {
	var billed []string
	for _, row := range rows {
		if len(billed) == 0 || billed[len(billed)-1] != row.Model {
			billed = append(billed, row.Model)
		}
	}
	return billed
}

func totalSeconds(rows []store.Request) int64 {
	var total int64
	for _, row := range rows {
		total += row.Meters[pricing.Seconds]
	}
	return total
}

func (s *RuntimeSuite) TestASessionRecordsRuntimeUntilItCloses() {
	s.manages()
	created, err := s.opens(Spec{Text: true})
	s.Require().NoError(err)

	time.Sleep(runtimeSlice*2 + runtimeSlice/2)
	s.Require().NoError(created.Close())

	rows := s.runtime(created.ID())
	s.Require().GreaterOrEqual(len(rows), 3, "a slice is written as each one ends, and the last when the session closes")
	s.Equal([]string{store.ModalityText}, modes(rows))
	s.EqualValues(1, totalSeconds(rows), "the slices add up to the session's life, rounded up to a second")
	for _, row := range rows {
		s.Equal(runtimeProvider, row.Provider)
		s.Equal(created.ID(), row.OperationID)
		s.Equal("key-runtime", row.APIKey, "billed to whoever opened the session")
		s.Zero(row.CostMicros, "time in writing is priced free here")
		s.False(row.Billable, "and so is recorded without being billed")
	}
}

func (s *RuntimeSuite) TestStartingVoiceSwitchesTheRuntimeMode() {
	s.manages()
	created, err := s.opens(Spec{Text: true})
	s.Require().NoError(err)

	time.Sleep(runtimeSlice)
	s.Require().NoError(created.StartVoice(s.ctx, Call{}))
	time.Sleep(runtimeSlice)
	created.SawVideo()
	time.Sleep(runtimeSlice)
	s.Require().NoError(created.StopVoice(s.ctx))
	created.SawVideo()
	time.Sleep(runtimeSlice)
	s.Require().NoError(created.Close())

	rows := s.runtime(created.ID())
	s.Equal([]string{store.ModalityText, store.ModalityVoice, store.ModalityVideo, store.ModalityText}, modes(rows),
		"billed as what held the session: writing, the call, the call with video, then writing again")
	s.EqualValues(1, totalSeconds(rows))
	for _, row := range rows {
		s.Equal(row.Model != store.ModalityText, row.Billable, "only the call's time is billed at these prices")
	}
}

func (s *RuntimeSuite) TestASessionThatFailsToOpenIsNotBilled() {
	s.manages()
	s.refuses = true
	id := "runtime-" + uuid.NewString()

	_, err := s.opens(Spec{ID: id})
	s.Require().Error(err)

	s.Empty(s.runtime(id), "the time spent failing to join is not the customer's")
}

func (s *RuntimeSuite) TestTheGreetingIsTheSessionsFirstRecordedTurn() {
	s.manages()
	created, err := s.opens(Spec{Greeting: "Hi, I'm listening."})
	s.Require().NoError(err)

	var turns []store.AgentResponse
	s.Require().Eventually(func() bool {
		s.Require().NoError(created.FlushRecords(s.ctx))
		turns, err = s.store.SessionResponses(s.ctx, s.customer, created.ID(), 10, nil)
		s.Require().NoError(err)
		return len(turns) > 0 && turns[0].Status == store.ResponseCompleted
	}, settleFor, 10*time.Millisecond, "the greeting was never written down")
	items, err := s.store.SessionItems(s.ctx, s.customer, created.ID(), turns[0].ID, 10, nil)
	s.Require().NoError(err)
	said := slices.IndexFunc(items, func(item store.AgentResponseItem) bool {
		return item.Kind == store.ItemAnswer && item.Text == "Hi, I'm listening."
	})
	s.GreaterOrEqual(said, 0, "the first turn is the greeting, said as written")
}
