//go:build integration

package session

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// RecordedCreateSuite reads back, from Postgres, what Create recorded about a session.
type RecordedCreateSuite struct {
	suite.Suite
	ctx      context.Context
	store    *store.Store
	customer string
}

func TestRecordedCreateSuite(t *testing.T) {
	suite.Run(t, new(RecordedCreateSuite))
}

func (s *RecordedCreateSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN is not set")
	}
	s.ctx = context.Background()
	db, err := store.Open(testenv.Database(dsn, "recorded_create"))
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.store = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })
}

func (s *RecordedCreateSuite) SetupTest() {
	s.customer = "recorded-" + uuid.NewString()
}

func (s *RecordedCreateSuite) TestAGreetingSlowerThanTheRowIsStored() {
	// The greeting's Said comes out before it is spoken, so with a voice that takes a while
	// it reaches the fan-out well before Create has queued the session's row.
	m := s.manager(newQuietEdge(), slowTTS{&stubTTS{emitter: tts.NewEmitter(64)}}, nil)

	created, err := m.Create(s.ctx, s.spec(Spec{Greeting: "Hi, I'm listening."}))
	s.Require().NoError(err)

	s.Require().Eventually(func() bool {
		s.Require().NoError(m.records.Flush(s.ctx))
		got, err := s.store.SessionResponses(s.ctx, s.customer, created.ID(), 10, nil)
		s.Require().NoError(err)
		return len(got) == 1
	}, settleFor, 20*time.Millisecond, "the greeting's turn is not stored")
}

func (s *RecordedCreateSuite) TestTheStoredToolsKeepSearch() {
	m := s.manager(newQuietEdge(), &stubTTS{emitter: tts.NewEmitter(64)}, stubSearch{})

	created, err := m.Create(s.ctx, s.spec(Spec{SearchTarget: "en-low-latency"}))
	s.Require().NoError(err)
	s.Require().NoError(created.Close())
	s.Require().NoError(m.records.Flush(s.ctx))

	stored, err := s.store.SessionTools(s.ctx, created.ID())
	s.Require().NoError(err)
	names := make([]string, 0, len(stored))
	for _, tool := range stored {
		names = append(names, tool.Name)
	}
	s.Contains(names, "search", "search starts on joining, and its tool is offered from then on")
}

func (s *RecordedCreateSuite) TestASessionWhoseCallRefusedItLeavesNoRow() {
	m := s.manager(refusingEdge{newQuietEdge()}, &stubTTS{emitter: tts.NewEmitter(64)}, nil)
	from := time.Now().UTC().Add(-time.Hour)

	_, err := m.Create(s.ctx, s.spec(Spec{}))
	s.Require().Error(err)
	s.Require().NoError(m.records.Flush(s.ctx))

	// Base 291f2e34c: no row, no activity and a peak of none.
	rows, err := s.store.QuerySessions(s.ctx, s.customer, store.SessionFilter{Limit: 10})
	s.Require().NoError(err)
	s.Empty(rows, "a session that never joined is listed")
	to := time.Now().UTC().Add(time.Hour)
	buckets, err := s.store.CustomerActivity(s.ctx, s.customer, store.ActivityDaily, from, to)
	s.Require().NoError(err)
	for _, bucket := range buckets {
		s.Zero(bucket.Sessions, "a session that never joined is counted as activity")
	}
	peak, err := s.store.CustomerConcurrencyPeak(s.ctx, s.customer, from, to)
	s.Require().NoError(err)
	s.Zero(peak.Sessions, "a session that never joined is counted in the peak")
}

// A turn taken while a slow greeting is spoken runs before Create queues the session's row.
// The title it names the conversation is not in the row, which takes the spec's.
func (s *RecordedCreateSuite) TestATitleNamedDuringTheGreetingIsKept() {
	m := s.manager(newQuietEdge(), &stubTTS{emitter: tts.NewEmitter(64)}, nil)
	joining := s.joining(m)
	joining.naming = &naming{titles: s.titler(m, "Refund a late order")}

	joining.name(agent.Responded{TurnID: "turn-1", Text: "It ships today."})
	joining.naming.titles.running.Wait()

	s.Equal("Refund a late order", s.queued(m, joining, sessionRow(joining)).Title)
}

// The rows below change while Create builds the row and before it queues it.
func (s *RecordedCreateSuite) TestAConnectionChosenAsTheRowIsQueuedIsKept() {
	m := s.manager(newQuietEdge(), &stubTTS{emitter: tts.NewEmitter(64)}, nil)
	joining := s.joining(m)
	row := sessionRow(joining)

	joining.chose("crm", "connection-1")

	s.Equal([]store.SessionConnectorSelection{{Name: "crm", ConnectionID: "connection-1"}},
		s.queued(m, joining, row).ConnectorSelections)
}

func (s *RecordedCreateSuite) TestFramesSeenAsTheRowIsQueuedMakeItAVideoOne() {
	m := s.manager(newQuietEdge(), &stubTTS{emitter: tts.NewEmitter(64)}, nil)
	joining := s.joining(m)
	row := sessionRow(joining)
	frames := framesRunner{parts: []llm.ContentPart{{Image: &llm.ImagePart{URL: "data:image/jpeg;base64,AA=="}}}}

	_, err := (&videoRunner{next: frames, session: joining}).Run(s.ctx, llm.ToolCall{Name: agent.VideoFramesTool})
	s.Require().NoError(err)

	s.Equal(store.ModalityVideo, s.queued(m, joining, row).Modality)
}

// A title named before the row is built is in the row.
func (s *RecordedCreateSuite) TestTheRowCarriesTheTitleNamedBeforeItWasBuilt() {
	m := s.manager(newQuietEdge(), &stubTTS{emitter: tts.NewEmitter(64)}, nil)
	joining := s.joining(m)
	joining.title, joining.description = "Refund a late order", "Where an order went."

	row := sessionRow(joining)

	s.Equal("Refund a late order", row.Title)
	s.Equal("Where an order went.", row.Description)
}

// The tests below read the spec while a login's chose writes it, on the goroutines Create and
// the manager read it from. They show their failure only under -race.
func (s *RecordedCreateSuite) TestTheSpecReadsRaceNoChose() {
	m := s.manager(newQuietEdge(), &stubTTS{emitter: tts.NewEmitter(64)}, nil)
	joining := s.joining(m)
	joining.naming = &naming{titles: s.titler(m, "Refund a late order")}
	m.sessions[joining.id] = joining
	// Shutdown closes what is registered, and this session has no agent to close.
	s.T().Cleanup(func() { delete(m.sessions, joining.id) })
	owner := OwnerOf(joining.spec)
	reads := map[string]func(){
		"the session row":   func() { sessionRow(joining) },
		"the call row":      func() { row(joining) },
		"the episode card":  func() { (*callCards)(nil).started(joining, streamapp.Bound{}) },
		"the title asked":   func() { _, _ = joining.naming.titles.ask(s.ctx, joining.Spec(), joining.id, nil) },
		"a lookup":          func() { m.Get(joining.id, owner) },
		"a listing":         func() { m.List(owner) },
		"the newest":        func() { m.newest(func(Spec) bool { return true }, nil) },
		"a seat superseded": func() { m.supersede(Spec{Text: true}) },
	}
	for name, read := range reads {
		s.Run(name, func() {
			var wg sync.WaitGroup
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := range 50 {
					joining.chose("crm", fmt.Sprint("connection-", i))
				}
			}()
			for range 50 {
				read()
			}
			wg.Wait()
		})
	}
}

// joining is a session as Create holds it while the greeting is spoken: handed the recorder,
// its row not queued yet.
func (s *RecordedCreateSuite) joining(m *Manager) *Session {
	return &Session{
		id: uuid.NewString(), spec: s.spec(Spec{}), modality: store.ModalityVoice,
		records: m.records, created: time.Now(),
	}
}

// queued queues row as Create does once the greeting is over, and reads it back.
func (s *RecordedCreateSuite) queued(m *Manager, joining *Session, row store.AgentSession) store.AgentSession {
	m.records.Opened(row, nil)
	joining.startRecording()
	s.Require().NoError(m.records.Flush(s.ctx))
	rows, err := s.store.QuerySessions(s.ctx, s.customer, store.SessionFilter{Limit: 10})
	s.Require().NoError(err)
	s.Require().Len(rows, 1)
	return rows[0]
}

// titler names every conversation title, on m's recorder.
func (s *RecordedCreateSuite) titler(m *Manager, title string) *titler {
	logger := slog.New(slog.DiscardHandler)
	config := stubConfig()
	config.Aliases[titleTarget] = routing.Alias{Languages: []string{"en"}, RequireRealtime: true}
	naming := llmrouter.NewRegistry()
	naming.Register("stub", func(routing.Spec) (llmrouter.Provider, error) {
		return &stubLLM{reply: `{"title": "` + title + `", "description": "Where an order went."}`}, nil
	})
	router, err := llmrouter.New(llmrouter.Options{Config: config, Registry: naming, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	named := newTitler(router, m.records, logger)
	s.T().Cleanup(named.Close)
	return named
}

// spec is a session of the test's customer on the stub models.
func (s *RecordedCreateSuite) spec(spec Spec) Spec {
	spec.CustomerID, spec.CallID = s.customer, "call-1"
	spec.LLMTarget, spec.STTTarget, spec.TTSTarget = "en-low-latency", "en-low-latency", "en-low-latency"
	return spec
}

// manager is a session manager over the suite's Postgres, joining every call on edge and
// speaking with voice. A nil finder leaves the deployment without search.
func (s *RecordedCreateSuite) manager(edge agent.Edge, voice tts.TTS, finder search.Provider) *Manager {
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
	speech.Register("stub", func(routing.Spec) (tts.TTS, error) { return voice, nil })
	speaker, err := ttsrouter.New(ttsrouter.Options{Config: stubConfig(), Registry: speech, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(speaker.Close)

	options := ManagerOptions{
		LLM: reasoner, STT: transcriber, TTS: speaker, Store: s.store, Logger: logger,
		Edge: func(context.Context, Spec, streamapp.Bound, *slog.Logger) (agent.Edge, error) { return edge, nil },
	}
	if finder != nil {
		searching := searchrouter.NewRegistry()
		searching.Register("stub", func(routing.Spec) (search.Provider, error) { return finder, nil })
		options.Search, err = searchrouter.New(searchrouter.Options{Config: stubConfig(), Registry: searching, Logger: logger})
		s.Require().NoError(err)
	}
	m, err := NewManager(options)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.Require().NoError(m.Shutdown()) })
	return m
}

// refusingEdge is a call that will not have the agent.
type refusingEdge struct{ *quietEdge }

func (refusingEdge) Join(context.Context) error { return errors.New("the call refused the agent") }

// slowTTS takes 300 ms to start speaking, longer than Create takes to queue the row.
type slowTTS struct{ *stubTTS }

func (s slowTTS) Synthesize(request tts.Request) error {
	time.Sleep(300 * time.Millisecond)
	return s.stubTTS.Synthesize(request)
}

// stubSearch finds nothing.
type stubSearch struct{}

func (stubSearch) Search(context.Context, search.Query) (search.Result, error) {
	return search.Result{}, nil
}
func (stubSearch) Start(context.Context) error { return nil }
func (stubSearch) Close() error                { return nil }
func (stubSearch) Provider() string            { return "stub" }
func (stubSearch) Model() string               { return "stub-model" }
