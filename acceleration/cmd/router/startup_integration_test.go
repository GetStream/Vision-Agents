//go:build integration

package main

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime/pprof"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// RouterStartSuite starts the router as its serve command does, on a database of the suite's
// own, and stops it with the signal a deployment sends.
type RouterStartSuite struct {
	suite.Suite
	ctx      context.Context
	settings config.Config
	db       *store.Store
}

func TestRouterStartSuite(t *testing.T) {
	suite.Run(t, new(RouterStartSuite))
}

func (s *RouterStartSuite) SetupSuite() {
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", dsnEnvVar)
	}
	s.ctx = context.Background()
	s.settings = config.Defaults()
	s.settings.Postgres.DSN = testenv.Database(dsn, "start")
	s.settings.Addr = "127.0.0.1:0"
	// api_key, the default, wants a keyring to seal the keys it issues; a router of no
	// customer needs none.
	s.settings.Auth.Mode = string(auth.NoAuth)
	// The router migrates the database itself; opening it here first is what lets the test
	// store the rows it ages before the router starts.
	db, err := openStore(s.ctx, s.settings)
	s.Require().NoError(err)
	s.T().Cleanup(func() { db.Close() })
	s.db = db
}

// The prunes and the redaction run as the router starts, not a day, an hour or ten minutes later.
func (s *RouterStartSuite) TestAStartingRouterPrunesAndRedactsWhatIsPastItsRetention() {
	customer := "customer-" + uuid.NewString()
	next := time.Now().UTC().Add(24 * time.Hour).Truncate(time.Second)
	task := store.ScheduledTask{
		CustomerID: customer, ConfigID: "config-1", UserID: "user-1",
		CallerKind: store.UserKindAuthenticated, Title: "Stretch", Instructions: "Remind them to stretch.",
		StartLocal: next, Timezone: "UTC", NextRunAt: &next, State: store.ScheduledTaskActive, CreatedBy: "agent",
	}
	s.Require().NoError(s.db.CreateScheduledTask(s.ctx, &task, 25))
	s.T().Cleanup(func() {
		s.NoError(s.db.DeleteScheduledTask(context.Background(), customer, task.ID))
	})
	_, err := s.db.DB().ExecContext(s.ctx, `
INSERT INTO scheduled_task_runs (id, task_id, customer_id, scheduled_for, trigger, state, finished_at)
VALUES (?, ?, ?, now() - interval '91 days', 'schedule', 'delivered', now() - make_interval(secs => ?))`,
		uuid.NewString(), task.ID, customer, (store.ScheduledTasksKept + 24*time.Hour).Seconds())
	s.Require().NoError(err)
	_, err = s.db.DB().ExecContext(s.ctx, `
INSERT INTO data_changes (customer_id, table_name, op, key, at)
VALUES (?, 'agent_configs', 'insert', '{}', now() - make_interval(secs => ?))`,
		customer, (s.settings.DataMove.Retention + 24*time.Hour).Seconds())
	s.Require().NoError(err)
	session := &store.AgentSession{ID: uuid.NewString(), CustomerID: customer, AgentName: "docs", ConfigID: "config-1",
		UserID: "user-1", CallerKind: store.UserKindAuthenticated, State: store.SessionClosed, CreatedAt: time.Now().UTC()}
	s.Require().NoError(s.db.SaveSession(s.ctx, session))
	response := uuid.NewString()
	s.Require().NoError(s.db.StartResponse(s.ctx, &store.AgentResponse{
		ID: response, SessionID: session.ID, CustomerID: customer, Said: "where is my order?", CreatedAt: time.Now().UTC(),
	}))
	due := time.Now().UTC().Add(-time.Hour)
	s.Require().NoError(s.db.AppendResponseItems(s.ctx, []store.AgentResponseItem{{
		ResponseID: response, SessionID: session.ID, Kind: store.ItemToolResult, ToolName: "orders",
		Text: `{"email":"someone@example.com"}`, At: time.Now().UTC(), RedactAt: &due,
	}}))
	left := func(query string) int {
		var n int
		s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, query, customer).Scan(&n))
		return n
	}
	const (
		runs       = "SELECT count(*) FROM scheduled_task_runs WHERE customer_id = ?"
		changes    = "SELECT count(*) FROM data_changes WHERE customer_id = ?"
		unredacted = "SELECT count(*) FROM agent_response_items i JOIN agent_responses r ON r.id = i.response_id " +
			"WHERE r.customer_id = ? AND i.redact_at IS NOT NULL"
	)
	s.Require().Equal([]int{1, 1, 1}, []int{left(runs), left(changes), left(unredacted)})

	stop := s.start(s.settings)

	pruned := func() bool {
		return left(runs) == 0 && left(changes) == 0 && left(unredacted) == 0
	}
	// A router that stopped before it was told to fails here, and its error is in stop's.
	deadline := time.After(30 * time.Second)
	for !pruned() {
		select {
		case <-deadline:
			s.Require().Failf("what is past its retention is still there",
				"scheduled task runs %d, recorded changes %d, unredacted tool results %d", left(runs), left(changes), left(unredacted))
		case <-time.After(50 * time.Millisecond):
		}
	}
	s.Require().NoError(stop())
}

// The router starts the sweep of the Daytona secrets a crashed router left behind, and a sweep
// still running ends when the router is told to stop.
func (s *RouterStartSuite) TestAStartingRouterSweepsSecretsUntilItStops() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR not set")
	}
	// Its own prefix, so a run beside it on the same Redis has its own sweep lease.
	prefix := "boot" + strings.ToLower(rand.Text())
	listing, ended, release := make(chan struct{}, 1), make(chan struct{}, 1), make(chan struct{})
	signal := func(c chan struct{}) {
		select {
		case c <- struct{}{}:
		default:
		}
	}
	daytona := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The SDK's client also opens Daytona's event socket, which the sweep never uses.
		if r.URL.Path != "/api/secret/paginated" || r.URL.Query().Get("name") != "va_"+prefix+"_" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		// The list is held, so the sweep is still running when the router is told to stop.
		signal(listing)
		select {
		case <-r.Context().Done():
			signal(ended)
		case <-release:
		}
	}))
	s.T().Cleanup(daytona.Close)
	s.T().Cleanup(func() { close(release) })
	s.T().Setenv("DAYTONA_API_KEY", "key-1")
	s.T().Setenv("DAYTONA_API_URL", daytona.URL+"/api")
	settings := s.settings
	settings.Redis.Addr = address
	settings.CodeSandbox.SecretPrefix = prefix

	stop := s.start(settings)
	select {
	case <-listing:
	case <-time.After(30 * time.Second):
		s.Require().Fail("the router did not list the secrets it left behind")
	}
	s.Require().NoError(stop())

	// The cancelled request ends at once; ten seconds is slack for a loaded machine.
	select {
	case <-ended:
	case <-time.After(10 * time.Second):
		s.Fail("the sweep went on after the router stopped")
	}
}

// The router starts the sweep that retires the status urls of old placed calls.
func (s *RouterStartSuite) TestAStartingRouterRetiresTheStatusURLsOfOldCalls() {
	previous := phoneSweepEvery
	// Well inside the wait below; the production ten minutes would outlast it.
	phoneSweepEvery = 50 * time.Millisecond
	s.T().Cleanup(func() { phoneSweepEvery = previous })
	call := store.PhoneCall{
		Token: uuid.NewString(), CustomerID: "customer-" + uuid.NewString(), CallType: "default", CallID: uuid.NewString(),
		Vendor: "telnyx", LocalNumber: "+15125550100", RemoteNumber: "+15125550199", Reports: true,
		// Past a status url's lifetime, phone.MaxCallLength and an hour (statusTokenLifetime in
		// internal/phone/service.go).
		PlacedAt: time.Now().UTC().Add(-phone.MaxCallLength - 2*time.Hour),
	}
	s.Require().NoError(s.db.InsertPhoneCall(s.ctx, &call))
	s.T().Cleanup(func() {
		_, err := s.db.DB().ExecContext(context.Background(), "DELETE FROM phone_calls WHERE id = ?", call.ID)
		s.NoError(err)
	})
	retired := func() bool {
		var gone bool
		err := s.db.DB().QueryRowContext(s.ctx, "SELECT token IS NULL FROM phone_calls WHERE id = ?", call.ID).Scan(&gone)
		return err == nil && gone
	}

	stop := s.start(s.settings)

	s.Eventually(retired, 30*time.Second, 50*time.Millisecond, "the old call's status url is still live")
	s.Require().NoError(stop())
	// startPhoneSweeper is the sweep's frame (cmd/router/phonesweep.go); its goroutine ends with
	// the context the router hands it.
	s.Eventually(func() bool { return !strings.Contains(goroutines(), "startPhoneSweeper") },
		10*time.Second, 50*time.Millisecond, "the phone sweep went on after the router stopped")
}

// The router starts the sweep that closes idle episodes where an agent config has the cards on.
func (s *RouterStartSuite) TestAStartingRouterClosesAnIdleEpisode() {
	customer := "customer-" + uuid.NewString()
	carded := &store.AgentConfig{CustomerID: customer, Name: "carded", EpisodeCards: true}
	s.Require().NoError(s.db.CreateAgentConfig(s.ctx, carded))
	s.T().Cleanup(func() { s.NoError(s.db.DeleteAgentConfig(context.Background(), customer, carded.ID)) })
	contact := store.ContactMapEntry{
		CustomerID: customer, AgentConfigID: carded.ID, Kind: store.ContactSlack,
		Address: "T1:U1", ConversationID: "agent:omni-" + uuid.NewString(),
	}
	_, err := s.db.MapContact(s.ctx, &contact)
	s.Require().NoError(err)
	episode := store.Episode{
		CustomerID: customer, ContactID: contact.ID, Source: store.EpisodeSlack, ThreadChannel: "agent:thread-" + uuid.NewString(),
		// Past episodes.idle_after, an hour by default (config.Defaults).
		StartedAt: time.Now().Add(-2 * time.Hour),
	}
	_, err = s.db.OpenEpisode(s.ctx, &episode)
	s.Require().NoError(err)
	// The episode's episode_activity row goes with it (ON DELETE CASCADE).
	s.T().Cleanup(func() {
		_, err := s.db.DB().ExecContext(context.Background(), "DELETE FROM episodes WHERE id = ?", episode.ID)
		s.NoError(err)
		_, err = s.db.DB().ExecContext(context.Background(), "DELETE FROM contact_map WHERE id = ?", contact.ID)
		s.NoError(err)
	})
	closed := func() bool {
		var status string
		err := s.db.DB().QueryRowContext(s.ctx, "SELECT status FROM episodes WHERE id = ?", episode.ID).Scan(&status)
		return err == nil && status != "in_progress"
	}

	stop := s.start(s.settings)

	s.Eventually(closed, 30*time.Second, 50*time.Millisecond, "the idle episode is still open")
	s.Require().NoError(stop())
}

// start runs the router as its serve command does, and waits for its public listener: a stop
// signal before that fails the start with the context's error. stop sends it the signal a
// deployment sends and gives what run returned. A failed wait ends the test with the router
// still up; run has a handler for SIGTERM only while it runs, so the cleanup stops it here and
// not by the signal killing the binary.
func (s *RouterStartSuite) start(settings config.Config) (stop func() error) {
	stopped := make(chan struct{})
	logs := &listenLog{listening: make(chan struct{})}
	var runErr error
	go func() {
		runErr = run(settings, slog.New(logs))
		close(stopped)
	}()
	stop = sync.OnceValue(func() error {
		select {
		case <-stopped:
			return fmt.Errorf("the router stopped before it was told to: %v", runErr)
		default:
		}
		if err := syscall.Kill(os.Getpid(), syscall.SIGTERM); err != nil {
			return err
		}
		select {
		case <-stopped:
			return runErr
		case <-time.After(30 * time.Second):
			return errors.New("the router did not stop on SIGTERM")
		}
	})
	s.T().Cleanup(func() { s.NoError(stop()) })
	select {
	case <-logs.listening:
	case <-stopped:
		s.Require().Failf("the router stopped as it started", "error: %v", runErr)
	// The 30 s the start test has always waited; a start here takes under a second.
	case <-time.After(30 * time.Second):
		s.Require().Fail("the router did not start listening")
	}
	return stop
}

// goroutines is every goroutine's stack, as the pprof endpoint prints it.
func goroutines() string {
	var out strings.Builder
	_ = pprof.Lookup("goroutine").WriteTo(&out, 1)
	return out.String()
}

// listenLog drops the router's logs and tells when it logs that it is listening.
type listenLog struct {
	listening chan struct{}
	once      sync.Once
}

func (l *listenLog) Enabled(context.Context, slog.Level) bool { return true }

func (l *listenLog) Handle(_ context.Context, record slog.Record) error {
	if record.Message == "listening" {
		l.once.Do(func() { close(l.listening) })
	}
	return nil
}

func (l *listenLog) WithAttrs([]slog.Attr) slog.Handler { return l }
func (l *listenLog) WithGroup(string) slog.Handler      { return l }
