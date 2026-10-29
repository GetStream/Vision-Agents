//go:build integration

package daytona

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/concurrency"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// org is Daytona's secret API as a sweep uses it: List, which matches the name anywhere in a
// secret's and answers two secrets a page, and Delete.
type org struct {
	server *httptest.Server

	mu      sync.Mutex
	secrets []orgSecret
	calls   []string
	// refuseLists answers that many Lists with a 500, and refusePaged every List of a page after
	// the first.
	refuseLists int
	refusePaged bool
	// refuseDelete answers the Delete of a secret whose name holds it with a 500.
	refuseDelete string
	// hold, when set, keeps every List waiting until it is closed, and each List says on listing
	// that it started.
	hold, listing chan struct{}
}

// orgSecret is a secret of the organization its key reaches: a key lists and deletes its own
// organization's secrets alone.
type orgSecret struct {
	id, name, key string
	created       time.Time
}

// deploymentKey is the key the suite's sweepers reach the deployment's organization with.
const deploymentKey = "key-1"

func newOrg() *org {
	o := &org{}
	o.server = httptest.NewServer(http.HandlerFunc(o.serve))
	return o
}

func (o *org) serve(w http.ResponseWriter, r *http.Request) {
	// The SDK's client also opens Daytona's event socket, which the sweep never uses.
	if r.Method != http.MethodDelete && r.URL.Path != "/api/secret/paginated" {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	o.mu.Lock()
	o.calls = append(o.calls, r.Method+" "+r.URL.Path)
	hold, listing := o.hold, o.listing
	o.mu.Unlock()
	key := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	if r.Method == http.MethodDelete {
		o.mu.Lock()
		defer o.mu.Unlock()
		id := strings.TrimPrefix(r.URL.Path, "/api/secret/")
		if o.refuseDelete != "" && slices.ContainsFunc(o.secrets, func(held orgSecret) bool {
			return held.id == id && strings.Contains(held.name, o.refuseDelete)
		}) {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		o.secrets = slices.DeleteFunc(o.secrets, func(held orgSecret) bool { return held.id == id && held.key == key })
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if hold != nil {
		select {
		case listing <- struct{}{}:
		default:
		}
		<-hold
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	cursor := r.URL.Query().Get("cursor")
	if o.refuseLists > 0 || o.refusePaged && cursor != "" {
		o.refuseLists--
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"message":"daytona is down"}`))
		return
	}
	var matched []map[string]any
	for _, held := range o.secrets {
		if held.key == key && strings.Contains(held.name, r.URL.Query().Get("name")) {
			matched = append(matched, map[string]any{"id": held.id, "name": held.name, "placeholder": "dtn_secret_" + held.id,
				"hosts": []string{"api.github.com"}, "createdAt": held.created, "updatedAt": time.Now()})
		}
	}
	from, _ := strconv.Atoi(cursor)
	to := min(from+2, len(matched))
	var next any
	if to < len(matched) {
		next = strconv.Itoa(to)
	}
	answer, _ := json.Marshal(map[string]any{"items": append([]map[string]any{}, matched[from:to]...), "total": len(matched),
		"nextCursor": next})
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(answer)
}

// add puts a secret named name in the deployment's organization, made age ago.
func (o *org) add(name string, age time.Duration) { o.addFor(deploymentKey, name, age) }

// addFor puts a secret named name in the organization key reaches, made age ago.
func (o *org) addFor(key, name string, age time.Duration) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.secrets = append(o.secrets, orgSecret{id: "secret-" + uuid.NewString(), name: name, key: key, created: time.Now().Add(-age)})
}

func (o *org) names() []string {
	o.mu.Lock()
	defer o.mu.Unlock()
	var names []string
	for _, held := range o.secrets {
		names = append(names, held.name)
	}
	return names
}

func (o *org) called(prefix string) int {
	o.mu.Lock()
	defer o.mu.Unlock()
	count := 0
	for _, call := range o.calls {
		if strings.HasPrefix(call, prefix) {
			count++
		}
	}
	return count
}

// SweepSuite is what a sweep deletes of the secrets at Daytona, with the sessions it sees as
// live held in a node directory in Redis.
type SweepSuite struct {
	suite.Suite
	ctx       context.Context
	redis     rueidis.Client
	org       *org
	directory *node.Directory
	// prefix is the test's secret prefix, and keys its Redis keys, each named for this PR.
	prefix string
	keys   []string
}

func TestSweepSuite(t *testing.T) {
	suite.Run(t, new(SweepSuite))
}

func (s *SweepSuite) SetupSuite() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR not set")
	}
	client, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.redis, s.ctx = client, context.Background()
}

func (s *SweepSuite) TearDownSuite() {
	if s.redis != nil {
		s.redis.Close()
	}
}

func (s *SweepSuite) SetupTest() {
	s.org = newOrg()
	s.prefix = "w15sweep" + strings.ReplaceAll(uuid.NewString(), "-", "")[:8]
	directory := "w15sweep:" + s.prefix
	var err error
	s.directory, err = node.NewDirectory(node.DirectoryOptions{Redis: s.redis, Address: "10.0.0.1:1", Prefix: directory,
		Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.keys = []string{"sandbox:secret-sweep:" + s.prefix}
}

func (s *SweepSuite) TearDownTest() {
	s.directory.Close()
	s.org.server.Close()
	s.Require().NoError(s.redis.Do(s.ctx, s.redis.B().Del().Key(s.keys...).Build()).Error())
}

// live has the directory say session runs on this router.
func (s *SweepSuite) live(session string) {
	s.directory.Hold(s.ctx, session, "")
	s.keys = append(s.keys, "w15sweep:"+s.prefix+":session:"+session)
}

// unreadable has the directory fail to say whether session is live: its claim's key holds a
// list, and Redis answers a GET of a list with WRONGTYPE.
func (s *SweepSuite) unreadable(session string) {
	key := "w15sweep:" + s.prefix + ":session:" + session
	s.Require().NoError(s.redis.Do(s.ctx, s.redis.B().Rpush().Key(key).Element("w17").Build()).Error())
	s.keys = append(s.keys, key)
}

// sweeper sweeps the test's prefix, with liveness read from directory, and asks it again at once.
func (s *SweepSuite) sweeper(directory *node.Directory) *Sweeper {
	return s.sweeperEvery(directory, 50*time.Millisecond)
}

func (s *SweepSuite) sweeperEvery(directory *node.Directory, every time.Duration) *Sweeper {
	sweeper, err := NewSweeper(SweepOptions{APIKey: deploymentKey, APIURL: s.org.server.URL + "/api", Prefix: s.prefix,
		Sessions: directory, Slots: concurrency.New(s.redis, nil), Every: every, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	sweeper.wait = func(context.Context) error { return nil }
	return sweeper
}

// named is this test's secret for session and binding.
func (s *SweepSuite) named(session, binding string) string {
	return "va_" + s.prefix + "_" + session + "_" + binding + "_0a1b2c3d"
}

func (s *SweepSuite) TestAnEndedSessionsSecretIsDeletedAndALiveOnesKept() {
	s.live("session-live")
	s.org.add(s.named("session-live", "github"), 2*time.Hour)
	s.org.add(s.named("session-ended", "github"), 2*time.Hour)

	s.Require().NoError(s.sweeper(s.directory).Sweep(s.ctx))

	s.Equal([]string{s.named("session-live", "github")}, s.org.names())
}

func (s *SweepSuite) TestACustomersOwnOrganizationIsSweptOnTheirKey() {
	s.live("session-live")
	s.org.addFor("customer-key", s.named("session-ended", "github"), 2*time.Hour)
	s.org.addFor("customer-key", s.named("session-live", "github"), 2*time.Hour)
	s.Require().NoError(s.sweeper(s.directory).Sweep(s.ctx))
	s.Len(s.org.names(), 2, "the deployment's key reaches nothing of the customer's")

	sweeper := s.sweeper(s.directory)
	sweeper.options.Accounts = func(context.Context) ([]string, error) { return []string{"customer-key"}, nil }
	s.Require().NoError(sweeper.Sweep(s.ctx))

	s.Equal([]string{s.named("session-live", "github")}, s.org.names(), "the ended session's, on the customer's key")
}

// A session id and a binding may both hold _, so the session is not just what comes before the
// first _ of the name.
func (s *SweepSuite) TestALiveSessionWhoseIdHoldsAnUnderscoreKeepsItsSecret() {
	s.live("session_live")
	s.org.add(s.named("session_live", "my_github"), 2*time.Hour)

	s.Require().NoError(s.sweeper(s.directory).Sweep(s.ctx))

	s.Equal([]string{s.named("session_live", "my_github")}, s.org.names())
}

func (s *SweepSuite) TestASecretYoungerThanAnHourIsKept() {
	s.org.add(s.named("session-young", "github"), 59*time.Minute)
	s.org.add(s.named("session-old", "github"), 61*time.Minute)

	s.Require().NoError(s.sweeper(s.directory).Sweep(s.ctx))

	s.Equal([]string{s.named("session-young", "github")}, s.org.names())
}

// Daytona's name filter matches anywhere in a name, so another environment's secret whose
// session id holds this prefix is listed too, also where that environment's prefix extends ours.
func (s *SweepSuite) TestAnotherEnvironmentsSecretIsKept() {
	other := []string{"va_dev_session-1_github_0a1b2c3d", "va_dev_" + s.named("session-2", "github"),
		"va_" + s.prefix + "2_" + s.named("session-4", "github")}
	for _, name := range other {
		s.org.add(name, 2*time.Hour)
	}
	s.org.add(s.named("session-3", "github"), 2*time.Hour)

	s.Require().NoError(s.sweeper(s.directory).Sweep(s.ctx))

	s.Equal(other, s.org.names())
}

// A live session drops out of the directory while its claim is gone, until its node writes the
// claim back.
func (s *SweepSuite) TestASessionWhoseClaimCameBackKeepsItsSecret() {
	s.org.add(s.named("session-lapsed", "github"), 2*time.Hour)
	sweeper := s.sweeper(s.directory)
	sweeper.wait = func(context.Context) error {
		s.live("session-lapsed")
		return nil
	}

	s.Require().NoError(sweeper.Sweep(s.ctx))

	s.Equal([]string{s.named("session-lapsed", "github")}, s.org.names())
}

// A sweep asks again only after recheckAfter, so one that must end sooner deletes nothing.
func (s *SweepSuite) TestASweepDeletesNothingBeforeItAsksAgain() {
	s.org.add(s.named("session-ended", "github"), 2*time.Hour)
	sweeper := s.sweeper(s.directory)
	sweeper.wait = waitToRecheck
	ctx, cancel := context.WithTimeout(s.ctx, time.Second)
	defer cancel()

	s.Error(sweeper.Sweep(ctx))

	s.Len(s.org.names(), 1)
}

// A sweep cancelled while it waits to ask again ends then, so a router that stops is not held
// up for recheckAfter.
func (s *SweepSuite) TestASweepCancelledWhileItWaitsEndsAtOnce() {
	s.org.add(s.named("session-ended", "github"), 2*time.Hour)
	sweeper := s.sweeper(s.directory)
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()
	sweeper.wait = func(ctx context.Context) error {
		time.AfterFunc(100*time.Millisecond, cancel)
		return waitToRecheck(ctx)
	}
	start := time.Now()

	s.ErrorIs(sweeper.Sweep(ctx), context.Canceled)

	s.Less(time.Since(start), recheckAfter/2)
	s.Len(s.org.names(), 1)
}

// A sweep that finds no secret to delete has nothing to ask again, so it does not wait.
func (s *SweepSuite) TestASweepWithNothingToDeleteDoesNotWait() {
	s.live("session-live")
	s.org.add(s.named("session-live", "github"), 2*time.Hour)
	s.org.add(s.named("session-young", "github"), time.Minute)
	sweeper := s.sweeper(s.directory)
	sweeper.wait = waitToRecheck
	start := time.Now()

	s.Require().NoError(sweeper.Sweep(s.ctx))

	s.Less(time.Since(start), recheckAfter/2)
	s.Len(s.org.names(), 2)
}

// A live session's claim is gone for at most its lifetime before its node writes it back, so a
// sweep asks again only after that.
func (s *SweepSuite) TestASweepAsksAgainOnlyAfterAClaimWouldBeWrittenBack() {
	s.live("session-live")

	lasts, err := s.redis.Do(s.ctx, s.redis.B().Pttl().Key("w15sweep:"+s.prefix+":session:session-live").Build()).AsInt64()

	s.Require().NoError(err)
	s.Require().Positive(lasts, "the claim is written with a lifetime")
	s.Greater(recheckAfter, time.Duration(lasts)*time.Millisecond)
}

// A sweep ends within the lease that keeps a second router from sweeping beside it, also where
// its caller would let it run longer.
func (s *SweepSuite) TestASweepEndsWithinItsLease() {
	s.org.add(s.named("session-ended", "github"), 2*time.Hour)
	sweeper := s.sweeper(s.directory)
	var deadline time.Time
	sweeper.wait = func(ctx context.Context) error {
		deadline, _ = ctx.Deadline()
		return nil
	}
	ctx, cancel := context.WithTimeout(s.ctx, 2*sweepLease)
	defer cancel()
	start := time.Now()

	s.Require().NoError(sweeper.Sweep(ctx))

	s.WithinRange(deadline, start.Add(sweepLease), time.Now().Add(sweepLease))
}

// A session whose liveness cannot be read keeps its secret, and the secrets listed after it are
// still swept.
func (s *SweepSuite) TestASecretListedAfterAnUnreadableOneIsStillDeleted() {
	s.unreadable("session-unreadable")
	s.org.add(s.named("session-unreadable", "github"), 2*time.Hour)
	s.org.add(s.named("session-ended", "github"), 2*time.Hour)

	s.Error(s.sweeper(s.directory).Sweep(s.ctx))

	s.Equal([]string{s.named("session-unreadable", "github")}, s.org.names())
}

// A session whose liveness cannot be read when the sweep asks again keeps its secret, and the
// sweep says it could not tell.
func (s *SweepSuite) TestASessionUnreadableWhenAskedAgainFailsTheSweep() {
	s.org.add(s.named("session-1", "github"), 2*time.Hour)
	sweeper := s.sweeper(s.directory)
	sweeper.wait = func(context.Context) error {
		s.unreadable("session-1")
		return nil
	}

	s.ErrorContains(sweeper.Sweep(s.ctx), "is the session of secret "+s.named("session-1", "github")+" live")

	s.Equal([]string{s.named("session-1", "github")}, s.org.names())
}

func (s *SweepSuite) TestANameThisRouterDidNotMakeIsKept() {
	stray := []string{"va_" + s.prefix + "_stray", "va_" + s.prefix + "_stray_0a1b2c3d"}
	for _, name := range stray {
		s.org.add(name, 2*time.Hour)
	}

	s.Require().NoError(s.sweeper(s.directory).Sweep(s.ctx))

	s.Equal(stray, s.org.names())
}

func (s *SweepSuite) TestEveryPageIsSwept() {
	for i := range 5 {
		s.org.add(s.named("session-"+strconv.Itoa(i), "github"), 2*time.Hour)
	}

	s.Require().NoError(s.sweeper(s.directory).Sweep(s.ctx))

	s.Empty(s.org.names())
	s.Equal(3, s.org.called("GET /api/secret/paginated"))
}

func (s *SweepSuite) TestAListErrorDeletesNothing() {
	for i := range 3 {
		s.org.add(s.named("session-"+strconv.Itoa(i), "github"), 2*time.Hour)
	}
	s.org.refusePaged = true

	s.Error(s.sweeper(s.directory).Sweep(s.ctx))

	s.Len(s.org.names(), 3)
	s.Zero(s.org.called("DELETE"))
}

func (s *SweepSuite) TestASecretDaytonaWouldNotDeleteLeavesTheOthersDeleted() {
	s.org.add(s.named("session-1", "github"), 2*time.Hour)
	s.org.add(s.named("session-2", "github"), 2*time.Hour)
	s.org.refuseDelete = "session-1"

	s.Error(s.sweeper(s.directory).Sweep(s.ctx))

	s.Equal([]string{s.named("session-1", "github")}, s.org.names())
}

func (s *SweepSuite) TestTheNextTickSweepsAfterAListError() {
	s.org.add(s.named("session-ended", "github"), 2*time.Hour)
	s.org.refuseLists = 1
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()

	go s.sweeper(s.directory).Run(ctx)

	s.Eventually(func() bool { return len(s.org.names()) == 0 }, 5*time.Second, 10*time.Millisecond)
	s.Equal(2, s.org.called("GET /api/secret/paginated"))
}

// Run sweeps when the router starts, and then once an hour by default.
func (s *SweepSuite) TestRunSweepsAtOnceAndThenHourly() {
	s.org.add(s.named("session-ended", "github"), 2*time.Hour)
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()

	go s.sweeperEvery(s.directory, 0).Run(ctx)

	s.Eventually(func() bool { return len(s.org.names()) == 0 }, 5*time.Second, 10*time.Millisecond)
	time.Sleep(300 * time.Millisecond)
	s.Equal(1, s.org.called("GET /api/secret/paginated"))
}

func (s *SweepSuite) TestASessionWhoseLivenessCannotBeReadKeepsItsSecret() {
	closed, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{os.Getenv("ROUTER_REDIS_ADDR")}, DisableCache: true})
	s.Require().NoError(err)
	closed.Close()
	unreadable, err := node.NewDirectory(node.DirectoryOptions{Redis: closed, Address: "10.0.0.1:1", Prefix: "w15sweep:" + s.prefix,
		Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	defer unreadable.Close()
	s.org.add(s.named("session-1", "github"), 2*time.Hour)

	s.Error(s.sweeper(unreadable).Sweep(s.ctx))

	s.Equal([]string{s.named("session-1", "github")}, s.org.names())
}

func (s *SweepSuite) TestOneRouterSweepsAtATime() {
	s.org.add(s.named("session-ended", "github"), 2*time.Hour)
	s.org.hold, s.org.listing = make(chan struct{}), make(chan struct{}, 1)
	first, second := s.sweeper(s.directory), s.sweeper(s.directory)
	swept := make(chan error, 1)
	go func() { swept <- first.Sweep(s.ctx) }()
	<-s.org.listing
	// Released on every path, since the org's server waits for the Lists it holds to close.
	release := sync.OnceFunc(func() { close(s.org.hold) })
	defer release()

	// Bounded, so a second sweep that lists fails rather than waits for hold.
	bounded, cancel := context.WithTimeout(s.ctx, 2*time.Second)
	defer cancel()
	s.Require().NoError(second.Sweep(bounded))
	s.Equal(1, s.org.called("GET /api/secret/paginated"))

	release()
	s.Require().NoError(<-swept)
	s.Empty(s.org.names())
}
