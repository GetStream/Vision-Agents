package audiofloor

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// replica answers as one of several chorus-asr servers behind one address does: it knows the
// agents it was told, by a name of their content that every replica gives them alike, and
// answers 404 for any other. inline has it take an agent told with a decision, in a part after
// the audio, and say so in GET /v1/info; one that doesn't reads the turn and the audio and
// stops there, as a server that takes no agent inline does.
type replica struct {
	mu     sync.Mutex
	inline bool
	known  map[string]bool
	// registered are the agents each registration named, missed how many decisions it answered
	// 404, and told the agent parts its decisions carried.
	registered []string
	missed     int
	told       [][]byte
}

func newReplica(inline bool) *replica { return &replica{inline: inline, known: map[string]bool{}} }

// contentName is a replica's name for an agent: its content's.
func contentName(spec []byte) string {
	sum := sha256.Sum256(spec)
	return hex.EncodeToString(sum[:8])
}

func (r *replica) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	r.mu.Lock()
	defer r.mu.Unlock()
	switch req.URL.Path {
	case "/v1/info":
		info := testInfo
		if r.inline {
			info = strings.Replace(info, `{"template":"t",`, `{"template":"t","inline_agents":true,`, 1)
		}
		_, _ = w.Write([]byte(info))
	case "/v1/agents":
		spec, _ := io.ReadAll(req.Body)
		name := contentName(spec)
		r.known[name], r.registered = true, append(r.registered, name)
		_ = json.NewEncoder(w).Encode(map[string]string{"agent": name})
	case "/v1/decide":
		name := req.URL.Query().Get("agent")
		names, parts, ok := everyPart(req)
		if !ok || len(names) < 2 || names[0] != "turn" || names[1] != "audio" {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if len(names) == 3 && names[2] == "agent" {
			r.told = append(r.told, parts[2])
			if r.inline && contentName(parts[2]) == name {
				r.known[name] = true
			}
		}
		if !r.known[name] {
			r.missed++
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":"this server doesn't know that agent: register it with POST /v1/agents"}`))
			return
		}
		_, _ = w.Write([]byte(`{"answer":"book","p_eot":0.9,"options":[],"ruling":{"disposition":"respond","floor":"continue"}}` + "\n"))
	}
}

// everyPart is every part of req's multipart body, by name and in order.
func everyPart(req *http.Request) ([]string, [][]byte, bool) {
	media, params, err := mime.ParseMediaType(req.Header.Get("Content-Type"))
	if err != nil || media != "multipart/form-data" {
		return nil, nil, false
	}
	reader := multipart.NewReader(req.Body, params["boundary"])
	var names []string
	var parts [][]byte
	for {
		part, err := reader.NextPart()
		if errors.Is(err, io.EOF) {
			return names, parts, true
		}
		if err != nil {
			return nil, nil, false
		}
		raw, err := io.ReadAll(part)
		if err != nil {
			return nil, nil, false
		}
		names, parts = append(names, part.FormName()), append(parts, raw)
	}
}

// replicas are servers behind one address, as a Service is: each request goes to the replica
// route names for it, in the order the requests come, and to the first once route has run out.
// went is where each went.
type replicas struct {
	mu    sync.Mutex
	all   []*replica
	route []int
	went  []string
}

func (rs *replicas) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	rs.mu.Lock()
	to := 0
	if len(rs.route) > 0 {
		to, rs.route = rs.route[0], rs.route[1:]
	}
	rs.went = append(rs.went, fmt.Sprintf("%s %c", req.URL.Path, 'A'+to))
	rs.mu.Unlock()
	rs.all[to].ServeHTTP(w, req)
}

// routed has the next requests go to the replicas named, and forgets where the last ones went.
func (rs *replicas) routed(to ...int) {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	rs.route, rs.went = to, nil
}

func (rs *replicas) wentTo() []string {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	return append([]string(nil), rs.went...)
}

// twoReplicas is a client of two replicas behind one address, A and B.
func twoReplicas(t *testing.T, inline bool) (*Client, *replicas) {
	t.Helper()
	pool := &replicas{all: []*replica{newReplica(inline), newReplica(inline)}}
	srv := httptest.NewServer(pool)
	t.Cleanup(srv.Close)
	return New(srv.URL, PCM, srv.Client()), pool
}

var weather = Agent{Instructions: "Say the weather."}

// A decision that reaches a replica that was never told its agent has that agent told again,
// and it alone: every agent's name is its content's, the same on every replica, so the others
// the client knows stand, and are not registered again on their next decision. Forgetting them
// all made each miss on one replica a registration of every agent on whichever replica came
// next, over and over while any replica had yet to learn them.
func TestAReplicaThatMissesAnAgentHasThatAgentAloneToldAgain(t *testing.T) {
	client, pool := twoReplicas(t, false)
	ctx := context.Background()
	for _, agent := range []Agent{booking, weather} {
		_, err := client.Decide(ctx, agent, Turn{Speaker: caller}, some)
		require.NoError(t, err)
	}
	require.Len(t, pool.all[0].registered, 2)

	pool.routed(1, 1, 1)
	_, err := client.Decide(ctx, booking, Turn{Speaker: caller}, some)
	require.NoError(t, err)
	require.Equal(t, []string{"/v1/decide B", "/v1/agents B", "/v1/decide B"}, pool.wentTo(), "the miss, the agent told B, and the decision again")

	pool.routed()
	_, err = client.Decide(ctx, weather, Turn{Speaker: caller}, some)
	require.NoError(t, err)
	require.Equal(t, []string{"/v1/decide A"}, pool.wentTo(), "the other agent's name stands")
	require.Equal(t, 2+1, len(pool.all[0].registered)+len(pool.all[1].registered), "the agent that missed, told again, and no other")
}

// A decision that misses its agent again once it was told, the registration and the second
// decision having gone to different replicas, is a server that is up: none of its errors is a
// server unreachable or misconfigured, so a floor that rules alone does not step aside on it.
func TestAnAgentMissedWhereverItWasToldIsNoServerDown(t *testing.T) {
	client, pool := twoReplicas(t, false)
	pool.routed(0, 1, 0, 1)
	_, err := client.Decide(context.Background(), booking, Turn{Speaker: caller}, some)
	require.Error(t, err)
	require.Equal(t, []string{"/v1/agents A", "/v1/decide B", "/v1/agents A", "/v1/decide B"}, pool.wentTo())
	require.ErrorIs(t, err, ErrAgentUnknown)
	var status *StatusError
	require.ErrorAs(t, err, &status)
	require.Equal(t, http.StatusNotFound, status.Code)
	require.False(t, Unreachable(err), "a replica that has not been told the agent")
	require.False(t, Misconfigured(err))
}

// A server that takes agents inline, as GET /v1/info says, is told a decision's agent with
// it, in a part after the audio, the JSON a registration sends: any replica answers, though
// it was never told the agent itself, and none misses it.
func TestAServerThatTakesAgentsInlineIsToldTheAgentWithEachDecision(t *testing.T) {
	client, pool := twoReplicas(t, true)
	_, err := client.PolicyNow(context.Background())
	require.NoError(t, err)
	pool.routed(0, 1)
	_, err = client.Decide(context.Background(), booking, Turn{Speaker: caller}, some)
	require.NoError(t, err)
	require.Equal(t, []string{"/v1/agents A", "/v1/decide B"}, pool.wentTo())
	b := pool.all[1]
	require.Zero(t, b.missed, "B learned the agent from the decision")
	require.Empty(t, b.registered)
	_, spec, err := agentKey(booking)
	require.NoError(t, err)
	require.Equal(t, [][]byte{spec}, b.told, "what a registration sends")
}

// A server that does not take agents inline is not sent them: a decision is its turn and its
// audio, as fakeServer, which takes no third part, has every other test check.
func TestAServerThatTakesNoAgentsInlineIsSentNone(t *testing.T) {
	client, pool := twoReplicas(t, false)
	_, err := client.PolicyNow(context.Background())
	require.NoError(t, err)
	_, err = client.Decide(context.Background(), booking, Turn{Speaker: caller}, some)
	require.NoError(t, err)
	require.Empty(t, pool.all[0].told)
}

// A floor that rules alone whose checks all reach a replica never told the agent, however
// often another is, does not step aside: the server answers, and the checks are no sign that
// it is down. Counted as unreachable, three such checks over failoverWindow handed the floor to
// the other transcriber.
func (s *FloorSuite) TestAFloorWhoseChecksMissTheAgentOnAReplicaDoesNotStepAside() {
	clock := time.Now()
	var mu sync.Mutex
	s.transcriber.now = func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		return clock
	}
	s.server.mu.Lock()
	s.server.unknown = true
	s.server.mu.Unlock()
	for i := range failoverAfter + 1 {
		s.say(600, true)
		s.say(220, false)
		s.answered()
		s.say(2600, false)
		s.True(s.writes(int64(i+1), stt.ModeFinal, "hello"), "still the floor's after %d", i+1)
		mu.Lock()
		clock = clock.Add(failoverWindow / 2)
		mu.Unlock()
	}
	s.False(s.transcriber.stepped.Load())
}

// A check a floor that rules alone held back while the server asked for time, over the agent,
// was never asked, and counts neither way: the checks the server did not answer before it are
// still in a row with those after it. Counted as answered, a held check started the count over,
// and a server down at every pause around it was never given up.
func (s *FloorSuite) TestAFloorCountsNoCheckHeldBackWhileTheServerAsksForTime() {
	clock := time.Now()
	var mu sync.Mutex
	s.transcriber.now = func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		return clock
	}
	s.server.mu.Lock()
	s.server.failDecide = true
	s.server.mu.Unlock()
	s.client.backOff("30")
	for i := range failoverAfter {
		if i == failoverAfter-1 {
			held := testutil.ToFloat64(backedOff.WithLabelValues(kindOverAgent))
			s.agent.speaks(true)
			s.say(1000, true)
			s.Eventually(func() bool { return testutil.ToFloat64(backedOff.WithLabelValues(kindOverAgent)) > held }, 5*time.Second, time.Millisecond,
				"a check over the agent held back")
			s.agent.speaks(false)
		}
		s.say(600, true)
		s.say(220, false)
		s.answered()
		if i < failoverAfter-1 {
			s.say(2600, false)
		}
		mu.Lock()
		clock = clock.Add(failoverWindow / 2)
		mu.Unlock()
	}
	gone := next[stt.Disconnected](&s.transcribing)
	s.False(gone.Clean)
}

// The router's client of the server closes a connection it has not used for a while before the
// server does (gophonic cmd/qwen3asr-decide, an idle timeout of 60 s), so that no request goes
// out on one the server is closing; and keeps as many a host as the router's default transport.
func TestTheClientClosesAnIdleConnectionBeforeTheServerDoes(t *testing.T) {
	const serverIdleTimeout = 60 * time.Second
	for name, client := range map[string]*Client{
		"open": Open(config.Agent{AudioFloorURL: "http://floor:8080"}),
		"new":  New("http://floor:8080", Opus, nil),
	} {
		transport, ok := client.http.Transport.(*http.Transport)
		require.True(t, ok, "%s: a transport of its own", name)
		require.Positive(t, transport.IdleConnTimeout, name)
		require.Less(t, transport.IdleConnTimeout, serverIdleTimeout, name)
		require.Equal(t, 256, transport.MaxIdleConnsPerHost, name)
	}
}

// A request that goes out on a kept-alive connection the server is closing, as one past its
// idle timeout or shutting down closes it, is sent again on a new one: every request the
// client sends is idempotent. net/http sends a POST again only when told it is; without that,
// the request failed as a connection, nothing having answered it.
func TestARequestOnAConnectionTheServerClosedIsSentAgain(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	answer := `{"transcript":{"text":"hello","language":"English"}}` + "\n"
	go func() {
		// The first connection answers one request, then is closed as the next comes in on it.
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		read := bufio.NewReader(conn)
		for i := range 2 {
			req, err := http.ReadRequest(read)
			if err != nil {
				break
			}
			_, _ = io.Copy(io.Discard, req.Body)
			if i == 0 {
				_, _ = fmt.Fprintf(conn, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s", len(answer), answer)
			}
		}
		_ = conn.Close()
		// The connections after it are answered.
		_ = http.Serve(listener, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.Copy(io.Discard, r.Body)
			_, _ = w.Write([]byte(answer))
		}))
	}()
	t.Cleanup(func() { _ = listener.Close() })
	client := New("http://"+listener.Addr().String(), PCM, &http.Client{Transport: &http.Transport{}})
	for i := range 2 {
		said, err := client.Transcribe(context.Background(), some, nil)
		require.NoError(t, err, "request %d", i+1)
		require.Equal(t, "hello", said.Text)
	}
}

// A decision or a transcription tells the server how long the router will wait for its
// answer (TimeoutHeader), in milliseconds from when it was sent: a decision with its words
// the decision's deadline, past which nobody waits for the words either. A request with no
// deadline says none.
func (s *ClientSuite) TestARequestSaysHowLongItIsWaitedFor() {
	within := func(timeout time.Duration, ask func(context.Context)) {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		ask(ctx)
	}
	within(time.Second, func(ctx context.Context) {
		_, err := s.client.Decide(ctx, booking, Turn{Speaker: caller}, some)
		s.Require().NoError(err)
	})
	within(400*time.Millisecond, func(ctx context.Context) {
		_, said, err := s.client.DecideWords(ctx, booking, Turn{Speaker: caller}, some)
		s.Require().NoError(err)
		s.Require().NoError((<-said).Err)
	})
	within(3*time.Second, func(ctx context.Context) {
		_, err := s.client.Transcribe(ctx, some, nil)
		s.Require().NoError(err)
	})
	_, err := s.client.Transcribe(context.Background(), some, nil)
	s.Require().NoError(err)
	s.server.mu.Lock()
	deadlines := append([]string(nil), s.server.deadlines...)
	s.server.mu.Unlock()
	s.Require().Len(deadlines, 4)
	for i, timeout := range []time.Duration{time.Second, 400 * time.Millisecond, 3 * time.Second} {
		ms, err := strconv.Atoi(deadlines[i])
		s.Require().NoError(err, "request %d: %q", i+1, deadlines[i])
		s.LessOrEqual(ms, int(timeout.Milliseconds()), "request %d", i+1)
		s.Greater(ms, int(timeout.Milliseconds())-200, "request %d", i+1)
	}
	s.Empty(deadlines[3], "no deadline, none said")
}

// A 503 with Retry-After has the client hold back what decides no turn until it has passed: a
// check over the agent is not sent, and says so (ErrBackingOff), which is neither a server down
// nor misconfigured; a pause's check is sent all the same, as it decides the turn.
func (s *ClientSuite) TestARetryAfterHoldsTheChecksOverTheAgentButNotThePauses() {
	var skew atomic.Int64
	s.client.now = skewed(&skew)
	over := Turn{Speaker: caller, Speaking: true, Unfinished: true}
	_, err := s.client.Decide(context.Background(), booking, over, some)
	s.Require().NoError(err, "sent before the server asked for time")

	s.server.mu.Lock()
	s.server.failing, s.server.failWith, s.server.retryAfter = 1, http.StatusServiceUnavailable, "2"
	s.server.mu.Unlock()
	_, err = s.client.Transcribe(context.Background(), some, nil)
	s.Require().Error(err)
	held := testutil.ToFloat64(backedOff.WithLabelValues(kindOverAgent))
	asked := s.decisionsAsked()
	_, err = s.client.Decide(context.Background(), booking, over, some)
	s.Require().ErrorIs(err, ErrBackingOff)
	s.False(Unreachable(err))
	s.False(Misconfigured(err))
	s.Equal("backed_off", outcome(err))
	s.Equal(asked, s.decisionsAsked(), "not sent")
	s.Equal(held+1, testutil.ToFloat64(backedOff.WithLabelValues(kindOverAgent)))

	_, err = s.client.Decide(context.Background(), booking, Turn{Speaker: caller}, some)
	s.Require().NoError(err, "a pause's check is sent")
	_, err = s.client.Decide(context.Background(), booking, Turn{Speaker: caller, Speaking: true}, some)
	s.Require().NoError(err, "and one at a pause over the agent")

	skew.Store(int64(2 * time.Second))
	_, err = s.client.Decide(context.Background(), booking, over, some)
	s.Require().NoError(err, "sent again once the Retry-After has passed")
}

// decisionsAsked is how many decisions the server was asked for, answered or not.
func (s *ClientSuite) decisionsAsked() int {
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	return len(s.server.decidesAt)
}

// A Retry-After is seconds or an HTTP date, heeded up to maxBackOff; one that has passed, is
// zero or is neither changes nothing, and a shorter one does not cut a longer one short.
func TestARetryAfterIsSecondsOrADateAndBounded(t *testing.T) {
	now := time.Date(2026, 11, 10, 12, 0, 0, 0, time.UTC)
	for retryAfter, want := range map[string]time.Duration{
		"2":    2 * time.Second,
		" 5 ":  5 * time.Second,
		"3600": maxBackOff,
		"0":    0,
		"-3":   0,
		"soon": 0,
		"":     0,
		now.Add(7 * time.Second).Format(http.TimeFormat):  7 * time.Second,
		now.Add(-7 * time.Second).Format(http.TimeFormat): 0,
		now.Add(time.Hour).Format(http.TimeFormat):        maxBackOff,
	} {
		client := New("http://floor:8080", PCM, nil)
		client.now = func() time.Time { return now }
		client.backOff(retryAfter)
		got := time.Duration(0)
		if !client.backOffUntil.IsZero() {
			got = client.backOffUntil.Sub(now)
		}
		require.Equal(t, want, got, "%q", retryAfter)
	}
	client := New("http://floor:8080", PCM, nil)
	client.now = func() time.Time { return now }
	client.backOff("10")
	client.backOff("1")
	require.Equal(t, now.Add(10*time.Second), client.backOffUntil)
}

// While the server's Retry-After lasts, a transcriber asks no interims and no checks over the
// agent, and asks about its pauses all the same, a pause's first check the decision and its
// words in one request, as the pauses decide the turn. Each request held back is counted.
func (s *PausesSuite) TestARetryAfterHoldsTheInterimsAndTheChecksOverTheAgentButNotThePauses() {
	s.transcriber.interims = true
	s.server.mu.Lock()
	s.server.failing, s.server.failWith, s.server.retryAfter = 1, http.StatusServiceUnavailable, "30"
	s.server.mu.Unlock()
	refused := testutil.ToFloat64(interims.WithLabelValues("refused"))
	s.say(1000, true)
	s.Eventually(func() bool { return testutil.ToFloat64(interims.WithLabelValues("refused")) > refused }, 5*time.Second, time.Millisecond,
		"the first interim, answered 503 with a Retry-After")

	heldInterims, heldOver := testutil.ToFloat64(backedOff.WithLabelValues(kindInterim)), testutil.ToFloat64(backedOff.WithLabelValues(kindOverAgent))
	s.say(2000, true)
	s.Greater(testutil.ToFloat64(backedOff.WithLabelValues(kindInterim)), heldInterims, "interims held back")
	// Over the agent its checks are what write the words so far, and no interim is asked.
	s.agent.speaks(true)
	s.say(1000, true)
	s.Eventually(func() bool { return testutil.ToFloat64(backedOff.WithLabelValues(kindOverAgent)) > heldOver }, 5*time.Second, time.Millisecond,
		"a check over the agent held back")
	over, _, transcripts := s.asked()
	s.Equal([2]int{0, 0}, [2]int{over, transcripts}, "no check over the agent and no interim reached the server")

	_, withWords, _ := s.counts()
	s.say(220, false)
	s.answered()
	over, atPauses, _ := s.asked()
	s.Zero(over)
	s.Positive(atPauses, "the pause over the agent is asked about")
	_, now, _ := s.counts()
	s.Greater(now, withWords, "with its words")
}

// asked is how many checks the server was asked for over the agent and at pauses over it, and
// how many requests for words alone.
func (s *PausesSuite) asked() (over, atPauses, transcripts int) {
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	for _, turn := range s.server.turns {
		switch {
		case turn.Unfinished:
			over++
		case turn.Speaking:
			atPauses++
		}
	}
	return over, atPauses, len(s.server.audio) - len(s.server.turns)
}

// A decision's words the server answered without, every lane that writes them busy, are
// counted busy: the server answered. They were counted a connection that failed, which had
// minutes of busy answers in the router's logs show as connection failures.
func (s *PausesSuite) TestAPausesWordsAnsweredBusyAreBusyNotAConnection() {
	s.server.mu.Lock()
	s.server.wordsBusy = true
	s.server.mu.Unlock()
	busy, connection := testutil.ToFloat64(transcribeFailures.WithLabelValues("busy")), testutil.ToFloat64(transcribeFailures.WithLabelValues("connection"))
	s.say(1000, true)
	s.say(220, false)
	s.Eventually(func() bool { return testutil.ToFloat64(transcribeFailures.WithLabelValues("busy")) == busy+1 }, 5*time.Second, time.Millisecond)
	s.Equal(connection, testutil.ToFloat64(transcribeFailures.WithLabelValues("connection")))
}

// Why a request failed is what the server answered, when it answered: an answer without the
// words is the server's, busy or not, and only a request nothing answered is a connection.
func TestAFailureIsWhatTheServerAnswered(t *testing.T) {
	for err, want := range map[error]string{
		&StatusError{Code: http.StatusServiceUnavailable}:                        "503",
		fmt.Errorf("%w: %w", ErrAgentUnknown, &StatusError{Code: 404}):           "404",
		&TranscriptError{Request: "/v1/decide: the words", Message: "busy"}:      "busy",
		&TranscriptError{Request: "transcribe", Message: "transcript_failed"}:    "no_words",
		fmt.Errorf("audiofloor: decide: %w", context.DeadlineExceeded):           "timeout",
		&net.OpError{Op: "dial", Err: errors.New("connect: connection refused")}: "connection",
	} {
		require.Equal(t, want, failure(err), "%v", err)
	}
}

// requestErrorsOf is how many requests of kind failed with code.
func requestErrorsOf(kind, code string) float64 {
	return testutil.ToFloat64(requestErrors.WithLabelValues(kind, code))
}

// setupTrips is how many requests to route came to outcome.
func setupTrips(t *testing.T, route, outcome string) uint64 {
	t.Helper()
	var metric dto.Metric
	require.NoError(t, setupRoundTrips.WithLabelValues(route, outcome).(prometheus.Metric).Write(&metric))
	return metric.GetHistogram().GetSampleCount()
}

// Every request that fails is counted by what it was for and the status the server answered,
// or why nothing did; and registering an agent and reading the policy are timed, as decisions
// and words are.
func (s *ClientSuite) TestFailuresAreCountedByStatusAndTheSetUpIsTimed() {
	info, agents := setupTrips(s.T(), routeInfo, "ok"), setupTrips(s.T(), routeAgents, "ok")
	_, err := s.client.PolicyNow(context.Background())
	s.Require().NoError(err)
	_, err = s.client.Decide(context.Background(), booking, Turn{Speaker: caller}, some)
	s.Require().NoError(err)
	s.Equal(info+1, setupTrips(s.T(), routeInfo, "ok"))
	s.Equal(agents+1, setupTrips(s.T(), routeAgents, "ok"))

	pause, words := requestErrorsOf(kindPause, "500"), requestErrorsOf(kindTranscribe, "503")
	s.server.mu.Lock()
	s.server.failingDecides, s.server.failing, s.server.failWith = 1, 1, http.StatusServiceUnavailable
	s.server.mu.Unlock()
	_, err = s.client.Decide(context.Background(), booking, Turn{Speaker: caller}, some)
	s.Require().Error(err)
	_, err = s.client.Transcribe(context.Background(), some, nil)
	s.Require().Error(err)
	s.Equal(pause+1, requestErrorsOf(kindPause, "500"))
	s.Equal(words+1, requestErrorsOf(kindTranscribe, "503"))

	missed := requestErrorsOf(kindPause, "404")
	s.server.mu.Lock()
	s.server.unknown = true
	s.server.mu.Unlock()
	_, err = s.client.Decide(context.Background(), booking, Turn{Speaker: caller}, some)
	s.Require().ErrorIs(err, ErrAgentUnknown)
	s.Equal(missed+2, requestErrorsOf(kindPause, "404"), "the miss, and the one after the agent was told again")
}
