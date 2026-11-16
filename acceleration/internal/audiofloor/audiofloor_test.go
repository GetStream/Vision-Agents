package audiofloor

import (
	"bytes"
	"cmp"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"github.com/thesyncim/gopus"
	"github.com/thesyncim/gopus/container/ogg"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// testInfo is /v1/info as a server publishes it: the measured frontier of a checkpoint, and
// presets naming its points.
const testInfo = `{"template":"t","end_of_turn":{"first_check_ms":200,"first_pause_logit":0.5,"interval_ms":300,"cap_ms":2500,` +
	`"default":"safe","presets":{"safe":1100,"fast":500},"frontier":[` + testFrontier + `]}}`

// testFrontier is that checkpoint's frontier: its 500 ms point fails the dictation guard, and
// its 1400 ms point ends a turn on two checks in a row.
const testFrontier = `{"max_wait_ms":500,"bars":[{"from_ms":0,"bar":0.8},{"from_ms":500,"bar":0.3}],"checks_in_a_row":1,` +
	`"measured":{"cut_ins":0.1451,"answer_ms_median":500,"answer_ms_p90":800,"dictation_pauses_cut":0.16,"slow_counts_cut":9,"slow_counts":9,"dictation_guard":false}},` +
	`{"max_wait_ms":1100,"bars":[{"from_ms":0,"bar":0.995},{"from_ms":1100,"bar":0.2}],"checks_in_a_row":1,` +
	`"measured":{"cut_ins":0.0454,"answer_ms_median":1100,"answer_ms_p90":1100,"dictation_pauses_cut":0.1257,"slow_counts_cut":0,"slow_counts":9,"dictation_guard":true}},` +
	`{"max_wait_ms":1400,"bars":[{"from_ms":0,"bar":0.99},{"from_ms":1100,"bar":0.2}],"checks_in_a_row":2,` +
	`"measured":{"cut_ins":0.0307,"answer_ms_median":1400,"answer_ms_p90":1400,"dictation_pauses_cut":0.1057,"slow_counts_cut":0,"slow_counts":9,"dictation_guard":true}}`

// fakeServer answers as a chorus-asr server does, with what a test sets.
type fakeServer struct {
	mu         sync.Mutex
	registered []agentJSON
	known      map[string]bool
	infos      int
	turns      []Turn
	audio      [][]byte
	types      []string
	languages  []string
	// refuse is the status /v1/agents answers an agent with tools, fail has /v1/decide
	// answer 500, info is /v1/info's body, and noPolicy has /v1/info answer 501, as a
	// server started without a policy for its model does.
	refuse   int
	fail     bool
	info     string
	noPolicy bool
	// failing is how many transcription requests to answer failWith, before answering,
	// busy how many to answer without words, every lane busy, failDecide has every decision
	// fail, and failingDecides the next that many.
	failing        int
	failWith       int
	busy           int
	failDecide     bool
	failingDecides int
	// decidesAt is when each decision was asked for, answered or not.
	decidesAt []time.Time
	// transcribeGate, when set, holds every transcription request until it is closed, or
	// with holdOne the next one alone, and held counts the requests waiting at it. failHeld
	// has a request let go of answer failWith.
	transcribeGate chan struct{}
	holdOne        bool
	held           int
	failHeld       bool
	// language is the language the server says it heard words in, English unless set, and
	// noise, when set, has it answer transcriptions with no words and that as the reason.
	language string
	noise    string
	// decideAfter is how long a decision takes, and wordsAfter how long its words take after
	// it.
	decideAfter time.Duration
	wordsAfter  time.Duration
	// wordsGate, when set, holds the words of the next decision asked for them until it is
	// closed: that decision's alone, the ones after it answering as they come.
	wordsGate chan struct{}
	// unfinishedAfter is how long a decision about words still coming, a check over the
	// agent, takes on top of decideAfter.
	unfinishedAfter time.Duration
	// withWords counts the decisions asked for their words too, and agents the agents each
	// decision was about.
	withWords int
	agents    []string
	decision  string
	words     string
	// speakingDecision and speakingWords, when set, answer a check at a pause over the agent
	// (speaking), and unfinishedDecision and unfinishedWords one over the agent without a
	// pause (unfinished), in place of decision and words: the server's other frames.
	speakingDecision, unfinishedDecision string
	speakingWords, unfinishedWords       *string
	// blanked is the reason a decision's words give for being none, as the server says why
	// it wrote nothing of a murmur it ruled ignored.
	blanked string
	// unknown has every decision answer 404, as a replica behind the address that was never
	// told the agent does, however often another is told; retryAfter is the Retry-After a
	// transcription answered failWith carries; wordsBusy has a decision's words answer that
	// every lane that writes them is busy; and deadlines are the TimeoutHeader of each
	// decision and transcription asked for.
	unknown    bool
	retryAfter string
	wordsBusy  bool
	deadlines  []string
}

// requests is how many decisions the server was asked for.
func (f *fakeServer) requests() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.audio)
}

func (f *fakeServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	switch r.URL.Path {
	case "/v1/info":
		f.infos++
		if f.noPolicy {
			w.WriteHeader(http.StatusNotImplemented)
			_, _ = w.Write([]byte(`{"error":"no policy for this model"}`))
			return
		}
		_, _ = w.Write([]byte(f.info))
	case "/v1/agents":
		var spec agentJSON
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &spec)
		f.registered = append(f.registered, spec)
		if f.refuse != 0 && len(spec.Tools) > 0 {
			w.WriteHeader(f.refuse)
			_, _ = w.Write([]byte(`{"error":"floor: 22 tools, and an agent can have 21"}`))
			return
		}
		id := fmt.Sprintf("agent-%d", len(f.registered))
		f.known[id] = true
		_ = json.NewEncoder(w).Encode(map[string]string{"agent": id})
	case "/v1/transcribe":
		if f.fail {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		f.deadlines = append(f.deadlines, r.Header.Get(TimeoutHeader))
		if f.failing > 0 {
			f.failing--
			if f.retryAfter != "" {
				w.Header().Set("Retry-After", f.retryAfter)
			}
			w.WriteHeader(f.failWith)
			_, _ = w.Write([]byte(`{"error":"every lane is busy"}`))
			return
		}
		if f.busy > 0 {
			f.busy--
			_, _ = w.Write([]byte(`{"transcript_error":"busy"}` + "\n"))
			return
		}
		// The audio comes as a decision's does, in a part of its own, which a gateway that
		// changes a raw body's type passes as it was sent.
		parts, types, ok := formParts(r, "audio")
		if !ok {
			w.WriteHeader(http.StatusUnsupportedMediaType)
			_, _ = w.Write([]byte(`{"error":"not a multipart body of one audio part"}`))
			return
		}
		f.audio, f.types = append(f.audio, parts[0]), append(f.types, types[0])
		f.languages = append(f.languages, r.URL.Query().Get("languages"))
		if gate := f.transcribeGate; gate != nil {
			f.held++
			if f.holdOne {
				f.transcribeGate = nil
			}
			f.mu.Unlock()
			<-gate
			f.mu.Lock()
			if f.failHeld {
				f.failHeld = false
				w.WriteHeader(f.failWith)
				_, _ = w.Write([]byte(`{"error":"every lane is busy"}`))
				return
			}
		}
		if f.noise != "" {
			_, _ = fmt.Fprintf(w, `{"transcript":{"text":"","language":%q,"noise":%q}}`+"\n", f.spoken(), f.noise)
			return
		}
		_, _ = fmt.Fprintf(w, `{"transcript":{"text":%q,"language":%q}}`+"\n", f.words, f.spoken())
	case "/v1/decide":
		f.decidesAt = append(f.decidesAt, time.Now())
		f.deadlines = append(f.deadlines, r.Header.Get(TimeoutHeader))
		if f.fail || f.failDecide || f.failingDecides > 0 {
			if f.failingDecides > 0 {
				f.failingDecides--
			}
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		if f.unknown || !f.known[r.URL.Query().Get("agent")] {
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":"this server doesn't know that agent: register it with POST /v1/agents"}`))
			return
		}
		parts, types, ok := formParts(r, "turn", "audio")
		if !ok {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		var turn Turn
		_ = json.Unmarshal(parts[0], &turn)
		f.audio, f.types = append(f.audio, parts[1]), append(f.types, types[1])
		f.turns = append(f.turns, turn)
		f.agents = append(f.agents, r.URL.Query().Get("agent"))
		if after := f.decideAfter + map[bool]time.Duration{true: f.unfinishedAfter}[turn.Unfinished]; after > 0 {
			f.mu.Unlock()
			time.Sleep(after)
			f.mu.Lock()
		}
		decision, words := f.decision, f.words
		switch {
		case turn.Unfinished:
			decision, words = cmp.Or(f.unfinishedDecision, decision), *cmp.Or(f.unfinishedWords, &words)
		case turn.Speaking:
			decision, words = cmp.Or(f.speakingDecision, decision), *cmp.Or(f.speakingWords, &words)
		}
		_, _ = w.Write([]byte(decision + "\n"))
		if r.URL.Query().Get("transcript") == "true" {
			f.withWords++
			if f.wordsAfter > 0 {
				w.(http.Flusher).Flush()
				f.mu.Unlock()
				time.Sleep(f.wordsAfter)
				f.mu.Lock()
			}
			if gate := f.wordsGate; gate != nil {
				f.wordsGate = nil
				w.(http.Flusher).Flush()
				f.mu.Unlock()
				<-gate
				f.mu.Lock()
			}
			if f.wordsBusy {
				_, _ = w.Write([]byte(`{"transcript_error":"busy"}` + "\n"))
				return
			}
			_, _ = fmt.Fprintf(w, `{"transcript":{"text":%q,"language":%q,"noise":%q},"transcript_ms":3}`+"\n", words, f.spoken(), f.blanked)
		}
	}
}

// formParts reads r's multipart/form-data body, which must be the parts names, in that order
// and no others: each one's bytes and type.
func formParts(r *http.Request, names ...string) ([][]byte, []string, bool) {
	media, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || media != "multipart/form-data" {
		return nil, nil, false
	}
	reader := multipart.NewReader(r.Body, params["boundary"])
	var parts [][]byte
	var types []string
	for _, name := range names {
		part, err := reader.NextPart()
		if err != nil || part.FormName() != name {
			return nil, nil, false
		}
		raw, err := io.ReadAll(part)
		if err != nil {
			return nil, nil, false
		}
		parts, types = append(parts, raw), append(types, part.Header.Get("Content-Type"))
	}
	if _, err := reader.NextPart(); err != io.EOF {
		return nil, nil, false
	}
	return parts, types, true
}

// spoken is the language the server says it heard. The caller holds the lock.
func (f *fakeServer) spoken() string { return cmp.Or(f.language, "English") }

// serverSuite is a Client of a fake server.
type serverSuite struct {
	suite.Suite
	server *fakeServer
	client *Client
}

func (s *serverSuite) SetupTest() {
	s.server = &fakeServer{known: map[string]bool{}, info: testInfo,
		decision: `{"answer":"book","answer_probability":0.9,"p_eot":0.95,"options":[{"name":"book","p":0.9},{"name":"wait","p":0.05}],"decide_ms":12,"ruling":{"disposition":"respond","floor":"continue","tool":"book"}}`}
	srv := httptest.NewServer(s.server)
	s.T().Cleanup(srv.Close)
	s.client = New(srv.URL+"/", PCM, srv.Client())
}

// ClientSuite asks a fake server.
type ClientSuite struct{ serverSuite }

func TestClientSuite(t *testing.T) {
	suite.Run(t, new(ClientSuite))
}

var booking = Agent{Instructions: "Take bookings.", Tools: []llm.Tool{{Name: "book", Description: "Book it.", Parameters: map[string]any{"required": []any{"day"}}}}}

// some is a moment of audio.
var some = harness.Audio{Type: pcmType, Data: []byte{1, 0, 0xff, 0xff}, Duration: time.Millisecond / 8}

func (s *ClientSuite) TestAnAgentIsRegisteredOnceAndTheMomentGoesInTheBody() {
	for range 2 {
		got, err := s.client.Decide(context.Background(), booking, Turn{Speaker: caller, Speaking: true, Said: "Votre traversée à 8 h"}, some)
		s.Require().NoError(err)
		s.Equal(Ruling{Disposition: "respond", Floor: "continue", Tool: "book"}, *got.Ruling)
		s.InDelta(0.95, *got.PEOT, 1e-9)
	}
	s.Len(s.server.registered, 1)
	s.Equal([]any{"day"}, s.server.registered[0].Tools[0].Parameters["required"])
	s.Equal("Votre traversée à 8 h", s.server.turns[0].Said)
	s.Equal([]byte{1, 0, 0xff, 0xff}, s.server.audio[0])
	s.Equal("audio/pcm;rate=16000;channels=1;format=s16le", s.server.types[0])
}

// The settings open a client whose transcribers write the words so far while the caller
// talks unless they say off.
func TestTheSettingsSayWhetherTheWordsAreWrittenWhileTheCallerTalks(t *testing.T) {
	for setting, interims := range map[string]bool{"": true, "on": true, "off": false} {
		client := Open(config.Agent{AudioFloorURL: "http://floor:8080", AudioFloorInterims: setting})
		require.NotNil(t, client, setting)
		require.Equal(t, interims, NewTranscriber(client, "", nil, slog.New(slog.DiscardHandler)).interims, "interims %q", setting)
	}
	require.Nil(t, Open(config.Agent{}))
}

// A gateway's address has a path of its own, and every route goes under it, with the key when
// there is one.
func TestEveryRequestGoesUnderTheAddressesPathWithTheKey(t *testing.T) {
	const prefix = "/environments/production/sync"
	for _, key := range []string{"", "a-key"} {
		fake := &fakeServer{known: map[string]bool{}, info: testInfo, decision: `{"answer":"respond","p_eot":0.9,"options":[{"name":"respond","p":0.9}],"ruling":{"disposition":"respond","floor":"continue"}}`, words: "hello"}
		var (
			mu     sync.Mutex
			routes []string
		)
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			route, found := strings.CutPrefix(r.URL.Path, prefix)
			require.True(t, found, r.URL.Path)
			wantAuth := ""
			if key != "" {
				wantAuth = "Bearer " + key
			}
			require.Equal(t, wantAuth, r.Header.Get("Authorization"), route)
			mu.Lock()
			routes = append(routes, route)
			mu.Unlock()
			r.URL.Path = route
			fake.ServeHTTP(w, r)
		}))
		defer srv.Close()
		// Written with and without the slash an address may end in.
		for _, address := range []string{srv.URL + prefix, srv.URL + prefix + "/"} {
			routes = nil
			client := Open(config.Agent{AudioFloorURL: address, AudioFloorAudio: "pcm", AudioFloorAPIKey: key})
			_, err := client.PolicyNow(context.Background())
			require.NoError(t, err)
			_, err = client.Decide(context.Background(), booking, Turn{Speaker: caller}, some)
			require.NoError(t, err)
			_, said, err := client.DecideWords(context.Background(), booking, Turn{Speaker: caller}, some)
			require.NoError(t, err)
			require.NoError(t, (<-said).Err)
			_, err = client.Transcribe(context.Background(), some, nil)
			require.NoError(t, err)
			require.Equal(t, []string{"/v1/info", "/v1/agents", "/v1/decide", "/v1/decide", "/v1/transcribe"}, routes)
		}
	}
}

// A gateway that turns the key away is the router's to fix, not a server that is down, and
// what it says, which is logged, doesn't carry the key, even when it quotes it.
func TestARefusedKeyIsMisconfiguredAndInNoError(t *testing.T) {
	const key = "a-key-to-keep-out-of-the-logs"
	for _, status := range []int{http.StatusUnauthorized, http.StatusForbidden} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(status)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": "no such key: " + r.Header.Get("Authorization")})
		}))
		defer srv.Close()
		client := New(srv.URL, PCM, srv.Client()).WithAPIKey(key)
		_, policy := client.PolicyNow(context.Background())
		_, decide := client.Decide(context.Background(), booking, Turn{Speaker: caller}, some)
		_, _, words := client.DecideWords(context.Background(), Agent{}, Turn{Speaker: caller}, some)
		_, transcribe := client.Transcribe(context.Background(), some, nil)
		for name, err := range map[string]error{"policy": policy, "decide": decide, "words": words, "transcribe": transcribe} {
			require.Error(t, err, name)
			require.True(t, Misconfigured(err), "%s: %d is misconfigured", name, status)
			require.False(t, Unreachable(err), "%s: %d is not a server down", name, status)
			require.NotContains(t, fmt.Sprintf("%+v", err), key, name)
			require.Contains(t, err.Error(), "[key]", name)
		}
	}
}

func (s *ClientSuite) TestAChangedAgentIsRegisteredAgain() {
	for _, instructions := range []string{"One.", "One.", "Two."} {
		_, err := s.client.Decide(context.Background(), Agent{Instructions: instructions}, Turn{}, some)
		s.Require().NoError(err)
	}
	s.Len(s.server.registered, 2)
}

// infos is how many times the server was asked for its policy.
func (s *serverSuite) infos() int {
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	return s.server.infos
}

// publish has the server publish another policy: the same, with default as its default.
func (s *serverSuite) publish(preset string) {
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	s.server.info = strings.Replace(testInfo, `"default":"safe"`, `"default":"`+preset+`"`, 1)
}

// A restarted server has forgotten its agents, and may serve another model and policy: the
// policy is read again, in the background, and the last one read stands until then.
func (s *ClientSuite) TestARestartedServerIsAskedAgain() {
	_, err := s.client.Policy(context.Background())
	s.Require().NoError(err)
	_, err = s.client.Decide(context.Background(), booking, Turn{}, some)
	s.Require().NoError(err)
	s.server.mu.Lock()
	s.server.known = map[string]bool{}
	s.server.mu.Unlock()
	s.publish("fast")
	_, err = s.client.Decide(context.Background(), booking, Turn{}, some)
	s.Require().NoError(err)
	s.Len(s.server.registered, 2)
	s.Eventually(func() bool { return s.infos() == 2 }, 5*time.Second, time.Millisecond)
	s.Eventually(func() bool {
		policy, err := s.client.Policy(context.Background())
		return err == nil && policy.Default == "fast"
	}, 5*time.Second, time.Millisecond)
}

// A policy stands for policyTTL, and is then read again in the background: the call that
// finds it old gets the last one read, and the next the new one.
func (s *ClientSuite) TestAnOldPolicyIsReadAgainInTheBackground() {
	var skew atomic.Int64
	s.client.now = skewed(&skew)
	first, err := s.client.Policy(context.Background())
	s.Require().NoError(err)
	s.publish("fast")
	again, err := s.client.Policy(context.Background())
	s.Require().NoError(err)
	s.Equal(first.Default, again.Default, "stands within the TTL")
	s.Equal(1, s.infos())

	skew.Store(int64(policyTTL))
	old, err := s.client.Policy(context.Background())
	s.Require().NoError(err)
	s.Equal("safe", old.Default, "the last one read, while it is read again")
	s.Eventually(func() bool {
		policy, err := s.client.Policy(context.Background())
		return err == nil && policy.Default == "fast"
	}, 5*time.Second, time.Millisecond)
	s.Equal(2, s.infos(), "read again once")
}

// reading says the client is reading its policy again in the background.
func (s *ClientSuite) reading() bool {
	s.client.mu.Lock()
	defer s.client.mu.Unlock()
	return s.client.reading
}

// A read that fails leaves the last policy standing, and is tried again after policyRetry.
func (s *ClientSuite) TestAFailedReadLeavesTheLastPolicyStanding() {
	var skew atomic.Int64
	s.client.now = skewed(&skew)
	_, err := s.client.Policy(context.Background())
	s.Require().NoError(err)
	s.server.mu.Lock()
	s.server.info = `{"end_of_turn":{}}`
	s.server.mu.Unlock()
	skew.Store(int64(policyTTL))
	_, err = s.client.Policy(context.Background())
	s.Require().NoError(err)
	s.Eventually(func() bool { return s.infos() == 2 }, 5*time.Second, time.Millisecond)
	// The server has answered; the client sets when to try again only once it has read the
	// answer, by its clock, so the clock waits for that.
	s.Eventually(func() bool { return !s.reading() }, 5*time.Second, time.Millisecond)
	policy, err := s.client.Policy(context.Background())
	s.Require().NoError(err)
	s.Equal("safe", policy.Default)
	s.Equal(2, s.infos(), "not before policyRetry")

	s.publish("fast")
	skew.Add(int64(policyRetry))
	_, err = s.client.Policy(context.Background())
	s.Require().NoError(err)
	s.Eventually(func() bool {
		policy, err := s.client.Policy(context.Background())
		return err == nil && policy.Default == "fast"
	}, 5*time.Second, time.Millisecond)
}

// An agent with more tools than the model has answers for is refused, and isn't offered
// again until it changes. The server may have moved to another model to refuse it, so its
// policy is read again, in the background.
func (s *ClientSuite) TestARefusedAgentIsRefusedUntilItChanges() {
	_, err := s.client.Policy(context.Background())
	s.Require().NoError(err)
	s.server.refuse = http.StatusUnprocessableEntity
	for range 2 {
		_, err := s.client.Decide(context.Background(), booking, Turn{}, some)
		s.ErrorIs(err, ErrRefused)
		s.ErrorContains(err, "22 tools")
	}
	s.Len(s.server.registered, 1)
	_, err = s.client.Decide(context.Background(), Agent{Instructions: booking.Instructions}, Turn{}, some)
	s.Require().NoError(err)
	s.Eventually(func() bool { return s.infos() == 2 }, 5*time.Second, time.Millisecond)
}

func (s *ClientSuite) TestAServerWithoutAPolicyIsUnsupported() {
	s.server.refuse = http.StatusNotImplemented
	_, err := s.client.Decide(context.Background(), booking, Turn{}, some)
	s.ErrorIs(err, ErrUnsupported)
	const times = `"first_check_ms":200,"interval_ms":300,"cap_ms":2500`
	point := func(bars string) string {
		return `{"max_wait_ms":1100,"bars":` + bars + `,"checks_in_a_row":1,"measured":{"cut_ins":0.05,"answer_ms_median":1100,"answer_ms_p90":1100}}`
	}
	for _, info := range []string{
		`{"end_of_turn":{}}`,
		`{"end_of_turn":{` + times + `,"default":"a","presets":{"a":1100}}}`,
		`{"end_of_turn":{` + times + `,"default":"b","presets":{"a":1100},"frontier":[` + point(`[{"from_ms":0,"bar":0.9}]`) + `]}}`,
		`{"end_of_turn":{` + times + `,"default":"a","presets":{"a":1100,"b":900},"frontier":[` + point(`[{"from_ms":0,"bar":0.9}]`) + `]}}`,
		`{"end_of_turn":{` + times + `,"default":"a","presets":{"a":1100},"frontier":[` + point(`[{"from_ms":500,"bar":0.9}]`) + `]}}`,
		`{"end_of_turn":{` + times + `,"default":"a","presets":{"a":1100},"frontier":[` + point(`[{"from_ms":0,"bar":0.9},{"from_ms":0,"bar":0.2}]`) + `]}}`,
		// Presets of bars, as servers published them before the frontier.
		`{"end_of_turn":{` + times + `,"default":"a","presets":{"a":{"bars":[{"from_ms":0,"bar":0.9}]}}}}`,
		`{"end_of_turn":{` + times + `,"first_bar":0.7,"later_bar":0.2}}`,
	} {
		s.server.info = info
		_, err = s.client.Policy(context.Background())
		s.ErrorIs(err, ErrUnsupported, info)
	}
}

// The words alone need no agent and no policy, and name the callers' languages. The audio goes
// in a part of its own, as a decision's does: a gateway may change a raw body's type.
func (s *ClientSuite) TestTheWordsAloneAreAskedForAlone() {
	s.server.words = "What time is the last ferry?"
	got, err := s.client.Transcribe(context.Background(), some, []string{"en", "pt-BR"})
	s.Require().NoError(err)
	s.Equal("What time is the last ferry?", got.Text)
	s.Equal("English", got.Language, "the language the server heard, by its name for it")
	s.Equal([]string{"en,pt-BR"}, s.server.languages)
	s.Equal([][]byte{some.Data}, s.server.audio)
	s.Equal([]string{pcmType}, s.server.types)
	s.Empty(s.server.registered)
	s.Zero(s.server.infos)
}

// The body carries no more than a prompt can show: the reply's last 800 characters, and the
// lines of the last six turns, a run of one side's lines being one turn.
func (s *ClientSuite) TestTheBodyCarriesWhatAPromptCanShow() {
	var lines []Line
	for i := range 9 {
		lines = append(lines, Line{Agent: i%2 == 1, Text: fmt.Sprintf("turn %d", i)})
	}
	lines = append(lines[:6], append([]Line{{Agent: true, Text: "turn 5, more"}}, lines[6:]...)...)
	lines[8].Text = "early " + strings.Repeat("late ", maxLineWords)
	_, err := s.client.Decide(context.Background(), Agent{}, Turn{Speaking: true, Said: "x" + strings.Repeat("é", maxSaid), Conversation: lines}, some)
	s.Require().NoError(err)
	got := s.server.turns[0]
	s.Equal(strings.Repeat("é", maxSaid), got.Said)
	want := slices.Clone(lines[3:])
	want[5].Text = strings.TrimSpace(strings.Repeat("late ", maxLineWords))
	s.Equal(want, got.Conversation)
	s.Equal("early "+strings.Repeat("late ", maxLineWords), lines[8].Text, "the caller's lines are theirs")
}

func (s *ClientSuite) TestNoAudioIsNotAsked() {
	_, err := s.client.Decide(context.Background(), booking, Turn{}, harness.Audio{})
	s.Require().Error(err)
	s.Zero(s.server.requests())
}

// The bar is the last of the bars to have begun by the check's time into the pause,
// raised on the turn's first pause.
func TestTheBarIsTheOneBegunByTheCheck(t *testing.T) {
	var info struct {
		EndOfTurn Policy `json:"end_of_turn"`
	}
	require.NoError(t, json.Unmarshal([]byte(testInfo), &info))
	p := info.EndOfTurn
	for _, c := range []struct {
		paused time.Duration
		want   float64
	}{{200 * time.Millisecond, 0.995}, {800 * time.Millisecond, 0.995}, {1099 * time.Millisecond, 0.995},
		{1100 * time.Millisecond, 0.20}, {2 * time.Second, 0.20}} {
		require.InDelta(t, c.want, p.Bar(p.DefaultEnding().Bars, c.paused, false), 1e-9, "%v into the pause", c.paused)
	}
	require.InDelta(t, 0.292, p.Bar(p.DefaultEnding().Bars, 1100*time.Millisecond, true), 1e-3)
	require.InDelta(t, 0.997, p.Bar(p.DefaultEnding().Bars, 800*time.Millisecond, true), 1e-3)
}

// HearerSuite reads the model's answers as the flow controller's rulings.
type HearerSuite struct {
	serverSuite
	hearer *Hearer
}

func TestHearerSuite(t *testing.T) {
	suite.Run(t, new(HearerSuite))
}

func (s *HearerSuite) SetupTest() {
	s.serverSuite.SetupTest()
	s.hearer = NewHearer(s.client, EndOfTurn{}, slog.New(slog.DiscardHandler))
}

func (s *HearerSuite) answering(pEOT float64, ruling string) {
	s.server.decision = fmt.Sprintf(`{"answer":"wait","answer_probability":%v,"p_eot":%v,"options":[],"decide_ms":1,"ruling":%s}`, 1-pEOT, pEOT, ruling)
}

// p_eot of 0.6 doesn't end a turn at a check 800 ms into the pause, where the bar is 0.995,
// and ends it at one 1.1 s in, where the bar is 0.2, as a ruling the server read past wait.
func (s *HearerSuite) TestATurnHasEndedWhenPEndReachesTheBar() {
	s.answering(0.6, `{"disposition":"respond","floor":"continue","tool":"book"}`)
	first, err := s.hearer.Check(context.Background(), harness.FlowTurn{Audio: some, Waited: true, Paused: 800 * time.Millisecond})
	s.Require().NoError(err)
	s.False(first.Ended)
	s.InDelta(0.995, first.Bar, 1e-9)
	s.Equal("preset:safe", first.Ending.Name)
	s.Equal(harness.Decided{Disposition: harness.Wait, Floor: harness.Continue, Again: 300 * time.Millisecond}, first.Ruling)

	later, err := s.hearer.Check(context.Background(), harness.FlowTurn{Audio: some, Waited: true, Paused: 1100 * time.Millisecond})
	s.Require().NoError(err)
	s.True(later.Ended)
	s.Equal(harness.Decided{Disposition: harness.Respond, Floor: harness.Continue}, later.Ruling)
	s.Equal("book", later.Answer.Ruling.Tool, "the tool wanted is the answer's, for the log")
}

func (s *HearerSuite) TestATurnHasEndedAtTheCapWhateverTheModelSays() {
	s.answering(0.01, `{"disposition":"respond","floor":"continue"}`)
	got, err := s.hearer.Check(context.Background(), harness.FlowTurn{Audio: some, Waited: true, Paused: 2500 * time.Millisecond})
	s.Require().NoError(err)
	s.Equal(harness.Respond, got.Ruling.Disposition)
}

// Over the agent there is no wait to weigh, and the ruling is the model's.
func (s *HearerSuite) TestOverTheAgentTheRulingIsTheModels() {
	s.server.decision = `{"answer":"stop","answer_probability":0.8,"options":[],"decide_ms":1,"ruling":{"disposition":"wait","floor":"stop"}}`
	got, err := s.hearer.Hear(context.Background(), harness.FlowTurn{Audio: some, Speaking: true, Unfinished: true, Reply: "Your crossing is"})
	s.Require().NoError(err)
	s.Equal(harness.Decided{Disposition: harness.Wait, Floor: harness.Stop}, got)
	s.Equal(Turn{Speaker: caller, Speaking: true, Said: "Your crossing is", Unfinished: true}, s.server.turns[0])
}

// A participant's name never reaches the model, the agent's reply only matters over it, and
// the conversation is the caller's and the agent's words.
func (s *HearerSuite) TestATurnIsSaidInTheModelsTerms() {
	history := []llm.Message{{Role: llm.System, Content: "x"}, {Role: llm.User, Content: "hi"}, {Role: llm.Assistant, Content: "hello"}, {Role: llm.User, Content: " "}}
	_, err := s.hearer.Hear(context.Background(), harness.FlowTurn{Audio: some, Participant: "Alice", AnotherVoice: true, Reply: "hello", Unfinished: true, History: history})
	s.Require().NoError(err)
	s.Equal(Turn{Speaker: anotherVoice, Conversation: []Line{{Text: "hi"}, {Agent: true, Text: "hello"}}}, s.server.turns[0])
}

// Over the agent the model is told what it was trained on (the rows of floor's testdata):
// the conversation before the reply being spoken, which is not a line of it but what the agent
// "has so far said", and that as far as the caller has heard it. The conversation is what was
// said on the call: a turn typed or handed in by the app ("greet the user"), and a tool's
// result, are no caller's words.
func (s *HearerSuite) TestOverTheAgentTheModelIsToldWhatWasSaidAndHeard() {
	s.server.decision = `{"answer":"continue","answer_probability":0.8,"options":[],"decide_ms":1,"ruling":{"disposition":"ignore","floor":"continue"}}`
	reply := "Tomorrow in Lisbon it will be sunny and warm, with a high of twenty four degrees."
	history := []llm.Message{
		{Role: llm.User, Content: "greet the user in one short sentence", Typed: true},
		{Role: llm.Assistant, Content: "Hello, how can I help you today?"},
		{Role: llm.User, Content: "What's the weather like in Lisbon tomorrow?"},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "get_weather"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"forecast":"sunny","high":24}`},
		{Role: llm.Assistant, Content: reply + " "},
	}
	_, err := s.hearer.Hear(context.Background(), harness.FlowTurn{Audio: some, Speaking: true, Unfinished: true, History: history,
		Reply: reply, Heard: "Tomorrow in Lisbon it will be", Measured: true})
	s.Require().NoError(err)
	s.Equal(Turn{Speaker: caller, Speaking: true, Unfinished: true, Said: "Tomorrow in Lisbon it will be", Conversation: []Line{
		{Agent: true, Text: "Hello, how can I help you today?"},
		{Text: "What's the weather like in Lisbon tomorrow?"},
	}}, s.server.turns[0])

	_, err = s.hearer.Hear(context.Background(), harness.FlowTurn{Audio: some, Speaking: true, Unfinished: true, History: history, Reply: reply})
	s.Require().NoError(err)
	s.Equal(reply, s.server.turns[1].Said, "what the caller heard unknown, the reply stands for it")
}

// The reply being said is no line of the conversation before it wherever the router keeps it:
// a reply that called a tool has the tool's result after it, which is no line either, and the
// reply was told twice, as a line of the agent's and as what it has so far said.
func (s *HearerSuite) TestTheReplyIsToldOnceWhenATurnFollowsItInTheHistory() {
	s.server.decision = `{"answer":"continue","answer_probability":0.8,"options":[],"decide_ms":1,"ruling":{"disposition":"ignore","floor":"continue"}}`
	reply := "Let me look that up for you."
	history := []llm.Message{
		{Role: llm.User, Content: "Where is my order?"},
		{Role: llm.Assistant, Content: reply, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "lookup_order"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"status":"shipped"}`},
	}
	_, err := s.hearer.Hear(context.Background(), harness.FlowTurn{Audio: some, Speaking: true, Unfinished: true, History: history,
		Reply: reply, Heard: "Let me look", Measured: true})
	s.Require().NoError(err)
	s.Equal(Turn{Speaker: caller, Speaking: true, Unfinished: true, Said: "Let me look", Conversation: []Line{
		{Text: "Where is my order?"},
	}}, s.server.turns[0])
}

// A turn the agent starts after its own line, such as the answer after a tool's result, has
// said and written nothing yet: the line before it was said whole and is the conversation's,
// and the agent has so far said nothing of the turn. Taken for the reply being said, that line
// was dropped from the conversation and the model told the agent had never said it.
func (s *HearerSuite) TestATurnThatHasSaidNothingYetLeavesTheLineBeforeIt() {
	s.server.decision = `{"answer":"continue","answer_probability":0.8,"options":[],"decide_ms":1,"ruling":{"disposition":"ignore","floor":"continue"}}`
	line := "Let me look that up for you."
	history := []llm.Message{
		{Role: llm.User, Content: "Where is my order?"},
		{Role: llm.Assistant, Content: line, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "lookup_order"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"status":"shipped"}`},
	}
	_, err := s.hearer.Hear(context.Background(), harness.FlowTurn{Audio: some, Speaking: true, Unfinished: true, History: history,
		Reply: line, Measured: true, Unsaid: true})
	s.Require().NoError(err)
	s.Equal(Turn{Speaker: caller, Speaking: true, Unfinished: true, Conversation: []Line{
		{Text: "Where is my order?"},
		{Agent: true, Text: line},
	}}, s.server.turns[0])
}

// TranscriberSuite runs the transcriber on frames of voice, of noise and of quiet, asking
// about no pauses: the words alone, which is how a session ends up once the server has said
// it can't rule on them, and what any pause falls to when a check of it fails.
type TranscriberSuite struct{ transcribing }

// transcribing is a transcriber of a fake server, and the frames it is sent.
type transcribing struct {
	serverSuite
	transcriber *Transcriber
	// wordsAlone opens each transcriber asking about no pauses (AskOnPauses(nil)), interims
	// has it write the words so far while the caller talks, which the suites that count each
	// request leave off, and languages are the callers' it is opened for.
	wordsAlone bool
	interims   bool
	languages  []string
	// rulesAlone opens each as a floor that rules alone (NewFloor), whose words another
	// transcriber writes, asking it for its final with finalize.
	rulesAlone bool
	finalize   func() (bool, error)
	// phase and at keep the voice going from one frame to the next, and noise the line's
	// noise, floor loud.
	phase float64
	at    int
	noise uint32
	floor float64
}

func TestTranscriberSuite(t *testing.T) {
	suite.Run(t, new(TranscriberSuite))
}

func (s *TranscriberSuite) SetupTest() {
	s.wordsAlone, s.interims = true, false
	s.transcribing.SetupTest()
}

func (s *transcribing) SetupTest() {
	s.serverSuite.SetupTest()
	s.server.words = "Two tickets to Bilbao, please."
	s.client.format = Opus
	s.languages = []string{"en", "pt-BR"}
	s.floor = 16 // about -66 dBFS
	// The voice and the noise start over, so that a test hears the same line whatever ran
	// before it: over noise near voiceLevel, which frames the detector takes for speech
	// depends on where the voice's pitch is.
	s.phase, s.at, s.noise = 0, 0, 0
	s.open()
}

// open starts the transcriber, and sends the speech detector's warm-up, as a call's first
// moments.
func (s *transcribing) open() { s.openReading(time.Now) }

// reopen closes the transcriber and opens another, reading the clock now.
func (s *transcribing) reopen(now func() time.Time) {
	s.Require().NoError(s.transcriber.Close())
	s.openReading(now)
}

func (s *transcribing) openReading(now func() time.Time) {
	if s.rulesAlone {
		s.transcriber = NewFloor(s.client, s.languages, s.finalize, slog.New(slog.DiscardHandler))
	} else {
		s.transcriber = NewTranscriber(s.client, "", s.languages, slog.New(slog.DiscardHandler))
		s.transcriber.interims = s.interims
	}
	s.transcriber.now = now
	if s.wordsAlone {
		s.transcriber.AskOnPauses(nil)
	}
	s.Require().NoError(s.transcriber.Start(context.Background()))
	transcriber := s.transcriber
	s.T().Cleanup(func() { s.Require().NoError(transcriber.Close()) })
	s.say(300, false)
}

// pauseMs is long enough to have the words written: the detector's window ends 10 ms before
// the newest sample, so a pause is heard a frame late.
const pauseMs = int((writeAfter + 2*frame) / time.Millisecond)

// requests are the server's transcription requests, by how much audio each carried.
func (s *transcribing) requests() []int {
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	var out []int
	for _, body := range s.server.audio {
		out = append(out, decoded(s.T(), body))
	}
	return out
}

// say sends ms of 20 ms frames of the line's noise, voiced or quiet.
func (s *transcribing) say(ms int, voiced bool) {
	for range ms / 20 {
		frame := make([]int16, frameSamples)
		for i := range frame {
			s.noise = s.noise*1664525 + 1013904223
			frame[i] = int16((float64(s.noise>>16)/65535*2 - 1) * s.floor)
		}
		if voiced {
			// The transcription service's voiced fixture: a wandering pitch, its harmonics
			// and a syllable's rhythm, which the speech detector scores as speech.
			for i := range frame {
				second := float64(s.at) / SampleRate
				pitch := 150 + 30*math.Sin(2*math.Pi*0.5*second) + 12*math.Sin(2*math.Pi*1.3*second)
				s.phase += 2 * math.Pi * pitch / SampleRate
				value := 0.6*math.Sin(s.phase) + 0.25*math.Sin(2*s.phase) + 0.15*math.Sin(3*s.phase)
				envelope := 0.5 + 0.5*math.Pow(0.5+0.5*math.Sin(2*math.Pi*4.2*second), 1.5)
				frame[i] += int16(24000 * value * envelope)
				s.at++
			}
		}
		s.send(frame)
	}
}

func (s *transcribing) send(frame []int16) {
	s.Require().NoError(s.transcriber.ProcessAudio(audio.PcmData{Samples: frame, SampleRate: SampleRate, Channels: 1}, stt.Participant{ID: "p"}))
}

// transcribed is how many requests for words alone (POST /v1/transcribe) the server has had.
func (s *transcribing) transcribed() int {
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	return len(s.server.audio) - len(s.server.turns)
}

// answered waits for the check asked to have answered.
func (s *transcribing) answered() {
	s.Eventually(func() bool {
		s.transcriber.mu.Lock()
		defer s.transcriber.mu.Unlock()
		return !s.transcriber.pause.inFlight
	}, 5*time.Second, time.Millisecond)
}

// next is the next event of type E.
func next[E stt.Event](s *transcribing) E {
	s.T().Helper()
	for {
		select {
		case event := <-s.transcriber.Events():
			if typed, ok := event.(E); ok {
				return typed
			}
		case <-time.After(5 * time.Second):
			s.FailNow("no event")
		}
	}
}

// none says no transcript comes.
func (s *transcribing) none() {
	s.T().Helper()
	for {
		select {
		case event := <-s.transcriber.Events():
			_, ok := event.(stt.Transcript)
			s.Require().False(ok, "a transcript: %+v", event)
		case <-time.After(100 * time.Millisecond):
			return
		}
	}
}

// An utterance's words are written at its pauses, the whole of it each time, and it ends
// at a pause of endAfter with one final on the words last written. The callers' languages
// go with each request.
func (s *TranscriberSuite) TestAnUtteranceIsWrittenAtItsPausesAndEndsOnce() {
	s.say(1700, false)
	s.say(1000, true)
	s.say(180, false)
	s.Empty(s.requests())
	s.say(pauseMs-180, false)
	said := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeReplacement, said.Mode)
	s.Equal(int64(1), said.Utterance)
	s.Equal("Two tickets to Bilbao, please.", said.Text)
	s.InDelta(float64(300+1000+writeAfter.Milliseconds()), said.AudioDurationMs, 40, "300 ms before the voice, then the voice and the pause")
	s.Equal("audio/ogg", s.server.types[0])
	s.InDelta(said.AudioDurationMs*SampleRate/1000-104, float64(s.requests()[0]), 1, "less the pre-skip")
	s.Equal("en,pt-BR", s.server.languages[0])

	s.say(500, true)
	s.say(pauseMs, false)
	more := next[stt.Transcript](&s.transcribing)
	s.Equal(int64(1), more.Utterance)
	s.InDelta(said.AudioDurationMs+float64(500+writeAfter.Milliseconds()), more.AudioDurationMs, 60)

	s.say(int(endAfter/time.Millisecond), false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal(int64(1), end.Utterance)
	s.Len(s.requests(), 2, "a final with no speech since the last words asks for none")
}

// An interim of the next utterance waits for the last one's final, which a slow server is
// still writing: asked meanwhile, it would go out first and the final behind it would be
// dropped as words of an utterance that is over, or arrive after the next utterance's words.
func (s *TranscriberSuite) TestAnInterimOfTheNextUtteranceWaitsForTheLastFinal() {
	s.interims = true
	s.reopen(time.Now)
	s.say(1000, true)
	first := next[stt.Transcript](&s.transcribing)
	s.True(first.Interim)
	s.Equal(int64(1), first.Utterance)

	gate := make(chan struct{})
	s.T().Cleanup(func() {
		select {
		case <-gate:
		default:
			close(gate)
		}
	})
	s.server.mu.Lock()
	s.server.transcribeGate, s.server.holdOne = gate, true
	s.server.mu.Unlock()
	s.say(int(endAfter/time.Millisecond)+40, false)
	s.Eventually(func() bool {
		s.server.mu.Lock()
		defer s.server.mu.Unlock()
		return s.server.held == 1
	}, 5*time.Second, time.Millisecond, "the pause's words, which the final is asked for on, are held")
	s.say(1200, true)
	s.none()
	s.Equal(2, s.transcribed(), "no interim of the next utterance while the last one's final is still to come")

	close(gate)
	written := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeReplacement, written.Mode, "the pause's words")
	s.False(written.Interim)
	s.Equal(int64(1), written.Utterance)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal(int64(1), end.Utterance)
	s.Equal("Two tickets to Bilbao, please.", end.Text)
	s.server.mu.Lock()
	s.server.words = "Is the ferry running tomorrow?"
	s.server.mu.Unlock()
	s.say(40, true)
	again := next[stt.Transcript](&s.transcribing)
	s.True(again.Interim)
	s.Equal(int64(2), again.Utterance)
	s.Equal("Is the ferry running tomorrow?", again.Text)
}

// Silence after a final writes nothing more: the utterance is over, and the next waits for
// speech.
func (s *TranscriberSuite) TestSilenceAfterAFinalWritesNothing() {
	s.say(1000, true)
	s.say(int(endAfter/time.Millisecond)+100, false)
	s.Equal(stt.ModeReplacement, next[stt.Transcript](&s.transcribing).Mode)
	s.Equal(stt.ModeFinal, next[stt.Transcript](&s.transcribing).Mode)
	s.say(15000, false)
	s.none()
	s.Len(s.requests(), 1)
}

// A line whose noise hovers about -48 dBFS, as loud as quiet speech, writes nothing of its
// own, however long the silence: only speech asks for words. Taken by its level alone, as
// the transcriber once did, the noise asked for the words again at every crossing.
func (s *TranscriberSuite) TestANoisyLineWritesNothingOfItsOwn() {
	s.Require().NoError(s.transcriber.Close())
	s.floor = 230 // a frame's root mean square about voiceLevel
	s.open()
	s.say(1000, true)
	// Over noise as loud as voiceLevel the pause is heard once the detector's score has let
	// go of the voice, about 200 ms later.
	s.say(int(endAfter/time.Millisecond)+400, false)
	s.Equal(stt.ModeReplacement, next[stt.Transcript](&s.transcribing).Mode)
	s.Equal(stt.ModeFinal, next[stt.Transcript](&s.transcribing).Mode)
	s.say(15000, false)
	s.none()
	s.Len(s.requests(), 1)
}

// Audio is never written twice: once an utterance has ended, the next one, however soon it
// begins, carries none of its audio.
func (s *TranscriberSuite) TestNoAudioIsWrittenTwice() {
	s.say(1000, true)
	s.say(int(endAfter/time.Millisecond)+100, false)
	next[stt.Transcript](&s.transcribing)
	s.True(next[stt.Transcript](&s.transcribing).Final())
	s.say(600, true)
	s.say(pauseMs, false)
	again := next[stt.Transcript](&s.transcribing)
	s.Equal(int64(2), again.Utterance)
	s.InDelta(float64(100+600+writeAfter.Milliseconds()), again.AudioDurationMs, 40, "the 100 ms since the final, the new speech and its pause")
	s.Len(s.requests(), 2, "one pass over each utterance")
}

// A caller who says more after a final starts a new utterance: a final is the last word on
// its own, and nothing about its audio follows it.
func (s *TranscriberSuite) TestSpeechAfterAFinalIsANewUtterance() {
	s.say(1000, true)
	s.say(int(endAfter/time.Millisecond)+100, false)
	next[stt.Transcript](&s.transcribing)
	first := next[stt.Transcript](&s.transcribing)
	s.True(first.Final())
	s.say(800, true)
	s.say(int(endAfter/time.Millisecond)+100, false)
	for _, mode := range []stt.Mode{stt.ModeReplacement, stt.ModeFinal} {
		said := next[stt.Transcript](&s.transcribing)
		s.Equal(mode, said.Mode)
		s.Equal(int64(2), said.Utterance)
	}
	s.none()
}

// The transcriber asks for the words alone: no agent, no decision and no policy, so a
// server that serves no policy transcribes as well.
func (s *TranscriberSuite) TestItNeedsNoAgentNorPolicy() {
	s.server.mu.Lock()
	s.server.info = ""
	s.server.mu.Unlock()
	s.say(1000, true)
	s.say(pauseMs, false)
	s.Equal("Two tickets to Bilbao, please.", next[stt.Transcript](&s.transcribing).Text)
	s.Empty(s.server.registered)
	s.Empty(s.server.turns)
	s.Zero(s.server.infos)
}

// Empty words are nothing heard: what the server dropped as noise, or what had no words.
func (s *TranscriberSuite) TestEmptyWordsAreNothingHeard() {
	s.server.mu.Lock()
	s.server.words = ""
	s.server.mu.Unlock()
	s.say(200, true)
	s.say(int(endAfter/time.Millisecond)+100, false)
	s.none()
	s.Len(s.requests(), 1, "the pause's words, and none at the end, which has nothing more")
}

// Words that come back empty at a later pause don't take back the words before them, and
// the utterance ends on those.
func (s *TranscriberSuite) TestAnUtteranceEndsOnItsLastWords() {
	s.say(1000, true)
	s.say(pauseMs, false)
	s.Equal("Two tickets to Bilbao, please.", next[stt.Transcript](&s.transcribing).Text)
	s.server.mu.Lock()
	s.server.words = ""
	s.server.mu.Unlock()
	s.say(300, true)
	s.say(int(endAfter/time.Millisecond)+100, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal("Two tickets to Bilbao, please.", end.Text)
}

// A server that is busy, down or restarting costs a request, not the session: the words it
// didn't write are asked for again, of the whole utterance, at its end.
func (s *TranscriberSuite) TestABusyServerCostsARequestNotTheSession() {
	for _, status := range []int{http.StatusServiceUnavailable, http.StatusInternalServerError, http.StatusNotFound, http.StatusTooManyRequests} {
		s.Run(http.StatusText(status), func() {
			s.SetupTest()
			failed := testutil.ToFloat64(transcribeFailures.WithLabelValues(strconv.Itoa(status)))
			s.server.mu.Lock()
			s.server.failing, s.server.failWith = 1, status
			s.server.mu.Unlock()
			s.say(1000, true)
			s.say(pauseMs, false)
			s.say(int(endAfter/time.Millisecond), false)
			end := next[stt.Transcript](&s.transcribing)
			s.Equal(stt.ModeFinal, end.Mode, "no error, and the utterance ends on its words")
			s.Equal("Two tickets to Bilbao, please.", end.Text)
			s.Len(s.requests(), 1, "the failed request carried nothing back, and the end asked again")
			s.InDelta(float64(300+1000+int(endAfter/time.Millisecond)), float64(s.requests()[0])*1000/SampleRate, 80, "of the whole utterance")
			s.Equal(failed+1, testutil.ToFloat64(transcribeFailures.WithLabelValues(strconv.Itoa(status))))
		})
	}
}

// A pause's words that fail after the utterance's end was asked for are asked for again at
// the end, as they are when they fail before it: a server slow to fail, or a session that
// hears faster than the server answers, costs a request, not the utterance.
func (s *TranscriberSuite) TestWordsThatFailAfterTheEndIsAskedForAreAskedForAtTheEnd() {
	gate := make(chan struct{})
	s.T().Cleanup(func() {
		select {
		case <-gate:
		default:
			close(gate)
		}
	})
	s.server.mu.Lock()
	s.server.transcribeGate, s.server.holdOne, s.server.failHeld, s.server.failWith = gate, true, true, http.StatusServiceUnavailable
	s.server.mu.Unlock()
	s.say(1000, true)
	s.say(pauseMs, false)
	s.Eventually(func() bool {
		s.server.mu.Lock()
		defer s.server.mu.Unlock()
		return s.server.held == 1
	}, 5*time.Second, time.Millisecond, "the pause's words are out")
	s.say(int(endAfter/time.Millisecond), false)
	close(gate)

	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode, "the utterance ends on its words")
	s.Equal("Two tickets to Bilbao, please.", end.Text)
	asked := s.requests()
	s.Require().Len(asked, 2, "the pause's words, which failed, and the end's")
	s.InDelta(float64(300+1000+int(endAfter/time.Millisecond)), float64(asked[1])*1000/SampleRate, 80, "of the whole utterance")
}

// Words written at a pause are what the utterance ends on, with nothing asked again.
func (s *TranscriberSuite) TestAnEndWithNothingNewEndsOnTheWordsWritten() {
	s.say(1000, true)
	s.say(int(endAfter/time.Millisecond), false)
	written := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeReplacement, written.Mode)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal("Two tickets to Bilbao, please.", end.Text)
	s.Len(s.requests(), 1, "the end had nothing new to write")
}

// A server that isn't there, or doesn't answer in time, is asked again as well.
func (s *TranscriberSuite) TestAServerThatIsGoneCostsARequestNotTheSession() {
	failed := testutil.ToFloat64(transcribeFailures.WithLabelValues("connection"))
	s.client.url = "http://127.0.0.1:1"
	s.say(1000, true)
	s.say(pauseMs+100, false)
	s.Eventually(func() bool { return testutil.ToFloat64(transcribeFailures.WithLabelValues("connection")) > failed }, 5*time.Second, 10*time.Millisecond)
	s.none()
}

// noFatal says no fatal error comes.
func (s *transcribing) noFatal() {
	s.T().Helper()
	for {
		select {
		case event := <-s.transcriber.Events():
			failure, ok := event.(stt.Error)
			s.Require().False(ok && failure.Fatal, "a fatal error: %+v", event)
		case <-time.After(100 * time.Millisecond):
			return
		}
	}
}

// skewed is a clock running skew ahead of the wall.
func skewed(skew *atomic.Int64) func() time.Time {
	return func() time.Time { return time.Now().Add(time.Duration(skew.Load())) }
}

// A server that answers none of three requests for words in a row, the last a window after
// the first, is down for the call rather than restarting: the session is fatal, and the call
// goes to the next transcriber rather than on without words. A connection refused counts,
// as a 5xx and a timeout would.
func (s *TranscriberSuite) TestAServerDownForTheCallIsFatal() {
	var skew atomic.Int64
	s.reopen(skewed(&skew))
	s.client.url = "http://127.0.0.1:1"
	s.say(1000, true)
	s.say(pauseMs, false)
	s.say(int(endAfter/time.Millisecond), false)
	s.noFatal()

	skew.Store(int64(failoverWindow))
	s.say(1000, true)
	s.say(pauseMs, false)
	failure := next[stt.Error](&s.transcribing)
	s.True(failure.Fatal)
	s.Equal("transcribe", failure.Context)
	s.ErrorContains(failure.Err, "none of 3 requests for words")
}

// A server shedding load with 503 or 429 is up, however long it does: those requests are
// asked again, and never give the session up. Words answered start the count over, so a
// server that fails now and then is never down for the call.
func (s *TranscriberSuite) TestABusyServerIsUpAndWordsAnsweredStartTheCountOver() {
	var skew atomic.Int64
	s.reopen(skewed(&skew))
	// failing has the server answer status to the next utterances' requests, two each: the
	// pause's and the end's, which asks again what the pause was refused. The clock moves a
	// window on once the transcriber has counted both, not once they are asked for: it
	// weighs a failure when it reads its answer, which on a busy machine is later.
	failing := func(status, utterances int) {
		s.server.mu.Lock()
		s.server.failing, s.server.failWith = 2*utterances, status
		s.server.mu.Unlock()
		counted := transcribeFailures.WithLabelValues(strconv.Itoa(status))
		for range utterances {
			before := testutil.ToFloat64(counted)
			s.say(1000, true)
			s.say(pauseMs, false)
			s.say(int(endAfter/time.Millisecond), false)
			s.Eventually(func() bool { return testutil.ToFloat64(counted) == before+2 }, 5*time.Second, time.Millisecond,
				"the pause's and the end's words, refused")
			skew.Add(int64(failoverWindow))
		}
	}
	failing(http.StatusServiceUnavailable, 2)
	failing(http.StatusTooManyRequests, 1)
	s.noFatal()

	// Two not answered, then words, then two more: never three in a row.
	failing(http.StatusInternalServerError, 1)
	s.say(1000, true)
	s.say(pauseMs, false)
	s.Equal(stt.ModeReplacement, next[stt.Transcript](&s.transcribing).Mode)
	failing(http.StatusInternalServerError, 1)
	s.noFatal()

	// The third since the words were answered, a window on: down.
	s.server.mu.Lock()
	s.server.failing, s.server.failWith = 1, http.StatusInternalServerError
	s.server.mu.Unlock()
	s.say(1000, true)
	s.say(pauseMs, false)
	s.True(next[stt.Error](&s.transcribing).Fatal)
}

// A server the router is misconfigured for will never answer, and hands the call to the
// next transcriber.
func (s *TranscriberSuite) TestAMisconfiguredServerIsFatal() {
	for _, status := range []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusUnsupportedMediaType, http.StatusNotImplemented} {
		s.Run(http.StatusText(status), func() {
			s.SetupTest()
			s.server.mu.Lock()
			s.server.failing, s.server.failWith = 1, status
			s.server.mu.Unlock()
			s.say(1000, true)
			s.say(pauseMs, false)
			s.True(next[stt.Error](&s.transcribing).Fatal)
		})
	}
}

// An utterance whose final was dropped, for a server too far behind, leaves its words to
// nothing after it: the next utterance, with no words of its own, writes nothing.
func (s *TranscriberSuite) TestADroppedFinalLeavesItsWordsBehind() {
	stream, err := s.client.NewStream(time.Second)
	s.Require().NoError(err)
	s.Require().NoError(stream.Write(speech(SampleRate)))
	heard := stream.Last(time.Second)
	participant := stt.Participant{ID: "p"}
	s.transcriber.requests <- request{participant: participant, utterance: 1, audio: heard}
	s.Equal("Two tickets to Bilbao, please.", next[stt.Transcript](&s.transcribing).Text)
	s.server.mu.Lock()
	s.server.words = ""
	s.server.mu.Unlock()
	// Utterance 1's final never came: the queue was full.
	s.transcriber.requests <- request{participant: participant, utterance: 2, audio: heard}
	s.transcriber.requests <- request{participant: participant, utterance: 2, final: true}
	s.none()
}

// speech is a voice-like test signal: a falling pitch with its harmonics, swelling and
// fading as syllables do.
func speech(n int) []int16 {
	out := make([]int16, n)
	phase := 0.0
	for i := range out {
		t := float64(i) / SampleRate
		pitch := 140 + 40*math.Sin(2*math.Pi*0.7*t)
		phase += 2 * math.Pi * pitch / SampleRate
		var v float64
		for h := 1.0; h <= 12; h++ {
			v += math.Sin(h*phase) / h
		}
		out[i] = int16(5000 * v * (0.6 + 0.4*math.Sin(2*math.Pi*3*t)))
	}
	return out
}

// demux reads an Ogg Opus body back: its packets and granule positions, and its pre-skip.
func demux(t *testing.T, body []byte) ([][]byte, []uint64, int) {
	t.Helper()
	r, err := ogg.NewReader(bytes.NewReader(body))
	require.NoError(t, err)
	require.EqualValues(t, 1, r.Channels())
	var packets [][]byte
	var granules []uint64
	for {
		packet, granule, err := r.ReadPacket()
		if errors.Is(err, io.EOF) {
			break
		}
		require.NoError(t, err)
		packets, granules = append(packets, slices.Clone(packet)), append(granules, granule)
	}
	return packets, granules, int(r.PreSkip())
}

// decode is what a decoder that starts on body's first packet hears: the audio at 16 kHz,
// less the pre-skip, to the last page's granule position.
func decode(t *testing.T, body []byte) []int16 {
	t.Helper()
	packets, granules, preSkip := demux(t, body)
	decoder, err := gopus.NewDecoder(gopus.DefaultDecoderConfig(SampleRate, 1))
	require.NoError(t, err)
	var pcm []int16
	buf := make([]int16, 5760)
	for _, packet := range packets {
		n, err := decoder.DecodeInt16(packet, buf)
		require.NoError(t, err)
		pcm = append(pcm, buf[:n]...)
	}
	return pcm[preSkip/3 : granules[len(granules)-1]/3]
}

func decoded(t *testing.T, body []byte) int { return len(decode(t, body)) }

// What the stream writes is the packets it encoded, as each arrived, in pieces of whatever
// size, and they decode back to the audio written: the frames of its last 16 s, from the
// first frame's start.
func TestAStreamDecodesBackToWhatItWasWritten(t *testing.T) {
	stream, err := newStream(Opus, Window)
	require.NoError(t, err)
	written := speech(30 * SampleRate)
	for at, sizes := 0, []int{160, 333, 960, 320, 7}; at < len(written); {
		n := min(sizes[at%len(sizes)], len(written)-at)
		require.NoError(t, stream.Write(written[at:at+n]))
		at += n
	}
	frames := len(written) / frameSamples
	require.EqualValues(t, frames, stream.Frames())

	got := stream.Last(Window)
	require.Equal(t, "audio/ogg", got.Type)
	require.Equal(t, Window, got.Duration)
	packets, granules, preSkip := demux(t, got.Data)
	held := int(Window / frame)
	require.Len(t, packets, held)
	require.LessOrEqual(t, bytes.Count(got.Data, []byte("OggS")), 2+held/255+1, "packets laced many to a page")
	for i, packet := range packets {
		require.Equal(t, stream.frames[(stream.next+i)%held], packet, "packet %d", i)
	}
	require.EqualValues(t, 312, preSkip, "the encoder's lookahead, 6.5 ms, at 48 kHz")
	require.EqualValues(t, held*960, granules[len(granules)-1])

	heard := decode(t, got.Data)
	require.Len(t, heard, held*frameSamples-104)
	// The audio heard is the audio written, in time: an encoder codes a sound, not its
	// waveform, so the two are compared by their loudness frame by frame, and by where
	// they line up best. A decoder that starts mid-stream converges within a few frames.
	written = written[(frames-held)*frameSamples:]
	require.Greater(t, correlation(loudness(written[960:len(heard)]), loudness(heard[960:])), 0.95)
	lag, best := 0, math.Inf(-1)
	for shift := -frameSamples; shift <= frameSamples; shift++ {
		if c := correlation(toFloat(written[2*frameSamples+shift:][:SampleRate]), toFloat(heard[2*frameSamples:][:SampleRate])); c > best {
			lag, best = shift, c
		}
	}
	require.InDelta(t, 0, lag, 4, "the pre-skip lines the audio up with what was written")
}

// A packet of 255 bytes or more takes several lacing values, and a page holds at most 255.
func TestOggLacesEveryPacketWhole(t *testing.T) {
	var packets [][]byte
	for i, size := range append([]int{1, 254, 255, 256, 510, 600}, slices.Repeat([]int{3}, 300)...) {
		packets = append(packets, bytes.Repeat([]byte{byte(i)}, size))
	}
	got, granules, _ := demux(t, oggOpus(packets, 312))
	require.Equal(t, packets, got)
	require.EqualValues(t, len(packets)*960, granules[len(granules)-1])
}

// loudness is each frame's root mean square.
func loudness(pcm []int16) []float64 {
	var out []float64
	for f := range slices.Chunk(pcm, frameSamples) {
		var sum float64
		for _, v := range f {
			sum += float64(v) * float64(v)
		}
		out = append(out, math.Sqrt(sum/float64(len(f))))
	}
	return out
}

func toFloat(pcm []int16) []float64 {
	out := make([]float64, len(pcm))
	for i, v := range pcm {
		out[i] = float64(v)
	}
	return out
}

// correlation is Pearson's of two series of the same length.
func correlation(a, b []float64) float64 {
	n := float64(min(len(a), len(b)))
	var sa, sb, saa, sbb, sab float64
	for i := range int(n) {
		sa, sb = sa+a[i], sb+b[i]
		saa, sbb, sab = saa+a[i]*a[i], sbb+b[i]*b[i], sab+a[i]*b[i]
	}
	return (sab - sa*sb/n) / math.Sqrt((saa-sa*sa/n)*(sbb-sb*sb/n))
}

// Less than the window is what was heard, and nothing is nothing.
func TestAStreamSendsWhatItHas(t *testing.T) {
	stream, err := newStream(Opus, Window)
	require.NoError(t, err)
	require.Empty(t, stream.Last(Window).Data)
	require.NoError(t, stream.Write(speech(frameSamples*5/2)))
	got := stream.Last(Window)
	require.Equal(t, 2*frame, got.Duration, "half a frame waits for the rest")
	require.Equal(t, 2*frameSamples-104, decoded(t, got.Data))
	require.Equal(t, frame, stream.Last(frame).Duration)
}

// PCM is the frames' samples as they were written.
func TestAPCMStreamIsTheSamples(t *testing.T) {
	stream, err := newStream(PCM, time.Second)
	require.NoError(t, err)
	written := speech(3 * SampleRate)
	require.NoError(t, stream.Write(written))
	got := stream.Last(time.Second)
	require.Equal(t, pcmType, got.Type)
	require.Equal(t, audio.PcmData{Samples: written[2*SampleRate:], SampleRate: SampleRate, Channels: 1}.Bytes(), got.Data)
}

// Writing a frame allocates nothing once its slot in the ring has room for the packet.
func TestAStreamWritesWithoutAllocating(t *testing.T) {
	stream, err := newStream(Opus, time.Second)
	require.NoError(t, err)
	quiet := make([]int16, frameSamples)
	for range 2 * 50 {
		require.NoError(t, stream.Write(quiet))
	}
	require.Zero(t, testing.AllocsPerRun(100, func() { _ = stream.Write(quiet) }))
}

// benchSpeech is CHORUS_ASR_WAV's audio, a 16 kHz mono 16-bit WAV, or the test signal.
func benchSpeech(b *testing.B, n int) []int16 {
	path := os.Getenv("CHORUS_ASR_WAV")
	if path == "" {
		return speech(n)
	}
	raw, err := os.ReadFile(path)
	require.NoError(b, err)
	at := bytes.Index(raw, []byte("data"))
	pcm := audio.FromBytes(raw[at+8:at+8+int(binary.LittleEndian.Uint32(raw[at+4:]))], SampleRate, 1).Samples
	for len(pcm) < n {
		pcm = append(pcm, pcm...)
	}
	return pcm[:n]
}

// What a participant's audio costs as it arrives: one frame encoded.
func BenchmarkAFrame(b *testing.B) {
	for _, format := range []Format{Opus, PCM} {
		b.Run(string(format), func(b *testing.B) {
			pcm := benchSpeech(b, 60*SampleRate)
			stream, err := newStream(format, Window)
			require.NoError(b, err)
			b.ResetTimer()
			for i := range b.N {
				at := i % (len(pcm) / frameSamples) * frameSamples
				_ = stream.Write(pcm[at : at+frameSamples])
			}
		})
	}
}

// What a decision costs the router: the body of a full window.
func BenchmarkADecisionsBody(b *testing.B) {
	for _, format := range []Format{Opus, PCM} {
		b.Run(string(format), func(b *testing.B) {
			stream, err := newStream(format, Window)
			require.NoError(b, err)
			require.NoError(b, stream.Write(benchSpeech(b, int(Window/time.Second)*SampleRate)))
			turn := Turn{Speaker: caller, Conversation: []Line{{Agent: true, Text: "Weather line, how can I help?"}}}
			var size int
			b.ResetTimer()
			for range b.N {
				body, _, err := decisionBody(turn, stream.Last(Window), nil)
				require.NoError(b, err)
				size = len(body)
			}
			b.ReportMetric(float64(size), "bytes")
		})
	}
}
