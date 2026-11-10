// Package audiofloor asks the chorus-asr model about a call: one speech model that hears the
// caller's last 16 s and answers the question the flow controller answers from the words,
// says which of the agent's tools is wanted, and can write what was said. It is served by an
// chorus-asr server, which speaks this protocol:
//
//	GET  /v1/info             what the server serves: its prompt template, whether a prompt
//	                          with the floor free says the conversation, the most tools an
//	                          agent can have, the audio a decision hears, and end_of_turn,
//	                          the policy fitted to its model for ending a turn
//	POST /v1/agents           {"instructions","tools"}: registers an agent by its content
//	                          and answers {"agent": ID}. More tools than the model has
//	                          answers for is a 422
//	POST /v1/decide?agent=ID  a multipart body: a "turn" part, the moment of the call as
//	                          JSON, then an "audio" part, Ogg Opus or 16 kHz mono PCM. The
//	                          answer is every option's probability, p_eot = 1 - p(wait)
//	                          with the floor free, and the ruling to act on once the turn
//	                          has ended. With &transcript=true it is two JSON lines: the
//	                          decision, about the audio's last 16 s, as soon as it is made,
//	                          then the words of all of it
//	POST /v1/transcribe       a multipart body of one "audio" part, as a decision's, and
//	                          ?languages= the callers speak: what was said, empty when
//	                          nothing was or the server heard noise
//
// A server that says "inline_agents": true in GET /v1/info takes a decision's agent with it: a
// third part, "agent", after the audio, the JSON POST /v1/agents takes, so that any of several
// replicas behind one address can answer for an agent another was told. It goes last because a
// server that does not take it reads the turn and the audio and stops there. Every decision and
// transcription says in X-Timeout-Ms how many milliseconds the router will go on waiting for its
// answer, a decision's first line with ?transcript=true, from when it was sent, so that a server
// can drop what nobody waits for any more. A 503 or a 429 with Retry-After has the client send
// only what decides turns, the pauses' checks and their words, until it has passed.
//
// The server writes the model's prompt, because the model knows only the words it was
// trained on. The server says what to do with the words; whether a turn with the floor free
// has ended is the router's to say, from p_eot, the pause and the policy. A server started
// without a policy for its model answers 501 to the first three, and transcribes all the
// same.
package audiofloor

import (
	"bytes"
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

const (
	// SampleRate is what the server takes and the model hears.
	SampleRate = 16000
	// Window is what a decision hears.
	Window = 16 * time.Second
	// maxSaid, maxTurns and maxLineWords are what a prompt shows of the reply, of the
	// conversation and of each of its lines, so sending more only spends the request.
	maxSaid      = 800
	maxTurns     = 6
	maxLineWords = 40
	// maxTurnBytes is the most the server takes in a turn.
	maxTurnBytes = 64 << 10
	// maxAgents bounds the IDs a client remembers. A router's agents are few, and one
	// forgotten costs a registration.
	maxAgents = 1024
	// responseLimit bounds an answer, a small JSON line.
	responseLimit = 1 << 20
	// wordsTimeout bounds the words a decision writes as well, which on a CPU take seconds,
	// with as long again as the audio lasts.
	wordsTimeout = 8 * time.Second
	// TimeoutHeader is how many milliseconds the router will go on waiting for a decision's or
	// a transcription's answer, counted from when it sent the request: a whole number, at least
	// 1. It is relative, so that the clocks of the router and the server need not agree.
	TimeoutHeader = "X-Timeout-Ms"
	// maxBackOff bounds how long a Retry-After is heeded, whatever it says.
	maxBackOff = 30 * time.Second
	// idleConnTimeout is how long a kept-alive connection to the server may sit unused before
	// the client closes it: under the 60 s after which the server closes one itself (gophonic
	// cmd/qwen3asr-decide, idleTimeout), so that the server is never the one to close it as a
	// request goes out on it. maxIdleConns is what the router's default transport keeps a host
	// (cmd/router), so that requests that overlap each keep a connection of their own.
	idleConnTimeout = 50 * time.Second
	maxIdleConns    = 256
)

var (
	// ErrRefused is an agent the server won't ask about, for more tools than the model has
	// answers for.
	ErrRefused = errors.New("audiofloor: the server refused the agent")
	// ErrUnsupported is a server with no policy for its model, which asks about no agent.
	ErrUnsupported = errors.New("audiofloor: the server has no policy for its model")
	// ErrAgentUnknown is a decision the server answered 404 for an agent it was told about
	// moments before, asked again and answered 404 again: a replica behind the same address that
	// has not been told yet. The server is up, and the next request may reach one that knows.
	ErrAgentUnknown = errors.New("audiofloor: the replica answering does not know the agent, though the server was told it")
	// ErrBackingOff is a request the client did not send, one that decides no turn (a check
	// over the agent), while a Retry-After the server answered 503 or 429 with lasts.
	ErrBackingOff = errors.New("audiofloor: the server asked for time (Retry-After), so only the pauses are asked about")
)

// StatusError is a request the server answered with an error status.
type StatusError struct {
	Method, Path string
	Code         int
	Status       string
	// Message is the server's own word on it.
	Message string
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("audiofloor: %s %s: %s: %s", e.Method, e.Path, e.Status, e.Message)
}

// Misconfigured says err is the router's to fix, and asking again won't help: a server with
// no policy, or a request it refused as malformed or unauthorised. A server that is busy,
// down, restarting or slow (5xx, 404, 408, 429, a timeout, a reset connection) may answer
// the next one.
func Misconfigured(err error) bool {
	if errors.Is(err, ErrUnsupported) {
		return true
	}
	var status *StatusError
	if !errors.As(err, &status) {
		return false
	}
	switch status.Code {
	case http.StatusNotFound, http.StatusRequestTimeout, http.StatusTooManyRequests:
		return false
	}
	return status.Code >= 400 && status.Code < 500
}

// TranscriptError is a request for words the server answered without them, saying why: it
// heard noise, or every lane that writes words was busy. The server is up, whatever it said.
type TranscriptError struct {
	// Request is which request: transcribe, or a decision's words.
	Request string
	Message string
}

func (e *TranscriptError) Error() string { return "audiofloor: " + e.Request + ": " + e.Message }

// Unreachable says the server did not answer err's request at all: the connection was
// refused, reset or dropped, the request timed out, or what answered was a 5xx, 404 or 408,
// as a proxy with nothing behind it answers. A server that answered 503 or 429 is up and
// shedding load, and one that answered without words (TranscriptError) or refused the
// request (Misconfigured) is up as well, as is one whose replica did not know an agent
// another was told (ErrAgentUnknown), and a request held back while it asked for time
// (ErrBackingOff) says nothing of it. A session that meets nothing else for a while has a
// server down for the call, rather than restarting.
func Unreachable(err error) bool {
	if Misconfigured(err) || errors.Is(err, ErrAgentUnknown) || errors.Is(err, ErrBackingOff) {
		return false
	}
	var transcript *TranscriptError
	if errors.As(err, &transcript) {
		return false
	}
	var status *StatusError
	if errors.As(err, &status) {
		return status.Code != http.StatusServiceUnavailable && status.Code != http.StatusTooManyRequests
	}
	return true
}

// The speakers a prompt can name. Training's rows name nobody else, so a participant's own
// name goes no further than this package.
const (
	caller       = "caller"
	anotherVoice = "another_voice"
)

// Policy is when a turn with the floor free has ended, fitted to the server's model. A
// pause is checked FirstCheckMs into it and every IntervalMs after. The frontier is the
// measured points, each bars and how many checks in a row must reach them: the turn has
// ended at the first check where p_eot reaches the bar of the last of a point's bars to have
// begun by then, raised by FirstPauseLogit in log-odds on the turn's first pause, with as
// many checks before it as its ChecksInARow asks. At CapMs of pause it has ended whatever
// the model says. Presets name points by their MaxWaitMs, and the server names its Default.
// An agent chooses a point by preset or by the most a caller may wait, or brings bars of its
// own (EndOfTurn).
type Policy struct {
	FirstCheckMs    int             `json:"first_check_ms"`
	FirstPauseLogit float64         `json:"first_pause_logit"`
	IntervalMs      int             `json:"interval_ms"`
	CapMs           int             `json:"cap_ms"`
	Default         string          `json:"default"`
	Presets         map[string]int  `json:"presets"`
	Frontier        []FrontierPoint `json:"frontier"`
}

// PolicyBar is the bar p_eot must reach at a check from FromMs into the pause on.
type PolicyBar struct {
	FromMs int     `json:"from_ms"`
	Bar    float64 `json:"bar"`
}

// raised is bar raised by logit in log-odds.
func raised(bar, logit float64) float64 {
	return 1 / (1 + math.Exp(-(math.Log(bar/(1-bar)) + logit)))
}

// valid says the policy has its times and a frontier of points a client can run, each with
// what it was measured at, and that its default and every preset name one of them.
func (p Policy) valid() bool {
	if p.FirstCheckMs <= 0 || p.IntervalMs <= 0 || p.CapMs <= 0 || len(p.Frontier) == 0 {
		return false
	}
	if _, ok := p.Presets[p.Default]; !ok {
		return false
	}
	for _, wait := range p.Presets {
		if _, ok := p.point(wait); !ok {
			return false
		}
	}
	for _, point := range p.Frontier {
		if !point.valid() {
			return false
		}
	}
	return true
}

// Interval and Cap are the policy's times.
func (p Policy) Interval() time.Duration { return time.Duration(p.IntervalMs) * time.Millisecond }
func (p Policy) Cap() time.Duration      { return time.Duration(p.CapMs) * time.Millisecond }

// Agent is what an agent was told and what it can do, and the languages its callers speak,
// BCP 47 tags, for the words a decision writes: a fragment in a script none of them is
// written in is noise. None is any language.
type Agent struct {
	Instructions string
	Tools        []llm.Tool
	Languages    []string
}

// Line is one line of the conversation so far.
type Line struct {
	Agent bool   `json:"agent,omitempty"`
	Text  string `json:"text"`
}

// Turn is the moment a decision is asked about.
type Turn struct {
	Speaker      string `json:"speaker,omitempty"`
	Speaking     bool   `json:"speaking,omitempty"`
	Said         string `json:"said,omitempty"`
	Unfinished   bool   `json:"unfinished,omitempty"`
	Conversation []Line `json:"conversation,omitempty"`
}

// Ruling is what to do once the turn has ended, in the flow controller's terms. Tool and
// Cannot say what only this model can: the tool the caller wants, or that none can do it.
type Ruling struct {
	Disposition string `json:"disposition"`
	Floor       string `json:"floor"`
	Tool        string `json:"tool,omitempty"`
	Cannot      bool   `json:"cannot,omitempty"`
}

// Answer is what the model made of a turn.
type Answer struct {
	Answer            string   `json:"answer"`
	AnswerProbability float64  `json:"answer_probability"`
	PEOT              *float64 `json:"p_eot"`
	DecideMs          float64  `json:"decide_ms"`
	Ruling            *Ruling  `json:"ruling"`
}

const (
	// policyTTL is how long a policy read from the server stands before it is read again,
	// in the background, so that a server moved to another model is followed within it;
	// policyRetry is how soon a read that failed is tried again; and policyReadTimeout
	// bounds a read made in the background, which is on no call's deadline.
	policyTTL         = 5 * time.Minute
	policyRetry       = 30 * time.Second
	policyReadTimeout = wordsTimeout
)

// Client asks one server. It is safe for concurrent use.
type Client struct {
	url    string
	format Format
	http   *http.Client
	// apiKey, when the server is behind a gateway that asks for one, goes with every
	// request and in nothing else.
	apiKey string
	// interims says its transcribers write the words so far while the caller talks.
	interims bool
	// overlapTimeout is how long a ruling on a caller talking over the agent is waited for
	// (HearDeadline), the flow controller's own share of its deadline when zero.
	overlapTimeout time.Duration
	// now is the clock the policy's age is read by: time.Now, but in tests.
	now func() time.Time

	mu sync.Mutex
	// policy is the last one read, nextRead when it is to be read again, and reading says a
	// read is out, in the background. inline says the server read last takes a decision's agent
	// with it (GET /v1/info's inline_agents).
	policy   *Policy
	nextRead time.Time
	reading  bool
	inline   bool
	agents   map[[sha256.Size]byte]registration
	// backOffUntil is when the last Retry-After the server answered 503 or 429 with runs out,
	// by now.
	backOffUntil time.Time
}

// registration is the server's name for an agent, or why it refused it: an agent refused
// once is refused until it changes, and isn't offered again on every turn.
type registration struct {
	id      string
	refused error
}

// New returns a client of the server at url, sending it audio in format, over client, or over
// one of the package's own when nil (serverClient).
func New(url string, format Format, client *http.Client) *Client {
	if client == nil {
		client = serverClient()
	}
	return &Client{url: strings.TrimRight(url, "/"), format: format, http: client, interims: true, now: time.Now, agents: map[[sha256.Size]byte]registration{}}
}

// WithAPIKey has the client send key, when there is one, as a bearer token on every request,
// and returns the client.
func (c *Client) WithAPIKey(key string) *Client {
	c.apiKey = key
	return c
}

// HearDeadline is how long the server has to rule on turn: the deployment's wait while the
// caller talks over the agent (config.Agent.AudioFloorOverlapTimeout), and otherwise, or with
// none set, the flow controller's share of its own deadline (harness.HearDeadline).
func (c *Client) HearDeadline(turn harness.FlowTurn) time.Duration {
	if turn.Unfinished && turn.Speaking && c.overlapTimeout > 0 {
		return c.overlapTimeout
	}
	return harness.HearDeadline(turn)
}

// NewStream returns a stream of a participant's audio, as the server is sent it, that keeps
// the last keep of it.
func (c *Client) NewStream(keep time.Duration) (*Stream, error) { return newStream(c.format, keep) }

// Policy is the server's policy for its model (GET /v1/info's end_of_turn). The first call
// reads it within ctx. From then on it is the last one read, which is read again in the
// background once it is policyTTL old, and sooner after an answer that says the server may
// have changed: a 404 for an agent it was told, or a 422 refusing one. No decision waits on
// a read, and one that fails leaves the last policy standing, to be read again after
// policyRetry.
func (c *Client) Policy(ctx context.Context) (Policy, error) {
	c.mu.Lock()
	known := c.policy
	if known != nil && !c.now().Before(c.nextRead) {
		c.rereadLocked()
	}
	c.mu.Unlock()
	if known != nil {
		return *known, nil
	}
	return c.readPolicy(ctx)
}

// PolicyNow reads the server's policy now, within ctx, for a caller on no call's path that
// wants it as it is: an agent's end_of_turn is checked against it as the agent is written.
// What it reads is what Policy answers from then on.
func (c *Client) PolicyNow(ctx context.Context) (Policy, error) { return c.readPolicy(ctx) }

// rereadLocked has the policy read again in the background, unless a read is out already.
// With none read yet there is nothing to stand in for it, and the next Policy reads it.
func (c *Client) rereadLocked() {
	c.nextRead = time.Time{}
	if c.policy == nil || c.reading {
		return
	}
	c.reading = true
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), policyReadTimeout)
		defer cancel()
		_, err := c.readPolicy(ctx)
		c.mu.Lock()
		defer c.mu.Unlock()
		c.reading = false
		if err != nil {
			c.nextRead = c.now().Add(policyRetry)
		}
	}()
}

// readPolicy reads GET /v1/info within ctx, and keeps the policy it answers, and whether the
// server takes a decision's agent with it.
func (c *Client) readPolicy(ctx context.Context) (Policy, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.url+"/v1/info", nil)
	if err != nil {
		return Policy{}, stack.Wrap(err)
	}
	var info struct {
		EndOfTurn    json.RawMessage `json:"end_of_turn"`
		InlineAgents bool            `json:"inline_agents"`
	}
	sent := time.Now()
	_, err = c.do(req, &info)
	measureSetup(ctx, routeInfo, sent, err)
	if err != nil {
		return Policy{}, err
	}
	// A policy of another shape, from a server older or newer than this router, is one it
	// can't run.
	var policy Policy
	if err := json.Unmarshal(info.EndOfTurn, &policy); err != nil {
		return Policy{}, stack.Wrap(fmt.Errorf("%w: its end_of_turn is not one this router reads: %w", ErrUnsupported, err))
	}
	if !policy.valid() {
		return Policy{}, stack.Wrap(fmt.Errorf("%w: its end_of_turn is incomplete", ErrUnsupported))
	}
	c.mu.Lock()
	c.policy, c.nextRead, c.inline = &policy, c.now().Add(policyTTL), info.InlineAgents
	c.mu.Unlock()
	return policy, nil
}

// Decide asks about audio under the moment turn. The agent is registered first if it is
// new, and again if the server has forgotten it.
func (c *Client) Decide(ctx context.Context, agent Agent, turn Turn, audio harness.Audio) (Answer, error) {
	answer, _, err := c.decide(ctx, agent, turn, audio, false)
	return answer, err
}

// Words is what a request for words wrote of its audio: all of it, as POST /v1/transcribe
// does and a decision's pass can, empty when nothing was said or the server heard noise.
// Language is the language the server heard the words in, by its own name for it
// ("English"), which it takes back as one of a request's languages; a server that does not
// say leaves it empty. Noise says why the text is empty when the server heard something it
// does not write: "language" is words in a language the callers don't speak.
type Words struct {
	Text     string
	Language string
	Noise    string
	Ms       float64
	Err      error
}

// transcriptJSON is the words as the server writes them, in a transcription's answer and
// a decision's second line.
type transcriptJSON struct {
	Text     string `json:"text"`
	Language string `json:"language"`
	Noise    string `json:"noise"`
}

func (t *transcriptJSON) words(ms float64) Words {
	return Words{Text: t.Text, Language: t.Language, Noise: t.Noise, Ms: ms}
}

// DecideWords is Decide that also has the server write the words of all of audio in the
// same pass, so that a pause costs one request. The answer is back as soon as the decision
// is, within ctx. The words follow on the channel, once and within wordsTimeout and the
// audio's length, whatever has become of ctx by then.
func (c *Client) DecideWords(ctx context.Context, agent Agent, turn Turn, audio harness.Audio) (Answer, <-chan Words, error) {
	return c.decide(ctx, agent, turn, audio, true)
}

func (c *Client) decide(ctx context.Context, agent Agent, turn Turn, audio harness.Audio, words bool) (Answer, <-chan Words, error) {
	if len(audio.Data) == 0 {
		return Answer{}, nil, stack.Wrap(errors.New("audiofloor: no audio heard yet"))
	}
	// Words still coming are said only over the agent (turnOf).
	kind := kindPause
	if turn.Unfinished {
		kind = kindOverAgent
	}
	if kind == kindOverAgent && c.backingOff() {
		// A check over the agent decides no turn: the flow controller has it while the server
		// asks for time, and the pauses' checks, which do, are what it is asked.
		backedOff.WithLabelValues(kind).Inc()
		return Answer{}, nil, stack.Wrap(ErrBackingOff)
	}
	key, spec, err := agentKey(agent)
	if err != nil {
		return Answer{}, nil, err
	}
	c.mu.Lock()
	inline := c.inline
	c.mu.Unlock()
	var told []byte
	if inline {
		told = spec
	}
	body, contentType, err := decisionBody(turn, audio, told)
	if err != nil {
		return Answer{}, nil, err
	}
	for attempt := 0; ; attempt++ {
		id, err := c.register(ctx, key, spec, attempt > 0)
		if err != nil {
			return Answer{}, nil, err
		}
		target := c.url + "/v1/decide?agent=" + id
		var answer Answer
		var said <-chan Words
		var status int
		sent := time.Now()
		if words {
			answer, said, status, err = c.decideWords(ctx, target+"&transcript=true", body, contentType, wordsTimeout+audio.Duration)
		} else {
			var req *http.Request
			if req, err = http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(body)); err != nil {
				return Answer{}, nil, stack.Wrap(err)
			}
			req.Header.Set("Content-Type", contentType)
			setDeadline(req.Header, ctx)
			status, err = c.do(req, &answer)
		}
		if err == nil && answer.Ruling == nil {
			err = stack.Wrap(errors.New("audiofloor: decide: the answer has no ruling"))
		}
		measure(ctx, kind, sent, err)
		if status != http.StatusNotFound {
			return answer, said, err
		}
		if attempt > 0 {
			// Told again, and still not known where the decision went: another replica than
			// the one told, as a server that takes no agent inline has. It is up.
			return Answer{}, nil, fmt.Errorf("%w: %w", ErrAgentUnknown, err)
		}
		// The replica that answered doesn't know the agent: one restarted, or one of several
		// behind the address that was never told. That agent alone is told again, the others'
		// names standing, as an agent's name is its content's and the same on every replica.
		// A restarted server may serve another model, so its policy is read again, in the
		// background, the last one read standing until then.
		c.mu.Lock()
		c.rereadLocked()
		c.mu.Unlock()
	}
}

// decideWords asks for a decision and its words, and reads the words on after the decision
// has come back: the request outlives ctx, which only bounds the decision.
func (c *Client) decideWords(ctx context.Context, target string, body []byte, contentType string, timeout time.Duration) (Answer, <-chan Words, int, error) {
	asking, cancel := context.WithTimeout(context.WithoutCancel(ctx), timeout)
	stop := context.AfterFunc(ctx, cancel)
	req, err := http.NewRequestWithContext(asking, http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		cancel()
		return Answer{}, nil, 0, stack.Wrap(err)
	}
	req.Header.Set("Content-Type", contentType)
	// What the server is told it has is the decision's deadline, ctx's: past it, nobody
	// waits for the words either.
	setDeadline(req.Header, ctx)
	given := func(err error) error {
		// The decision's deadline, rather than the request's own, is what ran out.
		if ctx.Err() != nil {
			return stack.Wrap(fmt.Errorf("audiofloor: decide: %w", ctx.Err()))
		}
		return stack.Wrap(err)
	}
	resp, err := c.send(req)
	if err != nil {
		stop()
		cancel()
		return Answer{}, nil, 0, given(err)
	}
	if resp.StatusCode != http.StatusOK {
		stop()
		defer cancel()
		defer resp.Body.Close()
		return Answer{}, nil, resp.StatusCode, stack.Wrap(c.statusError(req, resp))
	}
	lines := json.NewDecoder(io.LimitReader(resp.Body, responseLimit))
	var answer Answer
	if err := lines.Decode(&answer); err != nil {
		stop()
		cancel()
		resp.Body.Close()
		return Answer{}, nil, resp.StatusCode, given(fmt.Errorf("audiofloor: %s: %w", req.URL.Path, err))
	}
	if !stop() {
		cancel()
		resp.Body.Close()
		return Answer{}, nil, resp.StatusCode, given(ctx.Err())
	}
	said := make(chan Words, 1)
	go func() {
		defer cancel()
		defer resp.Body.Close()
		var line struct {
			Transcript *transcriptJSON `json:"transcript"`
			Error      string          `json:"transcript_error"`
			Ms         float64         `json:"transcript_ms"`
		}
		switch err := lines.Decode(&line); {
		case err != nil:
			said <- Words{Err: stack.Wrap(fmt.Errorf("audiofloor: %s: the words: %w", req.URL.Path, err))}
		case line.Transcript == nil:
			said <- Words{Err: stack.Wrap(&TranscriptError{Request: req.URL.Path + ": the words", Message: line.Error})}
		default:
			said <- line.Transcript.words(line.Ms)
		}
	}()
	return answer, said, resp.StatusCode, nil
}

// Transcribe is what was said in audio, by callers who speak languages, BCP 47 tags or the
// server's own names for them (Words.Language), or any language when there are none: empty
// when nothing was, or when the server heard noise.
func (c *Client) Transcribe(ctx context.Context, audio harness.Audio, languages []string) (Words, error) {
	if len(audio.Data) == 0 {
		return Words{}, stack.Wrap(errors.New("audiofloor: no audio heard yet"))
	}
	target := c.url + "/v1/transcribe"
	if len(languages) > 0 {
		target += "?languages=" + url.QueryEscape(strings.Join(languages, ","))
	}
	// The audio goes in a part, as a decision's does: a gateway may change a raw body's type.
	body, contentType, err := formBody(formPart{"audio", audio.Type, audio.Data})
	if err != nil {
		return Words{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		return Words{}, stack.Wrap(err)
	}
	req.Header.Set("Content-Type", contentType)
	setDeadline(req.Header, ctx)
	var said struct {
		Transcript *transcriptJSON `json:"transcript"`
		Error      string          `json:"transcript_error"`
		Ms         float64         `json:"transcript_ms"`
	}
	sent := time.Now()
	_, err = c.do(req, &said)
	if err == nil && said.Transcript == nil {
		err = stack.Wrap(&TranscriptError{Request: "transcribe", Message: said.Error})
	}
	measure(ctx, kindTranscribe, sent, err)
	if err != nil {
		return Words{}, err
	}
	return said.Transcript.words(said.Ms), nil
}

type toolJSON struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type agentJSON struct {
	Instructions string     `json:"instructions"`
	Tools        []toolJSON `json:"tools,omitempty"`
	Languages    []string   `json:"languages,omitempty"`
}

func agentKey(agent Agent) ([sha256.Size]byte, []byte, error) {
	spec := agentJSON{Instructions: agent.Instructions, Languages: agent.Languages}
	for _, tool := range agent.Tools {
		spec.Tools = append(spec.Tools, toolJSON{tool.Name, tool.Description, tool.Parameters})
	}
	raw, err := json.Marshal(spec)
	if err != nil {
		return [sha256.Size]byte{}, nil, stack.Wrap(err)
	}
	return sha256.Sum256(raw), raw, nil
}

// register returns the server's name for the agent spec, registering it if it isn't known, or
// again, though it is, when the server answering a decision about it did not know it.
func (c *Client) register(ctx context.Context, key [sha256.Size]byte, spec []byte, again bool) (string, error) {
	c.mu.Lock()
	known, ok := c.agents[key]
	c.mu.Unlock()
	if ok && !again {
		return known.id, known.refused
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url+"/v1/agents", bytes.NewReader(spec))
	if err != nil {
		return "", stack.Wrap(err)
	}
	req.Header.Set("Content-Type", "application/json")
	var registered struct {
		Agent string `json:"agent"`
	}
	sent := time.Now()
	status, err := c.do(req, &registered)
	measureSetup(ctx, routeAgents, sent, err)
	result := registration{id: registered.Agent}
	switch {
	case status == http.StatusUnprocessableEntity:
		result.refused = fmt.Errorf("%w: %w", ErrRefused, err)
	case err != nil:
		return "", err
	}
	c.mu.Lock()
	if len(c.agents) >= maxAgents {
		clear(c.agents)
	}
	c.agents[key] = result
	if result.refused != nil {
		// Refused for more tools than the model has answers for: the server may have moved
		// to another model, with another policy.
		c.rereadLocked()
	}
	c.mu.Unlock()
	return result.id, result.refused
}

// send is the one way a request leaves the client. Every request it sends is idempotent:
// reading the policy, registering an agent by its content, and a decision or the words of
// audio, which change nothing on the server. So each is marked so, with an Idempotency-Key
// that is not sent (net/http: a nil value is never written), and net/http sends it again on a
// new connection when a kept-alive one turns out closed by the server as the request went out
// on it, which it otherwise does only for a GET.
func (c *Client) send(req *http.Request) (*http.Response, error) {
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	if _, ok := req.Header["Idempotency-Key"]; !ok {
		req.Header["Idempotency-Key"] = nil
	}
	return c.http.Do(req)
}

// setDeadline says in header how long the router will wait for the answer, ctx's deadline
// from now (TimeoutHeader), when ctx has one that has yet to pass.
func setDeadline(header http.Header, ctx context.Context) {
	deadline, ok := ctx.Deadline()
	if !ok {
		return
	}
	if left := time.Until(deadline); left > 0 {
		header.Set(TimeoutHeader, strconv.FormatInt(max(1, left.Milliseconds()), 10))
	}
}

// backingOff says a Retry-After the server answered with has yet to pass.
func (c *Client) backingOff() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now().Before(c.backOffUntil)
}

// backOff heeds the Retry-After of a 503 or 429: seconds, or an HTTP date, up to maxBackOff
// from now. One that has passed, or that is neither, changes nothing.
func (c *Client) backOff(retryAfter string) {
	retryAfter = strings.TrimSpace(retryAfter)
	if retryAfter == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.now()
	var wait time.Duration
	if seconds, err := strconv.ParseInt(retryAfter, 10, 64); err == nil {
		wait = time.Duration(min(seconds, int64(maxBackOff/time.Second))) * time.Second
	} else if at, err := http.ParseTime(retryAfter); err == nil {
		wait = at.Sub(now)
	}
	if until := now.Add(min(wait, maxBackOff)); wait > 0 && until.After(c.backOffUntil) {
		c.backOffUntil = until
	}
}

// do sends req and reads the answer, a JSON line, into into.
func (c *Client) do(req *http.Request, into any) (int, error) {
	resp, err := c.send(req)
	if err != nil {
		return 0, stack.Wrap(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return resp.StatusCode, stack.Wrap(c.statusError(req, resp))
	}
	answer := json.NewDecoder(io.LimitReader(resp.Body, responseLimit))
	if err := answer.Decode(into); err != nil {
		return resp.StatusCode, stack.Wrap(fmt.Errorf("audiofloor: %s: %w", req.URL.Path, err))
	}
	return resp.StatusCode, nil
}

// statusError is the server's answer to req, which isn't 200: a StatusError, and
// ErrUnsupported for a server with no policy. Errors are logged, and a gateway may quote the
// key it turned away, so the key is taken out of what it said.
func (c *Client) statusError(req *http.Request, resp *http.Response) error {
	var failed struct {
		Error string `json:"error"`
	}
	_ = json.NewDecoder(io.LimitReader(resp.Body, responseLimit)).Decode(&failed)
	if c.apiKey != "" {
		failed.Error = strings.ReplaceAll(failed.Error, c.apiKey, "[key]")
	}
	var err error = &StatusError{Method: req.Method, Path: req.URL.Path, Code: resp.StatusCode, Status: resp.Status, Message: failed.Error}
	switch resp.StatusCode {
	case http.StatusNotImplemented:
		err = fmt.Errorf("%w: %w", ErrUnsupported, err)
	case http.StatusServiceUnavailable, http.StatusTooManyRequests:
		c.backOff(resp.Header.Get("Retry-After"))
	}
	return err
}

// decisionBody is a decision's multipart body: the turn, then the audio, then the agent when
// there is one to tell the server inline, the JSON POST /v1/agents takes. What a prompt won't
// show isn't sent.
func decisionBody(turn Turn, audio harness.Audio, agent []byte) ([]byte, string, error) {
	turn.Said = lastRunes(turn.Said, maxSaid)
	turn.Conversation = slices.Clone(lastTurns(turn.Conversation, maxTurns))
	for i, line := range turn.Conversation {
		if words := strings.Fields(line.Text); len(words) > maxLineWords {
			turn.Conversation[i].Text = strings.Join(words[len(words)-maxLineWords:], " ")
		}
	}
	var moment []byte
	for {
		var err error
		if moment, err = json.Marshal(turn); err != nil {
			return nil, "", stack.Wrap(err)
		}
		if len(moment) <= maxTurnBytes {
			break
		}
		if len(turn.Conversation) == 0 {
			return nil, "", stack.Wrap(errors.New("audiofloor: the moment of the call is too long to say"))
		}
		turn.Conversation = turn.Conversation[1:]
	}
	parts := []formPart{{"turn", "application/json", moment}, {"audio", audio.Type, audio.Data}}
	if agent != nil {
		parts = append(parts, formPart{"agent", "application/json", agent})
	}
	return formBody(parts...)
}

// formPart is a part of a multipart/form-data body: its name, type and bytes.
type formPart struct {
	name, contentType string
	raw               []byte
}

// formBody is a multipart/form-data body of parts, in their order, and its Content-Type. Every
// route that sends audio sends it so, in a part with its own type, since a gateway may change
// the type of a raw body.
func formBody(parts ...formPart) ([]byte, string, error) {
	size := 512 * len(parts)
	for _, part := range parts {
		size += len(part.raw)
	}
	var body bytes.Buffer
	body.Grow(size)
	writer := multipart.NewWriter(&body)
	for _, part := range parts {
		header := textproto.MIMEHeader{}
		header.Set("Content-Disposition", `form-data; name="`+part.name+`"`)
		header.Set("Content-Type", part.contentType)
		w, err := writer.CreatePart(header)
		if err != nil {
			return nil, "", stack.Wrap(err)
		}
		if _, err := w.Write(part.raw); err != nil {
			return nil, "", stack.Wrap(err)
		}
	}
	if err := writer.Close(); err != nil {
		return nil, "", stack.Wrap(err)
	}
	return body.Bytes(), writer.FormDataContentType(), nil
}

// lastTurns is the lines of the last n turns, a turn being one side's run of lines: the
// server joins a run into one line with the floor free, and shows the last n lines over the
// agent, so both are in them.
func lastTurns(lines []Line, n int) []Line {
	turns := 0
	for i := len(lines) - 1; i >= 0; i-- {
		if i == len(lines)-1 || lines[i].Agent != lines[i+1].Agent {
			turns++
		}
		if turns > n {
			return lines[i+1:]
		}
	}
	return lines
}

func lastRunes(s string, n int) string {
	at := len(s)
	for count := 0; at > 0 && count < n; count++ {
		_, size := utf8.DecodeLastRuneInString(s[:at])
		at -= size
	}
	return s[at:]
}

// serverClient is the client every Client asks its server over unless it is given one, one a
// process: net/http's default transport as the process has it, keeping as many idle
// connections a host as the router's (maxIdleConns), each closed after idleConnTimeout unused.
var serverClient = sync.OnceValue(func() *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxIdleConns, transport.MaxIdleConnsPerHost = maxIdleConns, maxIdleConns
	transport.IdleConnTimeout = idleConnTimeout
	return &http.Client{Transport: transport}
})

// Open returns a client of the server a deployment's settings name, or nil without one.
// Loading the settings checked them all.
func Open(settings config.Agent) *Client {
	if settings.AudioFloorURL == "" {
		return nil
	}
	client := New(settings.AudioFloorURL, Format(cmp.Or(settings.AudioFloorAudio, string(Opus))), nil).WithAPIKey(settings.AudioFloorAPIKey)
	client.interims = settings.AudioFloorInterims != "off"
	client.overlapTimeout = settings.AudioFloorOverlapTimeout
	return client
}
