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
//	POST /v1/transcribe       audio, and ?languages= the callers speak: what was said,
//	                          empty when nothing was or the server heard noise
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
)

var (
	// ErrRefused is an agent the server won't ask about, for more tools than the model has
	// answers for.
	ErrRefused = errors.New("audiofloor: the server refused the agent")
	// ErrUnsupported is a server with no policy for its model, which asks about no agent.
	ErrUnsupported = errors.New("audiofloor: the server has no policy for its model")
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

// The speakers a prompt can name. Training's rows name nobody else, so a participant's own
// name goes no further than this package.
const (
	caller       = "caller"
	anotherVoice = "another_voice"
)

// Policy is when a turn with the floor free has ended, fitted to the server's model. A
// pause is checked FirstCheckMs into it and every IntervalMs after, and the turn has ended
// at the first check where p_eot reaches the bar of the last of the preset's bars to have
// begun by then, raised by FirstPauseLogit in log-odds on the turn's first pause. At CapMs
// of pause it has ended whatever the model says. The presets are how eager turn-taking is,
// by name, and the server names its Default.
type Policy struct {
	FirstCheckMs    int               `json:"first_check_ms"`
	FirstPauseLogit float64           `json:"first_pause_logit"`
	IntervalMs      int               `json:"interval_ms"`
	CapMs           int               `json:"cap_ms"`
	Default         string            `json:"default"`
	Presets         map[string]Preset `json:"presets"`
}

// Preset is one choice of bars, each from a time into the pause on.
type Preset struct {
	Bars []PolicyBar `json:"bars"`
}

// PolicyBar is the bar p_eot must reach at a check from FromMs into the pause on.
type PolicyBar struct {
	FromMs int     `json:"from_ms"`
	Bar    float64 `json:"bar"`
}

// Bar is what p_eot must reach, by the default preset, at a check paused into the pause for
// the turn to have ended.
func (p Policy) Bar(paused time.Duration, firstPause bool) float64 {
	var bar float64
	for _, b := range p.Presets[p.Default].Bars {
		if paused >= time.Duration(b.FromMs)*time.Millisecond {
			bar = b.Bar
		}
	}
	if firstPause {
		bar = 1 / (1 + math.Exp(-(math.Log(bar/(1-bar)) + p.FirstPauseLogit)))
	}
	return bar
}

// valid says the policy has its times and a default preset, and that every preset has a
// bar for every check, each a probability.
func (p Policy) valid() bool {
	if _, ok := p.Presets[p.Default]; !ok || p.FirstCheckMs <= 0 || p.IntervalMs <= 0 || p.CapMs <= 0 {
		return false
	}
	for _, preset := range p.Presets {
		bars := preset.Bars
		if len(bars) == 0 || bars[0].FromMs != 0 {
			return false
		}
		for i, b := range bars {
			if b.Bar <= 0 || b.Bar >= 1 || (i > 0 && b.FromMs <= bars[i-1].FromMs) {
				return false
			}
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

// Client asks one server. It is safe for concurrent use.
type Client struct {
	url    string
	format Format
	http   *http.Client

	mu     sync.Mutex
	policy *Policy
	agents map[[sha256.Size]byte]registration
}

// registration is the server's name for an agent, or why it refused it: an agent refused
// once is refused until it changes, and isn't offered again on every turn.
type registration struct {
	id      string
	refused error
}

// New returns a client of the server at url, sending it audio in format.
func New(url string, format Format, client *http.Client) *Client {
	if client == nil {
		client = http.DefaultClient
	}
	return &Client{url: strings.TrimRight(url, "/"), format: format, http: client, agents: map[[sha256.Size]byte]registration{}}
}

// NewStream returns a stream of a participant's audio, as the server is sent it, that keeps
// the last keep of it.
func (c *Client) NewStream(keep time.Duration) (*Stream, error) { return newStream(c.format, keep) }

// Policy is the server's policy for its model (GET /v1/info's end_of_turn), asked once and
// again after the server has restarted.
func (c *Client) Policy(ctx context.Context) (Policy, error) {
	c.mu.Lock()
	known := c.policy
	c.mu.Unlock()
	if known != nil {
		return *known, nil
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.url+"/v1/info", nil)
	if err != nil {
		return Policy{}, stack.Wrap(err)
	}
	var info struct {
		EndOfTurn Policy `json:"end_of_turn"`
	}
	if _, err := c.do(req, &info); err != nil {
		return Policy{}, err
	}
	if !info.EndOfTurn.valid() {
		return Policy{}, stack.Wrap(fmt.Errorf("%w: its end_of_turn is incomplete", ErrUnsupported))
	}
	c.mu.Lock()
	c.policy = &info.EndOfTurn
	c.mu.Unlock()
	return info.EndOfTurn, nil
}

// Decide asks about audio under the moment turn. The agent is registered first if it is
// new, and again if the server has forgotten it.
func (c *Client) Decide(ctx context.Context, agent Agent, turn Turn, audio harness.Audio) (Answer, error) {
	answer, _, err := c.decide(ctx, agent, turn, audio, false)
	return answer, err
}

// Words is what a decision's pass wrote of its audio: all of it, as POST /v1/transcribe
// would, empty when nothing was said or the server heard noise.
type Words struct {
	Text string
	Ms   float64
	Err  error
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
	body, contentType, err := decisionBody(turn, audio)
	if err != nil {
		return Answer{}, nil, err
	}
	key, spec, err := agentKey(agent)
	if err != nil {
		return Answer{}, nil, err
	}
	for attempt := 0; ; attempt++ {
		id, err := c.register(ctx, key, spec)
		if err != nil {
			return Answer{}, nil, err
		}
		target := c.url + "/v1/decide?agent=" + id
		var answer Answer
		var said <-chan Words
		var status int
		if words {
			answer, said, status, err = c.decideWords(ctx, target+"&transcript=true", body, contentType, wordsTimeout+audio.Duration)
		} else {
			var req *http.Request
			if req, err = http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(body)); err != nil {
				return Answer{}, nil, stack.Wrap(err)
			}
			req.Header.Set("Content-Type", contentType)
			status, err = c.do(req, &answer)
		}
		if status == http.StatusNotFound && attempt == 0 {
			// A restarted server knows no agent, and may serve another model.
			c.mu.Lock()
			clear(c.agents)
			c.policy = nil
			c.mu.Unlock()
			continue
		}
		if err == nil && answer.Ruling == nil {
			err = stack.Wrap(errors.New("audiofloor: decide: the answer has no ruling"))
		}
		return answer, said, err
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
	given := func(err error) error {
		// The decision's deadline, rather than the request's own, is what ran out.
		if ctx.Err() != nil {
			return stack.Wrap(fmt.Errorf("audiofloor: decide: %w", ctx.Err()))
		}
		return stack.Wrap(err)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		stop()
		cancel()
		return Answer{}, nil, 0, given(err)
	}
	if resp.StatusCode != http.StatusOK {
		stop()
		defer cancel()
		defer resp.Body.Close()
		return Answer{}, nil, resp.StatusCode, stack.Wrap(statusError(req, resp))
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
			Transcript *struct {
				Text string `json:"text"`
			} `json:"transcript"`
			Error string  `json:"transcript_error"`
			Ms    float64 `json:"transcript_ms"`
		}
		switch err := lines.Decode(&line); {
		case err != nil:
			said <- Words{Err: stack.Wrap(fmt.Errorf("audiofloor: %s: the words: %w", req.URL.Path, err))}
		case line.Transcript == nil:
			said <- Words{Err: stack.Wrap(fmt.Errorf("audiofloor: %s: the words: %s", req.URL.Path, line.Error))}
		default:
			said <- Words{Text: line.Transcript.Text, Ms: line.Ms}
		}
	}()
	return answer, said, resp.StatusCode, nil
}

// Transcribe is what was said in audio, by callers who speak languages, BCP 47 tags, or any
// language when there are none: empty when nothing was, or when the server heard noise.
func (c *Client) Transcribe(ctx context.Context, audio harness.Audio, languages []string) (string, error) {
	if len(audio.Data) == 0 {
		return "", stack.Wrap(errors.New("audiofloor: no audio heard yet"))
	}
	target := c.url + "/v1/transcribe"
	if len(languages) > 0 {
		target += "?languages=" + url.QueryEscape(strings.Join(languages, ","))
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(audio.Data))
	if err != nil {
		return "", stack.Wrap(err)
	}
	req.Header.Set("Content-Type", audio.Type)
	var said struct {
		Transcript *struct {
			Text string `json:"text"`
		} `json:"transcript"`
		Error string `json:"transcript_error"`
	}
	if _, err := c.do(req, &said); err != nil {
		return "", err
	}
	if said.Transcript == nil {
		return "", stack.Wrap(fmt.Errorf("audiofloor: transcribe: %s", said.Error))
	}
	return said.Transcript.Text, nil
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

// register returns the server's name for the agent spec, registering it if it isn't known.
func (c *Client) register(ctx context.Context, key [sha256.Size]byte, spec []byte) (string, error) {
	c.mu.Lock()
	known, ok := c.agents[key]
	c.mu.Unlock()
	if ok {
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
	status, err := c.do(req, &registered)
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
	c.mu.Unlock()
	return result.id, result.refused
}

// do sends req and reads the answer, a JSON line, into into.
func (c *Client) do(req *http.Request, into any) (int, error) {
	resp, err := c.http.Do(req)
	if err != nil {
		return 0, stack.Wrap(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return resp.StatusCode, stack.Wrap(statusError(req, resp))
	}
	answer := json.NewDecoder(io.LimitReader(resp.Body, responseLimit))
	if err := answer.Decode(into); err != nil {
		return resp.StatusCode, stack.Wrap(fmt.Errorf("audiofloor: %s: %w", req.URL.Path, err))
	}
	return resp.StatusCode, nil
}

// statusError is the server's answer to req, which isn't 200: a StatusError, and
// ErrUnsupported for a server with no policy.
func statusError(req *http.Request, resp *http.Response) error {
	var failed struct {
		Error string `json:"error"`
	}
	_ = json.NewDecoder(io.LimitReader(resp.Body, responseLimit)).Decode(&failed)
	var err error = &StatusError{Method: req.Method, Path: req.URL.Path, Code: resp.StatusCode, Status: resp.Status, Message: failed.Error}
	if resp.StatusCode == http.StatusNotImplemented {
		err = fmt.Errorf("%w: %w", ErrUnsupported, err)
	}
	return err
}

// decisionBody is a decision's multipart body: the turn, then the audio. What a prompt
// won't show isn't sent.
func decisionBody(turn Turn, audio harness.Audio) ([]byte, string, error) {
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
	var body bytes.Buffer
	body.Grow(len(moment) + len(audio.Data) + 512)
	parts := multipart.NewWriter(&body)
	for _, part := range []struct {
		name, contentType string
		raw               []byte
	}{
		{"turn", "application/json", moment},
		{"audio", audio.Type, audio.Data},
	} {
		header := textproto.MIMEHeader{}
		header.Set("Content-Disposition", `form-data; name="`+part.name+`"`)
		header.Set("Content-Type", part.contentType)
		w, err := parts.CreatePart(header)
		if err != nil {
			return nil, "", stack.Wrap(err)
		}
		if _, err := w.Write(part.raw); err != nil {
			return nil, "", stack.Wrap(err)
		}
	}
	if err := parts.Close(); err != nil {
		return nil, "", stack.Wrap(err)
	}
	return body.Bytes(), parts.FormDataContentType(), nil
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

// Open returns a client of the server a deployment's settings name, or nil without one, and
// the mode its agents ask it in. Loading the settings checked both.
func Open(settings config.Agent) (*Client, Mode) {
	if settings.AudioFloorURL == "" {
		return nil, ""
	}
	return New(settings.AudioFloorURL, Format(cmp.Or(settings.AudioFloorAudio, string(Opus))), nil), Mode(settings.AudioFloorMode)
}
