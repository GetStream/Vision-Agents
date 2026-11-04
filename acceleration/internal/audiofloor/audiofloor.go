// Package audiofloor asks an audio floor model about a call: one speech model that hears the
// caller's last 16 s and answers the question the flow controller answers from the words,
// says which of the agent's tools is wanted, and can write what was said. It is served by an
// audio floor server, which speaks this protocol:
//
//	GET  /v1/info             what the server serves: its prompt template, whether a prompt
//	                          with the floor free says the conversation, the most tools an
//	                          agent can have, the audio a decision hears, and end_of_turn,
//	                          the policy fitted to its model for ending a turn
//	POST /v1/agents           {"instructions","tools"}: registers an agent by its content
//	                          and answers {"agent": ID}. More tools than the model has
//	                          answers for is a 422
//	POST /v1/decide?agent=ID  a multipart body: a "turn" part, the moment of the call as
//	                          JSON, then an "audio" part, 16 kHz mono PCM. The answer is
//	                          every option's probability, p_eot = 1 - p(wait) with the floor
//	                          free, and the ruling to act on once the turn has ended. With
//	                          transcript=true a second line follows with what was said
//
// The server writes the model's prompt, because the model knows only the words it was
// trained on. The server says what to do with the words; whether a turn with the floor free
// has ended is the router's to say, from p_eot, the pause and the policy. A server started
// without a policy for its model answers 501 to all three.
package audiofloor

import (
	"bytes"
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
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

const (
	// SampleRate is what the server takes and the model hears.
	SampleRate = 16000
	// windowSamples is what a decision hears.
	windowSamples = 16 * SampleRate
	// maxSaid and maxTurns are what a prompt shows of the reply and of the conversation, so
	// sending more only spends the request.
	maxSaid  = 800
	maxTurns = 6
	// maxTurnBytes is the most the server takes in a turn.
	maxTurnBytes = 64 << 10
	// maxAgents bounds the IDs a client remembers. A router's agents are few, and one
	// forgotten costs a registration.
	maxAgents = 1024
	// responseLimit bounds an answer, which is two small JSON lines.
	responseLimit = 1 << 20
)

var (
	// ErrRefused is an agent the server won't ask about, for more tools than the model has
	// answers for.
	ErrRefused = errors.New("audiofloor: the server refused the agent")
	// ErrUnsupported is a server with no policy for its model, which asks about no agent.
	ErrUnsupported = errors.New("audiofloor: the server has no policy for its model")
)

// The speakers a prompt can name. Training's rows name nobody else, so a participant's own
// name goes no further than this package.
const (
	Caller       = "caller"
	AnotherVoice = "another_voice"
)

// Info is what a server says it serves, in the parts the router reads.
type Info struct {
	Template  string `json:"template"`
	EndOfTurn Policy `json:"end_of_turn"`
}

// Policy is when a turn with the floor free has ended, fitted to the server's model. A
// pause is checked FirstCheckMs into it and every IntervalMs after, and the turn has ended
// at the first check where p_eot reaches the bar: FirstBar at the pause's first check and
// LaterBar after it, each raised by FirstPauseLogit in log-odds on the turn's first pause.
// At CapMs of pause it has ended whatever the model says.
type Policy struct {
	FirstCheckMs    int     `json:"first_check_ms"`
	FirstBar        float64 `json:"first_bar"`
	LaterBar        float64 `json:"later_bar"`
	FirstPauseLogit float64 `json:"first_pause_logit"`
	IntervalMs      int     `json:"interval_ms"`
	CapMs           int     `json:"cap_ms"`
}

// Bar is what p_eot must reach at a check for the turn to have ended.
func (p Policy) Bar(firstCheck, firstPause bool) float64 {
	bar := p.LaterBar
	if firstCheck {
		bar = p.FirstBar
	}
	if firstPause {
		bar = 1 / (1 + math.Exp(-(math.Log(bar/(1-bar)) + p.FirstPauseLogit)))
	}
	return bar
}

// FirstCheck, Interval and Cap are the policy's times.
func (p Policy) FirstCheck() time.Duration { return time.Duration(p.FirstCheckMs) * time.Millisecond }
func (p Policy) Interval() time.Duration   { return time.Duration(p.IntervalMs) * time.Millisecond }
func (p Policy) Cap() time.Duration        { return time.Duration(p.CapMs) * time.Millisecond }

// Agent is what an agent was told and what it can do.
type Agent struct {
	Instructions string
	Tools        []llm.Tool
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

// Option is one answer's probability.
type Option struct {
	Name string  `json:"name"`
	P    float64 `json:"p"`
}

// Answer is what the model made of a turn.
type Answer struct {
	Answer            string   `json:"answer"`
	AnswerProbability float64  `json:"answer_probability"`
	PEOT              *float64 `json:"p_eot"`
	Options           []Option `json:"options"`
	DecideMs          float64  `json:"decide_ms"`
	Ruling            *Ruling  `json:"ruling"`
	// Transcript is what was said, when asked for.
	Transcript string `json:"-"`
}

// Client asks one server. It is safe for concurrent use.
type Client struct {
	url  string
	http *http.Client

	mu     sync.Mutex
	info   *Info
	agents map[[sha256.Size]byte]registration
}

// registration is the server's name for an agent, or why it refused it: an agent refused
// once is refused until it changes, and isn't offered again on every turn.
type registration struct {
	id      string
	refused error
}

// New returns a client of the server at url.
func New(url string, client *http.Client) *Client {
	if client == nil {
		client = http.DefaultClient
	}
	return &Client{url: strings.TrimRight(url, "/"), http: client, agents: map[[sha256.Size]byte]registration{}}
}

// Info is what the server serves, asked once and again after the server has restarted.
func (c *Client) Info(ctx context.Context) (Info, error) {
	c.mu.Lock()
	known := c.info
	c.mu.Unlock()
	if known != nil {
		return *known, nil
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.url+"/v1/info", nil)
	if err != nil {
		return Info{}, stack.Wrap(err)
	}
	var info Info
	if _, err := c.do(req, &info, nil); err != nil {
		return Info{}, err
	}
	if p := info.EndOfTurn; p.FirstBar <= 0 || p.LaterBar <= 0 || p.FirstCheckMs <= 0 || p.IntervalMs <= 0 || p.CapMs <= 0 {
		return Info{}, stack.Wrap(fmt.Errorf("%w: its end_of_turn is incomplete", ErrUnsupported))
	}
	c.mu.Lock()
	c.info = &info
	c.mu.Unlock()
	return info, nil
}

// Decide asks about pcm, 16 kHz mono, under the moment turn, and for what was said in it
// when transcript is set. The agent is registered first if it is new, and again if the
// server has forgotten it.
func (c *Client) Decide(ctx context.Context, agent Agent, turn Turn, pcm []int16, transcript bool) (Answer, error) {
	body, contentType, err := decisionBody(turn, pcm)
	if err != nil {
		return Answer{}, err
	}
	key, spec, err := agentKey(agent)
	if err != nil {
		return Answer{}, err
	}
	query := "?agent="
	if transcript {
		query = "?transcript=true&agent="
	}
	for attempt := 0; ; attempt++ {
		id, err := c.register(ctx, key, spec)
		if err != nil {
			return Answer{}, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url+"/v1/decide"+query+id, bytes.NewReader(body))
		if err != nil {
			return Answer{}, stack.Wrap(err)
		}
		req.Header.Set("Content-Type", contentType)
		var (
			answer Answer
			words  *string
		)
		if transcript {
			words = &answer.Transcript
		}
		status, err := c.do(req, &answer, words)
		if status == http.StatusNotFound && attempt == 0 {
			// A restarted server knows no agent, and may serve another model.
			c.mu.Lock()
			clear(c.agents)
			c.info = nil
			c.mu.Unlock()
			continue
		}
		if err == nil && answer.Ruling == nil {
			err = stack.Wrap(errors.New("audiofloor: decide: the answer has no ruling"))
		}
		return answer, err
	}
}

type toolJSON struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type agentJSON struct {
	Instructions string     `json:"instructions"`
	Tools        []toolJSON `json:"tools,omitempty"`
}

func agentKey(agent Agent) ([sha256.Size]byte, []byte, error) {
	spec := agentJSON{Instructions: agent.Instructions}
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
	status, err := c.do(req, &registered, nil)
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

// do sends req and reads the answer's first JSON line into into, and the transcript's text
// from the second into words, when words is set.
func (c *Client) do(req *http.Request, into any, words *string) (int, error) {
	resp, err := c.http.Do(req)
	if err != nil {
		return 0, stack.Wrap(err)
	}
	defer resp.Body.Close()
	lines := json.NewDecoder(io.LimitReader(resp.Body, responseLimit))
	if resp.StatusCode != http.StatusOK {
		var failed struct {
			Error string `json:"error"`
		}
		_ = lines.Decode(&failed)
		err := fmt.Errorf("audiofloor: %s %s: %s: %s", req.Method, req.URL.Path, resp.Status, failed.Error)
		if resp.StatusCode == http.StatusNotImplemented {
			err = fmt.Errorf("%w: %w", ErrUnsupported, err)
		}
		return resp.StatusCode, stack.Wrap(err)
	}
	if err := lines.Decode(into); err != nil {
		return resp.StatusCode, stack.Wrap(fmt.Errorf("audiofloor: %s: %w", req.URL.Path, err))
	}
	if words == nil {
		return resp.StatusCode, nil
	}
	var said struct {
		Transcript *struct {
			Text string `json:"text"`
		} `json:"transcript"`
		Error string `json:"transcript_error"`
	}
	if err := lines.Decode(&said); err != nil {
		return resp.StatusCode, stack.Wrap(fmt.Errorf("audiofloor: transcript: %w", err))
	}
	if said.Transcript == nil {
		return resp.StatusCode, stack.Wrap(fmt.Errorf("audiofloor: transcript: %s", said.Error))
	}
	*words = said.Transcript.Text
	return resp.StatusCode, nil
}

// decisionBody is a decision's multipart body: the turn, then the audio. What a prompt
// won't show isn't sent.
func decisionBody(turn Turn, pcm []int16) ([]byte, string, error) {
	turn.Said = lastRunes(turn.Said, maxSaid)
	turn.Conversation = lastTurns(turn.Conversation, maxTurns)
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
	parts := multipart.NewWriter(&body)
	for _, part := range []struct {
		name, contentType string
		raw               []byte
	}{
		{"turn", "application/json", moment},
		{"audio", "audio/pcm;rate=" + strconv.Itoa(SampleRate) + ";channels=1;format=s16le",
			audio.PcmData{Samples: pcm, SampleRate: SampleRate, Channels: 1}.Bytes()},
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

// Ring keeps one participant's last 16 s. Write takes a frame from the edge's consumer and
// allocates nothing once it has its buffer.
type Ring struct {
	mu   sync.Mutex
	buf  []int16
	next int
	full bool
}

// Write adds samples, 16 kHz mono, dropping the oldest once the window is full.
func (r *Ring) Write(samples []int16) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.buf == nil {
		r.buf = make([]int16, windowSamples)
	}
	if len(samples) >= len(r.buf) {
		copy(r.buf, samples[len(samples)-len(r.buf):])
		r.next, r.full = 0, true
		return
	}
	n := copy(r.buf[r.next:], samples)
	if n < len(samples) {
		copy(r.buf, samples[n:])
		r.full = true
	}
	r.next += len(samples)
	if r.next >= len(r.buf) {
		r.next -= len(r.buf)
		r.full = true
	}
}

// Last returns what the window holds, oldest first, in a slice of the caller's.
func (r *Ring) Last() []int16 {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.full {
		return append([]int16(nil), r.buf[:r.next]...)
	}
	out := make([]int16, 0, len(r.buf))
	out = append(out, r.buf[r.next:]...)
	return append(out, r.buf[:r.next]...)
}
