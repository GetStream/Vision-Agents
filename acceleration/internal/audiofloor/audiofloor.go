// Package audiofloor asks an audio floor model about a call: one speech model that hears the
// caller's last 16 s and answers the question the flow controller answers from the words,
// says which of the agent's tools is wanted, and can write what was said. It is served by an
// audio floor server, which speaks this protocol:
//
//	GET  /v1/info             what the server serves: its prompt template, whether a prompt
//	                          with the floor free says the conversation, the most tools an
//	                          agent can have, the audio a decision hears, and end_of_turn,
//	                          the policy fitted to its model for ending a turn
//	POST /v1/agents           {"instructions","tools","languages"}: registers an agent by
//	                          its content and answers {"agent": ID}. More tools than the
//	                          model has answers for is a 422
//	POST /v1/decide?agent=ID  a multipart body: a "turn" part, the moment of the call as
//	                          JSON, then an "audio" part, Ogg Opus or 16 kHz mono PCM. The
//	                          answer is every option's probability, p_eot = 1 - p(wait)
//	                          with the floor free, and the ruling to act on once the turn
//	                          has ended. With transcript=true a second line follows with
//	                          what was said, empty when the server heard noise
//
// The server writes the model's prompt, because the model knows only the words it was
// trained on. The server says what to do with the words; whether a turn with the floor free
// has ended is the router's to say, from p_eot, the pause and the policy. A server started
// without a policy for its model answers 501 to all three.
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
	caller       = "caller"
	anotherVoice = "another_voice"
)

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

// Agent is what an agent was told and what it can do, and the languages its callers speak,
// the only ones the server writes their words in.
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
	// Transcript is what was said, when asked for: empty when nothing was, or when the
	// server heard noise.
	Transcript string `json:"-"`
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
	if _, err := c.do(req, &info, nil); err != nil {
		return Policy{}, err
	}
	if p := info.EndOfTurn; p.FirstBar <= 0 || p.LaterBar <= 0 || p.FirstCheckMs <= 0 || p.IntervalMs <= 0 || p.CapMs <= 0 {
		return Policy{}, stack.Wrap(fmt.Errorf("%w: its end_of_turn is incomplete", ErrUnsupported))
	}
	c.mu.Lock()
	c.policy = &info.EndOfTurn
	c.mu.Unlock()
	return info.EndOfTurn, nil
}

// Decide asks about audio under the moment turn, and for what was said in it when
// transcript is set. The agent is registered first if it is new, and again if the server
// has forgotten it.
func (c *Client) Decide(ctx context.Context, agent Agent, turn Turn, audio harness.Audio, transcript bool) (Answer, error) {
	if len(audio.Data) == 0 {
		return Answer{}, stack.Wrap(errors.New("audiofloor: no audio heard yet"))
	}
	body, contentType, err := decisionBody(turn, audio)
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
			c.policy = nil
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
