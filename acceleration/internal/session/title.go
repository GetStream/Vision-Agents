package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
)

// titleTarget names conversations. A fast tier: a title is a few words, and it is written
// while the person is still reading the answer it follows.
const titleTarget = "llm-fast"

// titleLimit is how many lines are read to name a conversation, from its beginning: what a
// conversation is about is what it opened with, and the lines after refine it.
const titleLimit = 40

// titleLineLimit clips one line, so a pasted log does not become the whole prompt.
const titleLineLimit = 600

const (
	titleTimeout     = 30 * time.Second
	titleTokens      = 200
	titleRunes       = 80
	descriptionRunes = 280
)

// titleEvery is how often, in answers, a conversation is renamed after its first: what it
// is about settles as it goes, and renaming it on every answer would cost a model call to
// be told the same thing.
const titleEvery = 3

// titleInstructions is what the namer is told to be.
const titleInstructions = `You name a conversation between a person and a support assistant, for a list of
conversations the person scans to find this one again.

Answer with JSON only: {"title": "...", "description": "..."}

The title is at most eight words in sentence case, with no quotes and no full stop. It names
what the person is trying to do, not that they asked: "Add push notifications to an Android
app", not "Question about notifications".

The description is one sentence of at most thirty words saying what was asked and where the
conversation got to.

Write both in the language the person wrote in.`

// naming is one session's share of the titler: where its name goes and what came before it.
type naming struct {
	titles  *titler
	service *persistent.Service
	// earlier is the conversation this session resumed, so a resumed conversation is named
	// for all of it rather than for what was said since.
	earlier []spoken
	// answered counts this session's finished answers. Only the event loop touches it.
	answered int
	// running keeps one naming per session in flight; an answer landing while one runs is
	// covered by the next.
	running atomic.Bool
}

// title is what the namer answers with.
type title struct {
	Title       string `json:"title"`
	Description string `json:"description"`
}

// titler names conversations off everybody's path, as the reviewer reviews calls: nobody
// waits for it, and a failure costs a title rather than an answer.
type titler struct {
	llm     *llmrouter.Router
	records *sessionRecorder
	logger  *slog.Logger

	ctx    context.Context
	cancel context.CancelFunc

	running sync.WaitGroup
}

// newTitler takes a nil records for a deployment with no store, which names the channel
// and not the row.
func newTitler(router *llmrouter.Router, records *sessionRecorder, logger *slog.Logger) *titler {
	ctx, cancel := context.WithCancel(context.Background())
	return &titler{llm: router, records: records, logger: logger, ctx: ctx, cancel: cancel}
}

// Close abandons the namings still running and waits for them to stop.
func (t *titler) Close() {
	t.cancel()
	t.running.Wait()
}

// name renames the conversation after its first answer and every titleEvery after that.
func (s *Session) name(event Event) {
	n := s.naming
	if n == nil {
		return
	}
	answer, ok := event.(agent.Responded)
	if !ok || answer.PendingWork || answer.Text == "" {
		return
	}
	n.answered++
	if (n.answered-1)%titleEvery != 0 || !n.running.CompareAndSwap(false, true) {
		return
	}
	said := append(append([]spoken(nil), n.earlier...), s.conversation()...)

	n.titles.running.Add(1)
	go func() {
		defer n.titles.running.Done()
		defer n.running.Store(false)

		ctx, cancel := context.WithTimeout(n.titles.ctx, titleTimeout)
		defer cancel()
		if err := s.retitle(ctx, said); err != nil {
			n.titles.logger.Warn("could not name the conversation",
				"session", s.id, "conversation", s.spec.ConversationID, "error", err)
		}
	}()
}

func (s *Session) retitle(ctx context.Context, said []spoken) error {
	n := s.naming
	named, err := n.titles.ask(ctx, s.spec, s.id, said)
	if err != nil {
		return err
	}

	s.labelMu.Lock()
	unchanged := s.title == named.Title && s.description == named.Description
	s.title, s.description = named.Title, named.Description
	s.labelMu.Unlock()
	if unchanged {
		return nil
	}

	if n.titles.records != nil {
		n.titles.records.Described(s.spec.CustomerID, s.id, named.Title, named.Description)
	}
	return n.service.Describe(ctx, s.spec.ConversationID, named.Title, named.Description)
}

func (t *titler) ask(ctx context.Context, spec Spec, id string, said []spoken) (title, error) {
	session, err := t.llm.Start(ctx, llmrouter.Request{
		CustomerID: spec.CustomerID,
		AgentID:    spec.AgentID,
		Tags:       spec.Tags,
		Target:     titleTarget,
	})
	if err != nil {
		return title{}, err
	}
	defer session.Close()

	stream, err := session.Create(ctx, llm.ResponseParams{
		ID:              id,
		Instructions:    titleInstructions,
		Input:           []llm.Message{{Role: llm.User, Content: quoted(said)}},
		MaxOutputTokens: titleTokens,
		Text:            llm.TextParams{Format: llm.FormatJSONObject},
	})
	if err != nil {
		return title{}, err
	}
	response, err := llm.Collect(stream)
	if err != nil {
		return title{}, err
	}
	return parseTitle(response.OutputText)
}

// quoted is the conversation as the namer reads it, quoted rather than replayed as turns
// for the reason conversation gives.
func quoted(said []spoken) string {
	if len(said) > titleLimit {
		said = said[:titleLimit]
	}
	var written strings.Builder
	written.WriteString("The conversation, in order:\n\n")
	for _, line := range said {
		speaker := "Person"
		if line.agent {
			speaker = "Assistant"
		}
		written.WriteString(speaker)
		written.WriteString(": ")
		written.WriteString(clip(strings.TrimSpace(line.text), titleLineLimit))
		written.WriteString("\n")
	}
	return written.String()
}

func parseTitle(answer string) (title, error) {
	var named title
	if err := json.Unmarshal([]byte(llm.Unfence(answer)), &named); err != nil {
		return title{}, fmt.Errorf("session: decode title: %w", err)
	}
	named.Title = clip(strings.TrimRight(strings.Trim(strings.TrimSpace(named.Title), `"'`), "."), titleRunes)
	named.Description = clip(strings.TrimSpace(named.Description), descriptionRunes)
	if named.Title == "" {
		return title{}, errors.New("session: the namer gave the conversation no title")
	}
	return named, nil
}

// clip shortens text to at most limit runes, marking the cut.
func clip(text string, limit int) string {
	if utf8.RuneCountInString(text) <= limit {
		return text
	}
	runes := []rune(text)
	return strings.TrimSpace(string(runes[:limit-1])) + "…"
}

// spokenOf is restored history as lines, keeping only what the person and the agent said.
func spokenOf(history []llm.Message) []spoken {
	var said []spoken
	for _, message := range history {
		if message.Content == "" || (message.Role != llm.User && message.Role != llm.Assistant) {
			continue
		}
		said = append(said, spoken{agent: message.Role == llm.Assistant, text: message.Content})
	}
	return said
}
