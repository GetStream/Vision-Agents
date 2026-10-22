package moderator

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

func TestTools(t *testing.T) {
	suite.Run(t, new(ToolsSuite))
}

// The two pages list_agent_feedback walks when the fake Stream is left on its default.
const (
	feedbackPageOne = `{"feedback":[{"id":"fb-1","item_id":"item-1","agent_review_id":"rev-1","nora_action":"remove","moderator_action":"keep","reason":"the phrase is allowed in this app","improvement":"do not remove this phrase when it targets nobody","labels":["profanity"],"entity_type":"stream:chat:v1:message","content_excerpt":"This market is fucking wild.","created_by":"mod-1","created_at":"2026-10-01T00:00:00Z"}],"next":"page-2"}`
	feedbackPageTwo = `{"feedback":[{"id":"fb-2","item_id":"item-2","agent_review_id":"rev-2","nora_action":"flag","moderator_action":"keep","reason":"quoted from a headline","improvement":"allow a quoted headline","labels":["harassment"],"entity_type":"stream:chat:v1:message","content_excerpt":"you idiot","created_by":"mod-2","created_at":"2026-10-02T00:00:00Z"}],"next":""}`
)

// ToolsSuite runs the tools against a Stream that answers every moderation read and keeps
// what it was asked.
type ToolsSuite struct {
	suite.Suite
	stream *httptest.Server
	runner *Runner
	next   *passedOn

	mu        sync.Mutex
	asked     []string
	queries   []string
	authTypes []string
	bodies    []map[string]any
	forever   atomic.Bool
}

// passedOn is the runner a call that is not a moderation tool reaches.
type passedOn struct{ names []string }

func (p *passedOn) Run(_ context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	p.names = append(p.names, call.Name)
	return llm.TextParts("from the caller"), nil
}

func (s *ToolsSuite) SetupTest() {
	s.asked, s.queries, s.authTypes, s.bodies = nil, nil, nil, nil
	s.forever.Store(false)
	s.stream = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		body := map[string]any{}
		_ = json.Unmarshal(raw, &body)
		s.mu.Lock()
		s.asked = append(s.asked, r.Method+" "+r.URL.Path)
		s.queries = append(s.queries, r.URL.RawQuery)
		s.authTypes = append(s.authTypes, r.Header.Get("Stream-Auth-Type"))
		s.bodies = append(s.bodies, body)
		s.mu.Unlock()
		if r.URL.Path == "/api/v2/moderation/agent_feedback" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, s.feedbackPage(r))
			return
		}
		answers := map[string]string{
			"/api/v2/moderation/review_queue":        `{"duration":"1ms","items":[{"id":"item-1","status":"pending"}],"next":"page-2"}`,
			"/api/v2/moderation/review_queue/item-1": `{"duration":"1ms","item":{"id":"item-1","flags":[{"reason":"spam"}],"actions":[{"type":"mark_reviewed"}]}}`,
			"/api/v2/moderation/configs":             `{"duration":"1ms","configs":[{"key":"chat:messaging","llm_config":{"rules":[{"label":"HARASSMENT"}]}}]}`,
			"/api/v2/moderation/moderation_rules":    `{"duration":"1ms","rules":[{"id":"rule-1","name":"Repeat offenders"}]}`,
			"/api/v2/blocklists":                     `{"duration":"1ms","blocklists":[{"name":"profanity","words":["darn"]}]}`,
			"/api/v2/moderation/logs":                `{"duration":"1ms","logs":[{"id":"log-1","type":"ban"}]}`,
		}
		answer, ok := answers[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, answer)
	}))
	client, err := getstream.NewClient("acme-key", "acme-secret", getstream.WithBaseUrl(s.stream.URL))
	s.Require().NoError(err)
	s.next = &passedOn{}
	s.runner = NewRunner(client, Tools(), s.next)
}

func (s *ToolsSuite) TearDownTest() { s.stream.Close() }

// feedbackPage is the agent_feedback body. The default is two pages. forever keeps handing
// back a new cursor so a walk with no cap cannot pass by luck.
func (s *ToolsSuite) feedbackPage(r *http.Request) string {
	if s.forever.Load() {
		n := 1
		if next := r.URL.Query().Get("next"); next != "" {
			parsed, err := strconv.Atoi(strings.TrimPrefix(next, "page-"))
			if err == nil && parsed > 0 {
				n = parsed
			}
		}
		if n > 40 {
			return `{"feedback":[],"next":""}`
		}
		id := strconv.Itoa(n)
		return `{"feedback":[{"id":"fb-` + id + `","nora_action":"remove","moderator_action":"keep","reason":"allowed","improvement":"allow it","labels":["profanity"],"content_excerpt":"phrase"}],"next":"page-` + strconv.Itoa(n+1) + `"}`
	}
	switch r.URL.Query().Get("next") {
	case "":
		return feedbackPageOne
	case "page-2":
		return feedbackPageTwo
	default:
		return `{"feedback":[],"next":""}`
	}
}

// run calls a tool and reads its answer back as JSON.
func (s *ToolsSuite) run(name, arguments string) map[string]any {
	parts, err := s.runner.Run(context.Background(), llm.ToolCall{ID: "1", Name: name, Arguments: arguments})
	s.Require().NoError(err)
	s.Require().Len(parts, 1)
	answer := map[string]any{}
	s.Require().NoError(json.Unmarshal([]byte(parts[0].Text), &answer))
	return answer
}

func (s *ToolsSuite) TestTheReviewQueueIsQueriedWithTheModelsFilterAndNeverLocked() {
	answer := s.run(QueryReviewQueueTool, `{"filter":{"status":"pending"},"limit":5,"next":"page-1"}`)

	s.Equal("page-2", answer["next"])
	s.Require().Equal([]string{"POST /api/v2/moderation/review_queue"}, s.asked)
	s.Equal(map[string]any{"status": "pending"}, s.bodies[0]["filter"])
	s.EqualValues(5, s.bodies[0]["limit"])
	s.Equal("page-1", s.bodies[0]["next"])
	s.NotContains(s.bodies[0], "lock_items", "reading the queue takes no item from a person reviewing it")
}

func (s *ToolsSuite) TestALimitPastTheSchemasIsCut() {
	s.run(QueryModerationLogsTool, `{"limit":500}`)

	s.EqualValues(maxLimit, s.bodies[0]["limit"])
}

func (s *ToolsSuite) TestAnItemIsReadWithItsFlagsAndActions() {
	answer := s.run(GetReviewQueueItemTool, `{"id":"item-1"}`)

	item := answer["item"].(map[string]any)
	s.Len(item["flags"], 1)
	s.Len(item["actions"], 1)
	s.Equal([]string{"GET /api/v2/moderation/review_queue/item-1"}, s.asked)
}

func (s *ToolsSuite) TestAnItemWithoutAnIdIsRefusedBeforeStreamIsAsked() {
	_, err := s.runner.Run(context.Background(), llm.ToolCall{ID: "1", Name: GetReviewQueueItemTool, Arguments: `{}`})

	s.Error(err)
	s.Empty(s.asked)
}

func (s *ToolsSuite) TestTheConfigsCarryTheirLLMLabels() {
	answer := s.run(QueryModerationConfigsTool, `{"filter":{"key":"chat:messaging"}}`)

	configs := answer["configs"].([]any)
	s.Require().Len(configs, 1)
	s.Contains(configs[0], "llm_config")
	s.Equal(map[string]any{"key": "chat:messaging"}, s.bodies[0]["filter"])
}

func (s *ToolsSuite) TestTheRulesAreRead() {
	answer := s.run(QueryModerationRulesTool, ``)

	s.Len(answer["rules"], 1)
	s.Equal([]string{"POST /api/v2/moderation/moderation_rules"}, s.asked)
}

func (s *ToolsSuite) TestTheBlocklistsAreReadWithTheirWords() {
	answer := s.run(ListBlocklistsTool, `{}`)

	blocklists := answer["blocklists"].([]any)
	s.Require().Len(blocklists, 1)
	s.Equal([]any{"darn"}, blocklists[0].(map[string]any)["words"])
}

func (s *ToolsSuite) TestTheLogsAreQueriedWithTheModelsSort() {
	answer := s.run(QueryModerationLogsTool, `{"sort":[{"field":"created_at","direction":-1}]}`)

	s.Len(answer["logs"], 1)
	s.Equal([]any{map[string]any{"field": "created_at", "direction": float64(-1)}}, s.bodies[0]["sort"])
}

func (s *ToolsSuite) TestAStreamFailureIsTheToolsError() {
	s.stream.Close()

	_, err := s.runner.Run(context.Background(), llm.ToolCall{ID: "1", Name: QueryModerationLogsTool})

	s.ErrorContains(err, QueryModerationLogsTool)
}

func (s *ToolsSuite) TestModeratorFeedbackIsListedAcrossPages() {
	answer := s.run(ListAgentFeedbackTool, `{}`)

	feedback := answer["feedback"].([]any)
	s.Require().Len(feedback, 2)
	first := feedback[0].(map[string]any)
	s.Equal("fb-1", first["id"])
	s.Equal("This market is fucking wild.", first["content_excerpt"])
	s.Equal("remove", first["nora_action"])
	s.Equal("keep", first["moderator_action"])
	s.Equal("the phrase is allowed in this app", first["reason"])
	s.Equal("do not remove this phrase when it targets nobody", first["improvement"])
	s.Equal([]any{"profanity"}, first["labels"])
	s.Equal("fb-2", feedback[1].(map[string]any)["id"])
	s.Equal([]string{
		"GET /api/v2/moderation/agent_feedback",
		"GET /api/v2/moderation/agent_feedback",
	}, s.asked)
	s.Contains(s.queries[0], "limit=25")
	s.Contains(s.queries[0], "api_key=acme-key")
	s.NotContains(s.queries[0], "next=")
	s.Contains(s.queries[1], "next=page-2")
	s.Contains(s.queries[1], "limit=24")
	s.Equal([]string{"jwt", "jwt"}, s.authTypes)
	s.NotContains(answer, "next")
}

func (s *ToolsSuite) TestModeratorFeedbackStopsAtTheAskedLimit() {
	answer := s.run(ListAgentFeedbackTool, `{"limit":1}`)

	s.Len(answer["feedback"].([]any), 1)
	s.Equal([]string{"GET /api/v2/moderation/agent_feedback"}, s.asked)
	s.Contains(s.queries[0], "limit=1")
}

func (s *ToolsSuite) TestModeratorFeedbackStopsWhenTheCursorNeverEnds() {
	s.forever.Store(true)

	answer := s.run(ListAgentFeedbackTool, `{"limit":25}`)

	s.Len(answer["feedback"].([]any), maxLimit)
	s.Len(s.asked, maxLimit)
}

func (s *ToolsSuite) TestAModeratorFeedbackFailureIsTheToolsError() {
	s.stream.Close()

	_, err := s.runner.Run(context.Background(), llm.ToolCall{ID: "1", Name: ListAgentFeedbackTool})

	s.ErrorContains(err, ListAgentFeedbackTool)
}

func (s *ToolsSuite) TestAnyOtherToolIsPassedOn() {
	parts, err := s.runner.Run(context.Background(), llm.ToolCall{ID: "1", Name: "search_docs"})

	s.Require().NoError(err)
	s.Equal("from the caller", parts[0].Text)
	s.Equal([]string{"search_docs"}, s.next.names)
	s.Empty(s.asked)
}
