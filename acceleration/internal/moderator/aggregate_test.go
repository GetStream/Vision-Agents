package moderator

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

func TestAggregates(t *testing.T) {
	suite.Run(t, new(AggregateSuite))
}

// AggregateSuite counts a week of a seeded app's review queue, served in pages. alice has
// three violations, one a moderator banned for. bob was flagged four times, but two of those
// were restored, so he has two. carol, a flagged user, has one.
type AggregateSuite struct {
	suite.Suite
	stream *httptest.Server
	runner *Runner
	pages  [][]string

	mu     sync.Mutex
	bodies []map[string]any
}

func item(id, creator, cid, label string, actions ...string) string {
	return fmt.Sprintf(`{"id":%q,"entity_type":"stream:chat:v1:message","entity_id":"m-%s","entity_creator_id":%q,`+
		`"flags_count":2,"flags":[{"type":"ai","labels":[%q]}],"message":{"cid":%q},"actions":[%s]}`,
		id, id, creator, label, cid, strings.Join(actions, ","))
}

func action(kind, reporter string) string {
	return fmt.Sprintf(`{"type":%q,"reporter_type":%q}`, kind, reporter)
}

func (s *AggregateSuite) SetupTest() {
	s.bodies = nil
	s.pages = [][]string{
		{
			item("a1", "alice", "messaging:general", "harassment", action("ban", "moderator")),
			item("b1", "bob", "messaging:random", "profanity", action("delete_message", "automod"), action("restore", "moderator")),
			item("a2", "alice", "messaging:general", "harassment", action("mark_reviewed", "moderator")),
			item("b2", "bob", "messaging:random", "profanity", action("agent_recommendation", "agent_reviewer")),
		},
		{
			item("a3", "alice", "messaging:general", "profanity"),
			item("b3", "bob", "messaging:random", "profanity", action("restore", "moderator")),
			item("b4", "bob", "messaging:general", "harassment"),
			`{"id":"c1","entity_type":"stream:user","entity_id":"carol","flags_count":1,"flags":[{"type":"user","labels":["spam"]}]}`,
		},
	}
	s.stream = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		body := map[string]any{}
		_ = json.Unmarshal(raw, &body)
		s.mu.Lock()
		s.bodies = append(s.bodies, body)
		s.mu.Unlock()
		page := 0
		if next, ok := body["next"].(string); ok {
			_, _ = fmt.Sscanf(next, "page-%d", &page)
		}
		next := ""
		if page+1 < len(s.pages) {
			next = fmt.Sprintf(`,"next":"page-%d"`, page+1)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"duration":"1ms","items":[%s]%s}`, strings.Join(s.pages[page], ","), next)
	}))
	client, err := getstream.NewClient("acme-key", "acme-secret", getstream.WithBaseUrl(s.stream.URL))
	s.Require().NoError(err)
	s.runner = NewRunner(client, Tools(), &passedOn{})
}

func (s *AggregateSuite) TearDownTest() { s.stream.Close() }

// run calls a tool and reads its answer back into answer.
func (s *AggregateSuite) run(name, arguments string, answer any) {
	parts, err := s.runner.Run(context.Background(), llm.ToolCall{ID: "1", Name: name, Arguments: arguments})
	s.Require().NoError(err)
	s.Require().Len(parts, 1)
	s.Require().NoError(json.Unmarshal([]byte(parts[0].Text), answer))
}

func (s *AggregateSuite) users(arguments string) topFlaggedUsers {
	answer := topFlaggedUsers{scan: &scan{}}
	s.run(TopFlaggedUsersTool, arguments, &answer)
	return answer
}

func (s *AggregateSuite) TestTheMostViolatingUserIsCountedAcrossEveryPage() {
	answer := s.users(`{}`)

	s.Equal(8, answer.ItemsRead)
	s.False(answer.Partial)
	s.Require().Len(answer.Users, 3)
	alice := answer.Users[0]
	s.Equal("alice", alice.UserID)
	s.Equal(3, alice.Violations)
	s.Equal(3, alice.FlaggedItems)
	s.Equal(6, alice.Flags)
	s.Equal(1, alice.ConfirmedByModerator, "marking an item reviewed confirms nothing")
	s.Equal(0, alice.Overturned)
	s.Equal([]labelCount{{Label: "harassment", Count: 2}, {Label: "profanity", Count: 1}}, alice.TopLabels)
}

func (s *AggregateSuite) TestOverturnedItemsAreNotViolations() {
	answer := s.users(`{}`)

	bob := answer.Users[1]
	s.Equal("bob", bob.UserID)
	s.Equal(4, bob.FlaggedItems, "bob has the most flagged items")
	s.Equal(2, bob.Overturned)
	s.Equal(2, bob.Violations, "yet his restored items do not count against him")
	s.Equal(1, bob.NoraRecommendations)
}

func (s *AggregateSuite) TestAFlaggedUserIsTheirOwnCreator() {
	answer := s.users(`{}`)

	s.Equal("carol", answer.Users[2].UserID)
	s.Equal(1, answer.Users[2].Violations)
}

func (s *AggregateSuite) TestALabelCountsOnlyItsItems() {
	answer := s.users(`{"label":"Profanity"}`)

	s.Equal("Profanity", answer.Label)
	s.Require().Len(answer.Users, 2)
	s.Equal("bob", answer.Users[0].UserID)
	s.Equal(1, answer.Users[0].Violations)
	s.Equal("alice", answer.Users[1].UserID)
	s.Equal(1, answer.Users[1].Violations)
}

func (s *AggregateSuite) TestTheLimitCutsTheRanking() {
	answer := s.users(`{"limit":1}`)

	s.Require().Len(answer.Users, 1)
	s.Equal("alice", answer.Users[0].UserID)
}

func (s *AggregateSuite) TestTheWindowIsTheLastWeekNewestFirst() {
	before := time.Now().UTC().AddDate(0, 0, -7)

	answer := s.users(`{"entity_type":"stream:chat:v1:message"}`)

	s.Equal(7, answer.Window.Days)
	s.WithinDuration(before, answer.Window.From, time.Minute)
	s.Require().Len(s.bodies, 2)
	filter := s.bodies[0]["filter"].(map[string]any)
	s.Equal(answer.Window.From.Format(time.RFC3339), filter["created_at"].(map[string]any)["$gte"])
	s.Equal("stream:chat:v1:message", filter["entity_type"])
	s.Equal([]any{map[string]any{"field": "created_at", "direction": float64(-1)}}, s.bodies[0]["sort"])
	s.NotContains(s.bodies[0], "lock_items")
	s.Equal("page-1", s.bodies[1]["next"])
}

func (s *AggregateSuite) TestAWindowPastTheCapIsCut() {
	answer := s.users(`{"days":365}`)

	s.Equal(maxDays, answer.Window.Days)
}

func (s *AggregateSuite) TestAQueueLongerThanThePageCapIsSaidToBePartial() {
	for len(s.pages) <= maxPages {
		s.pages = append(s.pages, s.pages[0])
	}

	answer := s.users(`{}`)

	s.True(answer.Partial)
	s.Len(s.bodies, maxPages)
	s.Equal(maxPages*4, answer.ItemsRead)
	s.Contains(answer.Note, "lower bound")
}

func (s *AggregateSuite) TestTheSummaryCountsByLabelActionAndChannel() {
	answer := moderationSummary{scan: &scan{}}

	s.run(ModerationSummaryTool, `{}`, &answer)

	s.Equal(8, answer.ItemsRead)
	s.Equal(2, answer.Overturned)
	s.Equal([]labelRow{
		{Label: "profanity", Items: 4, Overturned: 2},
		{Label: "harassment", Items: 3, Overturned: 0},
		{Label: "spam", Items: 1, Overturned: 0},
	}, answer.ByLabel)
	s.Equal([]actionRow{
		{Action: "restore", Count: 2, ByModerator: 2},
		{Action: "agent_recommendation", Count: 1, ByModerator: 0},
		{Action: "ban", Count: 1, ByModerator: 1},
		{Action: "delete_message", Count: 1, ByModerator: 0},
		{Action: "mark_reviewed", Count: 1, ByModerator: 1},
	}, answer.ByAction)
	s.Equal([]channelRow{
		{ChannelCID: "messaging:general", Items: 4, Overturned: 0},
		{ChannelCID: "messaging:random", Items: 3, Overturned: 2},
	}, answer.ByChannel)
}

func (s *AggregateSuite) TestAStreamFailureIsTheToolsError() {
	s.stream.Close()

	_, err := s.runner.Run(context.Background(), llm.ToolCall{ID: "1", Name: ModerationSummaryTool})

	s.ErrorContains(err, ModerationSummaryTool)
}
