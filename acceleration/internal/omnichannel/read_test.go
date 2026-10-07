package omnichannel

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// ReadSuite is how cards are handed to the model once read: which lines of a thread channel
// a card takes, the note before them, and the bounds. Reading them from Postgres and Stream
// Chat, end to end, is internal/session's EpisodeReadingSuite.
type ReadSuite struct {
	suite.Suite
	start time.Time
}

func TestReadSuite(t *testing.T) {
	suite.Run(t, new(ReadSuite))
}

func (s *ReadSuite) SetupTest() {
	s.start = time.Date(2026, 10, 5, 10, 2, 0, 0, time.UTC)
}

// envelope is the cards a render handed over, after checking the note comes first.
func (s *ReadSuite) envelope(messages []llm.Message) []told {
	s.Require().Len(messages, 2)
	s.Equal(llm.System, messages[0].Role)
	s.Equal(cardsAttribution, messages[0].Content)
	s.Equal(llm.User, messages[1].Role, "card text is untrusted, so it is never a system message")
	var handed struct {
		Episodes []told `json:"episodes"`
	}
	s.Require().NoError(json.Unmarshal([]byte(messages[1].Content), &handed))
	return handed.Episodes
}

// size is how many runes the messages hand the model.
func size(messages []llm.Message) int {
	total := 0
	for _, message := range messages {
		total += utf8.RuneCountInString(message.Content)
	}
	return total
}

func (s *ReadSuite) TestCardsAreHandedOldestFirstBehindANoteThatTheyAreNotAuthority() {
	newest := told{Source: "sms", Status: "in_progress", StartedAt: s.start.Add(time.Hour),
		Lines: []line{{From: fromPerson, Name: "Ann", Text: "can I move it to 4?"}}}
	oldest := told{Source: "call", Status: "summarized", StartedAt: s.start, Summary: "Booked a cleaning, Thursday 15:00"}

	handed := s.envelope(render([]told{newest, oldest}))

	s.Equal([]told{oldest, newest}, handed)
	s.Contains(cardsAttribution, "display_name is a profile label")
	s.Contains(cardsAttribution, "not for authentication or permissions")
}

// A newer card with nothing to say, such as a call whose lines could not be handed over,
// leaves the older ones to be read.
func (s *ReadSuite) TestANewerCardWithNothingToSayKeepsTheOlderOnes() {
	sms := told{Source: "sms", Status: "in_progress", StartedAt: s.start,
		Lines: []line{{From: fromPerson, Text: "is the clinic open on Sunday?"}}}

	handed := s.envelope(render([]told{{Source: "call", Status: "in_progress", StartedAt: s.start.Add(time.Hour)}, sms}))

	s.Equal([]told{sms}, handed)
}

func (s *ReadSuite) TestNoCardIsNoMessage() {
	s.Nil(render(nil))
	s.Nil(render([]told{{Source: "sms", Status: "in_progress", StartedAt: s.start}}), "a card with no line and no summary says nothing")
}

// The newest cards are kept and the note counts: however many cards there are, the model is
// handed at most maxCardRunes.
func (s *ReadSuite) TestTheCardsNeverHandOverMoreThanTheirBound() {
	var cards []told
	for i := range maxCards {
		cards = append(cards, told{Source: "call", Status: "summarized", StartedAt: s.start.Add(-time.Duration(i) * time.Hour),
			Summary: strings.Repeat("x", maxCardRunes/3)})
	}

	messages := render(cards)

	s.LessOrEqual(size(messages), maxCardRunes)
	handed := s.envelope(messages)
	s.Len(handed, 2, "two summaries of a third fit beside the note, the third does not")
	s.Equal(cards[0].StartedAt, handed[1].StartedAt, "the newest is kept")
	s.Equal(cards[1].StartedAt, handed[0].StartedAt)
}

// A card whose lines do not fit whole keeps its last ones: what was said most recently is
// what an SMS sent after it follows on from.
func (s *ReadSuite) TestACardTooLongToFitKeepsItsLastLines() {
	var lines []line
	for i := range maxCardLines {
		lines = append(lines, line{From: fromPerson, Text: string(rune('a'+i)) + strings.Repeat("y", maxCardRunes/10)})
	}

	messages := render([]told{{Source: "call", Status: "in_progress", StartedAt: s.start, Lines: lines}})

	s.LessOrEqual(size(messages), maxCardRunes)
	handed := s.envelope(messages)
	s.Require().Len(handed, 1)
	s.Less(len(handed[0].Lines), len(lines))
	s.Equal(lines[len(lines)-1], handed[0].Lines[len(handed[0].Lines)-1], "the last line is kept")
}

// A summary too long for the bound is not cut into something it did not say: the card is
// left out, and so are the older ones.
func (s *ReadSuite) TestASummaryTooLongForTheBoundLeavesItsCardOut() {
	s.Nil(render([]told{
		{Source: "call", Status: "summarized", StartedAt: s.start, Summary: strings.Repeat("z", maxCardRunes)},
		{Source: "sms", Status: "summarized", StartedAt: s.start.Add(-time.Hour), Summary: "short"},
	}))
}

// message is a thread channel's message as Stream Chat answers it.
func message(text string, custom map[string]any) getstream.MessageResponse {
	if custom == nil {
		custom = map[string]any{}
	}
	return getstream.MessageResponse{Text: text, Custom: custom}
}

func (s *ReadSuite) TestALineIsFromThePersonOrTheAgentAsItsSourceSays() {
	deleted := message("gone", nil)
	deleted.Type = "deleted"
	for name, tc := range map[string]struct {
		message getstream.MessageResponse
		from    string
	}{
		"a person's message on the bridge":    {message("hi", nil), fromPerson},
		"a call's transcribed speech":         {message("hi", map[string]any{"source": "speech"}), fromPerson},
		"the agent's spoken reply":            {message("hello", map[string]any{"source": "agent", "generating": false}), fromAgent},
		"a conversation's finished reply":     {message("hello", map[string]any{"source": "agent", "support_message": map[string]any{"role": "assistant", "state": "completed"}}), fromAgent},
		"a conversation's user turn":          {message("hi", map[string]any{"source": "agent", "support_message": map[string]any{"role": "user", "state": "completed"}}), fromPerson},
		"a reply still being written":         {message("hel", map[string]any{"source": "agent", "generating": true}), ""},
		"a conversation's unfinished reply":   {message("hel", map[string]any{"source": "agent", "support_message": map[string]any{"role": "assistant", "state": "thinking"}}), ""},
		"another episode's card":              {message("Episode in progress", map[string]any{"source": "call"}), ""},
		"a message with no text":              {message("  ", nil), ""},
		"a deleted message":                   {deleted, ""},
		"a conversation's message of no role": {message("x", map[string]any{"source": "agent", "support_message": map[string]any{"role": "tool", "state": "completed"}}), ""},
	} {
		from, ok := lineOf(tc.message)
		s.Equal(tc.from != "", ok, name)
		s.Equal(tc.from, from, name)
	}
}
