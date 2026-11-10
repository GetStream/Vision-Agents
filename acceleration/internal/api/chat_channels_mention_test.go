package api

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// MentionSuite is which messages mention the agent of a channel that answers only its mentions
// (answersTo): the Stream user rota, named Front Desk, its config's id and name.
type MentionSuite struct {
	suite.Suite
}

func TestMentionSuite(t *testing.T) {
	suite.Run(t, new(MentionSuite))
}

// mentioning is a message.new's message with text and the mentioned_users given, raw.
func mentioning(text, users string) messageEvent {
	var event messageEvent
	event.ChannelType, event.ChannelID = "callouts", "team-1"
	event.Message.Text = text
	if users != "" {
		event.Message.MentionedUsers = json.RawMessage(users)
	}
	return event
}

func (s *MentionSuite) TestAMessageMentioningTheAgentIsOne() {
	for _, c := range []struct{ name, text, users string }{
		{"picked from the list, text without @", "can you cover my 6pm?", `[{"id": "rota", "name": "Front Desk"}]`},
		{"listed beside another user", "can you two sort it out?", `[{"id": "ben"}, {"id": "rota"}]`},
		{"@id at the start", "@rota can you cover my 6pm?", ""},
		{"@name with a space, as Stream's clients write it", "@Front Desk I'm sick, I can't come at 6", ""},
		{"@name in another case", "hey @front desk, I'm sick", ""},
		{"@id in another case", "@ROTA can you cover?", ""},
		{"@id at the end", "I'm sick, I can't come at 6 @rota", ""},
		{"@id before punctuation", "who covers 6pm, @rota?", ""},
		{"@id ending a sentence", "I'm out at 6, sorry @rota.", ""},
		{"@id in brackets", "(@rota) I'm out at 6", ""},
		{"@id after a dash", "Anya—@rota I'm out", ""},
		{"@id after another mention", "@ben @rota I'm sick", ""},
		{"an address first, the mention after", "mail anya@rota.org or @rota", ""},
		{"text mention with a list of another shape", "@rota help", `["ben"]`},
	} {
		s.True(answersTo(mentionsOnly(), mentioning(c.text, c.users)), c.name)
	}
}

func (s *MentionSuite) TestAMessageNotMentioningTheAgentIsNone() {
	for _, c := range []struct{ name, text, users string }{
		{"no mention at all", "feel better!", ""},
		{"Ben mentioned alone, by list and text", "@Ben feel better!", `[{"id": "ben", "name": "Ben"}]`},
		{"the agent's id without @", "rota can you cover?", ""},
		{"the agent's name without @", "Front Desk can you cover?", ""},
		{"an address at the agent's id", "write to a@rota", ""},
		{"an address at the agent's id with a dot", "anya.b@rota.org", ""},
		{"another user whose id starts with the agent's", "@rotabot can you cover?", ""},
		{"another user whose id is the agent's with a dash", "@rota-2 can you cover?", ""},
		{"another user whose id is the agent's with an underscore", "@rota_old can you cover?", ""},
		{"another user whose name starts with the agent's", "@Front Desks are closed", ""},
		{"the agent's name's first word alone", "@Front can you cover?", ""},
		{"@ doubled", "@@rota", ""},
		{"a lone @", "@", ""},
		{"a prefix of the id", "@rot", ""},
		{"empty text", "", ""},
		{"a list of ids as strings, which Stream does not send", "can you cover?", `["rota"]`},
		{"a list that is no list", "can you cover?", `{"id": "rota"}`},
		{"null", "can you cover?", `null`},
		{"a listed user with no id", "can you cover?", `[{"name": "rota"}]`},
	} {
		s.False(answersTo(mentionsOnly(), mentioning(c.text, c.users)), c.name)
	}
}

// A type the config answers every message in, by saying so or by saying nothing, needs no
// mention, and neither does a type it answers only mentions in for another type.
func (s *MentionSuite) TestATypeAnsweringEveryMessageNeedsNoMention() {
	for _, channels := range [][]store.ChatChannel{
		{{Type: "callouts"}},
		{{Type: "callouts", Answer: store.ChatChannelAnswerAlways}},
		{{Type: "shifts", Answer: store.ChatChannelAnswerMentioned}, {Type: "callouts"}},
	} {
		config := store.AgentConfig{ID: "rota", Name: "Front Desk", ChatChannels: channels}
		s.True(answersTo(config, mentioning("feel better!", "")), "%+v", channels)
	}
}

// A config with no name has no name to mention: a lone @ is no mention of it.
func (s *MentionSuite) TestANamelessConfigIsMentionedByItsIDAlone() {
	config := mentionsOnly()
	config.Name = ""
	s.False(answersTo(config, mentioning("hey @ there", "")))
	s.True(answersTo(config, mentioning("hey @rota there", "")))
}

// mentionsOnly is a config answering only its mentions in callouts.
func mentionsOnly() store.AgentConfig {
	return store.AgentConfig{ID: "rota", Name: "Front Desk", ChatChannels: []store.ChatChannel{
		{Type: "shifts"}, {Type: "callouts", Answer: store.ChatChannelAnswerMentioned},
	}}
}
