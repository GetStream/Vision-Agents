package agent

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

type GivenNamesSuite struct{ suite.Suite }

func TestGivenNames(t *testing.T) { suite.Run(t, new(GivenNamesSuite)) }

func (s *GivenNamesSuite) TestAHotelTheCallerNamedIsNotRefusedAsAPersonsPlaceholder() {
	history := []llm.Message{
		{Role: llm.User, Content: "Checking in February third, checking out the fifth, two nights. The Manhattan House sounds good. Standard room."},
		{Role: llm.User, Content: "Yes, book it under my name."},
	}
	_, refused := placeholderName(history, llm.ToolCall{Name: "book_hotel", Arguments: `{"hotel_name":"The Manhattan House"}`})
	s.False(refused)
}

func (s *GivenNamesSuite) TestANameReturnedByAToolIsPreserved() {
	history := []llm.Message{{Role: llm.ToolResult, Content: `{"hotels":[{"name":"The Manhattan House"}]}`}}
	_, refused := placeholderName(history, llm.ToolCall{Name: "book_hotel", Arguments: `{"hotel_name":"The Manhattan House"}`})
	s.False(refused)
}

func (s *GivenNamesSuite) TestALongNameTheCallerGaveIsPreserved() {
	history := []llm.Message{{Role: llm.User, Content: "My name is Maria del Carmen de la Cruz."}}
	_, refused := placeholderName(history, llm.ToolCall{Name: "book_room", Arguments: `{"name":"Maria del Carmen de la Cruz"}`})
	s.False(refused)
}

func (s *GivenNamesSuite) TestAnAssistantInventingANameDoesNotGroundIt() {
	history := []llm.Message{{Role: llm.Assistant, Content: "I will book it for John Doe."}}
	_, refused := placeholderName(history, llm.ToolCall{Name: "book_room", Arguments: `{"name":"John Doe"}`})
	s.True(refused)
}

func (s *GivenNamesSuite) TestARealNameIsAllowedEvenWhenItLooksLikeAPlaceholder() {
	history := []llm.Message{{Role: llm.User, Content: "My name is John Doe."}}
	_, refused := placeholderName(history, llm.ToolCall{Name: "book_room", Arguments: `{"name":"John Doe"}`})
	s.False(refused)
}

func (s *GivenNamesSuite) TestARefusalQuotingAPlaceholderDoesNotGroundIt() {
	history := []llm.Message{{Role: llm.ToolResult, Content: placeholderResult("John Doe")}}
	_, refused := placeholderName(history, llm.ToolCall{Name: "book_room", Arguments: `{"name":"John Doe"}`})
	s.True(refused)
}

func (s *GivenNamesSuite) TestAPartialWordDoesNotGroundAPlaceholder() {
	history := []llm.Message{{Role: llm.User, Content: "Ask John Doerr."}}
	_, refused := placeholderName(history, llm.ToolCall{Name: "book_room", Arguments: `{"name":"John Doe"}`})
	s.True(refused)
}

func (s *GivenNamesSuite) TestSeparateToolValuesDoNotMakeAName() {
	history := []llm.Message{{Role: llm.ToolResult, Content: `["John", "Doe"]`}}
	_, refused := placeholderName(history, llm.ToolCall{Name: "book_room", Arguments: `{"name":"John Doe"}`})
	s.True(refused)
}

func (s *GivenNamesSuite) TestAGivenHotelDoesNotExcuseAMissingGuestName() {
	history := []llm.Message{{Role: llm.User, Content: "The Manhattan House sounds good."}}
	name, refused := placeholderName(history, llm.ToolCall{Name: "book_hotel", Arguments: `{"hotel_name":"The Manhattan House","guest_name":"John Doe"}`})
	s.True(refused)
	s.Equal("John Doe", name)
}

func (s *GivenNamesSuite) TestAStandInNameTheAgentWroteIsNotGroundedByAToolReadingItBack() {
	// A tool that reads back what the agent said or posted returns the name the agent wrote (AI-2886).
	said := llm.Message{Role: llm.Assistant, Content: "I will book it for John Doe."}
	posted := llm.Message{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "s1", Name: "send_message", Arguments: `{"text":"New booking: John Doe, 2 nights"}`}}}
	read := func(result string) []llm.Message {
		return []llm.Message{
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "r1", Name: "read_notes", Arguments: `{}`}}},
			{Role: llm.ToolResult, ToolCallID: "r1", Content: result},
		}
	}
	send := func(text string) llm.Message {
		arguments, err := json.Marshal(map[string]string{"text": text})
		s.Require().NoError(err)
		return llm.Message{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "s1", Name: "send_message", Arguments: string(arguments)}}}
	}
	given := llm.Message{Role: llm.ToolResult, Content: `{"guest":"John Doe","room":"204"}`}
	for _, c := range []struct {
		name    string
		history []llm.Message
		refused bool
	}{
		{"said, read back", append([]llm.Message{said}, read(`{"notes":"Agent: I will book it for John Doe."}`)...), true},
		{"posted, read back as text", append([]llm.Message{posted}, read("[bot] New booking: John Doe, 2 nights\n[ana] ok")...), true},
		{"said, read back on another line than a tool's", append([]llm.Message{said}, read(`{"notes":"Agent: I will book it for John Doe.\nFront desk: John Doe checked in"}`)...), false},
		{"given by a tool before the agent posted a copy of its line", append(append([]llm.Message{}, read(`{"note":"Guest John Doe is in room 5"}`)...), llm.Message{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "s2", Name: "send_message", Arguments: `{"text":"Guest John Doe"}`}}}), false},
		{"given by a tool before the agent said it", append([]llm.Message{given, said}, read(`{"notes":"Agent: I will book it for John Doe."}`)...), false},
		{"a name of the tool's where the agent's line split it inside words", append([]llm.Message{send("Room for Joh N. Doe")}, read("Room for John Doe")...), false},
		{"a name of the tool's where the agent's line held its letters across words", append([]llm.Message{send("Guest Jo Hndoe")}, read(`{"text":"Guest John Doe"}`)...), false},
		{"a name of the tool's where the agent's line ran its words together", append([]llm.Message{send("Ask Johndoe today")}, read("Ask John Doe today")...), false},
		{"a name the agent posted in capitals with a wider gap, read back", append([]llm.Message{send("Guest: JOHN  DOE")}, read("Guest: John Doe")...), true},
		{"a name of the tool's where the agent's line had its letters in a longer word", append([]llm.Message{send("John Doerr")}, read("Guest John Doe rr")...), false},
		{"a name of the tool's beside an agent's line with more letters in its last word", append([]llm.Message{send("Ask John Doerr")}, read("Ask John Doerr; guest John Doe")...), false},
	} {
		_, refused := placeholderName(c.history, llm.ToolCall{Name: "book_room", Arguments: `{"name":"John Doe"}`})
		s.Equal(c.refused, refused, c.name)
	}
}

func (s *GivenNamesSuite) TestAToolGivesANameInAValueWholeNotInAKey() {
	// A key names what a value is, it gives no name: a stand-in "User" is not given by a "user"
	// field. A value gives the name whole, wherever its words break.
	result := func(call, content string) []llm.Message {
		return []llm.Message{
			{Role: llm.User, Content: "Can you sort this out for me?"},
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "t1", Name: call, Arguments: `{}`}}},
			{Role: llm.ToolResult, ToolCallID: "t1", Content: content},
		}
	}
	for _, c := range []struct {
		name      string
		history   []llm.Message
		arguments string
		refused   bool
	}{
		{"a user key", result("read_channel", `{"messages":[{"user":"U0BOB0001","text":"printer is down"}]}`), `{"name":"User"}`, true},
		{"a customer key", result("get_order", `{"order_id":"A1","customer":{"id":"42"}}`), `{"customer_name":"Customer"}`, true},
		{"a guest key with no value", result("availability", `{"rooms":[{"guest":null,"number":"12"}]}`), `{"guest_name":"Guest"}`, true},
		{"a caller key", result("call_info", `{"caller":"+15550100199"}`), `{"name":"Caller"}`, true},
		{"a key that is the name", result("guests", `{"John Doe":{"room":"12"}}`), `{"name":"John Doe"}`, true},
		{"a name broken across lines in one value", result("guests", `{"guest":"John\nDoe"}`), `{"name":"John Doe"}`, false},
		{"a name in a value", result("guests", `{"guest":"John Doe"}`), `{"name":"John Doe"}`, false},
	} {
		_, refused := placeholderName(c.history, llm.ToolCall{Name: "create_ticket", Arguments: c.arguments})
		s.Equal(c.refused, refused, c.name)
	}
}
