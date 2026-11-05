package agent

import (
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
	given := llm.Message{Role: llm.ToolResult, Content: `{"guest":"John Doe","room":"204"}`}
	for _, c := range []struct {
		name    string
		history []llm.Message
		refused bool
	}{
		{"said, read back", append([]llm.Message{said}, read(`{"notes":"Agent: I will book it for John Doe."}`)...), true},
		{"posted, read back as text", append([]llm.Message{posted}, read("[bot] New booking: John Doe, 2 nights\n[ana] ok")...), true},
		{"said, read back on another line than a tool's", append([]llm.Message{said}, read(`{"notes":"Agent: I will book it for John Doe.\nFront desk: John Doe checked in"}`)...), false},
		{"given by a tool before the agent said it", append([]llm.Message{given, said}, read(`{"notes":"Agent: I will book it for John Doe."}`)...), false},
	} {
		_, refused := placeholderName(c.history, llm.ToolCall{Name: "book_room", Arguments: `{"name":"John Doe"}`})
		s.Equal(c.refused, refused, c.name)
	}
}
