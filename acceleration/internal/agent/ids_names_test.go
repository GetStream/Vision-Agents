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
