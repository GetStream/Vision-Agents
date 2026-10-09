package session

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ArgumentShapeSuite is the shape an invocation row keeps of a call's arguments (AI-990 F40):
// names, JSON types and lengths, never a value.
type ArgumentShapeSuite struct {
	suite.Suite
}

func TestArgumentShapeSuite(t *testing.T) {
	suite.Run(t, new(ArgumentShapeSuite))
}

func (s *ArgumentShapeSuite) TestEachArgumentIsItsNameTypeAndLengthSortedByName() {
	shape := argumentShape(`{"thread_ts": "", "text": "héllo", "channels": ["C1", "C2"], "limit": 20,
		"unfurl": false, "filter": {"secret": "x"}, "cursor": null}`)

	s.Equal([]store.ArgumentShape{
		{Name: "channels", Type: "array", Length: ptrTo(2)},
		{Name: "cursor", Type: "null"},
		{Name: "filter", Type: "object"},
		{Name: "limit", Type: "number"},
		{Name: "text", Type: "string", Length: ptrTo(5)},
		{Name: "thread_ts", Type: "string", Length: ptrTo(0)},
		{Name: "unfurl", Type: "boolean"},
	}, shape)
}

func (s *ArgumentShapeSuite) TestNoValueIsKept() {
	shape := argumentShape(`{"text": "the secret plan", "filter": {"query": "hidden"}, "ids": ["kept-out"]}`)

	for _, argument := range shape {
		s.NotContains(argument.Name+argument.Type, "secret")
		s.NotContains(argument.Name+argument.Type, "hidden")
		s.NotContains(argument.Name+argument.Type, "kept-out")
	}
}

// Empty arguments are the empty object, as the MCP source sends them.
func (s *ArgumentShapeSuite) TestEmptyArgumentsAreAnEmptyShape() {
	for _, arguments := range []string{"", "  ", "{}"} {
		shape := argumentShape(arguments)
		s.NotNil(shape, "%q", arguments)
		s.Empty(shape, "%q", arguments)
	}
}

func (s *ArgumentShapeSuite) TestArgumentsThatAreNotAnObjectHaveNoShape() {
	for _, arguments := range []string{"not json", "[1, 2]", `"text"`, "null", `{"a": `} {
		s.Nil(argumentShape(arguments), "%q", arguments)
	}
}

func ptrTo(n int) *int {
	return &n
}
