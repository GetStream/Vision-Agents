package api

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

type RespondFormatSuite struct {
	suite.Suite
}

func TestRespondFormatSuite(t *testing.T) {
	suite.Run(t, new(RespondFormatSuite))
}

// format is the text format respond.params resolves from a frame's format and the held one.
func (s *RespondFormatSuite) format(frame, held string) llm.TextFormat {
	params, err := respond{Format: frame}.params(options.LLM{Format: held})
	s.Require().NoError(err)
	s.Nil(params.Text.Schema)
	return params.Text.Format
}

func (s *RespondFormatSuite) TestAFrameNamingJSONObjectAsksForJSONObject() {
	s.Equal(llm.FormatJSONObject, s.format("json_object", ""))
}

func (s *RespondFormatSuite) TestAHeldJSONObjectAsksForJSONObject() {
	s.Equal(llm.FormatJSONObject, s.format("", "json_object"))
}

func (s *RespondFormatSuite) TestAFrameNamingJSONSchemaAsksForProse() {
	s.Equal(llm.FormatText, s.format("json_schema", ""))
}

func (s *RespondFormatSuite) TestAHeldJSONSchemaAsksForProse() {
	s.Equal(llm.FormatText, s.format("", "json_schema"))
}

func (s *RespondFormatSuite) TestTheFrameFormatBeatsAHeldJSONSchema() {
	s.Equal(llm.FormatJSONObject, s.format("json_object", "json_schema"))
}

func (s *RespondFormatSuite) TestNoFormatNamedLeavesItUnset() {
	s.Equal(llm.TextFormat(""), s.format("", ""))
}
