package core_test

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

type ToolPatternsSuite struct {
	suite.Suite
}

func TestToolPatternsSuite(t *testing.T) {
	suite.Run(t, new(ToolPatternsSuite))
}

func (s *ToolPatternsSuite) TestAToolPatternThatCannotBeReadIsRefused() {
	s.ErrorContains(core.CheckToolPatterns([]string{"read_[*"}), "read_[*")
	s.NoError(core.CheckToolPatterns([]string{"read_*", "search_files"}))
}
