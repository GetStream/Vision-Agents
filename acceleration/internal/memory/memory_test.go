package memory

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type MemorySuite struct {
	suite.Suite
}

func TestMemorySuite(t *testing.T) {
	suite.Run(t, new(MemorySuite))
}

func (s *MemorySuite) TestAScopeWithoutAUserIsRejected() {
	s.ErrorContains(Scope{AppID: "router"}.Validate(), "user id")
}

func (s *MemorySuite) TestAScopedMemoryBelongsToSomeone() {
	s.NoError(Scope{AppID: "router", UserID: "acme"}.Validate())
}

func (s *MemorySuite) TestNothingRecalledMeansNothingIsPrepended() {
	s.Empty(Prompt(nil), "an empty system message would only cost tokens")
}

func (s *MemorySuite) TestRecalledMemoriesBecomeASystemMessage() {
	prompt := Prompt([]Memory{
		{Text: "Prefers to be called Al"},
		{Text: "Is allergic to nuts"},
	})

	s.Equal("What you already know about this person:\n- Prefers to be called Al\n- Is allergic to nuts", prompt)
}
