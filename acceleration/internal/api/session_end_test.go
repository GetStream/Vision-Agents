package api

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// SessionEndSuite is how a stored session shows why it ended.
type SessionEndSuite struct{ suite.Suite }

func TestSessionEndSuite(t *testing.T) { suite.Run(t, new(SessionEndSuite)) }

// shown is a session rendered from a row that kept reason.
func (s *SessionEndSuite) shown(reason string) Session {
	var rendered Session
	mergeStored(&rendered, &store.AgentSession{Modality: store.ModalityVoice, EndReason: reason})
	return rendered
}

func (s *SessionEndSuite) TestAReasonInTheListShowsItsFaultAndStage() {
	// Catches the fault or stage of another reason shown, or none shown.
	shown := s.shown("start")
	s.Require().NotNil(shown.EndFault)
	s.Require().NotNil(shown.EndStage)
	s.Equal(SessionEndFault("carrier"), *shown.EndFault)
	s.Equal(SessionEndStage("ringing"), *shown.EndStage)
}

func (s *SessionEndSuite) TestARowWithNoReasonOrOneOutsideTheListShowsNoFaultOrStage() {
	// Catches an empty fault and stage shown for a session still running, or one that ended
	// before reasons were kept, and for a reason a newer router stored that this one does not list.
	running := s.shown("")
	s.Nil(running.EndReason)
	s.Nil(running.EndFault)
	s.Nil(running.EndStage)

	later := s.shown("a_later_reason")
	s.Require().NotNil(later.EndReason)
	s.Equal(SessionEndReason("a_later_reason"), *later.EndReason, "the stored reason is shown as it is")
	s.Nil(later.EndFault)
	s.Nil(later.EndStage)
}
