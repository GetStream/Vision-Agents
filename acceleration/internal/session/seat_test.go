package session

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// SeatSuite covers which watchers a session's tool calls are addressed to.
type SeatSuite struct {
	suite.Suite
}

func TestSeatSuite(t *testing.T) {
	suite.Run(t, new(SeatSuite))
}

func (s *SeatSuite) TestABackendsToolRunsOnABackendThatSaidNothing() {
	backends := s.openedBy(auth.KindServer, "alice")

	s.True(backends.seatRuns(Seat{Owner: Owner{Kind: auth.KindServer}}, "lookup_order"))
	s.False(backends.seatRuns(Seat{Owner: Owner{Kind: auth.KindAuthenticated, UserID: "alice"}}, "lookup_order"))
}

func (s *SeatSuite) TestABackendsToolNeverRunsInABrowserWhateverItDeclares() {
	backends := s.openedBy(auth.KindServer, "alice")
	browser := Seat{Owner: Owner{Kind: auth.KindAuthenticated, UserID: "alice"}, Declares: true, Tools: []string{"lookup_order"}}

	s.False(backends.seatRuns(browser, "lookup_order"))
}

func (s *SeatSuite) TestABrowsersToolRunsInThatBrowserAndNotInAnotherUsers() {
	alices := s.openedBy(auth.KindAuthenticated, "alice")

	s.True(alices.seatRuns(Seat{Owner: Owner{Kind: auth.KindAuthenticated, UserID: "alice"}}, "lookup_order"))
	s.False(alices.seatRuns(Seat{Owner: Owner{Kind: auth.KindAuthenticated, UserID: "bob"}}, "lookup_order"))
	s.False(alices.seatRuns(Seat{Owner: Owner{Kind: auth.KindServer}}, "lookup_order"),
		"a backend watching runs the browser's tool only when it says so")
	s.True(alices.seatRuns(Seat{Owner: Owner{Kind: auth.KindServer}, Declares: true, Tools: []string{"lookup_order"}}, "lookup_order"))
}

func (s *SeatSuite) TestAWatcherThatDeclaredRunsOnlyWhatItDeclared() {
	backends := s.openedBy(auth.KindServer, "")

	s.False(backends.seatRuns(Seat{Owner: Owner{Kind: auth.KindServer}, Declares: true}, "lookup_order"))
	s.False(backends.seatRuns(Seat{Owner: Owner{Kind: auth.KindServer}, Declares: true}, agent.VideoFramesTool))
}

func (s *SeatSuite) TestAToolNobodyDeclaredGoesToEveryWatcherThatSaidNothing() {
	backends := s.openedBy(auth.KindServer, "alice")

	s.True(backends.seatRuns(Seat{Owner: Owner{Kind: auth.KindAuthenticated, UserID: "alice"}}, agent.VideoFramesTool),
		"a phone hands over its video whoever opened the session")
}

func (s *SeatSuite) openedBy(kind auth.Kind, userID string) *Session {
	return &Session{spec: Spec{
		CallerKind: kind,
		Caller:     routing.Caller{UserID: userID},
		Tools:      []harness.Tool{{Name: "lookup_order"}},
	}}
}
