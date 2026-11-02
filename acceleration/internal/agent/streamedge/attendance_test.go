package streamedge

import (
	"time"

	sfu_events "github.com/GetStream/protocol/protobuf/video/sfu/event"
	sfu_models "github.com/GetStream/protocol/protobuf/video/sfu/models"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
)

// person is a phone on a call, as the SFU names it.
var person = &sfu_models.Participant{UserId: "sip-+15125550123", SessionId: "person-1", Name: "+15125550123"}

// attending is an edge that has joined nothing, to report arrivals on.
func (s *StreamEdgeSuite) attending() *Edge {
	edge, err := New(Options{CallID: "call-1", User: User{ID: "agent"}, APIKey: "key", APISecret: "secret"})
	s.Require().NoError(err)
	return edge
}

// reported is every arrival and departure the edge has reported so far.
func (s *StreamEdgeSuite) reported(edge *Edge) []agent.Attendance {
	var reported []agent.Attendance
	for {
		select {
		case attendance := <-edge.Attendance():
			reported = append(reported, attendance)
		case <-time.After(50 * time.Millisecond):
			return reported
		}
	}
}

func (s *StreamEdgeSuite) TestSomebodyTheSFUOnlyAnnouncesByTheirTrackArrives() {
	// The SFU may leave a join unannounced and name the participant on their track instead.
	edge := s.attending()

	edge.publishedBy(&sfu_events.TrackPublished{
		UserId: person.UserId, SessionId: person.SessionId, Type: sfu_models.TrackType_TRACK_TYPE_AUDIO, Participant: person,
	})

	reported := s.reported(edge)
	s.Require().Len(reported, 1)
	s.True(reported[0].Joined)
	s.Equal("person-1", reported[0].Participant.ID)
	s.Equal("sip-+15125550123", reported[0].Participant.UserID)
}

func (s *StreamEdgeSuite) TestATrackWithoutTheParticipantStillNamesWhoArrived() {
	edge := s.attending()

	edge.publishedBy(&sfu_events.TrackPublished{UserId: person.UserId, SessionId: person.SessionId, Type: sfu_models.TrackType_TRACK_TYPE_AUDIO})

	reported := s.reported(edge)
	s.Require().Len(reported, 1)
	s.Equal("person-1", reported[0].Participant.ID)
	s.Equal("sip-+15125550123", reported[0].Participant.UserID)
}

func (s *StreamEdgeSuite) TestSomebodyAnnouncedAndPublishingArrivesOnce() {
	edge := s.attending()

	edge.report(person, true)
	edge.publishedBy(&sfu_events.TrackPublished{UserId: person.UserId, SessionId: person.SessionId, Participant: person})
	edge.publishedBy(&sfu_events.TrackPublished{UserId: person.UserId, SessionId: person.SessionId, Type: sfu_models.TrackType_TRACK_TYPE_VIDEO})

	s.Len(s.reported(edge), 1)
}

func (s *StreamEdgeSuite) TestSomebodyWhoLeftAndPublishesAgainArrivesAgain() {
	edge := s.attending()

	edge.report(person, true)
	edge.report(person, false)
	edge.publishedBy(&sfu_events.TrackPublished{UserId: person.UserId, SessionId: person.SessionId, Participant: person})

	reported := s.reported(edge)
	s.Require().Len(reported, 3)
	s.True(reported[2].Joined)
}

func (s *StreamEdgeSuite) TestTheAgentsOwnTrackIsNobodyArriving() {
	edge := s.attending()

	edge.publishedBy(&sfu_events.TrackPublished{UserId: "agent", SessionId: "agent-1"})

	s.Empty(s.reported(edge))
}
