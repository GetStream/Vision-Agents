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

// published is the SFU saying a track of the person's is out, with the participant when
// named. Video, because subscribing to audio needs a call joined.
func published(named bool) *sfu_events.TrackPublished {
	track := &sfu_events.TrackPublished{UserId: person.UserId, SessionId: person.SessionId, Type: sfu_models.TrackType_TRACK_TYPE_VIDEO}
	if named {
		track.Participant = person
	}
	return track
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

	edge.trackPublished(s.ctx, published(true))

	reported := s.reported(edge)
	s.Require().Len(reported, 1)
	s.True(reported[0].Joined)
	s.Equal("person-1", reported[0].Participant.ID)
	s.Equal("sip-+15125550123", reported[0].Participant.UserID)
	s.Equal("+15125550123", reported[0].Participant.Name)
}

func (s *StreamEdgeSuite) TestATrackWithoutTheParticipantStillNamesWhoArrived() {
	edge := s.attending()

	edge.trackPublished(s.ctx, published(false))

	reported := s.reported(edge)
	s.Require().Len(reported, 1)
	s.Equal("person-1", reported[0].Participant.ID)
	s.Equal("sip-+15125550123", reported[0].Participant.UserID)
}

func (s *StreamEdgeSuite) TestSomebodyAnnouncedAndPublishingArrivesOnce() {
	edge := s.attending()

	edge.report(person, true)
	edge.trackPublished(s.ctx, published(true))
	edge.trackPublished(s.ctx, published(false))

	s.Len(s.reported(edge), 1)
}

func (s *StreamEdgeSuite) TestANameLearnedAfterTheTrackIsReported() {
	// Somebody first known only by an unnamed track is reported again once a join names them,
	// so whoever reads the arrivals gets their name.
	edge := s.attending()

	edge.trackPublished(s.ctx, published(false))
	edge.report(person, true)
	edge.report(person, true)

	reported := s.reported(edge)
	s.Require().Len(reported, 2)
	s.Empty(reported[0].Participant.Name)
	s.Equal("+15125550123", reported[1].Participant.Name)
	s.True(reported[1].Joined)
}

func (s *StreamEdgeSuite) TestSomebodyWhoLeftAndPublishesAgainArrivesAgain() {
	edge := s.attending()

	edge.report(person, true)
	edge.report(person, false)
	edge.trackPublished(s.ctx, published(true))

	reported := s.reported(edge)
	s.Require().Len(reported, 3)
	s.True(reported[2].Joined)
}

func (s *StreamEdgeSuite) TestTheAgentsOwnTrackIsNobodyArriving() {
	edge := s.attending()

	edge.trackPublished(s.ctx, &sfu_events.TrackPublished{UserId: "agent", SessionId: "agent-1", Type: sfu_models.TrackType_TRACK_TYPE_VIDEO})

	s.Empty(s.reported(edge))
}
