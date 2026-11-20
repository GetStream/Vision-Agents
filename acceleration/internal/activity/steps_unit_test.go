package activity

import (
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ReachedStepsSuite covers which steps areas reached and the day each was first answered.
type ReachedStepsSuite struct {
	suite.Suite
	// day is a fixed UTC day, so the days of a test are predictable.
	day time.Time
}

func TestReachedStepsSuite(t *testing.T) {
	suite.Run(t, new(ReachedStepsSuite))
}

func (s *ReachedStepsSuite) SetupTest() {
	s.day = time.Date(2026, 3, 1, 0, 0, 0, 0, time.UTC)
}

// answered is area's requests from client, the first answered n days after s.day.
func (s *ReachedStepsSuite) answered(area, client string, n int) store.ActivityArea {
	return store.ActivityArea{Area: area, Client: client, Requests: 1, FirstAnswered: s.on(n)}
}

func (s *ReachedStepsSuite) on(n int) time.Time { return s.day.Add(time.Duration(n) * 24 * time.Hour) }

// Defect: nothing recorded read as a step.
func (s *ReachedStepsSuite) TestNothingRecordedReachesNoStep() {
	s.Empty(ReachedSteps(nil))
}

// Defect: the day of the client read last kept rather than the earliest of every client.
func (s *ReachedStepsSuite) TestAStepsDayIsTheEarliestOfEveryClient() {
	s.Equal([]ReachedStep{{Step: AreaConversations, FirstAnswered: s.on(2)}}, ReachedSteps([]store.ActivityArea{
		s.answered(AreaConversations, store.AuditSourceAPI, 5),
		s.answered(AreaConversations, store.AuditSourceCLI, 2),
		s.answered(AreaConversations, store.AuditSourceSDK, 3),
	}))
}

// Defect: an area whose every request was refused, and so has no answered day, read as a
// step first answered on the zero day.
func (s *ReachedStepsSuite) TestAnAreaWithNoAnsweredDayIsNotReached() {
	refused := store.ActivityArea{Area: AreaVoiceVideo, Client: store.AuditSourceSDK, Requests: 2, Refused: 2}
	early := refused
	early.Area = AreaConversations

	s.Equal([]ReachedStep{{Step: AreaConversations, FirstAnswered: s.on(4)}}, ReachedSteps([]store.ActivityArea{
		early, s.answered(AreaConversations, store.AuditSourceAPI, 4), refused,
	}))
}

// Defect: steps in the order areas come in (by name), or areas that are no step listed.
func (s *ReachedStepsSuite) TestStepsComeInLadderOrderAndOtherAreasAreLeftOut() {
	s.Equal([]ReachedStep{
		{Step: AreaDashboardReads, FirstAnswered: s.on(0)},
		{Step: AreaAgentSetup, FirstAnswered: s.on(1)},
		{Step: AreaWorkers, FirstAnswered: s.on(1)},
		{Step: AreaConversations, FirstAnswered: s.on(3)},
		{Step: AreaVoiceVideo, FirstAnswered: s.on(5)},
		{Step: AreaPhoneSMS, FirstAnswered: s.on(2)},
	}, ReachedSteps([]store.ActivityArea{
		s.answered(AreaAgentSetup, store.AuditSourceCLI, 1),
		s.answered(AreaConversations, store.AuditSourceSDK, 3),
		s.answered(AreaDashboardReads, store.AuditSourceDashboard, 0),
		s.answered(AreaModelAPI, store.AuditSourceSDK, 0),
		s.answered(AreaPhoneSMS, store.AuditSourceAPI, 2),
		s.answered(AreaVoiceVideo, store.AuditSourceSDK, 5),
		s.answered(AreaWorkers, store.AuditSourceSDK, 1),
	}))
}
