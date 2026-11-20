package activity

import (
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ReachedStepsSuite covers which steps areas first answered and the day each was first answered.
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

// answered is area, first answered n days after s.day.
func (s *ReachedStepsSuite) answered(area string, n int) store.ActivityAreaDay {
	return store.ActivityAreaDay{Area: area, Day: s.on(n)}
}

func (s *ReachedStepsSuite) on(n int) time.Time { return s.day.Add(time.Duration(n) * 24 * time.Hour) }

// Defect: nothing recorded read as a step.
func (s *ReachedStepsSuite) TestNothingRecordedReachesNoStep() {
	s.Empty(ReachedSteps(nil))
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
	}, ReachedSteps([]store.ActivityAreaDay{
		s.answered(AreaAgentSetup, 1),
		s.answered(AreaConversations, 3),
		s.answered(AreaDashboardReads, 0),
		s.answered(AreaModelAPI, 0),
		s.answered(AreaPhoneSMS, 2),
		s.answered(AreaVoiceVideo, 5),
		s.answered(AreaWorkers, 1),
	}))
}
