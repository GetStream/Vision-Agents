package api

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/activity"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// FurthestStepSuite covers how far an organization's areas say it got (furthestStep).
type FurthestStepSuite struct {
	suite.Suite
}

func TestFurthestStepSuite(t *testing.T) {
	suite.Run(t, new(FurthestStepSuite))
}

func areaWith(area string, requests int64) store.ActivityArea {
	return store.ActivityArea{Area: area, Client: store.AuditSourceAPI, Requests: requests}
}

// Defect: nothing recorded read as a step.
func (s *FurthestStepSuite) TestNothingRecordedIsNoStep() {
	s.Equal(activityStepNone, furthestStep(nil, 0))
}

// Defect: the last area read kept rather than the furthest: areas come by name, and
// dashboard_reads sorts after conversations.
func (s *FurthestStepSuite) TestTheFurthestAreaIsTheStepWhateverItsOrder() {
	s.Equal(activityStepConversations, furthestStep([]store.ActivityArea{
		areaWith(activity.AreaAgentSetup, 1), areaWith(activity.AreaConversations, 1), areaWith(activity.AreaDashboardReads, 4),
	}, 0))
	s.Equal(activityStepAgentSetup, furthestStep([]store.ActivityArea{
		areaWith(activity.AreaAgentSetup, 1), areaWith(activity.AreaDashboardReads, 1),
	}, 0))
	s.Equal(activityStepDashboardReads, furthestStep([]store.ActivityArea{areaWith(activity.AreaDashboardReads, 1)}, 0))
}

// Defect: an area whose every request was refused counted as reached, so an app turned away
// at setup reads as set up.
func (s *FurthestStepSuite) TestAnAreaRefusedEveryTimeIsNotReached() {
	refused := areaWith(activity.AreaConversations, 2)
	refused.Refused = 2

	s.Equal(activityStepDashboardReads, furthestStep([]store.ActivityArea{
		areaWith(activity.AreaDashboardReads, 1), refused,
	}, 0))

	refused.Requests = 3
	s.Equal(activityStepConversations, furthestStep([]store.ActivityArea{refused}, 0))
}

// Defect: an area that is no step, such as model calls, read as one, or as the first.
func (s *FurthestStepSuite) TestAnAreaThatIsNoStepReachesNone() {
	s.Equal(activityStepNone, furthestStep([]store.ActivityArea{
		areaWith(activity.AreaModelAPI, 9), areaWith(activity.AreaWorkers, 9),
	}, 2))
}

// Defect: steady use from fewer than three days, or not from three whatever the areas say.
func (s *FurthestStepSuite) TestThreeDaysOfConversationsOrModelCallsAreSteadyUse() {
	areas := []store.ActivityArea{areaWith(activity.AreaConversations, 10)}

	s.Equal(activityStepConversations, furthestStep(areas, 2))
	s.Equal(activityStepSteadyUse, furthestStep(areas, 3))
	s.Equal(activityStepSteadyUse, furthestStep(nil, 3))
}
