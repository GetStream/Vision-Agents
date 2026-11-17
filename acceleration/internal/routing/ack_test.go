package routing

import (
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// AckSuite reads the usage service's answers to a shipment of three rows.
type AckSuite struct {
	suite.Suite
	rows []store.Request
}

func TestAckSuite(t *testing.T) {
	suite.Run(t, new(AckSuite))
}

func (s *AckSuite) SetupTest() {
	s.rows = []store.Request{{UsageID: uuid.New()}, {UsageID: uuid.New()}, {UsageID: uuid.New()}}
}

func (s *AckSuite) TestAnAnswerForEveryRowNamesTheRefusedOnes() {
	refused, err := refusedRows(s.rows, usage.Ack{Accepted: 1, Duplicates: 1, Rejected: []usage.Rejection{
		{Index: 2, UsageID: s.rows[2].UsageID, Reason: usage.ReasonMeter},
	}})

	s.Require().NoError(err)
	s.Equal(map[uuid.UUID]string{s.rows[2].UsageID: usage.ReasonMeter}, refused)
}

func (s *AckSuite) TestARejectionNamedByItsIndexAloneIsTaken() {
	refused, err := refusedRows(s.rows, usage.Ack{Accepted: 2, Rejected: []usage.Rejection{{Index: 0, Reason: usage.ReasonMalformed}}})

	s.Require().NoError(err)
	s.Equal(map[uuid.UUID]string{s.rows[0].UsageID: usage.ReasonMalformed}, refused)
}

func (s *AckSuite) TestAnAnswerForFewerRowsThanWereSentIsRefused() {
	_, err := refusedRows(s.rows, usage.Ack{Accepted: 2})

	s.ErrorContains(err, "answered for 2 events of 3")
}

func (s *AckSuite) TestARejectionPastTheRowsSentIsRefused() {
	_, err := refusedRows(s.rows, usage.Ack{Accepted: 2, Rejected: []usage.Rejection{{Index: 3, Reason: usage.ReasonMeter}}})

	s.ErrorContains(err, "refused event 3 of 3")
}

func (s *AckSuite) TestARejectionNamingAnotherRowsUsageIDIsRefused() {
	_, err := refusedRows(s.rows, usage.Ack{Accepted: 2, Rejected: []usage.Rejection{{Index: 1, UsageID: s.rows[0].UsageID, Reason: usage.ReasonMeter}}})

	s.ErrorContains(err, "as event 1")
}

func (s *AckSuite) TestOneRowRefusedTwiceIsRefused() {
	_, err := refusedRows(s.rows, usage.Ack{Accepted: 1, Rejected: []usage.Rejection{
		{Index: 1, Reason: usage.ReasonMeter}, {Index: 1, Reason: usage.ReasonMeter},
	}})

	s.ErrorContains(err, "event 1 twice")
}
