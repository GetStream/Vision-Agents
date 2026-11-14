package store

import (
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type MergeHiddenRangesSuite struct {
	suite.Suite
	base time.Time
}

func TestMergeHiddenRangesSuite(t *testing.T) {
	suite.Run(t, new(MergeHiddenRangesSuite))
}

func (s *MergeHiddenRangesSuite) SetupTest() {
	s.base = time.Date(2026, 3, 1, 10, 0, 0, 0, time.UTC)
}

// between is the range from minute from to minute to after base.
func (s *MergeHiddenRangesSuite) between(from, to int) HiddenRange {
	return HiddenRange{From: s.base.Add(time.Duration(from) * time.Minute), To: s.base.Add(time.Duration(to) * time.Minute)}
}

func (s *MergeHiddenRangesSuite) TestOverlappingRangesAreOne() {
	merged := MergeHiddenRanges([]HiddenRange{s.between(0, 10), s.between(5, 15)})

	s.Equal([]HiddenRange{s.between(0, 15)}, merged)
}

func (s *MergeHiddenRangesSuite) TestARangeStartingWhereAnotherEndsIsOneWithIt() {
	merged := MergeHiddenRanges([]HiddenRange{s.between(0, 10), s.between(10, 20)})

	s.Equal([]HiddenRange{s.between(0, 20)}, merged)
}

func (s *MergeHiddenRangesSuite) TestARangeStartingJustAfterAnotherEndsIsApart() {
	later := HiddenRange{From: s.base.Add(10*time.Minute + time.Microsecond), To: s.base.Add(20 * time.Minute)}

	merged := MergeHiddenRanges([]HiddenRange{s.between(0, 10), later})

	s.Len(merged, 2)
}

func (s *MergeHiddenRangesSuite) TestARangeInsideAnotherAddsNothing() {
	merged := MergeHiddenRanges([]HiddenRange{s.between(0, 30), s.between(5, 10)})

	s.Equal([]HiddenRange{s.between(0, 30)}, merged)
}

func (s *MergeHiddenRangesSuite) TestRangesApartStayApartInTheOrderTheyStart() {
	merged := MergeHiddenRanges([]HiddenRange{s.between(60, 70), s.between(0, 10), s.between(30, 40)})

	s.Equal([]HiddenRange{s.between(0, 10), s.between(30, 40), s.between(60, 70)}, merged)
}

func (s *MergeHiddenRangesSuite) TestRangesOutOfOrderStillMerge() {
	merged := MergeHiddenRanges([]HiddenRange{s.between(20, 30), s.between(50, 60), s.between(0, 20), s.between(25, 50)})

	s.Equal([]HiddenRange{s.between(0, 60)}, merged)
}

func (s *MergeHiddenRangesSuite) TestRangesFromTheSameMessageKeepItHiddenIfEitherDid() {
	inclusive := s.between(0, 5)
	inclusive.FromInclusive = true

	merged := MergeHiddenRanges([]HiddenRange{s.between(0, 10), inclusive})

	s.Equal([]HiddenRange{{From: s.base, To: s.base.Add(10 * time.Minute), FromInclusive: true}}, merged)
}

func (s *MergeHiddenRangesSuite) TestNoRangesAreNone() {
	s.Empty(MergeHiddenRanges(nil))
}
