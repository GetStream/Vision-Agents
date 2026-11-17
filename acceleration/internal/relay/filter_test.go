package relay

import (
	"strconv"
	"testing"

	"github.com/stretchr/testify/suite"
)

type FilterSuite struct {
	suite.Suite

	filter *Filter
}

func TestFilterSuite(t *testing.T) {
	suite.Run(t, new(FilterSuite))
}

func (s *FilterSuite) SetupTest() {
	s.filter = NewFilter()
}

func (s *FilterSuite) TestAKeyIsFoundOnceItIsAdded() {
	s.False(s.filter.Has("acme\x00jlahey"))

	s.filter.Add("acme\x00jlahey")

	s.True(s.filter.Has("acme\x00jlahey"))
}

func (s *FilterSuite) TestAKeyIsGoneOnceItIsRemoved() {
	s.filter.Add("acme\x00jlahey")
	s.filter.Remove("acme\x00jlahey")

	s.False(s.filter.Has("acme\x00jlahey"))
}

func (s *FilterSuite) TestAKeyTwoSocketsHoldSurvivesOneOfThemLeaving() {
	s.filter.Add("acme\x00jlahey")
	s.filter.Add("acme\x00jlahey")

	s.filter.Remove("acme\x00jlahey")
	s.True(s.filter.Has("acme\x00jlahey"), "the second socket is still watching")

	s.filter.Remove("acme\x00jlahey")
	s.False(s.filter.Has("acme\x00jlahey"))
}

// A removal with nothing behind it would delete whichever key shares the fingerprint,
// which is the one way a cuckoo filter can be made to say no about a key it holds.
func (s *FilterSuite) TestRemovingAKeyThatWasNeverAddedLeavesTheOthersAlone() {
	s.filter.Add("acme\x00jlahey")

	for i := range 10000 {
		s.filter.Remove("acme\x00absent-" + strconv.Itoa(i))
	}

	s.True(s.filter.Has("acme\x00jlahey"))
}

func (s *FilterSuite) TestKeysNothingHoldsAreAlmostAlwaysMissed() {
	for i := range 1000 {
		s.filter.Add("acme\x00watching-" + strconv.Itoa(i))
	}

	wrong := 0
	const asked = 10000
	for i := range asked {
		if s.filter.Has("acme\x00absent-" + strconv.Itoa(i)) {
			wrong++
		}
	}

	s.Less(float64(wrong)/asked, 0.01, "every false positive is a registry lookup for nothing")
}
