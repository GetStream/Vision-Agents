package users

import (
	"strconv"
	"testing"

	"github.com/stretchr/testify/suite"
)

// SeenSuite covers the in-memory tier on its own, where what it holds and what it drops
// can be counted.
type SeenSuite struct {
	suite.Suite
}

func TestSeenSuite(t *testing.T) {
	suite.Run(t, new(SeenSuite))
}

// fill adds n users to one app, named in order.
func (s *SeenSuite) fill(cache *seen, app string, n int) {
	for i := range n {
		cache.Add(app, strconv.Itoa(i))
	}
}

func (s *SeenSuite) TestAUserAddedIsFoundAgain() {
	cache := newSeen(10)
	cache.Add("app", "jlahey")

	s.True(cache.Get("app", "jlahey"))
	s.False(cache.Get("app", "randy"))
}

func (s *SeenSuite) TestTheSameIdUnderTwoAppsIsTwoUsers() {
	cache := newSeen(10)
	cache.Add("app", "jlahey")

	s.False(cache.Get("somebody-else", "jlahey"), "a user id belongs to the app that named it")
	s.Equal(1, cache.Len())
}

func (s *SeenSuite) TestAddingAUserTwiceHoldsThemOnce() {
	cache := newSeen(10)
	cache.Add("app", "jlahey")
	cache.Add("app", "jlahey")

	s.Equal(1, cache.Len())
}

func (s *SeenSuite) TestTheOldestUserIsDroppedWhenTheCacheIsFull() {
	cache := newSeen(3)
	s.fill(cache, "app", 4)

	s.False(cache.Get("app", "0"))
	s.True(cache.Get("app", "3"))
	s.Equal(3, cache.Len())
}

func (s *SeenSuite) TestAUserReadAgainOutlivesOneAddedBefore() {
	cache := newSeen(3)
	s.fill(cache, "app", 3)

	s.True(cache.Get("app", "0"))
	cache.Add("app", "3")

	s.True(cache.Get("app", "0"), "reading a user is using them")
	s.False(cache.Get("app", "1"))
}

func (s *SeenSuite) TestABusyAppTakesTheRoomAQuietOneIsNotUsing() {
	cache := newSeen(10)
	cache.Add("quiet", "jlahey")
	s.fill(cache, "busy", 12)

	// Ten in total, not ten each: the quiet app's one user is the oldest thing here, so
	// it goes the way any other oldest entry would.
	s.Equal(10, cache.Len())
	s.False(cache.Get("quiet", "jlahey"))
	s.True(cache.Get("busy", "11"))
}

func (s *SeenSuite) TestAnAppWhoseUsersAreAllEvictedIsForgotten() {
	cache := newSeen(2)
	cache.Add("quiet", "jlahey")
	s.fill(cache, "busy", 2)

	s.NotContains(cache.apps, "quiet", "an app nobody uses keeps no partition")
}

func (s *SeenSuite) TestRemovingAUserDropsThem() {
	cache := newSeen(10)
	cache.Add("app", "jlahey")
	cache.Remove("app", "jlahey")

	s.False(cache.Get("app", "jlahey"))
	s.Equal(0, cache.Len())
}

func (s *SeenSuite) TestRemovingAUserNobodyHeldChangesNothing() {
	cache := newSeen(10)
	cache.Add("app", "jlahey")
	cache.Remove("app", "randy")
	cache.Remove("somebody-else", "jlahey")

	s.True(cache.Get("app", "jlahey"))
}

func (s *SeenSuite) TestClearingDropsEveryApp() {
	cache := newSeen(10)
	cache.Add("app", "jlahey")
	cache.Add("somebody-else", "randy")

	cache.Clear()

	s.Equal(0, cache.Len())
	s.False(cache.Get("app", "jlahey"))

	// And it is usable afterwards, rather than a cache that has been emptied once.
	cache.Add("app", "jlahey")
	s.True(cache.Get("app", "jlahey"))
}

func (s *SeenSuite) TestACacheGivenNoCapacityHoldsTenThousand() {
	cache := newSeen(0)
	s.fill(cache, "app", DefaultCapacity+100)

	s.Equal(DefaultCapacity, cache.Len())
}
