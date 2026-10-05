package users

import (
	"container/list"
	"sync"
)

// DefaultCapacity is how many users one process keeps in memory, across every app it
// serves. It is a count rather than a size because the entries are two short strings
// each, so ten thousand of them is a few hundred kilobytes.
const DefaultCapacity = 10_000

// seen is which users this process has already recorded, newest use first.
//
// Partitioned per app so one app's users can be dropped without walking the rest, and
// because a user id belongs to the customer: the same id under two apps is two people.
// Eviction is across the partitions rather than within them, so an app serving nobody
// holds nothing and a busy one gets the room, which is what a cache of a fixed total size
// means.
//
// No value is kept beside the key. The only question asked of this is whether a user has
// already been written down, and keeping the row as well would be keeping a copy of
// Postgres that nothing reads.
type seen struct {
	capacity int

	mu    sync.Mutex
	order *list.List
	apps  map[string]map[string]*list.Element
}

// entry is what one element of the recency list holds, which is enough to find its
// partition when it is evicted from the far end.
type entry struct {
	app string
	id  string
}

func newSeen(capacity int) *seen {
	if capacity <= 0 {
		capacity = DefaultCapacity
	}
	return &seen{
		capacity: capacity,
		order:    list.New(),
		apps:     map[string]map[string]*list.Element{},
	}
}

// Get reports whether a user is held, and marks them as the most recently used.
func (s *seen) Get(app, id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	element, found := s.apps[app][id]
	if !found {
		return false
	}
	s.order.MoveToFront(element)
	return true
}

// Add holds a user, evicting the least recently used one across every app when the cache
// is full.
func (s *seen) Add(app, id string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if element, found := s.apps[app][id]; found {
		s.order.MoveToFront(element)
		return
	}

	partition, found := s.apps[app]
	if !found {
		partition = map[string]*list.Element{}
		s.apps[app] = partition
	}
	partition[id] = s.order.PushFront(entry{app: app, id: id})

	for s.order.Len() > s.capacity {
		s.evict(s.order.Back())
	}
}

// Remove drops a user, for one whose row has changed under them.
func (s *seen) Remove(app, id string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if element, found := s.apps[app][id]; found {
		s.evict(element)
	}
}

// Clear drops everything, which is what Redis asks for when it cannot say which keys went
// stale.
func (s *seen) Clear() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.order.Init()
	s.apps = map[string]map[string]*list.Element{}
}

// Len is how many users are held across every app.
func (s *seen) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.order.Len()
}

// evict removes one element and the partition it empties. The caller holds the lock.
func (s *seen) evict(element *list.Element) {
	if element == nil {
		return
	}
	evicted := s.order.Remove(element).(entry)
	partition := s.apps[evicted.app]
	delete(partition, evicted.id)
	if len(partition) == 0 {
		delete(s.apps, evicted.app)
	}
}
