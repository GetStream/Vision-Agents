package relay

import (
	"sync"

	cuckoo "github.com/seiflotfy/cuckoofilter"
)

// capacity is how many keys a node's filter is sized for. A key is one session owner with
// a socket on this node, so this is the number of people watching a conversation from
// here at once, and the filter costs about a byte each.
const capacity = 1 << 16

// Filter answers whether this node holds a socket for a key.
//
// Every message on the bus reaches every node, so the question is asked far more often
// than it is answered yes, and a node holding nothing has to be able to drop a message
// without taking a lock anything else wants. That is what a cuckoo filter buys: a hash
// and two bucket reads under a read lock, against a map whose writer is the registry a
// socket arriving contends for.
//
// A false positive costs one lookup in the registry behind it and nothing else, which is
// why the answer being approximate is allowed to be.
type Filter struct {
	mu     sync.RWMutex
	filter *cuckoo.Filter
	// held counts the sockets each key stands for. A cuckoo filter stores a fingerprint
	// rather than the key, so deleting one key's entry twice takes another key's: two
	// watchers of the same owner must insert once between them and delete once.
	held map[string]int
}

// NewFilter returns an empty filter.
func NewFilter() *Filter {
	return &Filter{filter: cuckoo.NewFilter(capacity), held: map[string]int{}}
}

// Add records one socket for a key.
func (f *Filter) Add(key string) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.held[key] == 0 {
		f.filter.Insert([]byte(key))
	}
	f.held[key]++
}

// Remove gives back one socket for a key. Removing a key that was never added does
// nothing, so a socket that failed to register cannot take another's entry with it.
func (f *Filter) Remove(key string) {
	f.mu.Lock()
	defer f.mu.Unlock()

	switch f.held[key] {
	case 0:
		return
	case 1:
		delete(f.held, key)
		f.filter.Delete([]byte(key))
	default:
		f.held[key]--
	}
}

// Has reports whether this node may hold a socket for a key. It is allowed to say yes
// about a key nothing here holds, and never says no about one it does.
func (f *Filter) Has(key string) bool {
	f.mu.RLock()
	defer f.mu.RUnlock()

	return f.filter.Lookup([]byte(key))
}
