package streamedge

import (
	"sync"
	"time"

	rtc "github.com/GetStream/getstream-go-webrtc"
)

// clientIdleFor is how long a shared SDK client no call is using stays open. The SDK keeps
// its connections warm for as long, so the next session of the same agent joins on them.
const clientIdleFor = 30 * time.Minute

// Clients shares SDK clients between the Edges given it: one per agent identity, kept
// open between their calls, so a session after the first joins on connections that are
// already open instead of opening new ones to the coordinator and the SFU. Edges sharing
// it must agree on the unexported test options, which the first one's client keeps.
type Clients struct {
	idleFor time.Duration

	mu      sync.Mutex
	clients map[clientKey]*sharedClient
	closed  bool
}

// clientKey is what an SDK client is built from.
type clientKey struct {
	apiKey, apiSecret, userToken string
	baseURL, wsURL               string
	userID, userName             string
}

type sharedClient struct {
	client *rtc.Client
	// users is how many Edges hold it; idle closes it once none have for idleFor.
	users int
	idle  *time.Timer
}

// NewClients returns an empty set of shared clients. Close it when the last Edge has left.
func NewClients() *Clients {
	return &Clients{idleFor: clientIdleFor, clients: map[clientKey]*sharedClient{}}
}

// acquire returns the client for key, built with build if there is none, and the release
// that gives it back.
func (c *Clients) acquire(key clientKey, build func() (*rtc.Client, error)) (*rtc.Client, func(), error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		client, err := build()
		if err != nil {
			return nil, nil, err
		}
		return client, func() { _ = client.Close() }, nil
	}
	shared, ok := c.clients[key]
	if !ok {
		client, err := build()
		if err != nil {
			return nil, nil, err
		}
		shared = &sharedClient{client: client}
		c.clients[key] = shared
	}
	if shared.idle != nil {
		shared.idle.Stop()
		shared.idle = nil
	}
	shared.users++
	var once sync.Once
	return shared.client, func() { once.Do(func() { c.release(key, shared) }) }, nil
}

func (c *Clients) release(key clientKey, shared *sharedClient) {
	c.mu.Lock()
	defer c.mu.Unlock()
	shared.users--
	if shared.users > 0 || c.closed {
		return
	}
	shared.idle = time.AfterFunc(c.idleFor, func() {
		c.mu.Lock()
		defer c.mu.Unlock()
		if shared.users > 0 || c.clients[key] != shared {
			return
		}
		delete(c.clients, key)
		_ = shared.client.Close()
	})
}

// Close closes every client, and makes Edges that acquire one later build their own.
func (c *Clients) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	for key, shared := range c.clients {
		if shared.idle != nil {
			shared.idle.Stop()
		}
		_ = shared.client.Close()
		delete(c.clients, key)
	}
}
