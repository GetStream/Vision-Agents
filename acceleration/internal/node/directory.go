// Package node reaches the other nodes of a deployment.
//
// A session lives in one process's memory, so an operation on it -- saying something,
// interrupting, reading what models it settled on -- only works on the node running it. A
// load balancer has no reason to prefer that node, so the request lands wherever it
// lands, and the relay beside this package only covers the events socket.
//
// The answer here is to carry the request itself to the node that can answer it: the
// Directory says which node that is, and the Forwarder hands it over by gRPC. The node
// receiving it runs its own handler, so it authenticates and authorizes the caller
// itself; a node is not trusted to have checked.
package node

import (
	"context"
	"errors"
	"log/slog"
	"maps"
	"sync"
	"time"

	"github.com/redis/rueidis"
)

// holdFor is how long a node's claim on a session lasts unrenewed. A node that crashes
// leaves its claim behind for at most this long, and during it a request for one of that
// node's sessions is forwarded to something that is not there.
const holdFor = 30 * time.Second

// renewEvery renews well inside that, so a Redis that drops a write has several more
// chances before a session this node is running starts to look unheld.
const renewEvery = 10 * time.Second

// DirectoryOptions configures a Directory.
type DirectoryOptions struct {
	// Redis is required, and is the same client the rest of the deployment uses.
	Redis rueidis.Client
	// Address is required: it is the host and port this node's peers reach it at.
	Address string
	// Prefix names the keys, so two deployments sharing one Redis do not forward each
	// other's requests.
	Prefix string
	Logger *slog.Logger
}

// Directory is who is running what, as far as Redis knows.
//
// A node writes its claims here and renews them until it lets go, so the entries are a
// statement about now rather than a record: a node that stops saying it runs a session
// stops being asked about it, whether it let go or died.
type Directory struct {
	redis    rueidis.Client
	address  string
	sessions string
	agents   string
	logger   *slog.Logger
	cancel   context.CancelFunc

	mu sync.Mutex
	// held maps a session this node is running to the agent it writes as, which is empty
	// for a session that writes to nobody.
	held map[string]string
}

// NewDirectory returns a Directory and starts renewing what it is given to hold.
func NewDirectory(options DirectoryOptions) (*Directory, error) {
	if options.Redis == nil {
		return nil, errors.New("node: a redis client is required")
	}
	if options.Address == "" {
		return nil, errors.New("node: an address this node's peers can reach it at is required")
	}
	prefix := options.Prefix
	if prefix == "" {
		prefix = "node"
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}

	ctx, cancel := context.WithCancel(context.Background())
	directory := &Directory{
		redis:    options.Redis,
		address:  options.Address,
		sessions: prefix + ":session:",
		agents:   prefix + ":agent:",
		logger:   logger,
		cancel:   cancel,
		held:     map[string]string{},
	}
	go directory.renew(ctx)

	return directory, nil
}

// Address is where this node's peers reach it.
func (d *Directory) Address() string { return d.address }

// Hold says this node is running a session, and goes on saying so until Release.
//
// The agent id is held beside it because an arriving message names a channel and no
// session at all, so the node that can answer it has to be findable by the agent writing
// there.
func (d *Directory) Hold(ctx context.Context, sessionID, agentID string) {
	d.mu.Lock()
	d.held[sessionID] = agentID
	d.mu.Unlock()

	d.write(ctx, sessionID, agentID)
}

// Release says this node is not running a session any more.
//
// Only the session's own claim is dropped. The agent's is left to expire, because a
// second session for the same agent may have started on another node and overwritten it,
// and deleting that would leave the agent unreachable rather than merely stale.
func (d *Directory) Release(ctx context.Context, sessionID string) {
	d.mu.Lock()
	_, held := d.held[sessionID]
	delete(d.held, sessionID)
	d.mu.Unlock()
	if !held {
		return
	}

	if err := d.redis.Do(ctx, d.redis.B().Del().Key(d.sessions+sessionID).Build()).Error(); err != nil {
		d.logger.Warn("could not take back this node's claim on a session",
			"session", sessionID, "error", err)
	}
}

// Node is where the node running a session can be reached, and empty when no node says it
// is running one.
func (d *Directory) Node(ctx context.Context, sessionID string) (string, error) {
	return d.lookup(ctx, d.sessions+sessionID)
}

// NodeByAgent is the same for the session writing as an agent, which is what an arriving
// message has to be answered by.
func (d *Directory) NodeByAgent(ctx context.Context, agentID string) (string, error) {
	return d.lookup(ctx, d.agents+agentID)
}

// Close stops renewing. What this node held expires rather than being given up, because
// a process on its way out has a shutdown grace period to finish the calls it is carrying
// and is still the only node that can answer for them.
func (d *Directory) Close() { d.cancel() }

func (d *Directory) lookup(ctx context.Context, key string) (string, error) {
	address, err := d.redis.Do(ctx, d.redis.B().Get().Key(key).Build()).ToString()
	if rueidis.IsRedisNil(err) {
		return "", nil
	}
	return address, err
}

// write says this node holds a session, for holdFor.
func (d *Directory) write(ctx context.Context, sessionID, agentID string) {
	commands := []rueidis.Completed{
		d.redis.B().Set().Key(d.sessions + sessionID).Value(d.address).Ex(holdFor).Build(),
	}
	if agentID != "" {
		commands = append(commands,
			d.redis.B().Set().Key(d.agents+agentID).Value(d.address).Ex(holdFor).Build())
	}
	for _, result := range d.redis.DoMulti(ctx, commands...) {
		if err := result.Error(); err != nil {
			d.logger.Warn("could not say this node is running a session",
				"session", sessionID, "error", err)
			return
		}
	}
}

// renew keeps every claim this node holds alive until the process stops.
func (d *Directory) renew(ctx context.Context) {
	ticker := time.NewTicker(renewEvery)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}

		d.mu.Lock()
		held := maps.Clone(d.held)
		d.mu.Unlock()

		for sessionID, agentID := range held {
			d.write(ctx, sessionID, agentID)
		}
	}
}
