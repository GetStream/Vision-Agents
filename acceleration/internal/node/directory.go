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
	"encoding/json"
	"errors"
	"log/slog"
	"maps"
	"slices"
	"strconv"
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
	// Shard is the shard this node is in, empty for a deployment of one. A node in another
	// shard is not on this one's relay, so a socket for a session it runs is carried there.
	Shard string
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
	shard    string
	sessions string
	agents   string
	workers  string
	hosted   string
	shards   string
	logger   *slog.Logger
	cancel   context.CancelFunc

	mu sync.Mutex
	// held maps a session this node is running to the agent it writes as, which is empty
	// for a session that writes to nobody.
	held map[string]string
	// waiting counts the dispatch workers connected here, by customer.
	waiting map[string]int
	// offered is what the workers here host, by customer and agent.
	offered map[offerKey][]byte
}

type offerKey struct{ customerID, agentID string }

// offerEntry is one node's offer as Redis holds it. A hash cannot expire its fields one at
// a time, so each says when it lapses.
type offerEntry struct {
	Until int64           `json:"until"`
	Offer json.RawMessage `json:"offer"`
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
		shard:    options.Shard,
		sessions: prefix + ":session:",
		agents:   prefix + ":agent:",
		workers:  prefix + ":workers:",
		hosted:   prefix + ":hosted:",
		shards:   prefix + ":shard:",
		logger:   logger,
		cancel:   cancel,
		held:     map[string]string{},
		waiting:  map[string]int{},
		offered:  map[offerKey][]byte{},
	}
	directory.sayShard(ctx)
	go directory.renew(ctx)

	return directory, nil
}

// Address is where this node's peers reach it.
func (d *Directory) Address() string { return d.address }

// InAnotherShard reports whether the node at an address is in a shard other than this one's.
// A node that has not said is taken to be in this one, which is every node of a deployment of
// one shard.
func (d *Directory) InAnotherShard(ctx context.Context, address string) (bool, error) {
	if d.shard == "" {
		return false, nil
	}
	shard, err := d.lookup(ctx, d.shards+address)
	return shard != "" && shard != d.shard, err
}

// sayShard says which shard this node is in, for holdFor.
func (d *Directory) sayShard(ctx context.Context) {
	if d.shard == "" {
		return
	}
	command := d.redis.B().Set().Key(d.shards + d.address).Value(d.shard).Ex(holdFor).Build()
	if err := d.redis.Do(ctx, command).Error(); err != nil {
		d.logger.Warn("could not say which shard this node is in", "shard", d.shard, "error", err)
	}
}

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

// Wait says this node has a dispatch worker connected for a customer, and goes on saying
// so until the function it returns is called.
//
// A call or a message arrives at whichever node the load balancer picked, and the worker
// that has to answer it may be connected to any other: this is how that node is found.
func (d *Directory) Wait(ctx context.Context, customerID string) func() {
	d.mu.Lock()
	d.waiting[customerID]++
	d.mu.Unlock()
	d.advertise(ctx, customerID)

	return func() {
		d.mu.Lock()
		d.waiting[customerID]--
		last := d.waiting[customerID] == 0
		if last {
			delete(d.waiting, customerID)
		}
		d.mu.Unlock()
		if !last {
			return
		}

		// The worker has gone, whether or not the caller's request is still running.
		ctx := context.WithoutCancel(ctx)
		command := d.redis.B().Zrem().Key(d.workers + customerID).Member(d.address).Build()
		if err := d.redis.Do(ctx, command).Error(); err != nil {
			d.logger.Warn("could not take back this node's claim on a customer's workers",
				"customer", customerID, "error", err)
		}
	}
}

// Waiting is where the other nodes with a dispatch worker connected for a customer can be
// reached. This node is never among them.
func (d *Directory) Waiting(ctx context.Context, customerID string) ([]string, error) {
	now := strconv.FormatInt(time.Now().UnixMilli(), 10)
	addresses, err := d.redis.Do(ctx, d.redis.B().Zrange().Key(d.workers+customerID).
		Min("("+now).Max("+inf").Byscore().Build()).AsStrSlice()
	if err != nil {
		return nil, err
	}
	return slices.DeleteFunc(addresses, func(address string) bool { return address == d.address }), nil
}

// Offer says what the workers here host for sessions under an agent, a JSON document the
// directory carries without reading, and goes on saying so until it is offered again. A nil
// offer withdraws it.
func (d *Directory) Offer(ctx context.Context, customerID, agentID string, offer []byte) {
	key := offerKey{customerID, agentID}
	d.mu.Lock()
	if offer == nil {
		delete(d.offered, key)
	} else {
		d.offered[key] = offer
	}
	d.mu.Unlock()

	if offer != nil {
		d.publishOffer(ctx, key, offer)
		return
	}
	command := d.redis.B().Hdel().Key(d.offerKey(key)).Field(d.address).Build()
	if err := d.redis.Do(context.WithoutCancel(ctx), command).Error(); err != nil {
		d.logger.Warn("could not withdraw the tools this node's workers hosted",
			"customer", customerID, "agent", agentID, "error", err)
	}
}

// Offers is what the other nodes' workers host for sessions under an agent, by where those
// nodes can be reached.
func (d *Directory) Offers(ctx context.Context, customerID, agentID string) (map[string][]byte, error) {
	entries, err := d.redis.Do(ctx, d.redis.B().Hgetall().Key(d.offerKey(offerKey{customerID, agentID})).Build()).AsStrMap()
	if err != nil {
		return nil, err
	}
	now := time.Now().UnixMilli()
	offers := map[string][]byte{}
	for address, raw := range entries {
		var entry offerEntry
		if address == d.address || json.Unmarshal([]byte(raw), &entry) != nil || entry.Until <= now {
			continue
		}
		offers[address] = entry.Offer
	}
	return offers, nil
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

// advertise says this node has workers for a customer, for holdFor.
//
// One sorted set per customer, scored by when each node's claim lapses, because a set
// cannot expire its members one at a time: a node that died is read past by its score and
// swept out by the next node to renew.
func (d *Directory) advertise(ctx context.Context, customerID string) {
	key := d.workers + customerID
	now := time.Now()
	for _, result := range d.redis.DoMulti(ctx,
		d.redis.B().Zadd().Key(key).ScoreMember().
			ScoreMember(float64(now.Add(holdFor).UnixMilli()), d.address).Build(),
		d.redis.B().Zremrangebyscore().Key(key).
			Min("-inf").Max(strconv.FormatInt(now.UnixMilli(), 10)).Build(),
		d.redis.B().Pexpire().Key(key).Milliseconds(holdFor.Milliseconds()).Build(),
	) {
		if err := result.Error(); err != nil {
			d.logger.Warn("could not say this node has workers for a customer",
				"customer", customerID, "error", err)
			return
		}
	}
}

// publishOffer says what the workers here host for an agent, for holdFor.
func (d *Directory) publishOffer(ctx context.Context, key offerKey, offer []byte) {
	entry, err := json.Marshal(offerEntry{Until: time.Now().Add(holdFor).UnixMilli(), Offer: offer})
	if err != nil {
		d.logger.Warn("could not say what this node's workers host", "agent", key.agentID, "error", err)
		return
	}
	hash := d.offerKey(key)
	for _, result := range d.redis.DoMulti(ctx,
		d.redis.B().Hset().Key(hash).FieldValue().FieldValue(d.address, string(entry)).Build(),
		d.redis.B().Pexpire().Key(hash).Milliseconds(holdFor.Milliseconds()).Build(),
	) {
		if err := result.Error(); err != nil {
			d.logger.Warn("could not say what this node's workers host",
				"customer", key.customerID, "agent", key.agentID, "error", err)
			return
		}
	}
}

func (d *Directory) offerKey(key offerKey) string {
	return d.hosted + key.customerID + ":" + key.agentID
}

// requestsFor is the channel the node at an address takes work for its workers on.
func (d *Directory) requestsFor(address string) string {
	return d.hosted + "requests:" + address
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

		d.sayShard(ctx)
		d.mu.Lock()
		held := maps.Clone(d.held)
		waiting := maps.Clone(d.waiting)
		offered := maps.Clone(d.offered)
		d.mu.Unlock()

		for sessionID, agentID := range held {
			d.write(ctx, sessionID, agentID)
		}
		for customerID := range waiting {
			d.advertise(ctx, customerID)
		}
		for key, offer := range offered {
			d.publishOffer(ctx, key, offer)
		}
	}
}
