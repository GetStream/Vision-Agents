package ledger

import (
	"context"
	"crypto/sha1" //nolint:gosec // EVALSHA names a script by its SHA-1
	_ "embed"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// KeyPrefix starts every key the mirror writes: usage:{<org>}:<subject>:<ids>.
const KeyPrefix = usage.CounterPrefix

// scriptChunk is the most totals one script call writes, so one organization's batch does not
// hold Redis, which runs a script alone, for long.
const scriptChunk = 1000

//go:embed mirror.lua
var mirrorScript string

// scriptSHA1 is the digest EVALSHA names the script by.
var scriptSHA1 = func() string {
	sum := sha1.Sum([]byte(mirrorScript))
	return hex.EncodeToString(sum[:])
}()

var (
	errDialing = errors.New("ledger: mirror: another write is connecting to redis")
	errClosed  = errors.New("ledger: mirror: closed")
)

// Mirror writes window totals to Redis, where admission reads them. Only the usage service
// writes these keys, and only with totals Postgres has committed.
type Mirror struct {
	options rueidis.ClientOption
	script  *rueidis.Lua
	logger  *slog.Logger
	timeout time.Duration
	every   time.Duration
	giveUp  time.Duration
	now     func() time.Time

	mu sync.Mutex
	// client is nil until a write connects. dialing is true while one does, which takes up
	// to the timeout when Redis is unreachable, so the writes behind it fail into the retries
	// at once rather than queue for their own dial.
	client  rueidis.Client
	dialing bool
	closed  bool
	// pending is every organization's totals waiting for Redis to come back, by counter id.
	pending map[string]map[string]waiting
}

// waiting is one total that could not be written, and since when its field has waited.
type waiting struct {
	counter
	since time.Time
}

// counter is one window total as the script takes it.
type counter struct {
	key, field, total, interval, keep string
	expireAt                          int64
	micros                            int64
}

// MirrorOptions configure a Mirror.
type MirrorOptions struct {
	// URL is the Redis to write to: redis://[user:password@]host:port[/db], rediss:// for
	// TLS, or a bare host:port.
	URL    string
	Logger *slog.Logger
	// Timeout bounds one write; defaults to 2 s.
	Timeout time.Duration
	// RetryEvery is how often a failed write is tried again; defaults to 1 s.
	RetryEvery time.Duration
	// GiveUpAfter is how long a failed total is retried; defaults to a minute. After it the
	// counter is repaired by its next write, or by a read through the service.
	GiveUpAfter time.Duration
	// Now is the clock the retries are timed by; nil is time.Now.
	Now func() time.Time
}

// NewMirror returns a mirror of the Redis at options.URL. It connects on its first write, so
// a Redis that is down when the service starts delays nothing but the mirror.
func NewMirror(options MirrorOptions) (*Mirror, error) {
	url := options.URL
	if !strings.Contains(url, "://") {
		url = "redis://" + url
	}
	clientOptions, err := rueidis.ParseURL(url)
	if err != nil {
		// Not wrapped: the parse error quotes the URL, and so its password, into the logs.
		return nil, errors.New("ledger: mirror: the Redis URL does not parse: want redis://[user:password@]host:port[/db], rediss:// for TLS, or host:port")
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	if options.Timeout == 0 {
		options.Timeout = 2 * time.Second
	}
	if options.RetryEvery == 0 {
		options.RetryEvery = time.Second
	}
	if options.GiveUpAfter == 0 {
		options.GiveUpAfter = time.Minute
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	clientOptions.DisableCache = true
	clientOptions.Dialer.Timeout = options.Timeout
	return &Mirror{
		options: clientOptions,
		script:  rueidis.NewLuaScript(mirrorScript),
		logger:  options.Logger,
		timeout: options.Timeout,
		every:   options.RetryEvery,
		giveUp:  options.GiveUpAfter,
		now:     options.Now,
		pending: map[string]map[string]waiting{},
	}, nil
}

// Close releases the connection.
func (m *Mirror) Close() {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	if m.client != nil {
		m.client.Close()
		m.client = nil
	}
}

// Redis is the client the mirror writes with, connecting if it has not yet. While another
// call connects it fails at once.
func (m *Mirror) Redis() (rueidis.Client, error) {
	m.mu.Lock()
	client, dialing, closed := m.client, m.dialing, m.closed
	if client == nil && !dialing && !closed {
		m.dialing = true
	}
	m.mu.Unlock()
	switch {
	case client != nil:
		return client, nil
	case closed:
		return nil, errClosed
	case dialing:
		return nil, errDialing
	}

	client, err := rueidis.NewClient(m.options)
	m.mu.Lock()
	defer m.mu.Unlock()
	m.dialing = false
	if err != nil {
		return nil, fmt.Errorf("ledger: mirror: connect: %w", err)
	}
	// Close ran while this one was connecting.
	if m.closed {
		client.Close()
		return nil, errClosed
	}
	m.client = client
	return client, nil
}

// Write writes the windows' totals, in one round trip, and reports whether Redis took them
// all. A total that cannot be written is counted, logged and retried in the background by
// Run: the batch it came from is durable in Postgres, and is not failed for it. A nil mirror
// writes nothing, and has taken only an empty list.
func (m *Mirror) Write(ctx context.Context, windows []UsageWindow) bool {
	if m == nil {
		return len(windows) == 0
	}
	if len(windows) == 0 {
		return true
	}
	byOrganization := map[string]map[string]counter{}
	for _, window := range windows {
		counters, ok := byOrganization[window.OrganizationID]
		if !ok {
			counters = map[string]counter{}
			byOrganization[window.OrganizationID] = counters
		}
		mergeCounter(counters, counterOf(window))
	}
	// The totals are committed, so a caller that has gone away does not stop them reaching
	// Redis.
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), m.timeout)
	defer cancel()
	failed, err := m.write(ctx, byOrganization)
	if len(failed) == 0 {
		return true
	}
	m.logger.Error("could not mirror window totals to redis, retrying", "organizations", len(failed), "error", err)
	m.mu.Lock()
	defer m.mu.Unlock()
	now := m.now()
	for _, organization := range failed {
		entries, ok := m.pending[organization]
		if !ok {
			entries = map[string]waiting{}
			m.pending[organization] = entries
		}
		for id, c := range byOrganization[organization] {
			existing, ok := entries[id]
			switch {
			case !ok:
				entries[id] = waiting{counter: c, since: now}
			case c.micros > existing.micros:
				entries[id] = waiting{counter: c, since: existing.since}
			}
		}
	}
	return false
}

// Run retries the writes that failed until ctx is done, and then logs what it drops.
func (m *Mirror) Run(ctx context.Context) {
	if m == nil {
		return
	}
	ticker := time.NewTicker(m.every)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			m.drop()
			return
		case <-ticker.C:
			m.retry(ctx)
		}
	}
}

// Pending is how many organizations have totals waiting to be written.
func (m *Mirror) Pending() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.pending)
}

// retry tries every waiting total once, and gives up on those waiting too long.
func (m *Mirror) retry(ctx context.Context) {
	m.mu.Lock()
	taken := make(map[string]map[string]counter, len(m.pending))
	for organization, entries := range m.pending {
		counters := make(map[string]counter, len(entries))
		for id, entry := range entries {
			counters[id] = entry.counter
		}
		taken[organization] = counters
	}
	m.mu.Unlock()
	if len(taken) == 0 {
		return
	}

	ctx, cancel := context.WithTimeout(ctx, m.timeout)
	defer cancel()
	failed, err := m.write(ctx, taken)
	stillFailing := make(map[string]bool, len(failed))
	for _, organization := range failed {
		stillFailing[organization] = true
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	now := m.now()
	for organization := range taken {
		entries := m.pending[organization]
		var dropped int
		var waited time.Duration
		for id, entry := range entries {
			switch {
			case !stillFailing[organization]:
				// A total that failed again meanwhile was merged into the entry, and the
				// one written now covers only what was taken; anything newer stays.
				if written, ok := taken[organization][id]; ok && written.micros >= entry.micros {
					delete(entries, id)
				}
			case now.Sub(entry.since) >= m.giveUp:
				delete(entries, id)
				dropped++
				waited = max(waited, now.Sub(entry.since))
			}
		}
		if dropped > 0 {
			MirrorGivenUp.Add(float64(dropped))
			m.logger.Error("gave up mirroring window totals to redis: the next write or read of each repairs it",
				"organization", organization, "counters", dropped, "waited", waited, "error", err)
		}
		if len(entries) == 0 {
			delete(m.pending, organization)
		}
	}
}

// drop forgets every total still waiting, as the service stops.
func (m *Mirror) drop() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for organization, entries := range m.pending {
		MirrorGivenUp.Add(float64(len(entries)))
		m.logger.Error("stopping with window totals not mirrored to redis: the next write or read of each repairs it",
			"organization", organization, "counters", len(entries))
	}
	clear(m.pending)
}

// write runs the script for each organization, in calls of at most scriptChunk totals, all in
// one round trip, and returns the organizations with a call that failed.
func (m *Mirror) write(ctx context.Context, byOrganization map[string]map[string]counter) ([]string, error) {
	started := time.Now()
	defer func() { mirrorSeconds.Observe(time.Since(started).Seconds()) }()

	organizations := make([]string, 0, len(byOrganization))
	for organization := range byOrganization {
		organizations = append(organizations, organization)
	}
	client, err := m.Redis()
	if err != nil {
		MirrorFailures.Add(float64(len(organizations)))
		return organizations, err
	}
	var calls []rueidis.LuaExec
	var owners []string
	for _, organization := range organizations {
		for _, call := range scriptCalls(byOrganization[organization]) {
			calls = append(calls, call)
			owners = append(owners, organization)
		}
	}
	var failed []string
	failing := map[string]bool{}
	var last error
	for i, result := range m.exec(ctx, client, calls) {
		if err := result.Error(); err != nil {
			if !failing[owners[i]] {
				failing[owners[i]] = true
				failed = append(failed, owners[i])
			}
			last = err
		}
	}
	MirrorFailures.Add(float64(len(failed)))
	return failed, last
}

// exec runs the calls with EVALSHA in one pipeline, and again with EVAL the ones Redis did
// not have the script for (a restarted or flushed Redis), which loads it for the next ones.
func (m *Mirror) exec(ctx context.Context, client rueidis.Client, calls []rueidis.LuaExec) []rueidis.RedisResult {
	commands := make(rueidis.Commands, 0, len(calls))
	for _, call := range calls {
		commands = append(commands, client.B().Evalsha().Sha1(scriptSHA1).Numkeys(int64(len(call.Keys))).Key(call.Keys...).Arg(call.Args...).Build())
	}
	results := client.DoMulti(ctx, commands...)
	for i, result := range results {
		if redisErr, ok := rueidis.IsRedisErr(result.Error()); ok && redisErr.IsNoScript() {
			results[i] = m.script.Exec(ctx, client, calls[i].Keys, calls[i].Args)
		}
	}
	return results
}

// scriptCalls are one organization's counters as the script's keys and arguments, at most
// scriptChunk counters a call.
func scriptCalls(counters map[string]counter) []rueidis.LuaExec {
	var calls []rueidis.LuaExec
	var call rueidis.LuaExec
	index := map[string]int{}
	written := 0
	for _, c := range counters {
		if written == scriptChunk {
			calls = append(calls, call)
			call, index, written = rueidis.LuaExec{}, map[string]int{}, 0
		}
		position, ok := index[c.key]
		if !ok {
			call.Keys = append(call.Keys, c.key)
			position = len(call.Keys)
			index[c.key] = position
		}
		call.Args = append(call.Args, strconv.Itoa(position), c.field, c.total, c.interval, c.keep, strconv.FormatInt(c.expireAt, 10))
		written++
	}
	if written > 0 {
		calls = append(calls, call)
	}
	return calls
}

// counterOf is a window's total as the script takes it.
func counterOf(window UsageWindow) counter {
	previous, _ := usage.Window(window.Interval, window.WindowStart.Add(-time.Nanosecond))
	return counter{
		key:      window.Ref().Key(KeyPrefix),
		field:    usage.Field(window.Interval, window.WindowStart),
		total:    strconv.FormatInt(window.SpentMicros, 10),
		micros:   window.SpentMicros,
		interval: string(window.Interval),
		keep:     usage.Field(window.Interval, previous),
		expireAt: window.WindowEnd.Add(Retention).Unix(),
	}
}

// mergeCounter keeps the larger total of one field.
func mergeCounter(counters map[string]counter, c counter) {
	id := c.key + "\x00" + c.field
	if existing, ok := counters[id]; ok && existing.micros >= c.micros {
		return
	}
	counters[id] = c
}
